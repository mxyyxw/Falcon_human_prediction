#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
StreamVLN Evaluator for Falcon framework.
专门用于 StreamVLN 生成式动作序列的评估器。
"""

import os
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch import nn

from habitat import logger
from habitat.utils.visualizations.utils import observations_to_image
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.rl.ppo.evaluator import Evaluator
from habitat_baselines.utils.common import (
    batch_obs,
    generate_video,
    inference_mode,
)


@baseline_registry.register_evaluator
class StreamVLNEvaluator(Evaluator):
    """
    StreamVLN 专用评估器。
    
    与标准 Falcon Evaluator 的主要区别：
    1. 使用 StreamVLN 的 generate_action_sequence() 直接生成动作
    2. 维护动作缓存，从生成的动作序列中逐步执行
    3. 支持多步规划（一次生成多个动作）
    4. 可选地计算辅助任务损失（仅用于监控，不更新）
    """

    def __init__(self, config, agent, envs, device):
        """
        初始化 StreamVLN Evaluator。
        
        Args:
            config: 配置对象
            agent: StreamVLN agent (包含 StreamVLNPolicy)
            envs: 环境向量
            device: 运行设备
        """
        self.config = config
        self.agent = agent
        self.envs = envs
        self.device = device
        
        # 动作缓存：每个环境维护一个动作队列
        self.num_envs = envs.num_envs
        self.action_buffers = [deque() for _ in range(self.num_envs)]
        
        # StreamVLN 配置
        self.generate_every_n_steps = config.habitat_baselines.rl.policy.get(
            "generate_every_n_steps", 1
        )  # 每 N 步生成一次新的动作序列
        
        # 辅助任务（可选）
        self.compute_aux_losses = config.habitat_baselines.rl.get(
            "compute_aux_losses_in_eval", False
        )
        if self.compute_aux_losses:
            self.aux_tasks = self._setup_aux_tasks()
        
        # 指标收集
        self.episode_stats = defaultdict(list)
        
        # Observation transforms
        self.obs_transforms = get_active_obs_transforms(config)
        self.obs_space = apply_obs_transforms_obs_space(
            envs.observation_space, self.obs_transforms
        )
        
        logger.info("StreamVLNEvaluator initialized")
        logger.info(f"  - Num environments: {self.num_envs}")
        logger.info(f"  - Generate every {self.generate_every_n_steps} steps")
        logger.info(f"  - Compute aux losses: {self.compute_aux_losses}")

    def _setup_aux_tasks(self):
        """设置辅助任务（用于监控，不更新）"""
        from falcon.auxiliary_tasks import (
            FutureTrajectoryPrediction,
            GuessHumanPosition,
            PeopleCounting,
        )
        
        aux_tasks = nn.ModuleDict()
        
        if hasattr(self.config.habitat_baselines.rl, 'auxiliary_losses'):
            aux_config = self.config.habitat_baselines.rl.auxiliary_losses
            
            if 'people_counting' in aux_config:
                aux_tasks['people_counting'] = PeopleCounting(
                    self.agent.actor_critic.net.recurrent_hidden_size,
                    max_human_num=aux_config.people_counting.max_human_num
                )
            
            if 'guess_human_position' in aux_config:
                aux_tasks['guess_human_position'] = GuessHumanPosition(
                    self.agent.actor_critic.net.recurrent_hidden_size,
                    max_human_num=aux_config.guess_human_position.max_human_num
                )
            
            if 'future_trajectory_prediction' in aux_config:
                aux_tasks['future_trajectory_prediction'] = FutureTrajectoryPrediction(
                    self.agent.actor_critic.net.recurrent_hidden_size,
                    max_human_num=aux_config.future_trajectory_prediction.max_human_num,
                    future_step=aux_config.future_trajectory_prediction.future_step
                )
        
        return aux_tasks.to(self.device)

    def _get_instruction_from_pointgoal(
        self, 
        observations: Dict[str, torch.Tensor],
        env_idx: int
    ) -> str:
        """
        从 pointgoal 生成简单的导航指令。
        
        在实际的 VLN 任务中，应该有语言指令。
        这里我们从 pointgoal 传感器生成一个简单的代理指令。
        """
        if 'agent_0_pointgoal_with_gps_compass' in observations:
            pointgoal = observations['agent_0_pointgoal_with_gps_compass'][env_idx]
            distance = pointgoal[0].item()
            angle = pointgoal[1].item()
            
            # 生成简单的指令
            if distance < 0.5:
                instruction = "you are near the goal, please stop"
            elif abs(angle) > 0.5:
                if angle > 0:
                    instruction = "turn left towards the goal"
                else:
                    instruction = "turn right towards the goal"
            else:
                instruction = "move forward to reach the goal"
            
            return instruction
        
        return "navigate to the goal"

    @torch.no_grad()
    def evaluate_agent(
        self,
        num_eval_episodes: Optional[int] = None,
        writer: Optional[TensorboardWriter] = None,
        checkpoint_index: int = 0,
    ):
        """
        评估 StreamVLN agent。
        
        Args:
            num_eval_episodes: 评估的回合数
            writer: Tensorboard writer
            checkpoint_index: Checkpoint 索引（用于日志）
        
        Returns:
            Dict: 评估指标
        """
        logger.info(f"Starting StreamVLN evaluation for {num_eval_episodes} episodes")
        
        # 重置环境
        observations = self.envs.reset()
        observations = self._prepare_batch(observations)
        
        # 重置统计
        self.episode_stats = defaultdict(list)
        episode_rewards = torch.zeros(self.num_envs, 1, device=self.device)
        episode_counts = torch.zeros(self.num_envs, 1, device=self.device)
        current_episode_steps = torch.zeros(self.num_envs, 1, dtype=torch.long)
        
        # 重置动作缓存
        for buffer in self.action_buffers:
            buffer.clear()
        
        # 重置 StreamVLN 的 episode state
        for env_idx in range(self.num_envs):
            self.agent.actor_critic.net.reset_episode_state()
        
        # 主评估循环
        pbar = range(num_eval_episodes) if num_eval_episodes else None
        step = 0
        
        while episode_counts.sum() < num_eval_episodes:
            current_episodes = self.envs.current_episodes()
            actions = []
            
            # 为每个环境生成/获取动作
            for env_idx in range(self.num_envs):
                if episode_counts[env_idx] >= num_eval_episodes / self.num_envs:
                    # 该环境已完成足够的回合
                    actions.append(0)  # STOP
                    continue
                
                # 检查是否需要生成新的动作序列
                if len(self.action_buffers[env_idx]) == 0:
                    # 生成新的动作序列
                    rgb = observations['agent_0_articulated_agent_jaw_rgb'][env_idx]
                    
                    # 获取指令
                    instruction = self._get_instruction_from_pointgoal(
                        observations, env_idx
                    )
                    
                    # 调用 StreamVLN 生成动作序列
                    try:
                        rgb_np = rgb.cpu().numpy()
                        if rgb_np.dtype != np.uint8:
                            if rgb_np.max() <= 1.0:
                                rgb_np = (rgb_np * 255).astype(np.uint8)
                            else:
                                rgb_np = rgb_np.astype(np.uint8)
                        
                        action_seq, llm_output = self.agent.actor_critic.net.generate_action_sequence(
                            rgb=rgb_np,
                            instruction=instruction,
                            env_idx=env_idx,
                            run_model=True
                        )
                        
                        logger.info(f"[Env {env_idx}] Generated action sequence: {action_seq}")
                        logger.info(f"[Env {env_idx}] LLM output: {llm_output[:100]}...")
                        
                        # 填充动作缓存
                        self.action_buffers[env_idx].extend(action_seq)
                    
                    except Exception as e:
                        logger.warning(f"[Env {env_idx}] Action generation failed: {e}")
                        # 回退：使用随机动作
                        self.action_buffers[env_idx].append(
                            np.random.randint(0, 4)
                        )
                
                # 从缓存中取出下一个动作
                if len(self.action_buffers[env_idx]) > 0:
                    action = self.action_buffers[env_idx].popleft()
                else:
                    action = 0  # STOP
                
                actions.append(action)
            
            # 执行动作
            actions = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
            outputs = self.envs.step([a.item() for a in actions])
            
            observations, rewards, dones, infos = [list(x) for x in zip(*outputs)]
            observations = self._prepare_batch(observations)
            
            # 更新统计
            rewards = torch.tensor(
                rewards, dtype=torch.float, device=self.device
            ).unsqueeze(1)
            episode_rewards += rewards
            current_episode_steps += 1
            
            # 处理完成的回合
            for env_idx in range(self.num_envs):
                if dones[env_idx]:
                    episode_counts[env_idx] += 1
                    
                    # 收集指标
                    info = infos[env_idx]
                    self.episode_stats['reward'].append(episode_rewards[env_idx].item())
                    self.episode_stats['length'].append(current_episode_steps[env_idx].item())
                    
                    if 'success' in info:
                        self.episode_stats['success'].append(info['success'])
                    if 'spl' in info:
                        self.episode_stats['spl'].append(info['spl'])
                    if 'distance_to_goal' in info:
                        self.episode_stats['distance_to_goal'].append(info['distance_to_goal'])
                    
                    # 重置
                    episode_rewards[env_idx] = 0
                    current_episode_steps[env_idx] = 0
                    self.action_buffers[env_idx].clear()
                    
                    logger.info(
                        f"[Env {env_idx}] Episode {int(episode_counts[env_idx].item())} completed: "
                        f"reward={self.episode_stats['reward'][-1]:.2f}, "
                        f"length={self.episode_stats['length'][-1]}"
                    )
            
            step += 1
        
        # 计算最终指标
        aggregated_stats = {}
        for k, v in self.episode_stats.items():
            if len(v) > 0:
                aggregated_stats[f"eval_{k}_mean"] = np.mean(v)
                aggregated_stats[f"eval_{k}_std"] = np.std(v)
        
        # 记录到 tensorboard
        if writer:
            for k, v in aggregated_stats.items():
                writer.add_scalar(k, v, checkpoint_index)
        
        # 打印结果
        logger.info("=" * 60)
        logger.info("Evaluation Results:")
        for k, v in sorted(aggregated_stats.items()):
            logger.info(f"  {k}: {v:.4f}")
        logger.info("=" * 60)
        
        return aggregated_stats

    def _prepare_batch(self, observations: List[Dict]) -> Dict[str, torch.Tensor]:
        """准备批量观测"""
        batch = batch_obs(observations, device=self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)
        return batch
