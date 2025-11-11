#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
NaVILA Evaluator for Falcon Framework
专门处理NaVILA策略的评估器，支持语言指令到动作的转换
"""

import copy
import json
import os
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import torch
import tqdm
from PIL import Image

from habitat import logger
from habitat.tasks.rearrange.rearrange_sensors import GfxReplayMeasure
from habitat.tasks.rearrange.utils import write_gfx_replay
from habitat.utils.visualizations.utils import (
    observations_to_image,
    overlay_frame,
)
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
)
from habitat_baselines.rl.ppo.evaluator import Evaluator, pause_envs
from habitat_baselines.rl.ppo.falcon_evaluator import FALCONEvaluator
from habitat_baselines.utils.common import (
    batch_obs,
    generate_video,
    get_action_space_info,
    inference_mode,
)
from habitat_baselines.utils.info_dict import extract_scalars_from_info

# NaVILA相关导入
try:
    from habitat_baselines.rl.ddppo.policy.navila.action_parser import NaVILAActionParser
    from habitat_baselines.rl.ddppo.policy.navila_policy import sample_and_pad_images
    
    from habitat_baselines.rl.ddppo.policy.navila.llava.constants import (
        IMAGE_TOKEN_INDEX,
    )
    from habitat_baselines.rl.ddppo.policy.navila.llava.conversation import (
        SeparatorStyle,
        conv_templates,
    )
    from habitat_baselines.rl.ddppo.policy.navila.llava.mm_utils import (
        KeywordsStoppingCriteria,
        process_images,
        tokenizer_image_token,
    )
    from habitat_baselines.rl.ddppo.policy.navila.llava.model.builder import (
        load_pretrained_model,
    )
    NAVILA_AVAILABLE = True
except ImportError as e:
    NAVILA_AVAILABLE = False
    print(f"Warning: NaVILA modules not available: {e}")


class NaVILAEvaluator(FALCONEvaluator):
    """
    NaVILA专用评估器
    
    继承自FALCONEvaluator，但使用NaVILA的语言模型直接生成动作，
    而不是通过policy网络。
    """
    
    def evaluate_agent(
        self,
        agent,
        envs,
        config,
        checkpoint_index,
        step_id,
        writer,
        device,
        obs_transforms,
        env_spec,
        rank0_keys,
    ):
        """
        评估NaVILA agent
        
        该方法重写父类方法，直接使用LLAVA模型生成动作指令。
        """
        if not NAVILA_AVAILABLE:
            raise ImportError("NaVILA modules are required for NaVILAEvaluator")
        
        # 加载NaVILA模型
        navila_config = config.habitat_baselines.rl.policy.agent_0
        navila_model_path = navila_config.get("navila_model_path", None)
        
        if navila_model_path is None or not os.path.exists(navila_model_path):
            raise ValueError(
                f"NaVILA model path is not provided or does not exist: {navila_model_path}"
            )
        
        logger.info(f"Loading NaVILA model from {navila_model_path}")
        model_name = os.path.basename(os.path.normpath(navila_model_path))
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            navila_model_path, model_name
        )
        model = model.to(device)
        model.eval()
        
        # 初始化动作解析器
        action_parser = NaVILAActionParser(
            forward_step=navila_config.get("forward_step", 25),
            turn_step=navila_config.get("turn_step", 15),
        )
        
        num_video_frames = navila_config.get("num_video_frames", 8)
        
        success_cal = 0
        observations = envs.reset()
        observations = envs.post_step(observations)
        batch = batch_obs(observations, device=device)
        batch = apply_obs_transforms_batch(batch, obs_transforms)
        
        current_episode_reward = torch.zeros(envs.num_envs, 1, device="cpu")
        
        stats_episodes: Dict[Any, Any] = {}
        ep_eval_count: Dict[Any, int] = defaultdict(lambda: 0)
        
        # 历史RGB帧缓存（每个环境一个）
        past_rgbs = [[] for _ in range(envs.num_envs)]
        
        # 动作队列（每个环境一个）
        action_queues = [[] for _ in range(envs.num_envs)]
        
        if len(config.habitat_baselines.eval.video_option) > 0:
            rgb_frames: List[List[np.ndarray]] = [
                [observations_to_image({k: v[env_idx] for k, v in batch.items()}, {})]
                for env_idx in range(config.habitat_baselines.num_environments)
            ]
        else:
            rgb_frames = None
        
        if len(config.habitat_baselines.eval.video_option) > 0:
            os.makedirs(config.habitat_baselines.video_dir, exist_ok=True)
        
        number_of_eval_episodes = config.habitat_baselines.test_episode_count
        evals_per_ep = config.habitat_baselines.eval.evals_per_ep
        if number_of_eval_episodes == -1:
            number_of_eval_episodes = sum(envs.number_of_episodes)
        else:
            total_num_eps = sum(envs.number_of_episodes)
            if total_num_eps < number_of_eval_episodes and total_num_eps > 1:
                logger.warn(
                    f"Config specified {number_of_eval_episodes} eval episodes, "
                    f"dataset only has {total_num_eps}. Evaluating with {total_num_eps} instead."
                )
                number_of_eval_episodes = total_num_eps
            else:
                assert evals_per_ep == 1
        
        assert number_of_eval_episodes > 0, (
            "You must specify a number of evaluation episodes with test_episode_count"
        )
        
        pbar = tqdm.tqdm(total=number_of_eval_episodes * evals_per_ep)
        actions_record = defaultdict(list)
        
        while (
            len(stats_episodes) < (number_of_eval_episodes * evals_per_ep)
            and envs.num_envs > 0
        ):
            current_episodes_info = envs.current_episodes()
            
            # 为每个环境生成动作
            step_data = []
            for i in range(envs.num_envs):
                # 如果动作队列中有动作，直接使用
                if len(action_queues[i]) > 0:
                    action = action_queues[i].pop(0)
                    logger.info(f"Env {i}: Using queued action {action}")
                else:
                    # 否则，使用NaVILA生成新动作
                    action = self._generate_navila_action(
                        batch, i, past_rgbs[i], num_video_frames,
                        current_episodes_info[i], model, tokenizer, 
                        image_processor, action_parser, action_queues[i], device
                    )
                
                step_data.append(action)
            
            # 执行动作
            outputs = envs.step(step_data)
            observations, rewards_l, dones, infos = [list(x) for x in zip(*outputs)]
            
            # 记录动作
            for i in range(envs.num_envs):
                episode_key = (
                    current_episodes_info[i].scene_id,
                    current_episodes_info[i].episode_id,
                    ep_eval_count[
                        (current_episodes_info[i].scene_id, current_episodes_info[i].episode_id)
                    ]
                )
                actions_record[episode_key].append({
                    "type": "scalar",
                    "value": int(step_data[i])
                })
            
            # 更新观察
            observations = envs.post_step(observations)
            batch = batch_obs(observations, device=device)
            batch = apply_obs_transforms_batch(batch, obs_transforms)
            
            # 添加当前RGB到历史
            for i in range(envs.num_envs):
                if "rgb" in batch:
                    curr_rgb = Image.fromarray(
                        np.uint8(batch["rgb"][i].cpu().numpy())
                    ).convert("RGB")
                    past_rgbs[i].append(curr_rgb)
            
            rewards = torch.tensor(rewards_l, dtype=torch.float, device="cpu").unsqueeze(1)
            current_episode_reward += rewards
            next_episodes_info = envs.current_episodes()
            envs_to_pause = []
            n_envs = envs.num_envs
            
            for i in range(n_envs):
                if (
                    ep_eval_count[(next_episodes_info[i].scene_id, next_episodes_info[i].episode_id)]
                    == evals_per_ep
                ):
                    envs_to_pause.append(i)
                
                disp_info = {k: v for k, v in infos[i].items() if k not in rank0_keys}
                
                if len(config.habitat_baselines.eval.video_option) > 0:
                    frame = observations_to_image({k: v[i] for k, v in batch.items()}, disp_info)
                    if dones[i]:
                        final_frame = observations_to_image(
                            {k: v[i] * 0.0 for k, v in batch.items()}, disp_info
                        )
                        final_frame = overlay_frame(final_frame, disp_info)
                        rgb_frames[i].append(final_frame)
                        rgb_frames[i].append(frame)
                    else:
                        frame = overlay_frame(frame, disp_info)
                        rgb_frames[i].append(frame)
                
                # Episode结束
                if dones[i]:
                    pbar.update()
                    if "success" in disp_info:
                        success_cal += disp_info['success']
                        logger.info(
                            f"Till now Success Rate: {success_cal/(len(stats_episodes)+1):.4f}"
                        )
                    
                    episode_stats = {"reward": current_episode_reward[i].item()}
                    episode_stats.update(extract_scalars_from_info(infos[i]))
                    current_episode_reward[i] = 0
                    k = (current_episodes_info[i].scene_id, current_episodes_info[i].episode_id)
                    ep_eval_count[k] += 1
                    stats_episodes[(k, ep_eval_count[k])] = episode_stats
                    
                    # 重置该环境的历史和队列
                    past_rgbs[i] = []
                    action_queues[i] = []
                    
                    if len(config.habitat_baselines.eval.video_option) > 0:
                        scene_id = current_episodes_info[i].scene_id.split('/')[-1].split('.')[0]
                        logger.info(
                            f"Scene ID: {scene_id}, Episode ID: {current_episodes_info[i].episode_id}"
                        )
                        
                        generate_video(
                            video_option=config.habitat_baselines.eval.video_option,
                            video_dir=config.habitat_baselines.video_dir,
                            images=rgb_frames[i][:-1],
                            scene_id=scene_id,
                            episode_id=f"{current_episodes_info[i].episode_id}_{ep_eval_count[k]}",
                            checkpoint_idx=checkpoint_index,
                            metrics=extract_scalars_from_info(disp_info),
                            fps=config.habitat_baselines.video_fps,
                            tb_writer=writer,
                            keys_to_include_in_name=config.habitat_baselines.eval_keys_to_include_in_name,
                        )
                        rgb_frames[i] = rgb_frames[i][-1:]
                    
                    gfx_str = infos[i].get(GfxReplayMeasure.cls_uuid, "")
                    if gfx_str != "":
                        write_gfx_replay(
                            gfx_str, config.habitat.task, current_episodes_info[i].episode_id
                        )
            
            # 暂停环境
            if envs_to_pause:
                # 同时暂停历史和队列
                past_rgbs = [past_rgbs[i] for i in range(n_envs) if i not in envs_to_pause]
                action_queues = [action_queues[i] for i in range(n_envs) if i not in envs_to_pause]
                
                not_done_masks = torch.tensor(
                    [[not done] for done in dones], dtype=torch.bool, device="cpu"
                )
                (
                    envs, _, not_done_masks, current_episode_reward, _, batch, rgb_frames,
                ) = pause_envs(
                    envs_to_pause, envs, None, not_done_masks, 
                    current_episode_reward, None, batch, rgb_frames,
                )
        
        pbar.close()
        
        # 聚合统计信息
        aggregated_stats = {}
        all_ks = set()
        for ep in stats_episodes.values():
            all_ks.update(ep.keys())
        for stat_key in all_ks:
            aggregated_stats[stat_key] = np.mean(
                [v[stat_key] for v in stats_episodes.values() if stat_key in v]
            )
        
        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.4f}")
        
        writer.add_scalar("eval_reward/average_reward", aggregated_stats["reward"], step_id)
        
        metrics = {k: v for k, v in aggregated_stats.items() if k != "reward"}
        for k, v in metrics.items():
            writer.add_scalar(f"eval_metrics/{k}", v, step_id)
        
        # 保存结果
        result_path = os.path.join("output/", "result.json")
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        evalai_result = {
            "SR": round(aggregated_stats.get("success", 0), 4),
            "SPL": round(aggregated_stats.get("spl", 0), 4),
            "PSC": round(aggregated_stats.get("psc", 0), 4),
            "H-Coll": round(aggregated_stats.get("human_collision", 0), 4),
            "Total": round(
                0.4 * aggregated_stats.get("success", 0)
                + 0.3 * aggregated_stats.get("spl", 0)
                + 0.3 * aggregated_stats.get("psc", 0),
                4,
            ),
        }
        
        with open(result_path, "w") as f:
            json.dump(evalai_result, f, indent=2)
        
        # 保存动作记录
        actions_output_path = os.path.join("output/", "actions.json")
        serializable_actions = {
            f"{scene_id}|{episode_id}|{eval_count}": actions
            for (scene_id, episode_id, eval_count), actions in actions_record.items()
        }
        with open(actions_output_path, "w") as f:
            json.dump(serializable_actions, f, indent=2)
    
    def _generate_navila_action(
        self, batch, env_idx, past_rgbs, num_video_frames,
        current_episode, model, tokenizer, image_processor, 
        action_parser, action_queue, device
    ):
        """
        使用NaVILA模型生成动作
        
        Args:
            batch: 观察批次
            env_idx: 环境索引
            past_rgbs: 历史RGB帧列表
            num_video_frames: 视频帧数
            current_episode: 当前episode信息
            model: LLAVA模型
            tokenizer: tokenizer
            image_processor: 图像处理器
            action_parser: 动作解析器
            action_queue: 动作队列（用于存储多步骤动作）
            device: 设备
            
        Returns:
            action: 动作ID (0-3)
        """
        # 获取当前RGB
        curr_rgb = Image.fromarray(
            np.uint8(batch["rgb"][env_idx].cpu().numpy())
        ).convert("RGB")
        
        # 构建视频序列
        past_and_current_rgbs = past_rgbs + [curr_rgb]
        sampled_frames = sample_and_pad_images(
            past_and_current_rgbs, num_frames=num_video_frames
        )
        
        # 获取指令（如果有的话）
        instruction = "Navigate to the goal location"
        if hasattr(current_episode, 'instruction'):
            instruction = current_episode.instruction.instruction_text
        
        # 构建提示
        interleaved_images = "<image>\n" * (len(sampled_frames) - 1)
        question = (
            f"Imagine you are a robot programmed for navigation tasks. You have been given a video "
            f'of historical observations {interleaved_images}, and current observation <image>\n. '
            f'Your assigned task is: "{instruction}" '
            f"Analyze this series of images to decide your next action, which could be turning left or right by a specific "
            f"degree, moving forward a certain distance, or stop if the task is completed."
        )
        
        # 构建对话
        conv_mode = "llama_3"
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        # 处理图像
        images_tensor = process_images(
            sampled_frames, image_processor, model.config
        ).to(device, dtype=torch.float16)
        
        # Tokenize
        input_ids = (
            tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .to(device)
        )
        
        # 停止条件
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        
        # 生成输出
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images_tensor.half().to(device),
                do_sample=False,
                temperature=0.0,
                max_new_tokens=32,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
                pad_token_id=tokenizer.eos_token_id,
            )
        
        # 解码
        output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        if output_text.endswith(stop_str):
            output_text = output_text[: -len(stop_str)].strip()
        
        logger.info(f"NaVILA output: {output_text}")
        
        # 解析动作
        action, num_repeats = action_parser.parse_action(output_text)
        
        # 将后续动作加入队列
        if num_repeats > 1:
            for _ in range(num_repeats - 1):
                action_queue.append(action)
            logger.info(f"Added {num_repeats - 1} actions to queue. Queue length: {len(action_queue)}")
        
        return action
