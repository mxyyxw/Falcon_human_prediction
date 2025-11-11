#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
NaVILA Policy for Falcon Framework (Habitat3)
基于视觉-语言模型的导航策略
"""

import copy
import os
from collections import OrderedDict
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import torch
from gym import spaces
from PIL import Image
from torch import nn as nn

from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ppo import Net, NetPolicy

# NaVILA相关导入
from habitat_baselines.rl.ddppo.policy.navila.action_parser import NaVILAActionParser

try:
    from habitat_baselines.rl.ddppo.policy.navila.llava.constants import (
        DEFAULT_IMAGE_TOKEN,
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
except ImportError:
    NAVILA_AVAILABLE = False
    print("Warning: NaVILA modules not available. Please ensure llava is properly installed.")

if TYPE_CHECKING:
    from omegaconf import DictConfig


def sample_and_pad_images(images, num_frames=8, width=512, height=512):
    """
    采样和填充图像序列到固定帧数
    
    Args:
        images: 图像列表
        num_frames: 目标帧数
        width: 图像宽度
        height: 图像高度
        
    Returns:
        采样后的图像列表
    """
    frames = copy.deepcopy(images)
    
    # 如果帧数不足，用黑色图像填充
    if len(frames) < num_frames:
        while len(frames) < num_frames:
            frames.insert(0, Image.new("RGB", (width, height), color=(0, 0, 0)))
    
    # 采样：均匀采样历史帧 + 最新帧
    latest_frame = frames[-1]
    sampled_indices = np.linspace(0, len(frames) - 1, num=num_frames - 1, endpoint=False, dtype=int)
    sampled_frames = [frames[i] for i in sampled_indices] + [latest_frame]
    
    return sampled_frames


@baseline_registry.register_policy
class NaVILAPolicy(NetPolicy):
    """
    基于视觉-语言模型(LLAVA)的导航策略
    
    该策略使用NaVILA模型生成语言指令，然后通过动作解析器转换为离散动作。
    适配Habitat3 Falcon框架，支持动态行人环境。
    """
    
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        hidden_size: int = 512,
        navila_model_path: Optional[str] = None,
        num_video_frames: int = 8,
        forward_step: int = 25,
        turn_step: int = 15,
        policy_config: "DictConfig" = None,
        aux_loss_config: Optional["DictConfig"] = None,
        instruction_sensor_uuid: str = "instruction",
        **kwargs,
    ):
        """
        初始化NaVILA策略
        
        Args:
            observation_space: 观察空间
            action_space: 动作空间
            hidden_size: 隐藏层大小（用于兼容性，实际不使用）
            navila_model_path: NaVILA预训练模型路径
            num_video_frames: 输入视频帧数
            forward_step: 前进步长（cm）
            turn_step: 转向步长（度）
            policy_config: 策略配置
            aux_loss_config: 辅助损失配置
            instruction_sensor_uuid: 指令传感器UUID
        """
        if not NAVILA_AVAILABLE:
            raise ImportError(
                "NaVILA modules are not available. "
                "Please ensure the navila package is properly installed."
            )
        
        if policy_config is not None:
            discrete_actions = (
                policy_config.action_distribution_type == "categorical"
            )
            self.action_distribution_type = (
                policy_config.action_distribution_type
            )
        else:
            discrete_actions = True
            self.action_distribution_type = "categorical"
        
        # 创建NaVILA网络
        net = NaVILANet(
            observation_space=observation_space,
            action_space=action_space,
            navila_model_path=navila_model_path,
            num_video_frames=num_video_frames,
            forward_step=forward_step,
            turn_step=turn_step,
            instruction_sensor_uuid=instruction_sensor_uuid,
        )
        
        super().__init__(
            net,
            action_space=action_space,
            policy_config=policy_config,
            aux_loss_config=aux_loss_config,
        )
    
    @classmethod
    def from_config(
        cls,
        config: "DictConfig",
        observation_space: spaces.Dict,
        action_space,
        **kwargs,
    ):
        """从配置创建策略"""
        # 排除用于渲染的相机
        ignore_names = [
            sensor.uuid
            for sensor in config.habitat_baselines.eval.extra_sim_sensors.values()
        ]
        filtered_obs = spaces.Dict(
            OrderedDict(
                (
                    (k, v)
                    for k, v in observation_space.items()
                    if k not in ignore_names
                )
            )
        )
        
        agent_name = None
        if "agent_name" in kwargs:
            agent_name = kwargs["agent_name"]
        
        if agent_name is None:
            if len(config.habitat.simulator.agents_order) > 1:
                raise ValueError(
                    "If there is more than an agent, you need to specify the agent name"
                )
            else:
                agent_name = config.habitat.simulator.agents_order[0]
        
        # 从配置中获取NaVILA特定参数
        navila_config = config.habitat_baselines.rl.policy[agent_name]
        
        return cls(
            observation_space=filtered_obs,
            action_space=action_space,
            navila_model_path=navila_config.get("navila_model_path", None),
            num_video_frames=navila_config.get("num_video_frames", 8),
            forward_step=navila_config.get("forward_step", 25),
            turn_step=navila_config.get("turn_step", 15),
            policy_config=config.habitat_baselines.rl.policy[agent_name],
            aux_loss_config=config.habitat_baselines.rl.auxiliary_losses,
            instruction_sensor_uuid=config.habitat_baselines.rl.policy[agent_name].get(
                "instruction_sensor_uuid", "instruction"
            ),
        )


class NaVILANet(Net):
    """
    NaVILA网络模型
    
    使用LLAVA视觉-语言模型处理RGB图像序列和指令文本，
    生成语言形式的导航动作。
    """
    
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        navila_model_path: Optional[str],
        num_video_frames: int = 8,
        forward_step: int = 25,
        turn_step: int = 15,
        instruction_sensor_uuid: str = "instruction",
    ):
        super().__init__()
        
        self.num_video_frames = num_video_frames
        self.instruction_sensor_uuid = instruction_sensor_uuid
        
        # 初始化动作解析器
        self.action_parser = NaVILAActionParser(
            forward_step=forward_step,
            turn_step=turn_step,
        )
        
        # 加载LLAVA模型
        if navila_model_path is None or not os.path.exists(navila_model_path):
            raise ValueError(
                f"NaVILA model path is not provided or does not exist: {navila_model_path}"
            )
        
        model_name = os.path.basename(os.path.normpath(navila_model_path))
        self.tokenizer, self.model, self.image_processor, self.context_len = (
            load_pretrained_model(navila_model_path, model_name)
        )
        
        # 将模型设置为评估模式
        self.model.eval()
        
        # 历史RGB帧缓存
        self.past_rgbs = []
        
        # 动作队列（用于处理需要重复的动作）
        self.action_queue = []
        
        # 输出大小（用于兼容性）
        self._output_size = 512
    
    @property
    def output_size(self):
        return self._output_size
    
    @property
    def is_blind(self):
        return False
    
    @property
    def num_recurrent_layers(self):
        return 1
    
    @property
    def recurrent_hidden_size(self):
        return self._output_size
    
    def forward(
        self,
        observations: Dict[str, torch.Tensor],
        rnn_hidden_states,
        prev_actions,
        masks,
        rnn_build_seq_info: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        前向传播
        
        Args:
            observations: 观察数据
            rnn_hidden_states: RNN隐藏状态（用于兼容性）
            prev_actions: 前一个动作
            masks: 掩码
            rnn_build_seq_info: RNN序列构建信息
            
        Returns:
            (features, rnn_hidden_states, aux_loss_state)
        """
        # 如果有动作队列，直接返回队列中的动作
        if len(self.action_queue) > 0:
            action = self.action_queue.pop(0)
            # 返回one-hot编码的动作
            batch_size = observations["rgb"].shape[0]
            features = torch.zeros(batch_size, self._output_size, device=observations["rgb"].device)
            features[0, action] = 1.0  # 简单编码
            return features, rnn_hidden_states, {}
        
        # 获取当前RGB观察
        rgb_obs = observations["rgb"]  # [batch, H, W, C]
        batch_size = rgb_obs.shape[0]
        
        # 目前只支持batch_size=1
        if batch_size != 1:
            raise NotImplementedError("NaVILA policy currently only supports batch_size=1")
        
        # 转换为PIL图像
        curr_rgb = Image.fromarray(np.uint8(rgb_obs[0].cpu().numpy())).convert("RGB")
        
        # 添加到历史帧
        self.past_rgbs.append(curr_rgb)
        
        # 采样和填充图像到固定帧数
        sampled_frames = sample_and_pad_images(
            self.past_rgbs, 
            num_frames=self.num_video_frames
        )
        
        # 获取指令文本
        if self.instruction_sensor_uuid in observations:
            instruction = observations[self.instruction_sensor_uuid][0]
            if isinstance(instruction, torch.Tensor):
                instruction = instruction.item() if instruction.numel() == 1 else str(instruction)
        else:
            # 如果没有指令，使用默认导航任务
            instruction = "Navigate to the goal location"
        
        # 构建提示
        interleaved_images = "<image>\n" * (len(sampled_frames) - 1)
        question = (
            f"Imagine you are a robot programmed for navigation tasks. You have been given a video "
            f'of historical observations {interleaved_images}, and current observation <image>\n. Your assigned task is: "{instruction}" '
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
            sampled_frames, 
            self.image_processor, 
            self.model.config
        ).to(self.model.device, dtype=torch.float16)
        
        # Tokenize输入
        input_ids = (
            tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .to(self.model.device)
        )
        
        # 停止条件
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        
        # 生成输出
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=images_tensor.half().to(self.model.device),
                do_sample=False,
                temperature=0.0,
                max_new_tokens=32,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # 解码输出
        output_text = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        output_text = output_text.strip()
        
        if output_text.endswith(stop_str):
            output_text = output_text[: -len(stop_str)]
        output_text = output_text.strip()
        
        # 解析动作
        action, num_repeats = self.action_parser.parse_action(output_text)
        
        # 如果需要重复多次，将后续动作加入队列
        if num_repeats > 1:
            for _ in range(num_repeats - 1):
                self.action_queue.append(action)
        
        # 返回特征（简单编码）
        features = torch.zeros(batch_size, self._output_size, device=rgb_obs.device)
        features[0, action] = 1.0
        
        return features, rnn_hidden_states, {}
    
    def reset_history(self):
        """重置历史帧和动作队列"""
        self.past_rgbs = []
        self.action_queue = []
