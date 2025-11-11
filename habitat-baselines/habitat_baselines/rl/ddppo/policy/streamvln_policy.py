#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
StreamVLN Policy integration for Falcon framework.
Adapts StreamVLN model (Habitat 2) to Falcon's policy interface (Habitat 3).
"""

import copy
import re
import itertools
from collections import OrderedDict
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import transformers
from gym import spaces
from PIL import Image

from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ppo import Net, NetPolicy
from habitat_baselines.utils.common import get_num_actions

# StreamVLN imports
from .streamvln.model.stream_video_vln import StreamVLNForCausalLM
from .streamvln.utils.utils import (
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
    DEFAULT_MEMORY_TOKEN,
    MEMORY_TOKEN_INDEX,
    DEFAULT_VIDEO_TOKEN,
    dict_to_cuda,
    IGNORE_INDEX,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig


@baseline_registry.register_policy
class StreamVLNPolicy(NetPolicy):
    """
    StreamVLN Policy for Falcon framework.
    
    This policy integrates the StreamVLN model (trained on Habitat 2) into
    Falcon's architecture (running on Habitat 3).
    
    Key features:
    - Visual-language navigation using StreamVLN's multimodal transformer
    - Memory-augmented streaming inference
    - Action sequence generation from language instructions
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        hidden_size: int = 1024,
        num_recurrent_layers: int = 1,
        rnn_type: str = "GRU",
        policy_config: "DictConfig" = None,
        aux_loss_config: Optional["DictConfig"] = None,
        model_path: str = None,
        num_frames: int = 32,
        num_history: int = 8,
        num_future_steps: int = 4,
        model_max_length: int = 4096,
        device: str = "cuda",
        **kwargs,
    ):
        """
        Initialize StreamVLN Policy.
        
        Args:
            observation_space: Observation space from Habitat
            action_space: Action space from Habitat
            hidden_size: Hidden size for RNN (not used, kept for compatibility)
            num_recurrent_layers: Number of RNN layers (not used, kept for compatibility)
            rnn_type: RNN type (not used, kept for compatibility)
            policy_config: Policy configuration
            aux_loss_config: Auxiliary loss configuration
            model_path: Path to pretrained StreamVLN model
            num_frames: Number of frames to process before resetting memory
            num_history: Number of history frames to keep
            num_future_steps: Number of future steps for action prediction
            model_max_length: Maximum sequence length for the model
            device: Device to run the model on
        """
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

        super().__init__(
            StreamVLNNet(
                observation_space=observation_space,
                action_space=action_space,
                hidden_size=hidden_size,
                model_path=model_path,
                num_frames=num_frames,
                num_history=num_history,
                num_future_steps=num_future_steps,
                model_max_length=model_max_length,
                device=device,
                discrete_actions=discrete_actions,
            ),
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
        """Create policy from config."""
        # Exclude cameras for rendering from the observation space
        ignore_names = []
        if hasattr(config, 'habitat_baselines') and hasattr(config.habitat_baselines, 'eval'):
            if hasattr(config.habitat_baselines.eval, 'extra_sim_sensors'):
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

        agent_name = kwargs.get("agent_name")
        if agent_name is None:
            if len(config.habitat.simulator.agents_order) > 1:
                raise ValueError(
                    "If there is more than an agent, you need to specify the agent name"
                )
            else:
                agent_name = config.habitat.simulator.agents_order[0]

        # Get StreamVLN specific config
        streamvln_config = config.habitat_baselines.rl.policy.get(agent_name, {})
        
        return cls(
            observation_space=filtered_obs,
            action_space=action_space,
            hidden_size=config.habitat_baselines.rl.ppo.hidden_size,
            policy_config=config.habitat_baselines.rl.policy[agent_name],
            aux_loss_config=config.habitat_baselines.rl.auxiliary_losses,
            model_path=streamvln_config.get("model_path", None),
            num_frames=streamvln_config.get("num_frames", 32),
            num_history=streamvln_config.get("num_history", 8),
            num_future_steps=streamvln_config.get("num_future_steps", 4),
            model_max_length=streamvln_config.get("model_max_length", 4096),
            device=streamvln_config.get("device", "cuda"),
        )


class StreamVLNNet(Net):
    """
    Network architecture for StreamVLN Policy.
    
    This network wraps the StreamVLN model and adapts it to Falcon's
    policy interface. It handles:
    - Image preprocessing and encoding
    - Language instruction processing
    - Memory management for streaming inference
    - Action sequence generation and decoding
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        hidden_size: int,
        model_path: str = None,
        num_frames: int = 32,
        num_history: int = 8,
        num_future_steps: int = 4,
        model_max_length: int = 4096,
        device: str = "cuda",
        discrete_actions: bool = True,
    ):
        super().__init__()
        
        self.device = torch.device(device)
        self.discrete_actions = discrete_actions
        self._hidden_size = hidden_size
        self.num_frames = num_frames
        self.num_history = num_history
        self.num_future_steps = num_future_steps
        self.model_max_length = model_max_length
        
        # Action mapping: StreamVLN to Habitat
        # StreamVLN actions: 0=STOP, 1=MOVE_FORWARD(↑), 2=TURN_LEFT(←), 3=TURN_RIGHT(→)
        # These need to be mapped to Habitat's action space
        self.actions2idx = OrderedDict({
            'STOP': 0,
            "↑": 1,  # MOVE_FORWARD
            "←": 2,  # TURN_LEFT
            "→": 3   # TURN_RIGHT
        })
        
        # Reverse mapping for action generation
        self.idx2actions = {v: k for k, v in self.actions2idx.items()}
        
        # Initialize StreamVLN model
        if model_path is None:
            raise ValueError("model_path must be provided for StreamVLN policy")
        
        print(f"Loading StreamVLN model from {model_path}")
        
        # Load tokenizer
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_path,
            model_max_length=model_max_length,
            padding_side="right"
        )
        
        # Add special tokens
        self.tokenizer.add_tokens(["<image>"], special_tokens=True)
        self.tokenizer.add_tokens(["<memory>"], special_tokens=True)
        
        # Load model config and model
        config = transformers.AutoConfig.from_pretrained(model_path)
        self.model = StreamVLNForCausalLM.from_pretrained(
            model_path,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            config=config,
            low_cpu_mem_usage=False,
        )
        
        self.model.model.num_history = num_history
        self.model.requires_grad_(False)
        self.model.to(self.device)
        self.model.eval()
        
        # Get image processor from vision tower
        self.image_processor = self.model.get_vision_tower().image_processor
        
        # Initialize conversation template
        prompt = f"<video>\nYou are an autonomous navigation assistant. Your task is to <instruction>. Devise an action sequence to follow the instruction using the four actions: TURN LEFT (←) or TURN RIGHT (→) by 15 degrees, MOVE FORWARD (↑) by 25 centimeters, or STOP."
        answer = ""
        self.conversation = [{"from": "human", "value": prompt}, {"from": "gpt", "value": answer}]
        
        # Conjunctions for instruction formatting
        self.conjunctions = [
            'you can see ',
            'in front of you is ',
            'there is ',
            'you can spot ',
            'you are toward the ',
            'ahead of you is ',
            'in your sight is '
        ]
        
        # Episode state
        self.reset_episode_state()
        
        # Camera intrinsics (default for VLN, can be overridden)
        self.intrinsic_matrix = np.array([
            [192.0, 0.0, 191.42857143, 0.0],
            [0.0, 192.0, 191.42857143, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        
        print("StreamVLN model loaded successfully")

    def reset_episode_state(self):
        """Reset per-episode state."""
        self.rgb_list = []
        self.depth_list = []
        self.pose_list = []
        self.intrinsic_list = []
        self.time_ids = []
        self.action_seq = []
        self.output_ids = None
        self.past_key_values = None
        self.step_id = 0
        self.last_image = None
        self.current_instruction = None
        
        # Reset model cache
        # Assuming single environment for now
        self.model.reset(1)

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return False

    @property
    def num_recurrent_layers(self):
        return 1

    @property
    def recurrent_hidden_size(self):
        return self._hidden_size

    @property
    def perception_embedding_size(self):
        return self._hidden_size

    def parse_actions(self, output: str) -> List[int]:
        """Parse action sequence from model output."""
        action_patterns = '|'.join(re.escape(action) for action in self.actions2idx)
        regex = re.compile(action_patterns)
        matches = regex.findall(output)
        actions = [self.actions2idx[match] for match in matches]
        actions = itertools.chain.from_iterable(
            [a] if isinstance(a, int) else a for a in actions
        )
        return list(actions)

    def preprocess_qwen(
        self,
        sources,
        has_image: bool = False,
        system_message: str = "You are a helpful assistant.",
        add_system: bool = False
    ):
        """Preprocess inputs for Qwen model."""
        roles = {"human": "user", "gpt": "assistant"}
        image_token_index = self.tokenizer.convert_tokens_to_ids("<image>")
        memory_token_index = self.tokenizer.convert_tokens_to_ids("<memory>")
        im_start, im_end = self.tokenizer.additional_special_tokens_ids
        
        # Reset Qwen chat templates
        chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        self.tokenizer.chat_template = chat_template

        # Apply prompt templates
        conversations = []
        input_ids = []
        
        for i, source in enumerate(sources):
            prompt = self.conjunctions[0] + DEFAULT_IMAGE_TOKEN
            
            if len(source[0]["value"]) != 0:
                source[0]["value"] += f" {prompt}."
            else:
                source[0]["value"] = f"{prompt}."
            
            if roles[source[0]["from"]] != roles["human"]:
                source = source[1:]

            input_id = []
            
            if add_system:
                input_id += self.tokenizer.apply_chat_template(
                    [{"role": "system", "content": system_message}]
                )

            for conv in source:
                try:
                    role = conv["role"]
                    content = conv["content"]
                except:
                    role = conv["from"]
                    content = conv["value"]

                role = roles.get(role, role)
                conv = [{"role": role, "content": content}]
                conversations.append(content)
                encode_id = self.tokenizer.apply_chat_template(conv)
                input_id += encode_id

            for idx, encode_id in enumerate(input_id):
                if encode_id == image_token_index:
                    input_id[idx] = IMAGE_TOKEN_INDEX
                if encode_id == memory_token_index:
                    input_id[idx] = MEMORY_TOKEN_INDEX

            input_ids.append(input_id)
        
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        return input_ids, conversations

    def forward(
        self,
        observations: Dict[str, torch.Tensor],
        rnn_hidden_states,
        prev_actions,
        masks,
        rnn_build_seq_info: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass of the network.
        
        This method processes observations and generates actions using the
        StreamVLN model.
        
        Args:
            observations: Dictionary of observations from Habitat
            rnn_hidden_states: RNN hidden states (not used in StreamVLN)
            prev_actions: Previous actions
            masks: Episode masks (1 = continue, 0 = reset)
            rnn_build_seq_info: Additional info for RNN (not used)
        
        Returns:
            - Features for action/value heads
            - Updated RNN hidden states (dummy for StreamVLN)
            - Auxiliary loss state dictionary
        """
        batch_size = observations['rgb'].shape[0]
        
        # Check for episode resets
        for i in range(batch_size):
            if masks[i].item() == 0:
                # Episode reset
                self.reset_episode_state()
        
        # Extract RGB observation
        # Expected shape: [batch, height, width, channels]
        rgb = observations['rgb']
        
        # Get instruction if available
        if 'instruction' in observations:
            instruction = observations['instruction']
            if isinstance(instruction, torch.Tensor):
                # Convert tensor to string if needed
                instruction = str(instruction[0].item())
            self.current_instruction = instruction
        elif self.current_instruction is None:
            # Default instruction
            self.current_instruction = "navigate to the goal"
        
        # Process images
        images_processed = []
        for i in range(batch_size):
            rgb_np = rgb[i].cpu().numpy()
            if rgb_np.dtype != np.uint8:
                rgb_np = (rgb_np * 255).astype(np.uint8)
            
            image = Image.fromarray(rgb_np).convert('RGB')
            image_tensor = self.image_processor.preprocess(
                images=image, return_tensors='pt'
            )['pixel_values'][0]
            images_processed.append(image_tensor)
            
            # Store for history
            self.rgb_list.append(image_tensor)
        
        # Prepare depth, pose, intrinsics (dummy values for now)
        depth = torch.zeros((batch_size, rgb.shape[1], rgb.shape[2], 1))
        pose = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
        intrinsic = torch.from_numpy(self.intrinsic_matrix).float().unsqueeze(0).repeat(batch_size, 1, 1)
        
        self.depth_list.extend([d for d in depth])
        self.pose_list.extend([p for p in pose])
        self.intrinsic_list.extend([i for i in intrinsic])
        self.time_ids.append(self.step_id)
        
        # Generate actions using StreamVLN
        # For now, we'll generate a simple action
        # In a full implementation, this would call the model
        
        # Prepare dummy features for action/value heads
        # These will be used by the policy to generate actions
        features = torch.randn(batch_size, self._hidden_size).to(rgb.device)
        
        # Dummy RNN hidden states (StreamVLN doesn't use RNN in the same way)
        new_rnn_hidden_states = rnn_hidden_states
        
        aux_loss_state = {
            "perception_embed": features,
            "rnn_output": features,
        }
        
        self.step_id += 1
        
        return features, new_rnn_hidden_states, aux_loss_state

    @torch.no_grad()
    def generate_action_sequence(
        self,
        rgb: np.ndarray,
        instruction: str,
        env_idx: int = 0,
        run_model: bool = True
    ) -> Tuple[List[int], str]:
        """
        Generate action sequence from RGB observation and instruction.
        
        This is the main inference method that directly uses StreamVLN's
        generation capabilities.
        
        Args:
            rgb: RGB image (H, W, C) in uint8 format
            instruction: Navigation instruction text
            env_idx: Environment index (for multi-env support)
            run_model: Whether to run the model or reuse last image
        
        Returns:
            - List of action indices
            - Raw LLM output text
        """
        # Preprocess image
        if run_model:
            image = Image.fromarray(rgb).convert('RGB')
            image = self.image_processor.preprocess(
                images=image, return_tensors='pt'
            )['pixel_values'][0]
            self.last_image = copy.deepcopy(image)
        else:
            image = self.last_image
        
        # Prepare dummy depth, pose, intrinsics
        depth = torch.zeros((rgb.shape[0], rgb.shape[1], 1)).float()
        pose = torch.eye(4)
        intrinsic = torch.from_numpy(self.intrinsic_matrix).float()
        
        self.time_ids.append(self.step_id)
        self.rgb_list.append(image)
        self.depth_list.append(depth)
        self.pose_list.append(pose)
        self.intrinsic_list.append(intrinsic)
        
        # Reset memory if needed
        if not run_model:
            if (self.step_id + 1) % self.num_frames == 0:
                print(f'Reset model at Step {self.step_id + 1}')
                self.model.reset_for_env(env_idx)
                self.output_ids = None
                self.past_key_values = None
                self.time_ids = []
            return [0], ""  # Return dummy action if not running model
        
        # Prepare input for model
        if self.output_ids is None:
            sources = copy.deepcopy(self.conversation)
            sources[0]["value"] = sources[0]["value"].replace(
                ' Where should you go next to stay on track?',
                f' Please devise an action sequence to follow the instruction which may include turning left or right by a certain degree, moving forward by a certain distance or stopping once the task is complete.'
            )
            if self.step_id != 0:
                sources[0]["value"] += f' You have visited these areas {DEFAULT_MEMORY_TOKEN}.'
            sources[0]["value"] = sources[0]["value"].replace(DEFAULT_VIDEO_TOKEN + '\n', '')
            sources[0]["value"] = sources[0]["value"].replace('<instruction>', instruction)
            add_system = True
        else:
            sources = [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}]
            add_system = False
        
        input_ids, conversations = self.preprocess_qwen(
            [sources], has_image=True, add_system=add_system
        )
        
        if self.output_ids is not None:
            input_ids = torch.cat([self.output_ids, input_ids.to(self.output_ids.device)], dim=1)
        
        # Prepare image history
        images = self.rgb_list[-1:]
        depths = self.depth_list[-1:]
        poses = self.pose_list[-1:]
        intrinsics = self.intrinsic_list[-1:]
        
        if self.step_id != 0 and self.step_id % self.num_frames == 0:
            if self.num_history is None:
                history_ids = slice(0, self.time_ids[0], self.num_future_steps)
            else:
                history_ids = slice(0, self.time_ids[0], (self.time_ids[0] // self.num_history))
            images = self.rgb_list[history_ids] + images
            depths = self.depth_list[history_ids] + depths
            poses = self.pose_list[history_ids] + poses
            intrinsics = self.intrinsic_list[history_ids] + intrinsics
        
        # Prepare input dict
        input_dict_raw = {
            'images': torch.stack(images).unsqueeze(0),
            'depths': torch.stack(depths).unsqueeze(0),
            'poses': torch.stack(poses).unsqueeze(0),
            'intrinsics': torch.stack(intrinsics).unsqueeze(0),
            'inputs': input_ids,
            'env_id': env_idx,
            'time_ids': [self.time_ids]
        }
        
        input_dict = dict_to_cuda(input_dict_raw.copy(), self.device)
        
        for key, value in input_dict.items():
            if key in ['images', 'depths', 'poses', 'intrinsics']:
                input_dict[key] = input_dict[key].to(torch.bfloat16)
        
        # Generate
        outputs = self.model.generate(
            **input_dict,
            task_ids=[0],
            do_sample=False,
            num_beams=1,
            max_new_tokens=512,
            use_cache=True,
            return_dict_in_generate=True,
            past_key_values=self.past_key_values
        )
        
        self.output_ids = outputs.sequences
        self.past_key_values = outputs.past_key_values
        
        # Decode output
        llm_output = self.tokenizer.batch_decode(
            self.output_ids, skip_special_tokens=False
        )[0].strip()
        
        # Parse actions
        action_seq = self.parse_actions(llm_output)
        
        if len(action_seq) == 0:
            action_seq = [0]  # Default to STOP
        
        self.step_id += 1
        
        return action_seq, llm_output
