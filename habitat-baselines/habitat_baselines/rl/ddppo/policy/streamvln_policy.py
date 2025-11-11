#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import logging
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from gym import spaces
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from transformers import AutoConfig, AutoTokenizer

from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ppo.policy import Policy, PolicyActionData
from habitat_baselines.rl.streamvln.model.stream_video_vln import (
    StreamVLNForCausalLM,
)
from habitat_baselines.rl.streamvln.utils import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_MEMORY_TOKEN,
    DEFAULT_VIDEO_TOKEN,
    IMAGE_TOKEN_INDEX,
    MEMORY_TOKEN_INDEX,
    dict_to_cuda,
)


def _resolve_dtype(dtype_str: Optional[str]) -> torch.dtype:
    if dtype_str is None:
        dtype_str = "bfloat16"
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
    }
    dtype = mapping.get(dtype_str.lower(), torch.bfloat16)
    if not torch.cuda.is_available() and dtype in {
        torch.float16,
        torch.bfloat16,
    }:
        return torch.float32
    return dtype


@dataclass
class _EnvState:
    pending_actions: List[int] = field(default_factory=list)
    output_ids: Optional[torch.Tensor] = None
    past_key_values: Optional[Any] = None
    step_id: int = 0


@baseline_registry.register_policy
class StreamVLNPolicy(Policy):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        policy_config: Optional[DictConfig] = None,
        **kwargs,
    ):
        super().__init__(action_space)
        self._logger = logging.getLogger(__name__)
        self.observation_space = observation_space
        self.policy_config = (
            OmegaConf.to_container(policy_config, resolve=True)
            if isinstance(policy_config, DictConfig)
            else (policy_config or {})
        )
        stream_cfg: Dict[str, Any] = self.policy_config.get("streamvln", {})

        self.model_path: str = stream_cfg.get("model_path", "")
        if not self.model_path:
            raise ValueError(
                "StreamVLNPolicy requires `streamvln.model_path` in the policy config."
            )

        self.device = torch.device(
            stream_cfg.get(
                "device", "cuda:0" if torch.cuda.is_available() else "cpu"
            )
        )
        self.visual_dtype: torch.dtype = _resolve_dtype(stream_cfg.get("dtype"))
        self.max_new_tokens: int = int(stream_cfg.get("max_new_tokens", 512))
        self.num_beams: int = int(stream_cfg.get("num_beams", 1))
        self.do_sample: bool = bool(stream_cfg.get("do_sample", False))
        self.add_system_prompt: bool = bool(
            stream_cfg.get("add_system_prompt", True)
        )
        self.system_prompt: str = stream_cfg.get(
            "system_prompt",
            "You are a helpful assistant.",
        )
        self.attn_implementation: str = stream_cfg.get(
            "attn_implementation", "flash_attention_2"
        )
        self.tokenizer_max_length: int = int(
            stream_cfg.get("tokenizer_max_length", 4096)
        )
        self.use_memory_tokens: bool = bool(
            stream_cfg.get("use_memory_tokens", True)
        )
        self.num_frames: int = int(stream_cfg.get("num_frames", 32))
        self.num_history: Optional[int] = stream_cfg.get("num_history", 8)
        self.num_future_steps: int = int(stream_cfg.get("num_future_steps", 4))
        self.max_envs: int = int(stream_cfg.get("max_envs", 16))
        self.prompts_conjunctions: List[str] = stream_cfg.get(
            "conjunctions",
            [
                "you can see ",
                "in front of you is ",
                "there is ",
                "you can spot ",
                "you are toward the ",
                "ahead of you is ",
                "in your sight is ",
            ],
        )

        self.prompt_template: str = stream_cfg.get(
            "prompt_template",
            "<video>\nYou are an autonomous navigation assistant. Your task is to <instruction>. Devise an action sequence to follow the instruction using the four actions: TURN LEFT (←) or TURN RIGHT (→) by 15 degrees, MOVE FORWARD (↑) by 25 centimeters, or STOP.",
        )
        self.default_action_token: str = stream_cfg.get(
            "default_action_token", "STOP"
        )

        self._load_model(stream_cfg)

        self.action_token_map: OrderedDict[str, int] = OrderedDict(
            stream_cfg.get(
                "action_token_map",
                {
                    "STOP": 0,
                    "↑": 1,
                    "←": 2,
                    "→": 3,
                },
            )
        )
        self._validate_action_map(action_space)
        self.action_regex = re.compile(
            "|".join(re.escape(tok) for tok in self.action_token_map.keys())
        )

        self.default_action: int = self.action_token_map.get(
            self.default_action_token, next(iter(self.action_token_map.values()))
        )

        self.rgb_key = self._resolve_sensor_key(stream_cfg.get("rgb_sensor"), "rgb")
        if self.rgb_key is None:
            raise ValueError(
                "StreamVLNPolicy could not determine the RGB sensor key. "
                "Please set `streamvln.rgb_sensor` in the policy config."
            )
        self.depth_key = self._resolve_sensor_key(
            stream_cfg.get("depth_sensor"), "depth"
        )
        self.instruction_key = stream_cfg.get("instruction_sensor", "instruction")
        if (
            self.instruction_key not in observation_space.spaces
            and self.instruction_key is not None
        ):
            # Allow missing instruction sensor but warn the user.
            self._logger.warning(
                "Instruction sensor `%s` not present in observation space. "
                "Falling back to empty instructions.",
                self.instruction_key,
            )
            self.instruction_key = None

        intrinsic = stream_cfg.get("intrinsic_matrix")
        if intrinsic is None:
            self.intrinsic_matrix = torch.eye(4, dtype=torch.float32)
        else:
            intrinsic_arr = np.array(intrinsic, dtype=np.float32)
            if intrinsic_arr.size != 16:
                raise ValueError(
                    "Expected `intrinsic_matrix` to describe a 4x4 matrix."
                )
            self.intrinsic_matrix = torch.from_numpy(
                intrinsic_arr.reshape(4, 4)
            )

        self.pose_template = torch.eye(4, dtype=torch.float32)

        self.conversation_template = [
            {"from": "human", "value": self.prompt_template},
            {"from": "gpt", "value": ""},
        ]

        self.env_states: List[_EnvState] = []
        self.current_env_count: int = 0

    def _load_model(self, stream_cfg: Dict[str, Any]) -> None:
        import importlib
        import sys

        self._logger.info("Loading StreamVLN model from %s", self.model_path)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            model_max_length=self.tokenizer_max_length,
            padding_side="right",
        )
        if "<image>" not in self.tokenizer.get_vocab():
            self.tokenizer.add_tokens(["<image>"], special_tokens=True)
        if "<memory>" not in self.tokenizer.get_vocab():
            self.tokenizer.add_tokens(["<memory>"], special_tokens=True)

        base_llava_pkg = importlib.import_module(
            "habitat_baselines.rl.streamvln.llava"
        )
        if "llava" not in sys.modules:
            sys.modules["llava"] = base_llava_pkg

        model_config = AutoConfig.from_pretrained(self.model_path)
        model = StreamVLNForCausalLM.from_pretrained(
            self.model_path,
            attn_implementation=self.attn_implementation,
            torch_dtype=self.visual_dtype,
            config=model_config,
            low_cpu_mem_usage=False,
        )
        model.model.num_history = self.num_history
        model.reset(self.max_envs)
        model.requires_grad_(False)
        model.to(self.device)
        model.eval()
        self.model = model
        self.image_processor = self.model.get_vision_tower().image_processor

    def _resolve_sensor_key(
        self, preferred_key: Optional[str], substring: str
    ) -> Optional[str]:
        if preferred_key:
            return preferred_key
        for key in self.observation_space.spaces.keys():
            if substring in key:
                return key
        return None

    def _validate_action_map(self, action_space) -> None:
        if not hasattr(action_space, "n"):
            raise ValueError(
                "StreamVLNPolicy currently supports discrete action spaces."
            )
        for token, action_idx in self.action_token_map.items():
            if action_idx < 0 or action_idx >= action_space.n:
                raise ValueError(
                    f"Action index {action_idx} mapped from token `{token}` "
                    f"is outside of the action space size {action_space.n}."
                )

    @property
    def should_load_agent_state(self) -> bool:
        return False

    def _ensure_env_states(self, num_envs: int) -> None:
        if self.current_env_count == num_envs:
            return
        self._logger.info(
            "Resetting StreamVLN model for %d environments.", num_envs
        )
        self.model.reset(num_envs)
        self.env_states = [_EnvState() for _ in range(num_envs)]
        self.current_env_count = num_envs

    def _reset_env(self, env_idx: int) -> None:
        if env_idx >= len(self.env_states):
            return
        self.env_states[env_idx] = _EnvState()
        if hasattr(self.model, "reset_for_env"):
            self.model.reset_for_env(env_idx)

    def _extract_instruction(
        self, observations: Dict[str, Any], env_idx: int
    ) -> str:
        if self.instruction_key is None:
            return ""
        if self.instruction_key not in observations:
            return ""
        value = observations[self.instruction_key]
        if isinstance(value, (list, tuple)):
            value = value[env_idx]
        elif isinstance(value, torch.Tensor):
            value = value[env_idx]
        elif isinstance(value, np.ndarray):
            value = value[env_idx]

        if isinstance(value, torch.Tensor):
            value = value.detach().cpu()
            if value.dtype == torch.int64 or value.dtype == torch.int32:
                arr = value.numpy()
                try:
                    return "".join(chr(c) for c in arr if c != 0)
                except ValueError:
                    return " ".join(str(int(c)) for c in arr if c != 0)
            elif value.dtype == torch.uint8:
                arr = value.numpy().tobytes()
                return arr.decode("utf-8", errors="ignore")
            else:
                return str(value.numpy())
        if isinstance(value, np.ndarray):
            if value.dtype.kind in {"U", "S", "O"}:
                return str(value.item())
            return "".join(chr(int(c)) for c in value.flatten() if c != 0)
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="ignore")
        if isinstance(value, str):
            return value
        return str(value)

    def _extract_rgb(
        self, observations: Dict[str, Any], env_idx: int
    ) -> np.ndarray:
        rgb_obs = observations[self.rgb_key]
        if isinstance(rgb_obs, torch.Tensor):
            data = rgb_obs.detach().cpu()
            if data.ndim == 4:
                data = data[env_idx]
            elif data.ndim != 3:
                raise ValueError(
                    f"Expected RGB observation with 3 or 4 dimensions, got {data.ndim}."
                )
            if data.shape[0] in {1, 3} and data.shape[-1] != 3:
                data = data.permute(1, 2, 0)
            elif data.shape[-1] == 3:
                data = data
            else:
                raise ValueError(
                    f"Unable to determine channel dimension for RGB shape {tuple(data.shape)}."
                )
            if data.dtype in {torch.float32, torch.float16, torch.bfloat16}:
                data = data.clamp(0.0, 1.0) * 255.0
            data = data.to(torch.uint8)
            return data.numpy()

        if isinstance(rgb_obs, np.ndarray):
            data = rgb_obs
            if data.ndim == 4:
                data = data[env_idx]
            if data.shape[0] in {1, 3} and data.shape[-1] != 3:
                data = np.transpose(data, (1, 2, 0))
            if data.dtype != np.uint8:
                data = np.clip(data, 0.0, 1.0) * 255.0
                data = data.astype(np.uint8)
            return data

        raise TypeError(f"Unsupported RGB data type: {type(rgb_obs)}")

    def _extract_depth(
        self, observations: Dict[str, Any], env_idx: int, shape_hw: Tuple[int, int]
    ) -> torch.Tensor:
        if self.depth_key is None or self.depth_key not in observations:
            h, w = shape_hw
            return torch.zeros((h, w, 1), dtype=torch.float32)
        depth_obs = observations[self.depth_key]
        if isinstance(depth_obs, torch.Tensor):
            data = depth_obs.detach().cpu()
            if data.ndim == 4:
                data = data[env_idx]
            elif data.ndim != 3 and data.ndim != 2:
                raise ValueError(
                    f"Expected depth tensor with 2 or 3 dimensions, got {data.ndim}."
                )
            if data.ndim == 2:
                data = data.unsqueeze(-1)
            if data.dtype != torch.float32:
                data = data.to(torch.float32)
            return data

        if isinstance(depth_obs, np.ndarray):
            data = depth_obs
            if data.ndim == 4:
                data = data[env_idx]
            if data.ndim == 2:
                data = data[..., None]
            if data.dtype != np.float32:
                data = data.astype(np.float32)
            return torch.from_numpy(data)

        raise TypeError(f"Unsupported depth data type: {type(depth_obs)}")

    def _preprocess_image(self, rgb_np: np.ndarray) -> torch.Tensor:
        image = Image.fromarray(rgb_np.astype(np.uint8)).convert("RGB")
        processed = self.image_processor.preprocess(
            images=image, return_tensors="pt"
        )["pixel_values"][0]
        return processed

    def _build_sources(self, env_state: _EnvState, instruction: str):
        if env_state.output_ids is None:
            sources = copy.deepcopy(self.conversation_template)
            instruction = instruction.strip()
            prompt = self.prompt_template.replace(DEFAULT_VIDEO_TOKEN + "\n", "")
            prompt = prompt.replace("<instruction>", instruction)
            if self.use_memory_tokens and env_state.step_id != 0:
                prompt += f" You have visited these areas {DEFAULT_MEMORY_TOKEN}."
            if self.prompts_conjunctions:
                prompt += f" {self.prompts_conjunctions[0]}{DEFAULT_IMAGE_TOKEN}."
            else:
                prompt += f" {DEFAULT_IMAGE_TOKEN}."
            sources[0]["value"] = prompt
            add_system = self.add_system_prompt
        else:
            sources = [
                {"from": "human", "value": ""},
                {"from": "gpt", "value": ""},
            ]
            add_system = False

        return sources, add_system

    def _preprocess_qwen(
        self, sources: List[Dict[str, str]], add_system: bool
    ) -> torch.Tensor:
        roles = {"human": "user", "gpt": "assistant"}
        image_token_index = self.tokenizer.convert_tokens_to_ids("<image>")
        memory_token_index = self.tokenizer.convert_tokens_to_ids("<memory>")
        im_start, im_end = self.tokenizer.additional_special_tokens_ids

        chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
        self.tokenizer.chat_template = chat_template

        input_id: List[int] = []
        if add_system:
            input_id += self.tokenizer.apply_chat_template(
                [{"role": "system", "content": self.system_prompt}]
            )

        for conv in sources:
            role = roles.get(conv.get("role", conv.get("from", "human")), "user")
            content = conv.get("content", conv.get("value", ""))
            encoded = self.tokenizer.apply_chat_template(
                [{"role": role, "content": content}]
            )
            input_id += encoded

        for idx, token in enumerate(input_id):
            if token == image_token_index:
                input_id[idx] = IMAGE_TOKEN_INDEX
            if token == memory_token_index:
                input_id[idx] = MEMORY_TOKEN_INDEX

        return torch.tensor([input_id], dtype=torch.long)

    def _run_streamvln(
        self,
        env_idx: int,
        env_state: _EnvState,
        rgb_np: np.ndarray,
        depth_tensor: torch.Tensor,
        instruction: str,
    ) -> Tuple[List[int], str]:
        pixel_values = self._preprocess_image(rgb_np)
        depth = depth_tensor
        pose = self.pose_template.clone()
        intrinsic = self.intrinsic_matrix.clone()

        sources, add_system = self._build_sources(env_state, instruction)
        input_ids = self._preprocess_qwen(sources, add_system)
        if env_state.output_ids is not None:
            input_ids = torch.cat(
                [env_state.output_ids, input_ids.to(env_state.output_ids.device)],
                dim=1,
            )

        images = pixel_values.unsqueeze(0).unsqueeze(0)
        depths = depth.unsqueeze(0).unsqueeze(0)
        poses = pose.unsqueeze(0).unsqueeze(0)
        intrinsics = intrinsic.unsqueeze(0).unsqueeze(0)
        time_ids = [[env_state.step_id]]

        input_dict = {
            "inputs": input_ids,
            "images": images,
            "depths": depths,
            "poses": poses,
            "intrinsics": intrinsics,
            "env_id": env_idx,
            "time_ids": time_ids,
        }

        input_dict = dict_to_cuda(input_dict, self.device)
        if torch.is_tensor(input_dict["images"]):
            input_dict["images"] = input_dict["images"].to(
                self.device, dtype=self.visual_dtype
            )
        if torch.is_tensor(input_dict["depths"]):
            input_dict["depths"] = input_dict["depths"].to(
                self.device, dtype=self.visual_dtype
            )
        if torch.is_tensor(input_dict["poses"]):
            input_dict["poses"] = input_dict["poses"].to(
                self.device, dtype=self.visual_dtype
            )
        if torch.is_tensor(input_dict["intrinsics"]):
            input_dict["intrinsics"] = input_dict["intrinsics"].to(
                self.device, dtype=self.visual_dtype
            )

        outputs = self.model.generate(
            task_type=[0],
            do_sample=self.do_sample,
            num_beams=self.num_beams,
            max_new_tokens=self.max_new_tokens,
            use_cache=True,
            return_dict_in_generate=True,
            past_key_values=env_state.past_key_values,
            **input_dict,
        )

        env_state.output_ids = outputs.sequences
        env_state.past_key_values = outputs.past_key_values
        env_state.step_id += 1

        llm_outputs = self.tokenizer.batch_decode(
            outputs.sequences, skip_special_tokens=False
        )[0].strip()
        action_seq = [
            self.action_token_map[token]
            for token in self.action_regex.findall(llm_outputs)
            if token in self.action_token_map
        ]
        if not action_seq:
            action_seq = [self.default_action]
        return action_seq, llm_outputs

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic: bool = False,
    ) -> PolicyActionData:
        if isinstance(observations, dict):
            num_envs = next(iter(observations.values())).shape[0]
        else:
            raise ValueError("Expected observations to be a dict for StreamVLN.")

        self._ensure_env_states(num_envs)

        action_device = (
            prev_actions.device
            if isinstance(prev_actions, torch.Tensor)
            else torch.device("cpu")
        )
        actions = torch.zeros(
            (num_envs, 1), dtype=torch.long, device=action_device
        )
        policy_info: List[Dict[str, Any]] = []

        mask_tensor = masks if isinstance(masks, torch.Tensor) else None
        if mask_tensor is not None:
            mask_tensor = mask_tensor.detach().cpu()

        for env_idx in range(num_envs):
            if mask_tensor is not None and mask_tensor[env_idx].item() == 0:
                self._reset_env(env_idx)

            env_state = self.env_states[env_idx]
            if not env_state.pending_actions:
                instruction = self._extract_instruction(observations, env_idx)
                rgb_np = self._extract_rgb(observations, env_idx)
                depth_tensor = self._extract_depth(
                    observations, env_idx, rgb_np.shape[:2]
                )
                pending, llm_outputs = self._run_streamvln(
                    env_idx, env_state, rgb_np, depth_tensor, instruction
                )
                env_state.pending_actions = pending
                policy_info.append(
                    {"generated": True, "llm_output": llm_outputs}
                )
            else:
                policy_info.append({"generated": False})

            action = env_state.pending_actions.pop(0)
            actions[env_idx, 0] = action

        should_inserts = torch.zeros(
            (num_envs, 1), dtype=torch.bool, device=actions.device
        )

        return PolicyActionData(
            actions=actions,
            rnn_hidden_states=rnn_hidden_states,
            policy_info=policy_info,
            should_inserts=should_inserts,
        )

    @classmethod
    def from_config(
        cls,
        config: "DictConfig",
        observation_space: spaces.Dict,
        action_space,
        **kwargs,
    ):
        agent_name = kwargs.get("agent_name")
        if agent_name is None:
            if len(config.habitat.simulator.agents_order) > 1:
                raise ValueError(
                    "If there is more than one agent, you need to specify the agent name."
                )
            agent_name = config.habitat.simulator.agents_order[0]

        policy_cfg = config.habitat_baselines.rl.policy[agent_name]
        return cls(
            observation_space=observation_space,
            action_space=action_space,
            policy_config=policy_cfg,
        )
