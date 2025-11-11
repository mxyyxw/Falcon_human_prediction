#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import copy
import logging
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from gym import spaces
from PIL import Image
from torch import Tensor

from habitat.core.spaces import ActionSpace
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ppo.policy import Policy, PolicyActionData

try:
    from transformers import AutoConfig, AutoTokenizer
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "StreamVLNPolicy requires the 'transformers' package. "
        "Please install it via `pip install -r requirements.txt`."
    ) from exc

LOGGER = logging.getLogger(__name__)


def _workspace_root(file_path: Path) -> Path:
    """
    Resolve the workspace root (repository root) from a file path inside
    habitat-baselines. We walk up five levels from the policy directory.
    """
    current = file_path.resolve()
    for _ in range(5):
        current = current.parent
    return current


def _ensure_streamvln_on_path(policy_file: Path) -> Path:
    """
    Adds the third_party/streamvln directory to sys.path so that the original
    StreamVLN modules (llava, streamvln, utils, etc.) can be imported without
    modification.
    """
    repo_root = _workspace_root(policy_file)
    streamvln_root = repo_root / "third_party" / "streamvln"
    if not streamvln_root.exists():
        raise RuntimeError(
            "未找到 StreamVLN 第三方代码目录，请确认已经在项目根目录下复制 `third_party/streamvln`。"
        )
    if str(streamvln_root) not in sys.path:
        sys.path.insert(0, str(streamvln_root))
    return streamvln_root


STREAMVLN_ROOT = _ensure_streamvln_on_path(Path(__file__))

# Import after sys.path adjustment
from streamvln.model.stream_video_vln import StreamVLNForCausalLM  # noqa: E402
from streamvln.utils.utils import DEFAULT_IMAGE_TOKEN  # noqa: E402
from streamvln.utils.utils import DEFAULT_MEMORY_TOKEN  # noqa: E402
from streamvln.utils.utils import IMAGE_TOKEN_INDEX  # noqa: E402
from streamvln.utils.utils import MEMORY_TOKEN_INDEX  # noqa: E402
from streamvln.utils.utils import dict_to_cuda  # noqa: E402


@dataclass
class StreamVLNAgentState:
    """
    保存每个环境实例的缓存状态，复用 StreamVLN 原始推理逻辑。
    """

    rgb_list: List[torch.Tensor] = field(default_factory=list)
    depth_list: List[torch.Tensor] = field(default_factory=list)
    pose_list: List[torch.Tensor] = field(default_factory=list)
    intrinsic_list: List[torch.Tensor] = field(default_factory=list)
    time_ids: List[int] = field(default_factory=list)
    action_seq: List[int] = field(default_factory=list)
    output_ids: Optional[torch.Tensor] = None
    past_key_values: Optional[Any] = None
    step_id: int = 0
    last_image: Optional[torch.Tensor] = None

    def reset(self) -> None:
        self.rgb_list.clear()
        self.depth_list.clear()
        self.pose_list.clear()
        self.intrinsic_list.clear()
        self.time_ids.clear()
        self.action_seq.clear()
        self.output_ids = None
        self.past_key_values = None
        self.step_id = 0
        self.last_image = None


def _to_numpy_image(obs: Union[Tensor, np.ndarray]) -> np.ndarray:
    """
    将 habitat 返回的 RGB 观测统一转为 uint8 numpy 数组，形状 [H, W, 3]。
    """
    if isinstance(obs, torch.Tensor):
        obs = obs.detach().cpu().numpy()
    if obs.dtype in (np.float32, np.float64):
        obs = np.clip(obs * 255.0, 0.0, 255.0).astype(np.uint8)
    elif obs.dtype != np.uint8:
        obs = obs.astype(np.uint8)
    if obs.ndim == 3 and obs.shape[0] in (3, 4):
        # Convert from CHW to HWC
        obs = np.transpose(obs[:3], (1, 2, 0))
    return obs


def _to_depth_tensor(obs: Optional[Union[Tensor, np.ndarray]], shape: Tuple[int, int]) -> torch.Tensor:
    """
    将深度观测转成 torch.Tensor，如果缺失则使用全零。
    """
    if obs is None:
        return torch.zeros((*shape, 1), dtype=torch.float32)
    if isinstance(obs, torch.Tensor):
        depth = obs.detach().cpu()
    else:
        depth = torch.from_numpy(obs)
    if depth.ndim == 2:
        depth = depth.unsqueeze(-1)
    depth = depth.float()
    return depth


def _expand_intrinsic(intrinsic: torch.Tensor) -> torch.Tensor:
    """
    确保内参矩阵为 float32，形状 [4, 4]。
    """
    intrinsic = intrinsic.float()
    if intrinsic.ndim == 2 and intrinsic.shape == (4, 4):
        return intrinsic
    raise ValueError("摄像机内参需要为 4x4 矩阵。")


@baseline_registry.register_policy
class StreamVLNPolicy(nn.Module, Policy):
    """
    将 StreamVLN 多模态语言导航模型集成到 Falcon (Habitat 3) 的策略接口中。
    该策略当前仅用于推理（evaluation/inference），不参与梯度更新。
    """

    def __init__(
        self,
        config,
        full_config,
        observation_space: spaces.Space,
        action_space: ActionSpace,
        orig_action_space: ActionSpace,
        num_envs: int,
        aux_loss_config,
        agent_name: Optional[str],
    ):
        Policy.__init__(self, action_space)
        nn.Module.__init__(self)

        self._num_envs: int = num_envs
        self._device = torch.device(
            getattr(config, "device", "cuda" if torch.cuda.is_available() else "cpu")
        )
        self._recurrent_hidden_size = 1
        self._action_space = action_space
        self.action_distribution_type = "categorical"

        # 配置参数
        self.model_path: str = getattr(config, "model_path", "")
        if not self.model_path:
            raise ValueError(
                "必须在策略配置中提供 `model_path`，指向训练好的 StreamVLN checkpoint。"
            )
        self.num_frames: int = getattr(config, "num_frames", 32)
        self.num_future_steps: int = getattr(config, "num_future_steps", 4)
        self.num_history: Optional[int] = getattr(config, "num_history", 8)
        self.model_max_length: int = getattr(config, "model_max_length", 4096)
        self.use_memory_tokens: bool = getattr(config, "use_memory_tokens", True)
        self.reset_every_episode: bool = getattr(config, "reset_every_episode", True)
        self.use_depth: bool = getattr(config, "use_depth", False)
        self.attn_impl: str = getattr(
            config, "attn_implementation", "flash_attention_2"
        )
        self.torch_dtype: torch.dtype = (
            torch.bfloat16
            if getattr(config, "use_bfloat16", True)
            else torch.float16
        )
        intrinsic_cfg = getattr(config, "intrinsic", None)
        if intrinsic_cfg is None:
            default_intrinsic = [
                [192.0, 0.0, 191.42857, 0.0],
                [0.0, 192.0, 191.42857, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
            intrinsic_cfg = default_intrinsic
        self.intrinsic_matrix = _expand_intrinsic(
            torch.tensor(intrinsic_cfg, dtype=torch.float32)
        )

        # 观测键
        self.rgb_sensor: str = getattr(config, "rgb_sensor", "rgb")
        self.depth_sensor: Optional[str] = getattr(config, "depth_sensor", None)
        self.instruction_sensor: str = getattr(
            config, "instruction_sensor", "instruction"
        )

        # 行动映射
        self.actions2idx = OrderedDict(
            {
                "STOP": [0],
                "↑": [1],
                "←": [2],
                "→": [3],
            }
        )

        self.conjunctions = getattr(
            config,
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

        self.base_prompt = (
            "<video>\n"
            "You are an autonomous navigation assistant. "
            "Your task is to <instruction>. "
            "Devise an action sequence to follow the instruction using the four actions: "
            "TURN LEFT (←) or TURN RIGHT (→) by 15 degrees, "
            "MOVE FORWARD (↑) by 25 centimeters, or STOP."
        )
        self.conversation_template = [
            {"from": "human", "value": self.base_prompt},
            {"from": "gpt", "value": ""},
        ]

        # 每个环境的状态缓存
        self._env_states: List[StreamVLNAgentState] = [
            StreamVLNAgentState() for _ in range(self._num_envs)
        ]

        # 加载 tokenizer & 模型
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            model_max_length=self.model_max_length,
            padding_side="right",
        )
        self.tokenizer.add_tokens(["<image>"], special_tokens=True)
        self.tokenizer.add_tokens(["<memory>"], special_tokens=True)

        config_hf = AutoConfig.from_pretrained(self.model_path)
        LOGGER.info("加载 StreamVLN 模型配置完成。")
        self.model = StreamVLNForCausalLM.from_pretrained(
            self.model_path,
            attn_implementation=self.attn_impl,
            torch_dtype=self.torch_dtype,
            config=config_hf,
            low_cpu_mem_usage=False,
        )
        self.model.model.num_history = self.num_history
        self.model.reset(self._num_envs)
        self.model.requires_grad_(False)
        self.model.to(self._device)
        self.model.eval()

        self.image_processor = self.model.get_vision_tower().image_processor
        LOGGER.info(
            "StreamVLNPolicy 初始化完成，设备：%s，模型路径：%s", self._device, self.model_path
        )

    # ------------------------------------------------------------------
    # 基类接口
    # ------------------------------------------------------------------
    @classmethod
    def from_config(
        cls,
        config,
        observation_space,
        action_space,
        orig_action_space,
        agent_name=None,
        **kwargs,
    ):
        if agent_name is None:
            if len(config.habitat.simulator.agents_order) > 1:
                raise ValueError(
                    "多智能体场景需要在配置中显式指定 agent_name。"
                )
            agent_name = config.habitat.simulator.agents_order[0]

        policy_cfg = config.habitat_baselines.rl.policy[agent_name]
        return cls(
            config=policy_cfg,
            full_config=config,
            observation_space=observation_space,
            action_space=action_space,
            orig_action_space=orig_action_space,
            num_envs=config.habitat_baselines.num_environments,
            aux_loss_config=config.habitat_baselines.rl.auxiliary_losses,
            agent_name=agent_name,
        )

    def to(self, device: Union[str, torch.device]) -> "StreamVLNPolicy":
        super().to(device)
        self._device = torch.device(device)
        self.model.to(self._device)
        return self

    @property
    def hidden_state_shape(self) -> Tuple[int, int]:
        return (1, self._recurrent_hidden_size)

    @property
    def hidden_state_shape_lens(self) -> List[int]:
        return [self._recurrent_hidden_size]

    @property
    def recurrent_hidden_size(self) -> int:
        return self._recurrent_hidden_size

    @property
    def num_recurrent_layers(self) -> int:
        return 1

    @property
    def should_load_agent_state(self) -> bool:
        return False

    def parameters(self) -> Iterable[Tensor]:
        # 模型仅用于推理，不暴露可训练参数
        return []

    def get_value(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
    ) -> torch.Tensor:
        batch = rnn_hidden_states.shape[0]
        return torch.zeros(batch, 1, device=rnn_hidden_states.device)

    # ------------------------------------------------------------------
    # 推理核心逻辑
    # ------------------------------------------------------------------
    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic: bool = False,
        **kwargs,
    ) -> PolicyActionData:
        batch_size = masks.shape[0]
        actions = torch.zeros_like(prev_actions, dtype=torch.int64)
        policy_info: List[Dict[str, Any]] = []

        for env_idx in range(batch_size):
            if masks[env_idx].item() == 0 and self.reset_every_episode:
                self._reset_env_state(env_idx)

            obs_env = self._slice_observations(observations, env_idx)
            try:
                action_idx, llm_output = self._run_streamvln_step(
                    env_idx, obs_env
                )
            except Exception as exc:  # pragma: no cover - robustness
                LOGGER.exception(
                    "StreamVLN 推理失败（env %d），回退为 STOP：%s", env_idx, exc
                )
                action_idx = 0
                llm_output = f"error: {exc}"

            actions[env_idx] = action_idx
            policy_info.append({"streamvln_output": llm_output})

        return PolicyActionData(
            take_actions=actions,
            actions=actions,
            rnn_hidden_states=rnn_hidden_states,
            policy_info=policy_info,
        )

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    def _slice_observations(
        self, observations: Dict[str, Any], env_idx: int
    ) -> Dict[str, Any]:
        sliced: Dict[str, Any] = {}
        for key, value in observations.items():
            if isinstance(value, dict):
                sliced[key] = {
                    sub_key: self._slice_single(sub_value, env_idx)
                    for sub_key, sub_value in value.items()
                }
            else:
                sliced[key] = self._slice_single(value, env_idx)
        return sliced

    @staticmethod
    def _slice_single(value: Any, env_idx: int) -> Any:
        if isinstance(value, torch.Tensor) and value.shape[0] > env_idx:
            return value[env_idx]
        if isinstance(value, np.ndarray) and value.shape[0] > env_idx:
            return value[env_idx]
        if isinstance(value, (list, tuple)) and len(value) > env_idx:
            return value[env_idx]
        return value

    def _reset_env_state(self, env_idx: int) -> None:
        LOGGER.debug("重置 StreamVLN 环境状态：env=%d", env_idx)
        self._env_states[env_idx].reset()
        self.model.reset_for_env(env_idx)

    def _run_streamvln_step(
        self, env_idx: int, observations: Dict[str, Any]
    ) -> Tuple[int, str]:
        state = self._env_states[env_idx]

        instruction_text = self._extract_instruction(observations)
        if not instruction_text:
            LOGGER.debug("env %d 缺少指令文本，默认 STOP", env_idx)
            return 0, ""

        rgb_obs = observations.get(self.rgb_sensor)
        if rgb_obs is None:
            LOGGER.debug("env %d 缺少 RGB 观测，默认 STOP", env_idx)
            return 0, ""

        rgb_np = _to_numpy_image(rgb_obs)
        image = Image.fromarray(rgb_np).convert("RGB")
        processed = self.image_processor.preprocess(
            images=image, return_tensors="pt"
        )["pixel_values"][0]

        state.last_image = processed.clone()

        depth_obs = None
        if self.use_depth and self.depth_sensor is not None:
            depth_obs = observations.get(self.depth_sensor, None)

        depth_tensor = _to_depth_tensor(
            depth_obs, shape=(rgb_np.shape[0], rgb_np.shape[1])
        )

        pose_tensor = torch.eye(4, dtype=torch.float32)
        intrinsic_tensor = self.intrinsic_matrix.clone()

        state.time_ids.append(state.step_id)
        state.rgb_list.append(processed)
        state.depth_list.append(depth_tensor)
        state.pose_list.append(pose_tensor)
        state.intrinsic_list.append(intrinsic_tensor)

        sources, add_system = self._build_conversation_sources(
            state, instruction_text
        )
        input_ids, _ = self._preprocess_qwen([sources], add_system=add_system)

        if state.output_ids is not None:
            input_ids = torch.cat(
                [state.output_ids, input_ids.to(state.output_ids.device)], dim=1
            )

        images_tensor = torch.stack(state.rgb_list[-1:]).unsqueeze(0)
        depths_tensor = torch.stack(state.depth_list[-1:]).unsqueeze(0)
        poses_tensor = torch.stack(state.pose_list[-1:]).unsqueeze(0)
        intrinsics_tensor = torch.stack(state.intrinsic_list[-1:]).unsqueeze(0)

        if (
            self.use_memory_tokens
            and state.step_id != 0
            and state.step_id % self.num_frames == 0
        ):
            if self.num_history is None:
                history_ids = slice(
                    0, state.time_ids[0], self.num_future_steps
                )
            else:
                history_ids = slice(
                    0,
                    state.time_ids[0],
                    max(1, state.time_ids[0] // self.num_history),
                )
            history_images = state.rgb_list[history_ids]
            history_depths = state.depth_list[history_ids]
            history_poses = state.pose_list[history_ids]
            history_intrinsics = state.intrinsic_list[history_ids]

            images_tensor = torch.stack(history_images + state.rgb_list[-1:])
            images_tensor = images_tensor.unsqueeze(0)
            depths_tensor = torch.stack(history_depths + state.depth_list[-1:])
            depths_tensor = depths_tensor.unsqueeze(0)
            poses_tensor = torch.stack(history_poses + state.pose_list[-1:])
            poses_tensor = poses_tensor.unsqueeze(0)
            intrinsics_tensor = torch.stack(
                history_intrinsics + state.intrinsic_list[-1:]
            )
            intrinsics_tensor = intrinsics_tensor.unsqueeze(0)

        input_dict_raw = {
            "images": images_tensor,
            "depths": depths_tensor,
            "poses": poses_tensor,
            "intrinsics": intrinsics_tensor,
            "inputs": input_ids,
            "env_id": env_idx,
            "time_ids": [state.time_ids],
        }
        input_dict = dict_to_cuda(
            copy.deepcopy(input_dict_raw), device=self._device
        )

        for key in ["images", "depths", "poses", "intrinsics"]:
            input_dict[key] = input_dict[key].to(self.torch_dtype)

        outputs = self.model.generate(
            **input_dict,
            task_type=[0],
            do_sample=False,
            num_beams=1,
            max_new_tokens=64,
            use_cache=True,
            return_dict_in_generate=True,
            past_key_values=state.past_key_values,
        )

        state.output_ids = outputs.sequences
        state.past_key_values = outputs.past_key_values
        decoded = self.tokenizer.batch_decode(
            state.output_ids, skip_special_tokens=False
        )[0].strip()

        actions = self._parse_actions(decoded)
        state.action_seq = actions
        state.step_id += 1

        if (
            self.use_memory_tokens
            and (state.step_id) % self.num_frames == 0
            and state.step_id != 0
        ):
            self.model.reset_for_env(env_idx)

        if len(actions) == 0:
            return 0, decoded
        return actions[0], decoded

    def _extract_instruction(self, observations: Dict[str, Any]) -> str:
        instruction_obs = observations.get(self.instruction_sensor)
        if instruction_obs is None:
            return ""
        if isinstance(instruction_obs, dict):
            value = instruction_obs.get("text")
            if value:
                return str(value)
        if isinstance(instruction_obs, (list, tuple)) and instruction_obs:
            return str(instruction_obs[0])
        if isinstance(instruction_obs, str):
            return instruction_obs
        return ""

    def _build_conversation_sources(
        self, state: StreamVLNAgentState, instruction_text: str
    ) -> Tuple[List[Dict[str, str]], bool]:
        if state.output_ids is None:
            sources = copy.deepcopy(self.conversation_template)
            sources[0]["value"] = sources[0]["value"].replace(
                " <instruction>.",
                f" {instruction_text}.",
            )
            if state.step_id != 0:
                sources[0]["value"] += (
                    f" You have visited these areas {DEFAULT_MEMORY_TOKEN}."
                )
            sources[0]["value"] = sources[0]["value"].replace(
                "<instruction>.", instruction_text
            )
            add_system = True
        else:
            sources = [{"from": "human", "value": ""}]
            add_system = False
        return sources, add_system

    def _preprocess_qwen(
        self,
        sources: List[List[Dict[str, str]]],
        has_image: bool = True,
        add_system: bool = False,
    ) -> Tuple[torch.Tensor, List[str]]:
        roles = {"human": "user", "gpt": "assistant"}
        image_token_index = self.tokenizer.convert_tokens_to_ids("<image>")
        memory_token_index = self.tokenizer.convert_tokens_to_ids("<memory>")
        im_start, im_end = self.tokenizer.additional_special_tokens_ids[:2]
        unmask_tokens_idx = [198, im_start, im_end]
        (nl_token,) = self.tokenizer("\n").input_ids

        chat_template = (
            "{% for message in messages %}"
            "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] "
            "+ '<|im_end|>' + '\\n'}}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|im_start|>assistant\\n' }}"
            "{% endif %}"
        )
        self.tokenizer.chat_template = chat_template

        input_ids = []
        conversations: List[str] = []
        for source in sources:
            if roles.get(source[0]["from"], "user") != "user":
                source = source[1:]
            input_id = []
            if add_system:
                input_id += self.tokenizer.apply_chat_template(
                    [{"role": "system", "content": "You are a helpful assistant."}]
                )
            for conv in source:
                role = roles.get(conv.get("role", conv.get("from")), "user")
                content = conv.get("content", conv.get("value", ""))
                conversations.append(content)
                encode_id = self.tokenizer.apply_chat_template(
                    [{"role": role, "content": content}]
                )
                input_id += encode_id

            for idx, encode in enumerate(input_id):
                if encode == image_token_index:
                    input_id[idx] = IMAGE_TOKEN_INDEX
                elif encode == memory_token_index:
                    input_id[idx] = MEMORY_TOKEN_INDEX
            input_ids.append(input_id)

        return torch.tensor(input_ids, dtype=torch.long), conversations

    def _parse_actions(self, output: str) -> List[int]:
        action_patterns = "|".join(map(repr, self.actions2idx)).replace("'", "")
        matches = []
        idx = 0
        while idx < len(output):
            matched = False
            for token, mapped in self.actions2idx.items():
                if output.startswith(token, idx):
                    matches.extend(mapped)
                    idx += len(token)
                    matched = True
                    break
            if not matched:
                idx += 1
        return matches
