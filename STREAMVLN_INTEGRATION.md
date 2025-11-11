# StreamVLN 集成到 Falcon 框架

本文档说明如何在 Falcon 框架（基于 Habitat 3）中使用 StreamVLN 模型（原本基于 Habitat 2）。

## 概述

StreamVLN 是一个视觉-语言导航模型，使用大型多模态模型（基于 Qwen2 和 LLaVA）进行指令跟随导航。本集成将 StreamVLN 的网络模型移植到 Falcon 的 policy 框架中。

## 项目结构

```
habitat-baselines/habitat_baselines/rl/ddppo/policy/
├── streamvln_policy.py          # StreamVLN Policy 主文件
├── streamvln/                   # StreamVLN 模块
│   ├── __init__.py
│   ├── habitat_compat.py        # Habitat 2 到 3 的适配层
│   ├── model/                   # StreamVLN 模型代码
│   │   └── stream_video_vln.py
│   ├── llava/                   # LLaVA 依赖
│   └── utils/                   # 工具函数
│       └── utils.py
└── __init__.py                  # 更新以注册 StreamVLN policy
```

## 配置文件

- `habitat-baselines/habitat_baselines/config/vln/streamvln_vln.yaml` - 主配置文件
- `habitat-baselines/habitat_baselines/config/vln/streamvln_inference.yaml` - 推理配置文件

## 安装依赖

### 基础依赖

```bash
# 安装 Falcon/Habitat 3 依赖
pip install -e habitat-lab
pip install -e habitat-baselines

# 安装 StreamVLN 依赖
pip install transformers torch torchvision
pip install pillow numpy

# 可选：Flash Attention 2 (推荐，显著提升推理速度)
pip install flash-attn --no-build-isolation
```

### StreamVLN 模型

下载预训练的 StreamVLN 模型：

```bash
# 示例：从 HuggingFace 或其他源下载模型
# 设置模型路径
export STREAMVLN_MODEL_PATH="/path/to/StreamVLN_Video_qwen_1_5_model"
```

## 使用方法

### 1. 推理模式

使用预训练的 StreamVLN 模型进行导航：

```bash
python -m habitat_baselines.run \
  --config-name=vln/streamvln_inference \
  habitat_baselines.rl.policy.main_agent.model_path=${STREAMVLN_MODEL_PATH} \
  habitat.dataset.data_path=data/datasets/vln/r2r/v1/val_seen/val_seen.json.gz
```

### 2. Python API 使用

```python
import torch
import numpy as np
from PIL import Image
from habitat_baselines.rl.ddppo.policy.streamvln_policy import StreamVLNNet

# 初始化模型
model = StreamVLNNet(
    observation_space=obs_space,
    action_space=action_space,
    hidden_size=1024,
    model_path="/path/to/streamvln/model",
    num_frames=32,
    num_history=8,
    num_future_steps=4,
    device="cuda"
)

# 生成动作序列
rgb_image = np.zeros((480, 640, 3), dtype=np.uint8)  # 你的 RGB 图像
instruction = "go to the kitchen"  # 导航指令

action_seq, llm_output = model.generate_action_sequence(
    rgb=rgb_image,
    instruction=instruction,
    env_idx=0,
    run_model=True
)

print(f"Generated actions: {action_seq}")
print(f"LLM output: {llm_output}")
```

### 3. 集成到 Falcon 环境

```python
from gym import spaces
import habitat
from habitat_baselines.rl.ddppo.policy import StreamVLNPolicy

# 创建环境
env = habitat.Env(config=config)

# 创建 policy
policy = StreamVLNPolicy.from_config(
    config=config,
    observation_space=env.observation_space,
    action_space=env.action_space
)

# 运行推理
obs = env.reset()
done = False
while not done:
    action = policy.act(obs)
    obs, reward, done, info = env.step(action)
```

## 关键特性

### 1. 动作映射

StreamVLN 使用以下动作空间：
- `0`: STOP - 停止
- `1`: ↑ (MOVE_FORWARD) - 前进 25cm
- `2`: ← (TURN_LEFT) - 左转 15°
- `3`: → (TURN_RIGHT) - 右转 15°

这些动作会自动映射到 Habitat 3 的动作空间。

### 2. 内存管理

StreamVLN 使用流式推理机制：
- 每 `num_frames` (默认 32) 帧后重置模型内存
- 保留 `num_history` (默认 8) 帧的历史信息
- 预测 `num_future_steps` (默认 4) 步的未来动作

### 3. Habitat 2 到 3 的适配

`habitat_compat.py` 模块处理以下适配：
- 观测空间格式转换
- 相机内参计算
- 深度图处理
- 传感器 API 差异

## 配置参数

### StreamVLN Policy 参数

在配置文件中设置：

```yaml
habitat_baselines:
  rl:
    policy:
      main_agent:
        name: "StreamVLNPolicy"
        model_path: "/path/to/streamvln/model"  # 必需
        num_frames: 32          # 内存重置间隔
        num_history: 8          # 历史帧数量
        num_future_steps: 4     # 未来步数预测
        model_max_length: 4096  # 最大序列长度
        device: "cuda"          # 设备
```

### 相机配置

StreamVLN 期望的相机配置：

```yaml
habitat:
  simulator:
    agents:
      main_agent:
        sim_sensors:
          rgb_sensor:
            height: 480
            width: 640
            hfov: 79
            position: [0, 1.25, 0]  # 相机高度 1.25m
```

## 注意事项

### Habitat 2 vs Habitat 3 差异

1. **观测格式**: 
   - Habitat 2: 使用 numpy 数组，RGB 值在 [0, 255]
   - Habitat 3: 可能使用 torch tensor，RGB 值可能归一化
   - 适配层会自动处理这些差异

2. **传感器 API**:
   - 传感器配置结构有所不同
   - 使用 `habitat_compat.py` 中的适配函数

3. **动作空间**:
   - 确保 Habitat 3 环境支持 StreamVLN 需要的动作
   - 可能需要自定义动作映射

### 性能优化

1. **Flash Attention 2**: 强烈推荐安装，可提升 2-3x 推理速度
2. **Batch Size**: StreamVLN 目前支持 batch_size=1，多环境支持需要进一步开发
3. **内存管理**: 调整 `num_frames` 和 `num_history` 以平衡性能和内存

## 故障排除

### 常见问题

**1. 模型加载失败**
```
Error: model_path must be provided for StreamVLN policy
```
解决：确保在配置文件中设置了正确的 `model_path`

**2. Flash Attention 错误**
```
ImportError: cannot import name 'flash_attn_func'
```
解决：降级到不使用 Flash Attention：
```python
# 在 streamvln_policy.py 中修改
attn_implementation="eager"  # 而不是 "flash_attention_2"
```

**3. CUDA 内存不足**
```
RuntimeError: CUDA out of memory
```
解决：
- 减少 `num_history`
- 降低图像分辨率
- 使用 `torch.bfloat16` 精度

**4. Habitat 版本不兼容**
确保使用 Habitat 3.x:
```bash
pip install habitat-sim==3.0.0
pip install habitat-lab==3.0.0
```

## 示例脚本

查看以下示例：
- `examples/streamvln_inference_example.py` - 基本推理示例
- `examples/streamvln_evaluation.py` - 评估脚本

## 参考资料

- [StreamVLN 论文](https://github.com/InternRobotics/StreamVLN)
- [Falcon 框架](https://github.com/Zeying-Gong/Falcon)
- [Habitat 3 文档](https://aihabitat.org/)
- [LLaVA 模型](https://llava-vl.github.io/)

## 贡献

欢迎提交 Issue 和 Pull Request！

## 许可证

本项目遵循 MIT 许可证。
