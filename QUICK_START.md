# StreamVLN 快速开始指南

## 1️⃣ 验证安装

```bash
# 测试集成是否正常
python test_streamvln_integration.py
```

## 2️⃣ 安装依赖

```bash
# 安装 StreamVLN 所需依赖
pip install -r streamvln_requirements.txt

# (可选) 安装 Flash Attention 2 以提升速度
pip install flash-attn --no-build-isolation
```

## 3️⃣ 准备模型

下载预训练的 StreamVLN 模型，或指定您的模型路径：

```bash
export STREAMVLN_MODEL_PATH="/path/to/your/streamvln/model"
```

## 4️⃣ 运行推理示例

### 方式 1: 使用示例脚本

```bash
python examples/streamvln_inference_example.py \
    --model-path ${STREAMVLN_MODEL_PATH} \
    --instruction "go to the kitchen and turn left" \
    --num-steps 10
```

### 方式 2: 使用配置文件

```bash
python -m habitat_baselines.run \
    --config-name=vln/streamvln_inference \
    habitat_baselines.rl.policy.main_agent.model_path=${STREAMVLN_MODEL_PATH}
```

### 方式 3: Python API

```python
from habitat_baselines.rl.ddppo.policy import StreamVLNPolicy
from gym import spaces
import numpy as np

# 创建观测和动作空间
obs_space = spaces.Dict({
    'rgb': spaces.Box(0, 255, (480, 640, 3), dtype=np.uint8)
})
action_space = spaces.Discrete(4)

# 创建配置对象 (简化版)
class Config:
    pass

config = Config()
# ... 设置配置 ...

# 初始化 policy
policy = StreamVLNPolicy.from_config(
    config=config,
    observation_space=obs_space,
    action_space=action_space
)

# 使用 policy
obs = env.reset()
action = policy.act(obs)
```

## 📚 更多信息

- 详细文档: `STREAMVLN_INTEGRATION.md`
- 项目总结: `INTEGRATION_SUMMARY.md`
- 文件清单: `CREATED_FILES.txt`

## 🐛 故障排除

### 问题: 导入错误

```bash
# 确保正确安装了 Habitat
pip install habitat-sim habitat-lab

# 确保在正确的目录
cd /workspace
```

### 问题: 模型加载失败

检查 `model_path` 是否正确：
- 路径应指向包含模型文件的目录
- 目录应包含 `config.json`, `pytorch_model.bin` 等文件

### 问题: CUDA 内存不足

降低内存使用：
```yaml
# 在配置文件中
habitat_baselines:
  rl:
    policy:
      main_agent:
        num_history: 4  # 从 8 减到 4
        num_frames: 16  # 从 32 减到 16
```

## ✅ 集成状态

- ✅ StreamVLN 模型已集成
- ✅ Habitat 2/3 适配完成
- ✅ Policy 接口已实现
- ✅ 配置文件已创建
- ✅ 示例和文档已完成

开始探索吧！🚀
