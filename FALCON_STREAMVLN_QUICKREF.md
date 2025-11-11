# StreamVLN + Falcon 快速参考

## 🎯 一句话总结

将 **StreamVLN** 视觉-语言导航模型集成到 **Falcon** 社交导航框架，替换 PointNavResNet，保留所有动态行人和辅助功能。

---

## 📁 配置文件位置

```
habitat-baselines/habitat_baselines/config/social_nav_v2/
├── streamvln_hm3d.yaml          ← 评估配置
└── streamvln_hm3d_train.yaml    ← 训练配置
```

---

## 🚀 快速开始

### 评估

```bash
python -m habitat_baselines.run \
  --config-name=social_nav_v2/streamvln_hm3d \
  habitat_baselines.rl.policy.agent_0.model_path=/path/to/streamvln/model
```

### 训练

```bash
python -m habitat_baselines.run \
  --config-name=social_nav_v2/streamvln_hm3d_train \
  habitat_baselines.rl.policy.agent_0.model_path=/path/to/streamvln/model
```

---

## 🔑 关键修改

| 项目 | Falcon 原始 | StreamVLN 版本 |
|------|-----------|---------------|
| **Policy** | PointNavResNetPolicy | **StreamVLNPolicy** |
| **RGB 传感器** | ❌ 注释 | ✅ **已启用** |
| **Hidden Size** | 512 | **1024** |
| **Num Environments** | 8 | **4** |
| **Train Encoder** | ✅ True | ❌ **False** |

---

## 📊 观测空间

### 评估模式
```yaml
obs_keys:
  - agent_0_articulated_agent_jaw_rgb       # ← 新增
  - agent_0_articulated_agent_jaw_depth
  - agent_0_pointgoal_with_gps_compass
```

### 训练模式（额外）
```yaml
obs_keys:
  # ... 评估模式的所有 keys ...
  - agent_0_localization_sensor              # ← 辅助损失
  - agent_0_human_num_sensor                 # ← 人数检测
  - agent_0_oracle_humanoid_future_trajectory # ← 轨迹预测
```

---

## ⚙️ StreamVLN 参数

```yaml
habitat_baselines:
  rl:
    policy:
      agent_0:
        name: "StreamVLNPolicy"
        model_path: "/path/to/model"  # ← 必须设置
        num_frames: 32       # 内存重置间隔
        num_history: 8       # 历史帧数
        num_future_steps: 4  # 未来步数
        model_max_length: 4096
        device: "cuda"
```

---

## 🎮 动作映射

| StreamVLN | 索引 | Falcon 动作 | 速度 |
|-----------|------|------------|------|
| STOP | 0 | discrete_stop | 0.0 |
| ↑ FORWARD | 1 | discrete_move_forward | 25.0 cm |
| ← LEFT | 2 | discrete_turn_left | 10.0° |
| → RIGHT | 3 | discrete_turn_right | -10.0° |

---

## 🏗️ 架构集成

```
Environment (RGB + Depth + GPS + Humans)
    ↓
StreamVLNNet.forward()
    ├─ 提取 RGB 观测
    ├─ StreamVLN Vision Tower
    └─ 特征提取
    ↓
NetPolicy.act()
    ├─ Action Distribution
    └─ Critic (Value)
    ↓
PolicyActionData
    ├─ actions
    ├─ values
    ├─ action_log_probs
    └─ rnn_hidden_states
    ↓
FALCONEvaluator / FalconTrainer
    ↓
Environment Step
```

---

## 💾 内存优化

### 默认（4 环境）
```yaml
num_environments: 4
num_frames: 32
num_history: 8
```
**内存**: ~12GB VRAM

### 节省内存（2 环境）
```yaml
num_environments: 2
num_frames: 16
num_history: 4
```
**内存**: ~8GB VRAM

---

## 🛠️ 故障排除速查

| 错误 | 原因 | 解决 |
|------|------|------|
| `model_path must be provided` | 未设置模型路径 | 在配置中设置 `model_path` |
| `CUDA out of memory` | 显存不足 | 减少 `num_environments` 和 `num_history` |
| `KeyError: agent_0_...rgb` | RGB 未启用 | 确保配置中包含 RGB 传感器 |
| 推理太慢 | StreamVLN 较大 | 这是正常的（~5-10 FPS） |

---

## 📈 性能对比

| 指标 | ResNet | StreamVLN |
|------|--------|----------|
| 推理速度 | ~30 FPS | ~5-10 FPS |
| 内存 | ~4GB | ~12GB |
| 参数量 | ~30M | ~7B |

---

## ✅ 保留的 Falcon 功能

- ✅ 动态行人（agent 1-6）
- ✅ Oracle 导航动作
- ✅ 辅助损失（people_counting, position, trajectory）
- ✅ FALCONEvaluator
- ✅ FalconTrainer
- ✅ 多agent 管理
- ✅ 所有原始传感器

---

## 📚 完整文档

- `STREAMVLN_FALCON_USAGE.md` - 详细使用指南
- `STREAMVLN_INTEGRATION.md` - StreamVLN 集成文档
- `INTEGRATION_SUMMARY.md` - 技术总结

---

## 🎓 关键概念

### Policy 替换原理

**Falcon 原始**:
```python
agent_0: PointNavResNetPolicy
  ├─ ResNet50 视觉编码器
  ├─ LSTM
  └─ 动作头
```

**StreamVLN 版本**:
```python
agent_0: StreamVLNPolicy
  ├─ StreamVLN Vision Tower (LLaVA)
  ├─ 特征提取（兼容 Falcon）
  └─ 动作头（复用 Falcon 的）
```

### 为什么有效？

1. **标准接口**: StreamVLNPolicy 继承 `NetPolicy`，实现相同的 `act()` 接口
2. **兼容输出**: 返回 `PolicyActionData`，与 Falcon evaluator/trainer 兼容
3. **观测适配**: 自动处理 Falcon 的观测格式（RGB + Depth + GPS）
4. **动作映射**: StreamVLN 的 4 个动作直接映射到 Falcon 的离散动作空间

---

## 🔬 实验建议

### 基线对比

```bash
# 1. 运行 Falcon 基线
python -m habitat_baselines.run \
  --config-name=social_nav_v2/falcon_hm3d

# 2. 运行 StreamVLN
python -m habitat_baselines.run \
  --config-name=social_nav_v2/streamvln_hm3d \
  habitat_baselines.rl.policy.agent_0.model_path=/path/to/model

# 3. 比较结果
```

### 消融实验

- 测试不同 `num_history` 值（4, 8, 16）
- 测试不同环境数量（1, 2, 4, 8）
- 比较冻结 vs 微调视觉编码器

---

*快速参考 - 2025-11-11*
