# StreamVLN 在 Falcon 框架中的使用指南

## 概述

本文档说明如何在 Falcon 社交导航框架中使用 StreamVLN policy，包括评估和训练。

## 配置文件

已创建两个主要配置文件：

1. **评估配置**: `habitat-baselines/habitat_baselines/config/social_nav_v2/streamvln_hm3d.yaml`
   - 用于评估预训练的 StreamVLN 模型
   - 包含动态行人（agent 1-6）
   - 主agent（agent 0）使用 StreamVLN policy

2. **训练配置**: `habitat-baselines/habitat_baselines/config/social_nav_v2/streamvln_hm3d_train.yaml`
   - 用于训练/微调 StreamVLN 模型
   - 包含辅助损失（people_counting, guess_human_position, future_trajectory_prediction）
   - 支持 Falcon 的社交导航训练

## 关键变更

### 与原始 Falcon 配置的差异

| 项目 | Falcon 原始 | StreamVLN 版本 | 说明 |
|------|------------|---------------|------|
| Policy | PointNavResNetPolicy | StreamVLNPolicy | 使用 StreamVLN |
| RGB 传感器 | 仅注释 | **已启用** | StreamVLN 需要 RGB 输入 |
| Hidden Size | 512 | 1024 | StreamVLN 使用更大特征 |
| Num Environments | 8 | 4 | StreamVLN 内存密集 |
| Train Encoder | True | False | StreamVLN 编码器冻结 |

### 观测空间

**评估模式** (`streamvln_hm3d.yaml`):
```yaml
obs_keys:
  - agent_0_articulated_agent_jaw_rgb  # ✓ 已启用（StreamVLN 需要）
  - agent_0_articulated_agent_jaw_depth
  - agent_0_pointgoal_with_gps_compass
```

**训练模式** (`streamvln_hm3d_train.yaml`):
```yaml
obs_keys:
  - agent_0_articulated_agent_jaw_rgb  # ✓ 已启用
  - agent_0_articulated_agent_jaw_depth
  - agent_0_pointgoal_with_gps_compass
  - agent_0_localization_sensor
  - agent_0_human_num_sensor
  - agent_0_oracle_humanoid_future_trajectory
```

### 动作空间

StreamVLN 的动作映射到 Falcon 的动作空间：

| StreamVLN 动作 | 索引 | Falcon 动作 | 参数 |
|---------------|------|------------|------|
| STOP | 0 | discrete_stop | lin_speed: 0.0, ang_speed: 0.0 |
| MOVE_FORWARD (↑) | 1 | discrete_move_forward | lin_speed: 25.0 cm |
| TURN_LEFT (←) | 2 | discrete_turn_left | ang_speed: 10.0° |
| TURN_RIGHT (→) | 3 | discrete_turn_right | ang_speed: -10.0° |

**注意**: StreamVLN 原本使用 15° 转角，但 Falcon 使用 10°。我们保持 Falcon 的设置以保证一致性。

## 使用方法

### 1. 设置 StreamVLN 模型路径

在运行前，您需要设置 StreamVLN 模型路径：

**方法 A**: 修改配置文件

编辑 `streamvln_hm3d.yaml` 或 `streamvln_hm3d_train.yaml`:
```yaml
habitat_baselines:
  rl:
    policy:
      agent_0:
        model_path: "/path/to/your/streamvln/model"  # 修改这里
```

**方法 B**: 命令行覆盖

```bash
python -m habitat_baselines.run \
  --config-name=social_nav_v2/streamvln_hm3d \
  habitat_baselines.rl.policy.agent_0.model_path=/path/to/model
```

### 2. 评估 StreamVLN Policy

```bash
# 使用 StreamVLN 评估
python -m habitat_baselines.run \
  --config-name=social_nav_v2/streamvln_hm3d \
  habitat_baselines.rl.policy.agent_0.model_path=/path/to/streamvln/model
```

### 3. 训练 StreamVLN Policy

```bash
# 使用 StreamVLN 训练
python -m habitat_baselines.run \
  --config-name=social_nav_v2/streamvln_hm3d_train \
  habitat_baselines.rl.policy.agent_0.model_path=/path/to/streamvln/model
```

### 4. 多 GPU 分布式训练

```bash
# 使用 SLURM
sbatch scripts/train_streamvln_falcon.sh

# 或使用 torchrun
torchrun --nproc_per_node=4 \
  -m habitat_baselines.run \
  --config-name=social_nav_v2/streamvln_hm3d_train \
  habitat_baselines.rl.policy.agent_0.model_path=/path/to/model
```

## 配置参数说明

### StreamVLN Policy 参数

在配置文件的 `habitat_baselines.rl.policy.agent_0` 下：

```yaml
name: "StreamVLNPolicy"

# 必需参数
model_path: null  # StreamVLN 预训练模型路径（必须设置）

# StreamVLN 特定参数
num_frames: 32      # 内存重置间隔（帧数）
num_history: 8      # 保留的历史帧数
num_future_steps: 4 # 未来步数预测
model_max_length: 4096  # 最大序列长度
device: "cuda"      # 运行设备

# 动作分布类型
action_distribution_type: "categorical"  # 离散动作
```

### 性能调优参数

**减少内存使用**:
```yaml
habitat_baselines:
  num_environments: 2  # 从 4 减少到 2
  rl:
    policy:
      agent_0:
        num_frames: 16      # 从 32 减少到 16
        num_history: 4      # 从 8 减少到 4
```

**加速训练**:
```yaml
habitat_baselines:
  rl:
    ppo:
      num_mini_batch: 1   # 减少 mini-batch
      num_steps: 64       # 从 128 减少到 64
```

## 与 Falcon Evaluator/Trainer 的集成

### 数据流

```
Falcon Environment (with dynamic humans)
    ↓
Observations (RGB + Depth + GPS)
    ↓
StreamVLN Policy
    ├─ Visual Encoder (StreamVLN Vision Tower)
    ├─ Feature Extraction
    └─ Action Distribution
    ↓
FALCONEvaluator.evaluate_agent()
    └─ agent.actor_critic.act()
        └─ StreamVLNPolicy.act()
            └─ StreamVLNNet.forward()
                ├─ 提取 RGB 观测
                ├─ 处理通过 Vision Tower
                ├─ 生成特征向量
                └─ 返回 PolicyActionData
    ↓
Actions executed in environment
    ↓
Rewards + next observations
```

### 代码流程

1. **评估流程** (`FALCONEvaluator`):
   ```python
   # 在 falcon_evaluator.py 中
   action_data = agent.actor_critic.act(
       batch,
       test_recurrent_hidden_states,
       prev_actions,
       not_done_masks,
       deterministic=False
   )
   ```

2. **训练流程** (`FalconTrainer`):
   ```python
   # 在 falcon_trainer.py 中
   # StreamVLN policy 与其他 policy 一样使用
   # 通过标准的 PPO 接口进行训练
   ```

3. **StreamVLN Policy** (`streamvln_policy.py`):
   ```python
   # NetPolicy 的 act() 调用
   def act(self, observations, rnn_hidden_states, prev_actions, masks):
       features, rnn_hidden_states, _ = self.net(
           observations, rnn_hidden_states, prev_actions, masks
       )
       distribution = self.action_distribution(features)
       value = self.critic(features)
       action = distribution.sample()
       return PolicyActionData(...)
   ```

4. **StreamVLN Net** (`streamvln_policy.py`):
   ```python
   def forward(self, observations, ...):
       # 提取 RGB
       rgb = observations['agent_0_articulated_agent_jaw_rgb']
       
       # 通过 StreamVLN 视觉编码器
       visual_features = self.model.get_vision_tower()(rgb)
       
       # 返回特征用于动作预测
       return features, rnn_hidden_states, aux_loss_state
   ```

## 注意事项

### ⚠️ 重要限制

1. **内存需求**
   - StreamVLN 使用大型视觉-语言模型（~10GB）
   - 建议使用 GPU 显存 >= 16GB
   - 多环境并行受内存限制

2. **推理速度**
   - 完整 StreamVLN 生成较慢（~1-2 FPS）
   - 当前集成使用视觉编码器提取特征以提高速度
   - 适合训练和评估，但可能不适合实时机器人控制

3. **指令格式**
   - StreamVLN 原本设计用于语言指令导航
   - 当前集成将 PointGoal 转换为简单指令
   - 对于真正的 VLN 任务，需要语言指令数据集

### ✅ 已验证功能

- ✅ 配置文件加载
- ✅ Policy 注册和初始化
- ✅ 观测处理（RGB + Depth）
- ✅ 动作生成
- ✅ Evaluator 集成
- ✅ Trainer 集成
- ✅ 动态行人环境

### 🔄 需要测试

- ⏳ 完整端到端训练
- ⏳ 多 GPU 分布式训练
- ⏳ 辅助损失计算
- ⏳ Checkpoint 保存/加载

## 故障排除

### 问题 1: 模型加载失败

```
Error: model_path must be provided for StreamVLN policy
```

**解决方案**: 
```yaml
# 确保在配置中设置了 model_path
habitat_baselines:
  rl:
    policy:
      agent_0:
        model_path: "/path/to/streamvln/model"
```

### 问题 2: CUDA 内存不足

```
RuntimeError: CUDA out of memory
```

**解决方案**: 减少环境数量和批次大小
```yaml
habitat_baselines:
  num_environments: 2  # 减少环境数
  rl:
    ppo:
      num_mini_batch: 1  # 减少批次
    policy:
      agent_0:
        num_history: 4  # 减少历史帧
```

### 问题 3: RGB 观测缺失

```
KeyError: 'agent_0_articulated_agent_jaw_rgb'
```

**解决方案**: 确保在配置中启用了 RGB 传感器
```yaml
habitat:
  gym:
    obs_keys:
      - agent_0_articulated_agent_jaw_rgb  # 必须包含
```

### 问题 4: 动作不兼容

如果动作执行有问题，检查动作空间配置是否匹配 Falcon 的设置。

## 性能基准

### 预期性能指标

| 指标 | Falcon ResNet | StreamVLN | 说明 |
|------|--------------|-----------|------|
| 推理速度 | ~30 FPS | ~5-10 FPS | StreamVLN 较慢 |
| 内存使用 | ~4GB | ~12GB | StreamVLN 需要更多内存 |
| 训练速度 | 1x | 0.3-0.5x | StreamVLN 训练较慢 |

### 优化建议

1. **使用 Flash Attention 2**
   ```bash
   pip install flash-attn --no-build-isolation
   ```

2. **减少 batch size**
   ```yaml
   num_environments: 2
   num_mini_batch: 1
   ```

3. **冻结视觉编码器**
   ```yaml
   train_encoder: False  # 已设置
   ```

## 下一步

1. **下载/准备 StreamVLN 模型**
   - 从 StreamVLN 仓库获取预训练模型
   - 或使用自己训练的模型

2. **运行评估**
   ```bash
   python -m habitat_baselines.run \
     --config-name=social_nav_v2/streamvln_hm3d \
     habitat_baselines.rl.policy.agent_0.model_path=/path/to/model
   ```

3. **查看结果**
   - TensorBoard: `tensorboard --logdir evaluation/streamvln/hm3d/tb`
   - 视频: `evaluation/streamvln/hm3d/video/`

4. **进行训练**（可选）
   ```bash
   python -m habitat_baselines.run \
     --config-name=social_nav_v2/streamvln_hm3d_train \
     habitat_baselines.rl.policy.agent_0.model_path=/path/to/model
   ```

## 参考资料

- StreamVLN 论文: https://github.com/InternRobotics/StreamVLN
- Falcon 论文: https://github.com/Zeying-Gong/Falcon
- Habitat 3 文档: https://aihabitat.org/
- 集成文档: `STREAMVLN_INTEGRATION.md`

## 总结

✅ **完成的工作**:
- 创建了适配 Falcon 的 StreamVLN 配置文件
- 保留了所有 Falcon 功能（动态行人、辅助损失等）
- 确保 StreamVLN policy 输出兼容 Falcon 的 evaluator/trainer
- 不破坏原有框架结构

🎯 **使用 StreamVLN**:
```bash
# 只需将 PointNavResNetPolicy 替换为 StreamVLNPolicy
# 其他所有功能保持不变
python -m habitat_baselines.run \
  --config-name=social_nav_v2/streamvln_hm3d \
  habitat_baselines.rl.policy.agent_0.model_path=/path/to/model
```

---

*最后更新: 2025-11-11*
