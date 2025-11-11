# StreamVLN 辅助任务状态参数集成说明

## 问题背景

Falcon 的辅助任务（auxiliary losses）需要从 policy 网络中提取**状态参数**来进行预测和计算损失。

## Falcon 的辅助任务

### 1. People Counting（人数统计）

**目标**: 预测场景中的人数（0到max_human_num）

**需要的状态参数**:
```python
scene_features = aux_loss_state['rnn_output']  # (batch_size, hidden_size)
```

**Ground Truth 来源**:
```python
target = batch["observations"]["human_num_sensor"]  # 人数传感器
```

### 2. Guess Human Position（人的位置预测）

**目标**: 预测场景中所有人的相对位置

**需要的状态参数**:
```python
scene_features = aux_loss_state['rnn_output']  # (batch_size, hidden_size)
human_num_features = batch["observations"]["human_num_sensor"]  # 人数
```

**Ground Truth 来源**:
```python
positions_gt = batch["observations"]["oracle_humanoid_future_trajectory"][:, :, 0, :]  # 当前位置
positions_gt_agent0 = batch["observations"]["localization_sensor"][:, [0, 2]]  # agent自身位置
# 计算相对位置
positions_gt_relative = positions_gt - positions_gt_agent0_repeated
```

### 3. Future Trajectory Prediction（未来轨迹预测）

**目标**: 预测场景中所有人未来4步的轨迹

**需要的状态参数**:
```python
scene_features = aux_loss_state["rnn_output"]  # (batch_size, hidden_size)
human_num_features = batch["observations"]["human_num_sensor"]  # 人数
position_features = batch["observations"]["oracle_humanoid_future_trajectory"][:, :, 0, :]  # 当前位置
```

**Ground Truth 来源**:
```python
positions_gt = batch["observations"]["oracle_humanoid_future_trajectory"][:, :, -future_step:, :]  # 未来轨迹
positions_gt_agent0 = batch["observations"]["localization_sensor"][:, [0, 2]]  # agent自身位置
# 计算相对轨迹
positions_gt_relative = positions_gt - positions_gt_agent0_repeated
```

## 原始 Falcon 中状态参数的提取

### 配置文件和数据源

在 `falcon_hm3d.yaml` 中：
```yaml
habitat:
  gym:
    obs_keys:
      # - agent_0_articulated_agent_jaw_rgb  # 注释掉，不使用RGB
      - agent_0_articulated_agent_jaw_depth    # ← 使用深度图
      - agent_0_pointgoal_with_gps_compass     # ← 目标位置
```

### 完整的提取流程

```
环境观测 → ResNetEncoder → 视觉特征 → 特征融合 → RNN → aux_loss_state
```

**步骤详解**:

#### 1. 环境产生观测
```python
observations = {
    'agent_0_articulated_agent_jaw_depth': torch.Tensor([batch, H, W, 1]),  # 深度图
    'agent_0_pointgoal_with_gps_compass': torch.Tensor([batch, 2]),         # [距离, 角度]
}
```

#### 2. ResNetEncoder 处理深度图
```python
# 在 ResNetEncoder.forward() 中
# 输入: depth (batch, H, W, 1)
depth = observations['agent_0_articulated_agent_jaw_depth']
depth = depth.permute(0, 3, 1, 2)  # → (batch, 1, H, W)
depth = depth.float() / 255.0       # 归一化
depth = F.avg_pool2d(depth, 2)      # 下采样
x = self.backbone(depth)            # ResNet50: → (batch, 2048, H', W')
visual_feats = self.compression(x)  # Flatten + Linear: → (batch, 512)
```

#### 3. PointNavResNetNet 特征融合
```python
# 在 PointNavResNetNet.forward() 中
x = []

# 3.1 视觉特征
visual_feats = self.visual_encoder(observations)  # (batch, 512)
visual_feats = self.visual_fc(visual_feats)       # (batch, 512)
aux_loss_state["perception_embed"] = visual_feats  # ← 保存原始视觉特征
x.append(visual_feats)

# 3.2 Goal 编码
goal_obs = observations['agent_0_pointgoal_with_gps_compass']  # (batch, 2)
# 极坐标变换
goal_obs_transformed = torch.stack([
    goal_obs[:, 0],                    # 距离
    torch.cos(-goal_obs[:, 1]),        # cos(角度)
    torch.sin(-goal_obs[:, 1]),        # sin(角度)
], -1)  # → (batch, 3)
goal_embed = self.tgt_embeding(goal_obs_transformed)  # Linear(3→32): → (batch, 32)
x.append(goal_embed)

# 3.3 动作历史
prev_actions = self.prev_action_embedding(prev_actions)  # Embedding: → (batch, 32)
x.append(prev_actions)

# 3.4 拼接
fused = torch.cat(x, dim=1)  # (batch, 512 + 32 + 32) = (batch, 576)
```

#### 4. RNN 处理
```python
# 通过 LSTM
rnn_output, rnn_hidden_states = self.state_encoder(
    fused,                # (batch, 576)
    rnn_hidden_states, 
    masks
)
# rnn_output: (batch, 512) ← RNN 输出
```

#### 5. 构建 aux_loss_state
```python
aux_loss_state = {
    "perception_embed": visual_feats,  # (batch, 512) - 原始视觉特征
    "rnn_output": rnn_output,          # (batch, 512) - RNN 输出 ← 辅助任务使用！
}

return rnn_output, rnn_hidden_states, aux_loss_state
```

### 关键数据来源总结

| 数据 | 来源 | 形状 | 用途 |
|------|------|------|------|
| **深度图** | `agent_0_articulated_agent_jaw_depth` | (batch, H, W, 1) | 输入到 ResNetEncoder |
| **目标位置** | `agent_0_pointgoal_with_gps_compass` | (batch, 2) | Goal 编码 |
| **视觉特征** | ResNet50 输出 | (batch, 512) | `perception_embed` |
| **融合特征** | 视觉+Goal+动作拼接 | (batch, 576) | 输入到 RNN |
| **RNN 输出** | LSTM 输出 | (batch, 512) | `rnn_output` ← **辅助任务使用** |

---

## ResNet Policy 的实现

在 `resnet_policy.py` 中，状态参数的构建过程：

```python
def forward(self, observations, rnn_hidden_states, prev_actions, masks, ...):
    x = []
    aux_loss_state = {}
    
    # 1. 视觉特征
    visual_feats = self.visual_encoder(observations)
    visual_feats = self.visual_fc(visual_feats)
    aux_loss_state["perception_embed"] = visual_feats  # ← 原始视觉特征
    x.append(visual_feats)
    
    # 2. Goal 信息
    goal_observations = observations['pointgoal_with_gps_compass']
    x.append(self.tgt_embeding(goal_observations))
    
    # 3. 之前的动作
    prev_actions_embed = self.prev_action_embedding(prev_actions)
    x.append(prev_actions_embed)
    
    # 4. 融合所有特征
    out = torch.cat(x, dim=1)
    
    # 5. 经过 RNN
    out, rnn_hidden_states = self.state_encoder(out, rnn_hidden_states, masks)
    
    # 6. 保存 RNN 输出 ← 关键！这是辅助任务使用的
    aux_loss_state["rnn_output"] = out
    
    return out, rnn_hidden_states, aux_loss_state
```

**关键点**:
- `perception_embed`: 原始视觉特征（RNN 之前）
- `rnn_output`: 融合的时序特征（RNN 之后），包含：
  - 视觉信息
  - Goal 信息
  - 动作历史
  - 时序依赖（通过 RNN）

## StreamVLN Policy 的实现

### 挑战

1. **StreamVLN 没有 RNN**: 它使用 Transformer
2. **需要提供等效的 `rnn_output`**: 必须包含足够的场景信息

### 解决方案

在 `streamvln_policy.py` 中，我们实现了类似的特征融合：

```python
def forward(self, observations, rnn_hidden_states, prev_actions, masks, ...):
    # 1. 提取视觉特征（通过 StreamVLN Vision Tower）
    visual_features = self.model.get_vision_tower()(images_batch)
    visual_features = visual_features.mean(dim=1)  # Pool
    features = visual_features  # (batch_size, feature_dim)
    
    # 2. 融合额外信息（模拟 ResNet policy 的行为）
    x = [features]  # 开始于视觉特征
    
    # 添加 goal 信息
    if 'agent_0_pointgoal_with_gps_compass' in observations:
        goal_obs = observations['agent_0_pointgoal_with_gps_compass']
        x.append(goal_obs)
    
    # 添加之前的动作
    if hasattr(self, 'prev_action_embedding'):
        prev_action_feat = self.prev_action_embedding(prev_actions)
        x.append(prev_action_feat)
    
    # 融合所有特征
    fused_features = torch.cat(x, dim=1)
    
    # 3. 准备辅助任务状态
    aux_loss_state = {
        "perception_embed": features,        # 原始视觉特征
        "rnn_output": fused_features,        # 融合特征（等效于 RNN 输出）
    }
    
    return fused_features, rnn_hidden_states, aux_loss_state
```

### 特征组成

**`perception_embed`** (原始视觉特征):
- 维度: (batch_size, vision_feature_dim)
- 来源: StreamVLN Vision Tower
- 内容: 纯视觉信息

**`rnn_output`** (融合特征):
- 维度: (batch_size, vision_feature_dim + goal_dim + action_dim)
- 来源: 视觉 + Goal + 动作的拼接
- 内容:
  - 视觉特征（来自 StreamVLN）
  - Goal 距离和角度（2维）
  - 之前的动作嵌入（32维）

### 为什么这样有效？

1. **包含场景信息**: 视觉特征捕获场景中的人和障碍物
2. **包含目标信息**: Goal embedding 提供导航目标
3. **包含动作历史**: 动作嵌入提供时序上下文
4. **足够丰富**: 辅助任务能从中预测人数、位置和轨迹

## 维度分析

### ResNet Policy

```
Visual: (batch, 512) 
  + Goal: (batch, 32) 
  + Action: (batch, 32)
  ↓ Concatenate
  = (batch, 576)
  ↓ RNN
  = (batch, 512)  ← rnn_output
```

### StreamVLN Policy

```
Visual: (batch, feature_dim)  # 来自 Vision Tower
  + Goal: (batch, 2)          # 距离 + 角度
  + Action: (batch, 32)       # 动作嵌入
  ↓ Concatenate
  = (batch, feature_dim + 34)  ← rnn_output (fused_features)
```

如果 `feature_dim` ≈ 1024，则 `rnn_output` ≈ (batch, 1056)

这比 ResNet 的 512 维更丰富，完全足够辅助任务使用。

## 观测需求

为了让辅助任务正常工作，需要确保配置中包含以下观测：

### 评估模式（最小集）
```yaml
obs_keys:
  - agent_0_articulated_agent_jaw_rgb          # StreamVLN 需要
  - agent_0_articulated_agent_jaw_depth
  - agent_0_pointgoal_with_gps_compass         # Goal 信息
```

### 训练模式（完整集）
```yaml
obs_keys:
  - agent_0_articulated_agent_jaw_rgb          # StreamVLN 需要
  - agent_0_articulated_agent_jaw_depth
  - agent_0_pointgoal_with_gps_compass         # Goal 信息
  - agent_0_human_num_sensor                   # ← 人数统计 GT
  - agent_0_localization_sensor                # ← 位置预测 GT
  - agent_0_oracle_humanoid_future_trajectory  # ← 轨迹预测 GT
```

## 完整的数据流

```
Environment
    ↓
Observations {
    rgb: (batch, H, W, 3)
    pointgoal: (batch, 2)
    human_num: (batch, 1)
    localization: (batch, 3)
    future_trajectory: (batch, max_human, future_step, 2)
}
    ↓
StreamVLNNet.forward()
    ├─ 视觉编码 → visual_features (batch, 1024)
    ├─ Goal 信息 → (batch, 2)
    ├─ 动作历史 → (batch, 32)
    └─ 拼接 → fused_features (batch, 1058)
    ↓
aux_loss_state = {
    "perception_embed": visual_features,
    "rnn_output": fused_features  ← 辅助任务使用这个
}
    ↓
Auxiliary Loss Modules
    ├─ PeopleCounting(rnn_output) → 预测人数
    ├─ GuessHumanPosition(rnn_output) → 预测位置
    └─ FutureTrajectoryPrediction(rnn_output) → 预测轨迹
    ↓
Auxiliary Losses {
    "people_counting": loss1,
    "guess_human_position": loss2,
    "future_trajectory_prediction": loss3
}
    ↓
Total Loss = policy_loss + aux_loss1 + aux_loss2 + aux_loss3
```

## 验证

### 检查状态参数是否正确

在训练时，可以添加调试代码：

```python
# 在 streamvln_policy.py 的 forward 方法中
print(f"perception_embed shape: {aux_loss_state['perception_embed'].shape}")
print(f"rnn_output shape: {aux_loss_state['rnn_output'].shape}")
```

**预期输出**:
```
perception_embed shape: torch.Size([batch_size, 1024])
rnn_output shape: torch.Size([batch_size, 1058])  # 1024 + 2 + 32
```

### 检查辅助损失是否工作

在训练日志中，应该能看到：

```
Losses:
  policy_loss: 0.234
  people_counting: 0.056
  guess_human_position: 0.123
  future_trajectory_prediction: 0.089
  total_loss: 0.502
```

## 与 ResNet Policy 的对比

| 特征 | ResNet Policy | StreamVLN Policy |
|------|--------------|-----------------|
| 视觉编码器 | ResNet50 | StreamVLN Vision Tower |
| 特征维度 | 512 | ~1024 |
| 时序建模 | LSTM | 无（直接拼接） |
| Goal 编码 | 嵌入层（32维） | 直接拼接（2维） |
| 动作历史 | 嵌入层（32维） | 嵌入层（32维） |
| `rnn_output` | LSTM 输出 | 拼接后的特征 |
| 信息丰富度 | 中等 | **高**（更大的视觉特征） |

## 优势

StreamVLN 的 `rnn_output` 实际上**更丰富**：

1. **更大的视觉特征**: 1024 vs 512
2. **来自预训练 VLM**: StreamVLN 的 Vision Tower 来自 LLaVA，在大规模数据上预训练
3. **更好的语义理解**: 能够理解场景中的人和物体

这使得辅助任务（人数统计、位置预测、轨迹预测）可能**表现更好**！

## 总结

✅ **StreamVLN 提供的状态参数**:
- `perception_embed`: 原始视觉特征（1024维）
- `rnn_output`: 融合特征（1058维 = 1024视觉 + 2goal + 32动作）

✅ **符合 Falcon 辅助任务的要求**:
- 所有三个辅助任务都使用 `aux_loss_state['rnn_output']`
- 包含足够的场景信息用于预测

✅ **实现方式**:
- 通过特征拼接模拟 RNN 的融合效果
- 不使用实际的 RNN，保持 StreamVLN 的架构

✅ **预期效果**:
- 辅助任务能正常训练和计算损失
- 可能由于更丰富的特征而表现更好

---

*最后更新: 2025-11-11*
