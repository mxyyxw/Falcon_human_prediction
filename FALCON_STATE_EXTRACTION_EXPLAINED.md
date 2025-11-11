# Falcon 状态参数提取详解

## 原始 Falcon (PointNavResNetPolicy) 的状态参数提取

让我们逐步追踪在 `falcon_hm3d.yaml` 配置下，状态参数是如何从观测中提取并构建的。

---

## 配置文件：falcon_hm3d.yaml

```yaml
habitat:
  gym:
    obs_keys:
      # - agent_0_articulated_agent_jaw_rgb  # 注释掉，不使用RGB
      - agent_0_articulated_agent_jaw_depth    # 使用深度图
      - agent_0_pointgoal_with_gps_compass     # 目标位置

habitat_baselines:
  rl:
    policy:
      agent_0:
        name: "PointNavResNetPolicy"  # ← 使用 ResNet policy
```

---

## 完整的状态参数提取流程

### 步骤 1: 环境产生观测

```python
# 在 Falcon 环境中
observations = {
    'agent_0_articulated_agent_jaw_depth': torch.Tensor([batch, H, W, 1]),  # 深度图
    'agent_0_pointgoal_with_gps_compass': torch.Tensor([batch, 2]),         # [距离, 角度]
}
```

### 步骤 2: 观测进入 Policy

```python
# 在 evaluator/trainer 中
action_data = agent.actor_critic.act(
    batch=observations,  # ← 包含深度图和 pointgoal
    rnn_hidden_states=rnn_hidden_states,
    prev_actions=prev_actions,
    masks=masks,
)
```

### 步骤 3: NetPolicy.act() 调用 forward()

`NetPolicy.act()` (在 `policy.py` 中):
```python
def act(self, observations, rnn_hidden_states, prev_actions, masks, deterministic=False):
    # 1. 调用网络的 forward 方法
    features, rnn_hidden_states, aux_loss_state = self.net(
        observations,           # ← 深度图 + pointgoal
        rnn_hidden_states,
        prev_actions,
        masks
    )
    
    # 2. 使用 features 生成动作分布
    distribution = self.action_distribution(features)
    value = self.critic(features)
    action = distribution.sample()
    
    # 3. 返回动作数据（aux_loss_state 也被保留用于训练）
    return PolicyActionData(
        values=value,
        actions=action,
        action_log_probs=distribution.log_probs(action),
        rnn_hidden_states=rnn_hidden_states,
    )
```

### 步骤 4: PointNavResNetNet.forward() 提取特征

**这是关键！** 在 `resnet_policy.py` 的 `PointNavResNetNet.forward()` 中：

```python
def forward(
    self,
    observations: Dict[str, torch.Tensor],
    rnn_hidden_states,
    prev_actions,
    masks,
    rnn_build_seq_info: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    
    x = []  # 用于收集所有特征
    aux_loss_state = {}  # ← 这里构建辅助任务的状态
    
    # ========== 1. 视觉特征提取 ==========
    if not self.is_blind:
        # 调用 ResNetEncoder
        visual_feats = self.visual_encoder(observations)
        # visual_feats shape: (batch, C, H, W) - ResNet 输出
        
        # 通过 FC 层压缩
        visual_feats = self.visual_fc(visual_feats)
        # visual_feats shape: (batch, 512)
        
        # ← 保存原始视觉特征到 aux_loss_state
        aux_loss_state["perception_embed"] = visual_feats
        
        x.append(visual_feats)  # 添加到特征列表
    
    # ========== 2. 其他 1D 传感器 ==========
    # 在 Falcon 中，这部分通常为空，因为其他传感器被排除了
    if len(self._fuse_keys_1d) != 0:
        fuse_states = torch.cat(
            [observations[k] for k in self._fuse_keys_1d], dim=-1
        )
        x.append(fuse_states.float())
    
    # ========== 3. PointGoal 嵌入 ==========
    if 'agent_0_pointgoal_with_gps_compass' in observations:
        goal_observations = observations['agent_0_pointgoal_with_gps_compass']
        # goal_observations shape: (batch, 2) - [距离, 角度]
        
        # 极坐标变换
        goal_observations = torch.stack([
            goal_observations[:, 0],           # 距离
            torch.cos(-goal_observations[:, 1]),  # cos(角度)
            torch.sin(-goal_observations[:, 1]),  # sin(角度)
        ], -1)
        # goal_observations shape: (batch, 3)
        
        # 通过 Linear 层嵌入
        goal_embed = self.tgt_embeding(goal_observations)
        # goal_embed shape: (batch, 32)
        
        x.append(goal_embed)  # 添加到特征列表
    
    # ========== 4. 之前的动作嵌入 ==========
    if self.discrete_actions:
        prev_actions = prev_actions.squeeze(-1)
        start_token = torch.zeros_like(prev_actions)
        # 使用 Embedding 层
        prev_actions = self.prev_action_embedding(
            torch.where(masks.view(-1), prev_actions + 1, start_token)
        )
        # prev_actions shape: (batch, 32)
    
    x.append(prev_actions)  # 添加到特征列表
    
    # ========== 5. 拼接所有特征 ==========
    out = torch.cat(x, dim=1)
    # out shape: (batch, 512 + 32 + 32) = (batch, 576)
    
    # ========== 6. 通过 RNN ==========
    out, rnn_hidden_states = self.state_encoder(
        out, rnn_hidden_states, masks, rnn_build_seq_info
    )
    # out shape: (batch, 512) - LSTM 输出
    
    # ← 保存 RNN 输出到 aux_loss_state
    aux_loss_state["rnn_output"] = out
    
    # ========== 7. 返回 ==========
    return out, rnn_hidden_states, aux_loss_state
```

---

## ResNetEncoder 内部：视觉特征提取

在 `ResNetEncoder.forward()` 中：

```python
def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
    if self.is_blind:
        return None
    
    cnn_input = []
    
    # ========== 提取视觉观测 ==========
    for k in self.visual_keys:  # visual_keys = ['agent_0_articulated_agent_jaw_depth']
        obs_k = observations[k]
        # obs_k shape: (batch, H, W, 1) - 深度图
        
        # 调整维度顺序：BHWC -> BCHW
        obs_k = obs_k.permute(0, 3, 1, 2)
        # obs_k shape: (batch, 1, H, W)
        
        # 归一化
        if self.key_needs_rescaling[k] is not None:
            obs_k = obs_k.float() * self.key_needs_rescaling[k]
        
        cnn_input.append(obs_k)
    
    # ========== 拼接多个视觉输入（如果有）==========
    x = torch.cat(cnn_input, dim=1)
    # x shape: (batch, 1, H, W) - 在 Falcon 中只有深度
    
    # ========== 下采样 ==========
    x = F.avg_pool2d(x, 2)
    # x shape: (batch, 1, H/2, W/2)
    
    # ========== 归一化 ==========
    x = self.running_mean_and_var(x)
    
    # ========== ResNet50 backbone ==========
    x = self.backbone(x)
    # x shape: (batch, 2048, H', W') - ResNet50 输出
    
    # ========== 压缩 ==========
    x = self.compression(x)
    # x shape: (batch, C, H'', W'')
    
    return x
```

然后在 `PointNavResNetNet` 中：
```python
visual_feats = self.visual_fc(visual_feats)
# Flatten + Linear: (batch, C*H''*W'') -> (batch, 512)
```

---

## 数据流总结

```
Environment 生成观测
    ↓
observations = {
    'agent_0_articulated_agent_jaw_depth': (batch, H, W, 1),  ← 来自深度相机
    'agent_0_pointgoal_with_gps_compass': (batch, 2),         ← 来自 GPS + 指南针
}
    ↓
PointNavResNetNet.forward()
    ↓
┌─────────────────────────────────────────┐
│ 1. ResNetEncoder (视觉编码器)            │
│    输入: depth (batch, H, W, 1)         │
│    输出: visual_feats (batch, 512)      │ ← 保存到 aux_loss_state["perception_embed"]
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 2. PointGoal 嵌入                        │
│    输入: pointgoal (batch, 2)            │
│    变换: 极坐标 -> (batch, 3)            │
│    Linear: (batch, 3) -> (batch, 32)    │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 3. 动作历史嵌入                          │
│    输入: prev_actions (batch,)           │
│    Embedding: -> (batch, 32)             │
└─────────────────────────────────────────┘
    ↓
Concatenate: (batch, 512) + (batch, 32) + (batch, 32) = (batch, 576)
    ↓
┌─────────────────────────────────────────┐
│ 4. RNN (LSTM)                            │
│    输入: (batch, 576)                    │
│    输出: (batch, 512)                    │ ← 保存到 aux_loss_state["rnn_output"]
└─────────────────────────────────────────┘
    ↓
返回: features (batch, 512), rnn_hidden_states, aux_loss_state
```

---

## 辅助任务使用状态参数

### People Counting
```python
def forward(self, aux_loss_state, batch):
    scene_features = aux_loss_state['rnn_output']  # (batch, 512)
    # ↑ 这包含：视觉(512) + goal(32) + action(32) → RNN → (512)
    
    # 使用 scene_features 预测人数
    lstm_output, _ = self.lstm(scene_features)
    attn_output, _ = self.attention(lstm_output, lstm_output, lstm_output)
    logits = self.classifier(attn_output)  # (batch, 7) - 0到6人
    
    # Ground truth
    target = batch["observations"]["human_num_sensor"]  # (batch, 1)
    
    loss = self.loss_fn(logits, target)
    return dict(loss=loss)
```

### Guess Human Position
```python
def forward(self, aux_loss_state, batch):
    scene_features = aux_loss_state['rnn_output']  # (batch, 512)
    # ↑ RNN 输出，包含场景的时序信息
    
    # 预测位置
    positions_pred = self.classifier(scene_features)  # (batch, 6, 2)
    
    # Ground truth - 来自观测！
    positions_gt = batch["observations"]["oracle_humanoid_future_trajectory"][:, :, 0, :]
    positions_gt_agent0 = batch["observations"]["localization_sensor"][:, [0, 2]]
    positions_gt_relative = positions_gt - positions_gt_agent0_repeated
    
    loss = self.loss_fn(positions_pred, positions_gt_relative)
    return dict(loss=loss)
```

### Future Trajectory Prediction
```python
def forward(self, aux_loss_state, batch):
    scene_features = aux_loss_state["rnn_output"]  # (batch, 512)
    
    # 还需要额外信息
    human_num_features = batch["observations"]["human_num_sensor"]  # (batch, 1)
    position_features = batch["observations"]["oracle_humanoid_future_trajectory"][:, :, 0, :]  # (batch, 6, 2)
    
    # 拼接
    features = torch.cat((scene_features, human_num_features, position_features.flatten(1)), dim=-1)
    
    # 预测未来轨迹
    positions_pred = self.classifier(features)  # (batch, 6, 4, 2)
    
    # Ground truth
    positions_gt = batch["observations"]["oracle_humanoid_future_trajectory"][:, :, -4:, :]
    
    loss = self.loss_fn(positions_pred, positions_gt_relative)
    return dict(loss=loss)
```

---

## 关键观测来源

| 观测名称 | 配置文件中 | 用途 | 来源 |
|---------|----------|------|------|
| `agent_0_articulated_agent_jaw_depth` | obs_keys | **视觉输入** | 深度相机传感器 |
| `agent_0_pointgoal_with_gps_compass` | obs_keys | **Goal 信息** | GPS + 指南针传感器 |
| `agent_0_human_num_sensor` | obs_keys (仅训练) | **辅助任务 GT** | 人数统计传感器 |
| `agent_0_localization_sensor` | obs_keys (仅训练) | **辅助任务 GT** | Agent 定位传感器 |
| `agent_0_oracle_humanoid_future_trajectory` | obs_keys (仅训练) | **辅助任务 GT** | Oracle 轨迹传感器 |

---

## StreamVLN 的对应实现

### 主要差异

| 组件 | ResNet Policy | StreamVLN Policy |
|------|--------------|-----------------|
| **视觉输入** | Depth (1通道) | **RGB (3通道)** ← 主要差异！ |
| **视觉编码器** | ResNet50 | **StreamVLN Vision Tower** |
| **视觉特征维度** | 512 | **1024** |
| **Goal 处理** | Linear(3→32) | **直接拼接(2)** |
| **动作处理** | Embedding(32) | Embedding(32) ✓ 相同 |
| **时序建模** | LSTM | **无（直接拼接）** |
| **rnn_output 维度** | 512 | **1058** |

### StreamVLN 的提取流程

```python
def forward(self, observations, rnn_hidden_states, prev_actions, masks, ...):
    # 1. 提取 RGB（而不是 Depth）
    rgb = observations['agent_0_articulated_agent_jaw_rgb']  # (batch, H, W, 3)
    
    # 2. StreamVLN Vision Tower
    visual_features = self.model.get_vision_tower()(rgb)  # (batch, 1024)
    
    # 3. 拼接 Goal
    goal_obs = observations['agent_0_pointgoal_with_gps_compass']  # (batch, 2)
    
    # 4. 动作嵌入
    prev_action_feat = self.prev_action_embedding(prev_actions)  # (batch, 32)
    
    # 5. 融合
    fused_features = torch.cat([visual_features, goal_obs, prev_action_feat], dim=1)
    # fused_features: (batch, 1024 + 2 + 32) = (batch, 1058)
    
    # 6. 构建 aux_loss_state
    aux_loss_state = {
        "perception_embed": visual_features,  # (batch, 1024)
        "rnn_output": fused_features,         # (batch, 1058) ← 辅助任务使用
    }
    
    return fused_features, rnn_hidden_states, aux_loss_state
```

---

## 总结对比表

| 特征 | ResNet (原始 Falcon) | StreamVLN (我的实现) |
|------|---------------------|---------------------|
| **输入图像** | Depth (H, W, 1) | **RGB (H, W, 3)** |
| **视觉编码器** | ResNet50 CNN | StreamVLN Vision Tower (Transformer) |
| **视觉特征** | 512维 | **1024维** |
| **Goal 编码** | 极坐标变换 + Linear(32) | 直接使用 (2维) |
| **动作编码** | Embedding(32) | Embedding(32) |
| **拼接后** | (512 + 32 + 32) = 576 | (1024 + 2 + 32) = 1058 |
| **时序建模** | LSTM → 512 | 无 RNN，直接使用 1058 |
| **perception_embed** | 512维 | 1024维 |
| **rnn_output** | 512维 (LSTM 输出) | **1058维 (拼接特征)** |
| **信息丰富度** | 中等 | **更高** |

---

## 为什么 StreamVLN 的实现是有效的？

1. **更丰富的视觉特征**: 1024 vs 512
2. **预训练优势**: Vision Tower 来自 LLaVA，在大规模数据上预训练
3. **RGB vs Depth**: RGB 包含更多语义信息（颜色、纹理）
4. **足够的上下文**: 包含视觉 + Goal + 动作历史
5. **辅助任务兼容**: `rnn_output` 维度更大，信息更丰富

虽然 StreamVLN 没有 RNN，但通过：
- **更大的特征维度**
- **更丰富的预训练知识**
- **RGB 的语义优势**

可以提供**同样甚至更好**的场景表示，供辅助任务使用！

---

*最后更新: 2025-11-11*
