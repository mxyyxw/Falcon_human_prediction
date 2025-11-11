# StreamVLN Policy 重构验证清单

## 📋 代码验证

### 1. Goal 编码

**文件**: `habitat-baselines/habitat_baselines/rl/ddppo/policy/streamvln_policy.py`

#### ✅ 初始化（Line ~278）
```python
# 检查是否存在
self.tgt_embeding = nn.Linear(3, 32).to(self.device)
```

#### ✅ 极坐标变换方法（Line ~335-349）
```python
def _polar_transform_goal(self, goal_observations: torch.Tensor):
    if goal_observations.shape[1] == 2:
        goal_observations = torch.stack([
            goal_observations[:, 0],              # distance
            torch.cos(-goal_observations[:, 1]),  # cos(-angle)
            torch.sin(-goal_observations[:, 1]),  # sin(-angle)
        ], -1)
    return goal_observations
```

#### ✅ Forward 中使用（Line ~543-550）
```python
if 'agent_0_pointgoal_with_gps_compass' in observations:
    goal_observations = observations['agent_0_pointgoal_with_gps_compass']
    goal_observations = self._polar_transform_goal(goal_observations)
    goal_embed = self.tgt_embeding(goal_observations)
    x.append(goal_embed)
```

---

### 2. Action 编码

#### ✅ 初始化（Line ~265-276）
```python
self._n_prev_action = 32
if discrete_actions:
    self.prev_action_embedding = nn.Embedding(
        action_space.n + 1, self._n_prev_action
    )
else:
    num_actions = get_num_actions(action_space)
    self.prev_action_embedding = nn.Linear(
        num_actions, self._n_prev_action
    )
```

#### ✅ Forward 中使用（Line ~553-563）
```python
if self.discrete_actions:
    prev_actions = prev_actions.squeeze(-1)
    start_token = torch.zeros_like(prev_actions)
    prev_actions = self.prev_action_embedding(
        torch.where(masks.view(-1), prev_actions + 1, start_token)
    )
else:
    prev_actions = self.prev_action_embedding(
        masks * prev_actions.float()
    )
x.append(prev_actions)
```

---

### 3. Visual FC

#### ✅ 动态创建（Line ~521-526）
```python
if visual_features.shape[-1] != self._hidden_size:
    if not hasattr(self, 'visual_fc'):
        self.visual_fc = nn.Sequential(
            nn.Linear(visual_features.shape[-1], self._hidden_size),
            nn.ReLU(True),  # ← 确认有 ReLU
        ).to(device)
    visual_features = self.visual_fc(visual_features)
```

---

### 4. 特征融合

#### ✅ 初始化（Line ~538-540）
```python
x = []  # ← 从空列表开始
aux_loss_state = {}
```

#### ✅ 顺序（Line ~542-567）
```python
# 1. 视觉
aux_loss_state["perception_embed"] = visual_feats
x.append(visual_feats)

# 2. Goal
if 'agent_0_pointgoal_with_gps_compass' in observations:
    ...
    x.append(goal_embed)

# 3. Action
...
x.append(prev_actions)

# 4. 拼接
out = torch.cat(x, dim=1)
```

#### ✅ 保存和返回（Line ~570, 577）
```python
aux_loss_state["rnn_output"] = out
...
return out, new_rnn_hidden_states, aux_loss_state
```

---

## 🧪 运行时验证

### 方法 1: 添加调试代码

在 `streamvln_policy.py` 的 `forward` 方法中添加：

```python
# 在 Line ~575 附近（return 之前）
if self.step_id == 0:  # 只在第一步打印
    print(f"\n{'='*60}")
    print(f"StreamVLN Policy 特征维度验证")
    print(f"{'='*60}")
    print(f"Visual features:    {visual_feats.shape}")
    print(f"Goal embedding:     {goal_embed.shape}")
    print(f"Action embedding:   {prev_actions.shape}")
    print(f"Concatenated (out): {out.shape}")
    print(f"perception_embed:   {aux_loss_state['perception_embed'].shape}")
    print(f"rnn_output:         {aux_loss_state['rnn_output'].shape}")
    print(f"{'='*60}\n")
```

**预期输出** (batch_size=4, hidden_size=1024):
```
============================================================
StreamVLN Policy 特征维度验证
============================================================
Visual features:    torch.Size([4, 1024])
Goal embedding:     torch.Size([4, 32])
Action embedding:   torch.Size([4, 32])
Concatenated (out): torch.Size([4, 1088])
perception_embed:   torch.Size([4, 1024])
rnn_output:         torch.Size([4, 1088])
============================================================
```

---

### 方法 2: 运行测试脚本

```bash
# 在配置好的环境中运行
python test_streamvln_refactor.py
```

**预期输出**: 所有测试通过 ✅

---

### 方法 3: 检查训练日志

运行训练后，检查日志中是否包含辅助任务损失：

```bash
python habitat-baselines/habitat_baselines/run.py \
  --config-name=social_nav_v2/streamvln_hm3d_train.yaml \
  habitat_baselines.rl.policy.agent_0.model_path=/path/to/model
```

**预期日志**:
```
Losses:
  policy_loss: 0.xxx
  people_counting: 0.xxx              ← 应该有值
  guess_human_position: 0.xxx         ← 应该有值
  future_trajectory_prediction: 0.xxx ← 应该有值
  total_loss: 0.xxx
```

如果这三个辅助任务损失都有正常值（不是 NaN），说明重构成功！

---

## 📊 对比验证

### 与 PointNavResNetNet 对比

| 检查项 | PointNavResNetNet | StreamVLNNet | 状态 |
|--------|------------------|--------------|------|
| Goal 编码器类型 | `nn.Linear(3, 32)` | `nn.Linear(3, 32)` | ✅ |
| Goal 变换 | 极坐标 → 笛卡尔 | 极坐标 → 笛卡尔 | ✅ |
| Action 编码器 | `nn.Embedding` | `nn.Embedding` | ✅ |
| Action 处理 | mask + start_token | mask + start_token | ✅ |
| Visual FC | Linear + ReLU | Linear + ReLU | ✅ |
| 拼接顺序 | Visual→Goal→Action | Visual→Goal→Action | ✅ |
| aux_loss_state | perception_embed + rnn_output | perception_embed + rnn_output | ✅ |
| 返回值 | (out, rnn_hidden, aux) | (out, rnn_hidden, aux) | ✅ |

---

## 🔍 常见问题排查

### 问题 1: 找不到 `tgt_embeding`

**症状**: `AttributeError: 'StreamVLNNet' object has no attribute 'tgt_embeding'`

**检查**:
- Line ~278 是否有 `self.tgt_embeding = nn.Linear(3, 32).to(self.device)`

---

### 问题 2: Goal 维度错误

**症状**: `RuntimeError: shape mismatch` 在 goal embedding

**检查**:
- Line ~547 是否调用了 `self._polar_transform_goal(goal_observations)`
- 确认输入是 (batch, 2)，输出是 (batch, 3)

---

### 问题 3: Action 维度错误

**症状**: `RuntimeError: shape mismatch` 在 prev_actions

**检查**:
- Line ~553-563 是否正确处理了 discrete_actions
- 是否使用了 `squeeze(-1)` 和 `start_token`

---

### 问题 4: 辅助任务损失为 NaN

**症状**: 训练时辅助任务损失显示为 NaN

**检查**:
1. `aux_loss_state["rnn_output"]` 是否正确保存 (Line ~570)
2. 特征维度是否匹配 (应为 batch, hidden_size+64)
3. 是否启用了必要的传感器 (human_num_sensor, localization_sensor 等)

---

### 问题 5: Visual features 维度不匹配

**症状**: `RuntimeError: shape mismatch` 在 visual_fc

**检查**:
- Line ~521-526 的 visual_fc 是否正确创建
- 确认 `visual_features.shape[-1]` 和 `self._hidden_size` 的值

---

## ✅ 最终检查清单

在运行训练/评估之前，确认以下所有项：

- [ ] Goal 编码器已添加 (`self.tgt_embeding`)
- [ ] 极坐标变换方法已实现 (`_polar_transform_goal`)
- [ ] Action 编码器已完整实现 (支持离散/连续)
- [ ] Visual FC 包含 ReLU 激活
- [ ] 特征融合顺序正确 (Visual → Goal → Action)
- [ ] `aux_loss_state` 包含 `perception_embed` 和 `rnn_output`
- [ ] Forward 返回 `(out, rnn_hidden_states, aux_loss_state)`
- [ ] 配置文件已更新 (`streamvln_hm3d.yaml`, `streamvln_hm3d_train.yaml`)
- [ ] RGB 传感器已启用 (`agent_0_articulated_agent_jaw_rgb`)
- [ ] StreamVLN 模型路径已配置

---

## 📚 参考文档

如果遇到问题，请查阅：

1. **STREAMVLN_POLICY_REFACTOR.md** - 详细的重构说明
2. **FALCON_STATE_EXTRACTION_EXPLAINED.md** - 原始 Falcon 实现
3. **REFACTOR_SUMMARY.md** - 使用指南
4. **REFACTOR_FILES_LIST.txt** - 代码行号对应表

---

## 🎯 成功标志

当看到以下情况时，说明重构成功：

✅ 训练/评估能够正常启动  
✅ 特征维度打印正确 (1088 for hidden_size=1024)  
✅ 三个辅助任务损失都有正常值  
✅ 没有维度相关的错误  
✅ 性能符合预期或更好  

---

*验证清单创建时间: 2025-11-11*
