# StreamVLN Evaluator 使用指南

## 概述

`StreamVLNEvaluator` 是专门为 StreamVLN 设计的评估器，与标准的 Falcon `Evaluator` 不同：

- ✅ **直接使用 StreamVLN 的生成能力**：调用 `generate_action_sequence()` 生成动作序列
- ✅ **支持多步规划**：一次生成多个动作，逐步执行
- ✅ **动作缓存机制**：避免每步都调用 LLM（提高效率）
- ✅ **可选的辅助任务监控**：可以计算辅助任务损失（仅监控，不更新）

---

## 与标准 Evaluator 的对比

| 特性 | Falcon Evaluator | StreamVLN Evaluator |
|------|-----------------|---------------------|
| **动作生成方式** | `agent.act()` → 单个动作 | `generate_action_sequence()` → 动作序列 |
| **策略类型** | Policy gradient (PPO) | 生成式 (LLM) |
| **每步调用** | 是 | 否（使用缓存） |
| **多步规划** | 否 | 是 |
| **辅助任务** | 完全集成 | 可选监控 |
| **适用场景** | PPO 训练和评估 | StreamVLN 纯推理 |

---

## 使用方法

### 1. 配置文件设置

创建或修改配置文件 `streamvln_hm3d_eval.yaml`：

```yaml
defaults:
  - /habitat_baselines: habitat_baselines_rl_config_base
  - /habitat_baselines/rl/policy/agent_0@habitat_baselines.rl.policy.agent_0: streamvln_policy
  - _self_

habitat_baselines:
  # 使用 StreamVLN Evaluator
  evaluator_class: StreamVLNEvaluator  # ← 关键！
  
  evaluate: True
  
  num_environments: 4  # StreamVLN 较慢，建议减少环境数
  
  rl:
    policy:
      agent_0:
        name: StreamVLNPolicy
        model_path: /path/to/streamvln/model  # ← StreamVLN 模型路径
        
        # StreamVLN 特定配置
        num_frames: 32
        num_history: 8
        generate_every_n_steps: 1  # 每 N 步生成新动作序列（1=每次都生成）
    
    # 可选：监控辅助任务损失
    compute_aux_losses_in_eval: False  # True 则计算辅助损失
    
    auxiliary_losses:
      people_counting:
        max_human_num: 6
      guess_human_position:
        max_human_num: 6
      future_trajectory_prediction:
        max_human_num: 6
        future_step: 4

habitat:
  gym:
    obs_keys:
      - agent_0_articulated_agent_jaw_rgb  # ← StreamVLN 需要 RGB
      - agent_0_articulated_agent_jaw_depth
      - agent_0_pointgoal_with_gps_compass
```

---

### 2. 运行评估

```bash
python habitat-baselines/habitat_baselines/run.py \
  --config-name=social_nav_v2/streamvln_hm3d_eval.yaml \
  habitat_baselines.rl.policy.agent_0.model_path=/path/to/streamvln/model \
  habitat_baselines.num_environments=4
```

---

### 3. 代码中使用

```python
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ppo.streamvln_evaluator import StreamVLNEvaluator

# 1. 创建 agent
config = get_config(...)
agent = agent_class(...)

# 2. 创建环境
envs = construct_envs(config, ...)

# 3. 创建 evaluator
evaluator = StreamVLNEvaluator(
    config=config,
    agent=agent,
    envs=envs,
    device=device
)

# 4. 运行评估
metrics = evaluator.evaluate_agent(
    num_eval_episodes=100,
    writer=tensorboard_writer,
    checkpoint_index=0
)

print(f"Success rate: {metrics['eval_success_mean']:.2%}")
print(f"SPL: {metrics['eval_spl_mean']:.2%}")
```

---

## 工作流程

### 主评估循环

```
1. 初始化
   ├─ 重置环境
   ├─ 清空动作缓存
   └─ 重置 StreamVLN episode state

2. 对每个环境
   ├─ 检查动作缓存
   │  ├─ 如果为空：
   │  │  ├─ 获取 RGB 观测
   │  │  ├─ 生成指令（从 pointgoal）
   │  │  ├─ 调用 generate_action_sequence()
   │  │  └─ 填充动作缓存
   │  └─ 如果不为空：
   │     └─ 从缓存取出动作
   └─ 执行动作

3. 更新统计
   ├─ 收集奖励、成功率等
   └─ 记录到 tensorboard

4. 处理完成的回合
   ├─ 收集 episode 指标
   ├─ 重置该环境
   └─ 清空该环境的动作缓存

5. 返回聚合指标
```

---

## 动作缓存机制

### 为什么需要动作缓存？

StreamVLN 的 LLM 生成很慢（~1秒/次），每步都调用会导致：
- ❌ 评估时间过长
- ❌ GPU 利用率低
- ❌ 不符合 StreamVLN 的设计（多步规划）

### 缓存策略

```python
# 每个环境维护一个动作队列
action_buffers = [deque() for _ in range(num_envs)]

# 在每步：
if len(action_buffers[env_idx]) == 0:
    # 缓存为空，生成新序列
    action_seq = generate_action_sequence(...)  # 例如: [1, 1, 2, 3, 0]
    action_buffers[env_idx].extend(action_seq)

# 取出下一个动作
action = action_buffers[env_idx].popleft()
```

### 配置选项

```yaml
# 控制生成频率（未实现，可扩展）
generate_every_n_steps: 1  # 每 N 步强制重新生成
```

---

## 辅助任务监控（可选）

### 启用方式

```yaml
habitat_baselines:
  rl:
    compute_aux_losses_in_eval: True  # ← 启用
    
    auxiliary_losses:
      people_counting:
        max_human_num: 6
      guess_human_position:
        max_human_num: 6
      future_trajectory_prediction:
        max_human_num: 6
        future_step: 4
```

### 工作原理

1. **提取特征**：
   ```python
   visual_features = agent.net.model.get_vision_tower()(rgb)
   features = cat([visual_features.mean(1), pointgoal])
   
   aux_loss_state = {
       "perception_embed": visual_features.mean(1),
       "rnn_output": features
   }
   ```

2. **计算损失**（不更新）：
   ```python
   with torch.no_grad():
       for aux_task in aux_tasks:
           loss = aux_task(aux_loss_state, batch)
           # 仅记录，不反向传播
   ```

3. **记录到 tensorboard**：
   - `eval_aux_people_counting_loss`
   - `eval_aux_guess_human_position_loss`
   - `eval_aux_future_trajectory_prediction_loss`

### 注意事项

- ⚠️ 辅助任务**不会更新** StreamVLN 的参数
- ⚠️ 这仅用于**监控**辅助任务在 StreamVLN 特征上的表现
- ⚠️ 如果要训练辅助任务，需要使用不同的 trainer（见下文）

---

## 指令生成

### 当前实现（简单代理）

由于 Falcon 是 point navigation 任务（没有语言指令），我们从 `pointgoal_with_gps_compass` 生成简单指令：

```python
def _get_instruction_from_pointgoal(observations, env_idx):
    pointgoal = observations['agent_0_pointgoal_with_gps_compass'][env_idx]
    distance = pointgoal[0].item()
    angle = pointgoal[1].item()
    
    if distance < 0.5:
        return "you are near the goal, please stop"
    elif abs(angle) > 0.5:
        if angle > 0:
            return "turn left towards the goal"
        else:
            return "turn right towards the goal"
    else:
        return "move forward to reach the goal"
```

### 扩展到真实 VLN

如果使用真实的 VLN 数据集（如 R2R, RxR）：

```python
# 从 episode 中获取语言指令
instruction = current_episodes[env_idx].instruction

# 或者从 observations 中获取
instruction = observations['instruction'][env_idx]
```

---

## 评估指标

### 标准指标

- `eval_reward_mean/std`: 平均奖励
- `eval_length_mean/std`: Episode 长度
- `eval_success_mean/std`: 成功率
- `eval_spl_mean/std`: Success weighted by Path Length
- `eval_distance_to_goal_mean/std`: 最终距离目标的距离

### StreamVLN 特有指标（可扩展）

```python
# 在 StreamVLNEvaluator 中添加
self.episode_stats['num_generations'].append(num_generations)
self.episode_stats['avg_action_seq_length'].append(avg_seq_len)
```

---

## 性能考虑

### StreamVLN 的瓶颈

1. **LLM 生成慢**：~1秒/次
   - 解决：动作缓存机制
   - 一次生成多步动作

2. **视觉编码慢**：Vision Tower 较大
   - 解决：使用 bfloat16
   - 减少环境数（`num_environments: 4`）

3. **Memory 管理**：StreamVLN 维护历史帧
   - 解决：定期重置（`num_frames: 32`）
   - 使用 `num_history` 控制历史长度

### 建议配置

```yaml
habitat_baselines:
  num_environments: 4  # 不要太多
  
  rl:
    policy:
      agent_0:
        num_frames: 32       # 每 32 帧重置一次 memory
        num_history: 8       # 保留 8 帧历史
        num_future_steps: 4  # 生成 4 步动作
```

---

## 训练 vs 评估

### StreamVLN 的推荐使用方式

| 阶段 | 方法 | 说明 |
|------|------|------|
| **预训练** | StreamVLN 原始训练 | 在 VLN 数据集上预训练（已完成） |
| **评估** | `StreamVLNEvaluator` | 使用本 evaluator 在 Falcon 环境评估 |
| **训练**（可选） | 方案 A 或 方案 C | 见下文 |

---

## 如果要训练？

### 方案 A：StreamVLN 作为特征提取器 + PPO 训练

使用当前的 `StreamVLNPolicy` + 标准 `FalconTrainer`：

```yaml
# 使用标准的 Falcon trainer
trainer_class: FalconTrainer  # 不是 StreamVLNEvaluator

rl:
  policy:
    agent_0:
      name: StreamVLNPolicy
      # ... StreamVLN 配置
  
  # PPO 训练
  ppo:
    clip_param: 0.2
    ppo_epoch: 4
    num_mini_batch: 2
```

**工作原理**：
1. `StreamVLNNet.forward()` 提取特征
2. `NetPolicy.act()` 使用 `action_distribution` 采样动作
3. PPO 更新 policy 和 critic
4. 辅助任务正常工作

**优点**：
- ✅ 兼容 Falcon 的 PPO 框架
- ✅ 辅助任务完全集成
- ✅ 可以 fine-tune

**缺点**：
- ❌ 不使用 StreamVLN 的生成能力
- ❌ LLM 部分被浪费

---

### 方案 C：混合方案

- **训练时**：使用方案 A（PPO + 特征提取）
- **评估时**：使用 `StreamVLNEvaluator`（生成式）

```python
class StreamVLNPolicy(NetPolicy):
    def __init__(self, ...):
        self.use_generation = False  # 默认关闭
    
    def act(self, observations, ...):
        if self.use_generation:
            # 生成式（评估）
            ...
        else:
            # PPO 采样（训练）
            ...
```

---

## 故障排查

### 问题 1: 找不到 StreamVLNEvaluator

**症状**：
```
KeyError: 'StreamVLNEvaluator'
```

**解决**：
1. 确认 `streamvln_evaluator.py` 已创建
2. 确认使用 `@baseline_registry.register_evaluator` 装饰器
3. 检查配置文件中 `evaluator_class: StreamVLNEvaluator`

---

### 问题 2: generate_action_sequence 失败

**症状**：
```
AttributeError: 'StreamVLNNet' object has no attribute 'generate_action_sequence'
```

**解决**：
确认 `streamvln_policy.py` 中有 `generate_action_sequence` 方法（Line 577+）

---

### 问题 3: RGB 观测缺失

**症状**：
```
KeyError: 'agent_0_articulated_agent_jaw_rgb'
```

**解决**：
在配置文件中启用 RGB 传感器：
```yaml
habitat:
  gym:
    obs_keys:
      - agent_0_articulated_agent_jaw_rgb  # ← 添加这个
```

---

### 问题 4: 评估很慢

**症状**：每个 episode 需要很长时间

**解决**：
1. 减少环境数：`num_environments: 4` → `num_environments: 2`
2. 减少生成频率（如果实现）
3. 使用 GPU 加速：确保 StreamVLN 在 GPU 上

---

## 总结

| 使用场景 | 推荐方案 | Evaluator |
|---------|---------|-----------|
| **只评估** StreamVLN | ✅ 推荐 | `StreamVLNEvaluator` |
| **训练** 新 policy | 方案 A | `FalconTrainer` |
| **训练** + **评估** | 方案 C | 训练用 `FalconTrainer`，评估用 `StreamVLNEvaluator` |

---

*文档创建时间: 2025-11-11*
