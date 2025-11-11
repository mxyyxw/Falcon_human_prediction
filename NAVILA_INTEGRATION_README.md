# NaVILA集成到Falcon框架说明文档

## 概述

本项目将NaVILA（Navigation with Vision-Language Actions）模型成功集成到Falcon框架（基于Habitat3）中，用于支持动态行人环境下的社交导航任务。

## 项目结构

```
workspace/
├── habitat-baselines/habitat_baselines/
│   ├── rl/ddppo/policy/
│   │   ├── navila/                          # NaVILA模块目录
│   │   │   ├── __init__.py                  # 模块初始化
│   │   │   ├── action_parser.py             # 语言指令到离散动作的解析器
│   │   │   ├── llava/                       # LLAVA视觉-语言模型
│   │   │   └── evaluation/                  # Habitat扩展（来自NaVILA）
│   │   ├── navila_policy.py                 # NaVILA策略（当前未使用）
│   │   └── __init__.py                      # Policy模块注册
│   └── rl/ppo/
│       ├── navila_evaluator.py              # NaVILA专用评估器
│       └── falcon_evaluator.py              # Falcon原始评估器
├── config/social_nav_v2/
│   ├── navila_falcon_hm3d.yaml             # NaVILA评估配置
│   └── navila_falcon_hm3d_train.yaml       # NaVILA训练配置
└── NAVILA_INTEGRATION_README.md            # 本文档
```

## 核心组件

### 1. 动作解析器 (action_parser.py)

**功能**: 将NaVILA生成的自然语言指令解析为Habitat3的离散动作。

**支持的动作**:
- `0`: STOP - 停止
- `1`: MOVE_FORWARD - 前进（每次25cm）
- `2`: TURN_LEFT - 左转（每次15度）
- `3`: TURN_RIGHT - 右转（每次15度）

**特性**:
- 自动提取距离和角度参数
- 支持多步骤动作（如"move forward 75cm"会被分解为3次前进动作）
- 容错处理，对于无法识别的指令返回默认动作

### 2. NaVILA评估器 (navila_evaluator.py)

**功能**: 专门处理NaVILA模型的评估流程。

**核心特性**:
- 直接使用LLAVA模型生成语言指令
- 管理每个环境的历史RGB帧序列
- 处理多步骤动作队列
- 在episode结束时自动重置历史和队列
- 兼容Falcon的动态行人环境

**关键方法**:
- `evaluate_agent()`: 主评估循环
- `_generate_navila_action()`: 使用LLAVA生成单个动作

### 3. 配置文件

#### navila_falcon_hm3d.yaml (评估配置)

**关键配置**:
```yaml
habitat_baselines:
  evaluator:
    _target_: habitat_baselines.rl.ppo.navila_evaluator.NaVILAEvaluator
  
  rl:
    policy:
      agent_0:
        name: "NaVILAPolicy"
        navila_model_path: "pretrained_model/navila_model"
        num_video_frames: 8
        forward_step: 25
        turn_step: 15
```

**特点**:
- 使用NaVILAEvaluator进行评估
- 保留所有动态行人配置（agent_1到agent_6）
- RGB输入已启用（agent_0_articulated_agent_jaw_rgb）
- 单环境运行（num_environments: 1）

#### navila_falcon_hm3d_train.yaml (训练配置)

**与评估配置的差异**:
- `evaluate: False` - 训练模式
- `trainer_name: "falcon_trainer"` - 使用Falcon训练器
- 包含auxiliary_losses配置（行人计数、位置预测等）

## 技术细节

### NaVILA与Habitat3的适配

**挑战**:
1. **版本差异**: NaVILA基于Habitat 0.1.7，Falcon基于Habitat3
2. **架构差异**: NaVILA直接生成语言指令，而Falcon使用policy网络输出动作分布

**解决方案**:
1. **模块隔离**: 将LLAVA模型和相关代码放在独立的navila目录下
2. **Evaluator层面集成**: 在evaluator中直接调用LLAVA，而不是通过policy网络
3. **动作队列机制**: 处理NaVILA的多步骤动作（如"前进75cm" = 3次前进25cm）

### 动作映射

NaVILA语言指令 → 动作解析器 → Habitat3离散动作

```python
# 示例
"The next action is move forward 50 cm" 
  → action=1 (MOVE_FORWARD), num_repeats=2
  → [执行MOVE_FORWARD] → [队列中添加1次MOVE_FORWARD]

"The next action is turn left 30 degree"
  → action=2 (TURN_LEFT), num_repeats=2
  → [执行TURN_LEFT] → [队列中添加1次TURN_LEFT]

"The next action is stop"
  → action=0 (STOP), num_repeats=1
  → [执行STOP]
```

### 历史帧管理

- 每个环境维护独立的历史RGB帧列表
- 最多保留8帧（可配置）
- Episode结束时自动清空
- 使用采样策略：均匀采样历史帧 + 最新帧

## 使用方法

### 1. 准备NaVILA模型

```bash
# 下载或准备NaVILA预训练模型
mkdir -p pretrained_model/navila_model
# 将模型文件放入该目录
```

### 2. 评估

```bash
cd habitat-baselines

# 使用NaVILA进行评估
python -m habitat_baselines.run \
  --config-name=social_nav_v2/navila_falcon_hm3d.yaml \
  habitat_baselines.eval_ckpt_path_dir=pretrained_model/navila_model
```

### 3. 训练（可选）

```bash
# 使用NaVILA进行训练（需要根据具体需求调整）
python -m habitat_baselines.run \
  --config-name=social_nav_v2/navila_falcon_hm3d_train.yaml
```

## 输出

评估完成后会生成：

1. **result.json**: 包含评估指标
   - SR: Success Rate
   - SPL: Success weighted by Path Length
   - PSC: Personal Space Compliance
   - H-Coll: Human Collision
   - Total: 综合得分

2. **actions.json**: 记录每个episode的动作序列

3. **视频**: 如果启用video_option，会生成可视化视频

## 环境配置说明

### 动作配置（在YAML中）

```yaml
habitat:
  task:
    actions:
      agent_0_discrete_stop:
        lin_speed: 0.0 
        ang_speed: 0.0
      agent_0_discrete_move_forward:
        lin_speed: 25.0  # 与forward_step对应
        ang_speed: 0.0
      agent_0_discrete_turn_left:
        lin_speed: 0.0
        ang_speed: 15.0  # 与turn_step对应
      agent_0_discrete_turn_right:
        lin_speed: 0.0
        ang_speed: -15.0
```

### 行人配置

配置保留了原Falcon的6个动态行人（agent_1到agent_6）:
- 使用OracleNavRandCoordAction_Obstacle
- human_joints运动控制
- 速度: 10.0 cm/s (linear), 10.0 deg/s (angular)

## 注意事项

1. **单环境限制**: 当前实现只支持单环境评估（num_environments: 1），因为：
   - LLAVA模型推理较慢
   - 需要维护每个环境的历史帧
   - 简化了队列管理

2. **GPU需求**: LLAVA模型需要GPU支持，建议至少8GB显存

3. **RGB输入**: 确保配置中启用了RGB传感器（agent_0_articulated_agent_jaw_rgb）

4. **指令传感器**: 如果任务有语言指令，需要在observation中提供instruction传感器

## 扩展与改进

### 可能的改进方向

1. **多环境支持**: 优化批处理和并行推理
2. **Fine-tuning**: 在Falcon的社交导航数据上fine-tune LLAVA
3. **Policy集成**: 将LLAVA嵌入到policy网络中，支持端到端训练
4. **动作空间扩展**: 支持更多动作类型（如后退、暂停）

### 添加新动作

1. 在action_parser.py中添加新的动作模式
2. 在配置文件中定义新动作的参数
3. 更新NaVILA的提示模板以生成新动作

## 故障排除

### 常见问题

1. **ImportError: NaVILA modules not available**
   - 检查llava模块是否正确复制到navila目录
   - 确认所有依赖已安装

2. **模型加载失败**
   - 检查navila_model_path是否正确
   - 确认模型文件完整

3. **CUDA out of memory**
   - 减小batch_size（已经是1）
   - 减小num_video_frames
   - 使用更小的LLAVA模型

4. **动作解析错误**
   - 检查NaVILA输出的语言格式
   - 在action_parser.py中添加更多容错逻辑

## 参考资料

- [NaVILA原始仓库](https://github.com/AnjieCheng/NaVILA)
- [Falcon原始仓库](https://github.com/Zeying-Gong/Falcon)
- [Habitat 3.0文档](https://aihabitat.org/)
- [LLAVA论文](https://arxiv.org/abs/2304.08485)

## 贡献者

集成完成于: 2025-11-11

## 许可证

本集成遵循MIT许可证。
