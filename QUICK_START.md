# NaVILA-Falcon集成快速入门指南

## 🎉 集成已完成！

NaVILA（Habitat 0.1.7）已成功集成到Falcon（Habitat3）框架中。

## 📋 前置要求

1. **Habitat3环境已安装**
2. **GPU环境**（至少8GB显存）
3. **NaVILA预训练模型**

## 🚀 快速开始

### 1. 准备模型

```bash
# 创建模型目录
mkdir -p pretrained_model/navila_model

# 将NaVILA预训练模型放入该目录
# 模型应包含：
# - tokenizer配置
# - LLAVA模型权重
# - 图像处理器配置
```

### 2. 运行评估

```bash
cd habitat-baselines

# 评估NaVILA策略
python -m habitat_baselines.run \
  --config-name=social_nav_v2/navila_falcon_hm3d.yaml \
  habitat_baselines.eval_ckpt_path_dir=pretrained_model/navila_model
```

### 3. 查看结果

```bash
# 查看评估指标
cat output/result.json

# 查看动作记录
cat output/actions.json

# 查看视频（如果启用）
ls -la evaluation/navila_falcon/hm3d/video/
```

## 📁 核心文件位置

```
workspace/
├── 配置文件:
│   ├── habitat-baselines/habitat_baselines/config/social_nav_v2/
│   │   ├── navila_falcon_hm3d.yaml         # 评估配置
│   │   └── navila_falcon_hm3d_train.yaml   # 训练配置
│
├── 核心代码:
│   ├── habitat-baselines/habitat_baselines/rl/ddppo/policy/
│   │   ├── navila/action_parser.py         # 动作解析器
│   │   ├── navila/llava/                   # LLAVA模型
│   │   └── navila_policy.py                # NaVILA策略
│   └── habitat-baselines/habitat_baselines/rl/ppo/
│       └── navila_evaluator.py             # NaVILA评估器
│
└── 文档:
    ├── NAVILA_INTEGRATION_README.md        # 完整文档
    ├── INTEGRATION_SUMMARY.md              # 集成总结
    └── QUICK_START.md                      # 本文档
```

## ⚙️ 配置说明

### 关键配置项（在YAML文件中）

```yaml
habitat_baselines:
  # 评估器类型
  evaluator:
    _target_: habitat_baselines.rl.ppo.navila_evaluator.NaVILAEvaluator
  
  # 模型路径
  eval_ckpt_path_dir: "pretrained_model/navila_model"
  
  # 环境数量（当前固定为1）
  num_environments: 1
  
  rl:
    policy:
      agent_0:
        # 策略类型
        name: "NaVILAPolicy"
        
        # NaVILA模型路径
        navila_model_path: "pretrained_model/navila_model"
        
        # 视频帧数
        num_video_frames: 8
        
        # 动作步长
        forward_step: 25    # 前进步长（cm）
        turn_step: 15       # 转向步长（度）
```

## 🎯 动作说明

NaVILA生成的语言指令会被自动转换为Habitat3动作：

| 语言指令 | 动作ID | Habitat3动作 | 说明 |
|---------|--------|-------------|------|
| "stop" | 0 | STOP | 停止移动 |
| "move forward N cm" | 1 | MOVE_FORWARD | 每次25cm |
| "turn left N degree" | 2 | TURN_LEFT | 每次15度 |
| "turn right N degree" | 3 | TURN_RIGHT | 每次15度 |

### 多步骤动作示例

```
NaVILA输出: "The next action is move forward 75 cm"
↓
动作解析: action=1 (MOVE_FORWARD), repeats=3
↓
执行: 
  - 立即执行 MOVE_FORWARD (1/3)
  - 队列添加 MOVE_FORWARD (2/3)
  - 队列添加 MOVE_FORWARD (3/3)
↓
后续步骤: 从队列中依次取出并执行
```

## 🔧 常见问题

### Q1: 导入错误 "No module named 'habitat'"
**A**: 这是正常的，说明habitat环境未安装。在实际使用时需要先安装habitat3。

### Q2: CUDA out of memory
**A**: 
- 减少`num_video_frames`（如改为4）
- 使用更小的LLAVA模型
- 增大GPU显存

### Q3: 模型加载失败
**A**: 检查模型路径配置：
```yaml
# 方法1: 在命令行指定
python -m habitat_baselines.run \
  --config-name=... \
  habitat_baselines.eval_ckpt_path_dir=/path/to/model

# 方法2: 在YAML中修改
eval_ckpt_path_dir: "/absolute/path/to/navila_model"
```

### Q4: 评估速度慢
**A**: 这是正常的，因为：
- LLAVA推理需要1-2秒/步
- 处理8帧历史图像
- 语言生成和解析
- 建议：使用GPU加速，考虑模型量化

## 📊 输出说明

### result.json 格式
```json
{
  "SR": 0.75,          // Success Rate（成功率）
  "SPL": 0.65,         // Success weighted by Path Length
  "PSC": 0.80,         // Personal Space Compliance
  "H-Coll": 0.05,      // Human Collision（人体碰撞率）
  "Total": 0.68        // 综合得分
}
```

### actions.json 格式
```json
{
  "scene_id|episode_id|eval_count": [
    {"type": "scalar", "value": 1},  // MOVE_FORWARD
    {"type": "scalar", "value": 2},  // TURN_LEFT
    ...
  ]
}
```

## 🧪 测试

### 运行单元测试
```bash
# 测试动作解析器（不需要habitat环境）
python3 test_action_parser_standalone.py

# 预期输出:
# ✓✓✓ 所有测试通过！
# 通过: 10/10
```

### 完整集成测试
```bash
# 需要habitat环境
python3 test_navila_integration.py
```

## 📚 更多信息

- **完整文档**: `NAVILA_INTEGRATION_README.md`
- **集成总结**: `INTEGRATION_SUMMARY.md`
- **源码注释**: 所有核心文件都有详细的中英文注释

## 🎯 核心功能

✅ **动态行人支持**: 保留Falcon的6个动态行人
✅ **语言指令解析**: 自动将NaVILA输出转为动作
✅ **历史帧管理**: 支持8帧视频输入
✅ **多步骤动作**: 自动处理复杂动作序列
✅ **评估指标**: 完整的评估指标输出

## 🚧 已知限制

1. **单环境**: 当前只支持num_environments=1
2. **RGB必需**: 必须启用RGB传感器
3. **推理速度**: LLAVA推理较慢（~1-2秒/步）

## 💡 提示

1. **首次运行**: 可能需要下载tokenizer和模型
2. **调试模式**: 设置`verbose: True`查看详细日志
3. **视频记录**: 启用`video_option: ["disk"]`生成可视化视频
4. **日志查看**: TensorBoard日志在`evaluation/navila_falcon/hm3d/tb/`

## 📞 需要帮助？

参考详细文档或检查代码注释：
- 动作解析: `habitat-baselines/habitat_baselines/rl/ddppo/policy/navila/action_parser.py`
- 评估流程: `habitat-baselines/habitat_baselines/rl/ppo/navila_evaluator.py`
- 配置参数: `habitat-baselines/habitat_baselines/config/social_nav_v2/navila_falcon_hm3d.yaml`

---

**准备好了吗？开始你的NaVILA-Falcon之旅！** 🚀
