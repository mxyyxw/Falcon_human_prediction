# NaVILA-Falcon Integration

<div align="center">

![Status](https://img.shields.io/badge/Status-✅_Completed-success)
![Habitat](https://img.shields.io/badge/Habitat-3.0-blue)
![NaVILA](https://img.shields.io/badge/NaVILA-Integrated-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

**将NaVILA视觉-语言导航模型集成到Falcon社交导航框架**

[English](#english) | [中文](#chinese)

</div>

---

<a name="chinese"></a>

## 🎯 项目概述

本项目成功将**NaVILA**（Navigation with Vision-Language Actions，基于Habitat 0.1.7）集成到**Falcon**框架（基于Habitat 3.0）中，实现了基于视觉-语言模型的社交导航策略，支持动态行人环境。

### 核心特性

- ✅ **视觉-语言导航**: 使用LLAVA模型将RGB图像序列转换为自然语言导航指令
- ✅ **语言到动作解析**: 自动将语言指令转换为Habitat3离散动作
- ✅ **动态行人环境**: 完全兼容Falcon的多智能体社交导航场景
- ✅ **多步骤动作**: 智能处理复杂动作序列（如"前进75cm" = 3步）
- ✅ **历史帧管理**: 支持视频输入（8帧历史RGB）
- ✅ **版本适配**: Habitat 0.1.7 → Habitat 3.0无缝迁移

## 📦 项目结构

```
workspace/
├── 📄 文档
│   ├── README_NAVILA_FALCON.md          # 本文档
│   ├── QUICK_START.md                   # 快速入门指南 ⭐
│   ├── NAVILA_INTEGRATION_README.md     # 完整技术文档
│   └── INTEGRATION_SUMMARY.md           # 集成总结
│
├── 🧪 测试
│   ├── test_navila_integration.py       # 完整集成测试
│   └── test_action_parser_standalone.py # 动作解析器测试 ✅
│
└── 🔧 代码
    └── habitat-baselines/habitat_baselines/
        ├── rl/ddppo/policy/
        │   ├── navila/                  # NaVILA模块
        │   │   ├── action_parser.py     # 动作解析器 ⭐
        │   │   ├── llava/               # LLAVA模型
        │   │   └── evaluation/          # Habitat扩展
        │   └── navila_policy.py         # NaVILA策略
        ├── rl/ppo/
        │   └── navila_evaluator.py      # NaVILA评估器 ⭐
        └── config/social_nav_v2/
            ├── navila_falcon_hm3d.yaml       # 评估配置 ⭐
            └── navila_falcon_hm3d_train.yaml # 训练配置
```

## 🚀 快速开始

### 1️⃣ 准备模型

```bash
mkdir -p pretrained_model/navila_model
# 将NaVILA预训练模型放入该目录
```

### 2️⃣ 运行评估

```bash
cd habitat-baselines
python -m habitat_baselines.run \
  --config-name=social_nav_v2/navila_falcon_hm3d.yaml \
  habitat_baselines.eval_ckpt_path_dir=pretrained_model/navila_model
```

### 3️⃣ 查看结果

```bash
cat output/result.json  # 评估指标
cat output/actions.json # 动作序列
```

**更多详细信息**: 请参阅 [QUICK_START.md](QUICK_START.md) ⭐

## 📊 技术架构

```mermaid
graph LR
    A[RGB图像序列] --> B[LLAVA模型]
    B --> C[语言指令]
    C --> D[动作解析器]
    D --> E[离散动作]
    E --> F[Habitat3环境]
    F --> G[行人 + 机器人]
```

### 动作映射表

| NaVILA输出 | 解析结果 | Habitat3动作 | 步长 |
|-----------|---------|-------------|------|
| "stop" | action=0 | STOP | - |
| "move forward 25 cm" | action=1, repeats=1 | MOVE_FORWARD | 25cm |
| "move forward 50 cm" | action=1, repeats=2 | MOVE_FORWARD×2 | 25cm×2 |
| "turn left 15 degree" | action=2, repeats=1 | TURN_LEFT | 15° |
| "turn left 30 degree" | action=2, repeats=2 | TURN_LEFT×2 | 15°×2 |
| "turn right 45 degree" | action=3, repeats=3 | TURN_RIGHT×3 | 15°×3 |

## 🧪 测试结果

```bash
$ python3 test_action_parser_standalone.py

============================================================
测试NaVILA动作解析器
Testing NaVILA Action Parser
============================================================
✓ 停止: 动作=0, 重复=1
✓ 前进25cm: 动作=1, 重复=1
✓ 前进50cm: 动作=1, 重复=2
✓ 前进75cm: 动作=1, 重复=3
✓ 左转15度: 动作=2, 重复=1
✓ 左转30度: 动作=2, 重复=2
✓ 左转45度: 动作=2, 重复=3
✓ 右转15度: 动作=3, 重复=1
✓ 右转30度: 动作=3, 重复=2
✓ 右转45度: 动作=3, 重复=3
------------------------------------------------------------
通过: 10/10 ✅
失败: 0/10
```

## 💡 关键创新

### 1. 版本适配策略
- **模块隔离**: NaVILA代码独立封装在`navila/`目录
- **Evaluator桥接**: 在evaluator层面集成，避免修改policy核心
- **兼容性保持**: Falcon原有功能完全保留

### 2. 动作队列机制
```python
# 示例：处理复杂动作
输入: "move forward 75 cm"
↓
解析: action=1, repeats=3
↓
执行:
  - 第1步: 立即执行 MOVE_FORWARD
  - 第2-3步: 加入队列
↓
后续: 从队列依次执行
```

### 3. 历史帧采样
```python
# 智能采样策略
历史帧: [frame1, frame2, ..., frame_n]
↓
采样: 均匀采样(n-1帧) + 最新帧
↓
输出: 固定8帧 → LLAVA模型
```

## 📈 性能指标

| 指标 | 说明 | 示例值 |
|-----|------|--------|
| SR | Success Rate（成功率） | 0.75 |
| SPL | Success weighted by Path Length | 0.65 |
| PSC | Personal Space Compliance | 0.80 |
| H-Coll | Human Collision（人体碰撞率） | 0.05 |
| **Total** | **综合得分** | **0.68** |

## 🔧 配置文件

### 评估配置 (navila_falcon_hm3d.yaml)
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

### 动态行人配置
```yaml
# 保留Falcon的6个动态行人
agent_1 到 agent_6:
  - OracleNavRandCoordAction_Obstacle
  - human_joints运动控制
  - 速度: 10 cm/s, 10 deg/s
```

## 📚 文档索引

| 文档 | 描述 | 目标读者 |
|------|------|---------|
| [QUICK_START.md](QUICK_START.md) ⭐ | 快速入门指南 | 初次使用者 |
| [NAVILA_INTEGRATION_README.md](NAVILA_INTEGRATION_README.md) | 完整技术文档 | 开发者 |
| [INTEGRATION_SUMMARY.md](INTEGRATION_SUMMARY.md) | 集成总结 | 项目管理者 |
| README_NAVILA_FALCON.md | 本文档 | 所有人 |

## ⚠️ 注意事项

### 系统要求
- ✅ GPU: 至少8GB显存
- ✅ Habitat3环境
- ✅ Python 3.8+
- ✅ PyTorch + CUDA

### 已知限制
- 🔸 单环境评估（num_environments=1）
- 🔸 RGB输入必需
- 🔸 LLAVA推理速度: ~1-2秒/步

## 🎓 引用

如果您使用了本项目，请引用：

```bibtex
@misc{navila_falcon_integration,
  title={NaVILA-Falcon Integration: Vision-Language Navigation in Dynamic Environments},
  author={Your Name},
  year={2025},
  howpublished={\url{https://github.com/your-repo}}
}
```

原始论文:
- **NaVILA**: [https://github.com/AnjieCheng/NaVILA](https://github.com/AnjieCheng/NaVILA)
- **Falcon**: [https://github.com/Zeying-Gong/Falcon](https://github.com/Zeying-Gong/Falcon)

## 📞 支持与反馈

- 📖 **文档问题**: 参考[NAVILA_INTEGRATION_README.md](NAVILA_INTEGRATION_README.md)
- 🐛 **Bug报告**: 检查代码注释和日志
- 💬 **讨论**: 欢迎提出改进建议

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

---

<a name="english"></a>

## 🎯 Project Overview (English)

This project successfully integrates **NaVILA** (Navigation with Vision-Language Actions, based on Habitat 0.1.7) into the **Falcon** framework (based on Habitat 3.0), enabling vision-language model-based social navigation with dynamic pedestrian environments.

### Key Features

- ✅ **Vision-Language Navigation**: Uses LLAVA model to convert RGB image sequences into natural language navigation instructions
- ✅ **Language-to-Action Parsing**: Automatically converts language instructions into Habitat3 discrete actions
- ✅ **Dynamic Pedestrian Environment**: Fully compatible with Falcon's multi-agent social navigation scenarios
- ✅ **Multi-step Actions**: Intelligently handles complex action sequences (e.g., "forward 75cm" = 3 steps)
- ✅ **Historical Frame Management**: Supports video input (8 historical RGB frames)
- ✅ **Version Adaptation**: Seamless migration from Habitat 0.1.7 → Habitat 3.0

## 🚀 Quick Start

```bash
# 1. Prepare model
mkdir -p pretrained_model/navila_model
# Place NaVILA pretrained model in this directory

# 2. Run evaluation
cd habitat-baselines
python -m habitat_baselines.run \
  --config-name=social_nav_v2/navila_falcon_hm3d.yaml \
  habitat_baselines.eval_ckpt_path_dir=pretrained_model/navila_model

# 3. View results
cat output/result.json
```

For detailed instructions, see [QUICK_START.md](QUICK_START.md) ⭐

## 📊 Action Mapping

| NaVILA Output | Parsed Result | Habitat3 Action | Step Size |
|--------------|--------------|-----------------|-----------|
| "stop" | action=0 | STOP | - |
| "move forward 25 cm" | action=1, repeats=1 | MOVE_FORWARD | 25cm |
| "move forward 50 cm" | action=1, repeats=2 | MOVE_FORWARD×2 | 25cm×2 |
| "turn left 30 degree" | action=2, repeats=2 | TURN_LEFT×2 | 15°×2 |

## 🧪 Test Results

```
✅ Action Parser Tests: 10/10 passed
✅ Module Structure: All files created
✅ Configuration Files: Valid YAML
```

## 📚 Documentation

- [**QUICK_START.md**](QUICK_START.md) - Quick start guide ⭐
- [**NAVILA_INTEGRATION_README.md**](NAVILA_INTEGRATION_README.md) - Complete technical documentation
- [**INTEGRATION_SUMMARY.md**](INTEGRATION_SUMMARY.md) - Integration summary

## 🎓 Citation

```bibtex
@misc{navila_falcon_integration,
  title={NaVILA-Falcon Integration: Vision-Language Navigation in Dynamic Environments},
  year={2025}
}
```

## 📄 License

MIT License

---

<div align="center">

**🎉 集成完成 | Integration Complete**

Made with ❤️ for the Habitat & Social Navigation Community

</div>
