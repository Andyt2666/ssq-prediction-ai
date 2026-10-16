# SSQ-Prediction-AI: 基于深度学习与统计分析的双色球智能预测与缩水系统

本项目是一个趣味性与硬核技术兼备的开源项目：探索如何运用**现代深度学习模型（TCN、Transformer/SSQ-GPT、CVAE）**结合**彩票统计物理学（遗漏值、和值分布、奇偶比、AC值、连号特征）**对双色球历史走势进行特征提取、规律分析、模拟回测与智能科学选号缩水。

> 💡 **理性声明**：彩票在数学上为独立同分布的随机事件，头奖概率为 1772 万分之一。本项目旨在进行高维序列与组合建模的学术工程探索，倡导**理性娱乐，量力而行**。

---

## 📁 目录结构

```text
ssq-prediction-ai/
├── docs/                      # 调研与技术方案
│   └── RESEARCH.md            # [核心调研] 深度学习双色球建模全面调研与系统设计
├── data/                      # 数据缓存目录 (被 .gitignore 忽略)
│   ├── raw/                   # 2003年至今全量历史开奖数据
│   └── processed/             # 处理后的特征矩阵与时序张量
├── src/                       # 核心源码
│   ├── data/                  # 历史开奖数据爬取与增量更新
│   ├── features/              # 彩票特征工程（遗漏、和值、AC值、奇偶比）
│   ├── models/                # 深度学习模型 (TCN双塔, SSQ-GPT, CVAE)
│   ├── strategy/              # 智能缩水与选号过滤器 (和值过滤、杀号、定胆)
│   └── utils/                 # 可视化与格式化输出工具
├── notebooks/                 # 走势图与统计学 EDA 分析
├── requirements.txt           # 项目 Python 依赖
├── .gitignore                 # Git 忽略配置
└── README.md                  # 项目总览
```

---

## 🚀 快速上手

### 1. 安装依赖
```bash
# 建议在原生 Linux (WSL2) 环境下运行
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 详细调研与技术架构
查看深度学习建模思路、数学理论与特征体系：
👉 [docs/RESEARCH.md](file:///home/aaa/projects/ssq-prediction-ai/docs/RESEARCH.md)
