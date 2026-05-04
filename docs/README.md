# Primer — 神经元脉冲序列分析入门教程

## 项目简介

本项目是一个基于 MATLAB 的神经元脉冲序列（spike train）分析教程代码库，配套相关学术论文使用。项目通过三个真实的神经科学数据集，系统性地演示了从原始数据到模型拟合、预测验证的完整分析流程，涵盖多种经典的神经编码分析方法。

## 数据集

项目包含三个不同神经系统层级的实验数据集：

| 数据集 | 编号系列 | 神经系统区域 | 数据目录 | 说明 |
|--------|---------|------------|---------|------|
| **视网膜 (Retina)** | 100 系列 | 视网膜神经节细胞 | `RetinaData/` | 白噪声棋盘格刺激及 53 个细胞的脉冲序列 |
| **触须/丘脑 (Whisker/Thalamus)** | 200 系列 | 大鼠丘脑 VPM 区 | `WhiskerData/` | 7 个丘脑细胞的触须位置与脉冲序列记录 |
| **运动皮层 (Motor Cortex)** | 300 系列 | 运动皮层 | `MotorData/` | 运动皮层神经元集群活动数据 |

## 分析方法

项目涵盖以下神经编码分析方法：

- **STA (Spike-Triggered Average)** — 脉冲触发平均，估计神经元的线性感受野
- **STC (Spike-Triggered Covariance)** — 脉冲触发协方差，捕获二阶特征
- **MNE (Maximally Noise Entropy)** — 最大噪声熵模型，非线性特征提取
- **GLM (Generalized Linear Model)** — 广义线性模型，包括：
  - 非耦合 GLM (Uncoupled GLM)
  - 耦合 GLM (Coupled GLM)，用于建模神经元网络交互
- **预测与验证 (Prediction & Validation)** — 模型预测性能评估与交叉验证

## 项目结构

```
primer-master/
├── RetinaData/          # 视网膜实验原始数据
├── RetinaScripts/       # 视网膜数据分析脚本 (100 系列)
│   └── RetinaMNEGLM/    # MNE/GLM 辅助函数
├── WhiskerData/         # 触须实验原始数据
├── WhiskerScripts/      # 触须数据分析脚本 (200 系列)
│   └── WhiskerMNEGLM/   # MNE/GLM 辅助函数
├── MotorData/           # 运动皮层实验原始数据
├── MotorScripts/        # 运动皮层数据分析脚本 (300 系列)
│   ├── MotorGLM/        # GLM 辅助函数
│   └── L1Group/         # L1 正则化优化工具
├── OtherScripts/        # 其他通用辅助函数
├── docs/                # 文档目录
└── readme.txt           # 原始说明文件
```

## 脚本组织

每个数据集的脚本分为两类：

- **`script_*`** — 数据处理与分析脚本：执行原始数据的预处理、模型拟合等计算，并保存结果
- **`makefig_*`** — 绘图脚本：加载已保存的分析结果并生成论文中的图表

> ⚠️ 使用顺序：必须先运行对应的 `script_*` 完成分析，再运行 `makefig_*` 生成图表。部分绘图脚本需要多个分析脚本的结果。

## 依赖与安装

1. 解压项目文件至目标目录
2. 将数据文件放入对应的数据目录（`RetinaData/`、`WhiskerData/`、`MotorData/`）
3. 将辅助函数目录添加到 MATLAB 搜索路径（包括 `MotorGLM/`、`L1Group/`、`OtherScripts/` 等）
4. 编译 L1Group 的 MEX 文件：运行 `MotorScripts/L1Group/mexAll.m`
5. 安装 [Jonathan Pillow 的 GLM 工具包](https://github.com/pillowlab/GLMspiketools)
6. 安装 [Mark Schmidt 的 L1 优化工具](https://www.cs.ubc.ca/~schmidtm/Software/minFunc.html)

> 💡 建议使用 MATLAB 的 **Cell Mode** 或 **Debug Mode** 逐步执行脚本，以便对照论文中的公式理解每一步的数据变换过程。

## 交叉验证脚本

300 系列中带有 `_crossval` 后缀的脚本用于五折交叉验证，用于选择耦合模型的最优正则化参数并估计标准误差。其中 `script_303_MotorData_SimCoupledGLM_crossval.m` 计算量非常大，需用户根据可用计算资源自行修改以在多 CPU/多机器上运行。这些脚本作为模板提供，不保证开箱即用。

## 许可与引用

本项目为学术论文的配套代码，使用时请引用相关论文。触须数据的详细说明请参见：

> Moore JD, Mercer Lindsay N, Deschênes M, Kleinfeld D (2015) Vibrissa Self-Motion and Touch Are Reliably Encoded along the Same Somatosensory Pathway from Brainstem through Thalamus. *PLoS Biol* 13(9): e1002253.
