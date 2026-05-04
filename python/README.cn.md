# Spike Train Primer — Python 复现环境

## 简介

本目录为原 MATLAB 项目（Analysis of Neuronal Spike Trains, Deconstructed）的 Python 复现开发环境。

## 快速开始

### 1. 创建虚拟环境

```bash
cd python
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
```

### 2. 安装依赖

```bash
# 方式一：使用 pip + requirements.txt
pip install -r requirements.txt

# 方式二：使用 pip + pyproject.toml (可编辑安装)
pip install -e ".[dev]"
```

### 3. 启动 Jupyter Lab

```bash
jupyter lab
```

## 目录结构

```
python/
├── pyproject.toml          # 项目配置与依赖定义
├── requirements.txt        # pip 依赖列表
├── README.md               # 本文件
├── src/
│   └── primer/
│       ├── __init__.py     # 包入口
│       ├── io.py           # 数据 I/O (读取 .raw, .isk, .mat)
│       ├── preprocessing.py # 数据预处理 (刺激构建、归一化、对齐)
│       ├── sta.py          # STA 脉冲触发平均
│       ├── stc.py          # STC 脉冲触发协方差
│       ├── mne_model.py    # MNE 最大噪声熵模型
│       ├── glm.py          # GLM 广义线性模型
│       ├── prediction.py   # 模型预测
│       ├── validation.py   # 模型验证与评估
│       └── plotting.py     # 可视化工具
└── notebooks/
    ├── 01_build_stimulus.ipynb
    ├── 02_sta.ipynb
    ├── 03_stc.ipynb
    ├── 04_mne.ipynb
    ├── 05_glm.ipynb
    └── 06_prediction_validation.ipynb
```

## 库依赖说明

| 库 | 用途 | 对应 MATLAB 功能 |
|---|------|-----------------|
| `numpy` | 矩阵运算、线性代数 | MATLAB 基础运算 |
| `scipy` | 信号处理、优化、`loadmat` 读取 .mat | MATLAB 内置函数 |
| `h5py` | 读取 v7.3 .mat 文件 | MATLAB load |
| `matplotlib` | 绑图 | MATLAB plot/figure |
| `scikit-learn` | PCA、交叉验证 | MATLAB 手写实现 |
| `statsmodels` | GLM 拟合 | Pillow GLM 工具包 |
| `pyglmnet` | 正则化 GLM | L1Group 优化 |
| `elephant` | STA、脉冲分析 | 手写 STA/STC |
| `spectrum` | 频谱/相干性 | Chronux 工具包 |
| `cvxpy` | 凸优化 (L1) | Mark Schmidt L1 |

## 可选 GPU 加速

```bash
pip install -e ".[gpu]"
```

安装 PyTorch 和 JAX 用于大规模模型拟合加速。
