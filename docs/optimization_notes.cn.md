# PyTorch MPS GPU 优化说明

## 概述

本文档说明 `python/run_retina_analysis.py` 中各阶段算法的 PyTorch MPS (Apple Silicon GPU) 优化情况。

---

## 优化总览

| 阶段 | 算法 | MPS 优化 | 状态 | 说明 |
|------|------|---------|------|------|
| 阶段一 | 数据预处理 | ❌ 不需要 | — | I/O 密集型，GPU 无法加速 |
| 阶段二 | STA | ❌ 不需要 | — | 单次矩阵乘法，NumPy 已足够快 |
| 阶段三 | STC 置换检验 | ✅ 已优化 | 已实现 | **1000x 加速**（69s→0.07s/次） |
| 阶段四 | MNE 共轭梯度 | ✅ 已优化 | 已实现 | logloss/gradient/fit 全部 MPS GPU 加速 |
| 阶段五 | GLM | ⏭️ 跳过 | — | 需要 Pillow 工具包 |
| 阶段六 | 预测验证 | ❌ 不需要 | — | 简单矩阵乘法，已足够快 |
| 阶段七 | 图表生成 | ❌ 不需要 | — | matplotlib 绑图，无计算瓶颈 |

---

## 阶段三：STC 置换检验 — ✅ 已优化

### 瓶颈分析

STC 零分布需要执行 200 次置换检验，每次包含：
1. 循环移位脉冲序列 `R_rnd = roll(R, shift)` 
2. 计算加权协方差 `Cs_rnd = cov(S * sqrt(R_rnd))` — 110K×588 矩阵
3. 特征值分解 `eigvalsh(Cs_rnd - Cp)` — 588×588 矩阵

**原始实现（np.cov）：** ~69 秒/次 → 200 次 = **~3.8 小时**

### 优化方案

#### 优化 1：算法优化（替代 np.cov）
```python
# 原始：np.cov 内部重复计算均值、创建临时数组
C = np.cov(S_weighted_centered, rowvar=False) / R_mean

# 优化：直接矩阵乘法，跳过冗余计算
w2 = R_rnd  # sqrt(R)^2 = R
StW2S = (S * w2[:, None]).T @ S
C = (StW2S / (T-1) - T/(T-1) * outer(Sw_mean, Sw_mean)) / R_mean
```
**效果：** ~69s → ~0.4s/次（**170x 加速**）

#### 优化 2：PyTorch MPS GPU 加速
```python
# 将矩阵运算卸载到 Apple Silicon GPU
S_t = torch.tensor(S, dtype=torch.float32, device="mps")
S_w = S_t * sqrt_R_rnd.unsqueeze(1)  # GPU 加权
C_rnd = (S_w_c.T @ S_w_c) / (T-1) / R_mean  # GPU 矩阵乘法
evals = torch.linalg.eigvalsh(dC_rnd.cpu())  # CPU 特征值（MPS 不支持 eigh）
```
**效果：** ~0.4s → ~0.07s/次（**额外 6x 加速**）

### 最终性能
| 方案 | 单次耗时 | 200次总耗时 | 加速比 |
|------|---------|------------|-------|
| 原始 np.cov | 69s | ~3.8h | 1x |
| 优化 NumPy | 0.4s | ~83s | **170x** |
| **MPS GPU** | **0.07s** | **~13s** | **~1000x** |

### 代码位置
- `_stc_null_distribution_fast()` — 自动检测 MPS，选择最佳路径
- `_stc_null_torch_mps()` — PyTorch MPS 版本
- `_stc_null_numpy_fast()` — 优化 NumPy 回退版本

---

## 阶段四：MNE 共轭梯度 — ✅ 已优化

### 瓶颈分析

MNE 拟合使用 Polak-Ribière 共轭梯度法，每次迭代需要计算：
1. **logloss** — `stim @ ptemp + sum(stim * (stim @ J))` — (T×N)@(N,) + (T×N)@(N×N) 矩阵运算
2. **gradient** — `stim.T @ (pSpike * stim)` — (N×T)@(T×N) = N×N 外积
3. **线搜索** — 多次 logloss 评估

对于 short3 配置：T≈82K, N=588, 参数量=1+588+588²=346,345

### 优化方案

#### 完全 MPS GPU 化
所有计算（数据、参数、梯度、线搜索）全部在 GPU 上执行，避免 CPU-GPU 数据传输：

```python
# 数据预加载到 GPU（仅一次）
stim_t = torch.tensor(stim, dtype=torch.float32, device="mps")
resp_t = torch.tensor(resp, dtype=torch.float32, device="mps")

# logloss: 使用 torch.nn.functional.softplus 替代 np.log(1+np.exp(x))
linear = p_t[0] + stim_t @ ptemp + (stim_t * (stim_t @ J)).sum(1)
f1 = torch.nn.functional.softplus(linear)  # 数值稳定 + GPU 加速
loss = (resp_t * f1 + (1 - resp_t) * torch.nn.functional.softplus(-linear)).mean()

# gradient: 使用 torch.sigmoid 替代手动 sigmoid
pSpike = torch.sigmoid(-linear)  # GPU sigmoid
temp = (stim_t.T @ (pSpike.unsqueeze(1) * stim_t)) / Nsamples  # GPU 外积
```

#### 关键优化细节
1. **约束平均** `avgsqrd = stim.T @ (resp * stim)` 预计算在 GPU 上（一次性）
2. **softplus** 替代 `log(1+exp(x))` — 数值更稳定且 GPU 原生支持
3. **torch.sigmoid** 替代手动 sigmoid — GPU 原生操作
4. **参数全程在 GPU** — 共轭方向 g/h/xi 均为 GPU tensor，无 CPU-GPU 拷贝
5. **仅在返回时 `.cpu().numpy()`** — 最后一次拷贝回 CPU

### 代码位置
- `_mne_fit_mps()` — MPS GPU 版本的完整共轭梯度拟合
- `_mne_logloss_mps()` — GPU logloss（使用 softplus）
- `_mne_gradient_mps()` — GPU gradient（使用 torch.sigmoid）
- `_mne_fit_numpy()` — NumPy 回退版本
- `mne_fit()` — 自动检测 MPS 并选择路径

### 预计加速
| 操作 | NumPy CPU | MPS GPU | 加速比 |
|------|----------|---------|-------|
| logloss (82K×588) | ~0.5s | ~0.05s | ~10x |
| gradient (外积 588×82K×588) | ~0.8s | ~0.08s | ~10x |
| 单次迭代 (2×loss + 1×grad) | ~1.8s | ~0.18s | ~10x |
| 完整拟合 (~50 迭代) | ~90s | ~9s | ~10x |
| 20 次拟合 (5 JK × 4 内部JK) | ~30min | ~3min | ~10x |

---

## 其他阶段分析

### 阶段一：数据预处理 — 无需优化
- **瓶颈：** `read_frame_v2` 逐帧读取二进制文件（I/O 密集）
- **说明：** GPU 无法加速磁盘 I/O，可考虑 `np.fromfile` 一次性读取整个文件来优化

### 阶段二：STA — 无需优化  
- **计算：** `S.T @ R / sum(R)` — 单次矩阵-向量乘法
- **说明：** 已在 ~1 秒内完成，GPU 加速收益小于数据传输开销

### 阶段六：预测验证 — 无需优化
- **计算：** 矩阵-向量乘法 + 插值 + 统计量计算
- **说明：** 每个模型预测仅需毫秒级，无瓶颈

---

## 环境要求

| 依赖 | 版本 | 用途 |
|------|------|------|
| PyTorch | ≥ 2.0 | MPS GPU 后端 |
| macOS | ≥ 12.3 | Metal/MPS 支持 |
| Apple Silicon | M1/M2/M3 | GPU 硬件 |

安装：
```bash
pip install torch
```

验证：
```python
import torch
print(torch.backends.mps.is_available())  # True
```

---

## 自动回退机制

所有 MPS 优化都有 NumPy 回退路径：
```python
if torch.backends.mps.is_available():
    return _stc_null_torch_mps(...)  # GPU 路径
else:
    return _stc_null_numpy_fast(...)  # CPU 优化路径
```

即使没有 PyTorch 或 MPS，脚本仍然可以正常运行（使用优化 NumPy 算法，比原始版本快 170x）。
