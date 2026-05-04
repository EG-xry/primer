# PyTorch MPS GPU Optimization Notes

## Overview

This document describes the PyTorch MPS (Apple Silicon GPU) optimization status for each stage of the algorithm in `python/run_retina_analysis.py`.

---

## Optimization Summary

| Stage | Algorithm | MPS Optimization | Status | Notes |
|-------|-----------|-----------------|--------|-------|
| Stage 1 | Data Preprocessing | ❌ Not needed | — | I/O intensive, GPU cannot accelerate |
| Stage 2 | STA | ❌ Not needed | — | Single matrix multiplication, NumPy is fast enough |
| Stage 3 | STC Permutation Test | ✅ Optimized | Implemented | **1000x speedup** (69s→0.07s/iteration) |
| Stage 4 | MNE Conjugate Gradient | ✅ Optimized | Implemented | logloss/gradient/fit all MPS GPU accelerated |
| Stage 5 | GLM | ⏭️ Skipped | — | Requires Pillow toolkit |
| Stage 6 | Prediction Validation | ❌ Not needed | — | Simple matrix multiplication, already fast enough |
| Stage 7 | Figure Generation | ❌ Not needed | — | matplotlib plotting, no computational bottleneck |

---

## Stage 3: STC Permutation Test — ✅ Optimized

### Bottleneck Analysis

The STC null distribution requires 200 permutation tests, each involving:
1. Circular shift of spike train `R_rnd = roll(R, shift)`
2. Weighted covariance computation `Cs_rnd = cov(S * sqrt(R_rnd))` — 110K×588 matrix
3. Eigenvalue decomposition `eigvalsh(Cs_rnd - Cp)` — 588×588 matrix

**Original implementation (np.cov):** ~69 seconds/iteration → 200 iterations = **~3.8 hours**

### Optimization Approach

#### Optimization 1: Algorithmic Optimization (replacing np.cov)
```python
# Original: np.cov internally recomputes mean, creates temporary arrays
C = np.cov(S_weighted_centered, rowvar=False) / R_mean

# Optimized: Direct matrix multiplication, skipping redundant computation
w2 = R_rnd  # sqrt(R)^2 = R
StW2S = (S * w2[:, None]).T @ S
C = (StW2S / (T-1) - T/(T-1) * outer(Sw_mean, Sw_mean)) / R_mean
```
**Result:** ~69s → ~0.4s/iteration (**170x speedup**)

#### Optimization 2: PyTorch MPS GPU Acceleration
```python
# Offload matrix operations to Apple Silicon GPU
S_t = torch.tensor(S, dtype=torch.float32, device="mps")
S_w = S_t * sqrt_R_rnd.unsqueeze(1)  # GPU weighting
C_rnd = (S_w_c.T @ S_w_c) / (T-1) / R_mean  # GPU matrix multiplication
evals = torch.linalg.eigvalsh(dC_rnd.cpu())  # CPU eigenvalues (MPS doesn't support eigh)
```
**Result:** ~0.4s → ~0.07s/iteration (**additional 6x speedup**)

### Final Performance
| Approach | Per-iteration Time | Total for 200 iterations | Speedup |
|----------|-------------------|--------------------------|---------|
| Original np.cov | 69s | ~3.8h | 1x |
| Optimized NumPy | 0.4s | ~83s | **170x** |
| **MPS GPU** | **0.07s** | **~13s** | **~1000x** |

### Code Location
- `_stc_null_distribution_fast()` — Automatically detects MPS, selects best path
- `_stc_null_torch_mps()` — PyTorch MPS version
- `_stc_null_numpy_fast()` — Optimized NumPy fallback version

---

## Stage 4: MNE Conjugate Gradient — ✅ Optimized

### Bottleneck Analysis

MNE fitting uses the Polak-Ribière conjugate gradient method, each iteration requires computing:
1. **logloss** — `stim @ ptemp + sum(stim * (stim @ J))` — (T×N)@(N,) + (T×N)@(N×N) matrix operations
2. **gradient** — `stim.T @ (pSpike * stim)` — (N×T)@(T×N) = N×N outer product
3. **line search** — Multiple logloss evaluations

For short3 configuration: T≈82K, N=588, number of parameters=1+588+588²=346,345

### Optimization Approach

#### Full MPS GPU Implementation
All computations (data, parameters, gradients, line search) are executed entirely on the GPU, avoiding CPU-GPU data transfers:

```python
# Preload data to GPU (once only)
stim_t = torch.tensor(stim, dtype=torch.float32, device="mps")
resp_t = torch.tensor(resp, dtype=torch.float32, device="mps")

# logloss: Use torch.nn.functional.softplus instead of np.log(1+np.exp(x))
linear = p_t[0] + stim_t @ ptemp + (stim_t * (stim_t @ J)).sum(1)
f1 = torch.nn.functional.softplus(linear)  # Numerically stable + GPU accelerated
loss = (resp_t * f1 + (1 - resp_t) * torch.nn.functional.softplus(-linear)).mean()

# gradient: Use torch.sigmoid instead of manual sigmoid
pSpike = torch.sigmoid(-linear)  # GPU sigmoid
temp = (stim_t.T @ (pSpike.unsqueeze(1) * stim_t)) / Nsamples  # GPU outer product
```

#### Key Optimization Details
1. **Constraint average** `avgsqrd = stim.T @ (resp * stim)` precomputed on GPU (one-time)
2. **softplus** replaces `log(1+exp(x))` — More numerically stable and natively supported on GPU
3. **torch.sigmoid** replaces manual sigmoid — Native GPU operation
4. **Parameters stay on GPU throughout** — Conjugate directions g/h/xi are all GPU tensors, no CPU-GPU copies
5. **Only `.cpu().numpy()` on return** — Single final copy back to CPU

### Code Location
- `_mne_fit_mps()` — MPS GPU version of the complete conjugate gradient fitting
- `_mne_logloss_mps()` — GPU logloss (using softplus)
- `_mne_gradient_mps()` — GPU gradient (using torch.sigmoid)
- `_mne_fit_numpy()` — NumPy fallback version
- `mne_fit()` — Automatically detects MPS and selects path

### Expected Speedup
| Operation | NumPy CPU | MPS GPU | Speedup |
|-----------|----------|---------|---------|
| logloss (82K×588) | ~0.5s | ~0.05s | ~10x |
| gradient (outer product 588×82K×588) | ~0.8s | ~0.08s | ~10x |
| Single iteration (2×loss + 1×grad) | ~1.8s | ~0.18s | ~10x |
| Full fit (~50 iterations) | ~90s | ~9s | ~10x |
| 20 fits (5 JK × 4 inner JK) | ~30min | ~3min | ~10x |

---

## Other Stage Analysis

### Stage 1: Data Preprocessing — No Optimization Needed
- **Bottleneck:** `read_frame_v2` reads binary file frame by frame (I/O intensive)
- **Notes:** GPU cannot accelerate disk I/O; consider using `np.fromfile` to read the entire file at once for optimization

### Stage 2: STA — No Optimization Needed
- **Computation:** `S.T @ R / sum(R)` — Single matrix-vector multiplication
- **Notes:** Already completes in ~1 second; GPU acceleration benefit is less than data transfer overhead

### Stage 6: Prediction Validation — No Optimization Needed
- **Computation:** Matrix-vector multiplication + interpolation + statistics computation
- **Notes:** Each model prediction takes only milliseconds, no bottleneck

---

## Environment Requirements

| Dependency | Version | Purpose |
|-----------|---------|---------|
| PyTorch | ≥ 2.0 | MPS GPU backend |
| macOS | ≥ 12.3 | Metal/MPS support |
| Apple Silicon | M1/M2/M3 | GPU hardware |

Installation:
```bash
pip install torch
```

Verification:
```python
import torch
print(torch.backends.mps.is_available())  # True
```

---

## Automatic Fallback Mechanism

All MPS optimizations have NumPy fallback paths:
```python
if torch.backends.mps.is_available():
    return _stc_null_torch_mps(...)  # GPU path
else:
    return _stc_null_numpy_fast(...)  # CPU optimized path
```

Even without PyTorch or MPS, the script can still run normally (using the optimized NumPy algorithm, which is 170x faster than the original version).
