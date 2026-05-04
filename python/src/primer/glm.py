"""
GLM — 广义线性模型
==================

对应 MATLAB:
    script_107_RetinaData_GLM.m
    makeFittingStruct_GLM_Retina.m

模型:
    λ(t) = exp(k^T s(t) + h^T r_hist(t) + b)
    
依赖: statsmodels, pyglmnet, scikit-learn
"""

import numpy as np
from scipy.linalg import svd
from typing import Tuple, Optional


def make_raised_cosine_basis(
    n_basis: int, n_bins: int, peak_range: Tuple[float, float] = (0, None)
) -> np.ndarray:
    """
    构建升余弦时间基函数。
    
    对应 MATLAB: makeBasis_StimKernel / makeBasis_PostSpike
    """
    if peak_range[1] is None:
        peak_range = (peak_range[0], n_bins)
    
    peaks = np.linspace(peak_range[0], peak_range[1], n_basis)
    width = (peak_range[1] - peak_range[0]) / (n_basis - 1) if n_basis > 1 else n_bins
    
    t = np.arange(n_bins)
    basis = np.zeros((n_bins, n_basis))
    for i, peak in enumerate(peaks):
        basis[:, i] = np.maximum(0, np.cos((t - peak) * np.pi / width / 2)) ** 2
    
    return basis


def build_history_matrix(
    response: np.ndarray, n_history: int
) -> np.ndarray:
    """构建脉冲历史设计矩阵。"""
    n = len(response)
    hist_matrix = np.zeros((n, n_history))
    for lag in range(1, n_history + 1):
        hist_matrix[lag:, lag - 1] = response[:-lag]
    return hist_matrix


def initialize_stimulus_filter(
    sta: np.ndarray, spatial_dims: Tuple[int, ...], n_temporal: int,
    rank: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用 STA 的 SVD 分解初始化时空分离的刺激滤波器。
    
    对应 MATLAB: makeFittingStruct_GLM_Retina.m 中的 SVD 初始化
    """
    n_spatial = int(np.prod(spatial_dims))
    sta_matrix = sta.reshape(n_temporal, n_spatial)
    U, s, Vt = svd(sta_matrix, full_matrices=False)
    
    k_temporal = U[:, :rank] * s[:rank]
    k_spatial = Vt[:rank, :].T
    
    return k_temporal, k_spatial


def fit_glm_poisson(
    stimulus: np.ndarray, response: np.ndarray,
    history_matrix: Optional[np.ndarray] = None,
    alpha: float = 0.0, l1_ratio: float = 0.0,
) -> dict:
    """
    拟合 Poisson GLM。
    
    Parameters
    ----------
    stimulus : 设计矩阵
    response : 脉冲响应
    history_matrix : 脉冲历史矩阵 (可选)
    alpha : 正则化强度
    l1_ratio : L1 正则化比例 (0=L2, 1=L1)
    
    Returns
    -------
    result : dict 包含 k_stim, h_hist, baseline, model
    """
    import statsmodels.api as sm
    
    if history_matrix is not None:
        X = np.hstack([stimulus, history_matrix])
    else:
        X = stimulus
    X = sm.add_constant(X)
    
    if alpha > 0:
        model = sm.GLM(response, X, family=sm.families.Poisson())
        result = model.fit_regularized(alpha=alpha, L1_wt=l1_ratio)
    else:
        model = sm.GLM(response, X, family=sm.families.Poisson())
        result = model.fit()
    
    params = result.params
    baseline = params[0]
    k_stim = params[1:stimulus.shape[1] + 1]
    h_hist = params[stimulus.shape[1] + 1:] if history_matrix is not None else None
    
    return {
        "k_stim": k_stim,
        "h_hist": h_hist,
        "baseline": baseline,
        "model": result,
    }


def simulate_glm(
    stimulus: np.ndarray, k_stim: np.ndarray,
    h_hist: Optional[np.ndarray] = None,
    baseline: float = 0.0, dt: float = 1.0,
    n_repeats: int = 1
) -> np.ndarray:
    """
    模拟 GLM 产生的脉冲序列。
    
    Returns
    -------
    rate : np.ndarray, shape (n_frames,)
        平均发放率 (跨重复试次)
    """
    n_frames = stimulus.shape[0]
    n_hist = len(h_hist) if h_hist is not None else 0
    
    all_spikes = np.zeros((n_repeats, n_frames))
    
    for rep in range(n_repeats):
        spikes = np.zeros(n_frames)
        for t in range(n_frames):
            linear = baseline + stimulus[t] @ k_stim
            if h_hist is not None and t >= n_hist:
                linear += spikes[t - n_hist:t][::-1] @ h_hist
            rate = np.exp(linear) * dt
            spikes[t] = np.random.poisson(rate)
        all_spikes[rep] = spikes
    
    return all_spikes.mean(axis=0)
