"""
STC — 脉冲触发协方差
====================

对应 MATLAB:
    script_103_RetinaData_STC_significance.m
    script_104_RetinaData_STC_model.m

C_STC = (1/n_spikes) * Σ (s(t)-STA)(s(t)-STA)^T - C_prior
"""

import numpy as np
from typing import Tuple


def compute_stc(
    stimulus: np.ndarray, response: np.ndarray, sta: np.ndarray
) -> np.ndarray:
    """
    计算脉冲触发协方差矩阵。
    
    Parameters
    ----------
    stimulus : np.ndarray, shape (n_frames, n_features)
    response : np.ndarray, shape (n_frames,)
    sta : np.ndarray, shape (n_features,)
    
    Returns
    -------
    stc : np.ndarray, shape (n_features, n_features)
    """
    spike_idx = np.where(response > 0)[0]
    n_spikes = len(spike_idx)
    
    spike_stim = stimulus[spike_idx] - sta
    stc_spike = (spike_stim.T @ spike_stim) / n_spikes
    
    # 先验协方差
    stim_centered = stimulus - stimulus.mean(axis=0)
    c_prior = (stim_centered.T @ stim_centered) / len(stimulus)
    
    return stc_spike - c_prior


def stc_eigendecomposition(
    stc: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """对 STC 矩阵进行特征值分解。"""
    eigenvalues, eigenvectors = np.linalg.eigh(stc)
    # 按特征值绝对值降序排列
    idx = np.argsort(np.abs(eigenvalues))[::-1]
    return eigenvalues[idx], eigenvectors[:, idx]


def marchenko_pastur_bounds(
    n_samples: int, n_features: int, variance: float = 1.0
) -> Tuple[float, float]:
    """
    计算 Marchenko-Pastur 分布的上下界。
    
    用作零假设: 超出该范围的 STC 特征值为统计显著。
    """
    gamma = n_features / n_samples
    lambda_plus = variance * (1 + np.sqrt(gamma)) ** 2
    lambda_minus = variance * (1 - np.sqrt(gamma)) ** 2
    return lambda_minus, lambda_plus


def find_significant_features(
    eigenvalues: np.ndarray, eigenvectors: np.ndarray,
    lambda_minus: float, lambda_plus: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    找出超出 Marchenko-Pastur 边界的显著特征方向。
    
    Returns
    -------
    sig_values : 显著特征值
    sig_vectors : 显著特征方向
    sig_idx : 显著特征索引
    """
    sig_idx = np.where(
        (eigenvalues > lambda_plus) | (eigenvalues < lambda_minus)
    )[0]
    return eigenvalues[sig_idx], eigenvectors[:, sig_idx], sig_idx


def estimate_nonlinearity(
    stimulus: np.ndarray, response: np.ndarray,
    feature_vector: np.ndarray, n_bins: int = 50
) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用 Bayes 定理估计沿特征方向的非线性函数。
    
    P(spike|projection) = P(projection|spike) * P(spike) / P(projection)
    """
    projections = stimulus @ feature_vector
    
    bins = np.linspace(projections.min(), projections.max(), n_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    
    # P(projection)
    p_proj, _ = np.histogram(projections, bins=bins, density=True)
    
    # P(projection | spike)
    spike_idx = response > 0
    p_proj_spike, _ = np.histogram(projections[spike_idx], bins=bins, density=True)
    
    # P(spike | projection)
    p_spike = response.mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        nonlinearity = np.where(p_proj > 0, p_proj_spike * p_spike / p_proj, 0)
    
    return bin_centers, nonlinearity
