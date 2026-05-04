"""
MNE — 最大噪声熵模型
====================

对应 MATLAB:
    script_105_RetinaData_MNE_fitting.m
    script_106_RetinaData_MNE_model.m
    MNEfit_RetinaData.m, logloss.m, dlogloss.m, frprmn_global_min.m

模型:
    P(spike|s) = σ(A + H^T s + s^T J s)
    其中 σ 为 sigmoid 函数
"""

import numpy as np
from scipy.optimize import minimize
from typing import Tuple


def sigmoid(x: np.ndarray) -> np.ndarray:
    """数值稳定的 sigmoid 函数。"""
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def pack_params(A: float, H: np.ndarray, J: np.ndarray) -> np.ndarray:
    """将 MNE 参数打包为一维向量。"""
    n = len(H)
    # J 为对称矩阵，只存上三角
    J_upper = J[np.triu_indices(n)]
    return np.concatenate([[A], H, J_upper])


def unpack_params(params: np.ndarray, n_features: int) -> Tuple[float, np.ndarray, np.ndarray]:
    """从一维向量解包 MNE 参数。"""
    A = params[0]
    H = params[1:1 + n_features]
    J_upper = params[1 + n_features:]
    J = np.zeros((n_features, n_features))
    J[np.triu_indices(n_features)] = J_upper
    J = J + J.T - np.diag(np.diag(J))  # 对称化
    return A, H, J


def mne_log_likelihood(
    params: np.ndarray, stimulus: np.ndarray, response: np.ndarray,
    n_features: int
) -> float:
    """
    MNE 模型的负对数似然 (对应 logloss.m)。
    
    L = -Σ [r * log(p) + (1-r) * log(1-p)]
    """
    A, H, J = unpack_params(params, n_features)
    linear = A + stimulus @ H + np.sum((stimulus @ J) * stimulus, axis=1)
    p = sigmoid(linear)
    p = np.clip(p, 1e-10, 1 - 1e-10)
    
    ll = -(response * np.log(p) + (1 - response) * np.log(1 - p)).mean()
    return ll


def mne_gradient(
    params: np.ndarray, stimulus: np.ndarray, response: np.ndarray,
    n_features: int
) -> np.ndarray:
    """MNE 模型的梯度 (对应 dlogloss.m)。"""
    A, H, J = unpack_params(params, n_features)
    linear = A + stimulus @ H + np.sum((stimulus @ J) * stimulus, axis=1)
    p = sigmoid(linear)
    
    residual = p - response  # shape (n,)
    n = len(response)
    
    dA = residual.mean()
    dH = (stimulus.T @ residual) / n
    dJ_full = (stimulus.T @ (residual[:, None] * stimulus)) / n
    dJ_full = (dJ_full + dJ_full.T) / 2  # 对称化
    dJ_upper = dJ_full[np.triu_indices(n_features)]
    
    return np.concatenate([[dA], dH, dJ_upper])


def fit_mne(
    stimulus_train: np.ndarray, response_train: np.ndarray,
    stimulus_test: np.ndarray = None, response_test: np.ndarray = None,
    order: int = 2, max_iter: int = 1000, tol: float = 1e-6,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    拟合 MNE 模型。
    
    Parameters
    ----------
    stimulus_train, response_train : 训练数据
    stimulus_test, response_test : 测试数据 (用于 early stopping)
    order : 1 仅线性, 2 包含二次项
    max_iter : 最大迭代次数
    tol : 收敛容差
    
    Returns
    -------
    A : float — 偏置
    H : np.ndarray — 线性特征向量
    J : np.ndarray — 二次特征矩阵
    """
    n_features = stimulus_train.shape[1]
    
    # 初始化
    p_spike = response_train.mean()
    A0 = np.log(p_spike / (1 - p_spike)) if p_spike > 0 and p_spike < 1 else 0.0
    H0 = np.random.randn(n_features) * 0.01
    
    if order >= 2:
        J0 = np.random.randn(n_features, n_features) * 0.001
        J0 = (J0 + J0.T) / 2
    else:
        J0 = np.zeros((n_features, n_features))
    
    params0 = pack_params(A0, H0, J0)
    
    result = minimize(
        mne_log_likelihood,
        params0,
        args=(stimulus_train, response_train, n_features),
        jac=mne_gradient,
        method="L-BFGS-B",
        options={"maxiter": max_iter, "ftol": tol, "disp": False},
    )
    
    A, H, J = unpack_params(result.x, n_features)
    return A, H, J


def mne_significant_features(
    J: np.ndarray, n_shuffles: int = 500
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    通过置换检验确定 J 矩阵的显著特征方向。
    
    对应 script_106: 重排 J 的对角/非对角元素生成零分布。
    """
    eigenvalues, eigenvectors = np.linalg.eigh(J)
    
    diag_vals = np.diag(J).copy()
    upper_idx = np.triu_indices_from(J, k=1)
    off_diag_vals = J[upper_idx].copy()
    
    null_eigenvalues = []
    for _ in range(n_shuffles):
        J_null = np.zeros_like(J)
        np.fill_diagonal(J_null, np.random.permutation(diag_vals))
        shuffled_off = np.random.permutation(off_diag_vals)
        J_null[upper_idx] = shuffled_off
        J_null = (J_null + J_null.T) / 2
        null_eigenvalues.append(np.linalg.eigvalsh(J_null))
    
    null_eigenvalues = np.array(null_eigenvalues)
    upper_bound = null_eigenvalues.max(axis=0)
    lower_bound = null_eigenvalues.min(axis=0)
    
    idx = np.argsort(eigenvalues)
    eigenvalues_sorted = eigenvalues[idx]
    eigenvectors_sorted = eigenvectors[:, idx]
    upper_sorted = np.sort(upper_bound)
    lower_sorted = np.sort(lower_bound)
    
    sig_mask = (eigenvalues_sorted > upper_sorted) | (eigenvalues_sorted < lower_sorted)
    
    return eigenvalues_sorted, eigenvectors_sorted, sig_mask
