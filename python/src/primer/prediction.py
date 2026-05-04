"""
模型预测模块
============

对应 MATLAB: script_108_RetinaData_Predictions.m

为各模型 (STA-LN, STC-LN, MNE, GLM) 在测试集上生成预测发放率。
"""

import numpy as np
from typing import Tuple, Optional


def predict_sta_ln(
    stimulus: np.ndarray, sta: np.ndarray,
    bin_centers: np.ndarray, nonlinearity: np.ndarray
) -> np.ndarray:
    """
    STA-LN 模型预测。
    
    投影 → 查表非线性
    """
    projections = stimulus @ sta
    return np.interp(projections, bin_centers, nonlinearity)


def predict_stc_ln(
    stimulus: np.ndarray, feature_vectors: np.ndarray,
    bin_centers_list: list, nonlinearity_list: list
) -> np.ndarray:
    """
    STC-LN 模型预测。
    
    多特征方向投影 → 各自非线性 → 求和
    """
    prediction = np.zeros(stimulus.shape[0])
    for i in range(feature_vectors.shape[1]):
        proj = stimulus @ feature_vectors[:, i]
        pred_i = np.interp(proj, bin_centers_list[i], nonlinearity_list[i])
        prediction += pred_i
    return prediction


def predict_mne(
    stimulus: np.ndarray, A: float, H: np.ndarray, J: np.ndarray
) -> np.ndarray:
    """
    MNE 模型预测。
    
    P(spike|s) = σ(A + H^T s + s^T J s)
    """
    linear = A + stimulus @ H + np.sum((stimulus @ J) * stimulus, axis=1)
    return 1 / (1 + np.exp(-linear))


def predict_glm(
    stimulus: np.ndarray, k_stim: np.ndarray,
    h_hist: Optional[np.ndarray] = None,
    baseline: float = 0.0
) -> np.ndarray:
    """
    GLM 条件强度函数预测 (不含脉冲历史反馈)。
    
    λ(t) = exp(k^T s(t) + b)
    """
    linear = baseline + stimulus @ k_stim
    return np.exp(linear)
