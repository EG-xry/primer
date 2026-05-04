"""
STA — 脉冲触发平均
==================

对应 MATLAB: script_102_RetinaData_STA.m

STA = (1/n_spikes) * Σ s(t)  对所有 spike 时刻 t
"""

import numpy as np


def compute_sta(stimulus: np.ndarray, response: np.ndarray) -> np.ndarray:
    """
    计算脉冲触发平均 (Spike-Triggered Average)。
    
    Parameters
    ----------
    stimulus : np.ndarray, shape (n_frames, n_features)
        设计矩阵 (已归一化)
    response : np.ndarray, shape (n_frames,)
        脉冲响应向量 (0/1 或脉冲计数)
        
    Returns
    -------
    sta : np.ndarray, shape (n_features,)
        脉冲触发平均向量
    """
    n_spikes = response.sum()
    if n_spikes == 0:
        return np.zeros(stimulus.shape[1])
    sta = (response @ stimulus) / n_spikes
    return sta


def sta_significance(
    sta: np.ndarray, stimulus: np.ndarray, response: np.ndarray, n_shuffles: int = 500
) -> float:
    """
    通过置换检验评估 STA 的统计显著性。
    
    Returns
    -------
    p_value : float
    """
    observed_norm = np.linalg.norm(sta)
    count = 0
    for _ in range(n_shuffles):
        shuffled_response = np.random.permutation(response)
        shuffled_sta = compute_sta(stimulus, shuffled_response)
        if np.linalg.norm(shuffled_sta) >= observed_norm:
            count += 1
    return count / n_shuffles
