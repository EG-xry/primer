"""
模型验证与评估模块
==================

对应 MATLAB: script_109_RetinaData_Validation.m

评价指标:
    - Pearson 相关系数
    - 对数似然 (log-likelihood)
    - 信息量 (bits/spike)
    - 均方误差 (MSE)
"""

import numpy as np
from typing import Tuple


def pearson_correlation(predicted: np.ndarray, actual: np.ndarray) -> float:
    """预测发放率与实际 PSTH 之间的 Pearson 相关系数。"""
    if predicted.std() == 0 or actual.std() == 0:
        return 0.0
    return np.corrcoef(predicted, actual)[0, 1]


def log_likelihood_bernoulli(
    predicted_prob: np.ndarray, actual: np.ndarray
) -> float:
    """
    Bernoulli 模型的对数似然。
    
    LL = Σ [r * log(p) + (1-r) * log(1-p)]
    """
    p = np.clip(predicted_prob, 1e-10, 1 - 1e-10)
    return (actual * np.log(p) + (1 - actual) * np.log(1 - p)).sum()


def log_likelihood_poisson(
    predicted_rate: np.ndarray, actual: np.ndarray, dt: float = 1.0
) -> float:
    """
    Poisson 模型的对数似然。
    
    LL = Σ [r * log(λ*dt) - λ*dt - log(r!)]
    """
    lam = np.clip(predicted_rate * dt, 1e-10, None)
    from scipy.special import gammaln
    return (actual * np.log(lam) - lam - gammaln(actual + 1)).sum()


def bits_per_spike(
    predicted_rate: np.ndarray, actual: np.ndarray, dt: float = 1.0
) -> float:
    """
    信息量: bits/spike。
    
    I = (1/n_spikes) * Σ r(t) * log2(λ(t) / <λ>)
    """
    mean_rate = predicted_rate.mean()
    if mean_rate <= 0:
        return 0.0
    n_spikes = actual.sum()
    if n_spikes == 0:
        return 0.0
    
    rate_ratio = np.clip(predicted_rate / mean_rate, 1e-10, None)
    return (actual * np.log2(rate_ratio)).sum() / n_spikes


def mse(predicted: np.ndarray, actual: np.ndarray) -> float:
    """均方误差。"""
    return np.mean((predicted - actual) ** 2)


def compare_models(
    predictions: dict, actual: np.ndarray, metric: str = "correlation"
) -> dict:
    """
    比较多个模型的预测性能。
    
    Parameters
    ----------
    predictions : dict
        {model_name: predicted_array}
    actual : np.ndarray
        实际响应
    metric : str
        "correlation", "log_likelihood", "bits_per_spike", "mse"
    
    Returns
    -------
    scores : dict
        {model_name: score}
    """
    metric_funcs = {
        "correlation": pearson_correlation,
        "log_likelihood": log_likelihood_bernoulli,
        "bits_per_spike": bits_per_spike,
        "mse": mse,
    }
    func = metric_funcs[metric]
    return {name: func(pred, actual) for name, pred in predictions.items()}
