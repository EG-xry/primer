"""
可视化工具模块
==============

对应 MATLAB: makefig_101 ~ makefig_106

提供复现论文图表所需的绑图函数。
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Tuple, Optional, List


def setup_plot_style():
    """设置论文级绑图样式。"""
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
    })


def plot_stimulus_frames(
    stimulus: np.ndarray, frame_indices: list,
    spatial_shape: Tuple[int, int], ax: Optional[plt.Axes] = None
):
    """绘制刺激帧序列 (对应 makefig_101 Panel A)。"""
    n_frames = len(frame_indices)
    if ax is None:
        fig, axes = plt.subplots(1, n_frames, figsize=(2 * n_frames, 2))
    else:
        axes = [ax]
    
    for i, idx in enumerate(frame_indices):
        frame = stimulus[idx].reshape(spatial_shape)
        axes[i].imshow(frame, cmap="gray", interpolation="nearest")
        axes[i].set_title(f"t={idx}")
        axes[i].axis("off")


def plot_eigenvalue_spectrum(
    eigenvalues: np.ndarray,
    mp_bounds: Optional[Tuple[float, float]] = None,
    ax: Optional[plt.Axes] = None, title: str = "Eigenvalue Spectrum"
):
    """绘制特征值谱 (对应 makefig_101 Panel B, makefig_103)。"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    
    ax.plot(np.arange(len(eigenvalues)), np.sort(eigenvalues)[::-1], "k.-")
    if mp_bounds is not None:
        ax.axhline(mp_bounds[0], color="r", linestyle="--", label="MP bounds")
        ax.axhline(mp_bounds[1], color="r", linestyle="--")
    ax.set_xlabel("Component index")
    ax.set_ylabel("Eigenvalue")
    ax.set_title(title)
    ax.legend()


def plot_spatial_filter(
    filter_vector: np.ndarray, spatial_shape: Tuple[int, int],
    n_temporal: int = 1, ax: Optional[plt.Axes] = None,
    title: str = "Spatial Filter"
):
    """绘制空间-时间滤波器 (对应 makefig_102)。"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 2))
    
    if n_temporal > 1:
        n_spatial = int(np.prod(spatial_shape))
        img = filter_vector.reshape(n_temporal, *spatial_shape)
        combined = np.hstack([img[t] for t in range(n_temporal)])
        ax.imshow(combined, cmap="RdBu_r", interpolation="nearest")
    else:
        img = filter_vector.reshape(spatial_shape)
        ax.imshow(img, cmap="RdBu_r", interpolation="nearest")
    
    ax.set_title(title)
    ax.axis("off")


def plot_nonlinearity(
    bin_centers: np.ndarray, nonlinearity: np.ndarray,
    ax: Optional[plt.Axes] = None, label: str = "",
    title: str = "Nonlinearity"
):
    """绘制非线性函数 (对应 makefig_102 Panel D)。"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(bin_centers, nonlinearity, label=label)
    ax.set_xlabel("Projection")
    ax.set_ylabel("P(spike)")
    ax.set_title(title)
    if label:
        ax.legend()


def plot_psth_comparison(
    actual: np.ndarray, predictions: dict,
    time_range: Optional[Tuple[int, int]] = None,
    ax: Optional[plt.Axes] = None
):
    """
    绘制实际 PSTH 与各模型预测的对比 (对应 makefig_106)。
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    
    if time_range is not None:
        sl = slice(*time_range)
        actual = actual[sl]
        predictions = {k: v[sl] for k, v in predictions.items()}
    
    t = np.arange(len(actual))
    ax.plot(t, actual, "k", linewidth=1.5, label="Actual", alpha=0.7)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for (name, pred), color in zip(predictions.items(), colors):
        ax.plot(t, pred, color=color, linewidth=1, label=name, alpha=0.8)
    
    ax.set_xlabel("Time (frames)")
    ax.set_ylabel("Firing rate")
    ax.set_title("PSTH: Actual vs Predicted")
    ax.legend()


def plot_model_comparison_bar(
    scores: dict, metric_name: str = "Correlation",
    ax: Optional[plt.Axes] = None
):
    """绘制模型性能对比柱状图 (对应 makefig_106)。"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    
    names = list(scores.keys())
    values = list(scores.values())
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"][:len(names)]
    
    ax.bar(names, values, color=colors)
    ax.set_ylabel(metric_name)
    ax.set_title(f"Model Comparison: {metric_name}")
