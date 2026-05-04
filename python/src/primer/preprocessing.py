"""
数据预处理模块
=============

对应 MATLAB: script_101_RetinaData_BuildStimulus.m

功能:
    - 从原始刺激中提取空间感受野区域
    - 构建时空联合刺激矩阵
    - 零均值化与归一化
    - 刺激-响应时间对齐
    - 训练/测试集划分
    - Jackknife 分组
"""

import numpy as np
from typing import Tuple


def extract_spatial_patch(
    stimulus: np.ndarray, x0: int, y0: int, patch_size: int
) -> np.ndarray:
    """从全场刺激中提取单个细胞的空间感受野区域。"""
    return stimulus[:, x0:x0 + patch_size, y0:y0 + patch_size]


def build_design_matrix(
    stimulus_patch: np.ndarray, n_temporal: int = 1
) -> np.ndarray:
    """
    构建时空联合设计矩阵。
    
    Parameters
    ----------
    stimulus_patch : np.ndarray, shape (n_frames, n_x)
        空间展平后的刺激
    n_temporal : int
        时间帧深度 (NT)
        
    Returns
    -------
    design_matrix : np.ndarray, shape (n_frames, n_x * n_temporal)
    """
    n_frames, n_x = stimulus_patch.shape
    if n_temporal == 1:
        return stimulus_patch
    
    design = np.zeros((n_frames, n_x * n_temporal))
    for t in range(n_temporal):
        design[t:, t * n_x:(t + 1) * n_x] = stimulus_patch[:n_frames - t]
    return design


def normalize_stimulus(stimulus: np.ndarray) -> np.ndarray:
    """对每个像素维度进行零均值化和单位方差归一化。"""
    mean = stimulus.mean(axis=0)
    std = stimulus.std(axis=0)
    std[std == 0] = 1.0
    return (stimulus - mean) / std


def align_stimulus_response(
    stimulus: np.ndarray, response: np.ndarray, lag: int
) -> Tuple[np.ndarray, np.ndarray]:
    """根据时间延迟参数对齐刺激和响应 (对应 MATLAB circshift)。"""
    stimulus_aligned = np.roll(stimulus, -lag, axis=0)
    return stimulus_aligned, response


def train_test_split(
    stimulus: np.ndarray, response: np.ndarray, test_fraction: float = 0.2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """将数据划分为训练集和测试集。"""
    n = len(response)
    n_test = int(n * test_fraction)
    return (
        stimulus[:-n_test], response[:-n_test],
        stimulus[-n_test:], response[-n_test:]
    )


def jackknife_split(n_samples: int, n_folds: int = 5) -> list:
    """生成 Jackknife 交叉验证的折叠索引。"""
    fold_size = n_samples // n_folds
    folds = []
    for i in range(n_folds):
        test_idx = np.arange(i * fold_size, min((i + 1) * fold_size, n_samples))
        train_idx = np.setdiff1d(np.arange(n_samples), test_idx)
        folds.append((train_idx, test_idx))
    return folds
