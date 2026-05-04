"""
数据 I/O 模块
============

负责读取原始实验数据文件，对应 MATLAB 的 ReadFramev2.m 和 load 函数。

支持格式:
    - .raw  : 白噪声棋盘格刺激 (二进制)
    - .isk  : 脉冲序列文件 (整数计数/帧)
    - .mat  : MATLAB 数据文件 (v5 via scipy, v7.3 via h5py)
"""

import numpy as np
from scipy.io import loadmat
import h5py
from pathlib import Path


def load_mat(filepath: str | Path) -> dict:
    """读取 .mat 文件，自动选择 scipy.loadmat 或 h5py。"""
    filepath = Path(filepath)
    try:
        return loadmat(str(filepath), squeeze_me=True)
    except NotImplementedError:
        # MATLAB v7.3 格式 (HDF5)
        data = {}
        with h5py.File(str(filepath), "r") as f:
            for key in f.keys():
                data[key] = np.array(f[key])
        return data


def read_stimulus_raw(filepath: str | Path, nx: int, ny: int) -> np.ndarray:
    """
    读取白噪声棋盘格刺激的原始二进制文件。
    
    对应 MATLAB: ReadFramev2.m
    
    Parameters
    ----------
    filepath : str or Path
        .raw 文件路径
    nx, ny : int
        棋盘格的空间维度
        
    Returns
    -------
    stimulus : np.ndarray, shape (n_frames, nx, ny)
        刺激帧序列
    """
    filepath = Path(filepath)
    raw_data = np.fromfile(str(filepath), dtype=np.uint8)
    n_pixels = nx * ny
    n_frames = len(raw_data) // n_pixels
    stimulus = raw_data[:n_frames * n_pixels].reshape(n_frames, nx, ny)
    return stimulus


def read_spike_train(filepath: str | Path) -> np.ndarray:
    """
    读取 .isk 脉冲序列文件。
    
    Parameters
    ----------
    filepath : str or Path
        .isk 文件路径
        
    Returns
    -------
    spikes : np.ndarray
        每帧的脉冲计数
    """
    filepath = Path(filepath)
    spikes = np.fromfile(str(filepath), dtype=np.int16)
    return spikes


def load_cell_parameters(filepath: str | Path) -> dict:
    """
    读取细胞参数文件 (RetinaCellParameters_*.mat)。
    
    包含: 空间感受野位置 (x0, y0)、大小 (Nv)、时间延迟 (lagshifts)、
         时间深度 (NT)、全场维度 (Nx, cx) 等。
    """
    return load_mat(filepath)
