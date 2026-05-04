#!/usr/bin/env python3
"""
视网膜模块完整复现脚本
======================

对应 MATLAB 脚本: script_101 ~ script_109, makefig_101 ~ makefig_106

本脚本按顺序执行以下分析流程:
    阶段一: 数据准备与预处理 (script_101)
    阶段二: STA 脉冲触发平均 (script_102)
    阶段三: STC 脉冲触发协方差 (script_103, script_104)
    阶段四: MNE 最大噪声熵模型 (script_105, script_106)
    阶段五: GLM (跳过 - 需要 Pillow 工具包)
    阶段六: 预测与验证 (script_108, script_109)
    阶段七: 生成图表

论文参考: Aljadeff, Lansdell, Fairhall & Kleinfeld (2016) Neuron, 91
链接: http://dx.doi.org/10.1016/j.neuron.2016.05.039
"""

import numpy as np
from scipy.io import loadmat
from scipy.interpolate import interp1d
from scipy.signal import welch, coherence
from pathlib import Path
import matplotlib.pyplot as plt
import os
import sys
import warnings

warnings.filterwarnings("ignore")

# ============================================================================
# 配置参数
# ============================================================================

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "RetinaData"
OUTPUT_DIR = PROJECT_ROOT / "python" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 分析参数
ICELL = 3           # 分析的细胞编号 (论文中展示的是 cell 3)
N_JK = 5            # Jackknife 折数
STIM_LENGTHS = ["short2", "short3", "long"]
NBINS = 15          # 用于非线性估计的 bin 数
NBINS_PLOT = 45     # 用于绘图的 bin 数
MNE_ORDER = 2       # MNE 模型阶数
MNE_NJACK = 4       # MNE 内部 jackknife 数
MNE_NULL_REP = 500  # MNE 零分布重复数
STC_NULL_REP = 1000 # STC 零分布重复数

print("=" * 70)
print("  视网膜模块代码复现 - Python 版本")
print("  Analysis of Neuronal Spike Trains, Deconstructed")
print("=" * 70)
print(f"\n项目根目录: {PROJECT_ROOT}")
print(f"数据目录:   {DATA_DIR}")
print(f"输出目录:   {OUTPUT_DIR}")
print(f"分析细胞:   Cell {ICELL}")


# ============================================================================
# 阶段一: 数据准备与预处理
# ============================================================================

def read_frame_v2(filepath, n_frames, nd, nv, cx, x0, y0):
    """
    读取白噪声棋盘格刺激的原始二进制文件。
    对应 MATLAB: ReadFramev2.m
    
    Parameters
    ----------
    filepath : Path - .raw 文件路径
    n_frames : int - 帧数
    nd : int - 全帧尺寸 (nd x nd)
    nv : int - patch 尺寸 (nv x nv)  
    cx : int - 空间下采样因子
    x0, y0 : int - patch 左上角坐标 (1-indexed in MATLAB, 0-indexed here)
    
    Returns
    -------
    stimulus : np.ndarray, shape (n_frames * fsize,)
    """
    fsize = (nv ** 2) // (cx ** 2)
    stimulus = np.zeros(n_frames * fsize)
    
    with open(filepath, "rb") as f:
        for j in range(n_frames):
            # 读取一帧: nd*nd 个 uint8 字节
            frame_data = np.frombuffer(f.read(nd * nd), dtype=np.uint8).astype(np.float64)
            if len(frame_data) < nd * nd:
                break
            # reshape 并转置 (MATLAB 按列存储, reshape 后转置等效于按行读取)
            frame = frame_data.reshape(nd, nd).T  # MATLAB: reshape(...,Nd,Nd)'
            # 提取 patch (MATLAB 1-indexed → Python 0-indexed)
            patch = frame[y0-1:y0-1+nv, x0-1:x0-1+nv]
            stimulus[j * fsize:(j + 1) * fsize] = patch.flatten()
    
    return stimulus


def build_stimulus_phase():
    """
    阶段一: 构建刺激矩阵与响应向量。
    对应 MATLAB: script_101_RetinaData_BuildStimulus.m
    """
    print("\n" + "=" * 70)
    print("  阶段一: 数据准备与预处理")
    print("=" * 70)
    
    results = {}
    
    for iL, stim_length in enumerate(STIM_LENGTHS):
        print(f"\n  处理刺激配置: {stim_length}")
        
        # 加载细胞参数
        param_file = DATA_DIR / f"RetinaCellParameters_{stim_length}.mat"
        params = loadmat(str(param_file), squeeze_me=True)
        
        lag = int(params["lagshifts"][ICELL - 1])  # MATLAB 1-indexed
        nv = int(params["Nv"][ICELL - 1])
        nx_full = int(params["Nx"])
        cx = int(params["cx"])
        x0 = int(params["x0"][ICELL - 1])
        y0 = int(params["y0"][ICELL - 1])
        nt = int(params["NT"])
        
        fsize = nv ** 2 // cx ** 2
        
        # 加载脉冲序列
        spike_file = DATA_DIR / f"whitenoisec{ICELL}.isk"
        with open(spike_file, "r") as f:
            R = np.array([int(line.strip()) for line in f.readlines()], dtype=np.float64)
        
        T_total = len(R)
        print(f"    总帧数: {T_total}, 空间维度: {fsize}, 时间深度: {nt}, 延迟: {lag}")
        
        # 加载刺激
        raw_file = DATA_DIR / "whitenoise.raw"
        S = read_frame_v2(raw_file, T_total, nx_full, nv, cx, x0, y0)
        
        NX = len(S) // T_total  # 空间维度
        N = NX * nt              # 总刺激维度
        
        # reshape 为 (T x NX)
        S = S.reshape(T_total, NX, order="C")
        
        # 零均值化和归一化
        S = S - S.mean(axis=0)
        std_s = S.std(axis=0)
        std_s[std_s == 0] = 1.0
        S = S / std_s
        
        # 对齐刺激和响应 (circshift)
        R = np.roll(R, -lag)
        R = R[:T_total - lag]
        S = S[:T_total - lag]
        T = len(R)
        
        # 多帧时间扩展
        if nt > 1:
            T = T - (nt - 1)
            S1 = np.zeros((T, N))
            for i in range(nt):
                S1[:, NX * i:NX * (i + 1)] = S[i:T + i]
            S = S1
            R = R[nt - 1:]
        
        Sa = S.copy()
        Ra = R.copy()
        Ta = T
        
        # Jackknife 划分
        for iJK in range(1, N_JK + 1):
            itest_start = round(Ta * (iJK - 1) / N_JK)
            itest_end = round(Ta * iJK / N_JK)
            itest = np.arange(itest_start, itest_end)
            ifit = np.setdiff1d(np.arange(Ta), itest)
            
            St = Sa[itest]
            Rt = Ra[itest]
            S_train = Sa[ifit]
            R_train = Ra[ifit]
            
            key = (stim_length, iJK)
            results[key] = {
                "S": S_train, "R": R_train,
                "St": St, "Rt": Rt,
                "N": N, "NX": NX, "NT": nt,
                "T": len(R_train), "Tt": len(Rt)
            }
        
        print(f"    完成: N={N}, NX={NX}, NT={nt}, 训练帧数≈{results[(stim_length, 1)]['T']}")
    
    print("\n  ✓ 阶段一完成: 数据预处理就绪")
    return results


# ============================================================================
# 阶段二: STA 脉冲触发平均
# ============================================================================

def histcn_1d(x, bins):
    """一维直方图, 返回 count 和 bin assignment。"""
    digitized = np.digitize(x, bins)
    # clip 到有效范围 [1, len(bins)]
    digitized = np.clip(digitized, 1, len(bins))
    count = np.bincount(digitized, minlength=len(bins) + 1)[1:len(bins) + 1]
    return count, digitized


def compute_sta_phase(data):
    """
    阶段二: 计算 STA 模型。
    对应 MATLAB: script_102_RetinaData_STA.m
    """
    print("\n" + "=" * 70)
    print("  阶段二: STA 脉冲触发平均")
    print("=" * 70)
    
    sta_results = {}
    
    for stim_length in ["short3", "long"]:  # 论文中使用这两个配置
        for iJK in range(1, N_JK + 1):
            key = (stim_length, iJK)
            d = data[key]
            S, R, T, N = d["S"], d["R"], d["T"], d["N"]
            
            # 1. 计算 STA (Equations 14, 15)
            sta = S.T @ R / R.sum() - S.mean(axis=0)
            sta = sta / np.linalg.norm(sta)
            
            # 2. 计算非线性函数 (Equations 6, 7)
            z = S @ sta  # 投影
            zx = np.max(np.abs(z)) + 1e-3
            
            # MATLAB 的 linspace(-zx,zx,nbns) 生成 nbns 个点作为 bin edges
            # 然后 bin centers 在 edges 之间
            bns = np.linspace(-zx, zx, NBINS)  # MATLAB: bns = linspace(-zx,zx,nbns)
            dbins = bns[1] - bns[0]
            ctr = bns[:-1] + dbins / 2  # nbns-1 个 bin centers
            
            bns_plot = np.linspace(-zx, zx, NBINS_PLOT)
            dbins_plot = bns_plot[1] - bns_plot[0]
            ctr_plot = bns_plot[:-1] + dbins_plot / 2
            
            # 计算每个 bin 的平均响应 (MATLAB histcn 风格)
            n_ctr = len(ctr)
            g = np.zeros(n_ctr)
            # 使用 digitize 将 z 分配到 bins
            digitized = np.digitize(z, bns)
            digitized = np.clip(digitized, 1, NBINS - 1)  # 有效范围 1..NBINS-1
            for ib in range(1, NBINS):
                mask = digitized == ib
                if mask.sum() > 0:
                    g[ib - 1] = R[mask].mean()
            g[np.isnan(g)] = 0
            
            # 刺激分布
            Pf = np.zeros(n_ctr)
            for ib in range(1, NBINS):
                Pf[ib - 1] = np.sum(digitized == ib)
            Pf = Pf / (T * dbins)
            
            Pf_plot_dig = np.digitize(z, bns_plot)
            Pf_plot_dig = np.clip(Pf_plot_dig, 1, NBINS_PLOT - 1)
            n_ctr_plot = len(ctr_plot)
            Pf_plot = np.zeros(n_ctr_plot)
            for ib in range(1, NBINS_PLOT):
                Pf_plot[ib - 1] = np.sum(Pf_plot_dig == ib)
            Pf_plot = Pf_plot / (T * dbins_plot)
            
            # 3. 构建 LN 模型
            # 平滑插值 → 整流 → 归一化
            sta_model_interp = interp1d(ctr, g, kind="cubic", 
                                         bounds_error=False, fill_value=0)
            
            # 整流 + 归一化
            model_pred = np.maximum(sta_model_interp(z), 0) + 1e-8
            norm_factor = model_pred.sum() / R.sum() if R.sum() > 0 else 1.0
            
            sta_results[key] = {
                "sta": sta,
                "g": g,
                "ctr": ctr,
                "Pf": Pf,
                "Pf_plot": Pf_plot,
                "ctr_plot": ctr_plot,
                "bns": bns,
                "norm_factor": norm_factor,
                "interp_func": sta_model_interp,
                "NX": d["NX"], "NT": d["NT"], "N": N
            }
        
        print(f"  {stim_length}: STA 计算完成 ({N_JK} 折)")
    
    print("\n  ✓ 阶段二完成: STA 模型就绪")
    return sta_results


# ============================================================================
# 阶段三: STC 脉冲触发协方差
# ============================================================================

def _stc_null_distribution_fast(S, R, Cp, N, T, n_rep):
    """
    优化的 STC 零分布计算。
    
    关键优化:
    1. 避免 np.cov (内部做了冗余的均值计算和归一化)，直接用矩阵乘法
    2. 预计算 S^T @ S 用于加速
    3. 使用 PyTorch MPS (Apple Silicon GPU) 加速矩阵运算
    4. 批量置换减少 Python 循环开销
    """
    import time as _time
    
    # 尝试使用 PyTorch MPS 加速
    use_mps = False
    try:
        import torch
        if torch.backends.mps.is_available():
            use_mps = True
    except ImportError:
        pass
    
    if use_mps:
        return _stc_null_torch_mps(S, R, Cp, N, T, n_rep)
    else:
        return _stc_null_numpy_fast(S, R, Cp, N, T, n_rep)


def _stc_null_torch_mps(S, R, Cp, N, T, n_rep):
    """PyTorch MPS 加速的零分布计算。"""
    import torch
    import time as _time
    
    device = torch.device("mps")
    print(f"      🚀 使用 Apple MPS GPU 加速", flush=True)
    
    # 转移到 GPU
    S_t = torch.tensor(S, dtype=torch.float32, device=device)
    R_t = torch.tensor(R, dtype=torch.float32, device=device)
    Cp_t = torch.tensor(Cp, dtype=torch.float32, device=device)
    R_mean = R_t.mean()
    
    edC_null = np.zeros(N * n_rep)
    t3 = _time.time()
    
    for ir in range(n_rep):
        shift = np.random.randint(1, T)
        # 在 GPU 上做 roll
        R_rnd = torch.roll(R_t, int(shift))
        sqrt_R_rnd = torch.sqrt(R_rnd)
        
        # 加权刺激
        S_w = S_t * sqrt_R_rnd.unsqueeze(1)
        S_w_mean = S_w.mean(dim=0)
        S_w_c = S_w - S_w_mean
        
        # 协方差: (S_w_c^T @ S_w_c) / (T-1) / R_mean
        C_rnd = (S_w_c.T @ S_w_c) / (T - 1) / R_mean
        
        # 差矩阵特征值
        dC_rnd = C_rnd - Cp_t
        # eigh 在 MPS 上可能不支持，回退到 CPU
        evals = torch.linalg.eigvalsh(dC_rnd.cpu()).numpy()
        edC_null[ir * N:(ir + 1) * N] = evals
        
        if (ir + 1) % 20 == 0 or ir == 0:
            elapsed = _time.time() - t3
            eta = elapsed / (ir + 1) * (n_rep - ir - 1)
            print(f"      置换 {ir+1}/{n_rep} (MPS) 耗时 {elapsed:.1f}s, 剩余 {eta:.0f}s", flush=True)
    
    print(f"      MPS 零分布完成 (总耗时 {_time.time()-t3:.1f}s)")
    return edC_null


def _stc_null_numpy_fast(S, R, Cp, N, T, n_rep):
    """
    纯 NumPy 优化的零分布计算。
    
    优化点:
    - 用 S^T @ (diag(w) @ S) 代替 np.cov（避免重复计算均值和归一化）
    - 预计算 S 的列和
    - 只计算上三角然后对称化
    """
    import time as _time
    
    print(f"      使用优化 NumPy 算法", flush=True)
    
    edC_null = np.zeros(N * n_rep)
    t3 = _time.time()
    
    # 预计算
    S_col_sum = S.sum(axis=0)  # (N,)
    
    for ir in range(n_rep):
        shift = np.random.randint(1, T)
        R_rnd = np.roll(R, shift)
        sqrt_R_rnd = np.sqrt(R_rnd)
        w = sqrt_R_rnd  # 权重
        
        # 加权刺激的均值: sum(w_i * S_i) / T
        w_sum = w.sum()
        Sw_sum = (S * w[:, None]).sum(axis=0)  # (N,)
        Sw_mean = Sw_sum / T
        
        # 协方差 = (1/(T-1)) * [S^T diag(w^2) S - T * Sw_mean^T Sw_mean]
        # 其中 S_w_c = S * w - Sw_mean, 展开后:
        # C = S^T diag(w^2) S / (T-1) - T/(T-1) * outer(Sw_mean, Sw_mean)
        w2 = w * w  # w^2 = R_rnd (因为 w = sqrt(R_rnd))
        
        # 关键: S^T @ diag(w2) @ S = (S * w2).T @ S — 矩阵乘法，比 np.cov 快
        StW2S = (S * w2[:, None]).T @ S  # (N, N)
        outer_mean = np.outer(Sw_mean, Sw_mean)
        
        C_rnd = (StW2S / (T - 1) - T / (T - 1) * outer_mean) / R.mean()
        
        edC_null[ir * N:(ir + 1) * N] = np.linalg.eigvalsh(C_rnd - Cp)
        
        if (ir + 1) % 20 == 0 or ir == 0:
            elapsed = _time.time() - t3
            eta = elapsed / (ir + 1) * (n_rep - ir - 1)
            print(f"      置换 {ir+1}/{n_rep} 耗时 {elapsed:.1f}s, 剩余 {eta:.0f}s", flush=True)
    
    print(f"      零分布完成 (总耗时 {_time.time()-t3:.1f}s)")
    return edC_null


def _compute_cov_fast(S, weights, R_mean, T):
    """快速计算加权协方差矩阵，避免 np.cov 开销。"""
    w = weights
    Sw = S * w[:, None]
    Sw_mean = Sw.mean(axis=0)
    Sw_c = Sw - Sw_mean
    return (Sw_c.T @ Sw_c) / (T - 1) / R_mean


def compute_stc_phase(data, sta_results):
    """
    阶段三: 计算 STC 模型。
    对应 MATLAB: script_103 + script_104
    
    优化方案:
    1. 用直接矩阵乘法替代 np.cov (避免冗余均值/归一化)
    2. PyTorch MPS GPU 加速 (Apple Silicon)
    3. 预计算共享数据减少重复计算
    """
    import time as _time
    
    print("\n" + "=" * 70)
    print("  阶段三: STC 脉冲触发协方差 [优化版]")
    print("=" * 70)
    
    # 检测加速方案
    try:
        import torch
        if torch.backends.mps.is_available():
            print("  🚀 检测到 Apple MPS GPU，将使用 GPU 加速")
        else:
            print("  ℹ️ MPS 不可用，使用优化 NumPy 算法")
    except ImportError:
        print("  ℹ️ PyTorch 未安装，使用优化 NumPy 算法")
    
    stc_results = {}
    
    total_configs = 2 * N_JK  # short3 + long, 各 N_JK 折
    config_idx = 0
    
    for stim_length in ["short3", "long"]:
        for iJK in range(1, N_JK + 1):
            config_idx += 1
            key = (stim_length, iJK)
            d = data[key]
            S, R, T, N = d["S"], d["R"], d["T"], d["N"]
            sta = sta_results[key]["sta"]
            
            t0 = _time.time()
            print(f"\n  [{config_idx}/{total_configs}] {stim_length} JK{iJK}: STC (N={N}, T={T})")
            sys.stdout.flush()
            
            # === script_103: STC 显著性检验 ===
            
            # 1. 先验协方差 (用快速方法)
            print(f"    → 步骤1/4: 先验协方差 Cp...", end="", flush=True)
            S_mean = S.mean(axis=0)
            S_c = S - S_mean
            Cp = (S_c.T @ S_c) / (T - 1)
            print(f" {_time.time()-t0:.1f}s")
            
            # 2. 脉冲触发协方差 (用快速方法)
            t1 = _time.time()
            print(f"    → 步骤2/4: 脉冲触发协方差 Cs...", end="", flush=True)
            sqrt_R = np.sqrt(R)
            Cs = _compute_cov_fast(S, sqrt_R, R.mean(), T)
            print(f" {_time.time()-t1:.1f}s")
            
            # 3. 协方差差矩阵
            t2 = _time.time()
            print(f"    → 步骤3/4: 特征分解...", end="", flush=True)
            dC = Cs - Cp
            edC, vdC = np.linalg.eigh(dC)
            idx = np.argsort(edC)
            edC = edC[idx]
            vdC = vdC[:, idx]
            print(f" {_time.time()-t2:.1f}s  范围:[{edC[0]:.4f}, {edC[-1]:.4f}]")
            
            # 4. 零分布 (优化版)
            n_rep = min(STC_NULL_REP, 200)
            print(f"    → 步骤4/4: 零分布 ({n_rep}次置换)...", flush=True)
            edC_null = _stc_null_distribution_fast(S, R, Cp, N, T, n_rep)
            
            max_enull = np.percentile(edC_null, 100)  # a=0
            min_enull = np.percentile(edC_null, 0)
            
            # 找出显著特征
            isig = np.where((edC > max_enull) | (edC < min_enull))[0]
            nsig = len(isig)
            
            if nsig > 0:
                isig = isig[np.argsort(np.abs(edC[isig]))[::-1]]
            
            total_time = _time.time() - t0
            print(f"    ✓ {stim_length} JK{iJK}: {nsig} 个显著 STC 特征 (总耗时 {total_time:.1f}s)")
            print(f"      零分布范围: [{min_enull:.4f}, {max_enull:.4f}]")
            sys.stdout.flush()
            
            # === script_104: 构建 STC 模型 ===
            
            # 移除与 STA 重叠过大的 STC 特征
            proj_th = 0.9
            kmax = 2
            
            if nsig > 0:
                proj_stcsta = np.abs(vdC[:, isig].T @ sta)
                keep = proj_stcsta < proj_th
                isig = isig[keep]
                nsig = len(isig)
            
            kmodel = 1 + min(nsig, kmax - 1)
            
            # 构建特征子空间
            stc_features = np.zeros((N, kmodel - 1))
            for i in range(kmodel - 1):
                v = vdC[:, isig[i]]
                v = v - (v @ sta) * sta  # 投影出 STA
                v = v / np.linalg.norm(v)
                stc_features[:, i] = v
            
            f_subspace = np.column_stack([sta] + [stc_features[:, i] for i in range(kmodel - 1)])
            
            # 投影刺激
            z = S @ f_subspace
            zx = np.max(np.abs(z)) + 1e-3
            bins = np.linspace(-zx, zx, NBINS + 1)
            dbins = bins[1] - bins[0]
            ctr = bins[:-1] + dbins / 2
            
            # 计算非线性 (根据维度)
            if kmodel == 1:
                g = np.zeros(NBINS)
                digitized = np.digitize(z.ravel(), bins)
                digitized = np.clip(digitized, 1, NBINS)
                for ib in range(1, NBINS + 1):
                    mask = digitized == ib
                    if mask.sum() > 0:
                        g[ib - 1] = R[mask].mean()
                g[np.isnan(g)] = 0
                
                stc_model_interp = interp1d(ctr, g, kind="linear",
                                            bounds_error=False, fill_value=0)
                model_pred = np.maximum(stc_model_interp(z.ravel()), 0) + 1e-8
                
            elif kmodel == 2:
                g = np.zeros((NBINS, NBINS))
                g_a = np.zeros(NBINS)
                g_c = np.zeros(NBINS)
                
                dig1 = np.clip(np.digitize(z[:, 0], bins), 1, NBINS)
                dig2 = np.clip(np.digitize(z[:, 1], bins), 1, NBINS)
                
                for ib1 in range(1, NBINS + 1):
                    mask1 = dig1 == ib1
                    if mask1.sum() > 0:
                        g_a[ib1 - 1] = R[mask1].mean()
                    for ib2 in range(1, NBINS + 1):
                        mask = mask1 & (dig2 == ib2)
                        if mask.sum() > 0:
                            g[ib1 - 1, ib2 - 1] = R[mask].mean()
                    mask2 = dig2 == ib1
                    if mask2.sum() > 0:
                        g_c[ib1 - 1] = R[mask2].mean()
                
                g[np.isnan(g)] = 0
                g_a[np.isnan(g_a)] = 0
                g_c[np.isnan(g_c)] = 0
                
                # 使用 2D 插值
                from scipy.interpolate import RegularGridInterpolator
                stc_model_interp = RegularGridInterpolator(
                    (ctr, ctr), g, method="linear",
                    bounds_error=False, fill_value=0
                )
                model_pred = np.maximum(stc_model_interp(z), 0) + 1e-8
            else:
                # kmodel >= 3 - 简化处理: 使用边缘非线性求和
                model_pred = np.zeros(T)
                for dim in range(kmodel):
                    g_dim = np.zeros(NBINS)
                    dig = np.clip(np.digitize(z[:, dim], bins), 1, NBINS)
                    for ib in range(1, NBINS + 1):
                        mask = dig == ib
                        if mask.sum() > 0:
                            g_dim[ib - 1] = R[mask].mean()
                    g_dim[np.isnan(g_dim)] = 0
                    f_interp = interp1d(ctr, g_dim, kind="linear",
                                       bounds_error=False, fill_value=0)
                    model_pred += np.maximum(f_interp(z[:, dim]), 0)
                model_pred = model_pred + 1e-8
                stc_model_interp = None
            
            # 归一化
            nT = R.sum()
            nT_model = model_pred.sum()
            norm_factor = nT / nT_model if nT_model > 0 else 1.0
            
            stc_results[key] = {
                "f_subspace": f_subspace,
                "kmodel": kmodel,
                "nsig": nsig,
                "edC": edC,
                "edC_null": edC_null,
                "max_enull": max_enull,
                "min_enull": min_enull,
                "g": g,
                "bins": bins,
                "ctr": ctr,
                "norm_factor": norm_factor,
                "interp_func": stc_model_interp,
            }
    
    print("\n  ✓ 阶段三完成: STC 模型就绪")
    return stc_results


# ============================================================================
# 阶段四: MNE 最大噪声熵模型 [MPS 优化版]
# ============================================================================

def sigmoid(x):
    """数值稳定的 sigmoid。"""
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def _detect_mps():
    """检测 MPS GPU 是否可用。"""
    try:
        import torch
        if torch.backends.mps.is_available():
            return True
    except ImportError:
        pass
    return False


# --- MPS 优化版 MNE 函数 ---

def _mne_logloss_mps(p_t, stim_t, resp_t, order, Ndim):
    """MPS GPU 版本的 logloss。"""
    import torch
    ptemp = p_t[1:Ndim + 1]
    if order > 1:
        J = p_t[Ndim + 1:Ndim + 1 + Ndim ** 2].reshape(Ndim, Ndim)
        linear = p_t[0] + stim_t @ ptemp + (stim_t * (stim_t @ J)).sum(1)
    else:
        linear = p_t[0] + stim_t @ ptemp
    
    # 数值稳定的 softplus: log(1+exp(x))
    f1 = torch.nn.functional.softplus(linear)
    f0 = torch.nn.functional.softplus(-linear)
    return (resp_t * f1 + (1 - resp_t) * f0).mean()


def _mne_gradient_mps(p_t, stim_t, avgs_t, order, Ndim, Nsamples):
    """MPS GPU 版本的 gradient。"""
    import torch
    ptemp = p_t[1:Ndim + 1]
    if order > 1:
        J = p_t[Ndim + 1:Ndim + 1 + Ndim ** 2].reshape(Ndim, Ndim)
        linear = p_t[0] + stim_t @ ptemp + (stim_t * (stim_t @ J)).sum(1)
    else:
        linear = p_t[0] + stim_t @ ptemp
    
    pSpike = torch.sigmoid(-linear)
    
    averages = torch.zeros(len(avgs_t), device=stim_t.device)
    averages[0] = pSpike.mean()
    averages[1:Ndim + 1] = (stim_t.T @ pSpike) / Nsamples
    
    if order > 1:
        temp = (stim_t.T @ (pSpike.unsqueeze(1) * stim_t)) / Nsamples
        averages[Ndim + 1:Ndim + 1 + Ndim ** 2] = temp.flatten()
    
    return avgs_t - averages


def mne_logloss(p, stim, resp, order):
    """MNE 对数损失函数 (NumPy 回退版)。"""
    Nsamples, Ndim = stim.shape
    ptemp = p[1:Ndim + 1]
    if order > 1:
        J = p[Ndim + 1:Ndim + 1 + Ndim ** 2].reshape(Ndim, Ndim)
        linear = p[0] + stim @ ptemp + np.sum(stim * (stim @ J), axis=1)
    else:
        linear = p[0] + stim @ ptemp
    f1 = np.log(1 + np.exp(np.clip(linear, -500, 500)))
    f0 = np.log(1 + np.exp(np.clip(-linear, -500, 500)))
    return np.mean(resp * f1 + (1 - resp) * f0)


def mne_gradient(p, stim, avgs, order):
    """MNE 梯度 (NumPy 回退版)。"""
    Nsamples, Ndim = stim.shape
    ptemp = p[1:Ndim + 1]
    if order > 1:
        J = p[Ndim + 1:Ndim + 1 + Ndim ** 2].reshape(Ndim, Ndim)
        linear = p[0] + stim @ ptemp + np.sum(stim * (stim @ J), axis=1)
    else:
        linear = p[0] + stim @ ptemp
    pSpike = sigmoid(-linear)
    averages = np.zeros(len(avgs))
    averages[0] = pSpike.mean()
    averages[1:Ndim + 1] = (stim.T @ pSpike) / Nsamples
    if order > 1:
        temp = (stim.T @ (pSpike[:, None] * stim)) / Nsamples
        averages[Ndim + 1:Ndim + 1 + Ndim ** 2] = temp.flatten()
    return avgs - averages


def mne_fit(stim, resp, test_stim, test_resp, order):
    """
    MNE 拟合 (共轭梯度法 + early stopping)。
    自动检测 MPS GPU 并使用加速版本。
    对应 MATLAB: MNEfit_RetinaData.m + frprmn_global_min.m
    """
    if _detect_mps():
        return _mne_fit_mps(stim, resp, test_stim, test_resp, order)
    else:
        return _mne_fit_numpy(stim, resp, test_stim, test_resp, order)


def _mne_fit_mps(stim, resp, test_stim, test_resp, order):
    """MPS GPU 加速的 MNE 拟合。"""
    import torch
    import time as _time
    
    device = torch.device("mps")
    Nsamples, Ndim = stim.shape
    
    # 将数据预加载到 GPU
    stim_t = torch.tensor(stim, dtype=torch.float32, device=device)
    resp_t = torch.tensor(resp, dtype=torch.float32, device=device)
    test_stim_t = torch.tensor(test_stim, dtype=torch.float32, device=device)
    test_resp_t = torch.tensor(test_resp, dtype=torch.float32, device=device)
    
    # 计算约束平均 (GPU)
    psp = resp_t.mean()
    avg = (stim_t.T @ resp_t) / Nsamples
    avgs_t = torch.cat([psp.unsqueeze(0), avg])
    if order > 1:
        avgsqrd = (stim_t.T @ (resp_t.unsqueeze(1) * stim_t)) / Nsamples
        avgs_t = torch.cat([avgs_t, avgsqrd.flatten()])
    
    # 初始化参数 (GPU)
    n_params = 1 + Ndim + (Ndim ** 2 if order > 1 else 0)
    p_t = torch.zeros(n_params, dtype=torch.float32, device=device)
    psp_val = psp.item()
    p_t[0] = np.log(1 / psp_val - 1) if 0 < psp_val < 1 else 0
    p_t[1:Ndim + 1] = torch.tensor(0.001 * (2 * np.random.rand(Ndim) - 1), dtype=torch.float32, device=device)
    if order > 1:
        temp = 0.0005 * (2 * np.random.rand(Ndim, Ndim) - 1)
        temp = (temp + temp.T) / 2
        p_t[Ndim + 1:] = torch.tensor(temp.flatten(), dtype=torch.float32, device=device)
    
    # 共轭梯度法 (Polak-Ribière) + early stopping — 全部在 GPU 上
    ITMAX = 400
    TALLY_MAX = 20
    
    xi = _mne_gradient_mps(p_t, stim_t, avgs_t, order, Ndim, Nsamples)
    g = -xi.clone()
    h = g.clone()
    xi = g.clone()
    
    best_test_loss = float("inf")
    best_p = p_t.clone()
    tally = 0
    t0 = _time.time()
    
    for it in range(ITMAX):
        # 线搜索 (backtracking)
        step = 0.2
        loss_old = _mne_logloss_mps(p_t, stim_t, resp_t, order, Ndim).item()
        for _ in range(10):
            p_new = p_t + step * xi
            loss_new = _mne_logloss_mps(p_new, stim_t, resp_t, order, Ndim).item()
            if loss_new < loss_old:
                break
            step *= 0.5
        
        p_t = p_t + step * xi
        
        # Early stopping on test set
        test_loss = _mne_logloss_mps(p_t, test_stim_t, test_resp_t, order, Ndim).item()
        
        if test_loss < best_test_loss * 0.999999 or it <= 1:
            best_test_loss = test_loss
            best_p = p_t.clone()
            tally = 0
        else:
            tally += 1
        
        if tally >= TALLY_MAX:
            break
        
        # 更新共轭方向
        xi = _mne_gradient_mps(p_t, stim_t, avgs_t, order, Ndim, Nsamples)
        gg = (g ** 2).sum().item()
        if gg == 0:
            break
        dgg = ((xi + g) * xi).sum().item()
        gam = dgg / gg
        g = -xi
        h = g + gam * h
        xi = h.clone()
    
    return best_p.cpu().numpy()


def _mne_fit_numpy(stim, resp, test_stim, test_resp, order):
    """纯 NumPy 的 MNE 拟合 (回退版)。"""
    Nsamples, Ndim = stim.shape
    
    psp = resp.mean()
    avg = (stim.T @ resp) / Nsamples
    avgs = np.concatenate([[psp], avg])
    if order > 1:
        avgsqrd = (stim.T @ (resp[:, None] * stim)) / Nsamples
        avgs = np.concatenate([avgs, avgsqrd.flatten()])
    
    p = np.zeros(1 + Ndim + (Ndim ** 2 if order > 1 else 0))
    p[0] = np.log(1 / psp - 1) if 0 < psp < 1 else 0
    p[1:Ndim + 1] = 0.001 * (2 * np.random.rand(Ndim) - 1)
    if order > 1:
        temp = 0.0005 * (2 * np.random.rand(Ndim, Ndim) - 1)
        temp = (temp + temp.T) / 2
        p[Ndim + 1:] = temp.flatten()
    
    ITMAX = 400
    TALLY_MAX = 20
    
    xi = mne_gradient(p, stim, avgs, order)
    g = -xi.copy()
    h = g.copy()
    xi = g.copy()
    
    best_test_loss = np.inf
    best_p = p.copy()
    tally = 0
    
    for it in range(ITMAX):
        step = 0.2
        for _ in range(10):
            p_new = p + step * xi
            loss_new = mne_logloss(p_new, stim, resp, order)
            loss_old = mne_logloss(p, stim, resp, order)
            if loss_new < loss_old:
                break
            step *= 0.5
        p = p + step * xi
        
        test_loss = mne_logloss(p, test_stim, test_resp, order)
        if test_loss < best_test_loss * 0.999999 or it <= 1:
            best_test_loss = test_loss
            best_p = p.copy()
            tally = 0
        else:
            tally += 1
        if tally >= TALLY_MAX:
            break
        
        xi = mne_gradient(p, stim, avgs, order)
        gg = np.sum(g ** 2)
        if gg == 0:
            break
        dgg = np.sum((xi + g) * xi)
        gam = dgg / gg
        g = -xi
        h = g + gam * h
        xi = h.copy()
    
    return best_p


def compute_mne_phase(data):
    """
    阶段四: MNE 模型拟合与后处理。
    对应 MATLAB: script_105 + script_106
    """
    print("\n" + "=" * 70)
    print("  阶段四: MNE 最大噪声熵模型")
    print("=" * 70)
    
    mne_results = {}
    
    # 仅对 short3 配置进行 MNE 拟合 (最常用)
    for stim_length in ["short3"]:
        for iJK in range(1, N_JK + 1):
            key = (stim_length, iJK)
            d = data[key]
            S, R_raw, T, N = d["S"], d["R"], d["T"], d["N"]
            
            # 二值化响应
            R = np.sign(R_raw).astype(np.float64)
            R[R < 0] = 0  # 确保非负
            
            import time as _time
            use_mps = _detect_mps()
            accel = "🚀 MPS GPU" if use_mps else "CPU NumPy"
            print(f"  {stim_length} JK{iJK}: 拟合 MNE (N={N}, T={T}, {MNE_NJACK}折内部JK, {accel})")
            print(f"    参数量: 1 + {N} + {N}² = {1 + N + N**2:,}", flush=True)
            print(f"    脉冲率: {R.mean():.4f} ({R.sum():.0f} spikes / {T} frames)")
            
            t_mne_start = _time.time()
            
            # 内部 jackknife 拟合
            Tj = T // MNE_NJACK
            mne_models = []
            
            for j in range(MNE_NJACK):
                t_jk = _time.time()
                jtst = np.arange(j * Tj, (j + 1) * Tj)
                jfit = np.setdiff1d(np.arange(T), jtst)
                
                S_fit = S[jfit]
                R_fit = R[jfit]
                S_tst = S[jtst]
                R_tst = R[jtst]
                
                print(f"    内部 JK {j+1}/{MNE_NJACK}: 训练={len(jfit)}, 测试={len(jtst)}...", end="", flush=True)
                pfinal = mne_fit(S_fit, R_fit, S_tst, R_tst, MNE_ORDER)
                mne_models.append(pfinal)
                elapsed_jk = _time.time() - t_jk
                print(f" {elapsed_jk:.1f}s")
            
            # 平均模型参数
            m = np.mean(mne_models, axis=0)
            
            # 提取参数
            A = m[0]
            H = m[1:N + 1]
            J = m[N + 1:].reshape(N, N)
            
            # 特征值分解
            eJ, vJ = np.linalg.eigh(J)
            idx = np.argsort(eJ)
            eJ = eJ[idx]
            vJ = vJ[:, idx]
            
            # 零分布 (简化版: 减少重复次数)
            n_rep = min(MNE_NULL_REP, 100)
            dJ = np.diag(J).copy()
            upper_idx = np.triu_indices(N, k=1)
            oJ = J[upper_idx].copy()
            
            eJ_null = np.zeros(N * n_rep)
            for ir in range(n_rep):
                J_null = np.zeros((N, N))
                np.fill_diagonal(J_null, np.random.permutation(dJ))
                J_null[upper_idx] = np.random.permutation(oJ)
                J_null = J_null + J_null.T - np.diag(np.diag(J_null))
                eJ_null[ir * N:(ir + 1) * N] = np.linalg.eigvalsh(J_null)
            
            max_enull = np.max(eJ_null)
            min_enull = np.min(eJ_null)
            
            isig = np.where((eJ > max_enull) | (eJ < min_enull))[0]
            nsig = len(isig)
            
            if nsig > 0:
                isig = isig[np.argsort(np.abs(eJ[isig]))[::-1]]
            
            print(f"    显著二次特征: {nsig} 个")
            
            mne_results[key] = {
                "A": A, "H": H, "J": J,
                "eJ": eJ, "vJ": vJ,
                "eJ_null": eJ_null,
                "max_enull": max_enull,
                "min_enull": min_enull,
                "isig": isig, "nsig": nsig,
            }
    
    print("\n  ✓ 阶段四完成: MNE 模型就绪")
    return mne_results


# ============================================================================
# 阶段五: GLM 广义线性模型 (使用 statsmodels 替代 Pillow 工具包)
# ============================================================================

def _glm_fit_torch_mps(S_pca, R, hist_matrix, n_components, n_hist, lr=0.01, max_iter=200):
    """PyTorch MPS 加速的 Poisson GLM 拟合 (梯度下降)。"""
    import torch
    import time as _time
    
    device = torch.device("mps")
    T = S_pca.shape[0]
    n_features = n_components + n_hist + 1  # +1 for bias
    
    # 构建设计矩阵 (含常数项)
    X = np.hstack([np.ones((T, 1)), S_pca, hist_matrix])
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    R_t = torch.tensor(R, dtype=torch.float32, device=device)
    
    # 初始化参数
    beta = torch.zeros(n_features, dtype=torch.float32, device=device, requires_grad=True)
    
    optimizer = torch.optim.Adam([beta], lr=lr)
    
    t0 = _time.time()
    for it in range(max_iter):
        optimizer.zero_grad()
        # Poisson GLM: λ = exp(X @ β), loss = -Σ[r*log(λ) - λ]
        linear = X_t @ beta
        lam = torch.exp(torch.clamp(linear, -20, 20))
        loss = -(R_t * linear - lam).mean()
        loss.backward()
        optimizer.step()
        
        if (it + 1) % 50 == 0:
            print(f"      迭代 {it+1}/{max_iter}: loss={loss.item():.6f} ({_time.time()-t0:.1f}s)", flush=True)
    
    params = beta.detach().cpu().numpy()
    return params


def compute_glm_phase(data, sta_results):
    """
    阶段五: GLM 拟合。
    使用 PyTorch MPS GPU 加速的 Poisson GLM (替代 Pillow 工具包)。
    回退方案: statsmodels IRLS。
    
    对应 MATLAB: script_107_RetinaData_GLM.m
    
    模型: λ(t) = exp(k^T s(t) + h^T r_hist(t) + b)
    """
    import time as _time
    
    print("\n" + "=" * 70)
    print("  阶段五: GLM 广义线性模型 [MPS 优化版]")
    print("=" * 70)
    
    use_mps = _detect_mps()
    if use_mps:
        print("  🚀 使用 PyTorch MPS GPU 加速 (Adam 优化器)")
    else:
        print("  ℹ️ 使用 statsmodels IRLS (CPU)")
    print("  模型: λ(t) = exp(k^T s(t) + h^T r_hist(t) + b)")
    
    glm_results = {}
    stim_length = "short3"
    
    for iJK in range(1, N_JK + 1):
        key = (stim_length, iJK)
        d = data[key]
        S, R, T, N = d["S"], d["R"], d["T"], d["N"]
        NX, NT = d["NX"], d["NT"]
        sta = sta_results[key]["sta"]
        
        t0 = _time.time()
        print(f"\n  {stim_length} JK{iJK}: 拟合 Poisson GLM (N={N}, T={T})")
        
        # PCA 降维
        n_components = min(20, N)
        print(f"    → PCA 降维: {N} → {n_components} 维...", end="", flush=True)
        
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_components)
        S_pca = pca.fit_transform(S)
        St_pca = pca.transform(d["St"])
        
        explained = pca.explained_variance_ratio_.sum() * 100
        print(f" 解释方差: {explained:.1f}%")
        
        # 脉冲历史
        n_hist = 10
        print(f"    → 脉冲历史 (n_hist={n_hist})", flush=True)
        
        hist_matrix = np.zeros((T, n_hist))
        for lag in range(1, n_hist + 1):
            hist_matrix[lag:, lag - 1] = R[:-lag]
        
        n_params = 1 + n_components + n_hist
        print(f"    → 参数: {n_params} (1 bias + {n_components} stim + {n_hist} hist)")
        
        try:
            if use_mps:
                # PyTorch MPS 加速版
                print(f"    → 拟合 (MPS GPU, Adam)...", flush=True)
                params = _glm_fit_torch_mps(S_pca, R, hist_matrix, n_components, n_hist)
            else:
                # statsmodels 回退版
                import statsmodels.api as sm
                print(f"    → 拟合 (statsmodels IRLS)...", end="", flush=True)
                t1 = _time.time()
                X_train = np.hstack([S_pca, hist_matrix])
                X_train = sm.add_constant(X_train)
                model = sm.GLM(R, X_train, family=sm.families.Poisson())
                result = model.fit(maxiter=100, method='IRLS')
                params = result.params
                print(f" {_time.time()-t1:.1f}s (AIC: {result.aic:.1f})")
            
            baseline = params[0]
            k_pca = params[1:n_components + 1]
            h_hist = params[n_components + 1:]
            k_stim = pca.components_.T @ k_pca
            
            print(f"      baseline={baseline:.4f}, |k|={np.linalg.norm(k_stim):.4f}, |h|={np.linalg.norm(h_hist):.4f}")
            
            # 预测
            hist_test = np.zeros((d["Tt"], n_hist))
            Rt_test = d["Rt"]
            for lag in range(1, n_hist + 1):
                hist_test[lag:, lag - 1] = Rt_test[:-lag]
            
            X_test = np.hstack([np.ones((d["Tt"], 1)), St_pca, hist_test])
            linear_test = X_test @ params
            Rt_glm = np.exp(np.clip(linear_test, -20, 20))
            Rt_glm = np.clip(Rt_glm, 1e-10, None)
            
            glm_results[key] = {
                "k_stim": k_stim, "h_hist": h_hist, "baseline": baseline,
                "params": params, "Rt_glm": Rt_glm, "pca": pca,
            }
            print(f"    ✓ GLM 完成 ({_time.time()-t0:.1f}s)")
            
        except Exception as e:
            print(f"    ⚠️ 失败: {e}")
            z_test = d["St"] @ sta
            Rt_glm = np.exp(z_test * 0.1) * R.mean()
            Rt_glm = np.clip(Rt_glm, 1e-10, None)
            glm_results[key] = {
                "k_stim": sta, "h_hist": np.zeros(n_hist),
                "baseline": np.log(max(R.mean(), 1e-10)), "Rt_glm": Rt_glm,
            }
    
    print("\n  ✓ 阶段五完成: GLM 模型就绪")
    return glm_results


# ============================================================================
# 阶段六: 预测与验证
# ============================================================================

def compute_predictions(data, sta_results, stc_results, mne_results):
    """
    阶段六: 模型预测与验证。
    对应 MATLAB: script_108 + script_109
    """
    print("\n" + "=" * 70)
    print("  阶段六: 预测与验证")
    print("=" * 70)
    
    pred_results = {}
    
    stim_length = "short3"  # 使用 short3 配置
    
    LL = np.zeros((4, N_JK))  # STA, STC, MNE, null
    
    for iJK in range(1, N_JK + 1):
        key = (stim_length, iJK)
        d = data[key]
        St, Rt = d["St"], d["Rt"]
        
        # --- STA 预测 ---
        sta_r = sta_results[key]
        sta = sta_r["sta"]
        z_sta = St @ sta
        Rt_sta = np.maximum(sta_r["interp_func"](z_sta), 0) + 1e-8
        Rt_sta = Rt_sta * (d["R"].sum() / (Rt_sta.sum() * sta_r["norm_factor"]))
        
        # --- STC 预测 ---
        stc_r = stc_results[key]
        f_sub = stc_r["f_subspace"]
        zt = St @ f_sub
        
        if stc_r["kmodel"] == 1:
            Rt_stc = np.maximum(stc_r["interp_func"](zt.ravel()), 0) + 1e-8
        elif stc_r["kmodel"] == 2 and stc_r["interp_func"] is not None:
            Rt_stc = np.maximum(stc_r["interp_func"](zt), 0) + 1e-8
        else:
            Rt_stc = Rt_sta.copy()  # fallback
        
        Rt_stc = Rt_stc * stc_r["norm_factor"]
        
        # --- MNE 预测 ---
        if key in mne_results:
            mne_r = mne_results[key]
            A, H, J = mne_r["A"], mne_r["H"], mne_r["J"]
            linear = A + St @ H + np.sum(St * (St @ J), axis=1)
            Rt_mne = sigmoid(linear)
        else:
            Rt_mne = Rt_sta.copy()
        
        # --- 对数似然 (Equation 48, 49) ---
        # 确保预测值为正
        Rt_sta_safe = np.clip(Rt_sta, 1e-10, None)
        Rt_stc_safe = np.clip(Rt_stc, 1e-10, None)
        Rt_mne_safe = np.clip(Rt_mne, 1e-10, None)
        mean_Rt = np.clip(Rt.mean(), 1e-10, None)
        
        LL[0, iJK - 1] = np.mean(Rt * np.log(Rt_sta_safe) - Rt_sta_safe)  # STA
        LL[1, iJK - 1] = np.mean(Rt * np.log(Rt_stc_safe) - Rt_stc_safe)  # STC
        LL[2, iJK - 1] = np.mean(Rt * np.log(Rt_mne_safe) - Rt_mne_safe)  # MNE
        LL[3, iJK - 1] = np.mean(Rt * np.log(mean_Rt) - mean_Rt)           # null
        
        # Pearson 相关系数
        corr_sta = np.corrcoef(Rt, Rt_sta)[0, 1] if Rt.std() > 0 and Rt_sta.std() > 0 else 0
        corr_stc = np.corrcoef(Rt, Rt_stc)[0, 1] if Rt.std() > 0 and Rt_stc.std() > 0 else 0
        corr_mne = np.corrcoef(Rt, Rt_mne)[0, 1] if Rt.std() > 0 and Rt_mne.std() > 0 else 0
        
        pred_results[key] = {
            "Rt": Rt, "Rt_sta": Rt_sta, "Rt_stc": Rt_stc, "Rt_mne": Rt_mne,
            "corr_sta": corr_sta, "corr_stc": corr_stc, "corr_mne": corr_mne,
        }
        
        print(f"  JK{iJK}: corr(STA)={corr_sta:.4f}, corr(STC)={corr_stc:.4f}, corr(MNE)={corr_mne:.4f}")
    
    # 对数似然差 (relative to null)
    dLL = LL[:3] - LL[3:4]
    print(f"\n  平均对数似然差 (vs null):")
    print(f"    STA: {dLL[0].mean():.6f} ± {dLL[0].std():.6f}")
    print(f"    STC: {dLL[1].mean():.6f} ± {dLL[1].std():.6f}")
    print(f"    MNE: {dLL[2].mean():.6f} ± {dLL[2].std():.6f}")
    
    pred_results["LL"] = LL
    pred_results["dLL"] = dLL
    
    print("\n  ✓ 阶段六完成: 预测与验证就绪")
    return pred_results


# ============================================================================
# 阶段七: 生成图表
# ============================================================================

def generate_figures(data, sta_results, stc_results, mne_results, pred_results):
    """
    阶段七: 生成论文图表。
    对应 MATLAB: makefig_101 ~ makefig_106
    """
    print("\n" + "=" * 70)
    print("  阶段七: 生成图表")
    print("=" * 70)
    
    stim_length = "short3"
    iJK = 1
    key = (stim_length, iJK)
    
    # ---- Figure 1: 刺激 PCA (makefig_101) ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    d = data[key]
    S = d["S"]
    NX, NT = d["NX"], d["NT"]
    
    # Panel A: 刺激帧示例
    n_show = min(5, S.shape[0])
    for i in range(n_show):
        axes[0].plot(S[i, :NX] + i * 3, "k-", linewidth=0.5)
    axes[0].set_title("Stimulus Frames (first spatial dim)")
    axes[0].set_xlabel("Pixel")
    axes[0].set_ylabel("Intensity (offset)")
    
    # Panel B: 刺激协方差特征值谱
    cov_S = np.cov(S, rowvar=False)
    eigvals_S = np.linalg.eigvalsh(cov_S)[::-1]
    
    # Marchenko-Pastur 边界
    gamma = d["N"] / d["T"]
    mp_plus = (1 + np.sqrt(gamma)) ** 2
    mp_minus = (1 - np.sqrt(gamma)) ** 2
    
    axes[1].plot(eigvals_S, "k.-", markersize=3)
    axes[1].axhline(mp_plus, color="r", linestyle="--", label=f"MP+ = {mp_plus:.2f}")
    axes[1].axhline(mp_minus, color="b", linestyle="--", label=f"MP- = {mp_minus:.2f}")
    axes[1].set_title("Stimulus Eigenvalue Spectrum")
    axes[1].set_xlabel("Component")
    axes[1].set_ylabel("Eigenvalue")
    axes[1].legend()
    
    # Panel C: 前几个主成分
    _, eigvecs_S = np.linalg.eigh(cov_S)
    eigvecs_S = eigvecs_S[:, ::-1]
    n_pc = min(6, d["N"])
    pc_img = eigvecs_S[:NX, :n_pc]
    axes[2].imshow(pc_img.T, aspect="auto", cmap="RdBu_r", interpolation="nearest")
    axes[2].set_title(f"Top {n_pc} PCs (spatial)")
    axes[2].set_xlabel("Pixel")
    axes[2].set_ylabel("PC index")
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig_101_stimulus_pca.png", dpi=150)
    plt.close()
    print("  ✓ fig_101_stimulus_pca.png")
    
    # ---- Figure 2: STA (makefig_102) ----
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    sta_r = sta_results[key]
    sta = sta_r["sta"]
    
    # Panel A: STA 空间滤波器
    nv = int(np.sqrt(NX))
    if NT > 1:
        sta_img = sta.reshape(NT, NX)
        for t in range(min(NT, 3)):
            axes[0, 0].plot(sta_img[t], label=f"t-{t}")
        axes[0, 0].legend()
    else:
        axes[0, 0].imshow(sta[:NX].reshape(nv, nv), cmap="RdBu_r")
    axes[0, 0].set_title("STA Filter")
    
    # Panel B: STA 作为图像
    if NX >= 4:
        sta_spatial = sta[:NX].reshape(nv, nv) if nv * nv == NX else sta[:NX].reshape(-1, 1)
        axes[0, 1].imshow(sta_spatial, cmap="RdBu_r", interpolation="nearest")
        axes[0, 1].set_title("STA (spatial, first frame)")
        axes[0, 1].axis("off")
    
    # Panel C: 投影分布
    axes[1, 0].bar(sta_r["ctr_plot"], sta_r["Pf_plot"], 
                   width=sta_r["ctr_plot"][1] - sta_r["ctr_plot"][0], alpha=0.7)
    axes[1, 0].set_title("P(s·STA)")
    axes[1, 0].set_xlabel("Projection")
    axes[1, 0].set_ylabel("Density")
    
    # Panel D: 非线性函数
    axes[1, 1].plot(sta_r["ctr"], sta_r["g"], "ko-", markersize=4)
    axes[1, 1].set_title("Nonlinearity g(z)")
    axes[1, 1].set_xlabel("Projection z")
    axes[1, 1].set_ylabel("P(spike|z)")
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig_102_sta.png", dpi=150)
    plt.close()
    print("  ✓ fig_102_sta.png")
    
    # ---- Figure 3: STC (makefig_103) ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    stc_r = stc_results[key]
    
    # Panel A: STC 特征值谱
    axes[0].plot(stc_r["edC"], "k.-", markersize=3)
    axes[0].axhline(stc_r["max_enull"], color="r", linestyle="--", label="Null bounds")
    axes[0].axhline(stc_r["min_enull"], color="r", linestyle="--")
    axes[0].set_title(f"STC Eigenvalue Spectrum ({stc_r['nsig']} significant)")
    axes[0].set_xlabel("Component")
    axes[0].set_ylabel("Eigenvalue")
    axes[0].legend()
    
    # Panel B: 零分布直方图
    axes[1].hist(stc_r["edC_null"], bins=50, density=True, alpha=0.7, color="gray")
    for e in stc_r["edC"]:
        axes[1].axvline(e, color="k", linewidth=0.5, alpha=0.3)
    axes[1].set_title("Null Eigenvalue Distribution")
    axes[1].set_xlabel("Eigenvalue")
    
    # Panel C: 显著特征方向 (如果有)
    if stc_r["nsig"] > 0 and stc_r["f_subspace"].shape[1] > 1:
        feat = stc_r["f_subspace"][:, 1]  # 第一个 STC 方向
        axes[2].plot(feat[:NX], "b-", label="STC_1 (spatial)")
        axes[2].plot(sta[:NX], "r--", label="STA (spatial)")
        axes[2].legend()
    axes[2].set_title("STC Feature vs STA")
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig_103_stc.png", dpi=150)
    plt.close()
    print("  ✓ fig_103_stc.png")
    
    # ---- Figure 4: MNE (makefig_104) ----
    if key in mne_results:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        mne_r = mne_results[key]
        
        # Panel A: MNE 线性特征 H vs STA
        axes[0].plot(mne_r["H"][:NX] / np.linalg.norm(mne_r["H"]), "b-", label="MNE H")
        axes[0].plot(sta[:NX], "r--", label="STA")
        axes[0].set_title("MNE Linear Feature vs STA")
        axes[0].legend()
        
        # Panel B: J 特征值谱
        axes[1].plot(mne_r["eJ"], "k.-", markersize=3)
        axes[1].axhline(mne_r["max_enull"], color="r", linestyle="--", label="Null bounds")
        axes[1].axhline(mne_r["min_enull"], color="r", linestyle="--")
        axes[1].set_title(f"MNE J Eigenvalues ({mne_r['nsig']} significant)")
        axes[1].set_xlabel("Component")
        axes[1].legend()
        
        # Panel C: 零分布
        axes[2].hist(mne_r["eJ_null"], bins=50, density=True, alpha=0.7, color="gray")
        for e in mne_r["eJ"]:
            axes[2].axvline(e, color="k", linewidth=0.5, alpha=0.3)
        axes[2].set_title("MNE Null Eigenvalue Distribution")
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "fig_104_mne.png", dpi=150)
        plt.close()
        print("  ✓ fig_104_mne.png")
    
    # ---- Figure 5: 预测对比 (makefig_106) ----
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    pr = pred_results[key]
    t_show = min(500, len(pr["Rt"]))
    t = np.arange(t_show) / 30.0  # 转换为秒 (30Hz)
    
    # Panel A: PSTH 对比
    axes[0].plot(t, pr["Rt"][:t_show], "k-", linewidth=1.5, label="Actual", alpha=0.7)
    axes[0].plot(t, pr["Rt_sta"][:t_show], "b-", linewidth=1, label=f"STA (r={pr['corr_sta']:.3f})", alpha=0.7)
    axes[0].plot(t, pr["Rt_stc"][:t_show], "g-", linewidth=1, label=f"STC (r={pr['corr_stc']:.3f})", alpha=0.7)
    axes[0].plot(t, pr["Rt_mne"][:t_show], "r-", linewidth=1, label=f"MNE (r={pr['corr_mne']:.3f})", alpha=0.7)
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Firing rate")
    axes[0].set_title("Predicted vs Actual Response")
    axes[0].legend()
    
    # Panel B: 对数似然柱状图
    dLL = pred_results["dLL"]
    model_names = ["STA", "STC", "MNE"]
    colors = ["#1f77b4", "#2ca02c", "#d62728"]
    means = dLL.mean(axis=1)
    stds = dLL.std(axis=1)
    
    axes[1].bar(model_names, means, yerr=stds, color=colors, capsize=5)
    axes[1].set_ylabel("ΔLog-Likelihood (vs null)")
    axes[1].set_title("Model Comparison: Log-Likelihood")
    axes[1].axhline(0, color="k", linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig_106_prediction_validation.png", dpi=150)
    plt.close()
    print("  ✓ fig_106_prediction_validation.png")
    
    # ---- 相关系数汇总图 ----
    fig, ax = plt.subplots(figsize=(8, 5))
    
    corrs_sta = [pred_results[(stim_length, iJK)]["corr_sta"] for iJK in range(1, N_JK + 1)]
    corrs_stc = [pred_results[(stim_length, iJK)]["corr_stc"] for iJK in range(1, N_JK + 1)]
    corrs_mne = [pred_results[(stim_length, iJK)]["corr_mne"] for iJK in range(1, N_JK + 1)]
    
    x = np.arange(3)
    means = [np.mean(corrs_sta), np.mean(corrs_stc), np.mean(corrs_mne)]
    stds = [np.std(corrs_sta), np.std(corrs_stc), np.std(corrs_mne)]
    
    ax.bar(x, means, yerr=stds, color=colors, capsize=5, tick_label=model_names)
    ax.set_ylabel("Pearson Correlation")
    ax.set_title("Model Prediction Accuracy (Cell 3, short3)")
    ax.set_ylim(0, max(means) * 1.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig_correlation_summary.png", dpi=150)
    plt.close()
    print("  ✓ fig_correlation_summary.png")
    
    print("\n  ✓ 阶段七完成: 所有图表已保存到", OUTPUT_DIR)


# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    import time as _time
    
    print("\n  使用数据目录:", DATA_DIR)
    t_total_start = _time.time()
    timing = {}
    
    # 阶段一: 数据准备
    t_start = _time.time()
    data = build_stimulus_phase()
    timing["阶段一: 数据预处理"] = _time.time() - t_start
    print(f"  ⏱ 阶段一耗时: {timing['阶段一: 数据预处理']:.1f}s\n")
    
    # 阶段二: STA
    t_start = _time.time()
    sta_results = compute_sta_phase(data)
    timing["阶段二: STA"] = _time.time() - t_start
    print(f"  ⏱ 阶段二耗时: {timing['阶段二: STA']:.1f}s\n")
    
    # 阶段三: STC
    t_start = _time.time()
    stc_results = compute_stc_phase(data, sta_results)
    timing["阶段三: STC"] = _time.time() - t_start
    print(f"  ⏱ 阶段三耗时: {timing['阶段三: STC']:.1f}s\n")
    
    # 阶段四: MNE
    t_start = _time.time()
    mne_results = compute_mne_phase(data)
    timing["阶段四: MNE"] = _time.time() - t_start
    print(f"  ⏱ 阶段四耗时: {timing['阶段四: MNE']:.1f}s\n")
    
    # 阶段五: GLM
    t_start = _time.time()
    glm_results = compute_glm_phase(data, sta_results)
    timing["阶段五: GLM"] = _time.time() - t_start
    print(f"  ⏱ 阶段五耗时: {timing['阶段五: GLM']:.1f}s\n")
    
    # 阶段六: 预测与验证
    t_start = _time.time()
    pred_results = compute_predictions(data, sta_results, stc_results, mne_results)
    timing["阶段六: 预测验证"] = _time.time() - t_start
    print(f"  ⏱ 阶段六耗时: {timing['阶段六: 预测验证']:.1f}s\n")
    
    # 阶段七: 生成图表
    t_start = _time.time()
    generate_figures(data, sta_results, stc_results, mne_results, pred_results)
    timing["阶段七: 图表生成"] = _time.time() - t_start
    print(f"  ⏱ 阶段七耗时: {timing['阶段七: 图表生成']:.1f}s\n")
    
    total_time = _time.time() - t_total_start
    
    print("\n" + "=" * 70)
    print("  ✅ 所有阶段完成!")
    print("=" * 70)
    print(f"\n  📊 运行时间统计:")
    print(f"  {'─' * 40}")
    for stage, elapsed in timing.items():
        pct = elapsed / total_time * 100
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"  {stage:<20s} {elapsed:>7.1f}s  {bar} {pct:>5.1f}%")
    print(f"  {'─' * 40}")
    print(f"  {'总计':<20s} {total_time:>7.1f}s")
    
    # 保存时间统计为 JSON（供 dashboard 读取）
    import json
    timing_data = {
        "stages": timing,
        "total": total_time,
        "timestamp": datetime.now().isoformat() if 'datetime' in dir() else _time.strftime("%Y-%m-%dT%H:%M:%S"),
        "mps_available": _detect_mps(),
    }
    try:
        from datetime import datetime as dt
        timing_data["timestamp"] = dt.now().isoformat()
    except:
        pass
    with open(OUTPUT_DIR / "timing.json", "w") as f:
        json.dump(timing_data, f, indent=2, ensure_ascii=False)
    print(f"\n  ⏱ 时间统计已保存: {OUTPUT_DIR / 'timing.json'}")
    
    print(f"\n  输出图表保存在: {OUTPUT_DIR}")
    
    # 自动生成 dashboard
    print("\n  📊 生成 Dashboard...")
    import subprocess
    dashboard_script = Path(__file__).parent / "dashboard.py"
    if dashboard_script.exists():
        subprocess.run([sys.executable, str(dashboard_script)], cwd=str(PROJECT_ROOT))
        result_html = OUTPUT_DIR / "dashboard.html"
        if result_html.exists():
            print(f"  ✓ Dashboard 已生成: {result_html}")
            # 在 macOS 上自动打开
            try:
                subprocess.Popen(["open", str(result_html)])
                print("  ✓ 已在浏览器中打开 Dashboard")
            except:
                pass
    else:
        print(f"  ⚠️ dashboard.py 不存在: {dashboard_script}")
