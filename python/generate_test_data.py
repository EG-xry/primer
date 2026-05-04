#!/usr/bin/env python3
"""生成合成 whitenoise.raw 用于测试复现脚本。"""
import numpy as np
import os

Nx = 40
n_frames = 137145
outpath = os.path.join(os.path.dirname(__file__), "..", "RetinaData", "whitenoise.raw")

print(f"Generating synthetic whitenoise.raw ({n_frames * Nx * Nx / 1e6:.1f} MB)...")
np.random.seed(42)
with open(outpath, "wb") as f:
    chunk = 10000
    written = 0
    while written < n_frames:
        n = min(chunk, n_frames - written)
        f.write(np.random.randint(0, 256, n * Nx * Nx, dtype=np.uint8).tobytes())
        written += n
print(f"Done: {os.path.getsize(outpath)} bytes")
