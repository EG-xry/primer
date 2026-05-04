"""
Spike Train Primer — Python 复现
================================

神经元脉冲序列分析入门教程的 Python 实现。

模块:
    io             - 数据 I/O (读取 .raw, .isk, .mat 文件)
    preprocessing  - 数据预处理 (刺激构建、归一化、时间对齐)
    sta            - STA 脉冲触发平均
    stc            - STC 脉冲触发协方差
    mne_model      - MNE 最大噪声熵模型
    glm            - GLM 广义线性模型
    prediction     - 模型预测
    validation     - 模型验证与评估
    plotting       - 可视化工具
"""

__version__ = "0.1.0"
