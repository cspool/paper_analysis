## Outlier Features in LLM Activations (LLM 激活值中的离群特征)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Outlier Features（离群值特征/异常值通道）是指 LLM 中间激活张量 X ∈ R^{T×d} 中某些特定通道（列）的数值幅度系统性地远大于其他通道的现象——某些 channel index c 在所有 token 上的 |X_{t,c}| 比其他通道大 20-100 倍（Wei et al., 2022 首次系统记录）。离群值特征集中在约 0.1% 的通道中，但其存在使得激活值的对称均匀量化几乎不可能：离群通道的 scale 由最大值决定（s_x = max(|X|)/q_max），导致非离群通道的有效量化精度极低。离群值特征的出现与 LayerNorm/RMSNorm 机制、Transformer 注意力残差结构、以及训练时的优化动态有关——RMSNorm 将每个 token 的向量除以其 L2 范数，将总能量分散，但特定方向（通道）仍保持极大的权重在内积中。从信号处理角度看，离群值特征可理解为模型的"强响应通道"——某些方向的语义信息（如 token identity、位置）高度集中。QuaRot 通过随机 Hadamard 变换从根源消除离群值：X → XQ，其中 Q = H_d diag(s)，每个输出通道变为所有输入通道的 ±1 加权和，大值被扩散到所有方向（图 1 验证变换后激活值从长尾分布变为类高斯分布）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
离群通道识别的典型伪代码（Atom/SmoothQuant 的 baseline 方法）：
```
# X ∈ R^{T×d}: 一批激活值
# 计算每个通道的最大绝对值（跨所有token）
channel_max = max(|X|, dim=0)       # [d]
# 计算所有通道最大值的中位数
threshold = 5.0 * median(channel_max)  # 或 6.0，超参数
# 识别离群通道
outlier_mask = (channel_max > threshold)  # [d], bool
# 离群通道保持FP16，其余INT4量化
X_normal = X[:, ~outlier_mask]   # 量化到INT4
X_outlier = X[:, outlier_mask]    # 保持FP16
```

QuaRot 的离群值消除方法（根本不需要识别）：
```
# 离线：计算随机Hadamard Q = H_d diag(s), s_i∈{±1}
# 融入权重：W_gate ← Q^T W_gate, W_down ← W_down Q, 等
# 在线：X_rotated = X @ Q  # 但无需显式计算，因为已融入权重
#       X_rotated 的所有通道幅值均匀（图1右）
#       直接用 per-token symmetric INT4 quant: s_x=max(|X_rotated|)/7
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
传统方法处理离群值特征的三种策略：(1) 混合精度（Atom, LLM.int8()）——离群通道保持 FP16/INT8，非离群通道量化；(2) Per-channel scaling（SmoothQuant）——将量化难度通过等价变换从激活值迁移到权重；(3) 校准集搜索（OmniQuant）——用可学习参数优化 per-channel 变换。QuaRot 用 Hadamard 旋转首度从根源消除离群值，使策略 (1)-(3) 中复杂机制不再必要。局限性：当存在大量"massive outliers"的 "pivot tokens" 时（如 Sun et al. 2024 发现某些 token 在整个序列中都产生极高激活值），全局 Hadamard 变换的效果可能不完全。当模型维度非 2 的幂时，需要使用已知 Hadamard 矩阵的 Kronecker 分解（H_d = H_{2^n} ⊗ H_m）。

涉及论文标题：
- QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs
