## Token-aware Quantization Estimator (TQE)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Token-aware Quantization Estimator (TQE) 是 Q-VDiT (ICML 2025) 提出的面向视频 Diffusion Transformer 的量化误差补偿模块。其理论基础为 Theorem 3.2：量化误差 Δ = W − Q̂(W) 的信息熵 H(Δ) ≤ H(W)（因为 round-to-nearest decimal truncation 是 surjection），因此 Δ 可在更低秩空间估计。TQE 使用两组低维向量参数 α∈R^{d_in} 和 β∈R^{d_out}（共 d_in+d_out 参数，vs 原始权重的 d_in×d_out）进行 rank=1 低秩误差估计，并从 token 维度和 feature 维度正交地补偿量化误差。token 维度的补偿通过 frame-aware 缩放因子 M∈R^t（t 为帧数）实现：M_i = η_i/ω_i，其中 η_i 衡量第 i 帧的量化误差权重（基于量化前后相似度），ω_i 衡量 token 序列的显著度量。TQE 本质上是 LoRA 模块的一个 rank=1 特殊实例，可被 LoRunner Kernel 融合以消除额外延迟。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
在 Video DiT 的 PTQ pipeline 中，TQE 修改了每个 Linear 层的前向传播：
```
# 输入: X ∈ R^{n×d_in}, W ∈ R^{d_out×d_in}, n = s × t (s个空间token, t帧)
# 参数: α∈R^{d_in}, β∈R^{d_out}, M∈R^t (仅 d_in+d_out+t 额外参数)

# Standard quantized forward:
Y_std = Q̂(X) @ Q̂(W)^T   # 量化+反量化后的矩阵乘法

# TQE error compensation:
for frame i in [0..t-1]:
    f_i = i * s   # frame start index
    # Token-aware scaling on quantized activations
    Δ̂[f_i:f_i+s, :] = (M_i ⊙ Q̂(X)[f_i:f_i+s, :]) @ α   # ∈ R^{s×1}

# Final output with error compensation:
Y = Y_std + Δ̂ @ β^T   # ∈ R^{n×d_out}
```

M 的初始化（Eq. 9）：η_i = exp[1-ρ(X_i, Q̂(X)_i)] / Σ_v exp[1-ρ(X_v, Q̂(X)_v)]，ω_i = Σ_τ|X_{i,τ}| / Σ_v Σ_τ|X_{v,τ}|，M_i = η_i/ω_i。α 用 Kaiming init，β 用 zero init。校准训练时同时优化 TQE 参数和量化参数。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
TQE 在 Q-VDiT 开源代码（https://github.com/wlfeng0509/Q-VDiT）中实现，作为 Linear 层的 wrapper 模块。TQE 参数仅 (d_in+d_out+t) 个，相比原权重减少 ~(d_in×d_out)/(d_in+d_out) 倍的参数量。推理时通过 LoRunner Kernel（来自 SVDQuant）将 rank=1 低秩分支与量化 GEMM kernel 融合——down projection (X→Δ̂) 与量化 kernel 融合共享激活张量消除额外内存访问，up projection (Δ̂→output) 与量化计算 kernel 融合——kernel 调用次数减半，额外延迟<5%。TQE 对 W4A6/W3A8/W3A6 均有效，结合 TMD 后在 W3A6 下 Scene Consistency 从 12.04 提升到 23.40（+94%）。

涉及论文标题：
- Q-VDiT Towards Accurate Quantization and Distillation of Video-Generation Diffusion Transformers

---
