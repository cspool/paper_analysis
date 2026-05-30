## Activation-Based Structured Mask for Quantization（基于激活的结构化量化掩码）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
基于激活的结构化量化掩码是 PTQ1.61 提出的用于极低位量化的显著权重识别方法。与 PB-LLM/BiLLM 的逐元素非结构化掩码（每权重 1-bit bitmap, shape mxn）不同，结构化掩码按权重矩阵的**行**（对应输入激活通道）标记显著通道，掩码形状为 mx1，额外存储开销从 ≥1-bit 降至 0.0002-bit。核心推导（Eq. 3-4）：量化误差 E ≤ Σ_i (|x_i| x Σ_j |w_{i,j}^q - w_{i,j}|)，其中 |x_i| 为第 i 通道输入激活幅值。激活幅值约为权重的 1000 倍（尤其 top-20% 通道），因此保留高幅值激活通道对应的权重行为 4-bit 可最大程度降低量化误差上界。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 对每个线性层 W in R^{n x m}:
X = forward_pass(X_calib, layer)            # [t, m]
ch_mag = ||X[:, i]|| for i in 1..m          # 每通道幅值 [m]
k = int(m * 0.2)
salient_idx = topk(ch_mag, k)               # top-20% 通道
mask = zeros(m); mask[salient_idx] = 1      # 一维掩码

for i in range(m):
    if mask[i]: W_q[:, i] = 4bit_quant(W[:, i])
    else:       W_q[:, i] = alpha_i * sign(W[:, i])
```
salient_ratio=20% 而非 30%：因为 30% 下位宽升至 1.91-bit，违反 sub 2-bit 定义。掩码基于激活幅值而非 Hessian（如 OWQ），因为 Hessian 近似在极低位下误差被放大。消融实验（Table 5）证实 OWQ Hessian 掩码替换后 LLaMA-7B PPL 从 12.50 崩溃至 22.11。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
掩码存储：一维 4096 bits (512 bytes) vs 非结构化 4096x4096 bits (2 MB) per layer，压缩 4096 倍。与 AWQ 区别：AWQ 用激活-权重关系做 per-channel scaling（grid search），没有掩码。与 OWQ 区别：OWQ 用 Hessian+Cholesky 做列选择保留 FP16，涉及多层近似。

涉及论文标题：
- PTQ1.61 Push the Real Limit of Extremely Low-Bit Post-Training Quantization
- AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration
- OWQ: Outlier-Aware Weight Quantization for Efficient Fine-Tuning and Inference of Large Language Models

---
