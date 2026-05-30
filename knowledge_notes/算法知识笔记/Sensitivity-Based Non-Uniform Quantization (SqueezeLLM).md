## Sensitivity-Based Non-Uniform Quantization (SqueezeLLM)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Sensitivity-Based Non-Uniform Quantization 是 SqueezeLLM 提出的 weight-only 后训练量化方法。核心思想是将量化问题转化为 weighted k-means clustering：优化目标不是最小化简单的 L2 reconstruction error `||W - W_Q||²`，而是最小化 Fisher-weighted error `Σ F_ii (w_i - Q(w_i))²`，其中 `F_ii` 是 Fisher 信息矩阵对角线（≈Hessian 对角），作为每个权重的 importance weight。这源于 Optimal Brain Damage (OBD) 框架：对 loss 做 Taylor 展开，假设模型已收敛（g≈0），`L(W_Q) ≈ L(W) + ½(W-W_Q)ᵀH(W-W_Q)`，用 Fisher 对角近似 Hessian 对角。结果是 k-means centroids 被"拉向"对最终 loss 更敏感的高 Fisher 值权重，在敏感区域提供更高的量化分辨率。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Step 1: 计算 Fisher 信息矩阵 (sensitivity)
# 对 calibration 数据集 D (10-100 samples)
F = zeros_like(W)  # per-weight Fisher diagonal
for each sample d in D:
    loss = cross_entropy(model(d), labels)
    g = backward(loss)  # gradient w.r.t. all weights
    F += g ⊙ g          # element-wise square
F /= |D|                # average over samples

# Step 2: Sensitivity-weighted k-means per output channel
# 目标: argmin_Q Σ_i F_ii * (w_i - Q(w_i))²
for each output channel c:
    k = 2^bit  # e.g., k=8 for 3-bit
    centroids = kmeans++_init(W[c,:], k)
    repeat until convergence:
        # E-step: assign each weight to nearest centroid
        for i in range(in_features):
            assignment[i] = argmin_j ||w_i - centroid_j||²
        # M-step: update centroids weighted by Fisher values
        for j in range(k):
            mask = (assignment == j)
            centroid_j = Σ_{i∈mask} F_ii * w_i / Σ_{i∈mask} F_ii
    LUT[c] = centroids  # FP16, k entries
    indices[c] = assignment  # b-bit per weight, packed
```

效果（LLaMA-7B 3-bit, C4 perplexity）：
- RTN (uniform): PPL 28.26
- Sensitivity-agnostic k-means: PPL 18.08
- Sensitivity-based k-means: PPL 7.75 (close to FP16 baseline 7.08)

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
SqueezeLLM 开源：https://github.com/SqueezeAILab/SqueezeLLM。Fisher 计算需一次完整 backward pass（A100: 7B=0.3min, 65B=2.5min），k-means 在 CPU 上执行（Xeon 48核: 7B=11min, 65B=80min）。关键权衡：(1) 仅需 10-100 校准样本即收敛（比 GPTQ/AWQ 的 128 更少）；(2) Fisher 计算的一次性峰值内存需求高（7B=33GB, 65B=292GB）；(3) 每 channel 需存储 k 个 FP16 centroid（如 3-bit: 8 FP16/channel），但 memory-bound 推理中 LUT dequant 额外计算可被内存带宽瓶颈掩盖。与传统 layer-wise objective（GPTQ: min ||WX - W_QX||²）相比，final-loss-based objective 在 PPL 上有约 0.3 的系统性优势（D.4 消融实验）。

涉及论文标题：
- SqueezeLLM Dense-and-Sparse Quantization
