## Partially-Binarized Quantization（部分二值化量化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
部分二值化量化（Partially-Binarized Quantization）是 PB-LLM（Shang et al., 2023）提出的一种极端低比特 LLM 量化策略。核心思想：不是将所有权重二值化（全部 → ±1），而是检测并保留少量（5%-30%）"显著权重"（salient weights）在高位宽（如 INT8），仅将剩余的 90%+ 非显著权重二值化为 ±1。这与传统的均匀量化（所有权重使用相同位宽）和完全二值化（所有权重 → 1-bit）都不同——它是一种混合精度策略，利用 LLM 中显著权重的存在性来实现接近 1-bit 的平均位宽。存储开销：N_bit ≤ 1 × r_binary + 8 × (1 − r_binary) + 1（bitmap index），例如保留 10% salient 权重为 INT8 时等效位宽约 2.7-bit。PB-LLM 在 PTQ 和 QAT 两种框架下均实现了部分二值化：PTQ 下通过 PB-GPTQ（Hessian 引导迭代量化 + 补偿），QAT 下通过冻结显著权重 + 最优缩放因子（α* = ||w_F||_1/n）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以 LLaMA-7B 某 Linear 层权重 W ∈ R^{d_o × d_i} 为例：
```
# Step 1: Salient Weight Detection
# Magnitude criterion (QAT):
salient_indices = topk(|W|.flatten(), k=int(d_o * d_i * salient_fraction))
salient_mask[i,j] = 1 if (i,j) in salient_indices else 0

# Hessian criterion (PTQ):
H = 2 * X @ X.T  # calibration data Hessian
H_inv = cholesky_inverse(H)
saliency[i,j] = W[i,j]^2 / H_inv[j,j]^2  # from SparseGPT
salient_indices = topk(saliency.flatten(), k)

# Step 2: Partial Binarization
W_salient = W * salient_mask       # 保留 INT8
W_unsalient = W * (1 - salient_mask)  # 二值化

# QAT 前向传播:
α = mean(|W_unsalient|, dim=1)      # column-wise optimal scaling
W_binary = α.unsqueeze(1) * sign(W_unsalient)
y = W_salient @ x + W_binary @ x    # 混合精度矩阵乘法

# PTQ (PB-GPTQ):
for col in range(d_i):
    w_sal = W_salient[:, col]; w_unsal = W_unsalient[:, col]
    ŵ_unsal = α[col] * sign(w_unsal)       # 二值化
    ŵ_sal = MinMaxQuant(w_sal, bit=8)       # INT8 量化
    error = W[:, col] - Ŵ[:, col]
    W[:, col+1:] += error / H_inv[col,col] * H_inv[col, col+1:]  # Hessian 补偿
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源实现：https://github.com/hahnyuan/PB-LLM。关键参数：`--low_frac`（二值化比例）、`--salient_metric`（magnitude 或 hessian）、`--high_bit`（显著权重位宽）。QAT 训练配置：AdamW optimizer, lr=2e-5, cosine decay, 10K iterations, batch size=1 per GPU。训练数据使用 RedPajama-simple-1B。部分二值化矩阵推理时存储为混合格式：INT8 salient weights + packed binary unsalient weights + column-wise α scaling factors + bitmap index。推理时二进制权重通过 dequant（α × sign 值）展开为 FP16 后执行标准 GEMM。论文主要关注 memory 压缩而非 kernel 加速，但理论上二进制权重可将 FP 乘法替换为 XNOR+Bitcount 实现 64× 理论加速。

涉及论文标题：
- PB-LLM Partially Binarized Large Language Models

---
