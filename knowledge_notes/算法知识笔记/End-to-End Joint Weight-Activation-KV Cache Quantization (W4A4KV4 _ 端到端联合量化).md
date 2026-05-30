## End-to-End Joint Weight-Activation-KV Cache Quantization (W4A4KV4 / 端到端联合量化)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
End-to-End Joint Quantization (W4A4KV4) 是指将 LLM 的所有线性层权重、中间激活值、以及 KV Cache 全部量化为 4-bit 精度（INT4）的量化方案。与传统仅权重量化（W4A16）或 8-bit 激活量化（W8A8）不同，W4A4KV4 要求：(1) 权重以 INT4 存储并在矩阵乘法时反量化；(2) 激活值在每次前向传播时在线量化为 INT4（per-token 对称量化），矩阵乘法在 INT4×INT4 精度下进行；(3) Key 和 Value 向量在存入缓存时量化为 INT4（asymmetric group-wise），解码时从缓存加载后反量化。QuaRot 是首个声称实现端到端 4-bit 量化的方法，其核心贡献在于通过随机 Hadamard 旋转从根源上消除所有三类张量中的离群值，使统一的对称/非对称 INT4 量化成为可能。LLAMA2-70B 在 W4A4KV4 下仅损失 0.47 WikiText-2 困惑度、保持 99% 零样本精度。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
QuaRot 中 W4A4KV4 的完整前向传播流程（单层 Transformer）：
```
# === 输入: FP16 X ∈ R^{T×d}, INT4 packed weights, INT4 KV cache ===

# 1. RMSNorm (FP32, 无scale)
X_norm = X / sqrt(mean(X^2) + eps)

# 2. Attention: Q/K/V projection (W4A4)
for W in [W_q, W_k, W_v]:   # W_q 等已离线融入随机Hadamard Q^T
    s_x = max(|X_norm|, dim=1) × 0.9 / 7   # per-token scale, clip=0.9
    X_q = round(clip(X_norm/s_x, -7, 7))    # INT4 activation
    Y_int = CUTLASS_INT4_GEMM(X_q, W_q)     # TensorCore INT4×INT4
    Y = (float(Y_int) * s_x[:, None] * s_w[None, :])  # dequant → FP16

# 3. RoPE + Post-RoPE Hadamard (在线 head-wise)
Q_h, K_h = RoPE(Q), RoPE(K)
Q_h = Q_h @ (I ⊗ H_{d_h})   # head-wise Walsh-Hadamard, O(d_h log d_h)
K_h = K_h @ (I ⊗ H_{d_h})

# 4. KV Cache 量化存储 (asymmetric group-wise, group=128)
for each group g in K_h:
    z_k = min(K_h[g]), s_k = (max(K_h[g]) - z_k) / 15
    K_q[g] = round(clip((K_h[g] - z_k)/s_k, 0, 15))   # INT4
# V 同理量化存储

# 5. Attention 计算 (FP16, 在线反量化KV)
scores = Q_h @ Dequant(K_q)^T / sqrt(d_h)
attn_out = softmax(scores) @ Dequant(V_q)

# 6. Hadamard heads (在线) + Out-projection (W4A4)
Z_h = attn_out @ (H_{n_h} ⊗ I)   # Kronecker Hadamard, reshape+WHT
Z_q = round(clip(Z_h/s_z, -7, 7))
O_int = CUTLASS_INT4_GEMM(Z_q, W_out_q)
O = dequant(O_int, s_z, s_wo)

# 7. FFN: Gate/Up projection (W4A4)
gate_q = round(clip(X_norm/s_x, -7, 7))
gate = SiLU(dequant(CUTLASS_INT4_GEMM(gate_q, W_gate_q)))
up = dequant(CUTLASS_INT4_GEMM(gate_q, W_up_q))

# 8. 在线 Hadamard (down-projection 前, FP16) + Down-projection (W4A4)
down_in = FastHadamard(gate * up)   # O(d log d) WHT
d_q = round(clip(down_in/s_d, -7, 7))
D_int = CUTLASS_INT4_GEMM(d_q, W_down_q)
D = dequant(D_int, s_d, s_wd)

# 9. Residual connection
X_out = X + O + D  # 所有 MatMul 均为 INT4×INT4
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现 W4A4KV4 需要：(1) 离线阶段：用计算不变性定理将旋转矩阵融入权重（RMSNorm scale 吸收 + 随机 Hadamard 融合），约 5 分钟（LLAMA2-70B on A100）；GPTQ 权重量化约 2 小时；(2) 在线推理：CUTLASS 提供 INT4 TensorCore GEMM kernel，FlashInfer 提供量化 KV Cache attention kernel，快速 Hadamard kernel 实现 O(d log d) 在线变换。加速比：LLAMA2-70B prefill 3.33× (batch=64, seq=2048, RTX 3090)，解码内存节省 3.89×。代码开源：https://github.com/spcl/QuaRot。

涉及论文标题：
- QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs

---
