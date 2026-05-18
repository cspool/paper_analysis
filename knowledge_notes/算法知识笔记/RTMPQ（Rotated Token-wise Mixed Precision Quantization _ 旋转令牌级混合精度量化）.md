## RTMPQ（Rotated Token-wise Mixed Precision Quantization / 旋转令牌级混合精度量化）

术语是什么？通过联网搜索让回答具体和精准。

RTMPQ（Rotated Token-wise Mixed Precision Quantization）是RoMeo提出的面向LLM 4-bit weight-activation量化的算法，通过"先旋转后混合精度"的两阶段策略处理双维度（channel+token）outlier。核心思路：(1) 用Hadamard rotation将channel-wise outliers平滑并迁移到token维度；(2) 对纯token-wise outliers用token-wise mixed precision（outlier tokens→INT8, normal tokens→INT4）。该算法使4-bit量化的LLM准确率显著优于仅处理单维度outlier的baseline（QuaRot、MixQ）。

从算法pipeline角度拆解术语：

RTMPQ完整算法流程（以Qwen3-8B Linear层推理为例）：

```
Algorithm: RTMPQ Forward Pass
Input:  Activation X [M, K] (FP16)
        Weight W [K, N] (offline rotated: W' = H^T·W)
        Outlier budgets: k_a (token outliers), k_w (weight outliers)
        
// === 离线预处理（Serving前完成） ===
1. W_rot = H^T · W           // Weight Hadamard rotation
2. max_w = reduce_max(|W_rot|, axis=1)  // per-column (对应per-channel)
3. O_W = topk_indices(max_w, k_w)
4. W_rot_Q = mixed_precision_quantize(W_rot, O_W)  // INT4+INT8

// === 在线推理 ===
5. X_rot = FWT(X)            // Fast Walsh-Hadamard Transform: O(MK log K)
6. max_x = reduce_max(|X_rot|, axis=1)  // per-token max
7. O_A = topk_indices(max_x, k_a)  // top-k outlier tokens (k_a ≈ 5% M)
8. X_rot_Q_int4 = INT4_quantize(X_rot)  // 全矩阵INT4
9. X_rot_Q_int8 = INT8_quantize(copy_outlier_tokens(X_rot, O_A))  // outlier buffer

// === Cross-Precision Multiplication ===
10. C_W4A4 = INT4_GEMM(X_rot_Q_int4, W_rot_Q_int4)     // 主体计算
11. C_W4A8 = INT4x8_GEMM(X_rot_Q_int4_normal, W_rot_Q_int8_outlier)
12. C_W8A4 = INT8x4_GEMM(X_rot_Q_int8_outlier, W_rot_Q_int4_normal)
13. C_W8A8 = INT8_GEMM(X_rot_Q_int8_outlier, W_rot_Q_int8_outlier)
14. C = dequantize_and_combine(C_W4A4, C_W4A8, C_W8A4, C_W8A8)

15. Post-mul Overwrite: C[O_A, :] = C_W8A*/C_W*A8结果覆盖对应行
```

RTMPQ的理论加速比（以m=n=4096, k_a=k_w=256为例）：
- P_INT4 = (4096-256)² / 4096² = 88%（纯INT4计算比例）
- S = 1 / (0.88/4 + 0.12/2) = 3.57× (vs FP16 baseline)

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- 实现方式：PyTorch nn.Module封装（可直接替换原始Linear），HadaCore提供FWT，CUTLASS提供INT4/INT8 GEMM kernel，Triton提供fused outlier detection+quantization kernel。
- Outlier budget配置：RoMeo使用5%作为默认值（k_a=0.05M, k_w=0.05N），实验显示outlier从0%到1.6%时perplexity改善最显著（Qwen3-8B: 0.40 PPL reduction），后续边际效益递减。
- Weight mixed precision的必要性：Hadamard rotation后的weight矩阵因H^T预乘amplify non-uniformity，也需要mixed precision处理（与activation对称，per-column outlier detection）。
- 精度优势：RoMeo在Qwen3-8B上PPL 10.97优于QuaRot 11.53（uniform INT4+rotation），Qwen3-14B上70.82 avg zero-shot accuracy优于QuaRot 70.04。
- 开源：https://github.com/thu-pacman/RoMeo

涉及论文标题：
- RoMeo: Mitigating Dual-dimensional Outliers with Rotated Mixed Precision Quantization

