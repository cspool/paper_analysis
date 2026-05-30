## CPU Chunked Attention Verification Kernel（CPU 分块注意力验证 Kernel）

术语是什么？通过联网搜索让回答具体和精准。
CPU Chunked Attention Verification Kernel 是 SpecMoEOff 为 speculative decoding 在 MoE offloading 场景下设计的 CPU 端 attention 算子。它处理 Q∈R^{n×d}, K∈R^{(l+n)×d}, V∈R^{(l+n)×d}（其中 n 为 draft tokens 数，l 为 prefix tokens 数）的 chunked attention 计算，是 speculative decoding verification 阶段的核心算子。

该 kernel 解决的关键问题：在 MoE offloading 场景下，target model 的 KV cache 全部存储在 CPU DRAM 中。若将 KV cache 传回 GPU 做 attention，会产生大量 CPU→GPU 传输开销；若在 CPU 上对每个 draft token 独立做 attention（GEMV），则需重复读取 KV cache n 次。CPU Chunked Attention Kernel 通过一次性读取 KV cache 并为所有 n 个 draft tokens 做 batch attention（GEMM），同时解决了两个问题。

从kernel调度角度拆解术语：
```
# CPU Chunked Attention Kernel 计算流程
输入: Q ∈ R^{n×d}  (n draft tokens queries)
      K ∈ R^{(l+n)×d}  (prefix + draft keys, from CPU DRAM KV cache)
      V ∈ R^{(l+n)×d}  (prefix + draft values, from CPU DRAM KV cache)
      M_draft ∈ {0,1}^{n×n}  (仅 draft-to-draft causal mask)
      # draft-to-prefix 全为 1, 不存储

# Step 1: Q@K^T via Intel MKL SGEMM
# [n, d] @ [d, l+n] → [n, l+n]
scores_full = mkl_sgemm(Q, K.T) / sqrt(d)

# Step 2: Apply mask (仅 draft 部分)
# scores_full[:, :l] 无需 mask（全 1）
scores_full[:, l:] += causal_mask(M_draft)  # M_draft: 下三角=0, 上三角=-inf

# Step 3: Softmax + Weighted Sum via Intel MKL SGEMM
attn_weights = softmax(scores_full, dim=-1)  # [n, l+n]
output = mkl_sgemm(attn_weights, V)          # [n, d]

# Batch 扩展: b 个 requests 各自独立并行
```

与 Baseline 方案的对比：

| 方案 | Q@K^T 次数 | KV Cache 读取 | CPU-GPU 传输 | Mask 内存 |
|------|-----------|-------------|------------|----------|
| GPU chunked attention | 1× (b×n GEMM) | 0 | KV cache 全量传输 | n×(l+n) |
| Naive CPU decode (per-token GEMV) | n× (b×1 GEMV) | n× 重复 | 0 | 无 |
| PyTorch CPU prefill | 1× | 重复计算 prefix | 0 | n×(l+n) |
| **SpecMoEOff CPU Chunked** | **1× (b×n GEMM)** | **1×** | **0** | **n×n** |

术语一般如何实现？如何使用？
基于 Intel oneAPI Math Kernel Library (MKL) 的 SGEMM 实现矩阵乘法，利用 CPU SIMD (AVX-512) 和 MIMD (multi-core) 能力。Mask 压缩：仅存储 n×n draft-to-draft 部分（draft-to-prefix 固定为 1，无需存储），内存从 O(n·(l+n)) 降至 O(n²)。在 SpecMoEOff 系统中，CPU Chunked Attention 随 draft length 增加逐渐成为 target model 的性能瓶颈（Table 3: 4.29s CPU Attention vs 3.53s GPU MoE），说明 CPU attention kernel 是系统性能的关键路径。

涉及论文标题：
- Accelerating Mixture-of-Experts Inference by Hiding Offloading Latency with Speculative Decoding

**BigMac 的 All-to-All 维度缩减（Jin et al., 2025）**：
BigMac 从算法/模型结构层面直接减少 All-to-All 通信量——通过 DCCA（descend-communicate-communicate-ascend）策略将 All-to-All 通信从 full hidden dimension h 移至压缩后的低维 r·h。通信量公式：$C_{BigMac} = 2 \times top\_k \times \frac{ep-1}{ep} \times b \times s \times (r \cdot h) = r \times C_{baseline}$。当 r=0.25 时，通信量减少 75%（如 GPT3-XL + 64 experts + ep=32: 1,488 GB → 372 GB）。该方法仅改变模型结构（projection 顺序 + expert 内部结构），不修改 All-to-All 通信原语，因此与 Tutel 的 2DH All-to-All、Lina 的 micro-op scheduling 等系统优化正交叠加。BigMac 在 Megatron 上训练加速 1.53-3.09×，在 Tutel 上加速 1.71-3.09×，在 DeepSpeed-Inference 上推理吞吐提升 1.62-3.11×。
