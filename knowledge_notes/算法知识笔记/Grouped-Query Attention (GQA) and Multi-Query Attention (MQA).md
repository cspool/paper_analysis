## Grouped-Query Attention (GQA) and Multi-Query Attention (MQA)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Grouped-Query Attention (GQA, Ainslie et al., 2023) 和 Multi-Query Attention (MQA, Shazeer, 2019) 是 Transformer decoder 中减少 KV Cache 内存占用的注意力机制变体。在标准 Multi-Head Attention (MHA) 中，每个 query head 拥有独立的 K/V head（即 hq = hkv，如 32 query heads 对应 32 KV heads），每层需要 32 组独立的 K/V 投影矩阵和 KV cache 存储。MQA 将共享推向极致——所有 query heads 共享同一组 K/V head（hkv = 1），每层仅需 2 个 K/V 投影矩阵（而非 MHA 的 2×hq 个），KV cache 内存降至 MHA 的 1/hq。GQA 是折中方案——将 query heads 分为若干组，每组共享一组 K/V head。共享度由 gq = hq/hkv 度量：gq=1 即 MHA，gq>1 为 GQA，hkv=1 (gq=hq) 为 MQA。

现代 LLM 广泛采用 GQA：Llama-3 (hq=32, hkv=8, gq=4)、Qwen3 (hq:hkv=4:1)、Llama-2-70B (hq=64, hkv=8, gq=8)。关键 trade-off：MQA 最大化 memory saving 但约束 attention 表达能力→GQA 在 memory 和 quality 间取平衡。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**GQA Attention Computation（nh=32, nkv=8 为例）**：
```
// 输入: Q ∈ R^{1×32×d_h}, K_cache, V_cache ∈ R^{T×8×d_h}
// gq = 32/8 = 4 query heads per KV head

// Step 1: Query projection (全部 32 个 Q heads)
Q = X @ W_Q  // shape: [1, 32*d_h]

// Step 2: K/V projection — 仅需 8 个 heads
K_new = X @ W_K  // shape: [1, 8*d_h]
V_new = X @ W_V  // shape: [1, 8*d_h]

// Step 3: Grouped attention — 每组 4 Q heads 共享 1 K/V
for kv_idx in range(8):
    q_group = Q[:, kv_idx*4 : (kv_idx+1)*4]  // [1, 4, d_h]
    K = K_cache[:, kv_idx, :]                 // [T, d_h]
    V = V_cache[:, kv_idx, :]                 // [T, d_h]
    // QK^T: 4 个 queries 联合与同一 K 做 GEMM → 更高 arithmetic intensity
    scores = q_group @ K^T / sqrt(d_h)       // [1, 4, T]
    attn = softmax(scores)                    // [1, 4, T]
    out[kv_idx*4:(kv_idx+1)*4] = attn @ V   // [1, 4, d_h]

// Step 4: Output projection
output = concat(all_outputs) @ W_O  // [1, 32*d_h]
```

**KV Cache 内存对比**（BF16，T=128K，L=36，d_h=64）：
| 变体 | hkv | KV Cache 大小 | 相对 MHA |
|------|-----|--------------|---------|
| MHA | 32 | 2×36×128K×64×32 ≈ 18.9GB | 1× |
| GQA (Llama-3) | 8 | 2×36×128K×64×8 ≈ 4.7GB | 0.25× |
| MQA | 1 | 2×36×128K×64×1 ≈ 0.59GB | 0.031× |

**论文发现（Cost-Optimal GQA）**：长上下文下进一步减少 head 数可以显著降低成本。T=128K 时，H=(8,1)（退化为 MQA）比 Llama-3 GQA H=(32,8) 的 KV cache 减少 87.5%，attention FLOPs 减少 75%，同时通过增大模型 N（1.8B vs 1.2B）补偿 loss。

术语一般如何实现？如何使用？

GQA/MQA 在训练阶段通过修改 attention layer 的 K/V projection 实现——将独立的 hq 个 K/V projection 矩阵合并/复用为 hkv 个。从 MHA checkpoint 转换为 GQA 可通过 mean pooling 已有 K/V heads 或 up-training。推理时无需特殊改动（KV cache 自动减少），与 FlashAttention、KV cache quantization、PagedAttention 等正交优化叠加。代码开源：https://github.com/THUNLP/cost-optimal-gqa。

**TransMLA 的理论贡献**：在 Appendix A 中严格证明了 GQA < MLA < MQA 的表达能力层级（相同 KV cache 大小下）。GQA 可表示为 MLA 的特例（W^{UK} 必须是 block-selector 稀疏矩阵，仅能产生 g 个独立 key/value 重复 h/g 次），而 MLA 的 dense W^{UK} 允许任意跨 head 混合，拥有严格更强的表达能力。这为从 GQA 迁移到 MLA 提供了理论基础。

涉及论文标题：
- Cost-Optimal Grouped-Query Attention for Long-Context LLMs
- GTA__Grouped-head_latenT_Attention
- KV-Compress__Paged_KV-Cache_Compression_with_Variable_Compression_Rates_per_Attention_Head
- KV-Compress__Paged_KV-Cache_Compression_with_Variable_Compression_Rates_per_Attention_Head
- LogQuant: Log-Distributed 2-Bit Quantization of KV Cache with Superior Accuracy Preservation
- Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention
- Q-Filters: Leveraging QK Geometry for Efficient KV Cache Compression
- TransMLA: Multi-Head Latent Attention Is All You Need
- ZSMerge: Zero-Shot KV Cache Compression for Memory-Efficient Long-Context LLMs

**Q-Filters 论文中的 GQA 处理**：Q-Filters 对 GQA 的处理方式为——对每组共享同一 KV head 的 Query heads，将其 Q-Filters（即各 Query head 的 SVD 第一右奇异向量）取平均，得到该 KV head 的统一 Q-Filter。推理时，用该平均 Q-Filter 对共享的 Key 向量做投影评估。论文验证该方法在 Llama-3.1-8B/70B（GQA, gq=4）上有效。

---
