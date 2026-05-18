## Grouped-Query Attention (GQA) and Multi-Query Attention (MQA)

术语是什么？通过联网搜索让回答具体和精准。

Grouped-Query Attention (GQA, Ainslie et al., 2023) 和 Multi-Query Attention (MQA, Shazeer, 2019) 是 Transformer decoder 中减少 KV cache 内存占用的注意力变体。标准 Multi-Head Attention (MHA) 中，每个 query head 有独立的 K/V head（hq = hkv）。MQA 极端化：所有 query heads 共享单组 K/V heads（hkv = 1, K/V 投影仅需 2 个矩阵而非 MHA 的 2×hq 个）。GQA 折中：将 query heads 分为若干组，每组共享一组 K/V head——共享度由 gq = hq/hkv 度量。gq=1 即 MHA, gq>1 为 GQA, hkv=1 (gq=hq) 为 MQA。GQA/MQA 降低 KV cache 容量至 MHA 的 1/gq，同时减少 decode 时 K/V head projection 的计算量。现代 LLM 广泛采用 GQA：LLaMA-3.1-8B/70B (hq=32, hkv=8, gq=4)、Qwen3 (hq/hkv=4:1)、DeepSeek-V3 使用 MLA（Multi-head Latent Attention, 进一步压缩的下一个演进）。关键 trade-off：MQA 最大化 memory saving 但约束 attention expressiveness→GQA 在 memory 和 quality 间取平衡，training from scratch 或 up-training (从 MHA checkpoint 转换) 均可。

从算法 pipeline 角度拆解术语：

```
// === GQA Attention Computation (decode step, gq=4, hq=32, hkv=8) ===
// 输入: Q ∈ R^{1×hq×d} (1 token × 32 query heads × head_dim)
//       K_cache, V_cache ∈ R^{L×hkv×d} (L tokens × 8 KV heads)

// Step 1: Query projection (所有32个Q heads并行)
Q = X × W_Q  // shape: [1, 32*d_head]

// Step 2: K/V projection — 仅需8个heads（vs MHA需要32个）
K_new = X × W_K  // shape: [1, 8*d_head]
V_new = X × W_V  // shape: [1, 8*d_head]

// Step 3: GQA attention computation
for each kv_head in range(8):  // 8个KV heads
    // 该KV head对应的4个query heads
    q_group_start = kv_head * 4;
    q_group = Q[q_group_start : q_group_start+4];  // shape: [4, d_head]
    
    // Step 3a: QK^T — 4个queries共享同一K
    // MHA: 每个Q head有独立K → 1×L GEMV × 32 heads
    // GQA: 4个Q × 同一K → 4×L GEMM × 8 groups → 更高arithmetic intensity
    scores = q_group × K_cache[:, kv_head, :]^T  // shape: [4, L]
    
    // Step 3b: Softmax (row-wise, 沿L维度)
    attn_weights = softmax(scores / sqrt(d_head))  // shape: [4, L]
    
    // Step 3c: PV — 4个attention weights共享同一V
    output[q_group_start:q_group_start+4] = attn_weights × V_cache[:, kv_head, :]
    // shape: [4, d_head]

// Step 4: Output projection
output = concat(all_outputs) × W_O  // [1, 32*d_head]
```

GQA 对系统性能的关键影响：从 MHA 到 GQA (gq=4)→K/V projection 计算量减至 1/4→K/V cache memory 减至 1/4→但 QK^T 的 M 维度从 1 增至 4（grouped queries 做联合 GEMM）→arithmetic intensity 提升→更利于 Tensor Cores。BitDecoding 在 RTX 4090 GQA 下 3× speedup vs QServe 仅 1.4×——QServe 的 CUDA Cores-only approach 无法利用 GQA 带来的 compute intensity 提升，而 BitDecoding 的 Tensor Cores 天然受益于更大的 M 维度。

术语一般如何实现？如何使用？

GQA/MQA 在 training 阶段通过修改 attention layer 的 K/V projection 实现——将独立的 hq 个 K/V projection 矩阵合并/复用为 hkv 个。从 MHA checkpoint 转换为 GQA 可通过 mean pooling 或 select-and-fine-tune 已有 K/V heads。Inference 使用不需要特殊改动（K/V cache 自动减少），但 kernel 实现可利用 grouping 优化（如 BitDecoding Query Transformation）。GQA/MQA 与 FlashAttention 等 fused attention kernels 兼容，且与 KV cache quantization、PagedAttention 等正交优化可叠加。

涉及论文标题：
- BitDecoding: Unlocking Tensor Cores for Long-Context LLMs with Low-Bit KV Cache

