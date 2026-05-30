## FlashInfer Attention Kernel（FlashInfer 注意力计算库）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

FlashInfer（https://github.com/flashinfer-ai/flashinfer）是一个高性能的 LLM 注意力计算库，为 LLM serving 和 training 场景提供优化的 CUDA/HIP attention kernel。与 FlashAttention 相比，FlashInfer 针对 serving 场景做了额外优化：(1) 支持 variable-length sequences（decode 阶段各请求的 KV cache 长度不同）；(2) 高效的 KV cache attention（将 KV cache 作为 persistent 数据，减少重复加载）；(3) 支持 speculative decoding 的 tree attention pattern；(4) 优化的 page table attention（兼容 PagedAttention 格式）。

在 MagicDec 论文中，FlashInfer 被用作 self-implemented backend 的 attention 引擎——在 prefill 阶段执行完整 dense attention，在 decode 阶段的 verify phase 对完整 KV cache 做 attention。FlashInfer 的高效注意力实现使 verification cost 降低，直接提升了 SD 加速比。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。

```
# FlashInfer 在 MagicDec verify phase 的使用

# 标准 attention: Q @ K^T → Softmax → @ V
# FlashInfer 优化: tiled attention + shared KV cache

# Verify phase kernel 调度（单次 forward，验证 γ 个候选 token）
# 输入: Q [B, γ+1, d_head], K_full [B, S_full, d_head], V_full [B, S_full, d_head]
#       page_table [B, max_num_pages] (PagedAttention format)

# FlashInfer kernel 伪代码:
for each tile_Q in range(0, γ+1, TILE_Q):     # Q 按 tile 分块
    q_tile = load Q[blockIdx.x, tile_Q, :]     # coaleased load from HBM → SRAM
    o_tile = zeros(TILE_Q, d_head)              # 输出累加器
    m_tile = -inf                                # softmax 最大值（online softmax）
    l_tile = 0                                   # softmax 归一化分母
    
    for each tile_KV in range(0, S_full, TILE_KV):
        k_tile = load K_full from page_table[blockIdx.x, tile_KV]  # paged loading
        v_tile = load V_full from page_table[blockIdx.x, tile_KV]
        
        # Compute QK^T
        s = q_tile @ k_tile^T                  # [TILE_Q, TILE_KV]
        s = s / sqrt(d_head)
        
        # Online softmax
        m_new = max(m_tile, row_max(s))
        l_new = exp(m_tile - m_new) * l_tile + row_sum(exp(s - m_new))
        o_tile = exp(m_tile - m_new) * o_tile + exp(s - m_new) @ v_tile
        m_tile, l_tile = m_new, l_new
    
    o_tile = o_tile / l_tile                    # 最终归一化
    store o_tile → HBM
    
# GPU 线程组织:
# blockIdx.x: batch element (0..B-1)
# blockIdx.y: Q tile index
# threadIdx: 内部分工（Q*KV 矩阵乘法的 warp tile）
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

FlashInfer 安装：`pip install flashinfer`（需要 CUDA 12.1+）。在 MagicDec 中使用方式——prefill: `flashinfer.prefill.single_prefill_with_kv_cache(q, k, v, ...)`；decode: `flashinfer.decode.single_decode_with_kv_cache(q, kv_cache, ...)`。在 SD verify phase 中，对 γ+1 个位置的 query 做 batch attention：`flashinfer.decode.batch_decode_with_kv_cache(q_all, kv_cache, ...)`。MagicDec 也使用了 FlashInfer 的 PagedAttention 接口处理 KV cache fragmentation。与 torch.compile 协同使用可进一步优化性能。

在 Quest 中，FlashInfer 被扩展了三个自定义 CUDA kernel：(a) Criticality estimation kernel——利用 per-page min/max Key metadata 与 Query 向量计算 upper-bound attention score；(b) Top-K filtering——调用 RAFT batched Top-K CUDA operator 选择关键 page；(c) Approximate attention——基于 PageAttention sparse page loading，仅对选中 page 执行 FlashAttention。RTX 4090 上 32K seq_len, 2048 token budget → 7.03× self-attention speedup vs FlashInfer full attention。

涉及论文标题：
- MagicDec: Breaking the Latency-Throughput Tradeoff for Long Context Generation with Speculative Decoding
- Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference
- XAttention: Block Sparse Attention with Antidiagonal Scoring
