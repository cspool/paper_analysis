## Pack-Forward-Merge Execution Paradigm（打包-转发-合并执行范式）

术语是什么？通过联网搜索让回答具体和精准。
Pack-Forward-Merge是PAT提出的面向decode attention的三阶段GPU kernel执行范式，直接针对memory-bound decode attention的两大瓶颈：(1) 共享prefix导致的redundant global memory accesses；(2) 动态query数和KV长度导致的resource inefficiency。Pack阶段将decode batch的block table转为prefix tree，用memory-centric profit model决定哪些queries应被打包进同一CTA（共享KV加载），生成CTA partition；Forward阶段为每个CTA选择最优tile配置并用multi-stream并行执行；Merge阶段用online softmax将同一query被拆分到多个CTA的partial results合并为最终输出。该范式区别于query-centric的one-query-per-CTA和KV-centric的fixed-tile packing，实现KV读复用+资源自适应。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Pack-Forward-Merge作为整体pipeline的执行流程：

```
// ====== Phase 1: Pack ======
Input: batch of queries + vLLM block_table
// block_table: each row = [block_id_0, block_id_1, ..., block_id_k]
// shared prefix → identical leading block_ids

// Step 1: Build prefix tree
prefix_tree = BuildPrefixTree(block_table)
// internal node u: {l_u: shared KV length, s_u: #sharing queries}
// leaf: one query with non-shared KV suffix

// Step 2: Memory-centric profit-driven packing
CTA_list = TreeHeuristic(root)
// Profit model for packing node u into a CTA:
//   Saving: (s_u - 1) * l_u * d  (减少s_u-1次shared KV加载)
//   Overhead: 8 * s_u * d  (FP32 partial intermediate写回+读回)
//   Profit ratio: l_u / 16 ≥ 1 (since KV block size ≥ 16)
// Inter-node: merge child c into parent u when 4*s_c > l_u
// Complexity: O(|V|+|E|) — linear in tree nodes+edges

// Step 3: Lazy update
if block_table unchanged since last step:
  reuse CTA_list from cache  // 跳过重建和调度

// ====== Phase 2: Forward ======
// Step 1: Tile selection per CTA
streams = {}  // map: (m,n) → CUDA stream
For each CTA in CTA_list:
  q = CTA.num_queries
  kv_len = CTA.KV_length
  (m, n) = TileSelector(q, kv_len)  // O(1) lookup
  CTA.tile_config = (m, n)
  streams[(m,n)].enqueue(CTA)

// Step 2: Long-KV split
mean_kv_len = mean(CTA.KV_length for CTA in CTA_list)
For each CTA in CTA_list:
  if CTA.KV_length > mean_kv_len:
    Split CTA into ceil(CTA.KV_length / mean_kv_len) sub-CTAs

// Step 3: Multi-stream parallel execution
For each (m,n) in streams:
  Launch kernel_{m,n} on CUDA stream_{(m,n)}
  // Different streams execute concurrently on GPU
  // Within each stream, CTAs execute sequentially

// Kernel_{m,n} internal:
For each CTA assigned to this stream:
  For kv_block in CTA.KV_blocks step by n:
    cp_async_load(K_tile, V_tile)  // async global→shared
    For each query in CTA:  // shared KV reuse within CTA
      QK^T [m, n]  // Tensor Core MMA
      Online softmax update (max, lse)
      PV [m, n]   // Tensor Core MMA
  Write partial results (max_i, lse_i, O_partial_i) to global memory

// ====== Phase 3: Merge ======
For each query q in batch:
  Gather all partial results for query q
  // Online softmax merge:
  m_final = max(m_1, m_2, ..., m_k)
  For each partial i:
    weight_i = exp(m_i - m_final) * lse_i
  l_final = sum(weight_i)
  O_final = sum(weight_i * O_partial_i) / l_final
  Write O_final to global memory
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
PAT将pack-forward-merge实现为vLLM的attention backend plugin。Pack scheduler在CPU上以异步线程运行，与pre-attention tasks（metadata preparation, QKV projection）重叠执行，平均scheduling latency比pre-attention task latency低42-50%，因此不增加端到端延迟。Forward和Merge kernel在GPU上以CUDA/CUTLASS实现。整个pipeline通过pybind11暴露给Python，约1.2k行Python glue code集成到vLLM v0.9.0。打包决策基于logical block IDs（而非physical KV cache data），复用vLLM的paged KV cache机制。开源：https://github.com/flashserve/PAT。

涉及论文标题：
- PAT: Accelerating LLM Decoding via Prefix-Aware Attention with Resource Efficient Multi-Tile Kernel

