## Prefix-Aware Attention Kernel（前缀感知注意力内核）

术语是什么？通过联网搜索让回答具体和精准。
Prefix-Aware Attention Kernel是PAT论文提出的面向LLM decode阶段的GPU attention kernel实现范式。其核心思想是：在decode batch内识别跨请求的共享KV cache prefix，将共享同一prefix的多个query打包进同一个CTA执行，使共享KV blocks仅从GPU global memory加载一次，在CTA内shared memory中复用。这与传统的query-centric attention kernel（每query独立CTA，重复加载共享KV）和KV-centric kernel（固定tile+padding）形成对比。PAT的prefix-aware kernel遵循pack-forward-merge范式：pack阶段将vLLM block table转为prefix tree并基于memory-centric profit model生成CTA partition；forward阶段用multi-tile kernel和multi-stream执行；merge阶段用online softmax合并partial results。在A100上，prefix-aware design使PAT相对FlashAttention的KV cache traffic减少4.1-7.5×，attention latency平均降低53.5%。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Prefix-Aware Attention在PAT中的执行流程（以Conversation trace三层prefix为例）：

```
// --- Pack Stage: Prefix Tree Construction ---
Input: decode_batch block_table  // 每行是query的KV block IDs
For each query:
  Path = block_table[query]  // 如 [KV-0, KV-1, KV-3, KV-6]
  Insert Path into prefix_tree  // shared leading blocks → internal nodes

// internal node u: {l_u: shared_KV_length, s_u: num_sharing_queries}
// leaf: one query's full KV path

// --- Pack Stage: TreeHeuristic ---
Function PackTree(root):
  P = []
  For each child c of root:
    if 4 * c.size < root.length:  // Scheme 1: Split
      P += PackTree(c)  // child becomes independent CTA
    else:  // Scheme 2: Merge
      P += PackTree(c with root.blocks)  // merge parent blocks into child's CTA
      root.RemoveQueries(c.queries)
  P += PackCTA(root.remaining_queries, root.shared_blocks)
  return P

// --- Forward Stage: Per-CTA Execution ---
For each CTA in P:
  q = CTA.num_queries         // e.g., 3 queries share prefix
  kv_len = CTA.KV_length      // e.g., 4096 tokens
  
  // Tile Selector
  m = ceil_pow2(q)            // e.g., q=3 → m=4, q=20 → m=32
  n = SelectKVTile(kv_len)    // long KV → n=128, short KV → n=32
  
  // Multi-tile forward kernel execution
  Launch on CUDA stream for (m,n):
    // Loop over KV tiles with double buffering
    for kv_tile in range(0, kv_len, n):
      cp_async_load(K_tile[kv_tile:kv_tile+n] → shared_mem)
      cp_async_load(V_tile[kv_tile:kv_tile+n] → shared_mem)
      
      // QK^T: Q[m, head_dim] × K[n, head_dim]^T → S[m, n]
      // 同一个shared_mem K tile被CTA内所有q个query复用
      for each query i in CTA:
        S[i] = mma(Q[i], K_shared)  // Tensor Core MMA
      
      // Online softmax stats
      m_new = max(m_old, rowmax(S))
      l_new = exp(m_old - m_new) * l_old + rowsum(exp(S - m_new))
      
      // PV: P[m, n] × V[n, head_dim] → O[m, head_dim]
      for each query i in CTA:
        P[i] = exp(S[i] - m_new) / l_new
        O[i] += mma(P[i], V_shared)  // 同一个shared_mem V tile复用

    // Output partial results per query per head
    WriteToGlobalMem(partial_max, partial_lse, partial_O)

// --- Merge Stage ---
For each query q:
  Load all partial results for query q from different CTAs
  // Online softmax merge across CTAs
  m_global = max(all partial_max)
  l_global = sum(exp(partial_max[i] - m_global) * partial_lse[i])
  O_final = sum(exp(partial_max[i] - m_global) * partial_O[i]) / l_global
  WriteToGlobalMem(O_final)
```

关键设计决策：(1) pack阶段决定哪些queries共享KV加载——这是prefix-aware的核心，通过profit model权衡KV读节省与intermediate写开销；(2) multi-tile用round-up规则避免query维度padding（如q=20选m=32而非m=64）；(3) multi-stream让不同tile配置的CTAs并行执行。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
PAT的prefix-aware attention kernel以约3k行Cutlass/CuTe + C++实现。Multi-tile kernel基于CUTLASS/CuTe的MMA抽象，使用cp_async + double buffering做global→shared memory异步搬运。在vLLM中通过设置环境变量`VLLM_ATTENTION_BACKEND=PAT`启用。Prefix-aware设计依赖三个前提：(1) decode batch中存在跨请求共享prefix（如system prompt、RAG context、tool templates）；(2) GPU memory-bound场景（compute/memory ratio高），HBM bandwidth是瓶颈；(3) serving framework的block table提供logical KV block ID mapping。论文指出prefix-aware attention对batch中共享prefix比例敏感：prefix ratio越高收益越大，无共享prefix时收益显著缩小（仅剩multi-tile/multi-stream的较小优化）。开源实现：https://github.com/flashserve/PAT。

涉及论文标题：
- PAT: Accelerating LLM Decoding via Prefix-Aware Attention with Resource Efficient Multi-Tile Kernel

