## Load-Balanced Attention Scheduling (Algorithm 1)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Load-Balanced Attention Scheduling 是 FlashInfer 提出的动态调度框架，用于解决 LLM serving 中 variable-length 序列 batch 的 attention 计算负载不均问题（wave quantization：处理短 KV 的 CTA 完成后 idle，等待处理长 KV 的 CTA）。调度器受 Stream-K (Osama et al., 2023) 启发，但设计为 deterministic（避免 atomic aggregation 引入非确定性输出，以满足 LLM serving 的 reproducibility 要求）。

核心思想：将 attention 计算的调度从 kernel 内部解耦到 runtime——compile-time 选择 tile sizes，runtime 根据实际 sequence length 信息动态分配 CTA workload。采用 persistent kernel 设计：kernel 以固定 grid size 启动（兼容 CUDAGraph），各 CTA 从 CPU 生成的 work queue 中消费 KV chunks。长 KV sequences 被 split 为多个 chunks，短 KV sequences 的 chunks 填充调度空隙，实现 SM 间负载均衡。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Algorithm 1: FlashInfer's Balanced Scheduling

```
// ===== Input =====
// {l_qo(i), l_kv(i)} for i = 1..batch_size  // query/KV lengths
// T_q: query tile size (compile-time selected)

// ===== Cost Function =====
// cost(l_q, l_kv) = α·l_q + β·l_kv  // α, β hyperparameters (default=1)

// Step 1: Compute max KV chunk size L_kv
// total_work = Σ_i ⌈l_qo(i)/T_q⌉ · l_kv(i)
// L_kv = total_work / #CTA  // target workload per CTA

// Step 2: Split query tiles into KV chunks
W = []  // work queue: list of (chunk_id, kv_length)
for i in range(batch_size):
    num_q_tiles = ceil(l_qo(i) / T_q)
    for each q_tile in range(num_q_tiles):
        // Split this query tile's KV into chunks ≤ L_kv
        remaining = l_kv(i)
        kv_start = 0
        while remaining > 0:
            chunk_len = min(L_kv, remaining)
            W.append((chunk_id, chunk_len))
            remaining -= chunk_len
            kv_start += chunk_len

// Step 3: Sort work chunks descending by length
W.sort(key=lambda x: x[1], reverse=True)

// Step 4: Greedy min-cost assignment
Q = MinPriorityQueue()  // (cta_id, current_cost)
for cta_id in range(num_CTA):
    Q.push((cta_id, 0))

for (chunk_id, kv_len) in W:
    cta_id, current_cost = Q.pop_min()
    new_cost = current_cost + cost(T_q, kv_len)
    assign chunk_id to CTA cta_id
    Q.push((cta_id, new_cost))

// ===== Output =====
// Plan info:
//   - CTA work queues: [(chunk_id, query_range, kv_range), ...] per CTA
//   - Partial→Final mapping: which partial outputs merge to which final positions
```

Plan info 的 life cycle：
1. **CPU planning**（`attn.plan(seqlen_info)`）：每 generation step 在 CPU 上运行 Algorithm 1，生成 plan info（CTA work queues + index mapping）
2. **Async copy to GPU**：plan info 通过 `cudaMemcpyAsync` 拷贝到 GPU workspace buffer 的固定 offset 区域
3. **GPU persistent kernel**（`g.replay()` via CUDAGraph）：各 CTA 读取自己的 work queue section → 处理分配的 KV chunks → 输出 partial attention states → contraction kernel 合并 partial states
4. **Reuse across layers**：同一 generation step 内所有 decode attention layers 可复用相同 plan info（sequence lengths 相同）

CUDAGraph 兼容性保证：
- Persistent kernel grid size 编译时固定（不变）✓
- Workspace buffer 各 section (partial O, plan info) 分配在固定 offset，指针不变 ✓
- Plan info **内容**（chunk assignments）每 step 变化，但指针不变 —— CUDAGraph 仅 capture kernel launch parameters，不 capture data ✓
- Plan function 在 CPU 上执行，不在 CUDAGraph 内 ✓

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

FlashInfer 的 load-balanced scheduler 实现：
- CPU 端用 C++ 实现 Algorithm 1（轻量级，overhead < 1ms per step，被 decoding loop 摊销）
- GPU 端 CUDA persistent kernel 从 workspace buffer 读取 plan info → 根据分配处理 KV chunks → 输出 partial AttentionState (O_partial, LSE_partial)
- Contraction kernel（可合并入同一 persistent kernel）执行 ⊕ composition 合并 partial states
- 与 Stream-K 的区别：FlashInfer 用 deterministic greedy assignment（保证 reproducibility），而非 atomic aggregation（Stream-K 的非确定性行为不适合 LLM serving 的确定性输出要求）
- 效果：uniform 和 skewed (Zipf) sequence length 分布下，FlashInfer decode/prefill kernel 的 bandwidth/FLOPs utilization 显著高于 FlashAttention（使用 static tile allocation）——Figure 8

涉及论文标题：
- FlashInfer Efficient and Customizable Attention Engine for LLM Inference Serving
