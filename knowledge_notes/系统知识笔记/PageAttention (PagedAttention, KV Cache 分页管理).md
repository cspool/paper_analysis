## PageAttention (PagedAttention, KV Cache 分页管理)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

PageAttention（亦称 PagedAttention）是 vLLM（Kwon et al., SOSP 2023）提出的 KV cache 内存管理机制，将操作系统虚拟内存的分页思想应用于 LLM 推理的 KV cache 管理。核心思想：将每请求的 KV cache 划分为固定大小的 page（block，典型大小为 16 tokens），通过 page table 将逻辑 page 映射到非连续的物理 GPU 内存块。这消除了传统方案中因预分配连续内存导致的碎片化问题（内存利用率从 ~40% 提升至 90%+），并支持 memory sharing（通过 copy-on-write 在 beam search 和 parallel sampling 中共享相同 prefix 的 KV cache page）。

Quest 利用 PageAttention 作为其 query-aware sparse attention 的基础：(1) 以 page 为粒度维护 per-page min/max Key metadata；(2) criticality estimation 以 page 为单位打分（而非逐 token），大幅减少 metadata 量和 Top-K 选择开销；(3) approximate attention 通过 FlashInfer 的 PageAttention 接口实现 sparse page loading——传入 Top-K page indices 直接索引物理内存块。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。

**PageAttention 在 Quest 中的运转流程**：

```
// vLLM/FlashInfer 中的 PageAttention 数据结构
Page Table: 每个请求维护逻辑 page → 物理 block 的映射
  logical_page_0 → physical_block_17
  logical_page_1 → physical_block_3
  logical_page_2 → physical_block_42
  ...

// Quest 扩展：每 physical block 附加 metadata
Metadata[physical_block_id] = {
    M: [max(K[0][i], K[1][i], ..., K[15][i]) for i in 0..127]  // 128 values FP16
    m: [min(K[0][i], K[1][i], ..., K[15][i]) for i in 0..127]  // 128 values FP16
}

// Decode 阶段稀疏注意力:
// 1. Quest 遍历所有 valid pages (通过 page table)
// 2. 仅加载 metadata (256 FP16 per page)，计算 per-page criticality score
// 3. Top-K selection → K 个 page indices
// 4. FlashInfer PageAttention: 使用 page_table + selected page indices
//    仅加载选中 page 对应的 physical blocks 的 K,V
//    flashinfer.decode.page_attention(Q, K_cache, V_cache,
//                                     page_table, selected_page_indices)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

PageAttention 由 vLLM（https://github.com/vllm-project/vllm）和 FlashInfer（https://github.com/flashinfer-ai/flashinfer）实现。核心 API：(a) `append_kv_cache(page_table, new_K, new_V)`——将新 KV 写入下一个空闲 page；(b) `page_attention(Q, page_table, K_cache, V_cache)`——通过 page_table 间接索引加载 K、V 并执行 attention。Quest 在此基础上新增：(c) `update_page_metadata(page_id, K)`——更新 per-page min/max Key；(d) `estimate_criticality(Q, all_metadata)`——计算 per-page upper-bound score；(e) `page_attention_with_sparse_indices(Q, page_table, K_cache, V_cache, selected_pages)`——仅加载选中 page 的 sparse attention。开源：https://github.com/mit-han-lab/Quest。

涉及论文标题：
- Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference
- Efficient Memory Management for Large Language Model Serving with PagedAttention (vLLM, Kwon et al., SOSP 2023)
