## PagedAttention (Virtual Memory/Paging for KV Cache Management)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

PagedAttention 是 vLLM（Kwon et al., 2023b）中引入的 KV cache 管理机制，灵感来源于操作系统中的虚拟内存和分页（paging）。它将 KV cache 划分为固定大小的 page blocks（而非为每个请求预分配连续的最大长度内存），动态按需分配小块 GPU 内存给 KV cache。当请求的生成长度增长时，可以分配新的 page 并链接到现有 page table，而非预先占用 `max_length` 的连续内存。这一设计解决了传统 KV cache 管理中的三个核心问题：(1) 内存碎片化——预分配方式导致大量未使用但被占用的内存（内部碎片）和无法分配大块连续内存（外部碎片）；(2) 内存超额分配——为最坏情况序列长度预留内存导致低 GPU 利用率；(3) KV cache 共享困难——prefix sharing 等优化在连续内存分配下难以实现。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。

**PagedAttention 在 LLM Serving 中的请求生命周期**：
```
1. 请求到达
   → Scheduler 分配初始 page blocks（e.g., 用于 prefill tokens）
   → Page table 记录 logical_token_pos → physical_page 的映射

2. Prefill 阶段
   → 所有 prompt tokens 的 KV cache 写入分配的 pages
   → 若 pages 不够，分配新 pages

3. Decode 阶段
   → 每生成一个新 token
   → 若当前 page 已满（page_size tokens 已写入）
       分配新 page → page table 更新
   → 新 token 的 K/V 写入当前 page 末尾

4. 请求完成（遇 EOS 或 max_tokens）
   → 释放所有 page blocks 回 free pool
```

**与 KV cache 压缩的不兼容点**：PagedAttention 假设同一 page 内所有 tensor 类型相同（均为 FP16），且 page size 固定。Window-based quantization（如 KIVI 保留最近 R token 为 FP16、其余为 INT4）需要同时管理两种 tensor 类型。Sparsity-based 方法（如 H2O）的 dynamic eviction 导致 KV cache length 不单调增长——与 page 只增不删假设冲突。

术语一般如何实现？如何使用？

实现于 vLLM（https://github.com/vllm-project/vllm）和 LMDeploy（https://github.com/InternLM/lmdeploy）。核心数据结构：block_table 映射 logical position → physical block index。Page size 通常为 16 或 32 tokens。论文 "Rethinking KV Cache Compression" 使用 LMDeploy v6.0.1 的 PagedAttention 评估压缩算法在真实 serving 框架上的吞吐，发现 PagedAttention+FlashAttention 下压缩算法的相对加速比大幅缩水。

涉及论文标题：
- Rethinking Key-Value Cache Compression Techniques for Large Language Model Serving
