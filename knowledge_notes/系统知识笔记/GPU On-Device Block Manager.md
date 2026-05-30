## GPU On-Device Block Manager

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

GPU On-Device Block Manager 是 KV-Compress 提出的将 vLLM 的 KV cache block 管理（分配、释放、计数）从 CPU 端移至 GPU 端的优化。原 vLLM 的 block manager（scheduler 的一部分）在 CPU 上运行，每个 scheduling step 需遍历所有 sequences 计算所需 blocks、查询 free block list、执行分配——这些操作用 Python/C++ loop 完成，runtime 随 block 数量线性增长。

KV-Compress 的 per-head per-layer block 布局使得 total block 数量为 l×H 倍（Llama-3.1-8B: 256 倍），若仍用 CPU 端调度，"scheduling loop 在部分情况下耗时超过 model forward pass"。GPU 端方案将所有 block 管理移至 device memory——block tables、context lengths、free/allocated tracking 均存储为 GPU tensors，利用 GPU 的 SIMT parallelism 并行完成所有 sequences 的 block 分配和释放。

从系统架构角度拆解术语：

**GPU Block Manager 运转流程**：
```
# 数据结构（均在 GPU device memory）
free_blocks: bool tensor [N]  # 1=free, 0=allocated
context_lengths: int tensor [B, l, H]  # per-sequence per-head token count
block_tables: int tensor [B, l, H, L_max/b]  # per-sequence per-head block indices

# Prefill 分配（parallel on GPU）
required_blocks = ceil(prompt_length / b)  # 标量，所有 head 相同
# GPU parallel prefix scan 找出 first required_blocks free blocks
alloc_indices = cumsum_prefix_scan(free_blocks)  # parallel on GPU
for each (s, m, h) in parallel:  # GPU threads
    T[s, m, h, :required_blocks] = alloc_indices[offset_s_m_h: ...]

# Decoding 分配（parallel on GPU）
for each (s, m, h) in parallel:
    if C[s, m, h] % b == 0:  # 需要新 block
        new_block = find_next_free(free_blocks)
        T[s, m, h, C[s, m, h] // b] = new_block
        C[s, m, h] += 1

# Preemption 释放（parallel on GPU）
for each preempted s:
    for each (m, h) in parallel:
        num_blocks_s_m_h = ceil(C[s, m, h] / b)
        free_blocks[T[s, m, h, :num_blocks_s_m_h]] = 1
```

术语一般如何实现？如何使用？

实现为 GPU kernel 或 PyTorch tensor 操作（cumsum, scatter, gather）。KV-Compress 在 vLLM v0.6.0 中实现。Block tables 和 context lengths 在每次 forward pass 前从 GPU block manager 直接传给 model runner（在 GPU 上，避免 CPU↔GPU 拷贝）。

适用场景：任何 block 数量极大的 paged-attention 系统（per-head per-layer、大 batch size、长 context）。代价是增加了 GPU memory 中 block manager metadata 的占用，但远小于 KV cache 本身。

涉及论文标题：
- KV-Compress__Paged_KV-Cache_Compression_with_Variable_Compression_Rates_per_Attention_Head

---
