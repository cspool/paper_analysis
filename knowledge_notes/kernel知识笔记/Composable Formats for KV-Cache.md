## Composable Formats for KV-Cache

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Composable Formats 是 FlashInfer 提出的多 BSR 矩阵分解 KV-cache 优化策略，受 SparseTIR (Ye et al., 2023) 启发。核心思想：不再用单一 BSR 矩阵（统一 block size）存储整个 KV-cache，而是利用 prior knowledge（如哪些 requests 共享 prefix）将 KV-cache 稀疏矩阵分解为多个不同 $(B_r, B_c)$ 的 BSR sub-matrices。各 sub-matrix 用不同的 AttentionWrapper（不同 tile sizes + block sizes → 编译为不同 CUDAGraphs），runtime 根据 KV-cache 配置选择最优组合。

动机：单一 block size 的 BSR 有内在 trade-off——大 $B_r$ 允许同 block 内 queries 共享 SMEM 中的 KV tile（high-bandwidth reuse），但增加 fragmentation（不在同一 block 的 requests 无法彼此访问 SMEM）；小 $B_r$ 减少 fragmentation 但失去 SMEM 复用。Composable formats 打破这一 trade-off：shared prefix 密集子矩阵用大 $B_r$ 存储和计算（高 SMEM 复用），unique suffix 稀疏子矩阵用小 $B_r$（per-query 独立访问，tolerate L2/global latency）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

以 parallel generation (n=4, Llama 3.1 8B) 为例：4 条 parallel generation 回复共享输入 prompt prefix 的 KV-cache，后续 suffix 各自不同。

```
// ===== Composable Format Decomposition =====
// Full KV-cache sparse matrix:
//   rows 0-3:  shared prefix (4 queries share same KV)
//   rows 4-7:  unique suffix query 0
//   rows 8-11: unique suffix query 1  ...

// Sub-matrix 1: Shared Prefix
//   B_r=3, B_c=1 (3 queries share KV page in SMEM)
//   对应 shared prefix 的 KV-cache pages
//   kv_indptr_1 = [0, num_prefix_pages, num_prefix_pages, ...]
//   kv_indices_1 = [page_0, page_1, ..., page_prefix-1] (重复)

// Sub-matrix 2: Unique Suffixes
//   B_r=1, B_c=1 (per-query processing)
//   对应各 unique suffix 的 KV-cache pages
//   kv_indptr_2 = [0, num_suffix_pages, 2*num_suffix_pages, ...]
//   kv_indices_2 = [per-query unique pages]

// ===== Compile-time =====
// 创建两个 AttentionWrapper（不同 CUDAGraph）
wrapper_prefix = AttentionWrapper(
    attn_spec, 
    task_info(B_r=3, T_q=3, ...),  // 大 block → SMEM 复用
    workspace
)
wrapper_suffix = AttentionWrapper(
    attn_spec, 
    task_info(B_r=1, T_q=1, ...),  // 小 block → per-query
    workspace
)

// ===== Runtime =====
// Shared prefix attention: 3 queries × same K/V tile in SMEM
//   → tensor core GEMM: Q(3, d) × K(l_prefix, d)^T
//   → 3× fewer global memory loads vs per-query processing
O_prefix, LSE_prefix = wrapper_prefix.run(Q_shared, KV_prefix)

// Unique suffix attention: per-query, via L2/global memory
for i in range(4):
    O_suffix[i], LSE_suffix[i] = wrapper_suffix.run(
        Q_suffix[i], KV_suffix[i])

// Merge: O_final = (O_prefix, LSE_prefix) ⊕ (O_suffix, LSE_suffix)
```

关键实现要点：
- **无数据移动**：KV-cache 数据不移动，仅需计算不同 sub-matrix 的 `kv_indptr` / `kv_indices` arrays（metadata-level 操作）。
- **多 CUDAGraph**：每种 composable format 配置编译为独立 CUDAGraph，runtime 根据 KV-cache 结构 select best graph。
- **性能提升来源**：shared prefix 部分 O(queries×prefix_len×d) 的 attention 计算从 per-query GEMV (CUDA core, low compute intensity) 升级为 batched GEMM (Tensor Core, high compute intensity)，同时减少 global memory traffic（3 queries 共享 1 份 KV load）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

FlashInfer composable formats 的实现：
- 在 FlashInfer Python API 层面，用户创建多个 `AttentionWrapper` 实例，分别指定不同 `task_info`（含 B_r 配置）
- 每个 wrapper 编译为独立 CUDAGraph（含对应的 JIT-compiled kernel + plan info）
- 集成入 MLC-Engine prefix-caching：framework 识别 shared prefix → 创建 composable format wrappers → 选择最优 CUDAGraph
- 论文实验显示：parallel generation (n=4-32) 下 ITL 降低 13-17%，TTFT 降低 16-23%（peak at n=4）

与相关工作的区别：RelayAttention、Hydragen、ChunkAttention 等也探索 shared prefix decoding，但需要分离的 KV-cache management for prefixes and suffixes。FlashInfer composable formats 支持 unified page table 管理下的 multi-level、multiple-prefix decoding，无需修改 serving framework 的 memory management 模块。

涉及论文标题：
- FlashInfer Efficient and Customizable Attention Engine for LLM Inference Serving
