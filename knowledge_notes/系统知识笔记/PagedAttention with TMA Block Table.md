## PagedAttention with TMA Block Table

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
PagedAttention (Kwon et al., SOSP 2023) 是一种 KV cache 内存管理技术，将 KV cache 划分为固定大小的 "pages"（blocks），通过 page table 将逻辑 KV block 位置映射到物理 HBM 地址，消除传统 contiguous KV cache 的内存碎片问题。FlashAttention-3 的 inference 实现中（contributed by Kai Londenberg），PagedAttention 与 TMA 首次结合：传统 TMA load 通过 tensor map descriptor（物理 tensor shape）确定坐标，不支持非连续物理地址。FlashAttention-3 定义新的 SM90_TMA_LOAD_PAGED_OP class 和 tensor map constructor，基于 virtual shape（continuous logical KV cache）构建 descriptor，TMA tensor 坐标对应用 virtual 空间计算，而 block table（logical block → physical address mapping）作为额外参数传入 TMA copy method，在 TMA 硬件内部完成地址翻译。

从系统架构角度拆解术语：
```
// 传统TMA: descriptor编码physical tensor shape，坐标=physical位置
cp.async.bulk.tensor.2d.shared.mbarrier(smem, &physical_desc, [i, j], mbarrier);

// PagedAttention+TMA: descriptor编码virtual tensor shape，坐标=virtual位置
// TMA copy method额外接收block_table参数
cp.async.bulk.tensor.2d.shared.mbarrier(smem, &virtual_desc, [i_logical, j_logical],
                                         mbarrier, block_table);
// TMA硬件内部: virtual_coord → logical_block_id → block_table[logical_block_id] → physical_address
```
这与 vLLM 的 PagedAttention 用户态 page table lookup（每个 token 一次）不同——TMA 版本将地址翻译委托给 TMA 硬件，减少 CUDA core 的地址计算开销，特别适合 memory-bound 的 decode 阶段（节省的 instruction slot 可用于发射更多 TMA/MMA 指令）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
PagedAttention with TMA 需：(1) H100+ GPU（TMA 支持）；(2) CUTLASS 3.x pipeline abstractions 的自定义 TMA copy atom（定义新的 SM90_TMA_LOAD_PAGED_OP class）；(3) host 端创建 virtual tensor map descriptor + 维护 block table。目前 FlashAttention-3 的 inference backend 是主要使用场景。与标准 PagedAttention（A100 兼容，用户态 page table lookup）相比，TMA 版本在 decode 阶段的 instruction overhead 更低，但仅限 Hopper+ 架构。FlashInfer 则从另一角度处理 PagedAttention——将 page table 统一映射为 BSR (Block-Sparse Row) 稀疏矩阵格式，通过 BSR indices arrays 在 kernel 内间接寻址 KV-cache pages，而非修改 TMA 硬件路径。这使得 FlashInfer 的 PagedAttention 支持兼容 A100/H100 全系列 GPU（不依赖 TMA），同时支持 vector-sparsity ($B_c=1$ per-page sparsity) 处理 fine-grained page-level selective attention。

涉及论文标题：
- FlashAttention-3 Fast and Accurate Attention with Asynchrony and Low-precision
- FlashInfer Efficient and Customizable Attention Engine for LLM Inference Serving
- Flex Attention: A Programming Model for Generating Optimized Attention Kernels
