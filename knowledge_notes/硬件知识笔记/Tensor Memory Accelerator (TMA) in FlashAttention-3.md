## Tensor Memory Accelerator (TMA) in FlashAttention-3

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Tensor Memory Accelerator (TMA) 是 H100 GPU 的专用硬件拷贝引擎，在 global memory (HBM) 和 shared memory (SMEM) 之间执行异步批量数据传输。核心特性：(1) 单线程发起——整个 warp 中仅需 1 个线程发射 TMA 指令，其余线程继续计算；(2) 无寄存器中转——数据直接在 HBM↔SMEM 间传输（vs A100 cp.async 需经寄存器）；(3) 硬件加速 mbarrier——SM 硬件专门加速 barrier wait，比软件 spin-wait 更低延迟；(4) multicast——从 HBM 同时拷贝到同一 threadblock cluster 内多个 SM 的 SMEM；(5) descriptor-based——通过 host 端 cuTensorMapEncodeTiled API 创建 tensor map descriptor 描述数据 shape、layout、stride。

从硬件架构角度拆解术语：
FlashAttention-3 中 TMA 的运转流程：
1. Host 端初始化：为 Q、K、V 分别创建 TMA tensor map descriptor（描述 2D tensor shape: [seqlen, head_dim] 或 [seqlen, nheads×head_dim]），传递到 device。
2. Producer warpgroup：在整个 CTA 生命周期内，producer warp（1 warp，register-deallocated via setmaxnreg）循环执行 TMA load：
   - `cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes(smem_addr, &desc, [coord_i, coord_j], mbarrier)`
   - 单线程发射 → TMA 硬件在后台异步执行 HBM→SMEM 传输 → commit mbarrier 通知 consumer
3. Consumer warpgroups：通过 mbarrier.try_wait 等待 K_j/V_j 加载完成后，执行 WGMMA 计算。
4. TMA + WGMMA overlap：TMA 异步加载 K_{j+1}/V_{j+1} 的同时，consumer 正对 K_j/V_j 执行计算——TMA latency 被完全隐藏。
5. Inference optimization：FlashAttention-3 for inference 使用 TMA block table (PagedAttention)——自定义 SM90_TMA_LOAD_PAGED_OP class，tensor map descriptor 的坐标基于 virtual shape，block table 通过额外参数传入 TMA copy method。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
CUTLASS 3.x 中 TMA 通过 `cute::copy` with TMA copy atoms 使用。Triton 通过 `tl._experimental_descriptor_load` 实验性访问。使用前提：(1) 数据访问模式可用规则 tensor 坐标描述（非随机访问）；(2) 需预先创建 tensor map descriptor（host-device 额外开销，适合 persistent kernel 分摊）；(3) H100+ 独有（A100 及以下不支持）。FlashAttention-3 中 TMA 对 variable sequence length 的处理：forward pass 中 TMA 固定加载 tile_size 行，超出原始 tensor 的部分 TMA 自动填零，S tensor masking 再掩盖多余行；backward pass 通过 preprocess kernel padding dQ/dPSum/LSE tensors to tile_size 对齐。

FlashFuser 中 TMA 的额外用途包括：(1) DSM 数据交换——TMA 的 `shared::cluster` 地址空间直接访问同 cluster 内其他 SM 的 shared memory，配合 mbarrier 实现 dsm_shuffle 的 ring communication；(2) inter-cluster reduction——TMA 的 `cp.reduce.async.bulk` 指令在 store 阶段执行跨 cluster 的异步原子归约，避免 global memory round-trip；(3) TMA multicast——从 HBM 同时拷贝数据到 cluster 内多个 SM 的 SMEM，用于 producer-consumer warp specialization 的数据预取。

MetaAttention 中 TMA 的使用：
MetaAttention 在 NVIDIA backend 中通过 CUTE 和 TileLang 两个 backend 框架使用 TMA。Attention Runtime 根据 scheduling plan（由 IntermediateTensor memory location + pipelineStage 决定）生成 kernel 模板——当 intermediate tensor 需要从 global memory 加载时，producer warp 使用 TMA `cp.async.bulk.tensor` 异步加载 Q/K/V tiles 到 SMEM 的 circular stage buffer；consumer warp 执行 WGMMA 计算。TMA 的异步特性与 pipeline stage 配置（通常为 2-stage pingpong）配合实现 load-compute overlap。TileLang backend 将 TMA 操作抽象为 high-level tile copy 原语，由 TileLang compiler 自动 lower 到 TMA 指令。

涉及论文标题：
- FlashAttention-3 Fast and Accurate Attention with Asynchrony and Low-precision
- FlashFuser: Expanding the Scale of Kernel Fusion for Compute-Intensive Operators via Inter-Core Connection
- MetaAttention: A Unified and Performant Attention Framework across Hardware Backends
