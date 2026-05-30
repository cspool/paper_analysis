## Weights Paging for MoE Offloading (MoE卸载中的权重分页)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Weights Paging 是 MoE-Lightning 提出的一种细粒度权重传输机制，将每层 MoE experts 的 FFN weights 从 CPU memory 传输到 GPU 时分页进行。类比操作系统中的内存分页（memory paging），weight paging 将整层所有 experts weights（例如 Mixtral 8x7B 每层 8 experts × 2 layers × 4096×14336 ≈ 1.8GB）切分为 n 个 page（n = 微批次数量），每个 page 大小 = total_layer_weights / n。GPU expert FFN kernel 通过 page table 查找对应 weight pages 的 GPU 地址来访问。分页的目的：在整层 weights 传输之间交错插入 hidden states H2D 传输（这些 hidden states 是下一个微批次 GPU PostAttn 计算的必要输入），避免整层 weights 一次性传输造成的长时间 H2D 阻塞。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
Paged Weight Transfer 流程（CGOPipe 中）：
1. **分配**：GPU 上分配 weight buffer 大小 = 2 × sizeof(per-layer-weights-on-CPU)，支持双缓冲（当前层使用 + 下一层预取）。
2. **分页**：将每层 experts weights 在 CPU memory 中按 expert 和层内维度切分为 n_pages。每页大小适配一次 cudaMemcpyAsync 调用的最优粒度。
3. **两阶段流水线传输**：
   - Stage 1: Page k 从 CPU DRAM 通过 memcpy 拷贝到 CPU pinned memory
   - Stage 2: Page k 从 CPU pinned memory 通过 cudaMemcpyAsync 异步传输到 GPU
   - Stage 1 和 Stage 2 在不同 pages 间重叠（Page k 的 Stage 2 与 Page k+1 的 Stage 1 并行）
4. **交错执行**：在 PostAttn(i, j) 执行前，仅传输 layer i 的 weight page j 到 GPU（不传输整层），随后立即执行该微批次 PostAttn。在两个微批次之间交错插入 hidden states H2D (D2)。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：MoE-Lightning 的 Memory Manager 模块（Appendix A.1），C++ 实现。(1) 维护 page table（映射 expert_id × page_id → GPU buffer offset）；(2) CUDA streams 管理异步传输；(3) 使用 CUDA events 进行 synchronization。
- 优点：消除 FlexGen 方案中整层 weights 一次性传输导致的后续微批次 H2D 阻塞，使 GPU 利用率更高。
- 局限：(1) 增加 page table lookup 开销（通常可忽略）；(2) 分页粒度需要在 I/O 效率与 kernel launch overhead 之间权衡。

涉及论文标题：
- MoE-Lightning: High-Throughput MoE Inference on Memory-constrained GPUs
