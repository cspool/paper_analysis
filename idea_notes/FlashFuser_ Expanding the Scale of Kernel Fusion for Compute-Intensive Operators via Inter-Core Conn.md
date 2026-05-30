## FlashFuser: Expanding the Scale of Kernel Fusion for Compute-Intensive Operators via Inter-Core Connection

- baseline方法是什么？
  **现有 kernel fusion 编译器和库（Chimera, BOLT, Welder, MC-Fuser）**：这些方法仅利用单个 SM 的 register 和 shared memory (SMEM) 来存储 kernel fusion 的中间结果。当融合的 GEMM 链中间 tensor 大小超出 SMEM 容量上限（H100 每 SM 227KB）时——例如 LLM 的 FFN 层中间 activation 通常远超此值——fusion 失败并回退到将中间结果经 global memory round-trip 的 low-efficiency 执行方式。这些方法采用单一的 block-level tiling hierarchy，不考虑 cluster-level 的数据分布和 inter-SM data exchange。

  全栈执行例子（Chimera on H100, GPT-6.7B FFN GEMM chain, M=128, N=16384, K=4096, L=4096）：
  - **模型推理算法层**：Standard FFN 的两个连续 GEMM: C = A×B (128×16384), E = C×D (128×4096)。中间 C ∈ R^{128×16384} ≈ 4.2MB (FP16)，远超单 SM 的 SMEM 上限 227KB。
  - **系统框架层**：PyTorch 调用 cuBLAS GEMM kernel。Chimera 尝试融合但失败于 SMEM capacity limitation。框架回退到 2 次独立 GEMM kernel launch——kernel1 写 C to HBM, kernel2 从 HBM 读 C。
  - **编译框架层**：Chimera 的 analytical model 仅分析 reg 和 SMEM 两级 cache 的数据 reuse，当中间 tensor 超出 SMEM 容量时直接判定 infeasible 并跳过。BOLT 使用 CUTLASS 模板但受限于固定 block execution order，也未考虑 DSM。Welder 分析 reg/SMEM data reuse 但同样无 DSM 支持。
  - **kernel调度层**：两独立 GEMM kernel——GEMM0: 加载 A/B tiles → Tensor core WGMMA → write C to HBM（~4.2MB per batch）；GEMM1: 从 HBM 读取 C + 加载 D tiles → Tensor core WGMMA → write E to HBM。中间 C 的 HBM read/write 产生了 2×4.2MB = 8.4MB 额外 global memory traffic per batch element。全局显存访问量约 2.4× more than FlashFuser。
  - **硬件架构层**：NVIDIA H100 GPU。SMEM 227KB/SM 是 fusion 的硬限制。DSM 硬件特性存在（inter-core connection via Crossbar），但被现有 software 完全忽略。Global memory bandwidth 3.35 TB/s，DSM bandwidth 在 cluster size 2 时约 8 TB/s（2.4× higher）。

  Baseline 缺陷：
  - (a) **SMEM capacity bottleneck 限制 fusion scope**：227KB/SM 上限导致中间 tensor 超过此大小的 GEMM chain 无法融合，只能 resort to costly HBM round-trip。大量 compute-intensive operators（FFN 占模型总执行时间 40-60%）受此制约。
  - (b) **完全忽略 DSM 硬件能力**：H100 的 inter-core connected architecture 提供了 cluster 内 SM 间高带宽低延迟的 inter-SMEM data path（DSM），但现有 compiler 和 library 无一利用。DSM bandwidth (up to ~8TB/s) 远高于 global memory (3.35TB/s)，DSM latency (~20ns) 远低于 global memory (~280ns)。
  - (c) **单一 tiling hierarchy 限制**：现有方法只考虑 block-level tiling（一个 SM 内的 tile 划分），缺少 cluster-level tiling 概念。引入 DSM 后需同时处理 spatial partition across SMs 和 temporal scheduling within SMs 的两级 hierarchy。
  - (d) **搜索空间爆炸但无处理方案**：引入 DSM 使 infeasible fusion 变 feasible，搜索空间从 ~10^4 膨胀至 ~10^6（GPT-6.7B），现有 pruning 策略不足以应对。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **FlashFuser：首个利用 DSM 进行 kernel fusion 的 DL compiler**。核心设计：(1) dsm_comm primitive——抽象 cluster 内 inter-SM data exchange 模式为形式化原语；(2) Dataflow Analyzer——在 reg→SMEM→DSM 三级 hierarchy 上量化数据搬移并生成 spill plan；(3) Fusion Search Engine——用解析 cost model + DSM-aware pruning 从 ~10^13 的搜索空间中找出最优 plan。

  全栈执行例子（FlashFuser on same GPT-6.7B FFN, H100 SXM, cluster size=(2,4,2,4)）：
  - **模型推理算法层**：同 Standard FFN 计算逻辑不变。FlashFuser 将两个 GEMM 融合为单个 fused CUDA kernel，中间 C 永远不离开 on-chip memory hierarchy (reg→SMEM→DSM)。
  - **系统框架层**：FlashFuser 作为 compiler 生成 CUTLASS-based fused CUDA kernel。离线搜索结果预编译，运行时通过 M-dimension binning + table lookup 选择最优 kernel。可直接嵌入 PyTorch/SGLang 替换 FFN layer 计算。
  - **编译框架层（核心创新——FlashFuser compiler）**：
    - **dsm_comm primitive**：定义三种通信原语统一描述 inter-SM dataflow——dsm_all_exchange（cluster 内 AllReduce/Mul for accumulation）, dsm_shuffle（ring communication within Shuffle Group for data redistribution）, dsm_reduce_scatter（hierarchical reduction for final output aggregation）。通过 cls_m/cls_n/cls_k/cls_l 参数导出 cls_shuffle 和 cls_reduce，精确控制 shuffle group 大小和 reduce group 数量。
    - **Dataflow Analyzer (Algorithm 1)**：遍历 graph 中所有 tensor——IO tensor 按 loop schedule 反序遍历计算 global memory data movement volume；reused tensor 用贪心 heuristic 从 reg→SMEM→DSM 逐级 spill，每级计算 data movement volume（重点分析 DSM traffic，因 DSM bandwidth 低于 SMEM）。最终输出总 D_V 和 placement plan。
    - **Fusion Search Engine (Algorithm 2)**：枚举 LoopSchedule (41 种 S/T 组合) × TilingSize (cluster-level 5^4 + block-level) × ResourceMapping → 5 条 pruning 规则（Divisible Tile, Cluster Size≤16, Activation=innermost loop, Dependency≠spatial L, Memory Capacity）→ Cost model C = max_l(V_l/B_l) → Top-11 硬件 profiling → 最优 plan。
  - **kernel调度层**：单 fused CUDA kernel，内部分为三阶段：
    - **GEMM0 Phase**: Block 内 Tensor core WGMMA 计算 partial C → cls_k=2 表示 K-dim spatial partition, dsm_all_exchange 执行 intra-cluster AllReduce 获得完整 C tile
    - **GEMM1 Phase**: dsm_shuffle ring communication in Shuffle Group (cls_shuffle=2 Blocks) 交换 C tile slices → 各 Block 获得所需 C slice → Tensor core WGMMA 计算 partial E
    - **Store Phase**: dsm_reduce_scatter 两次级归约 (intra-cluster + inter-cluster via TMA cp.reduce.async.bulk) → write final E to HBM
    - 中间 C tile 驻留 DSM（>227KB, 超出 SMEM 但 fit in DSM of multiple SMs），永不写入 HBM
  - **硬件架构层**：同一 H100 GPU。FlashFuser 将 intermediate data path 从 "SMEM → HBM → HBM → SMEM"（traditional round-trip）改为 "SMEM → DSM → SMEM"（direct on-chip path）。DSM bandwidth 约为 global memory 的 1.2-2.4×，latency 约 1/14。全局显存访问减少 58%（Nsight Compute 实测）。dsm_comm primitives 基于 TMA（data movement）+ mbarrier（many-to-many sync）实现，无自定义硬件修改。

  关键设计选择与 baseline 缺陷的对应：
  - **defect (a): SMEM capacity bottleneck** → 方案：DSM 作为 expanded on-chip memory pool——通过 inter-SM communication 将多个 SM 的 SMEM 聚合为虚拟大容量 memory。中间 tensor 超出单 SMEM 容量时 spill to DSM（而非 HBM），在 cluster 内通过 dsm_comm primitives 完成数据复用的通信。以 GPT-6.7B FFN 为例，中间 C tile 在 SMEM 中仅放 128×128×2B≈32KB per Block，但完整 C 需要 128×128×cls_n×cls_k×2B（cluster context），经 DSM exchange 后各 Block 持有完整 row。
  - **defect (b): 完全忽略 DSM 硬件能力** → 方案：dsm_comm primitive 是首个形式化的 DSM-based communication abstraction for kernel fusion。通过 cls_m/cls_n/cls_k/cls_l 参数化 cluster size，导出 cls_shuffle 和 cls_reduce 以精确控制数据交换模式。两种 Gated FFN mapping 策略——spatial partitioning (maximize parallelism) vs sequential execution (minimize DSM overhead)——展示了 DSM-based dataflow 的配置灵活性。
  - **defect (c): 单一 tiling hierarchy** → 方案：两级 hierarchical tiling——cluster-level tile 决定 work distribution across clusters 和 inter-block data exchange patterns，block-level tile 决定单个 Block 内 reg vs SMEM 分配。Loop Scheduling 中 Spatial dimensions 由多个 SM 并行处理（利用 DSM 同步），Temporal dimensions 由单 SM 串行处理。
  - **defect (d): 搜索空间爆炸** → 方案：4 条新增 DSM-aware pruning rules：Cluster Size Constraint (product ≤ 16 hardware limit, consecutive GEMMs' cluster dims must match), Activation Constraint (innermost accumulation dim), Dependency Constraint (L dim can't be spatial), Memory Capacity Limit (tensor ≤ lowest cache capacity)。Pruning 实现 >99.99% 缩减（2.75×10^13 → 1.15×10^6）。Cost model 的 minmax formulation (min max C_l) 精确识别 bottleneck stage 并选 Top-K=11 硬件 profiling，搜索比 brute-force 快 12-68×。
  - **额外贡献：Topology-agnostic design**：dsm_comm 在设计层面是 topology-agnostic 的 collective communication 概念。对 crossbar interconnects (H100, Graphcore IPU) 直接适用；对 mesh architectures (Cerebras WSE) 可通过将 shuffle groups 映射到邻近 core 实现。实现层面基于 CUTLASS + TMA + mbarrier，可移植到其他提供 inter-core connection 的硬件平台。
