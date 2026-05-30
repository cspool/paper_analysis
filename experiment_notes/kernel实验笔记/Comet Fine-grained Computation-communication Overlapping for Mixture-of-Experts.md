## Comet Fine-grained Computation-communication Overlapping for Mixture-of-Experts

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现是 Comet 的 **fine-grained fused MoE kernel**，核心包含三项 kernel 级优化：
  1. **Shared Tensor Based Dependency Resolving**：将 MoE layer0 的 shared tensor 沿 M（token）维度分解，将 layer1 的 shared tensor 沿 N（hidden）维度分解，并重新调度 GroupGEMM 的 tile 计算顺序——layer0 按 remote token 依赖最小化排序（local token 优先计算），layer1 按列方向交错计算（T^N 列完成后立即启动 reduce+通信）。这打破了通信(token级)和计算(tile级)之间的粒度不匹配。
  2. **Thread Block Specialization**：将通信（NVSHMEM I/O）和计算（CUTLASS GEMM）隔离到独立的 thread block，而非将它们垂直融合在同一 thread block 中。GEMM thread block 使用标准的 CUTLASS Hopper 实现（producer warp 用 TMA async 加载，consumer warp 执行 MMA），通信 thread block 独立执行 token 的 top-K reduce 和远程 NVSHMEM 读写。避免 fine-grained I/O 干扰计算 pipeline（尤其是 Hopper TMA 异步流水线）。
  3. **Adaptive Thread Block Assignment**：预编译多个不同 n^c/n^p（通信/计算 thread block 比例）的 kernel 变体，离线 profile 各配置下的最优分割点并存储为 metadata。运行时根据输入 token 长度 M 和并行策略（TP×EP）从 metadata 选择最优 kernel。最优 n^c 随 M 增大而增大（如 TP=8 时 M=4096→n^c=18, M=16384→n^c=26），随 TP 减小而增大（M=16384 时 TP=8→n^c=26, TP=4→n^c=46）。

  实验比较：
  - **Single MoE layer duration** (Figure 10): Comet vs Megatron-Cutlass, Megatron-TE, FasterMoE, Tutel，EP=8，M 从 256 到 16384 → Comet 1.28×-2.37× speedup，小 M 时优势更显著（kernel 内调度消除了 host 端 kernel launch 开销）
  - **MoE layer time breakdown** (Figure 11): EP=8, TP=1, E=8, topk=2, M=16384 → Comet hides 86.5% communication latency vs FasterMoE 29.2% and Tutel 68.6%
  - **Various parallelism** (Figure 12): E=8, topk=2, M=8192, EP×TP=8，不同 TP/EP 组合 → Comet 在所有配置下保持低延迟（shared tensor reschedule 消除 TP 引入的 fragmented GEMM 问题）
  - **Varying E and topk** (Figure 13): Comet 1.16×-1.83× speedup vs baselines
  - **Imbalanced token distribution** (Figure 14 left): std 从 0 到 0.05 → Comet 在所有分布下 consistently outperform
  - **L20 cluster (PCIe, 25 GB/s)** (Figure 14 right): Comet 1.19×-1.46× speedup vs baselines

- 后端平台是什么，配置是什么。
  **H800 集群**: 8× NVIDIA H800 GPU (80GB HBM)，NVLink 互联。CUDA 12.3, NVSHMEM 2.11, PyTorch 2.4.0, Megatron-LM (git-hash 6dbe4c)。
  **L20 集群**: 8× NVIDIA L20 GPU (46GB)，PCIe 桥互联，GPU-to-GPU 带宽约 25 GB/s。

- 评估性能的软件/脚本是什么。修改了什么。
  **评估工具**: Megatron-LM 框架上的端到端 MoE 模型推理/训练。使用 PyTorch Profiler 采集时间 breakdown。模型: Mixtral 8x7B (E=8, N=4096, K=14336), Qwen2-MoE-2.7B (E=64, N=2048, K=1408), Phi-3.5-MoE (E=16, N=4096, K=6400)。

  **Comet 修改内容（~12k lines C++/CUDA + 2k lines Python）**:
  1. **CUTLASS GEMM kernel 优化**: 在 layer0 中将 GEMM 输入的 row indices 缓存到寄存器，减少 global memory 访问。利用 CUTLASS 编程模板生成高效 GroupGEMM kernel。
  2. **Shared Tensor Decomposition & Reschedule**: 
     - Layer0 (communication→computation pipeline): 将 shared tensor [M×topk, N] 沿 M 维度分解为 sub-tensors。Token 按 source rank 排序 → GroupGEMM tile 计算顺序设计为 local token tile 优先 → 在 remote token 传输期间 local tokens 已开始计算。
     - Layer1 (computation→communication pipeline): 将 shared tensor 沿 N 维度分解。GroupGEMM 改为 column-wise 执行（先计算所有 expert 的前 T^N 列 → 启动 reduce+通信 → 继续后续列），而非 sequential per-expert 执行。
  3. **NVSHMEM Fused Kernel**: 使用 NVSHMEM 的 Unified Virtual Address 实现 token 级 fine-grained remote I/O，替代 NCCL 的粗粒度 all-to-all。NVSHMEM buffer 大小 = M×N（BF16/FP16 时 2MN），shared across layers and experts。
  4. **Thread Block Specialized Kernel**: 在 Hopper 架构上，GEMM thread blocks 使用 CUTLASS 标准实现（producer warp TMA async load → shared memory → consumer warp MMA），通信 thread blocks 读取 GEMM 输出 → top-K reduce → NVSHMEM write/read。两套 thread blocks 由 GPU hardware scheduler 并发调度在同一 kernel 内。
  5. **Adaptive Assignment Metadata**: 预编译多个 (n^c, n^p) 组合的 kernel 变体，profile 后存储为 metadata，运行时查表选择。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  **开源**: https://github.com/bytedance/flux（Project Page）。Comet 将开源，代码基于 Megatron-LM + CUTLASS + NVSHMEM。

  **Comet Fused Kernel 全流程（以 H800, EP=8, TP=1, E=8, topk=2, M=16384, 一个 MoE layer 为例）**:

  **=== MoE Layer0 (Communication→Computation Pipeline) ===**
  
  1. **输入**: 每个 rank 持有 M/W = 2048 tokens，经 Router 计算后需 dispatch 到持有对应 expert 的 rank
  
  2. **Shared Tensor Decomposition (沿 M 维)**:
     ```
     shared_tensor shape: [M×topk, N] = [32768, N]
     被分解为 M×topk = 32768 行（token 粒度）
     → tokens 按 source_rank 排序: local tokens 在前，remote tokens 在后
     ```
  
  3. **NVSHMEM Fine-grained Receive**: 通信 thread blocks 通过 NVSHMEM `nvshmem_getmem` 从 remote rank 逐 token 拉取数据到 shared tensor buffer。每个 token 独立可读——不等待全部 all-to-all 完成。
  
  4. **GroupGEMM Tile 重调度**:
     ```
     GroupGEMM 的 tile 计算顺序重新设计:
     先计算仅含 local tokens 的 tiles（无数据依赖，立即开始）
     → 再计算含部分 remote tokens 的 tiles（remote tokens 已通过 NVSHMEM 到达）
     → 最后计算纯 remote tokens 的 tiles
     在 GroupGEMM 执行早期 tiles 的同时，更多的 remote tokens 在并行到达
     ```
  
  5. **Thread Block Specialization (SM 分配)**:
     ```
     Total SMs = 132 (H800)
     通信 thread blocks (n^c): 执行 NVSHMEM remote read + token scatter
     计算 thread blocks (n^p): 执行 CUTLASS GroupGEMM
     # n^c/n^p 比例由 adaptive assignment metadata 决定
     
     GEMM TB (CUTLASS Hopper):
       Warp 0 (producer): TMA async copy global→shared memory
       Warp 1 (consumer): MMA (tensor core) shared→register→accumulator
     
     通信 TB:
       Warp 0..N: NVSHMEM get + scatter tokens to shared tensor buffer
     ```
  
  6. **输出**: 所有 expert 的 GEMM 完成 → layer0 output → 进入 layer1

  **=== MoE Layer1 (Computation→Communication Pipeline) ===**

  7. **Shared Tensor Decomposition (沿 N 维)**:
     ```
     shared_tensor shape: [M×topk, N] = [32768, N]
     被分解为 N/T^N 个列块（T^N 为 GroupGEMM tile N 维度大小）
     ```
  
  8. **Column-wise GroupGEMM + 通信重叠**:
     ```
     for col_block in [0, N/T^N):
       # 所有 expert 并行计算第 col_block 块
       for each expert on this rank:
         GEMM_tile_compute(expert, col_block)  # partial result along N dim
       
       # T^N 列完成后立即启动 reduce + 通信
       topk_reduce(partial_results[:, :col_block * T^N])
       NVSHMEM write tokens back to source ranks
     
     # 传统方案: 每个 expert 全部列计算完 → 才开始 reduce+通信
     # Comet: 每 T^N 列计算完 → 立即 reduce+通信 → 与剩余列计算重叠
     ```
  
  9. **Adaptive Assignment**: kernel 启动时，从 metadata 查表选择 (n^c, n^p) → 设定 thread block 数量 → 多 SM 并发执行
  
  10. **性能输出**: total MoE layer duration 记录为从 Router 开始到 combine 完成的时间。Comet hides 86.5% communication (Figure 11)，单层 1.96× speedup vs Megatron-Cutlass (Figure 10)。

  **与 Baseline 的关键差异**:
  - Megatron-Cutlass/TE: 零通信-计算重叠（通信完成→计算开始→通信开始，顺序执行）
  - FasterMoE: 将 expert 计算分 2 个 chunk 做 pipeline overlap（coarse-grained），仅 hide 29.2% 通信
  - Tutel: 通过优化 all-to-all primitive 和自适应并行 partial overlap，hide 68.6% 通信，但 expert 多时 scheduling overhead 增大
  - Comet: fine-grained token 级 fused kernel overlap，hide 86.5% 通信，且 kernel 内调度消除 host 端 overhead
