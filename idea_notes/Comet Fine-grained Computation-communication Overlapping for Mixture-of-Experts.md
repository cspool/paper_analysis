## Comet Fine-grained Computation-communication Overlapping for Mixture-of-Experts

- baseline方法是什么？
  **Baseline 为 Megatron-LM 中的标准 MoE 执行方式**，包括四种变体：

  1. **Megatron-Cutlass**: Megatron-LM 默认实现，使用 CUTLASS GroupGEMM 作为 expert FFN kernel，通信（NCCL all-to-all）和计算（GEMM）顺序执行，**无任何通信-计算重叠**。

  2. **Megatron-TE (Transformer Engine)**: 使用 NVIDIA Transformer Engine 的 GEMM 实现，同样无通信-计算重叠。TP 下因 Transformer Engine API 调用开销，部分配置下性能甚至差于 Megatron-Cutlass。

  3. **FasterMoE**: 通过自定义 Scatter 和 Gather 算子将 expert 计算分为 2 个 chunk 实现 pipeline overlap（pipeline degree=2，即 coarse-grained）。仅 hide 29.2% 通信延迟，同时引入了额外的 local indexing 开销。

  4. **Tutel**: 通过自适应并行策略切换、2D 分层 all-to-all 和启发式搜索实现 partial overlap。hide 68.6% 通信延迟，但 expert 数量大时（如 Qwen2 的 E=64）CPU 端 scheduling overhead 增大导致优势衰减。

  **Baseline 全栈执行例子（以 Mixtral 8x7B, EP=8, TP=1, M=16384, 一个 MoE layer 为例）**：

  - **算法层**: Router 计算 top-2 gating → 全量 all-to-all dispatch tokens → 等待通信完成 → 各 GPU 执行持有的 experts 的 GEMM（layer0 → activation → layer1）→ 全量 all-to-all combine → 等待通信完成 → top-K reduce → 输出。整个 pipeline 完全顺序执行：通信阻塞计算，计算阻塞通信。

  - **系统框架层**: Megatron-LM (Expert Parallelism + Data Parallelism)。MoE layer forward = `token_permutation → alltoall_dispatch (NCCL) → expert_gemm (CUTLASS GEMM, 每 expert 独立 kernel launch) → alltoall_combine (NCCL) → token_unpermutation → reduce`。Host CPU 需要为每步通信和计算分别 launch kernel，kernel launch overhead 在小 M 时占比显著。

  - **编译框架层**: PyTorch eager execution + NCCL + CUTLASS。无通信-计算融合编译优化。

  - **Kernel 调度层**: Expert GEMM 使用 CUTLASS group_gemm。所有 expert 的 GEMM tile 统一调度，无 tile 级重排序。通信使用 NCCL all-to-all，按完整大 tensor 一次传输。通信和计算通过独立的 CUDA streams 发射（如 Tutel/FasterMoE），但 coarse-grained chunk 划分导致重叠效率低——初始和最后的通信阶段无计算可重叠，产生 pipeline bubble。

  - **硬件架构层**: 8× H800 GPU (NVLink)。通信阶段 GPU SM 大量空闲（仅 NCCL kernel 使用少量 SM 做数据搬运），计算阶段通信链路空闲。GPU compute utilization 在通信期间接近 0。

  **Baseline 的核心缺陷**：
  1. **粒度不匹配（Granularity Mismatch）**: MoE 的通信以 token 为单位（单个 token 是最小数据搬运单元），但 GEMM 以 tile（如 128×128）为计算粒度——一个 GEMM tile 需要 128 个 token 的数据，这些 token 可能分布在多个 remote GPU 上。Coarse-grained pipeline（FasterMoE/Tutel）必须等一个 chunk 中所有 token 到齐才能启动计算，导致 tile 粒度以下的等待无法消除。
  2. **数据依赖复杂**: MoE gate 在运行时动态决定 token→expert 映射，每个 GEMM tile 所需的 token 随机分布在多 GPU 上。计算 tile 不能开始直到其依赖的所有 token（local + remote）可用，但在 coarse-grained 通信中远程 token 只能按 chunk 整体到达，无法按 tile 粒度就绪。
  3. **通信和计算负载动态变化**: MoE 的 token 分布不均衡（不同 expert 接收不同数量的 token），通信量和计算量在运行时动态变化。将通信和计算封装在独立 kernel 中（FasterMoE, Tutel 的做法）使得 GPU SM 资源分配在编译时固定，无法根据运行时负载自适应调整，导致重叠中的气泡（bubble）。
  4. **Host 端 scheduling overhead**: 多个独立 kernel launch 之间需要 CPU 端调度（尤其是 Tutel 的 adaptive scheduling 和 FasterMoE 的 multi-expert kernel），在小 M（短序列）时 CPU scheduling 成为 dominant overhead。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **Comet** 提出两个核心设计解决 MoE 中通信-计算的 fine-grained overlapping：

  **Defect → Design 映射**：

  | Baseline 缺陷 | Comet 设计选择 | 解决机制 |
  |---|---|---|
  | 粒度不匹配（token 通信 vs tile 计算） | **Shared Tensor Based Dependency Resolving**: 沿 M（layer0）或 N（layer1）维度分解 shared tensor → tile 计算重排序 | 将 token 级通信与 tile 级计算对齐——local token tiles 优先计算，remote token tiles 延后（此时通信已并发完成） |
  | 数据依赖复杂（tile 级等待） | Tile 按数据依赖重排序：layer0 按 source rank 排序优先计算最小依赖 tile，layer1 改为 column-wise 并行 | 每个 tile 能在其依赖的 tokens 就绪后立即开始（而非等待整个 chunk） |
  | 通信/计算负载动态变化 | **Adaptive Workload Assignment**: Thread block specialization + 自适应分配 n^c/n^p 比例 | 通信和计算在同一个 fused kernel 内执行但隔离到不同 thread blocks，运行时根据 M 和 parallelism 选择最优比例 |
  | Host 端 scheduling overhead | 通信和计算融合在单个 GPU kernel 中 | 消除多次 kernel launch 的 CPU↔GPU 往返延迟 |

  **Comet 方法全栈执行例子（同 Mixtral 8x7B, EP=8, TP=1, M=16384, 一个 MoE layer）**：

  - **算法层（不变）**: 同 baseline，Router 计算 top-2 gating → 决定 token→expert 映射。

  - **系统框架层**: Megatron-LM + Comet Python API。MoE layer forward 从多步 kernel launch 变为两个 Comet fused kernel：
    ```
    # Baseline: ~6+ kernel launches
    token_permute → alltoall_dispatch → expert0_gemm → expert1_gemm → ... → alltoall_combine → reduce
    
    # Comet: 2 fused kernel launches
    comet_layer0(shared_tensor, routing_map, expert_weights)  # NVSHMEM recv + GroupGEMM
    comet_layer1(shared_tensor, routing_map, expert_weights)  # column-wise GEMM + reduce + NVSHMEM send
    ```

  - **编译框架层**: 论文未明确说明（CUDA C++ + CUTLASS 模板 + NVSHMEM，无 Triton/TVM 编译层）。

  - **Kernel 调度层（核心创新）**:
    
    **Layer0 (Communication→Computation Pipeline)**:
    1. Shared Tensor 识别: layer0 的 shared tensor = dispatch buffer [M×topk, N]，是通信(producer)的输出和 GEMM(consumer)的输入
    2. 沿 M 维度分解: shared tensor 按行（token 粒度）分解 → 每个 token 独立可作为 GEMM 的输入
    3. Token 重排序: 所有需要参与 GroupGEMM 的 tokens 按 source rank 排序 → local tokens 聚集在前（无需通信，立即可用），remote tokens 聚集在后
    4. GroupGEMM tile 调度: tile 计算顺序重新编排——仅含 local tokens 的 tile 优先计算 → 含部分 remote tokens 的 tile 等 NVSHMEM 完成 → 纯 remote tile 最后。在计算早期 tiles 的同时，NVSHMEM 通信 thread blocks 正在拉取后续 tiles 所需的 remote tokens
    5. Thread block 隔离: 通信 TB 执行 NVSHMEM `get` → shared tensor buffer。计算 TB 执行 CUTLASS GroupGEMM（TMA async copy + tensor core MMA）。两套 TB 由 SM hardware scheduler 并发调度，互不干扰。

    **Layer1 (Computation→Communication Pipeline)**:
    1. Shared Tensor 识别: layer1 的 shared tensor = GEMM 输出 buffer [M×topk, N]，是 GEMM(producer)的输出和 reduce+通信(consumer)的输入
    2. 沿 N 维度分解: shared tensor 按列（hidden dim 粒度）分解为 N/T^N 个列块
    3. Column-wise GEMM: 所有 expert 并行计算第 1 个列块 → T^N 列完成后立即 top-K reduce → NVSHMEM write 回 source rank → 同时继续计算第 2 个列块 → ...
    4. 重叠效果: reduce+通信 与 后续列的 GEMM 计算完全重叠。Baseline 必须等所有 expert 全部列计算完才开始 reduce+通信。

    **Adaptive Assignment**: 预编译内核库含多个 n^c/n^p 比例（n^c=18/26/46...）→ deployment 前 profile → runtime 按 M 和 (EP,TP) 查表选择最优 kernel。

  - **硬件架构层**: 8× H800 GPU (NVLink)。NVSHMEM 分配 buffer = 2×M×N bytes（M=16384 时 128MB），跨所有 MoE layers 全局复用。Comet 对 GEMM thread block 使用标准 CUTLASS Hopper 实现（TMA + MMA），通信 thread block 额外占用 SM 资源但通过隔离避免了干扰 GEMM 的异步计算流水线。

  **对比 Baseline 的核心改进路径**：
  ```
  Baseline (Megatron-Cutlass/FasterMoE/Tutel):
  Tokens → NCCL all-to-all (full tensor, wait complete)
  → Expert GEMM (all tiles, sequential)
  → NCCL all-to-all (full tensor, wait complete)
  → Reduce
  通信占 47% total time, zero 或 partial overlap (29-69%)
  
  Comet:
  Tokens → [Fused Kernel Start]
    ├─ Comm TB: NVSHMEM get token-by-token → shared tensor buffer
    ├─ Compute TB: GroupGEMM tile 0 (local tokens, no wait)
    ├─ Comm TB: more NVSHMEM get (background)
    ├─ Compute TB: GroupGEMM tile 1 (ready tokens)
    ├─ ... (fine-grained interleaving)
    ├─ Compute TB: Column-wise GEMM col 0 → reduce → NVSHMEM send
    ├─ Compute TB: Column-wise GEMM col 1 (while col 0 reducing)
    └─ ...
  → [Fused Kernel End]
  Hide 86.5% communication, 1.96× single-layer speedup, 1.71× end-to-end speedup
  ```

  **关键设计对应关系**：
  | 设计选择 | 解决的具体问题 | 数值验证 |
  |---|---|---|
  | Shared tensor 沿 M/N 维分解 | 打破 coarse-grained 数据依赖，token 和 tile 粒度对齐 | 通信隐藏从 29-69% → 86.5% |
  | Tile 按数据依赖重排序（local→remote） | Complex data dependency 导致的 tile 级等待 | Expert compute efficiency 不受 partitioning 影响（t_1+t_2 ≈ t） |
  | Thread block specialization（隔离通信/计算） | Fine-grained I/O 拖慢 GEMM (尤其是 Hopper TMA 流水线) | GEMM TB 使用标准 CUTLASS，零性能退化 |
  | Adaptive n^c/n^p 分配 | 运行时负载动态变化（不同 M, TP, EP 下最优分配不同） | 不同配置自动选择最优 kernel, 无需人工 tuning |
  | NVSHMEM 替代 NCCL | Token 级 fine-grained remote I/O (NCCL 只支持 coarse-grained) | Unified Virtual Address 逐 token 访问 |
  | 单 fused kernel 替代多 kernel launch | Host 端 scheduling overhead (小 M 时 dominant) | 小 M (256-1024) 时 speedup 更高 (2.37×) |

  **创新总结**: Comet 的核心洞察是——MoE 通信和计算之间存在复杂的 token-tile 数据依赖，coarse-grained pipelining（按 chunk 重叠）无法消除这种依赖导致的等待。通过将 shared tensor 沿正确维度分解并重调度 tile 计算顺序，Comet 将粗粒度的 chunk 级重叠升级为 fine-grained 的 tile 级重叠。进一步地，通过 thread block specialization 将通信和计算隔离到同一 kernel 的不同 TB，避免了 fine-grained I/O 干扰 high-performance GEMM（尤其是 Hopper 的 TMA 异步流水线）。这种方法本质上是将 MoE 系统优化从 "kernel 间调度" 下沉到 "kernel 内调度"，消除了 CPU 端 scheduling overhead 并实现了精准的 GPU SM 资源分配。
