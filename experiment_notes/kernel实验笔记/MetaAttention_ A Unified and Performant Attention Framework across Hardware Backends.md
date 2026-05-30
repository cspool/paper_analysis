## MetaAttention: A Unified and Performant Attention Framework across Hardware Backends

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  MetaAttention 的 kernel 调度核心是 **IntermediateTensor-based scheduling** + **two-layer scheduling policy**。实现包括：(i) **IntermediateTensor 抽象**——将 attention 计算中所有 transient tensors（Q/K/V/scores/weights/output 及 customizable function 内部中间结果）统一建模为 IntermediateTensor 对象，每个对象携带三个可配置属性：TileShape（tile size, 通过 computation graph 传播推导所有 tensors 的 tile shape）、MemoryLocation（Register/Shared Memory/Global Memory, 逐级权衡延迟与容量）、PipelineStage（memory copy 与 computation 的重叠阶段数，决定 buffer 需求和调度灵活性）；(ii) **DeviceConfig 抽象**——封装 hardware-specific 约束：BaseTileShape（硬件对齐的 tile shape，如 H100 wgmma MMA instruction tile 和 memory transaction alignment）和 MemoryInfo（各 memory tier 容量：Register 256KB/SM, Shared Memory 228KB/SM, Global Memory 80GB）；(iii) **Tile Config Scheduling（外层）**——枚举所有合法 output tile sizes（对齐 basetile），沿 computation graph 传播 tile shape 约束到所有 IntermediateTensors（相邻 tensors 必须共享相同 tile size），生成候选 tile graphs；(iv) **Tile Resource Scheduling（内层）**——对每个 tile graph，初始化所有 tensors 到最高 memory tier (Register)，按 `(use_count, tile_size)` 排序逐步降级 memory location（Register→Shared Memory→Global Memory），在每级枚举 PipelineStage 并检查 memory constraint，返回所有合法 execution plans；(v) **Profiling-based Selection**——对所有合法 plans 通过 profiling 选 latency 最优者；(vi) **Attention Runtime Dispatch**——根据 scheduling plan 选择 Parallel 或 Recurrent kernel template，通过 code inlining 将 customizable function 的 hardware-mapped code 直接 fused 到 attention execution loop，实现零额外 kernel launch overhead。NVIDIA backend 利用 TMA (Tensor Memory Accelerator) 异步数据加载 + Tensor Core MMA；AMD backend 利用 Matrix Core + async copy。

  实验比较：(1) H100 上 10 种 attention 变体的 kernel 延迟 vs 手写 library（FA2/FA3/FlashMLA/FlashSigmoid/Mamba2/FLA）和编程模型方案（FlexAttention/FlashInfer），涵盖 Parallel、Recurrent、Customized、MLA、Sparse GQA 五类；(2) MI250 上 4 种 attention 变体（Softmax/ReLU/Mamba2/RetNet Recurrent）vs baselines，验证跨硬件调度能力；(3) 编译时间——分钟级（46-89s），significantly shorter than traditional DL compilers；(4) 开发工作量——22-90 LoC vs 手写 library 400-3000 LoC。

- 后端平台是什么，配置是什么。
  NVIDIA H100 SXM5 (132 SMs, 80GB HBM @ 3.35TB/s, 989 TFLOPS FP16, CUDA 12.4, Triton 2.3.1)，AMD Instinct MI250 (ROCm 6.2.4, Triton 3.1.0)。NVIDIA backend 使用 TMA (cp.async.bulk) + Tensor Core wgmma；AMD backend 使用 Matrix Core + async copy。DataType FP16。

- 评估性能的软件/脚本是什么。修改了什么。
  MetaAttention 为全新实现的框架（7.3k lines C++/Python），非修改现有软件。评估使用自编 benchmark 脚本测量 attention kernel wall-clock time，baseline 使用 FlashAttention-2 v2.7.4、FlashAttention-3、FlashMLA（blockSize=64）、FlashSigmoid、Mamba2 chunk kernel、Flash-Linear-Attention v0.2.0、FlexAttention、FlashInfer、PyTorch native。

  核心 kernel 调度实现（Algorithm Fig.10 伪代码对应）：
  ```
  // 外层: Tile Config Scheduling
  tiles = EnumerateTiles(g.output_shape, D.basetile)  // 对齐 MMA tile 枚举
  tensor_tile_graphs = PropagateTileGraphs(g, tiles)   // 沿 computation graph 传播 tile shape
  for tile_graph in tensor_tile_graphs:
    plans += TileResourceScheduling(tile_graph, D)    // 内层: memory + pipeline 调度
  for plan in plans:
    if Profile(plan) < best_latency:                  // profiling-based 选最优
      best_plan = plan
  
  // 内层: Tile Resource Scheduling
  tensor_list = GetIntermediateTensors(g)
  SetTile(tensor_list, g.tiles)
  SetMem(tensor_list, "L0")                           // 初始全放 Register
  tensor_list_sorted = sort(tensor_list, key=(use_count, tile_size))
  for tensor_i in tensor_list_sorted:
    plans = EnumerateUnsetAttributes(tensor_list)      // 枚举 pipelineStage
    for plan in plans:
      if not MeetMemoryConstraint(plan, D.memoryInfo): // 检查 Register/SMEM 容量
        plans.remove(plan)
    if not plans.isEmpty():
      return plans
    LowerMemLocation(tensor_i.mem)                     // Register→SMEM→Global 降级
  ```

  与 Handwritten Kernel Scheduler 的关键差异：
  - Baseline（FA2/FA3/FlashMLA/Mamba2）：kernel 内 tile size、memory placement、pipeline stage 全部 hardcode（如 FA2 固定 B_r=128/B_c=128，SMEM 分配固定），不同 attention 变体或不同 GPU 需重新手写
  - MetaAttention：scheduling 由 IntermediateTensor attributes + DeviceConfig constraints 自动推导；同一套 scheduling policy 适用于所有 attention variants（通过 computation graph 自动传播 tile shape）和所有 hardware backends（通过 DeviceConfig 控制约束）；tile size 非固定——对 dimqk≠dimv 的配置（如 Diff-Transformer-3B: dimqk=128, dimv=256）自动选择不等长的 tile sizes 避免 zero-padding waste

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  已开源：https://github.com/SJTU-IPADS/MetaAttention (MIT License)。Docker 环境：CUDA backend (`Dockerfile.cu128`, ~50 min build)，ROCm backend (`Dockerfile.rocm`, ~80 min build)。Performance test 复现 Figure 11 (H100, ~90 min) 和 Figure 14 (MI250X, ~20 min)。

  **Kernel 调度评估原理与流程**（以 H100 Softmax Attention (head=32, dimqk=128, dimv=128), batch=1, seqlen=2048 为例）：

  1. **Input 准备**：用户定义 Parallel Pattern attention template → MetaAttention 解析为 computation graph G = {Q, K, scores, weights, V, output} 六个 IntermediateTensors + customizable functions DAG（无自定义函数则为 identity）。DeviceConfig: H100 BaseTileShape={M=64/128, N=128/64} (wgmma MMA tile), MemoryInfo={RF:256KB, SMEM:228KB, GMEM:80GB}。

  2. **Tile Config Scheduling（外层）**：
     - EnumerateTiles: 枚举所有合法 output tile shapes——对 dimv=128, 可能的 output tile = {(Br,128) | Br ∈ {64,128,256,...}, Br × 128 × 2B ≤ SMEM free}，生成 ~10-30 个候选
     - PropagateTileGraphs: 沿 G 传播 tile shape——
       - output[B_r, d_v] → weights[B_r, B_c]（需 B_c 对应 V 的 seqlen tile）→ scores[B_r, B_c]（与 weights 同 shape）→ Q[B_r, d_qk] 和 K[B_c, d_qk]（与 scores 的首/末维度对应）
       - 约束：相邻 tensors 共享对应维度 → 所有合法 tile graphs 约 5-15 个

  3. **Tile Resource Scheduling（内层）对每个 tile graph**：
     - 初始 memory location: Q=RF, K=RF, scores=RF, weights=RF, V=RF, output=RF
     - Sorted by (use_count, tile_size): scores(used 2×: weights calc + online norm) > Q(1×) ≈ K(1×) ≈ V(1×) ≈ output(1×)
     - 枚举 pipelineStage: 对 MMA-heavy patterns, pipeline stages=2（async TMA load + compute 重叠）；对 memory-bound patterns, stages=1
     - Memory constraint check: Σ(tile_size × 2B per tensor) ≤ SMEM 228KB, register pressure ≤ 255/SM thread
     - 若 scores[B_r,B_c] + Q[B_r,d] + K[B_c,d] + V[B_c,d] 超 SMEM → 降级 scores 到 Global Memory → re-check → 若仍超 → 降级 V 到 Global Memory → ... → 找到满足约束的最优 placement

  4. **Profiling-based Selection**：
     - 对每个合法 plan，构造 execution time estimate（通过 lightweight microbenchmark 或 analytic cost model）→ 选 latency 最小的 plan
     - 可选实际 kernel launch profiling（更精确但更慢）

  5. **Attention Runtime Code Generation & Execution**：
     - 选择 Parallel Pattern kernel template → 根据 plan 配置 tile sizes, memory buffers, pipeline stages
     - Code inlining: 将 hardware-mapped customizable functions（若有）直接 inline 到 kernel mainloop
     - NVIDIA CUTE backend 生成：TMA cp.async.bulk 异步 load Q/K/V tiles (HBM→SMEM) → wgmma QK^T (SMEM→RF) → CUDA core online softmax (max/exp/rowsum in RF) → wgmma PV (RF accum) → TMA store output (RF→HBM)
     - Pipeline: producer warp (TMA load next K/V tile) ∥ consumer warp (compute current tile)，通过 mbarrier 同步

  6. **Performance measurement**：
     - CUDA event 记录 kernel wall-clock time → TFLOPs = attention FLOPs / time
     - 对比 FA3 (FlashAttention-3 handcrafted CUDA kernel with hardcoded scheduling)
     - 结果：MetaAttention achieves comparable or up to 1.61× (Diff-Transformer-3B forward) speedup over FA3，因 scheduler 可为 dimqk≠dimv 配置自动选择 non-padded tile sizes（FA3 固定 pad 到相同维度）

  7. **Cross-backend 验证（MI250）**：
     - 同 attention template → DeviceConfig 切换为 MI250（BaseTileShape 适应 AMD Matrix Core, MemoryInfo 适应 MI250 hierarchy）→ scheduling policy 自动生成 ROCm-optimized plan → TileLang backend lowering → ROCm kernel
     - MI250 上平均 3.3× forward / 2.0× backward speedup over baselines

  8. **Scheduling Time**：外层枚举 + 内层贪心 → 每次编译 46-89 秒（Table 4），远快于传统 auto-scheduling compiler (Ansor: minutes to hours)。Scheduling result 可 cache 复用（相同 attention config + device → 直接加载 plan，跳过 scheduling）。
