## MetaAttention: A Unified and Performant Attention Framework across Hardware Backends

- 属于编译框架的实现是什么？实验比较什么？
  MetaAttention 实现了一个端到端的 attention 编译框架，将用户定义的 attention template（Parallel/Recurrent pattern + customizable functions）自动 lowering 为硬件优化的 kernel。核心编译流程包括：(i) **Customizable Function Lowering**——将用户定义的 Python customizable functions 通过 tracing 转换为 directed acyclic graph (DAG) of computing primitives，每个 node 分类为 elementwise（add/tanh/mul 等，映射为 SIMT-style register-level fused execution）或 row-reduce（reduceSum/reduceMax 等，映射为 intra-warp parallel reduction）；(ii) **Two-layer Scheduling Policy**——外层 Tile Config Scheduling 枚举所有可能的 output tensor tile sizes，通过 computation graph 传播 tile shape 到所有 intermediate tensors 生成 tile graphs；内层 Tile Resource Scheduling 为每个 tile configuration 确定 memory location（register/shared memory/global memory）和 pipeline stages，逐步从最高 memory tier 降级直到满足 hardware constraint；(iii) **Attention Runtime Code Generation**——基于 scheduling plan 选择 kernel template（parallel 或 recurrent），将 hardware-mapped customizable functions 通过 code inlining 直接 fused 到 attention execution loop 中，零额外 kernel launch overhead；(iv) **Multi-backend Mapping**——NVIDIA GPU 使用 TileLang 和 CUTE 两种 backend 实现 TMA + Tensor Core MMA，AMD GPU 使用 TileLang backend 实现 Matrix Core + async copy。

  实验比较：H100 上 10 种 attention 变体的 kernel 延迟 vs FlashAttention-2/3、FlashSigmoid、FlashMLA、Mamba2 chunk kernel、Flash-Linear-Attention、FlexAttention、FlashInfer、PyTorch。特别验证编译器生成 kernel 与手写 library 的性能差距：(i) Softmax Attention forward/backward vs FA3——平均相当或更快（Diff-Transformer-3B 上 1.61×），因编译器可针对不同 dimqk≠dimv 优化 tile size 避免 zero-padding；(ii) 不支持变体（ReLU Attn/Sigmoid/Retention Parallel）——3.6× speedup vs 不支持的 baseline；(iii) 编译时间（Table 4）——分钟级（46-89 秒），比传统 DL compiler (Ansor) 快得多。

- 硬件平台是什么，配置是什么。
  NVIDIA H100 SXM5 (CUDA 12.4, Triton 2.3.1)，AMD Instinct MI250 (ROCm 6.2.4, Triton 3.1.0)。H100 评估 batch sizes 1/8, seqlen 2K/4K/8K，FP16。编译时间测量（Table 4）：Softmax Attention (H100 46s, MI250 64s)，Mamba2 SSM (H100 82s, MI250 89s)。

- 开源编译框架是什么。修改了什么。
  MetaAttention 为全新实现的框架（7.3k lines C++/Python），非修改现有编译器。开源：https://github.com/SJTU-IPADS/MetaAttention (MIT License)。

  底层依赖：
  - **TileLang** [29] (https://arxiv.org/abs/2504.17577)：Composable tiled programming model，用于 NVIDIA 和 AMD backend 的 kernel 代码生成
  - **CUTE** [7] (https://github.com/NVIDIA/cutlass)：NVIDIA CUTLASS 的 CUDA template library，用于 NVIDIA backend 的 TMA + MMA 实现
  
  核心编译组件：
  - **Customizable Function Tracer**：将 Python 函数 trace 为 DAG of tensor operations，nodes 分类为 elementwise（SIMT mapping with register-level fusion）或 row-reduce（intra-warp reduction mapping）
  - **IntermediateTensor + DeviceConfig 抽象**：IntermediateTensor（tile shape, memory location, pipeline stage）建模 scheduling space；DeviceConfig（base tile shape, memory hierarchy capacities）提供 hardware constraint
  - **Two-layer Scheduling Policy**：外层枚举 tile sizes → 传播 tile graph → 内层贪心确定 memory placement 和 pipeline stages → profiling-based 选最优
  - **Attention Runtime**：kernel template selection + code inlining + backend-specific lowering（TMA/Tensor Core for NVIDIA, Matrix Core for AMD）

- 开源情况。基于开源文档和论文，使用例子解释编译框架如何使用？作用是什么？至少具体到编译框架输入到输出的全过程。
  已开源：https://github.com/SJTU-IPADS/MetaAttention (MIT License)。Docker-based 环境，支持 CUDA 和 ROCm 两种 backend。Functional test: `python testing/test.py`（验证 Parallel 和 Recurrent pattern 正确性 vs PyTorch reference），Performance test: 复现 Figure 11/14。

  **编译框架完整流程**（以 RetNet Parallel Pattern, H100, seqlen=2048, head=32, dimqk=256, dimv=512 为例）：

  1. **用户输入（Programming Interface）**：
  ```python
  pattern: "Parallel"
  inputs: {Query: [B, H, S, 256], Key: [B, H, Skv, 256], Value: [B, H, Skv, 512]}
  customizable_functions:
    def scores_Mod(scores):
        return scores * mask          # 元素级: multiply with bool mask
    def scores_RowNorm_Online():
        def online_prologue():
            row_sum_wo_clamp = 0; row_sum = 0
            return row_sum_wo_clamp, row_sum
        def online_forward(scores, prev_sum_wo, prev_sum):
            row_sum_wo = prev_sum_wo + scores.reduceAbsSum()
            row_sum = max(row_sum_wo, 1)
            return scores / row_sum, row_sum_wo, row_sum
        def online_epilogue(...): ...
  ```

  2. **Customizable Function Lowering**：
     - Tracer 执行 `scores_Mod` 和 `scores_RowNorm_Online` → 生成 DAG：`[Mul node(mask)] → [ReduceAbsSum node] → [Max node] → [Div node]`
     - Node 分类：Mul → elementwise (SIMT mapping)，ReduceAbsSum → row-reduce (intra-warp reduction mapping)，Max → elementwise，Div → elementwise
     - 生成 hardware-mapped code snippets（elementwise → register-level SIMT fused ops，row-reduce → warp shuffle + sync reduction）

  3. **Scheduling Space Construction**：
     - IntermediateTensors 枚举：Q_tile, K_tile, scores_tile, weights_tile, V_tile, output_tile 及 customizable function 内部 intermediate tensors
     - DeviceConfig 约束：BaseTileShape（H100 wgmma tile: 64×128 or 128×128），MemoryInfo（Register 256KB/SM, Shared Memory 228KB/SM, Global Memory 80GB）

  4. **Two-layer Scheduling（Algorithm Fig.10）**：
     - **Tile Config Scheduling（外层）**：
       a. `EnumerateTiles(output_shape, D.basetile)` → 枚举所有合法 output tile sizes（受 DeviceConfig.basetile 约束，如 tile 必须对齐 MMA instruction tile shape）
       b. `PropagateTileGraphs(g, tiles)` → 将 output tile 通过 computation graph 反向传播到所有 IntermediateTensors（Q/K/V/scores/weights），确保相邻 tensors 共享相同 tile size（沿依赖边传播 tile shape constraint）
       c. 对每个 tile graph，调用 TileResourceScheduling
     - **Tile Resource Scheduling（内层）**：
       a. 初始化所有 IntermediateTensors memory location = "L0"（register，最小 I/O overhead）
       b. 按 `(use_count, tile_size)` 排序 tensors（高频使用+大 tile 优先留在高层 memory）
       c. 对每个 tensor_i：EnumerateUnsetAttributes（pipelineStage 枚举）→ 检查 memory constraint（total register/shared memory 使用 ≤ DeviceConfig.memoryInfo）
       d. 若无合法 plan → `LowerMemLocation(tensor_i.mem)`（Register → Shared Memory → Global Memory）→ 重新枚举
       e. 返回所有合法 execution plans

  5. **Profiling-based Selection**：对所有合法 plans，通过 hardware profiling（实际执行或 cost model）选 latency 最优者 → best_plan

  6. **Attention Runtime Code Generation**：
     - 选择 Parallel Pattern kernel template
     - Code inlining：将 elementwise ops (Mul/Div) fused 到 SIMT region of kernel loop，row-reduce ops (ReduceAbsSum) fused 为 warp reduction in softmax stage
     - Backend mapping：
       - **NVIDIA CUTE backend**：QKV load via TMA (cp.async.bulk) → QK^T via wgmma (Tensor Core) → customizable functions via CUDA core (SIMT + warp reduce) → PV via wgmma → output store via TMA
       - **NVIDIA TileLang backend**：同功能以 TileLang 表示 → TileLang compiler 生成 CUDA
       - **AMD TileLang backend**：Matrix Core 替代 Tensor Core，async copy 替代 TMA

  7. **输出**：单次 kernel launch 完成完整 attention computation（QK^T + mask + online L2 norm + PV + output），中间 tensors 全部驻留 on-chip memory（register/shared memory），无 HBM intermediate write。

  8. **编译时间**：82 秒（Mamba2 SSM, batch=1, seqlen=2048, H100, Table 4），显著低于传统 DL compiler (Ansor minutes to hours)。

  - **作用**：以声明式 Python 接口（22-90 LoC per attention variant）替代手写 CUDA/Triton kernel（400-3000 LoC），自动生成与手写 library 竞争或更优的跨硬件 attention kernel，消除 attention 变体的 "software lottery" 问题——任何用户可通过定义 template+functions 获得高性能 kernel，无需 expert CUDA 编程或 hardware-specific tuning。
