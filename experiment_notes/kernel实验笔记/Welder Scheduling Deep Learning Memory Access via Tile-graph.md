## Welder Scheduling Deep Learning Memory Access via Tile-graph

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现是WELDER的tile-graph调度引擎在GPU memory hierarchy上的kernel fusion和tile级数据调度。核心kernel调度机制包括：(1) Hierarchical Tile-Graph Execution——将DNN模型递归分层为register-level、shared-memory-level和global-memory-level的tile-graph，每层独立调度tile配置，通过四条硬件原语（Allocate/LoadTiles/ComputeTile/StoreTiles）递归展开执行；(2) Inter-Operator Tile Connection——通过SetConnect接口在同一memory level连接相邻operator-tile，使中间数据在shared memory或register中直接复用，消除global memory往返；(3) Hardware-Aligned Tile Search——枚举tile shape时加入硬件约束penalty：uncoalesced access按128B transaction计算额外traffic、并行度不足按core utilization比例加penalty、footprint超capacity则infinite penalty淘汰；(4) Block/ThreadIdx Remapping——支持Transpose等需线程重映射的算子连接，2D thread block映射到1D thread block；(5) TensorCore MMA 绑定——注册MMA axes annotations，对top-level operator-tiles绑定到warp执行MMA操作，加tile size为MLA fragment整数倍约束；(6) Shared Memory统一管理——对所有shared memory buffer做liveness分析+bestfit分配，考虑alignment要求（如32B对齐避免misaligned access），添加padding消除bank conflict。

  实验比较：端到端inference延迟 vs PyTorch/ONNXRuntime/Ansor/Rammer/TensorRT/FasterTransformer/BladeDISC/Nimble。消融实验：WELDER-none (无inter-operator tile connection，仅intra-operator) vs WELDER-base (仅register层连接) vs WELDER-full (register+shared memory连接)。Ablation结果：vs WELDER-none，WELDER-base减latency 52%、减kernel launch 67%、减global memory transactions 52%、减intermediate result size 66%；WELDER-full再减latency 29%、减kernel launches 60%、减transactions 25%、减IRS 65%。自动发现~300种fused subgraph pattern，其中89种含至少两个reduction-based operator不在Ansor规则覆盖范围内，最大fuse 48个算子为单kernel。对NeRF的7层MLP自动fuse为单GPU kernel（前6层TensorCore + 输出层SIMT Core，中间结果存shared memory），达5×加速。

- 后端平台是什么，配置是什么。
  NVIDIA V100 (16GB, SIMT Core + TensorCore)，NVIDIA RTX 3090 (Ampere)，AMD MI50 (16GB, ROCm 5.2.3)，GraphCore IPU (300MB device memory)。CUDA 11.0/11.3，ROCm 5.2.3。三级memory hierarchy：global memory (DRAM)、shared memory、register。已扩展支持host memory作为额外层处理超大输入（如UNet 8k×8k图像）。

- 评估性能的软件/脚本是什么。修改了什么。
  WELDER基于TVM用于kernel schedule编写、Roller用于枚举高效tile配置、Rammer用于端到端图优化。kernel评估通过硬件profiling（直接测量latency）。修改：(1) Tile-level kernel fusion——通过SetConnect/Propagate接口自动将TVM生成的独立kernel组合为fused kernel；(2) Load/Store Rewriting——TIR pass将standalone kernel的global memory访问改写为shared memory访问；(3) Block/ThreadIdx Remapping——从tensor expression推导Transpose等的blockIdx映射，2D→1D thread block映射；(4) Shared Memory Management——liveness分析+bestfit算法统一管理所有shared memory buffer；(5) Register-Level Connection——使用TVM compute_inline实现register级tile连接；(6) TensorCore annotations——对GEMM/BatchMatmul/Conv标注Warp-Level MMA axes。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  开源地址：https://github.com/microsoft/nnfusion/tree/osdi2023welder

  评估原理：WELDER将DNN kernel执行建模为分层tile-graph的递归展开。每个tile-graph的kernel从global memory加载tile → shared memory中复用中间数据 → register中执行compute → 结果通过各层写回。性能优势来自：shared memory level tile connection消除inter-operator global memory往返、register level connection消除kernel launch overhead、解析cost模型驱动最优tile配置。

  Kernel输入到性能输出全过程（以BERT attention block在V100上FP16 TensorCore执行为例，Q*K Matmul → Softmax fusion）：
  ```
  Host: WELDER编译BERT ONNX graph → 生成fused kernel binary
  Host: load input tensors (Q, K) in GPU DRAM

  GPU Kernel执行 (single fused kernel):

  Step 1 — Global→Shared Memory Load (LoadTiles):
    从DRAM加载Q tile [BM×BK] 到 shared memory buffer 0
    从DRAM加载K tile [BK×BN] 到 shared memory buffer 1
    (coalesced 128B transactions, aligned)

  Step 2 — Matmul Operator-Tile (ComputeTile, TensorCore):
    从 shared memory 加载 Q_tile → registers (ldmatrix, warp-level)
    从 shared memory 加载 K_tile → registers (ldmatrix)
    Warp-Level MMA: mma.sync.aligned.m16n8k16
      C_accum += Q_frag[16×16] × K_frag[16×16]
    // K维循环: 64/16 = 4次MMA迭代

  Step 3 — Inter-Operator Tile Connection (shared memory):
    Matmul输出 tile [BM×BN] 留在 shared memory ← SetConnect(edge, SharedMem)
    Softmax operator-tile 直接从 shared memory 读取中间结果
    // 消除了 Matmul→global memory write + Softmax→global memory read

  Step 4 — Softmax Operator-Tile (ComputeTile, SIMT Core):
    for each row in [BM×BN]:
      max_val = warp_reduce_max(row)
      exp_vals = exp(row - max_val)
      sum_exp = warp_reduce_sum(exp_vals)
      result = exp_vals / sum_exp
    // BM=16, BN=128 → 16行并行softmax

  Step 5 — Shared→Global Memory Store (StoreTiles):
    Softmax输出 [BM×BN] 从 shared memory 写回 DRAM ← StoreTiles

  Step 6 — Tile循环:
    重复 Step 1-5 覆盖全部 Q[seq_len×hidden_dim]@K^T 输出tiles
    // 24,576个输出tiles for BERT seq_len=128, hidden=768

  性能测量:
    - latency: CUDA Event start/stop, warmup + 多次迭代取平均
    - speedup vs Ansor: WELDER fused kernel 0.29ms vs Ansor separate kernels 0.36ms (1.26×)
    - memory traffic: 840MB (unfused, output tile [4×128])
                    → 264MB (fused, optimal output tile [16×128])
                    节省69% global memory traffic
    - TensorCore FP16: 2.72× vs Nimble, 1.53× vs TensorRT (V100)
  ```

  关键调度设计要点：
  - Inter-layer independence: L0 tile-graph的traffic仅由L0 output tile shape决定 → 各层独立优化
  - Intra-layer independence: 同层不同sub-graph的traffic互相独立 → 并行搜索
  - Propagation: 从output tile shape链式推断所有input tile shape → 自动对齐tile配置
  - Traffic cost model: Σ(input_tile_sizes + output_tile_size) × num_tile_graphs → 指导最优tile搜索
  - Reduction tiling: 含reduction轴的input tile可被partition为更小tile，顺序加载accumulate到output tile
  - Block size alignment: 所有operator-tile的线程数取GCD作为统一block size (≥128, ≤1024)
  - 2D→1D thread block mapping: 2D thread block可映射到1D，只要总线程数相等
