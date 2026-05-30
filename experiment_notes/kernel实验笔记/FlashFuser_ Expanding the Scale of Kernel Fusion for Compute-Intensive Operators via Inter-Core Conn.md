## FlashFuser: Expanding the Scale of Kernel Fusion for Compute-Intensive Operators via Inter-Core Connection

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  FlashFuser 的 kernel 调度核心是 DSM-based cluster 级别的 loop scheduling、tile selection 和 resource mapping。实现包括：(1) Loop Scheduling——将算子链的共依赖 loop 维度统一为集合 X={x_0,...,x_{J-1}}，划分为 Spatial dimension（多 SM 并行计算）和 Temporal dimension（单 SM 串行计算），共 41 种组合（2-4 个 spatial dims）。不同 loop schedule 影响中间 tensor 需要缓存的大小——MLNK order 需要存储完整 C tensor 可能 spill to DSM，MNLK order 每次迭代仅产生 partial E 结果；(2) Tile Selection——两级 hierarchical tiling：cluster-level tile（dictates work distribution across clusters，影响 inter-block data exchange）和 block-level tile（governs per-block tile size，影响 reg vs SMEM 分配决策）；(3) Resource Mapping——贪心 heuristic 将 reusable tensor 从 reg→SMEM→DSM 逐级放置，超出容量则 spill 到下一级，同时按 dsm_comm 定义的 cluster size 和 data footprint 计算 DSM traffic；(4) dsm_comm primitives——基于 TMA + mbarrier 实现三个 DSM 级通信原语：dsm_all_exchange（cluster 内 AllReduce/Mul）、dsm_shuffle（ring communication 交换 C tile 给不同 compute units）、dsm_reduce_scatter（hierarchical intra-cluster + inter-cluster atomic reduction via TMA cp.reduce.async.bulk）。
  实验比较：(1) GEMM/Gated FFN/Conv chains 的 kernel speedup vs PyTorch、TensorRT、BOLT、Chimera、Relay、TASO；(2) Ablation study——全系统 (DC+DA+SE) vs DC+DA (random config) vs DA only (仅 SMEM/global memory fusion)；(3) dsm_comm primitive bandwidth/utilization 随 cluster size 变化（1/2/4/8/16 SMs）；(4) 全局显存访问量 vs PyTorch (Nsight Compute profiling)；(5) Cost model 准确性和 Top-K 选择分析；(6) 搜索效率 vs Brute-Force；(7) 端到端 SGLang 推理 speedup。

- 后端平台是什么，配置是什么。
  NVIDIA H100 GPU (SXM)，132 SMs，HBM bandwidth 3.35 TB/s，SMEM 227KB/SM。DSM bandwidth 随 cluster size 变化（cluster=2: ~8TB/s; cluster=16: ~4TB/s，均高于 global memory bandwidth 3.35 TB/s）。DSM latency 在 cluster size=2 时约 20ns（vs global memory ~280ns），随 cluster size 增大而增长。Host: 双路 Intel Xeon Platinum 8468 (96 cores, 2.10GHz)。CUDA 12.4, PyTorch 2.6, CUTLASS, Nsight Compute 2025.2.0。

- 评估性能的软件/脚本是什么。修改了什么。
  基于 CUTLASS kernel 模板实现 FlashFuser 的 fused CUDA kernel。修改包括：
  (1) **Prologue**：扩展 semaphore 初始化到 DSM，准备 inter-CTA communication
  (2) **Mainloop 注入 dsm_comm**：
    - GEMM0 accumulation loop 完成后：dsm_all_exchange 执行 cluster 内 AllReduce（Standard FFN）或 Mul（Gated FFN SiLU gating branch multiply）
    - GEMM1 consumer accumulation loop 中：dsm_shuffle 实现 ring communication 在 Shuffle Group 内交换 intermediate C tile
  (3) **Epilogue**：dsm_reduce_scatter 执行 hierarchical two-level reduction——先 intra-cluster reduce（dsm_reduce_scatter），再 inter-cluster reduce（TMA cp.reduce.async.bulk 原子归约）
  (4) **Mbarrier-based synchronization**：不同于 CUTLASS 原生的 all-to-one cluster-sync，使用 mbarrier 实现仅必要 CTA group 之间的 many-to-many 同步
  (5) **Two approaches for Gated FFN**：spatial partitioning (cls_k=2, 不同 Block group 执行两个 GEMM branch) 最大化并行度 或 sequential execution within single Block 最小化 DSM communication

  Benchmark 脚本：Nsight Compute profiling 测量全局显存访问量；CUDA event timing 测量 kernel execution time；TFLOPs 计算 = GEMM FLOPs / runtime。End-to-end 通过替换 SGLang 的 attention+FFN kernel 测量 throughput。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  论文未明确声明独立开源仓库。基于 CUTLASS (https://github.com/NVIDIA/cutlass) 构建的代码框架。

  评估原理与流程（以 Standard FFN GPT-6.7B GEMM chain G5 为例, M=128, N=16384, K=4096, L=4096, H100 SXM）：

  1. **搜索最优执行计划（离线）**：
     - 输入 problem size (M,N,K,L) + device info (H100 memory hierarchy)
     - Search Engine 枚举 LoopSchedule × TilingSize × ResourceMapping 候选
     - Pruning 5 条规则后约 1.15×10^6 候选
     - Dataflow Analyzer 对每个候选计算 D_V (data movement volume per memory level)
     - Cost model 按 C = max(V_l/B_l) 选择 minmax Top-11 候选
     - 编译 Top-11 为 CUDA kernel → H100 硬件 profiling → 选最优

  2. **最优 kernel 结构**（以 cluster size (2, 4, 2, 4) 为例）：
     - Cluster 包含 cls_m×cls_n×cls_k×cls_l = 2×4×2×4 = 64 Blocks（clusters）
     - **GEMM0 Phase**:
       a. 每个 Block 加载 A tile (blk_m × blk_k) 和 B tile (blk_k × blk_n) from HBM → SMEM
       b. Tensor core WGMMA: partial C = A×B (FP16/BF16 → FP32 accumulate)
       c. cls_k=2 表示 K 维度 spatial partition 到 2 个并行 Block → 需 intra-cluster accumulation
       d. dsm_all_exchange: 在 cluster 内执行 AllReduce，每个 Block 获得完整 accumulated intermediate C tile
       e. 中间 C tile 驻留 DSM（200+ KB, 超出 SMEM 227KB 限制）
     - **GEMM1 Phase**:
       f. dsm_shuffle: 在 Shuffle Group（cls_shuffle = cls_l/cls_k = 2 Blocks）内 ring communication 交换 C tile slices
       g. 每个 Block 获得所需 C tile 后加载 D tile → Tensor core WGMMA: partial E = C×D
     - **Store Phase**:
       h. dsm_reduce_scatter: intra-cluster reduce (多个 Shuffle Groups 的 partial E 累加), Scatter pattern 下每个 Block 仅负责写回一部分 output
       i. inter_cluster_reduce: TMA cp.reduce.async.bulk 跨 cluster atomic reduction

  3. **Kernel 执行与测量**：
     - CUDA event 记录 kernel launch→completion wall time
     - TFLOPs = (2×M×N×K + 2×M×L×N) / runtime = (2×128×16384×4096 + 2×128×4096×16384) / 1e12 ≈ 34.4 GFLOPs per operator / runtime(ms)
     - Nsight Compute profiling 测量 global memory access (bytes read/written)
     - 对比 PyTorch（cuBLAS: 2 次独立 GEMM kernel，中间 C 经 HBM round-trip）——FlashFuser 减少 58% global memory access
     - 结果：G5 kernel speedup 约 4.1× over Chimera (SOTA compiler), 3.1× over PyTorch

  4. **dsm_comm bandwidth profiling**（独立 benchmark）：
     - 传输 32768×32768 tensor，切为 128×128 tiles
     - 在 cluster 内执行 dsm_comm 操作（排除 global read/store overhead）
     - 循环 1000 次测量 bandwidth
     - Bandwidth utilization = measured_bw / peak_DSM_bw (per cluster size)
     - Shuffle > Reduce ≈ Mul (Reduce/Mul 含额外计算 overhead)
     - Bandwidth utilization 随 cluster size 增大保持稳定

  5. **端到端 (SGLang)**：
     - 替换 SGLang 的 attention+FFN kernel 为 FlashFuser 预编译 kernel
     - Real-world models (Llama-3.2-3B, Qwen2.5, Qwen3 系列)
     - Sequence length=512, varying batch size
     - 平均 E2E speedup 1.24×（all scenarios），大模型 (70B/14B/32B) 上 1.16×-1.22×

  6. **Ablation 验证**：
     - 'DA only': 仅用 SMEM/global memory fusion (无 DSM) → 1.52× vs no-fusion baseline
     - 'DC+DA': DSM + Dataflow Analyzer, 随机 search → 2.11× vs baseline
     - 'All' (DC+DA+SE): 全系统 → 3.29× vs baseline
     - 说明 Search Engine (SE) 贡献最大增量 (2.11→3.29 = 1.56×)
