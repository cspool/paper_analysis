## WGMMA (Warpgroup Matrix Multiply-Accumulate)

术语是什么？通过联网搜索让回答具体和精准。

WGMMA (Warpgroup Matrix Multiply-Accumulate) 是 NVIDIA Hopper 架构（H100, SM90, Compute Capability 9.0）引入的新型Tensor Cores指令。与Ampere及之前架构中每个warp独立发射mma指令不同，WGMMA将同一CTA内多个warp编组为"warpgroup"（通常4个warp = 128 threads），warpgroup内所有warp协作执行单条wgmma指令完成大规模矩阵乘法。WGMMA的核心创新是异步执行模型：wgmma指令发射后立即返回（non-blocking），Tensor Cores在后台异步执行乘累加，warp可以继续发射后续指令——这与之前mma的同步阻塞模型（warp发射mma后stall等结果）根本不同。WGMMA提供了两种数据源模式：wgmma_RS（A和C来自Register，B来自Shared memory）和wgmma_SS（A/B/C均来自Shared memory）。Hopper的Tensor Cores峰值吞吐达989 TFLOPS (FP16 wgmma)，比A100的312 TFLOPS提升3×以上。

从硬件架构角度拆解术语：

WGMMA硬件执行流程（以BitDecoding on H100, decode step为例）：
1. **Warpgroup组织**：CTA内含4个warp (128 threads) 组成warpgroup，CTA其他warp可并行执行非wgmma工作（warp specialization）
2. **异步发射**：warpgroup leader发射`wgmma.fence` + `wgmma.commit_group` + `wgmma.m64n64k16` → 指令立即返回，Tensor Cores开始后台执行
3. **数据约束**：wgmma_RS要求B矩阵在shared memory中——这是BitDecoding使用STSM将dequantized K写入shared memory的原因（dequantized FP16 K从register→STSM→shared memory→wgmma_RS读取B矩阵）
4. **结果同步**：通过`wgmma.wait_group` barrier等待指定group的wgmma完成，读取accumulator
5. **Warp-specialized pipeline**：FlashAttention-3 (FA3) 利用此特性将producer warp（用TMA异步加载数据）和consumer warp（发射wgmma计算）解耦为异步pipeline，消除Ampere世代存在的35% throughput penalty

术语一般如何实现？如何使用？

WGMMA通过CUDA PTX指令（`wgmma.fence`、`wgmma.commit_group`、`wgmma.mma_async`、`wgmma.wait_group`）编程。CUTLASS 3.x提供C++ abstraction (WarpgroupMMA)。使用时需注意：① B矩阵必须在Shared Memory（wgmma_RS模式），这要求dequantized data先写入shared memory再被wgmma消费；② 需要显式fence/commit/wait管理异步执行和同步；③ accumulator layout与Ampere mma不同（m64n64k16 vs m16n8k16），为不同GPU世代设计portable kernel需处理layout差异；④ 可利用warp specialization将data movement (TMA) 和计算 (wgmma) 分离到不同warp以最大化overlap。

涉及论文标题：
- BitDecoding: Unlocking Tensor Cores for Long-Context LLMs with Low-Bit KV Cache

术语一般如何实现？如何使用？
SADDLE在Scheduler预标定阶段测量PIM和GPU的各自roofline ridge点（一次性offline profiling）。运行时Scheduler用effective micro-batch size和Shared Pool token count代入简化CI公式→与ridge阈值比较→决策。该方法的准确性依赖于CI估算的精确度——论文用代数近似（CI ≈ eff_bs for FC, CI ≈ avg_tokens × d_head for attention）而非完整profiling，牺牲精度换取亚微秒级决策延迟（仅占0.83% end-to-end latency）。消融实验验证了该trade-off的有效性：动态scheduling额外提升1.13×吞吐。

涉及论文标题：
- Adaptive Draft Sequence Length: Enhancing Speculative Decoding Throughput on PIM-Enabled Systems

