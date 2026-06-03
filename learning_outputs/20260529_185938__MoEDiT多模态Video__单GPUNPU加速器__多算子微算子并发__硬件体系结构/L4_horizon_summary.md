# L4: Kernel 调度 — 水平分类总结

## 问题覆盖概览

| Q-ID | 问题焦点 | 覆盖方法数 | 关键方法 |
|------|----------|-----------|----------|
| Q4.1 | Tile 切分与调度策略（tile size/shape、thread block→SM 映射、persistent kernel、grid-stride loop、dynamic parallelism） | 11 | WELDER, ThunderKittens LCSF, MPK tGraph, FlashMoE, Fused MoE, SLA, Block Order, Ascend TBE/TIK, XLA TPU, AMD wave-level, PAT multi-tile |
| Q4.2 | 指令编排与软件流水线（ILP、warp scheduling、async copy/TMA、dual-issue、VLIW） | 9 | Warp Scheduler, Software Pipelining cp.async, Hopper TMA+WS, ILP Tensor-Vector Parallelism, ImFP, Drawloom multi-stage, FlashMoE actor-based WS, RoMeo, TileLang T.Pipelined |
| Q4.3 | Memory Hierarchy 对 kernel 并发设计的影响（register、SMEM bank conflict、L2 cache thrashing、coalesced access、HBM bandwidth） | 10 | Register Tiling (MoE/DiT), SMEM Bank Conflict Avoidance, L2 Cache Affinity, Block Order Scheduling, Coalesced Access, LYNX MoE BW Reduction, Fused MoE, Kitsune L2 Ring Buffer, FlashAttention IO-Aware, PD Contention Guard |
| Q4.4 | Tile size/shape 根据硬件体系结构优化以最大化多算子/微算子并发 | 8 | FlashAttention IO-Aware Tiling, FlashMoE Persistent Kernel, CUTLASS v3 Grouped GEMM, ThunderKittens, Infera Zero-Tuning, ChituDiffusion dEngine, WELDER, PIT for MoE |
| Q4.5 | Kernel 调度框架/编译器后端/kernel 生成方法与编程模型 | 10 | Triton, CUTLASS v3, CUDA Graph, CUDA Stream, FlashMoE, MPK/Mirage, CANN/TBE/TIK, XLA/TPU, MLIR/Diffuse, KernelEvolve |
| Q4.6 | 各框架/方法具体实现细节、benchmark 与实验环境 | 8 | Triton Fused MoE (vLLM), SonicMoE 8-kernel suite, CUTLASS v3 Grouped GEMM, CUDA Graph MoE, Mirage MPK, CANN/TBE CoC, XLA/TPU GLaM, FP8 MoE PTQ |

---

## 按实验环境分类

| 分类 | 方法 | 具体方法描述 | 硬件平台 | Benchmark | 实现框架 | 来源 |
|------|------|-------------|----------|-----------|----------|------|
| **GPU/NVIDIA** | **WELDER tile search** | SubGraphTiling 中三级硬件感知 penalty 驱动 tile shape 搜索：Penalty 1 — Uncoalesced Memory Access（128B transaction 对齐）；Penalty 2 — Parallelism Underutilization（tile 数 < 128 → SM 吃不饱）；Penalty 3 — Capacity Overflow（footprint > target memory capacity）。额外 MMA Tensor Core 整除约束：M/N/K 必须整除 16（FP16 `mma.m16n8k16` fragment 粒度）。枚举空间 = 各维度 tile size 笛卡尔积，按 adjusted MemTraffic 排序 | V100 (80 SMs, 4 warp schedulers/SM) | MemTraffic (adjusted) | WELDER Compiler (SubGraphTiling) | Q4.1, Q4.4 |
| **GPU/NVIDIA** | **ThunderKittens LCSF** | 三层 tile 抽象直接映射 GPU 物理硬件：Warp → Register Tile (16×16 FP16，匹配 `mma.m16n8k16`)；Warpgroup → Shared Tile（自动 swizzle on 32/64/128B boundaries，naive row-major→8-way conflict，TK swizzle 64B→0-way conflict）；CTA → LCSF 模板（Load/Compute/Store/Finish 异步 producer-consumer pipeline，N-stage buffer N=1/2/3/4）。Grid 级：Persistent Grid（grid = SM_count blocks） + Block Order {8,N,M/8} 3D stride 最大化 L2 reuse。Pipeline depth 影响：1-stage 260 TFLOPS → 4-stage 760 TFLOPS GEMM 4096（H100） | H100 (132 SMs, 228KB SMEM/SM), A100 (108 SMs, 164KB SMEM/SM) | GEMM M=N=K=16384: 760 TFLOPS (4-stage); Block Order: 3D stride HBM 982 GB/s→805 TFLOPS vs row-major 3070 GB/s→392 TFLOPS | TK C++ library (<1MB), CUTLASS backend | Q4.1, Q4.4 |
| **GPU/NVIDIA** | **Mirage Persistent Kernel (MPK)** | 将整个 tensor program 编译为单个 mega-kernel，通过 SM 级任务图 (tGraph) 实现跨算子微算子并发。编译流程：(1) Operator decomposition 按 output 维沿 SM 数 partition；(2) Dependency analysis 枚举 producer-consumer task pair；(3) Event fusion 消除冗余同步点；(4) tGraph normalization（每 task ≤1 dep + 1 trig event）；(5) tGraph linearization (BFS)；(6) Device memory 紧凑存储。In-kernel runtime：每 SM 独立 event-driven task loop——spin-wait dep event → 执行 task（MatMul/Attention/RMSNorm/AllReduce）→ atomicAdd trig event counter → 激活下游。Scheduler:Worker = 16:128 warps 分区（H100: 4 SMs scheduler + 128 SMs worker；B200: 16+144） | A100 (108 SMs), H100 (132 SMs), B200 (148 SMs) | Qwen3-8B decode: 12.5ms/token (MPK) vs 14.5ms (vLLM); cross-task pipelining 1.2-1.3×; overall 1.0-1.7× | MPK Compiler (C++ 40K + CUDA 84K + Python 10K), Mirage superoptimizer, NVSHMEM | Q4.1, Q4.5, Q4.6 |
| **GPU/NVIDIA** | **FlashMoE Persistent Kernel** | 单 persistent kernel 融合 Gate + Dispatch + Expert FFN (GEMM0+GELU+GEMM1) + Combine，Actor 模型三种角色：(1) Processor (N-1 blocks) — CUTLASS GEMM + element-wise + NVSHMEM DMA；(2) Scheduler (OS block 内 1 warp) — work-conserving 多线程调度、event-driven polling；(3) Subscriber (OS block 内 3 warps) — 解码远程 tile packets → 通知 scheduler。Tile dimension=(128,64)，128 threads/block，CUTLASS ThreadBlock 128×128×32。对称 Tensor Layout（overprovision 2×r 倍内存）实现 write-write conflict-free one-sided DMA。1 次 launch 替代 Megatron 432-550 次 launch | 8×H100 80GB NVLink, CUDA 12.8 | 93.17% SM util (vs 14% Megatron-TE), 6× latency speedup, 5.7× throughput; FP32 FlashMoE 仍胜 FP16 baseline | CUDA/C++, CUTLASS, NVSHMEM, actor model | Q4.1, Q4.4, Q4.5 |
| **GPU/NVIDIA** | **Fused MoE Kernel (vLLM/Triton)** | Expert FFN 的 GEMM (gate+up) + SiLU + GEMM (down) + atomic scatter-add 融合为单 Triton kernel。BLOCK_M=64 (H100 SMEM 228KB 约束，H=4096, ffn_dim=14336)。Indirect token gather 通过 `sorted_token_ids` 按 expert_id 排序——同 expert tokens 连续→最大化 L2 cache 命中。Atomic scatter-add 处理 MoE top-k≥2 的多 expert 输出累积（H100 L2 atomic unit ~100M atomics/s）。MoE-SpeQ fuseMoE 变体：K=1408, N=2048 时所有 expert gate_proj+up_proj+SiLU+down_proj 融合为单 launch→batch 多 expert tokens 增大有效矩阵维度→提升 occupancy，fused kernel 贡献 31.8% 加速（8.88→13.02 tok/s） | H100, A100 | 15-20% throughput↑ (FP16); 25-30%↑ (FP8); 12-20% latency↓ | Triton (block-level SPMD), vLLM | Q4.1, Q4.3, Q4.6 |
| **GPU/NVIDIA** | **FlashAttention IO-Aware Tiling** | Online softmax rescaling + HBM→SRAM tiling：将 N×N attention matrix 完全在 SRAM 中计算而不 materialize 到 HBM。Tile size 推导：B_c = ceil(M / (4d))（M=SMEM 容量），B_r = min(B_c, d)。A100 d=64: B_c=384, B_r=64；H100 d=128: B_c≈228, B_r=128。HBM 访问量 552KB vs 标准 attention 8.5MB→~15× reduction。FA-3 (Hopper): warp-specialized producer-consumer pipeline + WGMMA + TMA，H100 达到 840 TFLOPs/s (85% 峰值 FP16)，FP8: 1.3 PFLOPs/s。register 约束：2 CTA/SM，occupancy 25%（有意为之——低 occupancy 换更多 regs/CTA） | A100 (108 SMs, 164KB SMEM), H100 (132 SMs, 228KB SMEM) | HBM 减少 ~15×; 2-4× speedup; FA-3 H100 840 TFLOPs/s | CUDA C++ / Triton / PyTorch 2.0+ SDPA; CUTLASS WGMMA/TMA for FA-3 | Q4.3, Q4.4 |
| **GPU/NVIDIA** | **SLA Fused Sparse-Linear Attention** | DiT denoising attention kernel：将稀疏 FlashAttention、线性 attention、negligible block skip 三种模式融合到单 kernel。Per-Q-block loop 条件执行——CRITICAL block→Tensor Core MMA (O(N²))，MARGINAL block→CUDA Core 线性 attention (O(1) via pre-aggregation)，NEGLIGIBLE→skip。DiT 特殊性：diffusion step 固定 latent resolution→tile size 可静态配置 | NVIDIA GPU | Kernel-only 13.7× vs FlashAttention-2 | CUDA C++ | Q4.1 |
| **GPU/NVIDIA** | **CUTLASS v3 Grouped GEMM** | 分层 tile 抽象：ThreadBlock (128×128×32) → Warp (64×64×32) → MMA Instruction (16×8×16 FP16 / 16×8×32 FP8)。MoE Grouped GEMM: Persistent kernel + TMA (HBM↔SMEM 直传，不经 register) + WGMMA (warp group async MMA) + FP8 MMA (E4M3×E4M3→FP32，1979 TFLOPS peak)。Software pipeline: NumStages=3-4，circular SMEM buffer。setmaxnreg 寄存器重分配：producer warp 释放寄存器（仅需 32 个），consumer warpgroup 用满 ~232 个。调度策略：Fixed-Split / Stream-K（沿 K 维分解动态分配）/ Token Rounding（SonicMoE 方案，消除 padding FLOPs） | H100 (SM90, Hopper), A100 (SM80, Ampere) | ~75% vs contiguous-packed ideal (Grouped GEMM); SonicMoE dH kernel ~420 TFLOPS (42% peak), Forward ~550 TFLOPS (56% peak) | CUTLASS v3 C++ template, CuTe DSL | Q4.4, Q4.5, Q4.6 |
| **GPU/NVIDIA** | **Hopper TMA + Warp Specialization 流水线** | 硬件辅助的 warp-specialized pipeline：TMA（独立硬件 DMA 引擎，单线程发起 HBM↔SMEM，不占 CUDA Core/warp scheduler，无寄存器中转）+ wgmma.mma_async（warp group 异步 Tensor Core 指令）+ Named Barriers / mbarrier（硬件同步原语，arrive ~10-20 cycles，单周期 wait）+ Warp Specialization（producer warp 1-2 个专做 TMA load，consumer warpgroup 6-7 warps 专做 WGMMA compute）。TMA 在仅 2KB 消息大小就能达到 74% 峰值 NVLink 带宽。Multicast 能力：单次操作将相同 tile 广播到 cluster 内多个 SM 的 shared memory | H100 (Hopper, SM90) | TMA 2KB→74% NVLink peak bandwidth; Pipeline PIPE=2-3; WGMMA 85% Tensor Core utilization | CUTLASS 3.x + CuTe, TileLang T.Pipelined, CUDA ≥12.0 | Q4.2, Q4.4, Q4.5 |
| **GPU/NVIDIA** | **ILP Tensor-Vector Parallelism (FlashAttention-T)** | 在 fused attention kernel 中将 softmax 拆分为 tensorized 部分（repurposed MMA on Tensor Core：scaling/FMA/rowsum）和 vectorized 部分（CUDA Core：REDUX max/MUFU.EX2 exp），通过 ILP interleaving 使两部分在同一 warp 内并行——Tensor MMA issue bubble 被 vector 指令填充。Horizontal split ratio≈1:1。Hopper TLP 替代方案：WGMMA row-sum 加入下一 iteration batch，与另一 warpgroup 的 vector S/O rescaling 并行→vector interval ratio 降至 2.7% | Ampere (ILP) / Hopper (TLP) | Hopper TLP vector interval ratio 2.7%; t'_softmax < t_vec (baseline) | 手写 CUDA PTX inline assembly | Q4.2 |
| **GPU/NVIDIA** | **ImFP (Implicit Fine-Grained Pipeline) / LiquidGEMM** | Single-producer multiple-consumer 模型替代 ExCP 三阶段流水线。Load WG (4 warps) 仅操作 TMA→SMEM；Compute WG_0 & WG_1 (各 4 warps) 从 SMEM 直接一站式 dequant+MMA（消除 SMEM↔RF round-trip）。跨 WG 异构重叠：WG_0 在 CUDA Core 做 dequant 时 WG_1 在 Tensor Core 做 MMA——天然重叠无 barrier。LiquidQuant (LQQ)：仅需 IMAD + XOR 两条指令处理四个元素，α=0.875 指令/element（vs QServe α≥10）。Dual-MMA packed layout：LDS.128 单指令加载 32 个 UINT4 | H100/H800 (Hopper) | α=0.875 指令/element; 消除 SMEM↔RF round-trip | CUTLASS/CuTe warp-specialized kernel | Q4.2 |
| **GPU/NVIDIA** | **Multi-stage Register Pipeline (Drawloom)** | SpMV on Tensor Core 的 5 阶段流水线：FillSMEM (async-copy)→FillREG (SMEM→REG+LDG)→Comp (TC MMA)→EmptySMEM→EmptyREG。delaySMEM=1 (double buffering), delayREG=2 (3 组 REG set 轮流)。消融：v4 +Multi-stage Pipeline vs v3 平均 1.46×，warp stall 改善 3.02×-3.13×，memory throughput +2.61×-2.75× | Ampere (A100) | warp stall 改善 3.02×-3.13×; mip1 达 5.68× | Drawloom CUDA: cp.async + LDG + TC MMA | Q4.2 |
| **GPU/NVIDIA** | **CUDA Graph (Kernel Graph)** | 将一系列 kernel launch 及其依赖预录制为 GPU-executable DAG。单次 `cudaGraphLaunch` 替代 N 次独立 kernel launch→eliminate CPU-GPU round-trip。支持 DAG fork/join 并发。限制：动态 shape 需重建 graph（40-100ms）；不兼容 MoE conditional routing（CUDA Graph 无法表达 conditional dispatch） | NVIDIA GPU (CUDA 10+) | Launch overhead: ~2-5μs replay vs ~5-20μs per-kernel (90-98%↓); 11 kernel→1 launch | CUDA Runtime API | Q4.5, Q4.6 |
| **GPU/NVIDIA** | **CUDA Stream 多流并发** | 多 stream 内 kernel 时间片/SM 空间分区并发。H100 最多 128 并发 stream，实际并发受 SM 资源（registers、SMEM）限制。Wave Quantization 效应：per-expert token 不均时 thread block 数 << SM 数→严重 SM idle，多 stream 不能解决→需 persistent kernel + work stealing | NVIDIA GPU | 实际并发受 SM 资源限制; 不解决 wave quantization | CUDA Runtime API | Q4.5 |
| **GPU/NVIDIA** | **SonicMoE 8-kernel suite** | CuTe-DSL (CUTLASS) 实现的 8 个 MoE kernel：Gather fusion + Ping-Pong scheduling (2 consumer warpgroups 交替 MMA+epilogue) + Token Rounding（per-expert token 数舍入到 128 倍数消除 padding FLOPs）。3 warpgroups/CTA：1 producer (4 warps, TMA) + 2 consumer (8 warps, WGMMA+epilogue) | H100, B300 | dH kernel ~420 TFLOPS (42% peak H100), ~700 TFLOPS (B300); Forward ~550 TFLOPS; 端到端训练 213B tok/day (64 H100) | CuTe-DSL, CUTLASS backend | Q4.6 |
| **GPU/NVIDIA** | **ChituDiffusion dEngine (Difflow)** | DiT diffusion 模型的数据属性感知编译：对每个 input property (fingerprint hash) 预编译多个 dEngine（tile 配置 + kernel variant），运行时根据 data property 动态选择最优 dEngine。Ragged operation regularization→等价 regular operator + round-robin tile→thread block mapping。OLS regression 辅助延迟估计 | A100/H100 | 1.58× avg throughput; 2.2× for correlative requests (H100) | Python/C++ / Triton / FlashAttention | Q4.4 |
| **GPU/NVIDIA** | **Infera Tile-Based Zero-Tuning** | 编译时全静态分析驱动多配置 micro-kernel 生成：tile shape 枚举→Multi-Version Generation（reg 64/96/128, smem 48/80/112/144 KiB, pipeline 2/3/4 stages）→Static Analysis（resource constraint + ILP 分析 + arithmetic intensity）→运行时选择。无需 GPU profiling，比 Ansor/MetaSchedule 快 2-3 个数量级 | V100/A100 | 2-3 orders of magnitude faster compile than search-based | Infera Compiler | Q4.4 |
| **GPU/NVIDIA** | **Kitsune L2-Resident Ring Buffer** | Inter-CTA queue pin 在 L2 cache 中实现 producer-consumer tile 传递：64-256KB tile 直接通过 L2 传输（SMEM→L2→SMEM），避免 HBM round-trip。依赖 L2 persistence——若被其他并发 kernel 刷掉则性能急剧下降 | H100 | 避免 HBM round-trip for inter-CTA data | CUDA + L2 persistence API | Q4.3 |
| **GPU/NVIDIA** | **LYNX MoE Bandwidth Reduction** | Batch-level dynamic expert selection 将 active expert 数从 ~55 降至 ~15→HBM 数据搬运量减少 ~73%→latency 降低 1.09-1.30×。MoE decode 带宽分解：Expert Weight Load 42%（核心瓶颈），Expert Computation 27%，Attention 19%，Other 12% | H200 | ~73% HBM traffic↓; 1.09-1.30× latency↓ | LYNX runtime | Q4.3 |
| **GPU/NVIDIA** | **RoMeo Separate-Kernels + Async** | 不同精度组合独立 kernel + CUDA Stream 异步并发。通过控制 shared memory 消耗释放寄存器用于 ILP。Separate-kernels 策略最大化 per-kernel register/occupancy 调优灵活性 | RTX 4090 (Ada Lovelace) | up to 4.68× vs BF16 GEMM; up to 2.10× end-to-end | CUTLASS + CUDA Graph + Triton | Q4.2 |
| **GPU/NVIDIA** | **TileLang T.Pipelined** | 编译器自动分析 loop body buffer 依赖→生成架构特定 pipeline 指令序列：Ampere→cp.async、Hopper→TMA、CDNA→buffer_load_dword。Live Variable Analysis→确定同步点→自动插入 barrier。Hopper 架构自动应用 warp specialization。FlashAttention 实现（~70 行 Python）达 FA-3（手写 CUDA）98% 性能 | Ampere/Hopper/AMD CDNA | 70 行 Python = 98% FA-3 性能 | TileLang JIT Compiler | Q4.2 |
| **NPU/Ascend** | **CANN/TBE/TIK (昇腾 Da Vinci)** | 达芬奇架构三单元异构执行：AI Core/Cube Unit（16×16×16 MAC 脉动阵列→MatMul 专用）+ AI Vector/Vector Unit（256-bit SIMD→element-wise/activation/AllReduce）+ Scalar Unit（控制流/地址计算）+ MTE（Memory Transfer Engine，HBM↔UB 异步 DMA）。三类单元有独立指令队列→支持同步取指并行执行（Cube 做 MatMul 时 Vector 同时做 AllReduce/LayerNorm）。CANN 软件栈：GE (Graph Engine) IR lowering + operator fusion→TBE (Tensor Boost Engine) Python DSL 自动 tiling→TIK (Tensor Iterator Kernel) C++ API 显式 buffer 管理→NPU 指令。Unified Buffer 256KB（类似 GPU SMEM）。CoC (Communication over Computation)：MTE 在 AI Core 计算当前 micro-batch 时同时发起下一 micro-batch 远程 DMA | Ascend 910B (64GB HBM, ~1.2TB/s BW), Ascend 910B3 (64GB HBM, 1.6TB/s, 20 AI Cores, 313 TFLOPS FP16) | 笔记未提供 kernel 级 MFU/TFLOPs; EPD-Serve operator-level 空间复用; ETR CoC 全节点训练加速 | CANN GE + TBE Python DSL + TIK C++ API; Ascend C (SPMD); PyTorch Ascend Adapter | Q4.1, Q4.5, Q4.6 |
| **TPU** | **XLA/TPU Kernel 调度** | TPU v5e/v5p MXU 128×128 (v5e) / 128×128 or 128×256 (v5p) systolic array→tile 必须适配 128 宽度（不可拆分，与 GPU 可组合 16×16 MMA tile 形成对比）。XLA 编译全自动：JAX HLO→Operator fusion（GemmFusion pass 将多 GEMM 合并为更大矩阵乘提升 MXU occupancy）→Layout assignment→VLIW bundle 打包→TPU binary。VPU memory（类似 GPU SMEM 但容量更大 ~16MB）→tile 需 fit VPU。无运行时并发——XLA compiler 静态 schedule 所有操作。VLIW 架构依赖于编译器的静态指令编排 vs GPU SIMT warp scheduler 动态调度 | TPU v4 (MXU 128×128), TPU v5e (128×128), TPU v5p (128×128 or 128×256) | GLaM 1.2T 参数训练 on 1,024 TPU-v4: 50-62% compute utilization; 笔记无 v5e/v5p MoE/DiT 推理 benchmark | JAX + XLA; HLO → LLO → TPU VLIW bundle | Q4.1, Q4.5, Q4.6 |
| **AMD GPU** | **AMD Wave Specialization / TileLang CDNA** | AMD CDNA3 静态寄存器分配：每 SIMD 512 regs 在所有 wave 间平均分配，producer wave 占用 register 但 consumer 无法回收→缩小 output tile size。无 TMA/wgmma/mbarrier→producer-consumer 退化（4P+8C: 893 TFLOPS vs 0P+8C: 1281 TFLOPS）。推荐 8-WAVE PING-PONG（无专职 producer）替代 producer-consumer。TileLang CDNA 路径使用 `s_waitcnt lgkmcnt` + `buffer_load_dword lds` 指令 | MI300X (CDNA3, 304 CU, 5.3 TB/s HBM3) | 0P+8C: 1281 TFLOPS (no producer-consumer overhead) | HIP/ROCm; TileLang CDNA backend | Q4.1, Q4.2 |
| **跨平台** | **MLIR/Diffuse (Multi-Dialect IR)** | MLIR 编译管线：cuPyNumeric→Distributed IR→Task Identification→Task Fusion→Data Promotion→Polyhedral Optimization (loop fusion+parallelization)→Temporary Elimination→gpu/nvvm lowering→GPU kernel。Multi-dialect: triton-gpu dialect (tile-level)、nvgpu dialect (NVIDIA TMA/WGMMA)、xla dialect (TPU)、amdGPU dialect、MTIA-MLIR dialect。跨 dialect lowering 允许同一 IR 编译到不同后端 | NVIDIA GPU, multi-core CPU, AMD GPU, MTIA v3 | 笔记未给出具体 benchmark | MLIR (LLVM), scf/affine/memref→gpu/nvvm lowering | Q4.5 |
| **跨平台** | **KernelEvolve Agentic Framework** | Agentic graph search 覆盖 Triton DSL→CuTe DSL→Triton-MLIR→PTX/CUBIN→AMDGPU/MTIA-MLIR→Hardware diagnostics 多层抽象。RAG + persistent knowledge base 编码硬件特定约束，使 LLM agent 为专有架构（MTIA v3）生成有效 kernel | NVIDIA H100/A100, AMD MI300X, Meta MTIA v3 | KernelBench 跨硬件加速 | Python + Triton + CuTe DSL; Meta 内部 | Q4.5 |
| **GPU/NVIDIA** | **Block Order Scheduling (L2 Cache Reuse)** | 3D stride block→data mapping：相邻 block 在 row 方向连续→A 矩阵行数据在相邻 block 间 L2 共享。GEMM {8,N,M/8} 3D stride: HBM 982 GB/s→805 TFLOPS；{N,M} row-major: HBM 3070 GB/s→仅 392 TFLOPS。关键洞察：更低 HBM 带宽 ≠ 更低性能——策略 A HBM 仅 982 GB/s 但 805 TFLOPS（大部分访问命中 L2 12 TB/s），策略 B HBM 高达 3070 GB/s 但仅 392 TFLOPS（L2 miss→从 HBM 3.35 TB/s 加载）。Attention {N,H,B} seq 连续: HBM 213 GB/s→600 TFLOPS；{B,H,N} batch 连续: HBM 2390 GB/s→494 TFLOPS | H100 (50MB L2, 132 SMs) | GEMM: HBM 982→805 TFLOPS (3D stride) vs 3070→392 TFLOPS (row-major); Attention: 213→600 vs 2390→494 | ThunderKittens persistent grid + block order | Q4.1, Q4.3 |
| **GPU/NVIDIA** | **Coalesced Memory Access 优化** | GPU global memory coalescing factor γ = A / (M×128)：γ=1 完美合并 (1-2 transaction/warp)；γ≈0.03 完全非合并 (32 transaction/warp)。MoE sorted tokens 同一 expert: coalesced→~3.0 TB/s effective BW；scatter access 不同 experts: uncoalesced→~90 GB/s (仅 3% peak)。Samoyeds data packing: sparse MoE metadata (2-bit/element)→32-bit aligned→确保 L2 高效利用。128B cache line = 32×FP32 elements/transaction | H100 (HBM3 3.35 TB/s), A100 (HBM2e 2.0 TB/s) | Coalesced γ≈1.0 perf; Uncoalesced γ≈0.03 → 仅 3% peak BW | CUDA (硬件 L1 cache line 128B coalescing); Samoyeds metadata packing | Q4.3 |
| **GPU/NVIDIA** | **Warp Scheduler 延迟隐藏 (GPU 硬件)** | Per-SM 4 warp scheduler units 独立管理 warp pool。策略推测为 "greedy-then-oldest"：优先保持 I-cache warm（prev warp 仍 ready→issue prev），否则 oldest ready warp。零成本 warp 切换（独立 per-warp register file + program counter，无 context switch）。调度粒度：每 scheduler 每 2 周期发射 1 条 warp 指令（Ampere），峰值 4×0.5×1.7GHz≈3.4 warp inst/clock/SM。不感知 CUDA stream priority | Ampere/Hopper/Ada (NVIDIA GPU) | 4 warp schedulers/SM; max 64 warps/SM; 零成本切换 | GPU 硬件固件（闭源），逆向工程推断 | Q4.2 |

---

## 按方法类别分类

| 分类 | 方法 | 具体方法描述 | 核心机制 | 来源 |
|------|------|-------------|----------|------|
| **Kernel Fusion (算子融合)** | Fused MoE Kernel? | Expert FFN: GEMM(gate+up) + SiLU + GEMM(down) + atomic scatter-add 融合为单 kernel。Sorted token indirection + L2 cache 友好的 block ordering。BLOCK_M=64 (H100 SMEM 约束) | 减少 6-8 kernel launch→1-2；消除中间 HBM round-trip；atomicAdd on L2 处理 top-k 多 expert 累积 | Q4.1, Q4.3, Q4.6 |
| **Kernel Fusion** | SLA Fused Sparse-Linear Attention? | DiT denoising attention: sparse FA + linear attention + negligible skip 三模式单 kernel 融合。CRITICAL→Tensor Core O(N²)，MARGINAL→CUDA Core O(1)，NEGLIGIBLE→skip | Per-Q-block conditional execution；static tile (diffusion step 固定 shape) | Q4.1 |
| **Kernel Fusion** | MPK / Mirage mega-kernel! | 全模型编译为单 persistent kernel——SM 级 tGraph + event-driven in-kernel runtime。Operator decomposition→dep analysis→event fusion→tGraph normalization→BFS linearization | SM task 级依赖（非 kernel 级）→跨算子 computation/communication 在 SM 间重叠；decentralized scheduling 无集中调度器 | Q4.1, Q4.5, Q4.6 |
| **Kernel Fusion** | FlashMoE Persistent Kernel* | Gate + Dispatch + Expert FFN + Combine 全融合单 persistent kernel。Actor model: Processor/Scheduler/Subscriber 三种 warp-specialized 角色。Tile (128,64)，128 threads/block | **1 launch 替代 432-550 kernel launches**；93.17% SM util；对称 Tensor Layout 实现 write-write conflict-free DMA | Q4.1, Q4.4, Q4.5 |
| **Kernel Fusion** | FlashAttention? | QK^T + Softmax + PV 融合单 kernel。Online softmax rescaling 避免 N×N 矩阵 materialize 到 HBM。FA-3: warp-specialized WGMMA+TMA async pipeline | HBM traffic 35.3GB→4.4GB (8× reduction)；FA-3 H100 840 TFLOPs/s (85% peak) | Q4.3, Q4.4 |
| **Kernel Fusion** | ChituDiffusion dEngine* | DiT/diffusion 模型的 dGraph 分解 + tile-based 编译。Data-property-aware 多 dEngine 预编译 + 运行时动态选择。Ragged operation regularization | **编译时多版本 + 运行时动态匹配** input property；round-robin tile→thread block mapping for ragged batching | Q4.4 |
| **Tile 切分与调度** | WELDER Hardware-Aligned Tile Search! | 三级硬件 penalty 驱动 tile shape 枚举：coalesced (128B transaction) + parallelism (≥128 tiles) + capacity (SMEM footprint)。MMA fragment 整除约束 (16 for FP16) | 枚举空间 = 各维度 tile size 笛卡尔积；penalty 各自 O(1) 计算；multiple candidates 可并行评估 | Q4.1, Q4.4 |
| **Tile 切分与调度** | ThunderKittens Tile Hierarchy? | Warp (16×16 register tile) → Warpgroup (shared tile, auto swizzle) → CTA (LCSF template, N-stage pipeline)。Grid: persistent grid + 3D stride block order | **三层 tile 抽象直接映射 GPU 物理硬件**；compile-time swizzle selection (32/64/128B)；4-stage pipeline→760 TFLOPS | Q4.1, Q4.4 |
| **Tile 切分与调度** | FlashAttention IO-Aware Tiling | B_c = ceil(M/(4d)), B_r = min(B_c, d)。SRAM 分配 155KB/192KB (A100)。MMA fragment: B_r 整除 16, B_c 整除 8 | 大 tile→更少 HBM passes 但更低 occupancy；小 tile→更多 passes 但更高 occupancy。FA 中前者收益更大 | Q4.4 |
| **Tile 切分与调度** | CUTLASS Tile Configuration? | ThreadBlock 128×128×32 → Warp 64×64×32 → MMA 16×8×16 (FP16)。FP8 (Hopper): mma 16×8×32, K 维对齐放宽至 32。WELDER penalty + MMA alignment 双重约束 | 分层 tile 迭代器：M 整除 16、N 整除 8、K 整除 16（FP16）；FP8 K 对齐放宽到 32；swizzle SMEM layout 消除 bank conflict | Q4.4 |
| **Tile 切分与调度** | Infera Zero-Tuning Tile Compilation! | 编译时 tile shape 枚举 + Multi-Version micro-kernel 生成 (reg 64/96/128, smem 48/80/112/144 KiB, pipeline 2/3/4) + 静态 ILP/Intensity 分析 | 无 GPU profiling→比 search-based 快 2-3 个数量级；36 种 baseline 组合覆盖 ILP/TLP/Intensity 三维空间 | Q4.4 |
| **Tile 切分与调度** | PIT for MoE (Permutation Invariant Transformation)* | 稀疏 micro-tile→dense tile 重组：permutation invariance 保证计算等价→高 GPU GEMM 利用率 | **运行时 micro-tile 重组**；置换不变性数学保证 | Q4.4 |
| **软件流水线 (Pipeline)** | Software Pipelining cp.async (Ampere)! | Double buffering: 2 组 SMEM buffer 交替——cp.async 填充 stage i 时 mma 消费 stage i-1。Prologue (预取前 2 stages)→Main Loop (steady state 重叠)→Epilogue (排空) | cp.async 16 字节粒度，L1 BYPASS，硬件自动完成；num_stages=2-4；SMEM 占用 = stages×tile_size | Q4.2 |
| **软件流水线** | Hopper TMA + WGMMA Pipeline! | TMA async HBM↔SMEM (单线程发起，无寄存器中转) + wgmma.mma_async (warp group 异步 MMA) + mbarrier (硬件同步) + warp specialization (producer/consumer 角色分工) | TMA→独立硬件单元不占 CUDA Core；WGMMA 异步发射不 block；mbarrier 单周期 wait；PIPE=2-3；multicast 广播到多 SM | Q4.2, Q4.5 |
| **软件流水线** | ImFP (Implicit Fine-Grained Pipeline)* | Single-producer (Load WG, TMA) multiple-consumer (Compute WG_0+WG_1, dequant+MMA)。**跨 WG 异构重叠——WG_0 做 dequant (CUDA Core) 时 WG_1 做 MMA (Tensor Core)**。消除 SMEM↔RF round-trip | Hardware atomic 竞争自动调度→无软件 barrier；LiquidQuant α=0.875 指令/element；Dual-MMA packed layout LDS.128 加载 32 UINT4 | Q4.2 |
| **软件流水线** | Multi-stage Register Pipeline (Drawloom)* | 5 阶段: **FillSMEM→FillREG→Comp→EmptySMEM→EmptyREG**。delaySMEM=1 (double buffering), delayREG=2 (3 REG sets) | 进一步解耦 SMEM→REG 和 computation→消除 data dependency stall；mip1 达 5.68× | Q4.2 |
| **软件流水线** | TileLang T.Pipelined* | 编译器自动分析 buffer 依赖→插入**架构特定异步指令** (Ampere→cp.async, Hopper→TMA, CDNA→buffer_load_dword) + 自动 warp specialization | 70 行 Python FA = 98% FA-3 性能；跨三架构统一 pipeline annotation | Q4.2 |
| **软件流水线** | SonicMoE Ping-Pong Scheduling* | **2 consumer warpgroups 交替执行 MMA+epilogue；1 producer warpgroup 持续 TMA prefetch。**3 warpgroups/CTA = 12 warps = 384 threads | Consumer0 执行 MMA 时 Consumer1 执行 epilogue (SwiGLU+TMA store)→Tensor Core 利用率 ~85% | Q4.6 |
| **Warp/Thread Specialization** | Warp Scheduler (NVIDIA GPU Hardware)? | 4 warp schedulers/SM，greedy-then-oldest 策略。零成本 warp 切换：独立 per-warp register file + program counter。64 warps/SM max | 更高 occupancy→更多 ready warps→更有效隐藏 memory latency (300-800 cycles)；与 register/thread 存在零和博弈 | Q4.2 |
| **Warp/Thread Specialization** | Producer-Consumer Warp Specialization (H100)* | **Producer warp (1-2, setmaxnreg=32): TMA load only。Consumer warpgroup (6-7, setmaxnreg=232): WGMMA compute only。**物理隔离→各自 register file 分区不互相挤占 | setmaxnreg 硬件寄存器重分配；load worker 仅需 ~32 regs→释放寄存器给 compute worker；mbarrier 同步 ~10-20 cycles arrive | Q4.2, Q4.5 |
| **Warp/Thread Specialization** | FlashMoE Actor-Based Warp Specialization* | 纯软件 actor 模型 (SM70+，不依赖 TMA)。Processor/Scheduler/Subscriber 三种角色 **event-driven polling**。NVSHMEM + atomic doorbell + GMEM task queue | N-1 blocks Processor + 1 block OS (Scheduler 1 warp + Subscriber 3 warps)；work-conserving 调度无 barrier | Q4.2, Q4.5 |
| **Warp/Thread Specialization** | AMD Wavefront Specialization* | CDNA3 静态 reg 分配：512 regs/SIMD 在所有 wave 间平均。**Producer-consumer 退化** (4P+8C: 893 vs 0P+8C: 1281 TFLOPS)。推荐 8-WAVE PING-PONG (无专职 producer) | 静态寄存器瓶颈→producer wave 占用 reg 但 consumer 无法回收→缩小 output tile size | Q4.1, Q4.2 |
| **指令级并行 (ILP)** | ILP Tensor-Vector Parallelism* | 同 warp 内交错 Tensor MMA 指令和 CUDA Core vector 指令。利用 GPU dual-issue 能力：Tensor MMA issue bubble 被 vector 指令填充。Horizontal split ratio≈1:1 | Hopper TLP vector interval ratio 降至 2.7%；ILP 提升受限于 register pressure 与 occupancy trade-off | Q4.2 |
| **指令级并行** | ImFP 跨 WG 指令混合* | **Load WG TMA→Compute WG_0 dequant (CUDA Core)** || Compute WG_1 MMA (Tensor Core)。天然重叠无软件 barrier | 异构计算单元 (CUDA Core vs Tensor Core 16.5× 吞吐差距) 的天然并发 | Q4.2 |
| **Memory Hierarchy 优化** | Register Tiling (MoE/DiT)? | MoE: data stationary—output accumulator 保留在 regs 跨 k 维迭代 (64 FP32 accumulators = 128 regs)。R_per_thread 128→25% occupancy vs 64→50%。DiT: Q tile + O accumulator + m/l stats→~48.5KB reg/CTA。Register spill→local memory (HBM)~400 cycles penalty | 每 SM 65536×32-bit regs；R_per_thread×threads_per_block 决定 max blocks/SM；warp specialization 缓解 (load warp 少 regs, compute warp 多 regs) | Q4.3 |
| **Memory Hierarchy 优化** | SMEM Bank Conflict Avoidance* | 32 banks×4B→naive row-major layout→8-way conflict→有效 BW = peak/8。TK 3 swizzle layouts auto-selected at compile-time: 32B (0 conflict, 窄 tile), 64B (0 conflict, 中 tile), 128B (4 conflict, 宽 tile)。**多微算子并发时不同算子的 bank 访问模式不同**→需 layout 转换或接受一定 conflict | Bank conflict 使有效 SMEM BW 退化 88% (8-way)；ldmatrix 对 SMEM layout 有特殊要求（需均匀分布 across banks） | Q4.3 |
| **Memory Hierarchy 优化** | L2 Cache Affinity (HyTiS)? | Wave 粒度量化 DRAM→L2 流量 V_i→选择合适的 Group-M/Group-N tile layout 最小化总 V_tol。Column-major layout: 沿 M 维相邻 SM 共享 A 行；Row-major: 沿 N 维相邻 SM 共享 B 列。DRAM read 量在不同 layout 下差异最高 64% | H100 50MB L2 shared across 132 SMs；自适应 layout 将 low DRAM read 区从 46% 降至 20% | Q4.3 |
| **Memory Hierarchy 优化** | Block Order Scheduling (L2 Reuse)? | 3D stride {8,N,M/8} block mapping→相邻 block 复用 A 矩阵行数据→L2 hit。vs Row-major {N,M}→L2 miss→被迫 HBM 重载→3× HBM BW 但性能减半 | HBM 982 GB/s→805 TFLOPS (L2 hit) vs 3070 GB/s→392 TFLOPS (L2 miss)；关键洞察：更低 HBM BW ≠ 更低性能 | Q4.1, Q4.3 |
| **Memory Hierarchy 优化** | Coalesced Memory Access* | Warp 内 32 threads 访问同一 128B cache line→1-2 transaction→γ≈1.0 perfect。MoE sorted tokens (同 expert 连续)→coalesced；scatter access→uncoalesced→up to 32 transactions→γ≈0.03→仅 3% peak BW | 128B L1 cache line = 32×FP32；MoE expert weight 分散访存是核心瓶颈——42% decode time on HBM expert weight load | Q4.3 |
| **Memory Hierarchy 优化** | MoE Bandwidth Reduction (LYNX)* | **Batch-level dynamic expert selection**: ~55→~15 active experts→HBM data 搬运量 ~73%↓→latency 1.09-1.30×↓。MoE decode 带宽分解: Expert Weight Load 42% (核心), Expert Computation 27%, Attention 19%, Other 12% | HBM bandwidth 是共享资源——所有并发 kernel 的全局访存经同一组 HBM stack；无法软件绕过 HBM ceiling→只能减少数据搬运量 | Q4.3 |
| **Memory Hierarchy 优化** | Kitsune L2-Resident Ring Buffer* | **Inter-CTA tile 传递** pin 在 L2→SMEM→L2→SMEM 替代 SMEM→HBM→SMEM→HBM→SMEM | 64-256KB tile 通过 L2 传输；依赖 L2 persistence；与多 kernel 并发天然冲突 (L2 thrashing 风险) | Q4.3 |
| **并发 Kernel 调度** | CUDA Graph (Kernel DAG)! | 预录制 kernel launch DAG→单次 replay。Static DAG fork/join 支持。不兼容 dynamic shape 和 conditional routing | Launch overhead: ~2-5μs replay vs ~5-20μs per-kernel (90-98%↓)；MoE 需重建 graph (40-100ms) 当 routing 变化 | Q4.5, Q4.6 |
| **并发 Kernel 调度** | CUDA Stream 多流并发! | 多 stream 内 kernel 时间片/SM 空间分区并发。实际并发受 SM 资源限制。Wave quantization 效应：per-expert token 不均时 thread block << SM→严重 SM idle | H100 max 128 concurrent streams；compute-bound 和 memory-bound kernel 的 SM 资源竞争 | Q4.5 |
| **并发 Kernel 调度** | MPK Event-Driven In-Kernel Runtime! | 每 SM 独立 event-driven task loop。atomicAdd counter→event activation→scheduler dispatch。Decentralized scheduling: scheduler 仅用 local state，无全局协调 | Per-task overhead 极小（仅 atomicAdd）；cross-task pipelining (1.2-1.3×)；compute-communication overlap (1.1×) | Q4.1, Q4.5 |
| **并发 Kernel 调度** | Persistent Kernel + Work Stealing* | Grid = SM_count blocks→每 block 循环处理多个 tile。atomicAdd global_tile_counter 获取下一个 tile。消除 kernel re-launch overhead + 最大化 L2 cache 复用 (相邻 block 连续执行) | FlashMoE: 1 launch vs 432-550→SM util 14%→93.17%；Block Order 配合: 3D stride→L2 reuse→HBM BW 降低 3× | Q4.1, Q4.4 |
| **NPU/TPU 专用调度** | CANN/TBE/TIK 异构调度* | Cube Unit (MatMul) + Vector Unit (element-wise) + Scalar Unit (control) 独立指令队列→同步取指并行执行。EPD-Serve: MatMul 用 AI Core 时 AllReduce 用 AI Vector | 硬件天然微算子并发 vs GPU 需软件 warp specialization 模拟；MTE DMA 与计算完全重叠 | Q4.5, Q4.6 |
| **NPU/TPU 专用调度** | XLA/TPU VLIW 静态调度* | MXU 128×128 systolic array 驱动 tile 并发。XLA compiler: HLO→fusion→layout→VLIW bundle→TPU binary。GemmFusion pass 将多 GEMM 合并为更大的矩阵乘提升 MXU occupancy | **完全编译时静态调度 (无运行时并发)**；VLIW 架构依赖编译器静态指令编排 vs GPU SIMT warp scheduler 动态调度 | Q4.5, Q4.6 |

---

## 分类详细问答

### 分类: Kernel Fusion (算子融合)

#### 方法: Fused MoE Kernel (vLLM Triton)

Expert FFN 的 GEMM (gate+up) + SiLU activation + GEMM (down) + atomic scatter-add 融合为单 Triton kernel。Block-level SPMD 编程模型，compiler 自动处理 tiling 和 shared memory allocation。

**核心机制**:
```
@triton.jit
def fused_moe_kernel(A, B, C, sorted_token_ids, expert_ids, topk_weights, ...):
    pid = tl.program_id(0)
    expert_id = tl.load(expert_ids + pid)
    // Step 1: Indirect token gather — 只加载当前 expert 的 tokens
    token_indices = sorted_token_ids[pid*BLOCK_M : (pid+1)*BLOCK_M]
    a_block = tl.load(A + token_indices[:,None] * H + range(H))  // [BLOCK_M, H]
    // Step 2: FC1 gate+up 融合 (单 GEMM, MMA tile 16×16×16)
    w1 = tl.load(B + expert_id * stride_E)  // stacked [H, 2×ffn_dim]
    gate = silu(tl.dot(a_block, w1_gate))    // [BLOCK_M, ffn_dim]
    up   = tl.dot(a_block, w1_up)             // [BLOCK_M, ffn_dim]
    hidden = gate * up                        // element-wise fused gating
    // Step 3: FC2 down projection
    w2 = tl.load(B + expert_id * stride_E + 2*ffn_dim*H)
    expert_out = tl.dot(hidden, w2)           // [BLOCK_M, H]
    // Step 4: Routing weight + atomic scatter-add
    routing_w = tl.load(topk_weights + ...)
    expert_out = expert_out * routing_w[:, None]
    tl.atomic_add(C + token_indices[:,None] * H + ..., expert_out)
```

- **实现**: Triton (Python DSL, MLIR-based Triton-IR→PTX→SASS); vLLM 推理框架
- **实验环境**: H100 (132 SMs, 228KB SMEM/SM), A100 (108 SMs, 164KB SMEM/SM)。MoE 模型: Mixtral-8x7B (8 experts, top-k=2), Qwen2-MoE。BLOCK_M=64 (H100 SMEM 约束, H=4096, ffn_dim=14336)。Benchmark: 15-20% throughput↑ (FP16), 25-30%↑ (FP8), 12-20% latency↓。MoE-SpeQ fuseMoE 变体: K=1408, N=2048, fused kernel 贡献 31.8% 加速
- **来源**: Q4.1, Q4.3, Q4.6; `knowledge_notes/kernel知识笔记/Fused MoE（融合 MoE Kernel）.md` (score: 41.5); `experiment_notes/系统实验笔记/MoE-Inference-Bench.md` (score: 1503.6); `knowledge_notes/编译知识笔记/CUTLASS.md` (score: 4621.2)

#### 方法: SLA Fused Sparse-Linear Attention Kernel

DiT (Diffusion Transformer) denoising attention kernel 将稀疏 FlashAttention、线性 attention、negligible block skip 三种模式融合到单 kernel。Per-Q-block loop 条件执行——CRITICAL block→Tensor Core MMA (O(N²))，MARGINAL block→CUDA Core 线性 attention (O(1) via pre-aggregation h_j/z_j per KV block)，NEGLIGIBLE→skip。

- **核心机制**: Phase 1 — Precompute for linear attention (所有 block 共享)。Phase 2 — Per-Q-block loop 三种模式条件执行：CRITICAL→FlashAttention S=QK^T + softmax rescale + PV；MARGINAL→H_i += h_j, Z_i += z_j (d×d matrix + d×1 vector additions)；NEGLIGIBLE→skip
- **实现**: CUDA C++，自定义 attention kernel
- **实验环境**: NVIDIA GPU。Kernel-only 13.7× vs FlashAttention-2。DiT 特殊性: diffusion step 固定 latent resolution→tile size 静态配置
- **来源**: Q4.1; `knowledge_notes/kernel知识笔记/Fused Sparse-Linear Attention GPU Kernel.md`

#### 方法: Mirage Persistent Kernel (MPK) Mega-Kernel Compiler + Runtime

将整个 tensor program (PyTorch model) 编译为单个 mega-kernel，通过 SM 级任务图 (tGraph) 实现跨算子微算子并发。这是从「kernel 级」到「SM 级」的抽象层次下沉。

- **核心机制**:
  - 编译流程: Operator decomposition (沿 output 维 partition by SM count)→Dependency analysis (per-task-pair overlap check→插入 fine-grained events)→Event fusion (successor-set + predecessor-set fusion 消除冗余同步)→tGraph normalization (fan-in/fan-out ≤ 1)→tGraph linearization (BFS)→GPU device memory 紧凑存储
  - 运行时: 每 SM 独立 event-driven task loop——spin-wait dep event→执行 task (MatMul/Attention/RMSNorm/AllReduce 等)→atomicAdd trig event counter→激活下游。Scheduler:Worker = 16:128 warps 分区 (H100: 4 SMs scheduler + 128 SMs worker)
  - Cross-Task Pipelining: 当前 task compute 期间通过 TMA 预取下一 task 输入 tile→1.2-1.3× speedup
  - Compute-Communication Overlap: MatMul→AllReduce task 对级依赖→compute 和 communication 在不同 SM 组并发→1.1× speedup
- **实现**: C++ (40K), CUDA (84K), Python (10K) ≈ 134K lines; Mirage superoptimizer; NVSHMEM; torch.compile(backend=MPK)
- **实验环境**: A100 (108 SMs), H100 (132 SMs), B200 (148 SMs)。Qwen3-8B decode: 12.5ms/token vs vLLM 14.5ms→1.16×。Overall 1.0-1.7× vs kernel-per-op。固定 prompt length=64, decode 1024 tokens offline setup
- **来源**: Q4.1, Q4.5, Q4.6; `paper_secs/secs_multimodal_kernel/Mirage Persistent Kernel.../span-idpage-2-0span2.2-Kernel-Fusion.md` (score: 4149); `idea_notes/Mirage Persistent Kernel.md` (score: 4222.1)

#### 方法: FlashMoE Actor-Based Persistent Kernel

单 persistent kernel 融合 MoE 的全流程 (Gate + Dispatch + Expert FFN + Combine)，Actor 模型实现 warp-specialized work stealing。

- **核心机制**: 三种 Actor 角色 (同一 kernel，无 barrier 同步):
  - Processor (N-1 blocks, 4 warps/block): 执行 CUTLASS GEMM (GEMM0 fused GELU, GEMM1) + NVSHMEM DMA。Tile dimension (128,64), 128 threads/block
  - Scheduler (OS block warp 0): work-conserving 多线程调度——poll events→dispatch ready tasks→enqueue to processor queues
  - Subscriber (OS block warps 1-3): 解码 remote tile packets→enqueue to scheduler event queue
  - 对称 Tensor Layout: overprovision 2×r 倍内存→write-write conflict-free one-sided DMA
- **实现**: CUDA/C++, CUTLASS, NVSHMEM。1 次 launch 替代 Megatron-LM+DeepEP 的 432-550 次 launch
- **实验环境**: 8×H100 80GB NVLink, CUDA 12.8。93.17% SM util (vs 14% Megatron-TE), 6× latency speedup, 5.7× throughput。FP32 FlashMoE 仍胜 FP16 baseline
- **来源**: Q4.1, Q4.4, Q4.5; `paper_secs/secs_multimodal_kernel/FlashMoE.../1-Introduction.md` (score: 2752.3); `knowledge_notes/kernel知识笔记/Actor-Based Warp Specialization for In-Kernel Scheduling.md` (score: 3179.7)

#### 方法: FlashAttention IO-Aware Fusion

Online softmax rescaling 使 N×N attention matrix 完全在 SRAM 中计算而不 materialize 到 HBM。FlashAttention-3 (Hopper): warp-specialized producer-consumer pipeline + WGMMA + TMA。

- **核心机制**: B_c = ceil(M/(4d)), B_r = min(B_c, d)。SRAM 分配: Q tile + K/V tile (双缓冲) + softmax workspace + pipeline buffers。FP16 MMA: mma.sync.aligned.m16n8k16。Hopper FA-3: Producer warp (setmaxnreg=32) TMA load Q/K/V→Consumer warpgroup (setmaxnreg=232) WGMMA QK^T + softmax + PV
- **实现**: CUDA C++ (FA-1), Triton (FA-2), CUTLASS WGMMA/TMA (FA-3); PyTorch 2.0+ SDPA
- **实验环境**: A100 HBM 访问减少 ~15× (8.5MB→552KB), 2-4× speedup。FA-3 H100: 840 TFLOPs/s (85% peak FP16), FP8: 1.3 PFLOPs/s。seqlen=8192, batch=2, nheads=16, hdim=128
- **来源**: Q4.3, Q4.4; `knowledge_notes/kernel知识笔记/Tiling in GPU Attention Kernel.md` (score: 6335.8); `experiment_notes/kernel实验笔记/FlashAttention-3...md` (score: 688.1)

#### 方法: ChituDiffusion dEngine (Difflow)

DiT diffusion 模型的数据属性感知编译：通过 fingerprint hash 检测输入 redundancy→枚举匹配 property condition 的 dEngine→选择 max batch size within hardware constraints。Ragged operation regularization 将不规则 operation 转换为等价 regular operator。

- **核心机制**: 编译时多 dEngine 预编译 (每个 dEngine = tile 配置 + kernel variant)→运行时 data property inference→OLS regression 辅助延迟估计→选择最优 dEngine
- **实现**: Python/C++ / Triton / FlashAttention
- **实验环境**: A100/H100。1.58× avg throughput, 2.2× for correlative requests (H100)
- **来源**: Q4.4; `paper_secs/secs_2026/29-Difflow.../4-Compile-Time-Optimizations.md` (score: 1680.2)

---

### 分类: Tile 切分与调度 (Tiling)

#### 方法: WELDER Hardware-Aligned Tile Search

三级硬件 penalty 驱动 tile shape 枚举搜索，使 tile 配置天然匹配 GPU 物理约束。SubGraphTiling 中自动搜索最优 tile shape。

- **核心机制**:
  - Penalty 1 (Uncoalesced Memory Access): V100 L1 cache line=128B=32×FP32/transaction。tile leading dim 不整除 32→非合并访问→额外 memory transaction
  - Penalty 2 (Parallelism Underutilization): V100 80 SMs × 4 warp schedulers→至少 128 并行 tile。tile 过大→SM 吃不饱→利用率按比例下降
  - Penalty 3 (Capacity Overflow): footprint > target memory capacity→直接淘汰
  - MMA Tensor Core 整除约束: M/N/K 必须整除 16 (FP16 `mma.m16n8k16`)
  - 枚举空间 = 各维度 tile size 笛卡尔积，三个 penalty 各自 O(1)，按 adjusted MemTraffic 排序
- **实现**: WELDER Compiler (SubGraphTiling)
- **实验环境**: V100 (80 SMs)
- **来源**: Q4.1, Q4.4; `knowledge_notes/kernel知识笔记/Hardware-Aligned Tile Search (WELDER).md` (score: 3808)

#### 方法: ThunderKittens Three-Level Tile Hierarchy

三层 tile 抽象直接映射到 GPU 物理硬件层级：Warp (register tile)→Warpgroup (shared tile)→CTA (LCSF template)。每层有明确的硬件约束和参数。

- **核心机制**:
  - Warp → Register Tile: 16×16 FP16/BF16 矩阵 tile，匹配 `mma.m16n8k16` fragment。Swizzle layout 最小化 SMEM bank conflict
  - Warpgroup → Shared Tile: 多 warp 协作 shared memory tile。Auto swizzle on 32/64/128B boundaries: naive row-major→8-way conflict; TK swizzle 64B→0-way conflict
  - CTA → LCSF Template: Load (HBM→SMEM) | Compute (SMEM→RF→MMA) | Store (RF→SMEM→HBM) | Finish。N-stage pipeline buffer (N=1,2,3,4)
  - Grid → Persistent Grid: grid = SM_count blocks, task_id→(row,col) 映射控制 L2 reuse
  - Block Order: {8,N,M/8} 3D stride→相邻 block 复用 A 矩阵行→L2 hit (HBM 982 GB/s→805 TFLOPS)
- **实现**: TK C++ library (<1MB), CUTLASS backend。跨 NVIDIA H100/4090 (CUDA) + Apple M2 (Metal)
- **实验环境**: H100: GEMM 4096→760 TFLOPS (4-stage pipeline)。1-stage→260 TFLOPS, 4-stage→760 TFLOPS (+150-200 TFLOPS/stage)
- **来源**: Q4.1, Q4.4; `paper_secs/secs_multimodal_kernel/ThunderKittens.../span-idpage-4-0span3-THUNDERKITTENS.md` (score: 2901)

#### 方法: CUTLASS v3 Tile Configuration Hierarchy

CUTLASS 将 GEMM/Conv/Attention 分解为分层 tile：ThreadBlock Tile (shared memory resident, 128×128×32)→Warp Tile (register resident, 64×64×32)→MMA Instruction Tile (1 cycle, 16×8×16 FP16 / 16×8×32 FP8)。FP8 (Hopper): K 维对齐放宽到 32。

- **核心机制**: MMA Fragment 对齐约束: FP16→M 整除 16, N 整除 8, K 整除 16。不满足→padding→计算浪费 (如 TileM=130→padded 144→利用率 130/144=90.3%)。Swizzle SMEM layout 自动 padded→确保所有 lane 访问不同 bank。WELDER 三惩罚 + MMA alignment 双重约束
- **实现**: CUTLASS v3 C++ template library; CuTe DSL; persistent kernel + TMA + WGMMA on Hopper
- **实验环境**: H100 (SM90), A100 (SM80)。Grouped GEMM ~75% vs contiguous ideal。SonicMoE dH kernel ~420 TFLOPS (42% peak)
- **来源**: Q4.4, Q4.5, Q4.6; `knowledge_notes/编译知识笔记/CUTLASS.md` (score: 4621.2); `knowledge_notes/kernel知识笔记/Hardware-Aligned Tile Search (WELDER).md` (score: 2522.1)

#### 方法: Infera Tile-Based Zero-Tuning Compilation

编译时全静态分析驱动 multi-version micro-kernel 生成：Operator Tiling (各 axis expand_toward_hardware_alignment)→Multi-Version Generation (reg 64/96/128, smem 48/80/112/144 KiB, pipeline 2/3/4, spatial tile variants)→Static Analysis (resource constraint + ILP + arithmetic intensity)→Runtime kernel selection。

- **核心机制**: 36 种 baseline 组合覆盖 ILP/TLP/Intensity 三维空间。R=128→16 warps/SM; R=64→32 warps/SM。SMEM=144KB→1 block/SM; SMEM=48KB→3 blocks/SM。峰值性能区域 = "green box" (ILP/TLP/Intensity 平衡点)
- **实现**: Infera Compiler
- **实验环境**: V100/A100。比 Ansor/MetaSchedule 快 2-3 个数量级 (无 GPU profiling)
- **来源**: Q4.4; `knowledge_notes/编译知识笔记/Tile-Based Zero-Tuning Compilation.md` (score: 59.9); `knowledge_notes/编译知识笔记/Multi-Version Micro Kernel Generation.md` (score: 414.7)

#### 方法: PIT for MoE (Permutation Invariant Transformation)

稀疏 micro-tile→dense tile 重组。置换不变性数学保证计算等价→高 GPU GEMM 利用率。

- **核心机制**: 将 sparse MoE 的 micro-tile 在运行时重组为 GPU 高效的 dense tile。Permutation invariance 保证重组后计算等价
- **实现**: Graph-level compiler pass
- **实验环境**: GPU (通用)
- **来源**: Q4.4; `knowledge_notes/kernel知识笔记/PIT (Permutation Invariant Transformation) for MoE.md` (score: 47.2)

---

### 分类: 软件流水线 (Software Pipelining)

#### 方法: Ampere Software Pipelining (cp.async + Double Buffering)

A100 上通过 cp.async (PTX LDGSTS 硬件实现) 实现 HBM→SMEM 异步双缓冲流水线。Prologue (预取前 2 stages)→Main Loop (steady state 重叠: cp.async.wait_group + mma + cp.async 下一 stage)→Epilogue (排空)。

- **核心机制**: Double buffering (2 组 SMEM buffer 交替)。cp.async 16 字节粒度，L1 BYPASS，硬件自动完成而线程继续。num_stages=2-4，SMEM 占用 = stages×tile_size。H100 256KB SMEM/SM 可支持 num_stages=2-3 (取决于 tile size)
- **实现**: CUDA cp.async / CUTLASS / Triton (num_stages); PTX LDGSTS→SASS
- **实验环境**: A100 (Ampere, SM80): 3rd-gen Tensor Core, cp.async
- **来源**: Q4.2; `knowledge_notes/kernel知识笔记/Software Pipelining for GPU Attention Kernels.md` (score: 3306.4); `knowledge_notes/硬件知识笔记/Asynchronous Memory Copy (async-copy).md` (score: 2924.1)

#### 方法: Hopper TMA + WGMMA + Warp Specialization Pipeline

H100 硬件辅助的 warp-specialized pipeline：TMA (独立硬件 DMA→HBM↔SMEM 直传，不经 register file，单线程发起 non-blocking) + wgmma.mma_async (warp group 异步 Tensor Core 指令，发射后不 block) + mbarrier (硬件同步，arrive ~10-20 cycles，单周期 wait) + Warp Specialization (producer warp 1-2 做 TMA load，consumer warpgroup 6-7 做 WGMMA compute)。

- **核心机制**:
  - setmaxnreg 寄存器重分配: producer warp 仅需 32 regs→释放寄存器给 consumer warpgroup (232 regs)
  - TMA 在 2KB 消息大小达 74% NVLink 峰值带宽 (vs Copy Engine 需 256MB)
  - Multicast: 单次 TMA 操作广播相同 tile 到 cluster 内多 SM shared memory
  - Pipeline depth PIPE=2-3，circular SMEM buffer
- **实现**: CUTLASS 3.x + CuTe; TileLang T.Pipelined; CUDA ≥12.0
- **实验环境**: H100 (Hopper, SM90); WGMMA 85% Tensor Core utilization
- **来源**: Q4.2, Q4.4, Q4.5; `knowledge_notes/硬件知识笔记/Tensor Memory Accelerator (TMA).md` (score: 5104.2); `knowledge_notes/硬件知识笔记/Warp Specialization (Warp 专业化).md` (score: 4004.5)

#### 方法: ImFP (Implicit Fine-Grained Pipeline) / LiquidGEMM

Single-producer multiple-consumer 模型替代传统 ExCP (Load→Dequant→MMA) 三阶段流水线。Load WG (4 warps) TMA→SMEM; Compute WG_0 & WG_1 (各 4 warps) 从 SMEM 一站式 dequant+MMA。跨 WG 异构重叠——WG_0 在 CUDA Core 做 dequant 时 WG_1 在 Tensor Core 做 MMA，天然重叠无软件 barrier。

- **核心机制**:
  - 消除 SMEM↔RF round-trip: ExCP 需 Load→Dequant (SMEM→RF→SMEM)→MMA (SMEM→RF)；ImFP 中 Compute WG 直接从 SMEM 读取并一站式完成 dequant+MMA
  - LiquidQuant (LQQ): 仅需 IMAD + XOR 两条 32-bit 指令处理四个元素→α=0.875 指令/element (vs QServe α≥10)
  - Dual-MMA packed layout: LDS.128 单指令加载 32 UINT4→充分利用 SMEM 带宽
- **实现**: CUTLASS/CuTe warp-specialized kernel
- **实验环境**: H100/H800 (Hopper)
- **来源**: Q4.2; `knowledge_notes/kernel知识笔记/Implicit Fine-Grained Pipeline (ImFP).md` (score: 1737.9); `knowledge_notes/硬件知识笔记/CUDA Cores vs Tensor Cores.md` (score: 333.4)

#### 方法: Multi-stage Register Pipeline (Drawloom)

5 阶段流水线: FillSMEM (async-copy)→FillREG (SMEM→REG + LDG)→Comp (TC MMA)→EmptySMEM (剩余 SMEM→REG)→EmptyREG (剩余计算)。delaySMEM=1 (double buffering), delayREG=2 (3 REG sets 轮流)。

- **核心机制**: 最多 3 个 REG set 同时 in-flight: REG set[0] 被 MMA 消费、REG set[1] 被 LDG 填充、REG set[2] 等待 SMEM 数据就绪。进一步解耦 SMEM→REG 和 computation→消除 data dependency stall
- **实现**: Drawloom CUDA: cp.async + LDG + TC MMA
- **实验环境**: Ampere (A100)。v4 +Multi-stage vs v3: avg 1.46× speedup; warp stall 改善 3.02×-3.13×; memory throughput +2.61×-2.75×; mip1 达 5.68×
- **来源**: Q4.2; `knowledge_notes/kernel知识笔记/Multi-stage Register Pipeline (for SpMV on Tensor Cores).md` (score: 3576.4)

#### 方法: TileLang T.Pipelined (Compiler-Automated Pipeline)

用户仅需 `T.Pipelined(K // block_K, num_stages=N)` annotation，编译器自动分析 loop body buffer 依赖→插入架构特定异步指令序列 + Live Variable Analysis 确定同步点 + Hopper 自动 warp specialization。

- **核心机制**: Ampere→cp.async; Hopper→TMA; AMD CDNA→buffer_load_dword lds + s_waitcnt lgkmcnt。FlashAttention 实现 (~70 行 Python) 达 FA-3 (手写 CUDA) 98% 性能
- **实现**: TileLang JIT Compiler
- **实验环境**: Ampere/Hopper/AMD CDNA 三架构统一
- **来源**: Q4.2; `knowledge_notes/kernel知识笔记/Software Pipeline (T.Pipelined).md` (score: 359.6)

#### 方法: SonicMoE Ping-Pong Scheduling

2 consumer warpgroups 交替执行 MMA+epilogue (SwiGLU activation + TMA store)；1 producer warpgroup 持续 TMA prefetch。3 warpgroups/CTA = 12 warps = 384 threads。Occupancy 18.75% (低 occupancy 在此场景可接受——TMA 已消除 memory latency stall, Tensor Core 是瓶颈)。

- **核心机制**: Consumer0 执行 MMA 时 Consumer1 执行上一 tile 的 epilogue (SwiGLU+TMA store)→交替→Tensor Core 利用率 ~85%。SMEM 按 128B aligned 分配→消除 bank conflict
- **实现**: CuTe-DSL (CUTLASS backend)
- **实验环境**: H100, B300。dH kernel ~420 TFLOPS (42% peak H100), ~700 TFLOPS (B300)
- **来源**: Q4.6; `experiment_notes/kernel实验笔记/SonicMoE Accelerating MoE with IO and Tile-aware Optimizations.md` (score: 1285.5)

---

### 分类: Warp/Thread Specialization

#### 方法: Warp Scheduler 延迟隐藏 (NVIDIA GPU Hardware)

Per-SM 4 warp scheduler units 独立管理 warp pool。策略推测为 "greedy-then-oldest"：优先保持 I-cache warm (prev warp 仍 ready→issue prev)，否则 oldest ready warp。零成本 warp 切换 (独立 per-warp register file + program counter，无 context switch)。

- **核心机制**: 调度粒度: 每 scheduler 每 2 周期发射 1 条 warp 指令→峰值 4×0.5×1.7GHz≈3.4 warp inst/clock/SM。更高 occupancy→更多 ready warps→更有效隐藏 memory latency (300-800 cycles)。不感知 CUDA stream priority
- **实现**: GPU 硬件固件 (闭源)，逆向工程推断
- **实验环境**: Ampere/Hopper/Ada (NVIDIA GPU)
- **来源**: Q4.2; `knowledge_notes/kernel知识笔记/Warp Scheduler (NVIDIA GPU).md` (score: 3339.9)

#### 方法: Producer-Consumer Warp Specialization (H100)

Producer warp (1-2, setmaxnreg=32) 仅做 TMA load→SMEM；Consumer warpgroup (6-7, setmaxnreg=232) 仅做 WGMMA compute。Physical isolation→各自 register file partition 不互相挤占。

- **核心机制**: setmaxnreg 寄存器重分配指令→producer warp 释放寄存器 (仅需 32 个用于 TMA 地址计算)→consumer warpgroup 用满 ~232 个→最大化 MMA tile size。mbarrier arrive→try_wait→硬件同步 ~10-20 cycles vs __syncthreads() ~100 cycles
- **实现**: CUTLASS 3.x + CuTe; CUDA ≥12.0
- **实验环境**: H100 (Hopper, SM90)
- **来源**: Q4.2, Q4.5; `knowledge_notes/硬件知识笔记/Warp Specialization (Warp 专业化).md` (score: 4004.5); `knowledge_notes/硬件知识笔记/Named Barriers (Hopper).md` (score: 895.4)

#### 方法: FlashMoE Actor-Based Warp Specialization (纯软件)

纯软件 event-driven actor 模型，可在 SM70+ GPU 运行，不依赖 Hopper TMA。Processor (N-1 blocks) + Scheduler (1 warp in OS block) + Subscriber (3 warps in OS block)。

- **核心机制**:
  - Subscriber: poll NVSHMEM flag→atomic retrieve→decode tile packet→write task queue (GMEM)→atomicAdd doorbell (SMEM)→notify Scheduler
  - Scheduler: sweep all doorbells→WarpInclusiveSum→signal idle Processor
  - Processor: await_scheduler_signal→warp broadcast task→switch(op): GEMM0 fused GELU / GEMM1 + NVSHMEM put / Combine (Hadamard)
- **实现**: FlashMoE CUDA: NVSHMEM + atomic + doorbell
- **实验环境**: SM70+ GPU (V100/A100/H100)。N-1 blocks Processor + 1 block admin
- **来源**: Q4.2, Q4.5; `knowledge_notes/kernel知识笔记/Actor-Based Warp Specialization for In-Kernel Scheduling.md` (score: 3179.7)

#### 方法: AMD Wavefront Specialization

CDNA3 静态寄存器分配：每 SIMD 512 regs 在所有 wave 间平均→producer wave 占用 register 但 consumer 无法回收→缩小 output tile size。无 TMA/wgmma/mbarrier→producer-consumer 退化 (4P+8C: 893 TFLOPS vs 0P+8C: 1281 TFLOPS)。

- **核心机制**: 推荐 8-WAVE PING-PONG (无专职 producer) 替代 producer-consumer。TileLang CDNA 路径: s_waitcnt lgkmcnt + buffer_load_dword lds 指令实现异步拷贝
- **实现**: HIP/ROCm; TileLang CDNA backend
- **实验环境**: MI300X (CDNA3, 304 CU, 5.3 TB/s HBM3)
- **来源**: Q4.1, Q4.2; `knowledge_notes/kernel知识笔记/Wave Specialization (Producer-Consumer) on AMD.md`; `knowledge_notes/kernel知识笔记/Wavefront Specialization（波前专业化）.md` (score: 2782.4)

---

### 分类: 指令级并行 (ILP)

#### 方法: ILP Tensor-Vector Parallelism (FlashAttention-T)

在 fused attention kernel 中将 softmax 拆分为 tensorized 部分 (repurposed MMA on Tensor Core: scaling/FMA/rowsum) 和 vectorized 部分 (CUDA Core: REDUX max/MUFU.EX2 exp)，通过 ILP interleaving 在同一 warp 内交错编排。

- **核心机制**: Horizontal split ratio≈1:1。指令交错序列: mma (scaling)→redux max (vector, 并行)→mma (FMA)→fma subtract max (vector)→mma (rowsum)→ex2 exp (vector, 并行)。利用 GPU dual-issue: Tensor MMA 有 issue bubble→vector 指令填充 bubble→t'_softmax < t_vec。Hopper TLP 替代: WGMMA row-sum 加入下一 iteration batch→vector interval ratio 降至 2.7%
- **实现**: 手写 CUDA PTX inline assembly
- **实验环境**: Ampere (ILP) / Hopper (TLP)。Hopper TLP vector interval ratio 2.7%
- **来源**: Q4.2; `knowledge_notes/kernel知识笔记/Tensor-Vector Parallelism Scheduling (ILP_TLP for Softmax).md` (score: 2455.4)

#### 方法: ImFP 跨 Warp Group 指令混合

Load WG TMA 填充 SMEM→Compute WG_0 dequant (CUDA Core, IMAD+XOR) || Compute WG_1 MMA (Tensor Core, WGMMA INT8)。硬件 atomic 竞争自动调度——无 software barrier。

- **核心机制**: CUDA Core vs Tensor Core 16.5× 吞吐差距 (H100: CUDA Core FP32≈60 TFLOPS, Tensor Core INT8≈990 TFLOPS) 是 dequant 成瓶颈的根因。LiquidQuant α=0.875 指令/element 确保 CUDA Core 不成为 pipeline 瓶颈
- **实现**: CUTLASS/CuTe warp-specialized kernel; Hopper (H100/H800)
- **实验环境**: H100/H800 (Hopper)
- **来源**: Q4.2; `knowledge_notes/kernel知识笔记/Implicit Fine-Grained Pipeline (ImFP).md` (score: 1737.9)

---

### 分类: Memory Hierarchy 优化

#### 方法: Register Tiling — MoE Expert FFN & DiT Attention

MoE: Data stationary 策略——output accumulator (64 FP32 = 128 regs/thread) 保留在 regs 跨 k 维迭代，仅 weight tile 每次更新。R=128→32768 regs/block→2 blocks/SM→25% occupancy; R=64→4 blocks/SM→50% occupancy。DiT: Q tile (regs) + O accumulator (FP32) + m/l (softmax stats)→~48.5KB reg/CTA。FA-3 total 321KB (reg+SMEM) 需求，2 CTA/SM→occupancy 25%。

- **核心机制**: 每 SM 65536×32-bit regs=256KB。Occupancy = resident_blocks × threads_per_block / 2048。Register spill→local memory (HBM)~400 cycles penalty。Warp specialization 缓解: load warp 少 regs (32), compute warp 多 regs (232)→reg file 分区不互相挤占
- **实现**: CUDA/CUTLASS; --maxrregcount / setmaxnreg; Samoyeds data stationary
- **实验环境**: H100 (256KB RF/SM), A100 (256KB RF/SM)
- **来源**: Q4.3; `knowledge_notes/硬件知识笔记/GPU Resource Constraints.md` (score: 1528.6); `paper_secs/.../Samoyeds/span-idpage-4-2span4.1.md` (score: 1980.1)

#### 方法: SMEM Bank Conflict Avoidance (ThunderKittens Swizzle)

32 banks×4B/bank→naive row-major layout→8-way conflict→有效 BW=peak/8≈2.4 TB/s (vs 19 TB/s peak)。TK 3 swizzle layouts (32/64/128B) compile-time auto-selection: 32B swizzle 0 conflict (窄 tile width≤64); 64B swizzle 2-way conflict (中 tile); 128B swizzle 4-way conflict (宽 tile width>128)。

- **核心机制**: Tensor Core register layout (每 thread 持有矩阵的分散片段) 与 SMEM row-major layout 不匹配→ldmatrix 需要从 32 bank 均匀读取。多微算子并发 (GEMM ldmatrix + softmax row-wise reduction + element-wise)→不同算子 bank 访问模式不同→需显式 layout 转换或接受一定 conflict
- **实现**: ThunderKittens compile-time layout selection; CUTLASS padded SMEM layout (swizzle mode)
- **实验环境**: H100/A100 SMEM 32 banks×4B
- **来源**: Q4.3; `paper_secs/.../ThunderKittens/span-idpage-4-0span3-THUNDERKITTENS.md` (score: 1327.9)

#### 方法: L2 Cache Affinity + Block Order Scheduling

3D stride {8,N,M/8} block mapping→相邻 block 在 row 方向连续→A 矩阵行数据 L2 共享 (1 次 load 服务 8 consecutive blocks, SUPER_M=8)。vs Row-major {N,M}→相邻 block 遍历不同 row→L2 miss→被迫 HBM 重载。核心洞察: 更低 HBM 带宽 ≠ 更低性能 (策略 A: HBM 982 GB/s→805 TFLOPS vs 策略 B: HBM 3070 GB/s→392 TFLOPS)。

- **核心机制**: HyTiS wave 粒度量化 DRAM→L2 流量 V_i→选择 Group-M/N tile layout 最小化 V_tol。DRAM read 量在不同 layout 下差异最高 64%→自适应 layout 将 low DRAM read 区从 46% 降至 20%
- **实现**: ThunderKittens persistent grid + block order; HyTiS tile layout selection
- **实验环境**: H100 (50MB L2, 12 TB/s, shared across 132 SMs)
- **来源**: Q4.1, Q4.3; `knowledge_notes/硬件知识笔记/L2 Cache Affinity.md` (score: 2176.8); `knowledge_notes/kernel知识笔记/Block Order Scheduling _ L2 Cache Reuse (GPU).md` (score: 32.7)

#### 方法: Coalesced Memory Access + MoE Token Sorting

128B L1 cache line = 32×FP32 elements/transaction。Coalescing factor γ = A/(M×128), γ=1 perfect→~3.0 TB/s effective BW; γ≈0.03→~90 GB/s (仅 3% peak)。MoE sorted tokens (同 expert tokens 连续)→coalesced access→L2 hit 提升；scatter access (不同 experts 权重分散)→uncoalesced→32 transaction/warp→带宽退化 97%。

- **核心机制**: Samoyeds data packing: sparse MoE metadata (2-bit/element)→32-bit aligned→确保 L2 高效。Video 3D conv: strided pattern along spatial+temporal dims→需 im2col 或 explicit padding 实现 coalesced。DiT attention: QKV dense tensor→row-major layout→天然 coalesced
- **实现**: CUDA 硬件 L1 cache line coalescing; Samoyeds data packing; Fused MoE token sorting
- **实验环境**: H100 HBM3 3.35 TB/s; coalesced→~3.0 TB/s vs uncoalesced→~90 GB/s
- **来源**: Q4.3; `knowledge_notes/硬件知识笔记/GPU Memory Coalescing.md` (score: 925.8); `paper_secs/.../Samoyeds/span-idpage-4-2span4.1.md` (score: 1980.1)

#### 方法: MoE Bandwidth Reduction (LYNX)

MoE decode 带宽分解: Expert Weight Load 42% (核心瓶颈), Expert Computation 27%, Attention 19%, Other 12%。LYNX batch-level dynamic expert selection: ~55→~15 active experts→HBM 数据搬运量 ~73%↓→latency 1.09-1.30×↓。

- **核心机制**: Arithmetic Intensity = B×k/N = 16×8/64 = 2 FLOPs/byte (Qwen2-57B, B=16, H200)。H200 critical AI = 67/3.35 ≈ 20→AI=2 << 20→strongly memory-bound。HBM bandwidth 是共享资源——所有并发 kernel 经过同一 HBM stack——无法软件绕过 ceiling→只能减少数据搬运 (量化/稀疏化/expert reduction)
- **实现**: LYNX runtime; batch-level expert selection
- **实验环境**: H200 (67 TFLOPS, 3.35 TB/s); Qwen2-57B MoE, B=16
- **来源**: Q4.3; `knowledge_notes/系统知识笔记/Memory-Bandwidth-Bound Decode in MoE.md` (score: 1288.4)

#### 方法: Kitsune L2-Resident Inter-CTA Ring Buffer

64-256KB tile 通过 L2 cache 在 CTA 间传递: SMEM→L2→SMEM (avoid SMEM→HBM→SMEM→HBM→SMEM round-trip)。双 entry ring buffer + seq_num 无锁同步。

- **核心机制**: CUDA API pin queue memory 在 L2 cache。依赖 L2 persistence——若被其他并发 kernel 刷掉→数据溢写到 HBM→性能急剧下降→与多 kernel 并发天然冲突
- **实现**: Kitsune CUDA: L2 persistence API + atomic lock-free sync
- **实验环境**: H100
- **来源**: Q4.3; `knowledge_notes/kernel知识笔记/Inter-CTA Ring Buffer Queue (L2-Resident).md` (score: 1591.8)

---

### 分类: 并发 Kernel 调度框架

#### 方法: CUDA Graph (Kernel DAG 预录制)

Phase 1 — Graph Capture: cudaStreamBeginCapture→record kernel launches→cudaStreamEndCapture。Phase 2 — Instantiation (一次性)。Phase 3 — Repeated Launch: 单次 cudaGraphLaunch 替代 N 次独立 launch (11 kernel: 1 router+8 expert FFN+1 attention+1 aggregation→launch overhead 55-220μs→2-5μs→90-98%↓)。

- **核心机制**: DAG dependency auto-detection (无依赖 expert FFN kernel→并发发射到不同 SM)。限制: 动态 MoE routing→每次 kernel 组合不同→需重建 graph (40-100ms); 不兼容 conditional dispatch; static shape 要求
- **实现**: CUDA Runtime API (CUDA 10+)
- **实验环境**: H100, A100。MPK baseline (SGLang/vLLM + CUDA Graphs): graph replay ~5μs vs per-kernel ~50μs
- **来源**: Q4.5, Q4.6; `knowledge_notes/系统知识笔记/CUDA Graph.md` (score: 4701.2)

#### 方法: CUDA Stream 多流异步并发

多 stream 内 kernel 时间片/SM 空间分区并发。H100 max 128 concurrent streams，实际并发受 SM 资源限制 (registers、SMEM)。Wave Quantization 效应: per-expert token 不均→thread block<<SM→严重 SM idle, 多 stream 不能解决→需 persistent kernel + work stealing。

- **核心机制**: GPU thread block scheduler 在并发 kernel 间 round-robin 分配 SM→compute-bound 和 memory-bound kernel 的资源竞争。NVIDIA TX2 研究显示 scheduler 不感知 kernel 的计算/访存特征
- **实现**: CUDA Runtime API
- **实验环境**: H100/A100
- **来源**: Q4.5; `paper_secs/.../Demystifying.../2.-CUDA-PROGRAMMING-MODEL.md` (score: 2817.8)

#### 方法: MPK Event-Driven Decentralized In-Kernel Runtime

每 SM 独立 event-driven task loop: spin-wait dep event→execute task→atomicAdd trig event counter→activate downstream。Scheduler:Worker = 16:128 warps (H100: 4 SMs scheduler + 128 SMs worker; B200: 16+144)。Task 和 event 队列为 GPU device memory circular buffer，enqueue/dequeue 仅用 atomicAdd。

- **核心机制**: Decentralized scheduling→scheduler 仅用 local state，无全局协调。Cross-task pipelining: 当前 task compute 期间 TMA 预取下一 task 输入 tile→1.2-1.3×。Compute-communication overlap: task 对级依赖→MatMul→AllReduce 在不同 SM 组并发→1.1×
- **实现**: MPK Compiler (134K lines) + Mirage superoptimizer + NVSHMEM; torch.compile(backend=MPK)
- **实验环境**: A100/H100/B200, multi-GPU NVLink/NVSwitch。Qwen3-8B decode: 12.5ms/token
- **来源**: Q4.1, Q4.5; `paper_secs/.../Mirage.../5-In-Kernel-Parallel-Runtime.md` (score: 11632.1)

---

### 分类: NPU/TPU 专用调度

#### 方法: CANN/TBE/TIK — Ascend Da Vinci 异构调度

达芬奇架构 Cube Unit (16×16×16 MAC 脉动阵列→MatMul 专用) + Vector Unit (256-bit SIMD→element-wise/activation/AllReduce) + Scalar Unit (control) + MTE (HBM↔UB 异步 DMA)。三单元独立指令队列→支持同时取指、并行执行→硬件天然微算子并发 (MatMul 在 Cube 执行时 AllReduce 在 Vector 执行)。

- **核心机制**: CANN 软件栈: GE (Graph Engine) IR lowering + operator fusion→TBE (Python DSL, auto tiling)→TIK (C++ API, 显式 buffer 管理)→NPU 指令→Cube/Vector/Scalar 硬件。Unified Buffer 256KB (类似 GPU SMEM)。CoC (Communication over Computation): MTE 在计算当前 micro-batch 时同时远程 DMA 下一 batch→计算与通信流水线重叠
- **实现**: CANN GE + TBE Python DSL + TIK C++ API; Ascend C (SPMD); PyTorch Ascend Adapter
- **实验环境**: Ascend 910B (64GB HBM, ~1.2TB/s BW, 32 AI Cores), Ascend 910B3 (64GB HBM, 1.6TB/s, 20 AI Cores, 313 TFLOPS FP16)。EPD-Serve operator-level 空间复用。笔记未提供 kernel 级 MFU 数据
- **来源**: Q4.1, Q4.5, Q4.6; `knowledge_notes/硬件知识笔记/Ascend NPU Architecture...md` (score: 3903.6); `knowledge_notes/硬件知识笔记/Huawei Ascend 910B3 NPU.md` (score: 444.0); `knowledge_notes/编译知识笔记/CANN.md` (score: 33.3)

#### 方法: XLA/TPU VLIW 静态编译调度

TPU v5e/v5p MXU 128×128/128×256 systolic array→tile 必须适配 128 宽度 (不可拆分，vs GPU 可组合 16×16 MMA tile)。XLA 全自动: JAX HLO→Operator fusion (GemmFusion pass 合并多 GEMM 提升 MXU occupancy)→Layout assignment→VLIW bundle 打包→TPU binary。TPU 使用 VLIW 架构→依赖编译器静态指令编排 (vs GPU SIMT warp scheduler 动态调度)。

- **核心机制**: VPU memory ~16MB (类似 GPU SMEM 但容量更大)→tile 需 fit VPU。MXU 128×128 粗粒度: 小 batch MoE (单 expert 少量 tokens)→MXU 大量 PE 空闲。MoE 场景 GPU tile 灵活性更优 (可组合 16×16 MMA 构建任意 tile shape)
- **实现**: JAX + XLA; HLO → LLO → TPU binary
- **实验环境**: TPU v4 (MXU 128×128, GLaM 1.2T 训练 on 1,024 chips: 50-62% compute utilization)。TPU v5e (128×128), TPU v5p (128×128 or 128×256)。笔记无 v5e/v5p MoE/DiT 推理 benchmark
- **来源**: Q4.1, Q4.5, Q4.6; `knowledge_notes/硬件知识笔记/TPU-v4.md` (score: 641.7)

---

## 方法间关系

### 替代关系

- **Mirage MPK ←→ CUDA Graph**: 两者都减少 kernel launch overhead。MPK 通过 mega-kernel (SM 级 tGraph) 彻底消除 kernel barrier bubble——kernel 间 computation/communication 可在不同 SM 上并发；CUDA Graph 仅消除 CPU-side launch overhead，kernel 间仍有全局 barrier (所有 SM 必须等 kernel A 完成后才能开始 kernel B)。MPK 适合动态路由 (MoE conditional dispatch)，CUDA Graph 不适合
- **CUTLASS v3 ←→ Triton**: CUTLASS 提供更底层控制 (warp tile→MMA instruction 显式管理、TMA/WGMMA 直接接入) 但编程复杂度高；Triton 提供 block-level 抽象、开发效率高但无法实现 warp specialization。Triton 适合快速原型和 element-wise/reduction/GEMM kernel；CUTLASS 适合极致优化 (Hopper FP8/TMA/WGMMA 新特性) 的生产级 kernel
- **Hopper TMA+WS ←→ Ampere cp.async Pipeline**: TMA 硬件独立 DMA (不占 CUDA Core，无寄存器中转，2KB 达 74% NVLink 峰值) + mbarrier 硬件同步→pipeline 更浅 (PIPE=2-3) 且延迟隐藏更深。cp.async 需软件 barrier (__syncthreads ~100 cycles)，但兼容 A100/4090
- **Producer-Consumer WS (NVIDIA) ←→ 8-WAVE PING-PONG (AMD)**: NVIDIA Hopper TMA+setmaxnreg+mbarrier 实现高效 producer-consumer→4-stage pipeline 760 TFLOPS。AMD CDNA3 静态 reg 分配导致 producer-consumer 退化 (4P+8C: 893 vs 0P+8C: 1281 TFLOPS)→推荐 8-WAVE PING-PONG 替代
- **FlashMoE Actor Model ←→ MPK tGraph**: 两者都是 mega-kernel 方案。FlashMoE 为 MoE 量身定制 (actor model + symmetric tensor layout + NVSHMEM DMA) 适应多 GPU；MPK 为通用模型 (PyTorch→tGraph→mega-kernel) 适应多模型但单 GPU 为主。FlashMoE 在 MoE 场景 93.17% SM util；MPK 在通用 LLM 场景 1.0-1.7× vs kernel-per-op

### 互补关系

- **CUTLASS v3 + TMA + WGMMA + Warp Specialization**: 四者组合构成 Hopper 世代最高效 kernel pipeline。CUTLASS 提供 tile 抽象，TMA 提供硬件异步拷贝，WGMMA 提供 warp group 异步 MMA，Warp Specialization 提供 producer-consumer 角色分工→FA-3 达到 840 TFLOPs/s (85% peak)
- **WELDER tile search + CUTLASS tile config**: WELDER 提供硬件对齐的 tile shape 搜索 (三 penalty)，CUTLASS 提供搜索到的 tile shape 的执行引擎 (tile iterator→MMA abstraction)。两者配合→硬件最优 tile + 高效执行
- **Block Order Scheduling + Persistent Kernel**: Persistent kernel 消除 block launch overhead，Block Order (3D stride) 最大化 L2 cache reuse。两者配合→HBM bandwidth 需求降低 3× 同时性能提升 2× (805 vs 392 TFLOPS)
- **Fused MoE (算子融合) + Token Sorting + Coalesced Access**: Token sorting (同 expert tokens 连续)→L2 cache 命中率提升→coalesced memory access→Fused MoE 单 kernel 消除中间 HBM round-trip。三者组合最大化 MoE 推理的 memory bandwidth 利用率
- **ILP Tensor-Vector Parallelism + Software Pipelining**: ILP 在单 warp 内重叠 Tensor MMA 和 CUDA Core 指令，Software Pipelining 在 warp 间重叠 HBM→SMEM 和 Compute。两者层次化覆盖指令级和 kernel 级的时间重叠

### 依赖关系

- **TMA → Hopper Warp Specialization**: Warp Specialization 的高效实现依赖 TMA 硬件 (fire-and-forget async copy 使 producer warp 仅需 1 线程 + ~32 regs)。Ampere 上 cp.async 虽然也能实现 WS 但效率较低 (需更多线程参与 copy)
- **setmaxnreg → Consumer Warpgroup 大 Tile**: Hopper setmaxnreg 指令允许 producer warp 释放寄存器 (仅需 32→释放 ~200 regs 给 consumer)→consumer warpgroup 可用 ~232 regs→支持更大 MMA tile→更高 arithmetic intensity
- **mbarrier → Low-Latency Producer-Consumer Sync**: Hopper mbarrier (10-20 cycles arrive, 1 cycle wait) 使 producer-consumer 同步开销可忽略→pipeline depth 可降至 PIPE=2-3 仍保持 85% Tensor Core utilization。Ampere __syncthreads() ~100 cycles→pipeline depth 需更深 (3-4) 才能隐藏同步开销→SMEM 占用更大
- **XLA GemmFusion → TPU MXU Occupancy**: TPU 的 128×128 MXU 粗粒度 tile→单 GEMM 难以填满。XLA GemmFusion pass 将多个小 GEMM 合并为大 GEMM→提升 MXU occupancy→但依赖编译期静态分析和算子间的可合并性
- **Persistent Kernel → Cross-Task Pipelining (MPK)**: MPK 的 cross-task pipelining (当前 task compute 期间 TMA 预取下一 task tile) 需要 persistent kernel 的 continuous execution model——传统 kernel-per-op 模型无法实现跨 task tile prefetch

---

## 本层不确定性

1. **TPU v5e/v5p 精确参数与 MoE/DiT Benchmark**: vault 中无 TPU 微架构专用笔记。MXU 128×128/128×256 约束基于公开文档推断。XLA GemmFusion pass 的 tile 自动推导逻辑、VPU memory 精确容量 (~16MB 为推断)、v5 系列 MoE/DiT 推理的 kernel 级 MFU benchmark 均缺失笔记证据。GLaM 50-62% compute utilization 为 TPU v4 训练数据，可能不适用于 v5 推理

2. **Ascend Cube Unit 精确规格与 Kernel Benchmark**: vault 笔记确认达芬奇架构的三单元分离设计 (Cube/Vector/Scalar)，但 Cube Unit MAC array 精确尺寸 (推测 16×16×16)、Vector Unit 256-bit SIMD 宽度、L0/L1 buffer 容量和带宽均未在笔记中明确。CANN/TBE/TIK 在 MoE/DiT/多模态/Video 工作负载上的 kernel 级 MFU/TFLOPs benchmark 数据不足。现有笔记主要覆盖系统级指标 (TTFT/throughput)

3. **AMD MI300X CDNA3 Tile 约束**: vault 中无 AMD tile 切分的详细笔记。仅 wave specialization 退化分析 (来自 HipKittens 论文的二手引用)。CDNA3 Matrix Core 的具体 tile 约束、ROCm/HIP kernel 实现与 benchmark 数据缺失

4. **Groq LPU / Cerebras CS-3**: vault 中无 Groq Tensor Streaming Processor 或 Cerebras Wafer-Scale Engine 的微架构笔记。确定性编译时调度的 tile 约束、SRAM-only 架构的 memory hierarchy 特性未覆盖

5. **DiT MLP/Conv 的特定 Tile 参数**: vault 笔记以 SLA fused attention 和 ChituDiffusion dEngine 为主，DiT MLP/Conv 的特定 tile shape 参数未独立记录。Video 3D Conv kernel tiling 笔记覆盖不足 (Video DiT 以 attention-based backbone 为主，3D Conv 主要在 VAE 中)

6. **多模态 Cross-Attention Fusion Tile**: vault 无专门的多模态 cross-attention fusion kernel 笔记。以上分析基于 attention tile 通用原理和 Flex Attention 的 attention abstraction 推断

7. **Dynamic Parallelism 实验数据**: 笔记仅提及 CDP (CUDA Dynamic Parallelism) 概念 (~5-10μs per device-side launch)，无 MoE/DiT 场景的 per-expert dynamic launch 定量 benchmark。与 CUDA Graph 不兼容的限制在笔记中提及但无实验对比

8. **Register Spilling 定量影响**: 笔记指出 register spilling 导致数据溢写到 HBM local memory (~400 cycles penalty)，但 vault 未提供 spilling 对具体 kernel (MoE expert FFN、DiT attention) 延迟的定量数据

9. **Warp Scheduler 精确调度算法**: NVIDIA 未公开文档化 warp scheduler 的精确调度策略。"greedy-then-oldest" 和 "loose round-robin" 基于 Olmedo et al. (2020) 的微 benchmark 逆向推断，可能在不同 GPU 代数间存在差异

10. **Persistent Kernel 的异质任务通用性**: ACS 论文指出 persistent thread 框架假设所有 tasks homogeneous (相同 register/SMEM 使用)，heterogeneous tasks 导致使用 max(regs)→降低 occupancy。多模态模型中算子多样性 (cross-attention + vision encoder + text encoder) 可能使 mega-kernel 效率下降，但 MPK 通过 compile-time task decomposition 和 Mirage superoptimizer 的部分缓解在笔记中缺乏多模态场景验证

11. **Apple ANE / Apple M2 GPU Memory Hierarchy**: vault 笔记中无 Apple Neural Engine 或 Apple Silicon GPU 的 memory hierarchy 详细笔记。仅 ThunderKittens 提及跨平台移植到 Apple M2 (Metal)

12. **Power Consumption 与 TOPS/W 效率**: 笔记中 H100/TPU v4 有 TDP 数据，但各框架在 MoE/DiT/多模态/Video 工作负载下的实测功耗、TOPS/W 效率对比在 vault 中不充分

---

[HORIZON_SUMMARY_DONE] L4
