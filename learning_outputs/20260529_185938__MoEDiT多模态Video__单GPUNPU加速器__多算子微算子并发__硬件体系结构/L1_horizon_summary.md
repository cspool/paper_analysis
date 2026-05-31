# L1: 算法 Pipeline — 水平分类总结

## 问题覆盖概览

| Q-ID | 问题主题 | 覆盖方法数 | 关键方法 |
|------|---------|-----------|----------|
| Q1.1 | MoE/DiT/多模态/Video 推理计算流程与硬件瓶颈 | 4 | MoE Forward Pass, DiT Denoising Loop, MLLM Concatenation Pipeline, Video DiT 时空 Attention |
| Q1.2 | 算法层面加速方法（量化/蒸馏/稀疏/推测解码等） | 8 | Capacity-Aware Token Drop, PTQ(DMQ/Q-VDiT/S²Q-VDiT), MoE KD(MoS), Speculative Decoding(MoESD/IFMoE), KV-Cache Compression, Step Distillation(CM), VisiPruner, Frame Skipping |
| Q1.3 | 各硬件平台实现框架与工具链 | 12 | vLLM, TensorRT-LLM, DeepSpeed-MoE, SGLang, Triton/CUTLASS/TileLang, FasterTransformer, Ascend CANN/MikPoly, Apple ANE, AMD ROCm/MIGraphX, TPU JAX/XLA, Groq LPU, Cerebras WSE |
| Q1.4 | 多算子/微算子并发算法可行性 | 13 | PROBE Phase-Locked Co-Scheduling, Pre-gated MoE, DeepSeek-V3 DualPipe, HATB, MMDiT, VisiPruner Vision Exit, Cypress, PIT Tiling, Shepherd Operator, Stratum NMP, HNLPU Attention Buffer, Irregular A2A(Lancet), EEVEE Modal Cache |
| Q1.5 | 并发面临的关键挑战与约束 | 5 大挑战 | 数据依赖串行化, 并发粒度 trade-off, 负载均衡(Double Penalty), Memory Bandwidth 竞争, 同步/控制流开销 |
| Q1.6 | 面向硬件的设计空间与最佳实践 | 10 | Nimble AoT Multi-Stream, MPK In-Kernel Runtime, Comet Tile-Level Overlap, Lancet Whole-Graph Pipeline, Welder Tile-Graph, FlashFuser DSM Fusion, SN40L Streaming Dataflow, SCAR Chiplet Scheduling, PROBE Balance Planning, ParallelKittens LCSC |

---

## 按实验环境分类

| 分类 | 方法 | 具体方法描述 | 硬件平台 | Benchmark | 实现框架 | 来源 |
|------|------|-------------|----------|-----------|----------|------|
| **GPU/NVIDIA** | MoE Standard Forward Pass | 逐 token Top-K Sparse Routing + Expert FFN（Router→Dispatch→Expert GEMM→Combine），含完整伪代码和张量形状；Expert FFN 间完美并行但受 token 分配不均影响 | H100/A100 | Mixtral-8x7B (46.7B) | PyTorch, CUTLASS GroupedGEMM | Q1.1; `MoE notes` (2847.6), `EPS-MoE` (12326.9) |
| **GPU/NVIDIA** | DiT Denoising Loop | T 步迭代去噪（T=50~1000），每步含 adaLN Modulation + MHA + MLP FFN + DDIM Step；步间严格串行，步内 Multi-Head 可并行 | H100 | DiT-XL/2 (L=28, N=256 patches) | PyTorch, Difflow compiler | Q1.1; `Difflow` (2240.7), `MMDiT notes` (262.0) |
| **GPU/NVIDIA** | MLLM Concatenation Pipeline | Vision Encoder (ViT) + Projector + LLM Backbone (三阶段跨模态交互：Shallow/Middle/Deep)，Vision 和 Text 编码可并行 | H100/A100 | LLaVA-v1.5 7B (N_v=576) | PyTorch, vLLM | Q1.1; `VisiPruner` (6481.1), `Three-Stage` (621.6) |
| **GPU/NVIDIA** | Video DiT (MMDiT) | 3D Patchify + 共享 Self-Attn + 独立 FFN (文本/视频双流)，Spatial Attn per frame 并行 + Temporal Attn cross-frame barrier | H100 | HunyuanVideo, EasyAnimate (F=14, 35840 tokens) | PyTorch, MMDiT | Q1.1; `S-DMA` (1645.0), `MoBA` (1735.2) |
| **GPU/NVIDIA** | Capacity-Aware Token Drop | Expert capacity 上限 + Expanded Drop（top-(K+K') 扩展候选集），30% 加速仅 0.9% 精度损失 | H100 (expert parallelism) | OLMoE, Mixtral-8×7B | 论文自有实现 | Q1.2; `Capacity-Aware Inference` (9475.4) |
| **GPU/NVIDIA** | PTQ 量化 (DMQ/Q-VDiT/S²Q-VDiT) | DMQ: LES + PTS + Adaptive Timestep Weighting；Q-VDiT: TQE + TMD；S²Q-VDiT: Hessian-aware SDS + Sparse Token Distillation | H100/H800 | CogVideoX-2B/5B, HunyuanVideo-13B | AWQ, GPTQ, TensorRT-LLM, ViDiT-Q, FlatQuant | Q1.2; `DMQ` (3729.4), `Q-VDiT` (5777.5), `S²Q-VDiT` (4754.4) |
| **GPU/NVIDIA** | MoE 知识蒸馏 (MoS) | Teacher MoE (8×7B) → Student MoE (4×3B)，~4× 参数减少，单卡 A100-40GB 可行 | A100 | Mixtral-8x7B, DeepSpeed-MoE | DeepSpeed-MoE, MoESys Graph Optimization | Q1.2; `KD for MoE` (8563.9) |
| **GPU/NVIDIA** | 推测解码 (MoESD/IFMoE) | Draft (2/6 experts) → Target (6/6 experts) 并行验证；MoESD 提出 target efficiency 度量；SpecMoEOff 支持 CPU-GPU offloading 场景 | H100/A100, CPU-GPU offload | LLM benchmarks (HumanEval) | MLC-LLM, vLLM | Q1.2; `SD notes` (4483.2), `MoESD` (~6500) |
| **GPU/NVIDIA** | KV-Cache 压缩 (H2O/Cross-Self Pruning) | H2O Heavy-Hitter eviction + Sliding Window；Cross-Self 区分 cross-KV（视觉）和 self-KV（文本）进行模态感知剪枝，多节省 30-50% | H100/A100 | LongBench, Needle-in-Haystack, MME, MMBench | FlashAttention, PagedAttention | Q1.2; `Rethinking KV-Cache` (4857.4), `Cross-Self KV Pruning` (4837.8) |
| **GPU/NVIDIA** | VisiPruner 视觉 Token 剪枝 | 跨模态注意力不连续性检测 → 仅 ~10/576 视觉 token 关键 → 剪枝 99% 视觉注意力计算，53.9% FLOPs 减少 | GPU | LLaVA-v1.5, InternVL2 | Training-free, plug-and-play | Q1.2; `VisiPruner` (3026.3) |
| **GPU/NVIDIA** | 帧跳过 / VideoNSA | Keyframe detection (帧间差异/Learnable Selector) + PivotKV 关键帧完整 KV + 非关键帧稀疏 KV；从 256 帧→32 帧减少 memory-bound 压力 | GPU | VideoLLM-Online, VideoNSA benchmarks | VideoNSA, ReTaKe | Q1.2; `V-Rex` (1634.4), `VideoNSA` (614.9) |
| **GPU/NVIDIA** | vLLM PagedAttention | Block 级 KV cache 虚拟内存管理（block_size=16/32），利用率 ~96%；CUDA Graph capture 消除 kernel launch overhead | H200/H100/A100 | Llama-3.3-70B-FP8, Mixtral-8x7B; ShareGPT | vLLM (Apache-2) | Q1.3; `Shift Parallelism` (2658.0), `Survey` (1287.6) |
| **GPU/NVIDIA** | TensorRT-LLM INT4/FP8 MoE | 图融合（FusedQKV/FusedFFN/MoE dispatch+CUTLASS GroupedGEMM）+ INT4 量化注入 + CUDA Graph，FP8 MFU 85-95% | H100/H800/A100 | LLaMA-2/3 7B-70B, Mixtral-8x7B; MLPerf v4.0 | TensorRT-LLM, CUTLASS, AWQ/GPTQ/SmoothQuant | Q1.3; `FasterTransformer` (171.9), `LiquidGEMM` (201.0) |
| **GPU/NVIDIA** | DeepSpeed-MoE 混合并行 | DP+TP+EP 四维组合 + 分层 All-to-All (intra-node NVLink + inter-node InfiniBand)；HAP ILP 搜索最优并行配置 | A100/A6000/V100/H100 | Mixtral-8x7B, Qwen1.5-MoE, Qwen2-57B-A14B | DeepSpeed-FastGen + HAP | Q1.3; `HAP` (3346.2), `All-to-All notes` (461.4) |
| **GPU/NVIDIA** | SGLang RadixAttention | 前缀树 (Radix Tree) 跨请求共享 KV cache；系统提示词节省 30-60% KV 占用；1343 T/s decode (Qwen3-30B-A3B @ 4×A6000) | A100/H100/H20/A6000 | Qwen3-30B-A3B, Mixtral-8x7B; GSM8K, Arena Hard | SGLang (开源) | Q1.3; `AI Agent Serving` (1799.7) |
| **GPU/NVIDIA** | Triton/CUTLASS/TileLang Kernel 编程 | 三层抽象：Python DSL (Triton) → Tile-Level (TileLang Layout/Pipeline Inference) → 细粒度 CUTLASS CuTe；TileLang ~50 行实现 DeepSeek MLA，95%+ CUDA 性能 | H100/A100/V100, MI300/MI250, NPU | GEMM/Attention/MoE kernel benchmarks | Triton, CUTLASS 3.x, TileLang | Q1.3; `TileLang` (672.8, 2410.9) |
| **GPU/NVIDIA** | PROBE Phase-Locked Co-Scheduling | 双轨执行模型：主轨道（确定性 MoE 算子序列）+ 辅助轨道（Lookahead Prediction → Balance Planning → Expert Prefetch），split-phase 避免带宽竞争 | 4×A100 (NVLink 600 GB/s) | Mixtral-8x7B, Qwen2-MoE; Prefill -32%, Decode +41% | NVSHMEM + single-SM kernel | Q1.4/Q1.5; `PROBE` (5623.0, 12085.9) |
| **GPU/NVIDIA** | Pre-gated MoE | 在 Layer L-1 用 pre-gating function 预测 Layer L 的 expert 选择；expert 权重预取与 L-1 计算重叠；1.5-3× speedup vs CPU offloading | 8×A100, NVLink | Mixtral-8x7B, DeepSeek-V2 | FasterTransformer v5.2 | Q1.4; `Pre-gated MoE` (4126.5, 4862.4) |
| **GPU/NVIDIA** | DeepSeek-V3 DualPipe | Micro-batch 级计算-通信流水线重叠；SM 分区隔离计算/通信；Node-Limited Routing 确保通信量可控；近 100% overlap | 2048×H800, NVLink+NVSwitch | DeepSeek-V3 (671B) | DeepSeek 自研 (CUDA Graph + NCCL fusion) | Q1.4; `DeepSeek-V3` (1971.0) |
| **GPU/NVIDIA** | Comet Tile-Level Fused MoE | Shared Tensor Decomposition + Tile Reordering (T_local/T_mixed/T_remote) + Thread Block Specialization；Hide 86.5% communication，单层 1.96× speedup | H800/L20 | MoE benchmarks (Megatron-Cutlass baseline) | CUTLASS + NVSHMEM + Megatron | Q1.6; `Comet` (1008.6, 978.9) |
| **GPU/NVIDIA** | Nimble AoT Multi-Stream | AoT CUDA Graph capture + Ford-Fulkerson 最大匹配自动发现 DAG 并行度 → 多 stream 并发；vs PyTorch up to 22.34× | V100/A100/H100 | NASNet-A (max concurrency=15) | PyTorch + CUDA Graph API | Q1.6; `Nimble` (967.6) |
| **GPU/NVIDIA** | MPK (Mirage) In-Kernel Runtime | SM 分区 (128 Workers + 4 Schedulers) + event-driven + cross-task software pipelining + paged SMEM (32KB pages)；12.5ms/token（下限 ~10ms） | A100/H100/B200 | GEMM, Attention workloads | torch.compile(backend=MPK) | Q1.6; `Mirage` (243.4) |
| **GPU/NVIDIA** | FlashFuser DSM Fusion | 利用 Hopper DSM 实现跨 SM cluster 的 kernel fusion；dsm_all_exchange/dsm_shuffle/dsm_reduce_scatter primitives；HBM access -58%，vs Chimera 4.1× | H100 SXM | Attention+FFN fusion chains; SGLang E2E 1.24× | CUTLASS + CUDA 12.4 | Q1.6; `FlashFuser` (656.5, 3228.3) |
| **GPU/NVIDIA** | Welder Tile-Graph Memory | Tile propagation 自动对齐 tile shape + traffic cost model + 双层搜索（Graph Connecting + Sub-Graph Tiling）；89 种非常规 fusion pattern 自动发现 | V100/A100 | BERT attention DRAM traffic -69% (840→264MB), NeRF 7-layer MLP 5× speedup | ONNX + tile-graph compiler | Q1.6; `Welder` (939.5) |
| **GPU/NVIDIA** | Lancet Whole-Graph Pipeline | 沿 batch 维度 P 个 partition → 4-stage pipeline (NMC→A2A→Expert→Post)，跨 partition 计算-通信重叠 | A100/V100 多节点 | MoE 训练 benchmarks | Tutel + NCCL + RAF | Q1.6; `Lancet` (1158.1) |
| **GPU/NVIDIA** | AEP/AMoE 异步 EP | µ-queuing + 去 barrier 同步 + 异步 expert 执行；token 到达即处理不等待 straggler；2.7× 吞吐提升 | Multi-GPU MoE | 原型 MoE 模型 | 论文自有实现 | Q1.5; `Toward Cost-Efficient` (971.6) |
| **GPU/NVIDIA** | MegaScale-MoE Tile Overlap | Tile 级 device memory barrier + SM 分配（少量 SM 通信 + 其余 SM 计算）+ Swizzling 重排 tile 顺序；Hide 86.5% 通信延迟 | 大规模 MoE 训练集群 | 生产环境 MoE | 自研 (ByteDance) | Q1.5; `MegaScale-MoE` (721.3) |
| **GPU/NVIDIA** | Kitsune Tile-Level Dataflow | L2-resident ring buffer queue + 双 arbiter grid scheduler (SIMT+Tensor) + 空间并发 CTA；1.3-2.3× 加速，41-98% off-chip traffic 减少 | A100 (108 SMs) | 5 类 DL 应用 | 需修改 GPU grid scheduler | Q1.5; `Kitsune` (848.9), `Synchronous Dataflow` (2058.2) |
| **GPU/NVIDIA** | HATB 并行 Self+Cross Attention | self-attn 与 cross-attn 共享 Q 投影，K/V 独立并行执行 + Adaptive Gating 门控融合；4/28 layers optimal | 4×A100 TP=4 | Qwen2-7B + Siglip-400M; +103M params (1.5% LLM) | mPLUG-Owl3 | Q1.4; `HATB` (56.4) |
| **GPU/NVIDIA** | Cypress Task-Based | Warp-specialized kernel: TMA 异步搬运 + Tensor Core MMA 流水线；0.88-1.06× cuBLAS GEMM | H100 | GEMM, Fused MHA | Cypress compiler | Q1.4; `Cypress` (3582.8) |
| **GPU/NVIDIA** | ParallelKittens LCSC | Intra-SM / Inter-SM overlapping 两种调度策略 + 8 通信原语模板 | H100 (NVLink) | MoE kernel | ThunderKittens + CUDA | Q1.6; `ParallelKittens` (300.5) |
| **GPU/NVIDIA** | SCAR Chiplet Scheduling | Heterogeneous dataflow MCM scheduling + inter-chiplet pipelining；O(10^56)→sec-level search | MCM accelerator (sim) | Multi-chiplet benchmarks | Custom scheduler | Q1.6; `SCAR` (2159.0) |
| | | | | | | |
| **NPU/Ascend** | Ascend CANN + MindSpore + MikPoly | Da Vinci Core（Cube Unit + Vector Unit + Scalar Unit + MTE）；MikPoly 两阶段 Micro-Kernel 聚合编译（离线生成 Top-40 micro-kernel + 在线 <1ms 聚合），NPU 1.70× vs CANN | Ascend 910B (64GB HBM, Da Vinci Core); 对比 A100 | DeepBench (166 shapes) + 真实应用 (1267 shapes) | CANN SDK v5.1, MindSpore v1.7, TBE, Ascend C | Q1.3; `MikPoly` (471.5), `All-to-All notes` (461.4) |
| **NPU/Ascend** | MTE Communication-over-Computation | 当前 batch MatMul 与下一 batch All-to-All 通信在 MTE 上并行执行；训练效率 5.4-46.6% 提升 (32N/64N/256N Ascend 集群) | Ascend NPU 集群 | MoE training | CANN + MTE 编程 | Q1.3/Q1.4 |
| | | | | | | |
| **加速器/Cerebras** | WSE-3 Weight Streaming | 900,000 PE 2D Mesh 片上路由（单跳 1 cycle）；44GB SRAM + 21 PB/s 片上带宽；权重从外部 MemoryX 流式加载；Llama 4: 2,522 tok/s (~2.5× H100) | CS-2 (WSE-2), CS-3 (WSE-3) | Qwen3-30B-A3B, Llama 4, gpt-oss 120B | Cerebras SDK (PyTorch API) | Q1.3; `WSE notes` (758.9), `Wafer-Scale MoE` (651.8) |
| **加速器/SambaNova** | SN40L Streaming Dataflow Fusion | 硬件原生 streaming dataflow: 1040 PCUs（可配置 SA/SIMD）+ 1040 PMUs；算子以 pipeline 方式流式执行，中间结果永不物化到 off-chip；Gated FFN 全融合为单 spatial pipeline | SN40L RDU (5nm, 2 dies, <650mm²) | 520MB SRAM + 64GB HBM + 1.5TB DDR; 638 BF16 TFLOPS | SambaFlow compiler (PnR) | Q1.6; `SambaNova SN40L` (358.1) |
| **加速器/Groq** | Groq LPU Compiler | Tensor Streaming Processor (TSP)；230MB 片上 SRAM/chip；确定性执行（无 cache miss/warp divergence）；编译器精确编排 memory→network→compute→write 时序 | Groq LPU (14nm) | — | Groq Compiler | Q1.3; `HNLPU` (1832.3, 36.6) |
| **加速器/HNLPU** | Hardwired Neuron LPU ASIC | Metal-Embedding 权重固化硅片；320MB 片上 KV Cache (20,000 bank, 80 TB/s)；5,555× throughput, 1,047× 能效 vs H100 | 5nm 16-chip, 13,232 mm² | GPT 类 LLM | 无软件栈 (ASIC) | Q1.3/Q1.4; `HNLPU` (36.6, 577.0) |
| | | | | | | |
| **AMD** | MI300X ROCm + TileLang | CDNA3 架构: 192GB HBM3, 5.3 TB/s, FP16 1307 TFLOPS；ROCm stack (MIGraphX/MIOpen/rocBLAS/HIP)；TileLang 跨后端支持 MI300/MI250 | MI300X | — | ROCm, MIGraphX, TileLang | Q1.3; `TileLang` (672.8) |
| | | | | | | |
| **TPU** | Google TPU v5e/v5p JAX/XLA | 128×128 MXU systolic array；JAX jit + XLA 编译 HLO IR→TPU 可执行代码；SPMD scatter/gather 支持 MoE routing | TPU v5e/v5p | — | JAX, XLA, SAX | Q1.3 (辅助推断) |
| | | | | | | |
| **跨平台** | TileLang 跨后端 Kernel | 三层编程抽象 (Python DSL → Tile-Level → 细粒度控制)；Layout/Pipeline Inference 自动推导内存布局和异步流水线；支持 CUDA/ROCm/NPU 后端 | H100/A100/V100, MI300/MI250, NPU | DeepSeek MLA (95%+ CUDA 性能) | TileLang + TileScale | Q1.3; `TileLang` (672.8, 2410.9) |
| **跨平台** | Shepherd Operator (Infera) | 小型算子合并为 virtual operator，per-micro-operator→per-shepherd-operator 调度降级；消除 micro-operator scheduling overhead | GPU | — | Infera compiler | Q1.4; `Shepherd Operator` (658.1) |
| **跨平台** | PIT Tiling for MoE | 稀疏 micro-tile→dense tile 重组，置换不变性保证等价；提升 MoE expert GEMM 利用率 | GPU | MoE expert GEMM | Compiler graph pass | Q1.4; `PIT` (47.3) |

---

## 按方法类别分类

| 分类 | 方法 | 具体方法描述 | 核心机制 | 来源 |
|------|------|-------------|----------|------|
| **推理计算流程** | MoE Forward Pass | Router(Softmax+TopK) → Token Dispatch(permute, memory-bound) → Expert FFN × E (并行, compute-bound) → Token Combine + 加权聚合 | Top-K Sparse Routing, Expert 间完美并行, Dispatch 占 step time 34.1% | Q1.1 |
| **推理计算流程** | DiT Denoising Loop | TimestepEmbed + Conditioning → adaLN Modulate → MHA(per head 并行) → MLP FFN → NoisePred + DDIM Step；T 步严格串行 | 步间 barrier, 步内 multi-head 并行, Patch token 数决定 Attention O(N²) | Q1.1 |
| **推理计算流程** | MLLM Concatenation Pipeline | Vision Encoder (ViT, 冻结) → Projector (MLP/Q-Former) → LLM Backbone (三阶段: Shallow/Middle/Deep) → Autoregressive Decode | Vision+Text 编码可并行; 跨模态 Attention 四区域 (V→V/V→T/T→V/T→T); 深层 Vision Exit | Q1.1 |
| **推理计算流程** | Video DiT (MMDiT) | 3D Patchify → Shared Self-Attn (text+video 统一交互) → 独立 FFN (text/video) → NoisePred + Denoising；Spatial Attn (per frame 并行) → Temporal Attn (跨帧 barrier) | 分解时空 Attention (Spatial+ Temporal); MMDiT 共享 Attn + 独立 FFN 可并行 | Q1.1 |
| | | | | |
| **稀疏化与路由优化** | Capacity-Aware Token Drop | Expert capacity 上限 (C×B×S/E) + Expanded Drop (top-(K+K') 扩展候选集)；消除 straggler expert 等待 | 30% 加速/0.9% 精度损失; SM 利用率 40-60%→70-85% | Q1.2 |
| **稀疏化与路由优化** | Pre-gated MoE | Pre-gating function 在 L-1 预测 L 的 expert 选择；expert 权重预取与 L-1 计算重叠 | 打破 Router→Expert 串行依赖; 1.5-3× vs CPU offload | Q1.4 |
| **稀疏化与路由优化** | VisiPruner Vision Exit | 跨模态注意力不连续性检测→深层丢弃视觉 token（仅文本 self-attention）；99% 视觉注意力减少，53.9% FLOPs 减少 | Training-free; Shallow/Middle/Deep 三阶段规律驱动 | Q1.2/Q1.4 |
| | | | | |
| **量化** | PTQ (通用 INT8/FP8/INT4/NF4) | 校准统计 + 量化推理 (INT_MATMUL + rescale)；INT8 2× GEMM 加速 vs FP16, FP8 1.7× | Tensor Core MMA 指令; K 维需对齐 16 | Q1.2 |
| **量化** | DMQ (扩散专用 PTQ) | LES (Learned Equivalent Scaling, channel-wise) + PTS (Power-of-Two Scaling, 移位替代乘法) + Adaptive Timestep Weighting (早期步权重高) | 针对扩散 out-of-distribution outlier; W4A8 低损失 | Q1.2 |
| **量化** | Q-VDiT (视频 DiT 量化) | TQE (Token-Quantization Error compensation, rank-1 低秩) + TMD (Temporal Model Distillation)；W4A6 几乎无损, W3A6 SC 23.40 | ViDiT-Q, FlatQuant, LoRunner Kernel | Q1.2 |
| **量化** | S²Q-VDiT (视频 DiT 量化+蒸馏) | Hessian-aware SDS + Sparse Token Distillation；3.94× 压缩, 1.56× 显存节省, 1.28× 加速 | CUDA kernel (ViDiT-Q+FlatQuant) | Q1.2 |
| **量化** | LiquidQuant W4A8 | 两级量化 FP16→INT8→UINT4；dequant 仅需 IMAD+XOR 两条 32-bit 指令处理 4 元素；H800 接近 FP8 性能 | LiquidServe (ByteDance) | Q1.3 |
| | | | | |
| **知识蒸馏** | MoE KD (Mixture-of-Students) | Teacher MoE (8×7B) → Student MoE (4×3B)；多个 students 通过 gating 分工；联合损失 (CE + KL) | ~4× 参数减少; HBM 占用 ~93GB→~24GB; 单卡可行 | Q1.2 |
| **知识蒸馏** | Consistency Models (步蒸馏) | 学习 f(x_t, t)→x_0 consistency function；多步去噪→1-4 步；8-12.5× 理论加速 | 去噪步数减少→HBM 带宽需求等比例降低 | Q1.2 |
| **知识蒸馏** | D²-DPM 双去噪 | 量化模型 (INT4/INT8) + FP 模型输出融合去噪；(1-β)·x_q + β·x_fp；大部分步在 INT8 Tensor Core 执行 | 与步蒸馏结合: 4 步 INT8 → 0.04× HBM 访问 vs 原始 | Q1.2 |
| | | | | |
| **推测解码** | Standard Speculative Decoding | Draft (小模型/少 expert, γ tokens) → Target (并行验证) → Rejection Sampling (lossless) | 加速比 = (γ·α+1)/(γ·c+1); MoE Self-Draft (IFMoE) 免额外模型 | Q1.2 |
| **推测解码** | MoESD (MoE 专用) | Target efficiency 度量 (draft 质量 × 验证效率)；MoE 中等 batch 下 SD 加速优于 dense | 验证阶段受益于 expert 稀疏性 (更少每 token 计算) | Q1.2 |
| | | | | |
| **KV-Cache 压缩** | H2O Heavy-Hitter Eviction | 累积注意力分数重要性 + 保留 top-budget + 最近 W token；Memory 减少 10-32× | Rethinking: 压缩后 layout 未优化 → 延迟未必减少 | Q1.2 |
| **KV-Cache 压缩** | Cross-Self KV Pruning | 区分 cross-KV (视觉) 和 self-KV (文本) 进行模态感知剪枝；比统一剪枝多节省 30-50% | 视觉 token 占 KV 80-90% → 模态感知剪枝收益大 | Q1.2 |
| | | | | |
| **多算子并发与调度** | PROBE Phase-Locked Co-Scheduling | 双轨执行: 主轨 (确定性 MoE) + 辅助轨 (预测→规划→预取)；split-phase 避免通信-计算带宽竞争 | Prefill -32%, Decode +41% | Q1.4/Q1.5 |
| **多算子并发与调度** | DeepSeek-V3 DualPipe | Micro-batch 级计算-通信流水线 + SM 分区 + Node-Limited Routing；近 100% overlap | Persistent kernel + NCCL fusion | Q1.4 |
| **多算子并发与调度** | Kitsune Tile-Level Spatial Dataflow | L2-resident ring buffer queue + 双 arbiter grid scheduler；Tensor Core + SIMT Core 同时活跃 | 1.3-2.3× 加速, 41-98% off-chip traffic 减少; 需修改 GPU HW | Q1.5 |
| **多算子并发与调度** | AEP/AMoE 异步 EP | µ-queuing + token 到达即处理 + 去 barrier；2.7× 吞吐, 近线性多节点扩展 | 代价: token 乱序需 reorder buffer | Q1.5 |
| **多算子并发与调度** | Nimble AoT Multi-Stream | AoT CUDA Graph capture + MEG + Ford-Fulkerson 最大匹配 → 多 stream 并发 | vs PyTorch up to 22.34×; max concurrency=15 | Q1.6 |
| **多算子并发与调度** | MPK (Mirage) In-Kernel Runtime | SM 分区 (128W+4S) + event-driven + cross-task pipelining + paged SMEM | 12.5ms/token (下限~10ms) | Q1.6 |
| **多算子并发与调度** | HATB (mPLUG-Owl3) | Self-Attn ‖ Cross-Attn 并行 + Adaptive Gating 融合；共享 Q 投影, K/V 独立 | 4/28 layers optimal; cross-attn 延迟隐藏 | Q1.4 |
| **多算子并发与调度** | Cypress Task-Based | Warp-specialized: TMA 异步搬运 + Tensor Core MMA 流水线 | 0.88-1.06× cuBLAS GEMM | Q1.4 |
| **多算子并发与调度** | EEVEE Modal Cache | 缓存 modality-specific module 输出消除跨请求重复计算 | 提升多模态 serving 吞吐 | Q1.4 |
| | | | | |
| **计算-通信重叠** | Comet Tile-Level Fused MoE | Shared Tensor Decomposition + Tile Reordering (T_local/T_mixed/T_remote) + TB Specialization；Hide 86.5% communication | 单层 1.96×, E2E 1.71× | Q1.6 |
| **计算-通信重叠** | Lancet Whole-Graph Pipeline | batch 分 P 个 partition + 4-stage pipeline (NMC→A2A→Expert→Post) | 跨 partition 重叠; 多 GPU 训练为主 | Q1.6 |
| **计算-通信重叠** | MegaScale-MoE Intra-op Overlap | Tile 级 device memory barrier + SM 分配 (少量通信+其余计算) + Swizzling | Hide 86.5% 通信延迟 | Q1.5 |
| **计算-通信重叠** | Irregular All-to-All (Lancet) | 双趟 A2A: 先交换 size→再传输数据；不传 padding tokens | 3.83% prediction error | Q1.4 |
| | | | | |
| **算子融合** | FlashFuser DSM Fusion | Hopper DSM 跨 SM cluster kernel fusion；dsm_all_exchange/shuffle/reduce_scatter primitives；~1.15×10^6 候选→Top-11 profiling | HBM access -58%, vs Chimera 4.1× | Q1.6 |
| **算子融合** | SN40L Streaming Dataflow | 硬件原生 streaming: PCU SA/SIMD + PMU composable mem；Gated FFN 全融合为单 spatial pipeline | 中间结果永不物化到 off-chip | Q1.6 |
| **算子融合** | Welder Tile-Graph Memory | Tile propagation 自动对齐 + traffic cost model + 双层搜索；89 种 fusion pattern 自动发现 | DRAM traffic -69% (BERT); NeRF 7-layer MLP 全融合 5× | Q1.6 |
| **算子融合** | Shepherd Operator | Micro-operator→virtual operator 合并; per-shepherd-operator 调度降级 | 消除 micro-operator scheduling overhead | Q1.4 |
| | | | | |
| **Memory Planning** | Welder Tile-Graph | Tile propagation 反向推导 + SetConnect memory level 选择 (L0/L1/L2) + traffic cost model | Inter-layer independence 解耦优化 | Q1.6 |
| **Memory Planning** | MPK Paged SMEM | 32KB fixed pages + interval graph coloring 复用 + cross-task pipelining 预取 | H100: 7 pages/SM, A100: 5 pages/SM | Q1.6 |
| **Memory Planning** | PROBE Greedy Planning | T_window 约束 + water-filling 策略 + max kmax=16 iterations | 每 rank T_window 按计算-带宽比动态确定 | Q1.6 |
| | | | | |
| **实现框架** | vLLM | PagedAttention (block 级 KV 虚拟内存) + Continuous Batching + CUDA Graph | KV 利用率 ~96%, H200 peak 69,147 tok/s | Q1.3 |
| **实现框架** | TensorRT-LLM | 图融合 + INT4/FP8 量化注入 + In-flight Batching + CUDA Graph | FP8 MFU 85-95% | Q1.3 |
| **实现框架** | DeepSpeed-MoE | DP+TP+EP 混合并行 + 分层 All-to-All (NVLink+InfiniBand) | HAP ILP 搜索 1.01-1.77× vs TP | Q1.3 |
| **实现框架** | SGLang | RadixAttention (前缀树 KV 共享) + Structured Prompt Programming | KV 节省 30-60%, 与 vLLM PagedAttention 互补 | Q1.3 |
| **实现框架** | Triton/CUTLASS/TileLang | 三层 kernel 编程抽象；TileLang Layout/Pipeline Inference；跨 CUDA/ROCm/NPU | TileLang ~50 行 DeepSeek MLA, 95%+ CUDA 性能 | Q1.3 |
| **实现框架** | Ascend CANN/MindSpore | Da Vinci Core (Cube+Vector+Scalar+MTE)；MikPoly 两阶段 micro-kernel 聚合编译 | NPU 1.70× vs CANN | Q1.3 |
| **实现框架** | Cerebras SDK | WSE Weight Streaming + 2D Mesh 片上路由；Single-GPU-like PyTorch API | Llama 4: 2,522 tok/s | Q1.3 |
| | | | | |
| **硬件体系结构** | GPU SIMT (H100) | 132 SMs, 4th-gen Tensor Core (989 TFLOPS BF16, 1979 TOPS FP8), 80GB HBM3 3.35 TB/s, NVLink 900 GB/s | Warp scheduler 动态调度; CUDA Graph capture; MIG 物理分区 | Q1.3/Q1.5 |
| **硬件体系结构** | NPU SA (Ascend 910B) | Da Vinci Core × N; Cube Unit (矩阵乘) + Vector Unit (激活) + Scalar Unit (控制流) + MTE (通信卸载); 64GB HBM ~1.2 TB/s | 需软件显式管理 L1 Buffer; MTE CoC 通信-计算重叠 | Q1.3 |
| **硬件体系结构** | Dataflow RDU (SN40L) | 1040 PCUs + 1040 PMUs; 520MB SRAM; 三级存储 (SRAM→HBM→DDR); 硬件原生 streaming fusion | 无 kernel launch overhead; 编译器 PnR 映射 | Q1.6 |
| **硬件体系结构** | WSE-3 (Cerebras) | 900,000 PE 2D Mesh; 44GB SRAM 片上; 21 PB/s 片上带宽; Weight Streaming from MemoryX | MoE 通信瓶颈消除 (片上路由 1 cycle/hop vs HBM ~300ns) | Q1.3 |
| **硬件体系结构** | HNLPU (ASIC) | 320MB 片上 KV Cache, 20,000 bank, 80 TB/s; Metal-Embedding 权重固化; 5,555× throughput vs H100 | 1,047× 能效 vs H100; 仅能运行一个模型 | Q1.3/Q1.4 |

---

## 分类详细问答

### 分类: 推理计算流程与数据流

#### 方法: MoE Standard Forward Pass

**完整描述**（来自 Q1.1，基于 `knowledge_notes/算法知识笔记/Mixture of Experts (MoE).md` score: 2847.6 和 `EPS-MoE` score: 12326.9）：

MoE 单请求推理的逐 Token 前向传播流程（以 Mixtral-8x7B 为例，d_model=4096, N=8 experts, K=2）：

```
=== MoE 单请求推理：逐 Token 前向传播 ===
输入: x ∈ R^{T×d_model}  (T 个 token)
参数: N 个 Expert FFN E_i, Router W_r ∈ R^{d_model×N}, K=top-k

for each MoE layer l = 1..L:                         # L=32 for Mixtral
    # Step 1: Self-Attention (Dense, 所有 token)
    x_norm = RMSNorm(x)                              # [T, d_model]
    Q, K, V = x_norm @ W_Q, x_norm @ W_K, x_norm @ W_V
    A = FlashAttention(Q, K, V)                      # O(T·d_model²)
    x = x + A

    # Step 2: Router (Gate Network)
    logits = x_norm @ W_r                            # [T, N]
    probs = Softmax(logits)
    topk_weights, topk_indices = TopK(probs, K)      # [T, K]

    # Step 3: Token-to-Expert Dispatch (memory-bound!)
    for e = 0..E-1:
        tokens_e = {x_norm[t] | e ∈ topk_indices[t]} # 形状 [T_e, d_model], T_e 不均匀

    # Step 4: Expert FFN 并行计算
    for e = 0..E-1:                                   # E 个 expert 可独立并行
        gate = SiLU(tokens_e @ W_gate[e])            # [T_e, d_ff=14336]
        up   = tokens_e @ W_up[e]
        act  = gate * up
        out_e = act @ W_down[e]                      # [T_e, d_model]
        # 总 FLOPs per token: 3·d_model·d_ff ≈ 176M FLOPs

    # Step 5: Token Un-permute + 加权聚合
    y = zeros([T, d_model])
    for t = 0..T-1:
        for k = 0..K-1:
            y[t] += topk_weights[t, k] * out_e[idx_map[t, e]]
    x = x + y
```

**核心机制**：
- 计算复杂度：Attention O(T²·d_model) (compute-bound)，Router O(T·d_model·N) (可忽略)，Expert FFN O(T·K·d_model·d_ff) (dominant)，Dispatch O(T) (memory-bound)
- 数据依赖：Step 1→2→3 严格串行；Step 4 各 expert 无依赖可全并行；Step 5 依赖 Step 4 全部完成
- 硬件瓶颈（来自 EPS-MoE, score: 12326.9）：Token Dispatch 是 memory-bound（All-to-All 占 step time 34.1%，GPU SM efficiency 仅 3.7%）；小 batch Expert GEMM Tensor Core 利用率 <30%；Expert 参数占 96% 总参数量

**来源**: Q1.1; `knowledge_notes/算法知识笔记/Mixture of Experts (MoE).md` (score: 2847.6); `paper_secs/secs_moe/EPS-MoE...` (score: 12326.9); `knowledge_notes/算法知识笔记/Top-K Routing _ Gating Mechanism.md` (score: 3989.1)

---

#### 方法: DiT Denoising Loop

**完整描述**（来自 Q1.1，基于 `Difflow` score: 2240.7 和 `MMDiT notes` score: 262.0）：

```
=== DiT 单请求推理：迭代去噪流程 ===
输入: noise z_T ∈ R^{H×W×C} (如 32×32×4 latent), condition c
参数: DiT blocks × L 层, adaLN + MHA + MLP
timesteps t = T, T-1, ..., 1 (T=50 for SDXL-Turbo)

z = z_T
for step s = T down to 1:                             # ★ 严格串行!
    t_emb = TimestepEmbedding(s)
    c_emb = ConditionEmbedding(c)
    conditioning = t_emb + c_emb

    z_patches = Patchify(z)                           # [N_patches, d_model]
                                                      # N_patches=256 for DiT-XL/2

    h = z_patches
    for l = 1..L:                                      # L=28 for DiT-XL/2
        scale_1, shift_1, gate_1 = AdaLN_MLP(conditioning)
        scale_2, shift_2, gate_2 = AdaLN_MLP(conditioning)

        # MHA (multi-head 可并行)
        h_norm = LayerNorm(h) * (1 + scale_1) + shift_1
        attn = Softmax(Q@K^T/sqrt(d_head)) @ V
        h = h + gate_1 * attn

        # MLP FFN
        h_norm = LayerNorm(h) * (1 + scale_2) + shift_2
        mlp = GELU(h_norm @ W_1) @ W_2
        h = h + gate_2 * mlp

    epsilon_pred = Unpatchify(h)
    z = DenoisingStep(z, epsilon_pred, s)             # DDIM/DDPM update

image = VAE_Decoder(z)
```

**核心机制**：
- Denoising Loop 是最外层串行瓶颈——T=50 步 × ~10ms/步 (H100 DiT-XL) → ~500ms 总延迟；步间无法流水线化（马尔可夫链性质）
- 步内 Multi-Head 可并行（通过 batch-GEMM 实现），但 Attention 和 FFN 之间串行（残差连接依赖）
- 硬件瓶颈：小 batch (N_patches=256) Tensor Core 利用率 <20%；Attention O(N²) 随分辨率暴涨（N=1024 → 16× FLOPs）

**来源**: Q1.1; `paper_secs/secs_2026/29-Difflow...` (score: 2240.7); `knowledge_notes/算法知识笔记/MMDiT (Multi-Modal Diffusion Transformer).md` (score: 262.0)

---

#### 方法: MLLM Concatenation Pipeline

**完整描述**（来自 Q1.1，基于 `VisiPruner` score: 6481.1 和 `Three-Stage Cross-Modal Interaction` score: 621.6）：

```
=== 多模态 MLLM 单请求推理 (LLaVA-1.5 7B) ===
输入: Image I, Text prompt P

# Phase 1: 模态编码 (可并行)
# Vision (GPU Tensor Core)
patches = PatchEmbed(I)                               # [N_v=576, d_vis=1024]
for vit_layer in ViT.blocks (×24):                    # ViT-L/14
    patches = vit_layer(patches)
V = patches[:, 1:]                                    # [576, 1024]

# Text (CPU/GPU)
T_ids = Tokenizer(P) → T_emb = Embedding(T_ids)       # [T_text, 4096]

# Phase 2: Projector
V_proj = Projector(V)                                 # [576, 4096]
# Projector: Linear(GELU(Linear(V))), ~8.4M params

# Phase 3: LLM Backbone (三阶段跨模态交互)
H = Concat([V_proj, T_emb])                           # [596, 4096]
for l = 1..L (32 layers):
    # Attention 四区域: [V→V][V→T][T→V][T→T]
    # 三阶段行为:
    #   Shallow (L1-8): 无实质跨模态融合
    #   Middle  (L9-23): 稀疏 ~10/576 视觉 token 驱动融合
    #   Deep   (L24-32): 视觉 token 不再需要 (Vision Exit)
    attn_out = Softmax(Q@K^T/sqrt(d_head)) @ V
    H = H + attn_out + FFN(RMSNorm(H))

# Phase 4: Autoregressive Decode
# 每步仅计算 1 个新 token + KV-cache 复用
```

**核心机制**：
- Vision Encoder 和 Text Tokenization 可并行（在不同硬件单元）；Projector 是数据搬运瓶颈（V 从 HBM→L2→SM 3 次往返）
- 三阶段跨模态规律：深层 ~1/3 总层数视觉计算可完全消除
- 硬件瓶颈：视觉 token N_v=576 → KV-cache ~302MB (FP16)；Attention O((N_v+T_text)²) 中 V→V 区域 (>95%) 冗余

**来源**: Q1.1; `knowledge_notes/算法知识笔记/Three-Stage Cross-Modal Interaction in MLLMs.md` (score: 621.6); `paper_secs/secs_multimodal_kernel/vLLM-Omni...` (score: 886.0)

---

### 分类: 稀疏化与路由优化

#### 方法: Capacity-Aware Token Drop

**完整描述**（来自 Q1.2，基于 `Capacity-Aware Inference` score: 9475.4）：

MoE 的 sparse gating 在 expert parallelism 下产生 Straggler 效应——token-to-expert 分配不均衡导致某些 expert 过载。Capacity-Aware Token Drop 通过强制 expert capacity 上限来消除尾部延迟：

```
# Capacity-Aware Token Drop
tokens_per_expert = capacity_factor * ceil(B*S / E)

for expert_id in 0..E-1:
    expert_tokens = hidden_states[expert_mask]         # [N_e, D]
    if N_e > tokens_per_expert:
        sorted_idx = argsort(gate_score[expert_mask], descending=True)
        keep_idx = sorted_idx[:tokens_per_expert]
        drop_idx = sorted_idx[tokens_per_expert:]      # 丢弃多余 token
        expert_tokens = expert_tokens[keep_idx]        # [C, D]
    expert_output = expert_ffn[expert_id](expert_tokens)

# Expanded Drop: top-(K+K') 扩展候选集
```

- **变量含义**：capacity_factor C∈[1.0, 2.0] 控制每 expert token 上限；tokens_per_expert 是硬件友好的上界值
- **硬件适配**：均衡后 SM 利用率 40-60%→70-85%；Expanded Drop 在 Mixtral-8×7B 上 1.85× 加速
- **实验**：OLMoE 上 30% 推理加速仅 0.9% 精度损失

**来源**: Q1.2; `paper_secs/secs_moe/Capacity-Aware Inference...` (score: 9475.4)

---

### 分类: 量化

#### 方法: DMQ — 扩散模型专用 PTQ

**完整描述**（来自 Q1.2，基于 `DMQ` score: 3729.4）：

DMQ 针对扩散过程的离群值分布和迭代去噪特性设计专用量化策略：

```
# DMQ 核心机制
1. Learned Equivalent Scaling (LES):
   for each channel c in weight:
       W'_c = W_c / α_c, X'_c = X_c * α_c
       目标: min_α ||W_c·X_c - Q(W_c/α_c)·Q(X_c·α_c)||²

2. Adaptive Timestep Weighting:
   # 扩散早期步骤的量化误差累积效应更大
   w_t = exp(-β·t/T)  # 早期步权重更高

3. PTS (Power-of-Two Scaling):
   # 2 的幂次缩放 → rescale 退化为移位 → 无乘法器
```

- **INT8 vs FP16**：Tensor Core INT8 2× 吞吐（H100: INT8=3958 TOPS vs FP16=1979 TFLOPS）
- **量化格式适配**：INT8 (GPU Tensor Core/NPU Cube 原生), FP8 (H100 原生, E4M3/E5M2), INT4 (4× 压缩但需软件支持), NF4 (非均匀, memory-bound 友好)
- **NPU 适配**：昇腾 910B Cube Unit 原生支持 INT8，与 FP16 使用相同硬件但 2× 吞吐

**来源**: Q1.2; `paper_secs/secs_model_quant/DMQ...` (score: 3729.4); `Q-VDiT` (score: 5777.5); `S²Q-VDiT` (score: 4754.4); `knowledge_notes/算法知识笔记/Post-Training Quantization (PTQ).md` (score: 1171.5)

---

### 分类: 多算子并发与调度

#### 方法: PROBE Phase-Locked Co-Scheduling

**完整描述**（来自 Q1.4/Q1.5，基于 `PROBE` score: 5623.0/12085.9）：

PROBE 通过双轨执行模型将 expert 预取的通信与主路径计算重叠：

```
时间轴 ──────────────────────────────────────────►
主轨道 (Deterministic):
  Layer L-1: [─ Attn ─][─ A2A Dispatch ─][─ Expert Compute ─][─ A2A Combine ─]
                       ↑ 通信密集              ↑ 计算密集
辅助轨道 (Lookahead):
  Layer L:    [─ Predict ─][─ Plan ─][─ P2P Expert Transfer ─]
                   ↑                        ↑
              Gate-Initialized          利用主轨道计算密集期
             Lookahead Predictor        的 NVLink 空闲带宽
```

- **核心机制**：主轨道执行确定性 MoE 算子序列；辅助轨道异步执行 Lookahead Prediction → Balance Planning → Expert Prefetch 三阶段流水线
- **Imbalance Ratio (IR) 量化**：IR = max(L_r) / mean(L_r)；Prefill IR >2.6 (batch ~32K tokens)，Decode IR=1.43-2.28；约 50% 全局计算能力在 barrier 处空闲
- **Double Penalty 效应**：热门 expert GPU 的计算延迟 + 网络拥塞顺序叠加（非并行掩盖），stagger GPU 决定全局延迟
- **split-phase transmission**：expert 权重传输被切分为与 All-to-All 正交的相位，避免通信带宽竞争
- **实验**：4×A100 NVLink 600 GB/s；Prefill latency -32%，Decode throughput +41%

**来源**: Q1.4/Q1.5; `paper_secs/secs_moe/PROBE...` (score: 5623.0, 12085.9)

---

#### 方法: DeepSeek-V3 DualPipe

**完整描述**（来自 Q1.4，基于 `DeepSeek-V3` score: 1971.0）：

DeepSeek-V3 的 DualPipe 实现 micro-batch 级计算-通信流水线重叠：

```
DualPipe 调度 (2 micro-batch, EP=8):
Time ─────────────────────────────────────────►
Micro-batch 0: [─ Attn ─][─ Dispatch ─][─ Expert ─][─ Combine ─][─ FFN ─]
Micro-batch 1:           [─ Attn ───────][─ Dispatch ─][─ Expert ─][─ Combine ─]
                             ↑               ↑             ↑
                        与 mb0 Dispatch  与 mb0 Expert  与 mb0 FFN
                        的通信重叠      计算重叠        通信重叠
```

- **核心机制**：forward pass 的计算和通信分别调度到不同 SM 分区（persistent kernel on 部分 SM + 通信在其余 SM/Copy Engine）
- **Node-Limited Routing**：限制每 token 最多路由到 M 个节点，确保通信量可控
- **硬件要求**：H800 NVLink + NVSwitch；SM 分区 + CUDA Graph + NCCL fusion
- **实验**：2048×H800；近 100% computation-communication overlap

**来源**: Q1.4; `paper_secs/secs_moe/DeepSeek-V3...` (score: 1971.0)

---

#### 方法: Kitsune Tile-Level Spatial Dataflow

**完整描述**（来自 Q1.5，基于 `Kitsune` score: 848.9 和 `Synchronous Dataflow Execution on GPUs.md` score: 2058.2）：

Kitsune 实现 tile 级空间并发——不同算子的 CTA 同时驻留在不同 SM 上，通过 L2-resident ring buffer queue 传递 tile 级中间数据：

```
// MLP: Linear1 → ReLU → Linear2 空间并发
cudaPipeline pipeline;
pipeline.addKernel(kernel_Linear1, CTA_count=64, type=TENSOR);
pipeline.addKernel(kernel_ReLU,   CTA_count=44, type=SIMT);
pipeline.addKernel(kernel_Linear2, CTA_count=44, type=TENSOR);
pipeline.addQueue(queue0, producer=Linear1, consumer=ReLU);
pipeline.addQueue(queue1, producer=ReLU, consumer=Linear2);
pipeline.launch();

// SM 空间并发:
// SM_0: Linear1_CTA_0 (TensorCore) + ReLU_CTA_0 (SIMT)
// SM_1: Linear1_CTA_1 + ReLU_CTA_1
// ...
// 数据流: Linear1 tile → queue0 (L2 cache) → ReLU → queue1 → Linear2
// 全程无 global barrier, 无 DRAM round-trip
```

- **关键创新 vs BSP**：(a) 无 global barrier between operators; (b) 中间数据通过 L2 queue 传递非 DRAM round-trip; (c) Tensor Core + SIMT Core 同时活跃
- **性能**：128-256KB payload 时 aggregate BW 达 2 TB/s (A100 108 SMs)；1.3-2.3× 加速，41-98% off-chip traffic 减少
- **硬件约束**：需双 arbiter grid scheduler (SIMT+Tensor) 实现异质 CTA co-location；当前 GPU 无此支持

**来源**: Q1.5; `knowledge_notes/kernel知识笔记/Synchronous Dataflow Execution on GPUs.md` (score: 2058.2); `paper_secs/secs_multimodal_kernel/Kitsune...` (score: 848.9)

---

### 分类: 计算-通信重叠

#### 方法: Comet Tile-Level Fused MoE Kernel

**完整描述**（来自 Q1.6，基于 `Comet` score: 1008.6/978.9）：

Comet 通过 Shared Tensor Decomposition + Tile Reordering + Thread Block Specialization 将 MoE 计算-通信重叠从 coarse chunk 级升级为 fine tile 级：

```
# Comet 三层设计

Layer0 (Communication→Computation):
  shared_tensor 沿 M 分解 → token 粒度
  Tile 分类:
    T_local:  仅含 local tokens → 零等待，立即计算
    T_mixed:  含部分 remote tokens → 中等优先级
    T_remote: 仅含 remote tokens → 最后计算
  # 计算 T_local 时，通信 TB 正在 NVSHMEM 拉取 remote tokens

Layer1 (Computation→Communication):
  # Column-Wise GEMM + 通信重叠
  for col_block in [0, N/T^N):
    所有 expert 并行计算第 col_block 列
    完成后立即 top-K reduce + NVSHMEM write
    # 不等所有列完成！reduce+通信与后续列 GEMM 重叠

Layer2 (TB Specialization):
  通信 TB (n^c): NVSHMEM get token-by-token
  计算 TB (n^p): CUTLASS GroupGEMM (TMA + MMA)
  # n^c/n^p 由 Adaptive Assignment metadata 决定
```

- **核心洞察**：单 token 是最小通信单元，但 GEMM tile (128×128) 需 128 tokens 数据。Coarse pipeline 必须等整个 chunk 到齐；Comet tile reordering 使 local-only tile 立即开始
- **性能**：Hide 86.5% communication（vs FasterMoE 29.2%, Tutel 68.6%）；单层 1.96× vs Megatron-Cutlass；小 M 时 speedup 2.37×
- **硬件**：H800/L20；CUTLASS + NVSHMEM + Megatron

**来源**: Q1.6; `idea_notes/Comet...` (score: 1008.6); `experiment_notes/kernel实验笔记/Comet...` (score: 978.9)

---

### 分类: 算子融合

#### 方法: FlashFuser DSM-Based Cluster-Level Kernel Fusion

**完整描述**（来自 Q1.6，基于 `FlashFuser` score: 656.5/3228.3）：

FlashFuser 利用 Hopper 架构的 Distributed Shared Memory (DSM) 实现跨 SM cluster 的 kernel fusion，突破单 SM shared memory 227KB 限制：

```
# FlashFuser 编译期搜索
1. 算子链统一 Loop 维度建模:
   X = {x_0, x_1, ..., x_{J-1}}  # 共依赖 loop 维度
   每个 x_j: Spatial (多 SM 并行) | Temporal (单 SM 串行)
   组合数: 最多 41 种

2. 两级 Hierarchical Tiling + Resource Mapping:
   Cluster-level tile → inter-block data exchange
   Block-level tile  → reg vs SMEM 分配
   贪心: reusable tensor reg→SMEM→DSM→HBM 逐级放置

3. Cost Model: D_V = data movement volume per memory level
   Cost C = max(V_l / B_l)  # 最慢 memory level 决定延迟
   Pruning → ~1.15×10^6 候选 → Top-11 profiling → 选最优

# DSM Primitives
dsm_all_exchange:   cluster 内 AllReduce (TMA + mbarrier)
dsm_shuffle:        ring communication 交换 C tile
dsm_reduce_scatter: hierarchical two-level reduction
```

- **DSM Bandwidth**：cluster=2 SM: ~8TB/s, cluster=16 SM: ~4TB/s；延迟 ~20ns vs HBM ~280ns
- **Gated FFN 策略**：Spatial Partitioning (不同 Block group 执行两个 GEMM branch) vs Sequential (同一 Block 串行，min DSM communication)
- **性能**：vs Chimera 4.1×, vs PyTorch 3.1×; HBM access -58%; SGLang E2E 1.24×
- **硬件**：H100 SXM, CUDA 12.4 + CUTLASS

**来源**: Q1.6; `experiment_notes/kernel实验笔记/FlashFuser...` (score: 656.5); `paper_secs/.../FlashFuser...` (score: 3228.3)

---

### 分类: Memory Planning

#### 方法: Welder Tile-Graph Memory Optimization

**完整描述**（来自 Q1.6，基于 `Welder` score: 939.5）：

Welder 将 memory 优化从 "手工融合规则" 转变为 "在 tile-graph 上搜索最优 tile 连接配置"：

```
# Welder 四阶段

1. Tile-Graph 构造:
   每个 operator → operator-tile 集合 (每个 tile 处理一个 [BM, BN] output)
   SetConnect(v_a.tile_i → v_b.tile_j, memory_level)
   memory_level ∈ {L0(register), L1(shared memory), L2(global memory)}

2. Tile Propagation (自动对齐):
   从 output tile shape 反向推导整个 graph:
     给定 Matmul O[i,j]=Σ_k A[i,k]×B[k,j]
     output tile [i₀:i₁, j₀:j₁] → A 需 [i₀:i₁, :], B 需 [:, j₀:j₁]
   # 相邻 operator tile 自动对齐，无需手工规则

3. Traffic Cost Model:
   memory_traffic = Σ (input_tile_sizes + output_size) × num_tiles
   Inter-layer independence: 每层 traffic 仅由该层 output tile 配置决定
   # 搜索空间从指数级降为线性级
   双层搜索: 外层 Graph Connecting (L0/L1/L2) + 内层 Sub-Graph Tiling

4. 代码生成: LoadTiles → ComputeTile → StoreTiles
```

- **关键性质**：Inter-layer independence 将耦合的多层优化解耦为独立子问题
- **性能**：89 种非常规 fusion pattern 自动发现（含 48-operator fusion chain）；BERT attention DRAM traffic 节省 69%（840→264MB）；NeRF 7-layer MLP 全融合 5× speedup

**来源**: Q1.6; `idea_notes/Welder Scheduling...via Tile-graph.md` (score: 939.5)

---

### 分类: 实现框架

#### 方法: vLLM PagedAttention

**完整描述**（来自 Q1.3，基于 `Shift Parallelism` score: 2658.0 和 `Survey` score: 1287.6）：

```
vLLM PagedAttention 推理 Pipeline:
====================================
1. Scheduler (Iteration-Level): Orca-style 无 padding 请求选择
2. KV Cache Block 管理: 逻辑 block→物理 block 映射（类似 OS 页表）
   block_table 维护映射关系; block_size=16/32
3. Model Runner — 逐层 Transformer:
   for layer ℓ in 0..L-1:
       # PagedAttention kernel (Online Safe Softmax)
       for block_id in block_table[req_id]:
           K_block = KV_cache_k[block_id]  # [16, head_dim], HBM→SMEM
           V_block = KV_cache_v[block_id]
           scores = q_i @ K_block^T / sqrt(head_dim)
           # running max/sum 在线更新
       # MoE Expert Routing (MoE 层)
       if is_moe_layer: topk_e, topk_w = TopK(softmax(gate_logits), K=2)
4. 采样: next_token = Sample(lm_head(output))

硬件适配:
- H200 (141GB HBM, 4.8 TB/s, FP8 1979 TFLOPS), 8×H200 NVSwitch 900 GB/s
- KV cache 利用率 ~96% (vs 预分配 ~20-30%)
- CUDA Graph capture → kernel launch overhead ~5μs→~0.5μs/kernel
- Llama-3.3-70B-FP8: TTFT 148ms, TPOT 51ms, peak 69,147 tok/s
```

**来源**: Q1.3; `experiment_notes/系统实验笔记/Shift Parallelism...` (score: 2658.0); `experiment_notes/系统实验笔记/A Survey...` (score: 1287.6)

---

#### 方法: TensorRT-LLM INT4/FP8 MoE

**完整描述**（来自 Q1.3，基于 `FasterTransformer` score: 171.9 和 `LiquidGEMM` score: 201.0）：

```
TensorRT-LLM INT4 MoE 推理 Pipeline (H100, Mixtral-8x7B):
===========================================================
离线编译:
1. 图优化 Pass: LayerNorm+QKV 融合, GELU/SwiGLU+MatMul 融合, MoE dispatch+CUTLASS GroupedGEMM+token reorder 融合
2. INT4 量化注入 (W4A16, AWQ per-channel scaling, group_size=128)
   Dequant 融合进 GEMM epilogue, 不产生独立 kernel
3. CUDA Graph capture → TRT Engine

在线推理:
for layer ℓ in 0..31:
    # FlashAttention-3 (H100 Hopper)
    Q,K,V = FusedQKV(h, W_qkv_int4[ℓ])  # INT4 CUTLASS GEMM
    attn_out = FlashAttention3(Q,K,V)

    # MoE Block: 3× INT4 GroupedGEMM per expert
    for expert_id in topk_e:
        gate_out = INT4_GEMM(h, W_gate[expert_id])    # [1, 14336]
        up_out   = INT4_GEMM(h, W_up[expert_id])
        hidden   = SiLU(gate_out) * up_out
        down_out = INT4_GEMM(hidden, W_down[expert_id])
    h = h + Σ topk_w[e] * down_out

量化方案: AWQ (per-channel), GPTQ (per-group), SmoothQuant (W8A8 per-token), FP8 (H100 原生 Transformer Engine)
FP8 1.7× vs FP16, <0.1% PPL 退化; MFU 85-95% on H100
```

**来源**: Q1.3; `knowledge_notes/编译知识笔记/FasterTransformer.md` (score: 171.9); `experiment_notes/算法实验笔记/LiquidGEMM...` (score: 201.0)

---

### 分类: 硬件体系结构差异

#### 方法: GPU SIMT vs NPU SA vs Dataflow RDU 并发模式对比

**完整描述**（来自 Q1.3/Q1.5/Q1.6 综合）：

```
硬件并发架构三范式:

┌─ GPU SIMT (H100) ─────────────────────────────────────┐
│ 132 SMs × 4 Warp Scheduler × 4 Tensor Core              │
│ 存储: HBM(80GB, 3.35TB/s)→L2(50MB)→L1/SMEM(227KB/SM)→Reg│
│ 并发: 多 stream + CUDA Graph + MIG                      │
│ 算子融合: 编译器 fusion (SMEM 227KB 限制)               │
│ 最佳模式: TB specialization (Comet) + DSM fusion        │
│ 关键约束: kernel launch ~100µs, inter-CTA 通信弱        │
├─────────────────────────────────────────────────────────┤
│ NPU SA (Ascend 910B)                                    │
│ Da Vinci Core: Cube(矩阵乘)+Vector(激活)+Scalar(控制流) │
│         + MTE(Memory Transfer Engine, 通信卸载)          │
│ 存储: HBM(64GB,~1.2TB/s)→L1 Buffer→Unified Buffer       │
│ 并发: MTE CoC 通信-计算重叠 + 多级 buffer 分区           │
│ 算子融合: CANN 图编译 + TBE 自定义算子                   │
│ 最佳模式: GEMM systolic + SIMD vector 交替执行           │
│ 关键约束: 需软件显式管理 L1 Buffer; FP8 生态待完善       │
├─────────────────────────────────────────────────────────┤
│ Dataflow RDU (SN40L)                                    │
│ 1040 PCUs(可配 SA/SIMD) + 1040 PMUs(scratchpad)         │
│ 存储: DDR(1.5TB)→HBM(64GB)→PMU SRAM(520MB)              │
│ 并发: 硬件原生 streaming fusion; 无 kernel launch       │
│ 算子融合: 编译器 PnR 自动映射算子树到 PCU/PMU 阵列       │
│ 最佳模式: 空间 streaming pipeline + 多 expert 并发       │
│ 关键约束: 需编译器 PnR; 动态 routing 映射复杂            │
├─────────────────────────────────────────────────────────┤
│ WSE-3 (Cerebras)                                        │
│ 900K PE 2D Mesh, 44GB SRAM, 21PB/s 片上 BW              │
│ 并发: 片上路由 1cycle/hop + 海量 PE 天然并发             │
│ 最佳模式: Weight Streaming + 片上 mesh token 路由         │
│ 关键优势: MoE 通信瓶颈消除 (vs GPU All-to-All 30-50%)   │
│ 关键约束: 44GB SRAM 限制模型规模; 系统功耗 15kW          │
└─────────────────────────────────────────────────────────┘
```

| 维度 | GPU SIMT (H100) | NPU SA (Ascend 910B) | Dataflow RDU (SN40L) | WSE-3 (Cerebras) |
|------|-----------------|---------------------|----------------------|------------------|
| 计算范式 | Warp-SIMT + Tensor Core | 32×32 Systolic Array | Configurable SA/SIMD | 900K PE 2D Mesh |
| 并发粒度 | Thread Block (1024 threads) | PE Array tile (m×32) | PCU pipeline stage | PE 级 (单 cycle) |
| 算子融合 | 编译器 (SMEM 227KB 限制) | CANN layer fusion | 硬件原生 streaming | 片上路由天然融合 |
| 通信 Overlap | Multi-stream + NVSHMEM + TB spec | MTE CoC (专用 DMA) | RDN + P2P protocol | 片上 mesh 路由 |
| INT4/FP8 量化 | ★★★★★ (Tensor Core MMA) | ★★★☆☆ (Cube INT8, FP8 待完善) | ★★★☆☆ | ★★☆☆☆ (Weight streaming 下收益折半) |
| 稀疏 Gating | ★★★★★ (CUTLASS GroupedGEMM) | ★★★★☆ (MTE CoC) | ★★★★☆ (Streaming 多 expert) | ★★★★★ (片上路由消除通信) |
| Speculative Decoding | ★★★★★ (TRT-LLM/vLLM 原生) | ★★☆☆☆ | ★★★☆☆ | ★★★☆☆ |
| KV Cache 压缩 | ★★★★★ (PagedAttention) | ★★★☆☆ (vLLM-Ascend 集成中) | ★★★★☆ (SRAM 免 KV 加载) | ★★★★★ (44GB SRAM) |
| 多算子并发 | ★★★★★ (Streams+MPS+MIG) | ★★★★☆ (多级 buffer+MTE) | ★★★★★ (原生 streaming) | ★★★★★ (900K PE) |

**来源**: Q1.3/Q1.5/Q1.6; `Systolic-array Accelerator` (score: 3882.9); `StreamTensor` (score: 5208.6); `SambaNova SN40L` (score: 358.1); `WSE notes` (score: 758.9); `NVIDIA Ampere Architecture` (score: 4068.9); `Comet` (score: 978.9)

---

## 方法间关系

### 替代关系

- **vLLM ←→ TensorRT-LLM**：同为 GPU 推理框架。vLLM 以 PagedAttention 内存效率见长（KV 利用率 ~96%），TensorRT-LLM 以图融合和量化极致性能见长（FP8 MFU 85-95%）。SGLang 的 RadixAttention 在 vLLM 基础上增加跨请求 KV 共享，形成 vLLM → SGLang 的增强替代链
- **DeepSpeed-MoE ←→ SGLang/TensorRT-LLM MoE backend**：DeepSpeed-MoE 以混合并行（DP+TP+EP）和分层 All-to-All 见长，适合多 GPU；TensorRT-LLM 单 GPU MoE kernel 更优
- **Comet ←→ Lancet ←→ MegaScale-MoE**：三者均解决计算-通信重叠但粒度不同——Comet (tile 级, 单 GPU kernel 内)、Lancet (partition 级, 多 GPU 跨 partition pipeline)、MegaScale-MoE (tile 级 device barrier, 生产环境)。Comet hide 86.5% 在单 GPU 单请求场景最优
- **PROBE Phase-Locked ←→ Pre-gated MoE ←→ DeepSeek DualPipe**：三者均解决 MoE 计算-通信瓶颈——PROBE (双轨 lookahead + split-phase)、Pre-gated (打破 router→expert 串行依赖)、DualPipe (micro-batch 级流水线)。PROBE 和 DualPipe 互补：PROBE 侧重 expert 预取均衡，DualPipe 侧重跨 micro-batch 流水线
- **Kitsune ←→ MPK ←→ Cypress**：三者均追求更细粒度并发——Kitsune (tile 级 spatial dataflow, 需修改 GPU HW)、MPK (in-kernel runtime, SM 分区)、Cypress (warp-specialized, TMA 异步)。当前 H100 上 MPK 和 Cypress 可行，Kitsune 需硬件改动
- **AEP/AMoE ←→ Standard BSP EP**：异步 EP 替代 barrier EP——AEP 去 barrier 同步，token 到达即处理，2.7× 吞吐提升但引入乱序 overhead
- **H2O Eviction ←→ Sliding Window ←→ Cross-Self Pruning**：KV-cache 压缩三种策略——H2O (重要性 eviction)、Sliding Window (简单丢弃老 token)、Cross-Self (模态感知)。Cross-Self 对多模态最优（30-50% 额外节省）

### 互补关系

- **量化 (DMQ/Q-VDiT) + 步蒸馏 (Consistency Models)**：两者组合产生乘数效应——INT8 量化 2× 压缩 × 4-step CM 12.5× 加速 → 理论 25× 端到端加速。D²-DPM 的双去噪方案提供了具体融合路径
- **FlashFuser DSM Fusion + Comet Tile Overlap**：FlashFuser 解决算子链融合（减少 HBM 访问），Comet 解决 MoE 通信-计算重叠。组合使用可同时减少 on-chip 数据搬运和 off-chip 通信
- **Welder Tile-Graph + Nimble Multi-Stream**：Welder 优化 memory traffic（tile 级），Nimble 优化 execution scheduling（stream 级）。Welder 的 tile-graph 可指导 Nimble 的 stream 分配
- **VisiPruner + KV-Cache 压缩**：VisiPruner 剪枝 visual token → 减少需缓存的 KV entries → KV 压缩压力降低。两者叠加：先剪枝 token 再压缩 KV
- **SGLang RadixAttention + vLLM PagedAttention**：SGLang 在 vLLM 的 PagedAttention 之上增加 Radix Tree 前缀共享，两者共享 block 级内存管理基础设施
- **MPK In-Kernel Runtime + Welder Tile-Graph**：MPK 提供 mega-kernel 并行运行时，Welder 提供 tile 级 memory 优化——前者解决调度，后者解决 memory

### 依赖关系

- **All-to-All 通信 ← Router Gating Decision**：MoE 推理的硬依赖链——Router→Dispatch→Expert→Combine
- **Expert FFN 计算 ← Token Dispatch**：Expert 计算依赖 token 完成路由分配
- **Denoising Step(t) → Denoising Step(t-1)**：DiT 推理的时序硬依赖——步骤间不可并行
- **Temporal Attention ← Spatial Attention (all frames)**：Video 推理的帧间 barrier——时序 attention 依赖所有帧的空间特征就绪
- **Cross-Attention ← Vision Encoder + Text Encoder**：多模态推理的模态汇聚 barrier
- **CUTLASS GroupedGEMM ← Token Reorder (by expert)**：MoE kernel 融合的数据布局前置依赖
- **TMA + Tensor Core MMA ← Hopper Architecture**：Cypress 和 FlashFuser 的核心依赖 Hopper TMA 和 DSM
- **NVSHMEM ← NVLink/NVSwitch**：Comet 的 fine-grained 通信依赖 NVSHMEM one-sided 操作
- **TileLang ← 后端编译器 (CUDA/ROCm/NPU)**：TileLang 的跨平台能力依赖各后端的成熟度

---

## 本层不确定性

1. **NPU/TPU 平台实验数据缺失**：华为昇腾 Ascend 910B、Google TPU v5e/v5p、Apple ANE 上的 MoE/DiT/多模态/Video 推理具体 benchmark 数据（latency/throughput/MFU）在 vault 笔记中极少覆盖。Q1.3 的 NPU 部分主要基于 MikPoly 论文（score: 471.5）扩展到 GEMM 动态 shape 优化，非端到端推理。Apple ANE 部分完全为推断。

2. **DiT 推理的精确硬件利用率**：笔记未提供 DiT-XL 在 H100 上单请求推理的 Tensor Core MFU 数值。EPS-MoE 提供了 MoE 的 GEMM 效率分析，但 DiT-specific 硬件效率数据缺失。

3. **DiT Step 间推测执行**：笔记未说明是否存在类似 LLM speculative decoding 的 DiT step 推测机制——能否用轻量去噪步预测后续步结果？

4. **Video Factorized vs Full 3D Attention 硬件偏好**：笔记未提供两种时空 Attention 范式在 H100 上的详细对比——哪种在单请求场景延迟更低取决于 F 和 (H,W) 的相对大小。

5. **多模态 Partial Cross-Attention**：笔记未说明 cross-attention 能否在 encoder 未完全完成时开始（partial fusion 可行性）。

6. **片上互连的 Bank 级并发细节**：L2 crossbar 端口数、HBM channel 级并发分配等微架构细节在笔记中未覆盖。SMEM/Register 划分策略中的具体数值基于 kernel 设计一般原理推断。

7. **Apple ANE / Core ML**：笔记库中无 Apple Neural Engine 在 MoE/DiT/多模态推理上的具体实现和 benchmark。

8. **AMD MI300X 量化性能**：笔记未提供 MI300X 上 INT4/FP8 量化的具体 benchmark。ROCm 生态成熟度评估为推断。

9. **Groq LPU 具体 MoE 数据**：HNLPU 论文提到 Groq LPU 作为 baseline，但未包含 Groq 运行 MoE 的详细 benchmark。Groq 确定性架构与 MoE 动态 routing 的根本性矛盾在笔记中未量化分析。

10. **Token 合并 (ToMe) 的 paper_secs 覆盖缺失**：omnisearch 搜索 "ToMe bipartite matching token merging" 在 paper_secs 中返回空结果，该方法在 vault 中未被覆盖。

11. **FP8 精度影响**：笔记 "FP8 量化 <0.1% PPL 退化" 是基于推理框架的一般经验值，非针对所有 MoE/DiT 模型的系统基准测试。

12. **DiT/多模态/Video 特定框架**：Q1.3 涵盖这些模型负载，但 vault 中实现框架信息主要聚焦于 LLM MoE。Video DiT 在各硬件平台上的 serving 框架覆盖薄弱。

---

[HORIZON_SUMMARY_DONE] L1
