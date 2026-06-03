# L2: Serving 调度 — 水平分类总结

## 问题覆盖概览

| Q-ID | 覆盖方法数 | 关键方法 |
|------|-----------|----------|
| Q2.1 | 10 | EP Barrier, AEP/AMoE, JANUS AEBS, METRO, DiT step调度, CFG双流, VisiPruner三阶段, EEVEE Modal Cache, LiveStar SVeD, 长视频分块 |
| Q2.2 | 11 | vLLM, SGLang, TensorRT-LLM, TGI, LightLLM, Mooncake, S-LoRA, vLLM-Omni, ModServe, EPD-Serve, TetriServe |
| Q2.3 | 10 | Infera Tile-Based, Nimble Stream Assign, vLLM Fused MoE, FlashMoE Megakernel, EEVEE Module Mux, EPD-Serve Stage Disagg, MixFusion Patch, MPK tGraph, MegaScale-MoE Intra-op, FinDEP |
| Q2.4 | 10 | MuxWise SLO-Aware, TZ-LLM Priority Greedy, ACS Out-of-Order, HuntKTm Resource-Aware, QoServe Hybrid, Infera Runtime, SADDLE CI-Aware, MPipeMoE Stream Pipeline, Shepherd Operator, Bullet Dynamic SM Repartitioning |
| Q2.5 | 7 | vLLM Scheduler详细, SGLang Scheduler详细, TensorRT-LLM Runtime详细, LightLLM三层架构, Mooncake Store, MixServe Fused AR-A2A, AMoE µ-queue |
| Q2.6 | 12 | PAT, Shift Parallelism, MegaScale-Infer, Faster MoE Expert Skipping, SpecMoEOff, IFMoE, D2MoE, LiquidServe, μShare, FastTree, FlashAttention-T, HNLPU ASIC |

---

## 按实验环境分类

### GPU/NVIDIA 平台

| 分类 | 方法 | 具体方法描述 | 硬件平台 | Benchmark | 实现框架 | 来源 |
|------|------|-------------|----------|-----------|----------|------|
| GPU/H100 | MuxWise SLO-Aware Dispatcher | Decode优先SM分区（GreenContext CUDA 12.4+），Contention-Tolerant Estimator建模HBM带宽竞争，layer-wise prefill multiplexing，重分区延迟~4.1μs | H100 (132 SM), H200 (141GB) | Conversation + Tool&Agent traces, TBT SLO 50ms(8B)/100ms(70B) | SGLang 0.4.10post2 + PyTorch 2.6.0 + CUDA 12.8 | Q2.4, 笔记: paper_secs 3.4-SLO-aware-Dispatcher (5286.6) |
| GPU/H100 | JANUS AEBS | GPU kernel同步无关expert调度（<90μs开销），单副本→固定实例/多副本→贪心最低负载，NVSHMEM one-sided put绕过CPU，15min间隔SLO-aware scaling | H100, 900GB/s NVLink, 400Gbps InfiniBand | TPOT SLO约束下min(n_a+n_e)配置 | SGLang + NVSHMEM | Q2.1, 笔记: experiment_notes/JANUS (1266.9) |
| GPU/H100 | vLLM Fused MoE | Router→Dispatch→GEMM→Combine单Triton JIT kernel融合，GroupedGEMM（Triton block-level parallelism），token sorting + BLOCK_M=64 padding | H100 | Mixtral-8x7B (MoE-Inference-Bench) | vLLM Triton JIT (`fused_moe.py`)，15-20%吞吐提升 | Q2.3/Q2.5, 笔记: Fused MoE (400.9/390.2) |
| GPU/H100 | FlashMoE Megakernel | Persistent kernel: 1 OS Block (Scheduler warp + Subscriber warps) + N-1 Processor Blocks；doorbell monotonic counter并行sweep实现O(1) task discovery；CUTLASS device-side GEMM；1 kernel launch vs 432 | GPU（分布式MoE） | — | CUDA C++ (~6820行), CUTLASS device-side, NVSHMEM | Q2.3, 笔记: Megakernel (455.7) |
| GPU/H100 | MixFusion Patch-Level | GCD-based uniform patch (patch_size=256)；CSP格式4×integer array O(1)查找；Operator Taxonomy分类（pixel-wise>70%/Self-Attention~20%/Conv~10%）；Patch Edge Stitcher CUDA kernel | H100 80GB | SDXL/SD3多分辨率(512/768/1024) | PyTorch custom ops (CSP, PES CUDA kernel)，Sequential 17.8s→Batched 9.5s | Q2.3, 笔记: Operator Taxonomy (751.0) |
| GPU/H100 | MPK tGraph Mega-Kernel | SM级tGraph替代kernel级DAG；128 Workers + 4 Scheduler-SMs；per-task-pair event（352 bytes metadata/task）；Hybrid JIT+AOT（attention JIT+MatMul AOT）；cross-task pipelining消除pipeline bubble | H100, B200 | Qwen3-8B/1.7B decode | torch.compile(backend=MPK) + Mirage superoptimizer，E2E 1.0-1.7× vs SGLang/vLLM | Q2.3, 笔记: MPK (59.2) |
| GPU/H100 | FastTree | Radix tree→query grouping，tree-structured attention kernel，tensor core GEMM替代decode GEMV | H100 80GB | Llama-2-7B | SGLang v0.2.13 plugin，throughput up to 2.2× vs FlashInfer | Q2.6, 笔记: FastTree (2198.70) |
| GPU/H100 | TensorRT-LLM FP8 | Graph Optimization Pipeline（离线融合+memory planning+kernel auto-tuning）；FP8 Tensor Core 1979 TFLOPS（vs FP16 989 TFLOPS）；TMA warp-specialization（Producer TMA异步预取+Consumer WGMMA计算完全重叠）；In-Flight Batching | H100 SXM, H200, B200 | Llama-3.1-8B/70B, Qwen3 | C++闭源runtime + Python API，NVLink one-sided alltoall | Q2.2/Q2.5, 笔记: FasterTransformer (188.2/46.3) |
| GPU/A100 | vLLM PagedAttention | KV-cache分页（16 token/block），block table逻辑→物理映射，prefix caching via content hash + Copy-on-Write，GPU-side block table消除CPU↔GPU同步 | A100 (108 SM, 80GB) | Toolagent trace (59% cache hit), Conversation trace | Python+C++/CUDA，CUDA Graph预录制整图 | Q2.2/Q2.5, 笔记: PagedAttention (24.0), vLLM (472.9) |
| GPU/A100 | SGLang RadixAttention | Token-level KV-cache pool + Radix Tree前缀树自动前缀复用；FlashInfer BSR格式Paged KV-cache；torch.compile + Triton JIT kernel生成 | A100/H100原生 | — | Python+PyTorch/Triton，CUDA Graph decode | Q2.2/Q2.5, 笔记: SGLang (291.2) |
| GPU/A100 | Infera Runtime | Tile-level DAG调度(G(u) asynchrony wavefront metric)；warp-level SASS binary水平融合；CDP daemon kernel fire-and-forget launch (<10μs)；GDRCopy bypass DMA (<100ns small payload) | A100-PCIE-40GB, CUDA 12.0 | BERT + ViT concurrent multi-model | ~17K LoC C++ kernel-space module from scratch | Q2.3/Q2.4, 笔记: Tile-Based Compilation (1309.2), Shepherd Operator (1413.2) |
| GPU/A100 | Nimble Stream Assign | DAG→MEG（去除冗余传递边）→Bipartite Matching（最大匹配=一组可并行算子）→CUDA Stream分配；理论上实现maximum logical concurrency + minimum synchronizations | A100 40GB ×4 | NASNet-A, BERT, GPT-2 | PyTorch JIT trace + AoT stream assignment，max concurrency=15，1.88× speedup | Q2.3, 笔记: Stream Assignment (29.3) |
| GPU/A100 | EEVEE Module Multiplexing | Visual encoder (70% SM) ↔ Text decoder (30% SM) CUDA MPS空间分区；Synergistic Greedy Search策略生成；Modal Cache复用visual tokens | A100 80GB, RTX 3090 | CLIP, BLIP-2, LLaVA-1.5, InternVL2.5-8B | Python (~5000行) + vLLM + CUDA MPS，max capacity +157%，latency -90% | Q2.1/Q2.3, 笔记: Module Multiplexing (14.6) |
| GPU/A100 | EP Barrier (Baseline MoE) | 四阶段循环：Attention→Router Gating→Dispatch All-to-All (barrier)→Expert FFN→Combine All-to-All (barrier)；GPU stall可达70%，A2A占59.2%延迟 | A100/H100 | DeepSeek-V2-Lite 1A4E | SGLang/vLLM | Q2.1, 笔记: All-to-All Communication (2258.7) |
| GPU/A100 | AEP/AMoE | µ-queue每(block, expert)独立队列 + Defragging Scheduler贪心max-score层选择 + ZeroMQ metadata + NCCL P2P异步传输；消除all-to-all barrier；cold tokens积累到高效batch size (>128) | A100 | — | AMoE Python+C++/pybind11（兼容vLLM API） | Q2.1, 笔记: AEP (1116.3/206.1) |
| GPU/A100 | QoServe Hybrid Prioritization | EDF+SRPF插值优先级（α=8 ms/token最优），三队列（Prefill→Decode→Relegated），Eager Relegation（仅5% relegation维持SLO） | A100/H100 16×集群 | ShareGPT + Azure Conv/Code traces | Sarathi-Serve + vLLM扩展，Goodput 1.5-2.4× vs FCFS | Q2.4, 笔记: QoServe Design (3756.4) |
| GPU/A40 | μShare Multi-Model Co-location | Half-plus blocksize shaping使不同kernel在同一SM内co-locate；time-shifted kernel launch；feedback-based batch sizing | A40 (84 SMs), A800 (108 SMs) | Azure INFless production trace, 10-model co-located | PyTorch 2.2.0 LD_PRELOAD，3046 QPS，SM utilization 15.10% | Q2.6, 笔记: μShare (2141.18) |
| GPU/异构 | MegaScale-Infer | H20 (高内存带宽，attention) + L40S (高计算362 TFLOPS，expert FFN) 异构集群；M2N通信库(~4900行C++)替代NCCL | A800同构/H20+L40S异构 | Mixtral 8x22B | 自研，per-cost/power throughput，1.5-2.0× serving cost降低 | Q2.6, 笔记: MegaScale-Infer (33.94) |
| GPU/端侧 | D2MoE On-Device | Bit-Width-Aware I/O-Compute Pipeline + HEBF调度 + Memory Budget Scheduler | RTX 3060 (6GB), Jetson AGX Orin (64GB) | Mixtral 8×7B, LLaMA-MoE | 自研 (~2500行Python+CUDA)，38-83 tok/s | Q2.6, 笔记: D2MoE (160.49) |
| GPU/端侧 | SpecMoEOff | Speculative decoding隐藏CPU-GPU offloading延迟，CPU-GPU微批次流水线 | A30 (24GB, 165 TFLOPS), RTX 4090D (83 TFLOPS) | Mixtral-8x7B + EAGLE draft | SGLang + MoE-Lightning (~20,000+行) | Q2.6, 笔记: SpecMoEOff (4143.76) |

### NPU/Ascend 平台

| 分类 | 方法 | 具体方法描述 | 硬件平台 | Benchmark | 实现框架 | 来源 |
|------|------|-------------|----------|-----------|----------|------|
| NPU/Ascend | EPD-Serve 三阶段解耦 | 7种E/P/D部署拓扑；物理共置关键机制——AI Core MatMul + AI Vector AllReduce算子互补空间复用（Cube Unit 256 MACs/cycle + Vector Unit 256-bit SIMD异构并发）；Mooncake Store传输接口 | Atlas 800I A2 (Ascend 910B, 64GB) | VisualWebInstruct (256+256 mixed) | PyTorch/Ascend + Mooncake Store，(E-P)-D吞吐+57-69%，E-P-D SLO attainment 94.34% | Q2.2/Q2.3, 笔记: EPD Disaggregation (19537.8/21.7) |
| NPU/Ascend | MixServe Fused AR-A2A | 基于vLLM（Ascend 910B适配）；Fused AR-A2A通信算法减少跨节点通信瓶颈；HCCL多stream异步通信+计算重叠 | Ascend 910B + H20 | DeepSeek-R1 | vLLM + Tutel，TTFT 2.67× vs vLLM TP+PP，1.70× vs vLLM DP+EP | Q2.5, 笔记: Fused AR-A2A (21.7) |
| NPU/Ascend | vLLM-Ascend CANN适配 | 通过CANN（Compute Architecture for Neural Networks）移植vLLM；ACL算子库替代cuBLAS/cuDNN；Ascend Graph（Task Descriptor Chain→硬件TS解析）替代CUDA Graph；Mooncake Store集成（2025.08） | Ascend 910B | — | vLLM Python Scheduler复用 + CANN Adapter API映射层 | Q2.2/Q2.5, 笔记: Mooncake Store (1686.3/30.2) |
| NPU/端侧 | TZ-LLM Priority Greedy + Preemptive | 四类operator DAG扩展（ALLOC/LOAD/DECRYPT/COMPUTE）；COMPUTE优先DAG调度；~64KB micro-operator抢占粒度；REE control plane + TEE data plane co-driver架构 | RK3588 (4×A76 + 3-core NPU ~6 TOPS, 16GB LPDDR4X) | Llama-3-8B, Qwen2.5-7B | llama.cpp扩展 (~1.2K LoC) + NPU co-driver (~1K LoC)，Arm TrustZone，pipeline距lower bound 0.01-9.9% | Q2.4, 笔记: Pipelined Parameter Restoration (141.3), TZ-LLM (187.6) |

### 加速器/ASIC 平台

| 分类 | 方法 | 具体方法描述 | 硬件平台 | Benchmark | 实现框架 | 来源 |
|------|------|-------------|----------|-----------|----------|------|
| ASIC | HNLPU Custom ASIC | 硬连线权重（嵌入金属互联非硅器件）；HN Array accumulate-multiply-accumulate架构；VEX Unit (FlashAttention/RMSNorm/SwiGLU/softmax, 32 cached KV-heads/cycle)；on-chip Control Unit调度+inter-layer pipeline | 5nm ASIC, 16-chip系统 (13,232 mm²), KV Cache 320MB 20,000 banks 80 TB/s | gpt-oss 120B FP4, 2K token length | Verilog RTL (Synopsys DC+ICC+PrimeTime PX)，vs H100(TensorRT-LLM)+Cerebras WSE-3 | Q2.6, 笔记: Hardwired Neuron LPU (36.59) |
| 加速器 | Mooncake Store 分布式KV-Cache | Transfer Engine拓扑感知多协议（RDMA/TCP/CXL/Shared Memory），40GB KVCache达190 GB/s (8×400Gbps RoCE)；Store分布式缓存池(DRAM/VRAM/NVMe分层)；Conductor KV-cache感知全局调度 | 跨平台传输层（已集成vLLM-Ascend 2025.08、SGLang 2025.09、TensorRT-LLM 2025.12） | — | C++/Python，GPUDirect RDMA零拷贝，CXL/共享内存低延迟 | Q2.2/Q2.5, 笔记: Mooncake Store (1686.3/30.2) |

---

## 按方法类别分类

### 分类1: MoE Expert 调度策略

| 分类 | 方法 | 具体方法描述 | 核心机制 | 来源 |
|------|------|-------------|----------|------|
| MoE调度 | EP Barrier (Baseline) | Attention→Router Gating→Dispatch All-to-All (barrier)→Expert FFN→Combine All-to-All (barrier)四阶段循环；GPU stall可达70%，A2A占59.2%延迟（DeepSeek-V2-Lite）；cold expert GPU SM利用率<10%，memory-bound small batch weight loading主导延迟 | Barrier同步→严格串行瓶颈，最慢GPU决定完成时间 | Q2.1, vault: knowledge_notes/All-to-All Communication (2258.7) |
| MoE调度 | AEP/AMoE 异步专家并行 | µ-queue每(block, expert)独立token队列 + Defragging Scheduler贪心max-score层选择（LScore lookahead加权前方token密度）；Receptor Thread（C++ POSIX接收tokens→µ-queue）+ Dispatcher Thread（ZeroMQ metadata + NCCL P2P异步传输）；完全消除all-to-all barrier，GPU自决策执行层 | 异步P2P替代barrier，cold tokens积累到>128 batch size→SM利用率<10%→接近峰值 | Q2.1, vault: knowledge_notes/AEP (1116.3/206.1) |
| MoE调度 | JANUS AEBS 解耦调度 | Attention GPU (n_a) + MoE GPUs (n_e) 独立子集群；AEBS GPU kernel同步无关expert调度（<90μs，no CPU-GPU同步）；NVSHMEM putmem_signal + GPUDirect RDMA替代NCCL；SLO-Aware Scaling: 15min间隔Monte Carlo â_max查找表→enum (n_a, n_e)满足TPOT SLO | GPU kernel内贪心负载均衡（单副本固定/多副本贪心最低负载），vs SGLang per-GPU throughput最高4.7× | Q2.1, vault: experiment_notes/JANUS (1266.9) |
| MoE调度 | METRO Expert-Level Load Balancing | 关键insight: memory-bound decode阶段平衡token数反而降低性能（增加激活expert数加剧HBM压力）。转而min(max_g 激活expert数 per GPU g)；AllGather全局top-k（替代all-to-all）+ GPU并行求解routing optimization | 激活expert数平衡>token数平衡（memory-bound阶段weight loading受HBM带宽约束） | Q2.1, vault: paper_secs/METRO (13897.1) |
| MoE调度 | FinDEP 细粒度DEP调度? | r₁(batch)×r₂(token)两级pipeline；shared expert感知最大重叠；ASAS/AASS两种策略选择 | 细粒度token级调度，attention AG与expert compute流水线重叠 | Q2.1, vault: paper_secs/FinDEP (4142.6) |
| MoE调度 | Faster MoE Expert Skipping/Pruning? | Expert skipping (na 6→2): throughput up to 1.32×；Expert pruning (ne 64→16): up to 2.3× speedup；SGLang v0.4.4 post1 on A800/H200 with DeepSeek-V2-Lite/V3 | **动态减少**激活expert数/剪枝冗余expert | Q2.6, vault: Faster MoE (2480.89) |
| MoE调度 | MixServe Fused AR-A2A | 基于vLLM Ascend 910B适配；Fused AR-A2A通信算法；HCCL异步通信+TP-EP-DP自动推导混合策略；TTFT 2.67× vs vLLM TP+PP，1.70× vs vLLM DP+EP | 通信-计算融合（含HCCS 60 GB/s vs NVLink 900 GB/s跨节点瓶颈） | Q2.1/Q2.5, vault: experiment_notes/MixServe (2220.9/21.7) |

### 分类2: DiT/Diffusion 调度策略

| 分类 | 方法 | 具体方法描述 | 核心机制 | 来源 |
|------|------|-------------|----------|------|
| DiT调度 | DiT Denoising Step 迭代调度 | Step-level循环（T→1逆扩散），每step内：Timestep Embedding→MMDiT Block Forward (N=24-48层，Self-Attention文本+图像共享QKV→AdaLN Modulation时间条件注入→独立FFN模态特定)→Noise Prediction→DDIM/SDE Step；Step间复用c_text conditioning cache | Step间严格数据依赖，每step计算量≈N×(2×4HWC×d²)，50 steps×per-step FLOPs | Q2.1, vault: knowledge_notes/MMDiT (262.0) |
| DiT调度 | CFG Batch 双流调度 | Conditional + Unconditional latent合并batch=2单次forward pass；GEMM M维度翻倍→Tensor Core利用率~50%→~80%+；HBM weight loading减半（仅加载一次）；对比串行双流（2×延迟）和双GPU并行CFG | Batch双流在单GPU最优——减少HBM weight loading次数+提高GEMM效率 | Q2.1, vault: knowledge_notes/MMDiT (262.0, 笔记推断) |
| DiT调度 | TetriServe Deadline-Aware Round-Based | 连续时间切分为固定时长round，每round动态选择请求和SP并行度；FLUX.1-dev/SD3上vs固定SP度xDiT baseline提升up to 32% SLO attainment | Round-based scheduler + deadline-aware SP selection | Q2.2, vault: paper_secs/TetriServe (2956.1) |
| DiT调度 | MixFusion Patch-Level 并行分解 | GCD-based uniform patch统一不同分辨率；CSP格式4个integer array O(1)查找；Operator Taxonomy: pixel-wise ops (>70%，batch全29 patches) vs Self-Attention (~20%，per-resolution分组) vs Conv (仅U-Net，PES边界缝合) | DiT无Convolution→patched inference自然100% accuracy；H100 sequential 17.8s→batched 9.5s | Q2.3, vault: knowledge_notes/Operator Taxonomy (751.0) |

### 分类3: 多模态 Serving 调度

| 分类 | 方法 | 具体方法描述 | 核心机制 | 来源 |
|------|------|-------------|----------|------|
| 多模态调度 | VisiPruner 三阶段跨模态调度 | Shallow (L1-8): 视觉+文本token独立演化→跳过cross-attention+视觉self-attention，节省~25%；Middle (L9-23): ~10/576关键视觉token驱动跨模态融合→Influence-based Token Selection仅保留instruction-relevant tokens（~10-50 vs 576），节省~90%；Deep (L24+): Vision Exit完全丢弃视觉tokens→纯文本self-attention，节省~20% | 分层token pruning基于跨模态信息流量化（FlowMM ρ^l metric指导） | Q2.1, vault: knowledge_notes/Three-Stage Cross-Modal (621.3), Cross-Modal Info Flow (707.2) |
| 多模态调度 | EEVEE Modal Cache | Controller计算图像content hash (64-bit)→GPU shared memory Modal Cache查找；命中→跳过视觉编码；Encoder-Decoder: 缓存cross-attention KV pairs；Decoder-Only: 缓存visual prefix self-attention KV；Critical Modal Cache（压缩至~170 tokens/30% retention via attention score排序）优先GPU memory | 跨请求复用消除重复visual encoding；Pipeline overlap: 新请求encoder computation与cache loading重叠 | Q2.1/Q2.3, vault: knowledge_notes/Modal Cache (382.2), Module Multiplexing (14.6) |
| 多模态调度 | vLLM-Omni Stage Graph? | 复杂Any-to-Any多模态模型分解为独立stage（Thinker LLM→Talker LLM→DiT Vocoder），每stage由独立execution engine服务；unified connector (NCD/shared memory/Mooncake RDMA)传输中间数据；Qwen2.5-Omni RTF降低61.4%，Qwen3-Omni RTF降低90.7% | Stage Disaggregation + Unified Connector传输 | Q2.2, vault: paper_secs/vLLM-Omni (2910.6) |
| 多模态调度 | ModServe Modality-Aware Disaggregation? | Image Instances (CPU preprocessing + GPU encoding) ↔ Text Instances (LLM prefill + decode)独立autoscaling；modality-aware routing | 模态级独立扩缩容，InternVL-26B上6.8× throughput vs vLLM monolith | Q2.2, vault: paper_secs/ModServe (2224.6) |
| 多模态调度 | EPD-Serve 三阶段解耦? | Encode (ViT, compute-heavy+大activation) → Prefill (LLM, compute-heavy+产KV Cache) → Decode (memory-bound token-by-token)；7种物理共置拓扑 | 三阶段不同计算/内存特征→解耦消除硬件利用率低；AI Core+AI Vector算子互补复用 | Q2.2/Q2.3, vault: EPD Disaggregation (19537.8/21.7) |

### 分类4: Video Serving 调度

| 分类 | 方法 | 具体方法描述 | 核心机制 | 来源 |
|------|------|-------------|----------|------|
| Video调度 | LiveStar SVeD Streaming 逐帧因果? | Vision Encoding (InternViT, 16 tokens/frame)→Streaming SVeD Loop: Cache Lookup (Inter-dialogue Streaming Cache)→Verification Forward Pass (PPL计算)→Gate Decision (PPL>α×PPL_ref→Response Mode生成新描述；else Silence Mode维持状态)；Peak-End压缩（保留高PPL峰值+最新帧） | PPL-gated响应-沉默决策；Strict因果约束（不可预看未来帧）；双级KV Cache (Intra-dialogue + Inter-dialogue) | Q2.1, vault: knowledge_notes/Online Video-LLM (192.5), Streaming KV Cache (556.9) |
| Video调度 | 长视频分块与上下文窗口滑动? | Sliding Window: W=64 frames, S=32 frames每chunk做spatial-temporal encoding + attention KV cache保留最近K chunks；Hierarchical Chunk: L1 short-term (8f)→L2 medium-term (8 chunks, 64f)→L3 long-term (相似度搜索选择关键segments)；Context Window Sliding: LLM 128K token窗口→注意力分驱逐+attention sink保留 | 分层时间表示降低KV cache膨胀；粗粒度segment搜索+细粒度frame分析 | Q2.1, vault: knowledge_notes/Online Video-LLM (192.5, 笔记推断) |

### 分类5: 计算图分解策略

| 分类 | 方法 | 具体方法描述 | 核心机制 | 来源 |
|------|------|-------------|----------|------|
| 算子级分解 | Infera Tile-Based + Shepherd Operator* | 大op→micro-op tile切分（Register/Shared/Global memory三级tile size静态分析决策）+ 小op→Shepherd Operator合并（降低per-op调度overhead）；Multi-version micro kernel (pipeline stage 2/3/4 + async copy + warp specialization)；Zero-tuning（无需GPU profiling反馈） | 自适应粒度控制：tile切分上限+Shepherd合并下限；协同编译-调度：**离线multi-version + 在线动态kernel选择** | Q2.3/Q2.4, vault: Tile-Based Compilation (1309.2), Shepherd Operator (1413.2) |
| 算子级分解 | Nimble DAG→MEG→Stream Assign | DAG→MEG去除冗余传递边→Bipartite Matching（最大匹配=一组可并行算子）→**CUDA Stream分配**+仅在跨stream MEG边界插入CUDA event同步；理论保证maximum logical concurrency + minimum synchronizations | MEG暴露真实直接依赖关系（消除DAG max_flow误导） | Q2.3, vault: Stream Assignment Algorithm (29.3) |
| 算子内Tile级 | MegaScale-MoE Intra-op Tile-Level | 4类tile级通信-计算barrier融合：**A2A+GEMM、GEMM+A2A、AG+Scatter+GroupedGEMM、GroupedGEMM+Gather+RS；per-tile device memory barrier替代kernel-level barrier**；tile到达即计算不等全量通信完成 | Tile级barrier→通信与计算在tile粒度重叠（非operator粒度） | Q2.3, vault: Intra-operator Communication-Computation Overlap (275.9) |
| 层/模块级分解 | vLLM Fused MoE + FlashMoE Megakernel* | Fused MoE: Router→Token Sorting (moe_align_block_size, BLOCK_M=64)→Grouped GEMM Triton kernel (per-expert block-level parallelism)→Weighted Combine atomic aggregation；FlashMoE: **Persistent kernel内含GPU内work-conserving scheduler (doorbell monotonic counter并行sweep)** + CUTLASS device-side GEMM | 消除per-expert kernel launch overhead（6-8次→1-2次，15-20%吞吐提升）；Megakernel进一步融合跨GPU通信→1次launch vs 432次 | Q2.3/Q2.5, vault: Fused MoE (400.9/390.2), Megakernel (455.7) |
| 层/模块级分解 | EEVEE Module Multiplexing* | Visual Encoder (70% SM, B_vis=2) ↔ Text Decoder (30% SM, B_txt=8) **CUDA MPS空间分区**；Offline Synergistic Greedy Search策略生成（逐步增加batch→平衡SM再分配）；Stage-Level Pipeline: CPU预处理↔GPU inference重叠 | MPS SM Partition硬约束（需CUDA初始化前设定→active-standby机制ms级切换） | Q2.3, vault: Module Multiplexing (14.6) |
| Token/Patch级分解 | MixFusion Patch-Level | GCD-based uniform patch + CSP格式 + Operator Taxonomy分类（pixel-wise/context-dependent） | Pixel-wise ops全并行batch；Self-Attention per-resolution分组；Conv PES边界缝合 | Q2.3, vault: Operator Taxonomy (751.0) |
| Mega-Kernel化 | MPK tGraph* | SM级tGraph替代kernel级DAG；per-task-pair event (352 bytes/task metadata, 全部GPU device memory连续数组)；128 Workers + 4 Scheduler-SMs物理分区；Cross-task pipelining (TMA prefetch next weight tile during current compute) | SM级表示暴露kernel barrier遮蔽的细粒度并行；Hybrid JIT+AOT：attention JIT dynamic load balance + MatMul AOT消除dispatch overhead | Q2.3, vault: MPK (59.2) |
| 阶段级分解 | EPD-Serve E/P/D 解耦* | 7种部署拓扑；AI Core (MatMul) ↔ AI Vector (AllReduce) 算子互补空间复用；**Ascend NPU Hardware TS直接解析Task Descriptor Chain**（vs GPU GigaThread Engine软件层干预） | 阶段级解耦依据三阶段不同计算/内存特征；物理共置利用硬件异构计算单元独立性 | Q2.3, vault: EPD Disaggregation (19537.8/21.7) |

### 分类6: Dispatcher 设计方法

| 分类 | 方法 | 具体方法描述 | 核心机制 | 来源 |
|------|------|-------------|----------|------|
| SLO感知调度 | MuxWise SLO-Aware Dispatcher + Bullet Dynamic SM Repartitioning *| Decode SLO优先→Contention-Tolerant Estimator建模HBM带宽竞争→确定SM分区（GreenContext CUDA 12.4+，重分区~4.1μs）；layer-wise prefill multiplexing穿插decode执行；Prefill Preemption（不递归抢占）；Bullet: 每prefill layer group/decode step后读shared metadata buffer→SLO感知重分区 | **CUDA Event异步prefill/decode stream同步**；SM mask硬件寄存器操作（libsmctrl）→后续kernel自动在新SM子集执行 | Q2.4, vault: SLO-aware-Dispatcher (5286.6), Dynamic SM Repartitioning (46.3) |
| 优先级调度 | TZ-LLM Priority Greedy + Preemptive Micro-Operator | 四类operator DAG扩展（ALLOC→LOAD→DECRYPT→COMPUTE每层）、COMPUTE优先级最高；大restoration op切分为~64KB micro-op支持CPU computation就绪时抢占（抢占开销~2.5μs）；REE control plane + TEE data plane co-driver架构 | Pipeline效率距critical path lower bound仅0.01-9.9% | Q2.4, vault: Pipelined Parameter Restoration (141.3), TZ-LLM (187.6) |
| 乱序调度 | ACS Out-of-Order Kernel Dispatch* | Scheduling Window (N=32~64 slots) 运行时read/write segment overlap依赖检测→标记READY kernel并发发射；**ACS-HW: GPU Command Processor内硬件调度窗口（N=32→~1KB SRAM，dispatch overhead ~50-100ns 64 cycles）；ACS-SW: 多线程调度器+cudaStreamWaitEvent** | 模仿CPU乱序执行——消除CPU-GPU同步（StreamSync），kernel完成事件GPU内部处理；特别适合input-dependent irregular DAG | Q2.4, vault: ACS Key Observations (2441.0) |
| 资源感知调度 | HuntKTm Resource-Aware Multi-Dimensional* | 三维SM资源评估（threads/registers/shared_memory归一化到[0,1]→木桶原理选bottleneck维度）+ **hardware queue计数**；非抢占式（task一旦分配不迁移→避免micro-operator中途迁移开销） | 不同micro-operator不同资源瓶颈（GEMM registers/attention shared memory/elementwise threads）→单维度错配 | Q2.4, vault: Resource-Aware Task Dispatching (556.8) |
| 混合优先级调度 | QoServe Hybrid Prioritization + Eager Relegation | EDF+SRPF插值优先级（α=8 ms/token最优，低负载α↓优化tail latency/高负载α↑优先短请求）；**三队列（Prefill→Decode→Relegated）**；Dynamic Chunking：增大chunk→更多prefill tokens与decode融合→HBM bandwidth利用率提升；仅prefill可抢占（不抢占decode/不递归） | Eager Relegation：仅5% relegation在极端过载下维持SLO；复杂度O(log N_new)仅priority queue操作 | Q2.4, vault: QoServe Design (3756.4) |
| 编译-调度协同 | Infera Runtime (SelectKernels→FuseKernels→LaunchKernel)* | SelectKernels: G(u) asynchrony wavefront metric递归传播children并发收益→选zero in-degree + max G(u) data blocks + 回归模型选最优kernel版本（IPC_est, TLP≥4约束）；FuseKernels: SASS binary level warp级水平融合（BAR.SYNC重组+warp交错排布）；LaunchKernel: HKQ→GDRCopy→DKQ→CDP daemon kernel fire-and-forget (<10μs) | **最深度的硬件-软件协同dispatcher——绕过DMA (GDRCopy)、GPU内自主launch (CDP)、SASS级binary manipulation** | Q2.4, vault: Infera Runtime (59.0/95.5/117.3) |
| 算术强度感知 | SADDLE CI-Aware PIM-GPU Dispatch* | 预标定PIM ridge (16.7 FLOP/Byte) vs GPU ridge (208 FLOP/Byte)→**运行时CI vs ridge比较动态operator-to-device映射**；CI<PIM_ridge→memory-bound→PIM高带宽；CI>GPU_ridge→compute-bound→GPU高算力 | Ridge point差距13×驱动异构调度决策 | Q2.4, vault: Arithmetic Intensity-Aware (333.5) |
| 微批次流水线 | MPipeMoE 3-Stream Micro-Batch Pipeline* | **3 CUDA stream (comp/comm/mem)**；offline interference profiling (μ/σ/η slowdown因子)；4种memory reuse策略 (S1-S4) + n∈{1,2,4,8,16}自适应选择→min T_total | σ≈1（计算几乎不受干扰→通信-计算重叠可行）；N大(64 GPU): S4最优（避免memory bandwidth竞争） | Q2.4, vault: MPipeMoE (56.3) |
| 调度粒度优化 | Shepherd Operator* | 编译期识别小operator子图（elementwise add/activation执行仅数μs→scheduling overhead >> 执行时间）→合并为Shepherd Operator（内部按拓扑调用子kernel，对外单一调度单元） | 每次CUDA kernel launch约5-10μs PCIe+command processor overhead→合并10个2μs小op节省9次launch (~45-90μs) | Q2.4, vault: Shepherd Operator (658.1) |

### 分类7: Serving 框架架构与实现

| 分类 | 方法 | 具体方法描述 | 核心机制 | 来源 |
|------|------|-------------|----------|------|
| 框架实现 | vLLM Scheduler 详细* | Python-frontend + C++/CUDA backend；Scheduler._schedule() 七步主循环：请求选择→Preemption(RECOMPUTE/SWAP策略)→Block Table管理(每block 16 tokens)→Prefix Caching Hash匹配→Continuous Batching组装→Model Runner执行(CUDA Graph快速路径 vs PyTorch eager路径)→完成检查+资源回收；GPU-side block table消除CPU↔GPU同步（BrownoutServe方案）；Framework scheduling overhead 6步分解: **Operator Dispatch~20-30μs + Shape Inference~10-20μs + Kernel Selection~5-50μs + Argument Preparation~5-10μs + Kernel Launch~3-5μs + Memory Allocation→CUDA Graph降至<100μs total** | CUDA Graph预录制整图单次launch替代per-op launch；FlashInfer BSR format Paged KV-cache (gather/scatter→contiguous SMEM→Tensor Core MMA) | Q2.5, vault: vLLM (472.9/433.2/390.5), Framework Scheduling Overhead (5079.9) |
| 框架实现 | SGLang Scheduler 详细* | RadixAttention: Radix Tree压缩前缀树（节点存token_seq/kv_cache_ptr/ref_count/children）；HTTP请求→Radix Tree前缀匹配（找到最长匹配前缀→复用其KV节点，ref_count+1）→Token-Level KV Pool逐token分配剩余tokens→In-Flight Batching混合prefill/decode→FlashInfer BSR attention kernel执行→CUDA Graph decode launch→Token采样+Radix Tree更新 | Token级（非block级）KV cache→任意粒度跨请求共享但metadata overhead更高；Bullet方案SM Spatial Sharing: **prefill/decode两个独立MPS进程+libsmctrl SM masking+CUDA IPC共享GPU memory** | Q2.5, vault: SGLang (291.2) |
| 框架实现 | TensorRT-LLM Runtime 详细* | 离线Phase 1: 模型导入(HuggingFace→GraphIR)→Graph Optimization Pass(Operator Fusion+Memory Planning liveness分析+Kernel Auto-Tuning cuBLASLt/CUTLASS+FP8 Precision Pass Transformer Engine)→TRT Engine序列化；运行时Phase 2: **Execution Context初始化(预分配buffers+CUDA Graph预录制+多CUDA stream)**→In-Flight Batching调度循环(Context Phase+Generation Phase同批混合)→MoE层：Router→CUB radix sort token dispatch→CUTLASS Grouped GEMM (FP8/FP16/INT4)→Token Combine→NVLink One-Sided AlltoAll | H100深度整合：FP8 Tensor Core(1979 TFLOPS) + TMA warp-specialization pingpong + Transformer Engine scaling factor自动管理；NVLink one-sided alltoall替代AllGather+ReduceScatter | Q2.5, vault: FasterTransformer (188.2/46.3), All-to-All Comm (16.5), FP8 MoE (17.8) |
| 框架实现 | LightLLM/PiLLM 三层架构* | API Layer→Global Scheduling Layer (Dispatcher按计算特征路由+Instance Count Manager动态分配)+**Execution Instances (Prefill Instances输入处理/Decode Instances输出生成)**；Token-level KV cache = linked list（非block-level）→batch内memory sharing灵活 | Token粒度分配/释放→batch内共享灵活；per-token metadata overhead更高 | Q2.5, vault: PiLLM 5-Impl (1058.1) |
| 框架实现 | Mooncake Store 分布式KV-Cache* | **Transfer** Engine: 拓扑感知RDMA/TCP/CXL/Shared Memory多协议（40GB KVCache 190 GB/s 8×400Gbps RoCE）+ Store: DRAM/VRAM/NVMe分层缓存池+Conductor: KV-cache感知全局调度器→路由请求到KV cache已在本地或最近节点的worker | 分离式架构核心存储引擎→vLLM-Ascend(2025.08)/SGLang(2025.09)/TensorRT-LLM(2025.12)三大框架已集成 | Q2.5, vault: Mooncake Store (1686.3/30.2) |

### 分类8: KV-Cache 管理与优化

| 分类 | 方法 | 具体方法描述 | 核心机制 | 来源 |
|------|------|-------------|----------|------|
| KV-Cache | PagedAttention (vLLM) | 固定大小block (16 tokens)→block table映射逻辑序列到非连续物理GPU内存；prefix caching: content hash + Copy-on-Write语义共享block；GPU-side block table (BrownoutServe)消除CPU↔GPU同步 | OS虚拟内存分页类比→消除碎片+实现内存共享 | Q2.2/Q2.5, vault: PagedAttention (24.0) |
| KV-Cache | RadixAttention (SGLang) | Token-level KV-cache pool + Radix Tree自动前缀复用；tree节点ref_count生命周期管理；比vLLM block-level共享粒度更细但tree维护开销更大 | Token vs block粒度权衡：共享效率↑ vs metadata overhead↑ | Q2.2/Q2.5, vault: SGLang (291.2) |
| KV-Cache | Streaming KV Cache Dual-Level (LiveStar) | **Intra-dialogue (clip内) + Inter-dialogue (跨clip) 双级KV Cache(长视频Cache压缩策略)**；Peak-End压缩（保留高PPL峰值+最新帧）；SVeD Swap：响应→沉默时swap cache末尾位置 | 5min video FPS 3.82 (Both) vs 2.50 (No Cache)，1.53×加速 | Q2.1, vault: Streaming KV Cache (556.9) |
| KV-Cache | Mooncake Store 分布式缓存池 | DRAM/VRAM/NVMe分层存储；拓扑感知多协议传输；大对象条带化+并行I/O+多副本；Conductor全局调度器根据cache位置路由 | 分离式架构下KV-cache跨节点预取/eviction | Q2.2/Q2.5, vault: Mooncake Store (1686.3/30.2) |
| KV-Cache | EEVEE Modal Cache | 视觉编码器输出跨请求缓存复用；64-bit content hash→GPU shared memory查找；Critical Modal Cache压缩（token-wise attention score排序→~30% retention, 576→170 tokens）；Global LRU + host memory spill | 同一图像多问题场景→后续请求完全跳过视觉编码 | Q2.1, vault: Modal Cache (382.2) |

---

## 分类详细问答

### 分类: MoE Expert 调度策略

#### 方法: EP Barrier 调度 (Baseline)

**笔记证据**: `idea_notes/Toward Cost-Efficient Serving of Mixture-of-Experts with Asynchrony.md` (score: 1622.8), `knowledge_notes/kernel知识笔记/All-to-All Communication in MoE Expert Parallelism...md` (score: 2258.7)

MoE单请求推理在标准Expert Parallelism (EP)下按decoding block的barrier同步模式执行。调度流程为四阶段循环：

```
=== MoE EP Barrier 调度流程 (SGLang/vLLM, 单请求, 4 GPU EP) ===

请求 → Tokenizer → [Decoding Block Loop] → Sampler → 输出

每个 Decoding Block 的调度时间线（以 DeepSeek-V2, 1A4E 为例）:

Step 1: Attention 计算 (所有 GPU 并行)
  GPU_0 (持有 KV Cache):  
    QKV_proj → FlashAttention → output_proj → hidden_states [B, S, 7168]
  GPU_1-3 (仅持有 expert 权重): 
    idle / 等待
    
Step 2: Router Gating (GPU_0)
  gate_logits = hidden_states @ W_gate           # [B, S, 256 experts]
  topk_weights, topk_indices = TopK(softmax(gate_logits), k=8)
  
Step 3: Dispatch All-to-All ← BARRIER
  # NCCL alltoallv: 所有 GPU 必须同时调用，最慢 GPU 决定完成时间
  # DeepSeek-V2-Lite: all-to-all 占 MoE layer forward 延迟的 59.2%
  
Step 4: Expert FFN Compute (所有 GPU 并行)
  各 GPU 对收到的 tokens 执行本地 expert FFN
  # GPU 间负载不均衡: hot expert GPU 处理大量 tokens，cold expert GPU 处理极少 tokens

Step 5: Combine All-to-All ← BARRIER
  # GPU stall 可达总执行时间的 70% (AMoE Fig.4)

Step 6: Token Merge & Residual
  final_output = Σ(topk_weights[e] * expert_out[e]) + attention_residual
```

**硬件视角**:
- **SM 占用率**: Cold expert GPU 的 SM 利用率极低——当只有少量 tokens 到达时，GEMM 的 M 维度极小，Tensor Core 利用率 < 10%
- **HBM 带宽压力**: Cold expert 小 batch 场景下，weight loading 主导延迟（HBM bandwidth-bound），而非计算
- **脉动阵列利用率**: NPU 上同样存在 batch size < 128 时吞吐量远低于线性的问题

**来源**: Q2.1, `Q2.1_L2_answer.md`, vault: `knowledge_notes/All-to-All Communication...md` (score: 2258.7)

---

#### 方法: AEP 异步专家并行调度 (AMoE)

**笔记证据**: `knowledge_notes/系统知识笔记/Asynchronous Expert Parallelism (AEP _ 异步专家并行).md` (score: 1116.3), `idea_notes/Toward Cost-Efficient Serving of Mixture-of-Experts with Asynchrony.md` (score: 1622.8)

AEP彻底消除all-to-all barrier，将每个GPU从同步等待中解放。核心机制是µ-queue（微队列）+ Defragging Scheduler：

```
初始化:
  每个 GPU 维护 µ-queues: Q[block_id][expert_id] = deque()

Scheduling Loop (每个 GPU 独立执行, 无全局 barrier):
  [Receptor Thread (C++ POSIX)]
    从 Communicator 接收其他 GPU 发来的 tokens
    for each received token:
        Q[layer_id][expert_id].append(token)
  
  [Defragging Scheduler (主线程)]
    while True:
        # LScore: lookahead 加权前方 token 密度
        Score = LScore(b, e) + Q[b][e].size()
        # 选 Score 最高的 (block, expert) pair
        tokens = Q[b][e].drain_all()
        Executor.run(best_layer, tokens)
  
  [Dispatcher Thread (C++ POSIX)]
    Communicator.async_send(output_tokens)  # ZeroMQ + NCCL P2P 异步传输
```

**关键差异 vs Barrier EP**:
- EP: token batch → barrier A2A → expert compute → barrier A2A → 下一层
- AEP: token → µ-queue → [GPU 独立选层执行] → 异步转发 → cold tokens 积累无需 stall

**硬件视角**:
- **SM 占用率提升**: Cold expert 的 tokens 在 µ-queue 中积累到高效 batch size (>128) 才被调度执行 → SM 利用率从 <10% 提升至接近峰值
- **HBM 带宽压力缓解**: 批量执行将 weight loading 开销摊销到更大 batch → 从 memory-bound 转向 compute-bound
- **通信开销消除**: 异步 P2P (ZeroMQ + NCCL P2P) 替代 barrier all-to-all，通信自然与计算重叠

**实现框架**: AMoE (Python+C++/pybind11)，兼容vLLM API

**来源**: Q2.1, `Q2.1_L2_answer.md`, vault: `knowledge_notes/AEP...md` (score: 1116.3)

---

#### 方法: JANUS 解耦 Attention-MoE + AEBS 调度

**笔记证据**: `experiment_notes/系统实验笔记/JANUS_ Disaggregating Attention and Experts for Scalable MoE Inference.md` (score: 1266.9)

JANUS将Attention和MoE层部署到独立GPU子集群，通过AEBS (Activated-Expert-Balanced Scheduling) GPU kernel实现同步无关的expert调度：

```
配置: n_a=1 attention GPU, n_e=6 MoE GPUs

Step 1: Attention 计算 (GPU A)
  SGLang continuous batching → in-flight decode batch
  MLA (Multi-head Latent Attention) + Shared Expert overlap

Step 2: 跨子集群数据传输 (每 MoE 层)
  Phase 1 (Intra-node): 同节点 NVLink NCCL 聚合激活
  Phase 2 (Inter-node): NVSHMEM putmem_signal + GPUDirect RDMA

Step 3: AEBS GPU Kernel 调度 (每 MoE GPU 独立执行)
  # GPU kernel, 无 CPU-GPU 同步, 无跨 GPU 协调, 开销 < 90μs
  单副本 expert → 固定分配到唯一持有实例
  多副本 expert → 贪心分配到当前负载最低的实例
  重写路由结果为物理 replica IDs → dispatch_tokens

Step 4: Expert FFN 计算 + 反向两阶段通信
Step 5: SLO-Aware Scaling (15 min 间隔)
  MoE Controller: Monte Carlo â_max 查找表 + 枚举 (n_a, n_e)
  → 选择 min(n_a+n_e) 满足 TPOT SLO 的配置
```

**核心指标**: vs SGLang per-GPU throughput 最高 4.7×，AEBS < 90μs 开销

**硬件平台**: H100, 900 GB/s NVLink (intra-node) + 400 Gbps InfiniBand (inter-node)

**来源**: Q2.1, `Q2.1_L2_answer.md`, vault: `experiment_notes/JANUS...md` (score: 1266.9)

---

#### 方法: METRO Expert-Level Load Balancing

**笔记证据**: `paper_secs/secs_moe/Efficient MoE Serving in the Memory-Bound Regime...md` (score: 13897.1)

METRO的关键发现：在memory-bound的decode阶段，平衡"token数"反而降低性能——因为它增加了激活expert数，加剧memory压力。METRO转而平衡"激活expert数" per GPU：

```
核心 Insight:
  传统 EP: 平衡 tokens/GPU → 每个 GPU 处理等量 tokens
  问题: 为平衡 tokens，token 被分散到更多 expert replicas
       → 激活 expert 总数增加
       → 每个 GPU 需加载更多 expert weights
       → memory-bound 阶段 HBM 带宽压力剧增
       → 端到端延迟反而上升 (!)

METRO 方案:
  目标函数: min(max_g 激活 expert 数 per GPU g)
  而非:     min(max_g token 数 per GPU g)
  
  实现:
  1. AllGather 全局 top-k 信息 (替代 all-to-all，开销更小)
  2. 联合优化: 最小化 routing quality loss + 约束激活 expert 分布
  3. GPU 并行求解: 利用 GPU parallel processing power
  4. 输出: token routing 决策 → 最小化每 GPU 激活 expert 数
```

**关键适用场景**: single-request decode（自回归逐 token 生成，batch size=1，HBM 带宽瓶颈）

**来源**: Q2.1, `Q2.1_L2_answer.md`, vault: `paper_secs/METRO` (score: 13897.1)

---

### 分类: DiT/Diffusion 调度策略

#### 方法: DiT Denoising Step 迭代调度 + CFG Batch 双流

**笔记证据**: `knowledge_notes/算法知识笔记/MMDiT (Multi-Modal Diffusion Transformer)...md` (score: 262.0), `knowledge_notes/系统知识笔记/Inflight Batching...md` (score: 181.4)

DiT推理与LLM自回归decode根本不同——DiT在固定数量denoising steps（如50 steps）中迭代精炼全量latent tokens。调度围绕step-level编排：

```
Step-level 调度循环 (50 steps):
  for t = T,...,1:
    Sub-step 1: Timestep Embedding + Conditioning Merge
    Sub-step 2: MMDiT Block Forward (N=24-48层)
      a) Self-Attention (文本+图像共享 QKV)
      b) AdaLN Modulation (时间条件注入)
      c) 独立 FFN (模态特定)
    Sub-step 3: Noise Prediction
    Sub-step 4: DDIM/SDE Step → z_{t-1}

Step 间优化:
  - Feature Cache 复用: c_text 在所有 steps 间不变 → 缓存在 GPU memory
  - KV-Cache 不适用: DiT 非自回归模型
  - Conditioning Cache: CFG 正/负 prompt 条件嵌入复用
```

**CFG Batch 双流调度**: Conditional + Unconditional latent 合并为 batch=2 单次 forward pass → GEMM M 维度翻倍（Tensor Core 利用率 ~50%→~80%+），HBM weight loading 减半

**硬件视角**: DiT 是 compute-bound（全量 latent tokens 的 attention/FFN 打满 SM）；大分辨率 (1024²) GEMM 充分填充 Tensor Core/脉动阵列

**来源**: Q2.1, `Q2.1_L2_answer.md`, vault: `knowledge_notes/MMDiT.md` (score: 262.0, 笔记推断)

---

### 分类: 多模态 Serving 调度

#### 方法: VisiPruner 三阶段跨模态调度

**笔记证据**: `knowledge_notes/算法知识笔记/Three-Stage Cross-Modal Interaction in MLLMs...md` (score: 621.3), `knowledge_notes/算法知识笔记/Cross-Modal Information Flow in MLLMs.md` (score: 707.2)

多模态MLLM推理调度协调vision encoder→text encoder→cross-modal fusion→LLM decode四阶段：

```
Phase 1: Vision Encoding (CLIP ViT-L, GPU) — 单体 forward, 不可拆分
Phase 2: Vision Projection (MLP Connector) — 轻量 2-layer, 可流水线重叠
Phase 3: Multimodal Prefill (三阶段跨模态调度, VisiPruner指导)
  
  Sub-phase 3a: Shallow Layers (1-8) — Task Recognition
    视觉+文本 token 独立演化，无有意义跨模态融合
    → 跳过 cross-attention + 视觉 self-attention → 节省 ~25%
  
  Sub-phase 3b: Middle Layers (9-23) — Sparse Cross-Modal Grounding
    跨模态融合由 ~10/576 关键视觉 token 驱动
    → Influence-based Token Selection: 仅保留 10-50 tokens
    → 节省 ~90% cross-attention 计算
  
  Sub-phase 3c: Deep Layers (24-32) — Linguistic Alignment
    视觉信息已集成到文本表示中
    → Vision Exit: 完全丢弃视觉 tokens → 节省 ~20%

Phase 4: Autoregressive Decode — 标准自回归，KV Cache 仅含文本
```

**来源**: Q2.1, `Q2.1_L2_answer.md`, vault: `knowledge_notes/Three-Stage...md` (score: 621.3)

---

### 分类: 计算图分解策略

#### 方法: Infera Tile-Based 分解 + Shepherd Operator

**笔记证据**: `knowledge_notes/编译知识笔记/Tile-Based Zero-Tuning Compilation.md` (score: 1309.2), `knowledge_notes/编译知识笔记/Shepherd Operator.md` (score: 1413.2)

Infera compiler将模型forward pass的每个DNN operator按tile size决策切分为micro operator：

```
编译流程:
  Step 1: ONNX Model → TVM Relay Frontend → Computation Graph
  Step 2: Tile-Tailored Partition
    ├── 大 operator: Conv2D → Conv_tile_0, Conv_tile_1,... (每tile独立micro-kernel)
    └── 小 operator: Add + ReLU + BatchNorm → Shepherd_0 (合并为单一调度单元)
  Step 3: Tile Size 决策 (Top-Down 静态分析, 无需GPU profiling)
    ├── Register file level: 平衡 ILP vs TLP
    ├── Shared memory level: spatial tile = thread_tile × thread_count
    └── Global memory level: spatial tile = block_tile × grid_size
  Step 4: Multi-Version Micro Kernel Generation (pipeline stage 2/3/4 + async copy + warp specialization)
  Step 5: CUDA Binary Static Library → 注册到 Inference Server Model Pool

在线推理 (TEU 三阶段):
  Phase 1 - SelectKernels: DAG中选zero in-degree且最大化asynchronous wavefront
  Phase 2 - FuseKernels: CUDA binary level warp-level horizontal fusion
  Phase 3 - LaunchKernel: HKQ→GDRCopy→DKQ→daemon kernel CDP launch
```

**Shepherd Operator**: 小operator（elementwise add/activation，执行仅数μs）合并→降低per-micro-operator调度overhead（每次CUDA kernel launch约5-10μs）

**来源**: Q2.3/Q2.4, `Q2.3_L2_answer.md`/`Q2.4_L2_answer.md`, vault: `knowledge_notes/编译知识笔记` (score: 1309.2/1413.2)

---

#### 方法: vLLM Fused MoE + FlashMoE Megakernel

**笔记证据**: `knowledge_notes/kernel知识笔记/Fused MoE（融合 MoE Kernel）.md` (score: 400.9), `knowledge_notes/kernel知识笔记/Megakernel _ Fused Persistent Kernel for Distributed MoE.md` (score: 455.7)

**vLLM Fused MoE** (Mixtral-8x7B, H100):
```
Step 1: Router → Top-K
Step 2: Token Sorting (moe_align_block_size, padding to BLOCK_SIZE=64)
Step 3: Fused MoE Kernel (单次 launch, Grid = E × ceil(tokens/BLOCK_M))
  For expert_id in 0..7:
    For token_block in expert's tokens:
      gate = silu(a_block @ w1_gate)  # FC1 gate+up 融合
      up   = a_block @ w1_up
      hidden = gate * up
      out = hidden @ w2  # FC2 down
      out = out * topk_weights  # routing weight 内联
      atomic_add(C[sorted_token_ids], out)
```
**效果**: 6-8次kernel launch→1-2次，吞吐提升15-20% (大batch, H100)

**FlashMoE Megakernel** (分布式MoE, 多GPU):
```
Grid: N blocks/GPU
OS Block (block N-1): Warp 0 Scheduler (work-conserving, doorbell monotonic counter sweep) + Warp 1-3 Subscriber (poll remote flags→enqueue→doorbell Scheduler)
Processor Block: CUTLASS device-side GEMM (Fused GEMM+GELU) + NVSHMEM put to remote combine buffer
```
**效果**: 1 kernel launch vs Megatron-LM+DeepEP的432次；SM occupancy: tile=(128,64), block_size=128, registers=255/thread, max 2 blocks/SM, 0 spill

**来源**: Q2.3, `Q2.3_L2_answer.md`, vault: `knowledge_notes/kernel知识笔记` (score: 400.9/455.7)

---

#### 方法: MPK tGraph Mega-Kernel

**笔记证据**: `idea_notes/Mirage Persistent Kernel_ A Compiler and Runtime for Mega-Kernelizing Tensor Programs.md` (score: 59.2)

将分解后的细粒度task重新融合为单个mega-kernel，以SM级tGraph替代kernel级computation graph：

```
离线编译:
  Step 1: Operator Decomposition (MatMul→132 tasks, Attention→动态task数)
  Step 2: tGraph 构建 (每个task 352 bytes metadata; events连续数组存GPU device memory)
  Step 3: Dependency Analysis (task-pair level — 仅当I/O区域重叠才插入event)

在线执行 (单次 kernel launch):
  SM 物理分区: 128 Workers + 4 Scheduler-SMs (16 warp-schedulers)
  
  T0: Start event → Scheduler分发Q/K/V projection tasks (AOT pre-assigned)
  T1: Worker SM_i: Q_proj (TMA preload W_Q→Tensor Core MMA)
      同时TMA预取K_proj weight tile (cross-task pipelining)
  T2: Q/K/V完成→Scheduler分发attention tasks (JIT, dynamic load balance)
  T3: All attention done→barrier→MLP tasks (AOT)
  T4: TMA/Tensor Cores/CUDA Cores三种硬件持续饱和
```

**性能**: MPK 12.5ms vs SGLang/vLLM 14.5ms，仅比理论下限10ms高~25%；E2E 1.0-1.7×

**来源**: Q2.3, `Q2.3_L2_answer.md`, vault: `idea_notes/MPK` (score: 59.2)

---

### 分类: Dispatcher 设计方法

#### 方法: ACS Out-of-Order Kernel Dispatch

**笔记证据**: `paper_secs/secs_multimodal_kernel/ACS Concurrent Kernel Execution on Irregular, Input-Dependent Computational Graphs/D.-Key-Observations.md` (score: 2441.0)

模仿CPU乱序执行——在GPU Command Processor中维护调度窗口，运行时检测kernel间数据依赖：

```
=== ACS-SW (纯软件实现) ===
SchedulingWindow: fixed[N] slots {
    kernel_id, state ∈ {READY, PENDING, EXECUTING},
    upstream_list: [kernel_id],
    read_segments/write_segments: [(start_addr, size)]
}

Thread_WindowModule: 管理窗口 + 依赖检测
  loop:
    kernel = FIFO.pop()
    for each sw_kernel in SW:
        if overlap(kernel.write_segments, sw_kernel.read_segments) OR
           overlap(kernel.write_segments, sw_kernel.write_segments):
            kernel.upstream_list.append(sw_kernel.id)
    on_kernel_complete: 遍历所有slot移除对应upstream ID → 空→标记READY

Thread_SchedulerModule (N_threads, 每线程绑定1 CUDA stream):
  ready_kernel = SW.poll_ready() → launch(ready_kernel, my_stream)

=== ACS-HW (硬件加速) ===
硬件调度窗口 (GPU Command Processor内):
  N=32 slots × [(N-1)×8B upstream IDs + 2bit state] SRAM → ~1KB
  插入: CPU发kernel+upstream list→HW窗口
  更新: kernel完成→HW遍历所有slot→某slot upstream为空→标记READY→发射
  延迟: N=64时 ~64 cycles (~50-100ns) overhead (!)
```

**适用场景**: ACS特别适合input-dependent irregular DAG（动态路由MoE、instance-aware DNN），因为运行时才知道kernel依赖

**来源**: Q2.4, `Q2.4_L2_answer.md`, vault: `paper_secs/ACS` (score: 2441.0)

---

#### 方法: MuxWise SLO-Aware Dispatcher + Bullet Dynamic SM Repartitioning

**笔记证据**: `paper_secs/secs_2026/1-Towards High-Goodput LLM Serving with Prefill-decode Multiplexing/3.4-SLO-aware-Dispatcher.md` (score: 5286.6), `knowledge_notes/系统知识笔记/Dynamic SM Repartitioning（动态SM重分区）.md` (score: 46.3)

```
Function SLO_AWARE_DISPATCH(prefill_queue, decode_batch, SLO_TBT):
    // Step 1: Decode SLO优先
    T_d = ContentionTolerantEstimator.estimate_decode(...)
    
    // Step 2: SM分区 (GreenContext CUDA 12.4+)
    SM_decode = ceil((T_d / SLO_TBT) * total_SMs)
    SM_prefill = total_SMs - SM_decode
    libsmctrl_set_stream_mask(stream_decode, build_mask(0, SM_decode))
    libsmctrl_set_stream_mask(stream_prefill, build_mask(SM_decode, SM_prefill))
    // 重分区耗时: ~4.1μs on A100
    
    // Step 3: layer-wise prefill multiplexing
    N_PL = ceil((T_d * N_layers) / T_P)  // 此次发射的prefill层数
    launch_async(decode_kernel, stream_decode, all_decode_tokens)
    launch_async(prefill_layers[0:N_PL], stream_prefill, prefill_tokens)
    
    // Step 4: Prefill完成→合并到decode batch
    // Step 5 (可选): Prefill Preemption (不递归抢占)
```

**Bullet Dynamic SM Repartitioning**: 每prefill layer group/decode step后触发——读shared metadata buffer (p90_tpot, queue_depth, current_sm_split)→SLO感知重分区（16 SM/步粒度）

**硬件依赖**: NVIDIA SM mask机制（通过libsmctrl/GreenContext API操作硬件寄存器），非开源GPU无法直接移植

**来源**: Q2.4, `Q2.4_L2_answer.md`, vault: `paper_secs/SLO-aware-Dispatcher` (score: 5286.6)

---

### 分类: Serving 框架架构与实现

#### 方法: vLLM Scheduler 详细实现

**笔记证据**: `knowledge_notes/系统知识笔记/vLLM.md` (score: 472.9), `knowledge_notes/kernel知识笔记/Framework Scheduling Overhead.md` (score: 5079.9)

vLLM Scheduler实现在`vllm/core/scheduler.py`（Python）配合C++/CUDA backend：

```
T=0ms [Scheduler._schedule() 被调用]
  Step 1: 从waiting_queue取出新请求，检查gpu_free_blocks
  Step 2: Preemption检查（内存不足→RECOMPUTE或SWAP策略）
  Step 3: Block Table管理（per-request逻辑→物理映射表, BLOCK_SIZE=16）
  Step 4: Prefix Caching匹配（prompt blocks content hash→命中则复用+跳过prefill）
  Step 5: Continuous Batching（组装prefill+decode混合running_batch）
  Step 6: Model Runner执行
    - 纯decode batch → CUDA Graph快速路径（单次cudaGraphLaunch）
    - 混合batch → PyTorch eager路径（MoE内部FusedMoE kernel）
  Step 7: 完成检查+资源回收（free blocks, update ref_counts）
```

**Framework Scheduling Overhead 6步分解** (per-operator):
1. Operator Dispatch: ~20-30μs
2. Output Shape Inference: ~10-20μs
3. Kernel Selection (cuDNN auto-tune): ~5-50μs
4. Argument Preparation: ~5-10μs
5. Kernel Launch (cudaLaunchKernel→GPU queue): ~3-5μs
6. Memory Allocation: amortized
→ **Total CPU overhead per op: ~50-115μs**，GPU idle ratio ~90%+
→ **CUDA Graph降至<100μs total per iteration**，GPU idle ratio <5%

**硬件特定优化**:
- FusedMoE (Triton JIT): Router→Dispatch→GroupedGEMM→Combine融合，15-20%吞吐提升
- GPU-side Block Table (BrownoutServe): 消除CPU↔GPU cudaMemcpy传输（~5-10μs/iteration）
- FlashInfer BSR: gather/scatter分散memory→contiguous SMEM→Tensor Core MMA

**来源**: Q2.5, `Q2.5_L2_answer.md`, vault: `knowledge_notes/vLLM.md` (score: 472.9), `Framework Scheduling Overhead` (score: 5079.9)

---

#### 方法: TensorRT-LLM C++ Runtime + Graph Optimization Pipeline

**笔记证据**: `knowledge_notes/编译知识笔记/FasterTransformer.md` (score: 188.2), `knowledge_notes/kernel知识笔记/All-to-All Communication in MoE.md` (score: 16.5)

TensorRT-LLM采用**离线图优化+运行时C++执行**架构：

**离线编译 Phase 1**:
- 模型导入(HuggingFace→GraphIR)→注入并行策略(TP/EP/PP)
- Graph Optimization Pass Pipeline:
  - Operator Fusion (GEMM+GELU/SwiGLU→fused; LayerNorm+QKV→fused; MoE全流程→FusedMoE)
  - Memory Planning (liveness分析→最小化peak memory预分配)
  - Kernel Auto-Tuning (cuBLASLt/CUTLASS在线profile选最优; FA2/FA3 attention; NVLink one-sided alltoall)
  - FP8 Precision Pass (H100 SM90: 量化校准FP16→FP8 E4M3 + Transformer Engine动态scaling)
- TRT Engine序列化→磁盘跨session复用

**运行时 Phase 2**:
- Execution Context初始化: 预分配buffers + CUDA Graph预录制 + 多CUDA stream
- In-Flight Batching调度循环: Context Phase(Prefill) + Generation Phase(Decode)同批混合
- MoE层: Router→CUB radix sort token dispatch→CUTLASS Grouped GEMM (FP8/FP16/INT4)→Token Combine→NVLink One-Sided AlltoAll

**H100深度整合**:
- FP8 Tensor Core: 1979 TFLOPS (vs FP16 989 TFLOPS)
- TMA warp-specialization: Producer TMA异步预取↔Consumer WGMMA计算完全重叠
- NVLink one-sided alltoall: 替代AllGather+ReduceScatter，更少同步点

**跨平台限制**: 深度绑定NVIDIA CUDA生态，NPU/TPU移植最困难

**来源**: Q2.2/Q2.5, `Q2.2_L2_answer.md`/`Q2.5_L2_answer.md`, vault: `FasterTransformer` (score: 188.2)

---

### 分类: 跨硬件平台对比

#### 三大硬件平台 Serving 适配关键差异

```
┌──────────────────┬──────────────────┬──────────────────┬──────────────────┐
│   适配维度        │  NVIDIA GPU      │  华为 Ascend NPU  │  Google TPU      │
│                  │  (CUDA/H100)     │  (CANN/910B)     │  (XLA/v5p)       │
├──────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ 编程模型          │ CUDA C++/Python  │ CANN C++/Python  │ XLA HLO          │
│                  │ 灵活kernel开发   │ ACL算子库+TS调度  │ (编译器驱动)     │
├──────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ KV Cache管理      │ PagedAttention   │ PagedAttention   │ 编译器静态管理   │
│                  │ (动态block分配)  │ (需CANN适配)     │ (运行时不可变)   │
├──────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ 片上存储         │ Shared Memory    │ L1 Buffer (分离) │ Vector Memory    │
│                  │ (256KB/SM,可编程)│ + L0 Buffer      │ (编译器管理)     │
├──────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ 图调度机制        │ CUDA Graph       │ Ascend Graph     │ XLA编译期调度    │
│                  │ (Record→Replay)  │ (Task Descriptor │ (静态图)         │
│                  │                  │  Chain→硬件TS)   │                  │
├──────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ Kernel Launch    │ ~3-10μs/kernel   │ 笔记未明确说明   │ 0 (编译期固定)   │
│ Overhead         │ (CUDA Graph解决) │                  │                  │
├──────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ 集合通信          │ NCCL (NVLink     │ HCCL (HCCS       │ ICI (Inter-Chip  │
│                  │  900 GB/s)       │  60 GB/s)        │  Interconnect)   │
├──────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ Dispatch可控性   │ 低 (GigaThread   │ 中 (TS可配置     │ 无 (编译器决定)  │
│                  │  Engine闭源)     │  queue depth)    │                  │
├──────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ 框架移植难度      │ 原生支持         │ 中等 (CANN适配)  │ 高 (XLA路径      │
│                  │ (基准平台)       │                  │  完全重编译)     │
└──────────────────┴──────────────────┴──────────────────┴──────────────────┘
```

**注解**:
- Ascend NPU Task Scheduler (TS)直接解析Task Descriptor Chain→消除了GPU GigaThread Engine的软件层block调度延迟
- HCCS片间60 GB/s vs NVLink 900 GB/s→跨节点通信成为Ascend集群主要瓶颈（MixServe的Fused AR-A2A算法专为解决此问题）
- TPU的XLA编译期固定调度→运行时零scheduling overhead，但完全丧失动态batch灵活性。MoE动态expert routing通过编译器静态unrolling处理

**来源**: Q2.5, `Q2.5_L2_answer.md`, vault: `knowledge_notes/硬件知识笔记/Ascend NPU Architecture` (score: 24.0), `EPD-Serve...ASCEND` (score: 17442.3)

---

## 方法间关系

### 替代关系

- **EP Barrier ←→ AEP/AMoE**: AEP以µ-queue + Defragging Scheduler + async P2P完全替代barrier A2A→A2A loop，GPU stall从70%降至接近零。AEP是EP的升级替代而非互补——两者解决同一问题（MoE expert通信瓶颈）但方式截然不同（同步barrier vs 异步自调度）
- **EP Barrier ←→ JANUS AEBS**: JANUS的解耦架构将expert调度从all-to-all通信改为GPU kernel内同步无关的负载均衡，同时将attention和MoE部署到不同GPU子集群。JANUS is 替代EP barrier的另一种路径（解耦而非异步）
- **METRO激活expert平衡 ←→ 传统token平衡**: METRO证伪了「平衡tokens→更好性能」的传统认知——在memory-bound decode阶段，平衡激活expert数优于平衡token数。两者是同一问题（负载均衡目标函数）的互斥选择
- **vLLM block-level prefix caching ←→ SGLang RadixAttention token-level prefix sharing**: 共享粒度不同（16 tokens/block vs 1 token），RadixAttention共享效率更高但tree维护开销更大。在单请求场景下差异缩小（prefix少），高并发下SGLang token-level更优
- **CUDA Graph (vLLM/SGLang) ←→ TensorRT-LLM AoT Graph Optimization**: CUDA Graph是运行时预录制+replay（固定shape），TensorRT-LLM是离线全图编译（包含fusion+kernel selection+memory planning），后者优化更深但灵活性更低
- **GPU GigaThread Engine ←→ Ascend NPU Hardware TS**: GPU闭源不可配置的left-over scheduling vs NPU可配置优先级的task descriptor chain。NPU TS可配置queue depth是Dispatcher设计的重要自由度
- **Infera tile-based分解 ←→ MPK tGraph mega-kernel化**: 表面相反——Infera分解大op为micro-op tiles再warp-level水平融合，MPK将分解后的task重新融合为SM级tGraph单kernel。实则殊途同归：分解是手段，消除kernel barrier后的高效执行单元是共同目标
- **CFG串行双流 ←→ CFG Batch双流**: Batch双流将2× latency降至1×+merge overhead——单GPU下替代串行双流

### 互补关系

- **AEP/AMoE + Fused MoE**: AEP解决通信调度（消除barrier），Fused MoE解决计算融合（减少kernel launch）——两者作用于MoE layer的不同阶段，可组合。AMoE论文提及"兼容vLLM API"暗示了这种组合的可行性
- **vLLM PagedAttention + Mooncake Store**: PagedAttention管理单GPU KV-cache分页，Mooncake Store管理跨节点KV-cache分布式池化——两者分别解决单节点和多节点的KV-cache管理
- **MuxWise SLO-Aware Dispatcher + Bullet Dynamic SM Repartitioning**: MuxWise提供SLO感知的decode/prefill调度决策，Bullet提供微秒级SM重分区机制——前者是决策层，后者是执行层
- **EEVEE Module Multiplexing + EEVEE Modal Cache**: SM MPS分区负责模块级空间复用，Modal Cache负责跨请求视觉特征复用——两者在同一系统（EEVEE）内互补
- **VisiPruner三阶段调度 + EEVEE Modal Cache**: VisiPruner提供分层token pruning策略（何时丢弃视觉tokens），Modal Cache提供跨请求视觉token复用——前者优化单请求内，后者优化跨请求
- **EPD-Serve阶段解耦 + Ascend NPU AI Core/Vector算子互补**: EPD-Serve的E/P/D分离提供粗粒度拓扑选择，AI Core/Vector互补提供细粒度物理共置算子空间复用——两层互补
- **MixFusion patch级分解 + MPK mega-kernel化**: MixFusion将不同分辨率图像分解为uniform patches，MPK将task图形融合为单kernel——理论上可组合：patch级分解+per-patch SM级tGraph mega-kernel
- **Infera Runtime SelectKernels + FuseKernels + LaunchKernel**: 三阶段pipeline内互补——SelectKernels选最优kernel版本，FuseKernels做warp-level融合减少barrier，LaunchKernel通过CDP daemon消除host→device launch overhead
- **QoServe Hybrid Prioritization + MuxWise SLO-Aware Dispatcher**: QoServe提供多请求优先级调度框架（EDF+SRPF），MuxWise提供硬件级SM分区执行——分别作用于调度策略层和硬件资源分配层
- **TetriServe deadline-aware scheduling + CFG Batch双流**: TetriServe的round-based调度决定SP并行度，CFG Batch双流在单step内优化条件/无条件流——互补的粗细粒度调度

### 依赖关系

- **AEP/AMoE → NCCL P2P + ZeroMQ**: AEP的异步通信依赖NCCL P2P（GPU-GPU数据传输）和ZeroMQ（CPU metadata传递）作为底层通信基础设施
- **JANUS AEBS → NVSHMEM putmem_signal + GPUDirect RDMA**: AEBS GPU kernel的无CPU同步调度依赖NVSHMEM one-sided通信语义和GPUDirect RDMA的硬件能力
- **vLLM Fused MoE → Triton JIT compiler**: Fused MoE kernel依赖Triton的block-level parallelism和grouped GEMM支持
- **FlashMoE Megakernel → CUTLASS device-side API**: Megakernel内的expert GEMM依赖CUTLASS device-side API在persistent kernel内直接执行（绕过host launch）
- **MPK tGraph → Mirage superoptimizer**: tGraph中的每个task依赖Mirage自动生成优化CUDA实现→无此编译器则无法构建tGraph
- **MuxWise dispatcher → GreenContext (CUDA 12.4+) + libsmctrl**: SM分区依赖GreenContext API的进程内SM分区和libsmctrl的SM mask硬件寄存器操作
- **Infera Runtime → GDRCopy + CUDA Dynamic Parallelism**: daemon kernel launch依赖CDP的GPU内fire-and-forget能力，HKQ→DKQ快速拷贝依赖GDRCopy bypass DMA
- **ACS-HW → GPU Command Processor 修改**: ACS硬件加速方案需要修改GPU Command Processor添加调度窗口SRAM→在商业GPU上不可行（需NVIDIA合作）
- **EPD-Serve → Ascend NPU AI Core/Vector 独立指令队列**: 算子互补空间复用依赖Ascend NPU的异构计算单元具有独立指令队列
- **Mooncake Store → RDMA/GPUDirect RDMA/CXL**: Transfer Engine的高带宽低延迟传输依赖RDMA（190 GB/s 8×400Gbps RoCE）和CXL的硬件能力
- **SADDLE CI-Aware → HBM-PIM硬件**: per-iteration PIM/GPU动态映射依赖HBM-PIM PE array的存在→传统GPU-only系统无法受益

---

## 跨模型硬件体系结构对比

### 调度粒度与硬件并行能力匹配

| 模型类型 | 调度粒度 | GPU SM 匹配度 | NPU 脉动阵列匹配度 | HBM 带宽压力 | 最优硬件平台特征 |
|----------|----------|---------------|---------------------|-------------|------------------|
| **MoE** | Expert-level (粗粒度) / Token-level (AEP 细粒度) | 中-高 (GEMM batch size 依赖) | 中 (A2A 通信开销大) | 高 (expert weights 反复加载) | 高 NVLink 带宽 + 大 HBM 容量 |
| **DiT** | Step-level (粗粒度) | 高 (全量 latent tokens 填充 SM) | 高 (大 GEMM 打满脉动阵列) | 中 (compute-bound) | 高 Tensor Core 密度 |
| **多模态** | Stage-level (粗粒度，ViT vs LLM 交替) | 中 (阶段切换有 idle) | 中-低 (异构单元负载不均) | 高 (visual tokens KV Cache) | 大 HBM + 多计算单元异构调度 |
| **Video** | Frame-level (细粒度，逐帧) | 低 (单帧编码 SM 利用率低) | 低 (逐帧 GEMM 太小) | 极高 (长视频 KV Cache 膨胀) | 大内存容量 + 高效 eviction/swap |

### 调度策略对硬件利用率的影响汇总

| 调度策略 | SM 占用率影响 | HBM 带宽影响 | 通信开销影响 |
|----------|--------------|-------------|-------------|
| EP Barrier (Baseline) | ↓↓↓ Cold expert GPU idle | ↓↓↓ Small batch weight loading dominant | ↑↑↑ Barrier A2A 占 59% |
| AEP 异步调度 | ↑↑↑ Cold tokens 积累后批量执行 | ↑↑↑ 批量执行摊销 weight loading | ↓↓↓ Async P2P 替代 barrier |
| AEBS GPU Kernel | ↑↑ Expert load 均衡 → SM 均匀利用 | → 解耦架构 KV cache 常驻 | ↓↓ NVSHMEM one-sided |
| METRO Expert-Balanced | ↑↑ 减少激活 expert 数 | ↑↑ Memory→Compute 转变 | ↓ AllGather vs All-to-All |
| CFG Batch 双流 | ↑↑ GEMM M 维度翻倍 | ↑ Weight 加载减半 | N/A (单 GPU) |
| Vision Exit (多模态) | ↑ Deep layers 计算量降低 | ↑ 视觉 KV Cache 释放 | N/A |
| Streaming KV Cache | → 逐帧编码固有低利用率 | ↓↓ 避免重编码历史帧 | N/A |
| MuxWise SM分区 | ↑↑ Prefill+Decode并发执行 | → Contention-tolerant modeling | ↓ CUDA Event async sync |
| Fused MoE kernel | ↑↑ 减少 kernel launch idle | ↑↑ 中间 tensor 不写回 HBM | N/A (单 GPU) |

---

## 本层不确定性

1. **NPU Dispatcher 具体机制**: 笔记中关于NPU (昇腾/寒武纪)的task descriptor queue格式、command processor架构、barrier instruction语义等信息缺失。对NPU上微算子dispatcher的设计只能从TZ-LLM的RK3588 NPU co-driver间接推断。搜索"NPU task descriptor queue"返回0结果

2. **TPU Device-Side Scheduling**: 笔记未包含TPU v5e/v5p运行时dispatcher的详细设计。XLA编译期调度与运行时调度的边界在笔记中未明确。TPU上MoE/DiT/多模态/Video模型的算子级/模块级分解benchmark数据在笔记中完全缺失

3. **DiT/Video 模型 Serving 实验证据缺失**: DiT模型的denoising step调度和Video模型的帧间调度实验在vault中缺乏覆盖。Video模型的spatial encoder + temporal encoder独立调度分解的具体方法和实验数据笔记未明确说明

4. **Groq LPU / Cerebras CS-3 / AMD MI300X 的调度策略**: 笔记中未找到这些加速器平台的Serving调度实验和分解策略证据

5. **跨框架对比标准不统一**: vLLM、SGLang、TensorRT-LLM的实验使用不同模型、不同GPU、不同batch配置和不同指标，直接对比存在困难

6. **能效指标量化不足**: 仅HNLPU笔记提供了硬件级功耗和面积数据，GPU平台的tokens/J实验数据在现有笔记中基本缺失

7. **调度开销绝对值缺失**: 多数笔记报告的调度开销为定性描述（如"overhead <1ms"），缺乏μs级精度的CPU-side scheduling overhead breakdown（DAG拓扑排序耗时、就绪队列管理耗时、资源查询耗时）

8. **单请求场景的特殊性**: 大部分MoE serving论文（JANUS、AMoE）的实验以multi-request continuous batching为默认场景，其调度设计的单请求适用性需进一步验证

9. **Video专用benchmark**: 笔记未找到Video-MME或类似Video推理benchmark上与计算图分解/调度策略相关的实验数据

10. **Command Processor 修改可行性**: ACS-HW需要修改GPU硬件command processor——这在商业GPU上不可行。NPU/TPU是否有等价的command processor可修改性，笔记未明确说明

---

[HORIZON_SUMMARY_DONE] L2
