# L3: 编译框架 — 水平分类总结

## 问题覆盖概览

| Q-ID | 覆盖方法数 | 关键方法 |
|------|-----------|----------|
| Q3.1 (IR表示) | 10 | XLA HLO all-to-all+GSPMD、MLIR scf.forall+async.execute、Cypress Event IR、Difflow DFG/dGraph、Triton IR program_id、Hidet IR、Tilus SIMB、CUTLASS模板、PIMphony MLIR+IREE、Infera Multi-Version |
| Q3.2 (算子融合) | 7 | FlashFuser DSM-based GEMM链融合、Group GEMM (MoE水平融合)、Mirage Persistent Kernel mega-kernel、Triton Autotuner、Multi-Stream (Nimble)、CANN GE/IRFusion |
| Q3.3 (图优化与自动调优) | 12 | NetMoE ILP设备放置、MegaScale-MoE Holistic Scheduling+SAR、HuntKTm Stream Scheduler、ACS OoO调度、HyTiS auto-tuning、Infera zero-tuning、FlowMoE BO调优、MoE-GEN DAG+DP、ELK O(KN)调度、Brainstorm动态水平融合、MoEBlaze Selective SiLU Checkpointing、KernelEvolve进化搜索 |
| Q3.4 (编译框架实现) | 7 | XLA/GSPMD (6阶段SPMD编译)、TVM (Relay→TensorIR→Relax+AutoScheduler)、Triton (TTIR→TTGIR→PTX)、MLIR/IREE (多dialect+polyhedral+HAL)、PyTorch 2.0 (Dynamo→Inductor→CUDA Graph)、CUTLASS 3.x (CuTe DSL+warp spec)、CANN (GE+IRFusion+AscendC) |
| Q3.5 (实验环境) | 8 | MoE-CAP (全GPU谱系)、FlashAttention-2/FlashInfer (kernel benchmark)、PAT (vLLM prefix-aware)、DS-MoE (sparse inference)、EPD-Serve (Ascend NPU多模态)、ElasticMoE (CloudMatrix384)、SambaNova SN40L (dataflow)、KernelEvolve (跨平台) |
| Q3.6 (Codegen) | 10 | CUDA Graph、ACS OoO调度、Warp Specialization+TMA、Software Pipeline (cp.async/TMA)、GSPMD+XLA TPU、Groq确定性编译、Cerebras Dataflow、TileLang T.Pipelined、MLIR async dialect、CANN NPU ILP编排 |

---

## 按实验环境分类

| 分类 | 方法 | 具体方法描述 | 硬件平台 | Benchmark | 实现框架 | 来源 |
|------|------|-------------|----------|-----------|----------|------|
| **GPU/NVIDIA** | FlashFuser | 基于H100 DSM (Distributed Shared Memory) 的GEMM链融合编译器。五级存储层次 (Reg→SMEM→DSM→L2→HBM) 突破单SM SMEM 227KB限制，cluster内有效片上空间 ~3.6MB。dsm_comm 四种原语 (all_exchange/shuffle/reduce_scatter/inter_cluster_reduce) + Dataflow Analyzer (贪心逐级放置中间张量 + Loop Schedule搜索 MNLK vs MLNK) + Fusion Search Engine (5条剪枝规则将2.75×10^13搜索空间降至1.15×10^6)。Tile Selection分为Cluster-level (跨cluster工作分布+是否需要跨cluster通信) 和Block-level (单block内存占用+Reg/SMEM使用) 两级 | H100 (80GB HBM3) | GEMM Chains, Convolutional Chains, Gated FFNs (SwiGLU) | CUTLASS 3.x + CUDA 12.4 | Q3.2, paper_secs FlashFuser (16491), dsm_comm 知识笔记 (29.5) |
| **GPU/NVIDIA** | Mirage Persistent Kernel (MPK) | Mega-kernel极值融合：将整个模型的计算+通信融合为单个persistent kernel。tGraph编译流程：Operator Decomposition (SM级任务切分)→Dependency Analysis (精细事件依赖，AllReduce task_i仅依赖MatMul task_i)→Event Fusion (按predecessor-set/successor-set合并)→tGraph Normalization (每task最多1依赖+1触发)→tGraph Linearization (BFS保证同事件task连续)→GPU Device Memory紧凑存储。Worker-Scheduler运行时：SM分区为workers (FIFO task queue) 和schedulers (维护task依赖，依赖满足时分配)，全异步、零kernel launch overhead | A100/H100/B200 | LLM serving (LLaMA-like), 多GPU推理 (含AllReduce) | CUDA + MPK runtime | Q3.2, paper_secs MPK (8593) |
| **GPU/NVIDIA** | Group GEMM (MoE水平融合) | 多expert FFN合并为单kernel launch：Gate+Up projection 通过GroupedMatMul将所有active expert的权重concat为单次GEMM，tile sizes通过offline profiling为不同activation pattern选择。Sentinel filtering在dispatch阶段过滤被跳过的expert，仅需warp-level元素操作 (masked comparison, overhead <1%)。Kernel launch从O(E)降至O(1) | A100/H100 (CUDA) | Mixtral 8x7B, DeepSeek-V2 MoE | Custom CUDA kernel + offline profiled tile selection | Q3.2, Group GEMM 知识笔记 (37.6) |
| **GPU/NVIDIA** | Triton Autotuner | Block-level编程模型+Triton IR (TTIR→TTGIR→LLVM IR→PTX)。@triton.autotune装饰器枚举配置空间 (tile sizes, num_warps, num_stages, 通常20-50 candidates)→在目标GPU上benchmark→选择最优缓存。program_id沿grid轴实现独立CTA并行。Autotuner key=['M','N','K']：运行时参数变化时重新benchmark。TTGIR lowering自动处理memory coalescing+shared memory padding (避免bank conflict)+sync insertion (cp.async.commit_group/wait_group) | NVIDIA (sm_70+), AMD CDNA3/4 | Attention, MoE FFN, GEMM intensive workloads | OpenAI Triton MIT License | Q3.1 (Triton IR 1762/213), Q3.2, Q3.4 (Triton笔记) |
| **GPU/NVIDIA** | HuntKTm Stream Scheduler | 编译时auto multi-stream：DFG构建→依赖分析→拓扑分层 (同level无依赖kernel可并发)→stream分配 (load-balanced, 选择当前队列最短的stream)→冗余同步消除 (transitively covered sync剪除)→内存生命周期管理 (non-overlapping live range→alias共享物理内存)。两层并发：kernel-level (CUDA stream, max 32 streams) + task-level (MPS) | A100 (单/多GPU) | DNN training/inference | LLVM pass + CUDA | Q3.3, paper_secs HuntKTm (4288.9) |
| **GPU/NVIDIA** | ACS OoO Kernel Scheduling | 编译时标注RW-segments (每个kernel标注read/write memory ranges)+运行时sliding scheduling window (N=32)。Window Module (CPU线程) 做O(segments²)地址重叠检测→upstream依赖分析→on_kernel_complete更新。Scheduler Module (多CPU线程，各绑1 CUDA stream) 发射READY kernel。ACS-HW增强GPU Command Processor：1KB调度窗口SRAM集成，kernel完成后硬件自动更新upstream list | NVIDIA GPU (RTX 3060模拟, 真实GPU) | 不规则计算图 (DeepRL物理仿真, 动态DNN) | ACS CUDA runtime (SW) / GPU CP修改 (HW) | Q3.3, Q3.6, paper_secs ACS (2386.8/2060.4), OoO知识笔记 (2407.6) |
| **GPU/NVIDIA** | CUDA Graph | 编译时DAG静态化：Stream Capture (运行时捕捉kernel序列→编译时固化)或Explicit API (显式构建DAG)。Instance化阶段GPU驱动做全局register allocation+SMEM预分配+依赖边→硬件同步原语编译。Replay仅一次CPU→GPU MMIO写入 (~3μs total vs ~5-20μs/kernel传统launch)。致命缺陷：动态图每次需重建graph (ACS实验Brax仿真中构建占47%总时间) | NVIDIA GPU (A100/H100) | DiT static denoising loop, Video固定帧数, MoE expert FFN链 | CUDA Driver API | Q3.4, Q3.6, CUDA Graph知识笔记 (5770.5) |
| **GPU/NVIDIA** | Warp Specialization + TMA (Hopper) | Hopper GEMM kernel编译为DMA warp (tid 128-159, 单线程TMA_load非阻塞) + Compute warpgroup (tid 0-127, 4 warp×32线程=128线程协作发射wgmma指令)。PIPE=2/3深度multi-buffered SMEM (通过mbarrier硬件同步)。TMA硬件完全独立于CUDA Core (无寄存器占用, 支持多播)。编译时pipeline depth = min(hardware_max, ceil(TMA_latency/GEMM_latency), smem_size/tile_size) | H100 SM90 | GEMM, Attention | CUTLASS CuTe DSL → nvcc → PTX → SASS | Q3.6, Task-Based Tensor paper (1736.1), TMA知识笔记 (12311.5) |
| **GPU/NVIDIA** | Software Pipeline (cp.async) | Ampere A100异步copy：cp.async.commit_group→cp.async发射(非阻塞)→cp.async.wait_group(N-1)→mma计算（计算与数据搬运重叠）。TileLang T.Pipelined编译器自动选择Ampere cp.async或Hopper TMA路径 | A100 (Ampere), H100 (Hopper) | GEMM kernel | PTX (cp.async), TileLang compiler | Q3.6, async-copy知识笔记 (10327.9), Software Pipeline知识笔记 (795.9) |
| **GPU/NVIDIA** | HyTiS Auto-tuning | 离线profiling+自适应搜索空间：对每个candidate micro-kernel tile K_i=(bM,bN,bK,layout)实测latency+SMEM/REG usage→构建S_TO (Throughput-Oriented, diff(T,max)<l1) + S_LO (Latency-Oriented, diff(t_wave,min)<l2)候选集。运行时两级tile scheduling：TO全波+LO尾波 (partial_tiles≤N_SM时有效)。H100平均搜索空间14配置 (max 66)，搜索空间-86% vs Inductor-Triton固定19配置。Profiling成本：H100 ~19min, A100 ~36min (per device, 一次完成) | H100/A100 | GEMM各种problem shape | Python + Triton/Inductor | Q3.3, HyTiS知识笔记 (690.5, 42.8) |
| **GPU/NVIDIA** | FlashAttention-2 | Split-Q warp调度：对每个(batch,head,row_block)组合launch 1 thread block (2048 blocks >> 108 SMs→~100% occupancy)。Warp内split-Q (每warp持有Q的32 rows, K/V由所有warp共享)→无warp间通信。Online softmax (MUFU.EX2指令)+一次性rescale→最小化non-matmul FLOPs (A100 FP32 CUDA Core仅19.5 TFLOPS vs Tensor Core 312 TFLOPS→比16:1) | A100 SXM4 80GB (108 SM, 192KB L1/SMEM), H100 | seq_len 512-16K, head_dim 64/128, causal/non-causal | CUTLASS 3.x CUDA kernel (nvcc编译) | Q3.5, FA2 实验笔记 (651.6) |
| **GPU/NVIDIA** | PAT Prefix-Aware Attention | Prefix tree构建→CTA partition→pack-forward-merge pipeline。Pack scheduler在每次decode step前将共享prefix信息编译为静态CTA partition (类似CUDA Graph的kernel launch优化)。Lazy update复用上次调度结果 (增量编译策略)。Scheduler与pre-attention tasks异步重叠执行 | A100-SXM4-80GB/H100 单卡, 4×A100 (TP=2 PP=2 for 72B) | Llama-3-8B, Qwen3-8B, Qwen3-30B-A3B (MoE), Qwen2.5-72B | vLLM v0.9.0 + PAT backend, CUDA 12.4 | Q3.5, PAT实验笔记 (760.3) |
| **GPU/NVIDIA** | MoE-CAP Benchmark | 自动化MoE评测流水线：CAP Profiler在每层router附近植入轻量级probe (记录expert激活布尔值+router top-k+BS+延迟, overhead 2.7%)→计算S-MBU (模型带宽利用率)+S-MFU (模型FLOPS利用率)+硬件成本→CAP雷达图三维度综合评估。覆盖consumer到datacenter全GPU谱系 | A100-80G-SXM4/PCIe, H100, H20, A6000, A5000, RTX4090, Apple M3, Orin | Qwen3 MoE, DeepSeek-V2-Lite, DeepSeek-R1, GSM8K/HumanEval/MMLU | vLLM/SGLang/MoE-Infinity/K-Transformers | Q3.5, MoE-CAP实验笔记 (1119.8) |
| **NPU/Ascend** | CANN GE/IRFusion | 图编译 + 规则驱动算子融合。IRFusion融合模式：Conv→BN→ReLU (Cube输出tile→Vector直连→HBM), LayerNorm→Linear (LN输出直连GEMM L1 buffer), Gated FFN水平融合 (gate_proj+up_proj→[W_gate|W_up]单GEMM)。硬件融合约束：Cube Unit 16×16 systolic array对齐要求、Vector Unit 32-lane SIMD对齐、L1 buffer 1MB/AI Core容量上限 | Ascend 910B/910C (32 AI Cores, 256 TFLOPS FP16) | PanGu-Σ MoE, Stable Diffusion CANN移植 | MindSpore MINI IR / torch_npu → CANN GE IR → AscendC DSL | Q3.4, Q3.6, CANN知识笔记 |
| **NPU/Ascend** | EPD-Serve 三阶段解耦 | E/P/D三阶段拆分为独立进程：Encode(视觉编码)→Prefill(LLM预填充)→Decode(自回归生成)。7+种部署拓扑 (E-P-D/EP-D/(E-P)-D/(E-PD)×2等)。分层分组KV传输：Prefill计算L+1层时L层KVCache异步传输至Decode→通信聚合优化 (1024 seq下overlap ratio +58%)。AI Core (矩阵乘)+AI Vector (向量/通信) 功能单元分离→算子级并行 | Ascend Atlas 800I A2 (单NPU 64GB) | openPangu-7B-VL (ViT 0.7B+LLM 7B), Qwen3-VL-8B, VisualWebInstruct/ShareGPT-4o | vLLM v0.11.0 + Mooncake Store | Q3.5, EPD-Serve实验笔记 (2710.1), paper_secs §4 (1262.7) |
| **NPU/Ascend** | ElasticMoE CloudMatrix384 | 384×Ascend 910C超节点弹性伸缩：zero-copy (rtIpc跨进程共享权重)、p2p-copy (HCCL isend/irecv绕过host memory)、vpage-remap (虚拟内存动态重映射expert权重)、disk-copy (按名称/partition/layer选择性加载)。CANN图编译 (GE+IRFusion) 自动融合+内存优化。Scale-up latency 2.43s (DP3→DP4, DeepSeek V2 Lite) ≈ 0.11× Cold Restart | CloudMatrix384: 384×910C (64GB/NPU), 192×Kunpeng 920, 24节点, Unified Bus | DeepSeekV2 Lite, Qwen3-30B-A3B, DeepSeek V3 | ascend-vLLM + CANN | Q3.5, ElasticMoE实验笔记 (817.0) |
| **NPU** | llm.npu Chunk-Sharing Graph | W8A8量化+chunk-sharing graph：120/144静态子图可共享 (Linear, LayerNorm等)→内存-75% (7.2GB节省)。Shadow outlier execution: NPU整数计算+CPU浮点outlier补偿 (<0.3% outlier channels)。Out-of-order子图调度消除37% bubble rate。NPU执行时间~315ms (Qwen1.5-1.8B中NPU是critical path) | 移动端NPU (实验环境) | Qwen1.5-1.8B | llm.npu runtime | Q3.6, Fast On-device LLM paper (1393.6) |
| **加速器/SambaNova** | SN40L Dataflow Compiler | 编译器空间融合 (Spatial Fusion)：整个decoder layer编译为单kernel launch→所有算子空间映射到不同PCU/PMU组→中间结果通过RDN (Reconfigurable Dataflow Network) 2D mesh流式传递，不回写HBM。三种kernel执行模式：Unfused (每op独立kernel)、Fused+Software Orchestrated、Fused+Hardware Orchestrated (AGCUs硬件调度，消除host→device往返)。静态垃圾回收：编译器通过符号生命周期分析分配设备虚拟地址。三级存储：DDR (1.5TiB, ~200GB/s)→HBM (64GiB, ~1.8TB/s)→片上SRAM (520MiB) | SN40L RDU (TSMC 5nm, 638 BF16 TFLOPS, 1040 PCU+1040 PMU) | Samba-CoE (150 experts >1T params), Llama 3.1 | SambaNova自研编译器 | Q3.5, SN40L实验笔记 (954.4, 9028.5) |
| **加速器/Groq** | Groq确定性编译 | 全静态编译：所有并发由编译器静态决定→算子直接映射到芯片特定计算单元 (spatial mapping)→运行时零调度/零同步/零cache miss。编译器决定每个cycle的精确数据路由路径→确定性指令流 (完全loop unrolling到具体cycle)。片上SRAM为主内存 (无HBM)→70B模型需数百张卡 | Groq LPU (TSP架构) | Llama-3, Mixtral | Groq Compiler | Q3.6, RPU RELATED WORK (122.2) |
| **加速器/Cerebras** | Cerebras Dataflow Compilation | 晶圆级芯片 (WSE-3: ~900,000 PE, 44GB片上SRAM, 2D mesh互联) 的dataflow编译。MoE模型：各expert FFN映射到不同物理区域→全并行执行。每expert内部MatMul→Act→Norm编译为相邻核心间pipeline (流水线深度=算子链长度, token流水线通过物理核心) | Cerebras CS-3 (46,225 mm²) | Cerebras-GPT | Cerebras SDK | Q3.6, RPU RELATED WORK (122.2) |
| **加速器/TPU** | GSPMD+XLA SPMD Codegen | 6阶段编译：User Annotation (<10 key tensors sharding hint)→Sharding Completion Pass (优先队列约束传播, O(N)图节点, 冲突时插入Reshard/AllGather/AllToAll)→SPMD Program Generation (所有devices同程序, 不同数据分片)→XLA Fusion Pipeline (Instruction/Multi-Output Fusion+Layout Optim+Memory Planning+Async Op Scheduling)→GPU CUDA (Thunk IR) / TPU VLIW Codegen→Runtime execution。TPU ICI (Inter-Chip Interconnect) 支持collective-permute硬件加速+2D/3D torus拓扑优化 | TPU v4 (512-device, 2D torus), TPU v5e/v5p | GLaM 1.2T MoE, Gemini MoE, PaLM | XLA/OpenXLA + JAX pmap/shard_map | Q3.1, Q3.4, Q3.6, GSPMD知识笔记 (41.8) |
| **加速器/ELK** | ELK O(KN)归纳调度 | ICCA (Inter-Core Connected AI Chip) 编译框架：从最后算子归纳推导每个算子的最优preload number (O(KN)复杂度, 非指数枚举)→Pareto最优片上内存分配（执行算子与预加载算子间两层tradeoff）→Cost model使用linear tree model (以tile shape为输入, profiled execution time为输出)。Preload重排：编译时静态决定所有preload-async+execute并发plan | ICCA芯片 (IPU-POD4) | LLM decoder | ELK编译器 (C++) | Q3.3, paper_secs ELK (380.1) |

---

## 按方法类别分类

| 分类 | 方法 | 具体方法描述 | 核心机制 | 来源 |
|------|------|-------------|----------|------|
| **IR设计** | XLA HLO all-to-all + GSPMD | HLO: `all-to-all` (NCCL alltoall映射)→`fusion` (multi-output: gate_proj+up_proj共享输入→单kernel)→逆`all-to-all` combine。HLO→LHLO (buffer assignment)→LMHLO (MLIR HLO dialect)→gpu.launch (Thunk-based runtime)。GSPMD: device mesh=[8,64], sharding hint→约束传播(O(N))→自动插入Reshard collective→SPMD program生成 | IR中并发表达：`all-to-all`为op一等公民，GSPMD自动插入sharding+collective，async all-to-all (实验性flag) | Q3.1, Q3.4 (XLA/GSPMD) |
| **IR设计** | MLIR scf.forall + async.execute | 四层lowering：StableHLO (all_to_all op)→Mixed dialect (scf.forall标注expert迭代独立→编译器并发发射)+Async dialect (async.execute token→显式并发+async.await barrier)→GPU lowering (async.execute→独立CUDA Stream, async.await→cudaEventRecord/WaitEvent)。Polyhedral affine融合：affine-loop-fusion→scalar-replacement→parallelize→GPU lowering (gpu.launch+map-parallel-loops) | scf.forall适合规则并行 (各expert计算量相近, 可fusion/tiling)；async.execute适合不规则并行+灵活device placement+通信overlap | Q3.1, Q3.4 (MLIR笔记578/2553) |
| **IR设计** | Cypress Event-Based IR | SSA异步event建模：每异步操作显式生成event值，支持`[(N, WARP)]`级别并行完成追踪。Event数组promote→硬件同步lowering (`__syncwarp`/`mbarrier`/`warpgroup sync`)。纯编译时construct，无运行时开销 | SSA event值编码异步依赖→WARP/THREAD粒度并行追踪，broadcast [:] 索引→硬件同步 | Q3.1, Event-Based IR知识笔记 (3521) |
| **IR设计** | Difflow DFG/dGraph | Denoising loop展开到收敛(≤5步)→符号属性传播(redundant? T/F)→dGraph分解(按输出属性表达式相同性对连续算子分组)→dEngine multi-version编译(每属性组合→不同dEngine, 运行时根据实际属性选择)。Symbolic shape propagation: ragged帧数→合并到batch维或round-robin到GPU blocks | 符号属性传播(Table 1规则, 如BatchMatmul [NHW] redundant传播)→跨迭代数据复用的编译时发现→消除冗余HBM读取 | Q3.1, paper_secs Difflow §4 (1488) |
| **算子融合** | FlashFuser DSM Fusion | 见实验环境分类GPU/NVIDIA行 | 五级存储层次+dsm_comm原语+Dataflow Analyzer贪心放置+Fusion Search Engine剪枝搜索 | Q3.2 (FlashFuser 16491) |
| **算子融合** | MPK Mega-Kernel | 见实验环境分类GPU/NVIDIA行 | SM-level tGraph+事件融合/归一化/线性化+Worker-Scheduler全异步运行时 | Q3.2 (MPK 8593) |
| **算子融合** | Group GEMM (MoE水平融合) | 见实验环境分类GPU/NVIDIA行 | 多expert FFN→单kernel launch, offline profiled tile选择, sentinel filtering | Q3.2 (Group GEMM 37.6) |
| **图优化** | MegaScale-MoE Holistic Scheduling + SAR | 手动编排MoE层算子：前向RMSNorm→All-Gather→Scatter→GroupedGEMM (3个dot fused)→SiLU*→GroupedGEMM→Gather+Reduce-Scatter→ResidualAdd。反向含SAR (Selective Activation Rematerialization)：仅保留{hidden,ln1_out,qkv,attn,ln2_out,fc2_out}，丢弃{fc2_in,ffn_in,fc1_out,fc3_out}→重计算与gradient通信交织。激活内存节省：Mixtral-8x7B 45.5%, MFU差异<0.5% | 算子类型决定keep/drop策略 (GEMM类保留, RMSNorm/通信类丢弃)；inter-operator级别双CUDA stream (S_compute+S_comm) 并发 | Q3.3, SAR知识笔记 (1601.5), Holistic Scheduling知识笔记 (212.8) |
| **图优化** | NetMoE ILP Device Placement | 将token-to-expert映射建模为ILP问题：变量SmpDev(i)∈[I]*_n (token i分配到节点n的expert集合中)，目标minimize Σ_i[t_intra+t_inter]，约束每个token恰好top-k节点、节点上expert已预放置、显存不溢出。Per-layer分解+贪心预热→1000 GPU规模下<1s编译。运行时placement_plan查询O(1) | ILP求解在编译时offline完成；运行时各GPU独立CUDA stream并发发射expert FFN | Q3.3, paper_secs NetMoE (889.4) |
| **图优化** | HuntKTm Stream Scheduler | 见实验环境分类GPU/NVIDIA行 | DFG→拓扑分层→同level无依赖kernel→不同stream→冗余同步消除+内存生命周期管理 | Q3.3 (HuntKTm 4288.9) |
| **自动调优** | HyTiS Auto-tuning | 见实验环境分类GPU/NVIDIA行 | 离线profiling→S_TO+S_LO候选集→两级tile scheduling (TO全波+LO尾波)→实测benchmark | Q3.3 (HyTiS 690.5) |
| **自动调优** | Infera Zero-Tuning | Tile size静态约束推导：Register File Level (32-bit reg/thread limit=64/96/128→平衡ILP vs TLP)→Shared Memory Level (usage/block=48/80/112/144 KiB)→Global Memory Level (grid_size fixed 64)→三维multi-version kernel生成 (reg×smem×pipeline_stages) | 完全跳过GPU profiling，基于静态资源约束推导；编译并行化→CPU核心增加时编译时间按比例缩短 | Q3.3, Tile-Based Zero-Tuning知识笔记 (297.9) |
| **自动调优** | FlowMoE BO Auto-tuning | 贝叶斯优化搜索all-reduce chunk size S_p：高斯过程拟合f(S_p) posterior distribution→Expected Improvement采集函数→约8次采样收敛 (BO开销<1% per-iteration time, 大模型评估~32s/次) | BO扩展到系统参数搜索 (非kernel参数)→环境变化时自动重新搜索 | Q3.3, BO知识笔记 (667.2) |
| **编译框架实现** | TVM (Relay→TensorIR→Relax+AutoScheduler) | Relay IR: `parallel` annotation标注算子级并发 (8 experts→8 CUDA streams独立)。TensorIR: `T.parallel`/`T.vectorize(4)`/`T.bind(threadIdx.x)`轴→微算子级并发。AutoScheduler: Compute DAG→Sketch Generation→Evolutionary Search (128 population, XGBoost cost model)→实测验证→收敛。Joint Tuning: cost_model.predict惩罚SM oversubscription/HBM BW oversubscription/Register pressure | compute/schedule分离；cost model XGBoost基于实测数据 (vs XLA纯启发式)；编译时间122s (cold)/5s (warm cached) for MoE 8 experts | Q3.4 (TVM笔记) |
| **编译框架实现** | PyTorch 2.0 (Dynamo+Inductor+CUDA Graph) | TorchDynamo: Python bytecode劫持→FX Graph capture (SSA形式编码数据依赖→无依赖node可标记并发)。Inductor: Fusion Group Formation (LN→QKV垂直融合, gate+up水平融合concat为[W_gate|W_up]单GEMM, silu+mul+down垂直融合)→Tiling Heuristic (SM shared_mem_capacity+register_file_size硬约束)→Triton codegen。CUDA Graph: capture所有kernel launch+event sync→单次replay (~3μs total vs ~32μs 传统4 kernel launch) | FX Graph SSA→自动并发发现；max-autotune mode自动探索multi-stream调度；FlexAttention: Python callable→Dynamo子图→Inductor lower为Triton代码块→注入手工attention kernel模板主循环 | Q3.4 (GPT-Fast笔记) |
| **编译框架实现** | CUTLASS 3.x CuTe DSL | CuTe C++ template→nvcc编译→PTX→SASS。Warp specialization: Producer (2 warps) TMA_load→Consumer (6 warps, warpgroup) wgmma。Persistent Tile Scheduler: CTA与tile解耦→atomicAdd动态work stealing→自动负载均衡 (快CTA多取tile)。Ping-Pong 4-stage pipeline消除TMA latency bubble。TiledMMA/TiledCopy/Pipeline异步调度→直接SASS/PTX生成 | CuTe不走MLIR抽象，直接控制异步硬件的精确timing；vs Triton: 无法在Triton中表达warp-specialized异步调度/Ping-Pong pipeline/cluster-level sync | Q3.4, Q3.6, CUTLASS/CuTe知识笔记 |
| **Codegen** | CUDA Graph | 见实验环境分类GPU/NVIDIA行 | DAG静态化+GPU端单次launch (Stream Capture或Explicit API)→Instance化全局资源分配→Replay | Q3.6 (CUDA Graph 5770.5) |
| **Codegen** | ACS OoO调度 | 见实验环境分类GPU/NVIDIA行 | 编译时RW-segments标注+运行时调度窗口+O(segments²)依赖检测+ACS-HW GP CP集成 | Q3.6 (ACS 2407.6/2060.4) |
| **Codegen** | Warp Specialization + TMA | 见实验环境分类GPU/NVIDIA行 | DMA warp (tid 128-159)∥Compute warpgroup (tid 0-127)+TMA硬件解耦+mbarrier同步 | Q3.6 (TMA 12311.5, Task-Based paper 1736.1) |
| **Codegen** | Software Pipeline | 见实验环境分类GPU/NVIDIA行 | Ampere cp.async.commit_group/wait_group或Hopper TMA+mbarrier→计算-搬运重叠 | Q3.6 (async-copy 10327.9, Software Pipeline 795.9) |
| **Codegen** | GSPMD+XLA TPU Backend | 见实验环境分类加速器/TPU行 | SPMD partition+collective-permute异步overlap+VLIW bundle编排 (scalar+vector+matrix+dma同cycle并发)+scratchpad编译时静态分配 | Q3.6 (GSPMD 41.8) |
| **Codegen** | Groq/Cerebras确定性编译 | 见实验环境分类加速器行 | 全静态空间映射→算子→chip特定计算单元+每cycle数据路由预计算→确定性延迟 (编译时可知) | Q3.6 (RPU 122.2) |

---

## 分类详细问答

### 分类: IR设计 — MoE模型

#### 方法: XLA HLO all-to-all + GSPMD

MoE的编译IR核心是将expert dispatch/combine的通信语义与expert FFN的计算语义在IR中统一表达。

**XLA HLO中MoE层的IR lowering链**：

```
Step 1: PyTorch/JAX → 计算图
  gate_logits = Linear(hidden)              // [B×S, E]
  topk_indices, topk_weights = TopK(gate_logits, k=2)
  dispatched = Dispatch(tokens, topk_indices)  // [G×C, D] per GPU
  expert_out = ExpertFFN(dispatched)           // 各expert独立计算
  combined = Combine(expert_out, topk_indices, topk_weights)

Step 2: XLA HLO表达
  %dispatched = all-to-all(%tokens, replica_groups={{0,1,...,G-1}})
      // 底层: NCCL alltoall (intra-node NVLink 900GB/s / inter-node IB 400GB/s)
  %expert_out = fusion(
      dot(%dispatched, %w1),     // gate_proj
      dot(%dispatched, %w2),     // up_proj  
      multiply(%gate, %up),      // SiLU activation
      dot(%result, %w3)          // down_proj
  )  // XLA multi-output fusion: 3个dot共享输入→融合为单kernel
  %combined = all-to-all(%expert_out, replica_groups=...)

Step 3: XLA GPU Codegen
  HLO → LHLO (buffer assignment) → LMHLO (MLIR HLO dialect) →
  gpu.launch (CUDA kernel) + Thunk-based runtime (XLA:GPU)
```

**GSPMD 6阶段编译**：
```
Phase 1: User Annotation (<10 key tensors sharding hint)
  mesh = Mesh(devices.reshape(8, 64), ('expert', 'hidden'))
  sharding_W_expert = NamedSharding(mesh, P('expert', None))

Phase 2: Sharding Completion Pass (优先级队列约束传播, O(N))
  PriorityQueue<HloInstruction*> worklist;
  while (!worklist.empty()):
    inst = worklist.pop()  // elementwise优先, MatMul次之
    infer_output_sharding(inst)
    // 冲突时插入 Reshard (AllReduce/AllGather/ReduceScatter/AllToAll)

Phase 3: SPMD Program Generation
  所有512 devices执行同一program，各处理不同数据分片

Phase 4: XLA Fusion Pipeline
  Instruction Fusion + Multi-Output Fusion + Layout Optim + 
  Memory Planning + Async Op Scheduling 
  (xla_gpu_enable_latency_hiding_scheduler=true ← 实验性)

Phase 5: GPU/TPU Codegen
  GPU: HLO→LHLO→Thunk IR (Thunk_AlltoAll→Thunk_FFN→Thunk_AlltoAll_combine)→CUDA kernel
  TPU: HLO→TPU HLO→VLIW instruction bundles (collective-permute+compute重叠)

Phase 6: Executable→Runtime
  512 devices: All-to-All dispatch (async) → Expert FFN (fused GEMM) → All-to-All combine → LayerNorm+residual
```

**硬件约束**：all-to-all的硬件映射——X-MoE发现intra-node (Infinity Fabric 200GB/s) 与inter-node (Slingshot 25GB/s) 带宽不对称达8:1。HLO `all-to-all` codegen需感知此拓扑（intra-node NVLink p2p, inter-node IB send/recv）。Irregular all-to-all (Lancet双趟协议: 先交换size再交换数据) 在IR层面需支持variable-length communication——对传统HLO固定shape假设构成挑战。

**并发机会**：expert FFN内部3个dot (gate_proj, up_proj, down_proj) 传统XLA以融合单kernel串行执行。多expert (8个) 之间数据独立，若IR能标注每个expert为独立async.execute region，可在多CUDA stream上并发——但standard XLA以串行为主，async all-to-all支持有限。

- **核心机制**: HLO `all-to-all` op + GSPMD自动sharding推导 (约束传播O(N)) + device mesh拓扑感知partition + SPMD program生成
- **实现**: XLA/OpenXLA (Apache 2.0), JAX pmap/shard_map, PyTorch-XLA
- **实验环境**: TPU v4 512-device 2D torus (GSPMD论文); Frontier HPC Dragonfly topology (X-MoE: intra-node IF 200GB/s vs inter-node Slingshot 25GB/s)
- **来源**: Q3.1, Q3.4, GSPMD知识笔记 (41.8), Irregular All-to-All知识笔记 (1797), X-MoE paper_secs (9305)

#### 方法: MLIR scf.forall + async.execute

**四层lowering**：
```
Level 1: StableHLO
  stablehlo.all_to_all(%tokens) → stablehlo.custom_call @expert_ffn → stablehlo.all_to_all

Level 2: Mixed dialect
  scf.forall (%e) in (0 to 8) {  // 标注expert并发
    linalg.matmul ins(%expert_in, %W_expert[%e]) outs(%buf)
  }
  // scf.forall 硬件映射: GPU→每iteration映射为一个thread block或独立kernel

Level 3: Async dialect (显式并发)
  %tok_e0 = async.execute { call @expert_0(%in0) }
  %tok_e1 = async.execute { call @expert_1(%in1) }  // 并发于e0
  async.await %tok_e0, %tok_e1, ...  // gather barrier

Level 4: GPU lowering
  async.execute → 独立CUDA Stream
  async.await → cudaEventRecord/cudaStreamWaitEvent
  gpu.launch → CUDA kernel launch on assigned stream
```

**scf.forall vs async.execute选择**：scf.forall适合规则并行（各expert计算量相近，编译器可fusion/tiling）；async.execute适合不规则并行，支持更灵活的设备放置和通信overlap。硬件约束：单GPU并发CUDA stream数受限于Hyper-Q（通常32硬件工作队列），8 experts并发可行，超32 expert需compile-time stream assignment策略。

- **核心机制**: 多dialect progressive lowering (StableHLO→Linalg→Async→GPU)，scf.forall无依赖迭代标注+async.execute token-based异步+async.await barrier
- **实现**: LLVM MLIR (Apache 2.0), C++ API + Python bindings
- **实验环境**: GPU (CUDA/Vulkan), PIM, NPU; MLIR PIM Compiler (PIMphony) paper_secs (2553)
- **来源**: Q3.1, Q3.4, MLIR知识笔记 (578), MLIR PIM Compiler知识笔记 (2553)

### 分类: 算子融合 — GPU GEMM链融合

#### 方法: FlashFuser DSM-based GEMM Chain Fusion

FlashFuser是首个利用H100 GPU DSM (Distributed Shared Memory) 进行GEMM链算子融合的编译框架。DSM是H100 Thread Block Cluster内多SM之间共享内存(SMEM)的互联通道(L1.5 cache)，提供比HBM更低延迟更高带宽的片上数据通路。FlashFuser将中间张量从传统Reg→SMEM→HBM路径扩展为Reg→SMEM→DSM→L2→HBM五级层次，突破单SM SMEM 227KB容量上限(cluster内可达~3.6MB)。

**编译流水线**：
```
输入: GEMM算子链计算图 (Standard/Gated FFN, Conv Chain)
  ↓
dsm_comm原语定义 (编码Cluster内SM划分与数据流):
  dsm_all_exchange: cluster内AllReduce/Mul
  dsm_shuffle: Shuffle Group环形通信
  dsm_reduce_scatter: scatter-reduce聚合
  inter_cluster_reduce: TMA cp.reduce.async.bulk跨cluster原子归约
  ↓
Dataflow Analyzer (Algorithm 1):
  Loop Scheduling (MNLK vs MLNK选择: 前者需本地存储完整tensor C→DSM压力大;
   后者每次LK迭代后产出partial E→可在寄存器中累加但受限于register file容量)
  + Tile Selection (两级: Cluster-level决定跨cluster通信需求;
    Block-level决定单block内存占用)
  + Resource Mapping (贪心: Reg→SMEM→DSM→L2→HBM逐级放置中间张量)
  ↓
Fusion Search Engine:
  5条Pruning Rules (消除不合法cluster配置+约束tile size为hardware-friendly值+
   消除冗余loop schedule+资源映射不超硬件限制+DSM带宽感知cost-based pruning)
  搜索空间: 2.75×10^13 → 1.15×10^6 (减少>99.99%)
  cost model对每个候选计算总数据搬运量
  ↓
最优执行计划 (Loop Schedule, Tile Size, Resource Mapping) → CUDA/TMA Kernel codegen
```

**Standard FFN垂直融合 (GEMM(A,B)→C→GEMM(C,D)→E, cluster size=(2,4,2,4))**：
1. GEMM0: 2 blocks沿K维并行计算partial C
2. dsm_all_exchange沿K维cluster内AllReduce→完整C tile驻留DSM，不写回HBM
3. GEMM1: dsm_shuffle (ShuffleGroup_0, ring_communication) 分发C tile→各block计算partial E
4. Store: dsm_reduce_scatter + inter_cluster_reduce (TMA cp.reduce.async.bulk) 两级归约

**Gated FFN (SwiGLU) 变体**：Gate和Up两个并行GEMM可通过spatial partitioning (cls_k=2, 两分支不同block group) 最大化并行度，或sequential execution最小化DSM通信。dsm_all_exchange操作从Add变为Mul (完成SiLU(gate)⊙up)。

**五级存储层次**：
- Reg (L0): ~256KB/SM, ~200TB/s
- SMEM (L1): 227KB/SM, ~4TB/s
- DSM (L1.5): cluster互联, 带宽随cluster size递减 (8-SM ~2TB/s vs 4-SM ~3TB/s)
- L2: 50MB/chip, ~12TB/s
- HBM: 80GB, 3TB/s

- **核心机制**: DSM (H100 Thread Block Cluster多SM SMEM互联) 扩展有效片上空间→突破单SM 227KB限制；dsm_comm四种通信原语+Dataflow Analyzer (贪心放置+Loop Schedule搜索)+Fusion Search Engine (5条剪枝规则+全配置枚举cost model)；中间58% memory access消除
- **实现**: CUTLASS 3.x kernel模板三阶段插入 (prologue: DSM semaphore/mbarrier初始化; mainloop: dsm_all_exchange+dsm_shuffle注入; epilogue: dsm_reduce_scatter+inter_cluster_reduce); TMA shared::cluster地址空间SM-to-SM搬移; mbarrier many-to-many sync
- **实验环境**: H100 80GB, CUDA 12.4, PyTorch 2.6, TVM 0.9, Triton 3.2; benchmark: GEMM chains/Convolutional chains/Gated FFNs (SwiGLU); baseline: cuBLAS, Chimera, BOLT, Welder, MCFuser, TVM, Triton; 3.3× vs cuBLAS, 4.1× vs compilers, 1.24× end-to-end speedup
- **来源**: Q3.2, paper_secs FlashFuser (16491), dsm_comm知识笔记 (29.5), GEMM-based Operator Chain知识笔记 (33.1)

#### 方法: Mirage Persistent Kernel (MPK) — Mega-Kernel融合

MPK将kernel fusion推向极致——整个模型的计算+通信全部融合为单个persistent kernel (mega-kernel)，通过SM-level图表示(tGraph)和device-memory同步原语协调所有SM执行。

**tGraph编译流程**：
```
Step 1: Operator Decomposition → SM-level Tasks
  输入: MatMul(QKV) → Attention → AllReduce → RMSNorm
  Q_i (SM_i负责Q的1/S行), K_i, V_i, A_i, O_i, R_i

Step 2: Dependency Analysis → 精确事件依赖
  传统: 所有MatMul完成→AllReduce才能开始
  MPK: AllReduce task_i仅依赖MatMul task_i→MatMul task_j (j≠i)可与AllReduce task_i并发

Step 3: Event Fusion
  Successor-set fusion: OutTasks(e1)=OutTasks(e2)→合并事件
  Predecessor-set fusion: InTasks(e1)=InTasks(e2)→合并事件

Step 4: tGraph Normalization (每task最多1依赖+1触发)
  fan-out>1: 引入空dummy task relay; fan-in>1: 同上
  新增开销<1% (实际模型以"深"为主)

Step 5: tGraph Linearization (BFS保证同事件触发的task连续)
  Algorithm 1: E←{所有无依赖事件}; 事件fan-out仅需存first/last task index

Step 6: 存储为GPU Device Memory紧凑结构
  每个task: [dependent_event_idx, trigger_event_idx]
  每个event: [trigger_count_required, first_task_idx, last_task_idx]
```

**Worker-Scheduler运行时**：GPU SM分区为workers (维护FIFO task queue) 和schedulers (维护task依赖，依赖满足时分配task)。事件驱动、全异步执行，GPU在推理期间零kernel launch。消除kernel launch overhead (从数百次→1次)、支持跨算子prefetch (当前算子计算时预取下一算子数据)、支持计算-通信重叠 (MatMul task与AllReduce task交错执行，隐藏通信延迟)。

- **核心机制**: SM-level tGraph (sub-kernel粒度依赖表达→跨算子软件流水线→compute-communication overlap) + 事件融合(按predecessor/successor set) + 图归一化(每task≤1依赖+1触发) + BFS线性化 + Worker-Scheduler全异步运行时
- **实现**: CUDA kernel + MPK runtime; 与FlashFuser互补——FlashFuser聚焦GEMM链compute-intensive fusion (DSM扩展片上缓存)，MPK聚焦全模型mega-kernel (tGraph跨算子并发)
- **实验环境**: A100/H100/B200; LLM serving (LLaMA-like) + 多GPU推理; baseline: PyTorch+CUDA Graphs+torch.compile, SGLang, vLLM; 1.0-1.7× speedup on both single- and multi-GPU
- **来源**: Q3.2, paper_secs MPK §2.2 (8593)

### 分类: 图优化 — 子图切分与设备放置

#### 方法: NetMoE ILP Device Placement

将MoE训练中的token-to-expert映射建模为整数线性规划(ILP)：

```
变量: 对每个token i∈[I], 目标节点n∈[N]: SmpDev(i)∈[I]*_n
目标: minimize Σ_i [t_intra(SmpDev(i)) + t_inter(SmpDev(i))]
约束: (a) 每token恰好top-k节点 (k=2 for Mixtral)
      (b) 节点n上SmpDev(i)涉及的expert必须在Node(n)上
      (c) 每GPU token数不超过显存容量

求解: Gurobi/OR-Tools, per-layer分解+贪心预热→1000 GPU规模<1s编译
运行时: placement_plan = lookup(optimal_placement, router_output); O(1)查询
        各GPU独立CUDA stream并发发射expert FFN
```

- **核心机制**: ILP建模通信-计算最优映射 (intra-node NVLink vs inter-node IB/RoCE代价不对称)→per-layer分解秒级求解→运行时O(1)查询
- **实现**: Gurobi/OR-Tools ILP求解器 (offline); FlexMoE补充动态expert迁移→编译时多版本IR生成
- **实验环境**: 1000 GPU规模分布式训练 (paper_secs NetMoE 889.4)
- **来源**: Q3.3, paper_secs NetMoE (889.4)

#### 方法: MegaScale-MoE Holistic Scheduling + SAR

MegaScale-MoE完整调度时间线 (2 CUDA streams: S_compute主计算, S_comm通信)：

```
Forward:
T0-T3:   RMSNorm [S_compute]
T4-T8:   All-Gather [S_comm]
T9-T12:  Scatter→ffn_in [S_compute]
T13-T28: GroupedGEMM(fc1,fc3)→SiLU*fc3→GroupedGEMM(fc2) [S_compute]
T45-T48: Gather+Reduce-Scatter [S_comm]
T49-T50: ResidualAdd [S_compute]

Backward (含SAR):
仅保留 {hidden, ln1_out, qkv, attn, ln2_out, fc2_out}
丢弃 {fc2_in, ffn_in, fc1_out, fc3_out}→反向重计算
T51-T54: All-Gather(Δffn_out) [S_comm]
T51-T60: GroupedGEMM_bwd(Δfc2_out) [S_compute]  // 与gradient all-gather并发!
T61-T64: Recompute SiLU [S_compute]  // memory-bound, 开销可被通信掩盖
T65-T72: GroupedGEMM_bwd(Δfc2_in) [S_compute]
T73-T76: Reduce-Scatter [S_comm]; Recompute RMSNorm [S_compute] // 重计算与通信重叠
T77-T84: FlashAttention backward [S_compute]

激活内存: Mixtral-8x7B节省45.5%, MFU差异<0.5%
```

- **核心机制**: 手动编排算子顺序+inter-operator双CUDA stream (计算/通信并发)+SAR (保留GEMM输出/丢弃RMSNorm+通信输出)→重计算与gradient通信交织→激活内存-50%
- **实现**: 当前为手动hand-tailored (论文§7指出未来方向是自动搜索); 不依赖torch.autograd monolithic反向
- **实验环境**: NVIDIA H800 GPU; Mixtral-8x7B MoE训练
- **来源**: Q3.3, SAR知识笔记 (1601.5), Holistic Scheduling知识笔记 (212.8)

### 分类: 自动调优 — Tile Size搜索

#### 方法: HyTiS Offline Profiling + 自适应搜索空间

**阶段1: Offline Profiling (一次完成)**：
```
对每个candidate micro-kernel tile K_i=(bM,bN,bK,layout):
  t(K_i) = measure_latency(K_i)         // CUDA event计时
  SMEM(K_i) = query_shared_mem_usage(K_i)
  REG(K_i) = query_register_usage(K_i)

构建S_TO (Throughput-Oriented):
  约束1: SMEM(K_i)<=SMEM_capacity, REG_spill==0
  约束2: 无更大维度K'同时满足资源约束
  约束3: ISA (H100 wgmma要求bM%64==0)
  T(K_i)=(M_i*N_i)/(n0*t(K_i)); S_TO={K_i|diff(T,max(T))<l1}

构建S_LO (Latency-Oriented):
  t_wave(K_i)=t(K_i)/n0; S_LO={K_i|diff(t_wave,min(t_wave))<l2}
```

**阶段2: 运行时Auto-tuning**：
```
function autotune(P(M,N,K)):
  for K1 in S_TO:
    total_tiles = ceil(M/K1.bM) * ceil(N/K1.bN)
    partial_tiles = total_tiles % N_SM
    for K2 in S_LO:
      if partial_tiles==0: valid=true  // TO-only
      elif partial_tiles<=N_SM: valid=true  // two-level: TO全波+LO尾波
      else: continue
      for layout in [GM_opt, GN_opt]:
        bench(HyTiS_GEMM, K1, K2, layout, P)
  best_config = argmin(latency); cache.store(P, best_config)

// H100: 平均搜索空间14配置 (max 66) vs Inductor-Triton固定19→搜索空间-86%
// Profiling: H100 ~19min, A100 ~36min (per device, 一次完成)
```

**硬件建模假设**："SM架构独立性"——micro-kernel在单个SM上的性能特征在不同problem shape下相对稳定，profiling结果可跨workload复用。对NVIDIA SM成立，对NPU脉动阵列不保证。

- **核心机制**: 离线profiling构建S_TO+S_LO两层候选集→运行时两级tile scheduling (TO全波填满所有SM+LO尾波处理partial tiles)→benchmark实测选最优
- **实现**: Python + Triton/Inductor; 不修改Triton compiler本身
- **实验环境**: H100/A100; profiling H100 ~19min, A100 ~36min (per device, 一次性)
- **来源**: Q3.3, HyTiS知识笔记 (690.5, 42.8)

#### 方法: Infera Zero-Tuning

完全跳过GPU profiling，基于静态资源约束推导tile size：
```
Register File Level: 32-bit reg/thread limit = 64/96/128→平衡ILP vs TLP
Shared Memory Level: usage/block = 48/80/112/144 KiB→spatial tile = thread_tile×thread_count
Global Memory Level: spatial tile = block_tile×grid_size (grid_size fixed 64)

Multi-Version Micro Kernel Generation:
for reg_config in {64, 96, 128}:
  for smem_config in {48, 80, 112, 144}KB:
    for pipeline_stage in {2, 3, 4}:
      if resource_feasible(reg, smem, stage):
        generate_kernel(reg, smem, stage)
// 无需GPU profiling→编译并行化→CPU核心增加时编译时间按比例缩短
// vs Ansor/MetaSchedule编译时间低2-3个数量级
```

- **核心机制**: 静态资源约束直接推导 (Roller: 仅考虑tile size因素即可生成高性能kernel)→multi-version kernel枚举 (ILP/TLP/intensity trade-off)
- **实现**: TVM Relay→TensorIR→CUDA; 编译并行化
- **实验环境**: NVIDIA GPU (CUDA)
- **来源**: Q3.3, Tile-Based Zero-Tuning知识笔记 (297.9)

### 分类: 编译框架实现 — 完整编译栈

#### 方法: XLA/GSPMD (6阶段SPMD编译)

见IR设计分类XLA行。核心6阶段：User Annotation→Sharding Completion Pass (O(N)约束传播)→SPMD Program Generation (all devices同program不同数据分片)→XLA Fusion Pipeline (Instruction+Multi-Output Fusion+Layout+Memory+Async)→GPU (Thunk IR→CUDA)/TPU (VLIW) Codegen→Runtime execution。

**核心并发模型**：SPMD (全device同时执行同一program)。同一device内传统XLA Thunk executor串行执行kernel为主。GPU backend async all-to-all依赖实验性flag (`xla_gpu_enable_latency_hiding_scheduler`)。

- **核心机制**: GSPMD自动sharding推导 (优先级队列约束传播, O(N)) + 自动Reshard插入 (AllReduce/AllGather/ReduceScatter/AllToAll) + 2D device mesh (TPU 2D torus拓扑优化)
- **实现**: Google XLA/OpenXLA (Apache 2.0); HLO→LHLO→LMHLO→Thunk IR→CUDA/TPU VLIW
- **实验环境**: TPU v4 512-device (GSPMD 2048 TPUv3编译<1min); GPU (CUDA) async支持状态实验性
- **来源**: Q3.1, Q3.4, Q3.6, GSPMD知识笔记 (41.8)

#### 方法: TVM (Relay→TensorIR→Relax + AutoScheduler)

**Relay IR**: `parallel` annotation标注算子级并发 (8 experts→8 CUDA streams独立)。

**TensorIR**: `T.parallel`轴 (tile跨SM并行) + `T.vectorize(4)` (128-bit SIMD load/store) + `T.bind(threadIdx.x)` (thread-level并行)。`T.block`为计算原子单元 (指定计算逻辑+数据访问region)，`T.buffer`为显式内存抽象 (带scope: global/shared/local)，buffer的producer-consumer关系自动建立依赖DAG。

**AutoScheduler**: Compute DAG→Sketch Generation→Evolutionary Search (128 population, XGBoost cost model trained on measured data)→实测top candidates→convergence。tile_structure枚举=product([32,64,128,256], [32,64,128], [8,16,32])，loop_attributes∈{parallel, vectorize, unroll, bind}，memory_scope∈{global, shared, local}。

**Joint Tuning (多算子并发)**: cost_model.predict(joint_config)惩罚 SM oversubscription (total_blocks>SM×max_blocks) + HBM BW oversubscription (>80% peak) + Register pressure (per-SM reg>65536)。实际采用贪心逐个kernel tune + 联合验证。

- **核心机制**: compute/schedule分离+三层IR (Relay→TensorIR→Relax) 逐级增加并发精度+AutoScheduler (XGBoost cost model based on measured data, vs XLA pure heuristic)+multi-kernel stream assignment (8 streams, 每stream ~16 SM)
- **实现**: Apache TVM; CUDA C++ template codegen (T.parallel→blockIdx, T.vectorize→float4, T.bind→threadIdx)
- **实验环境**: A100; MoE 8 experts; 编译时间122s (cold)/5s (warm cached tuning logs)
- **来源**: Q3.4, RAF Compiler知识笔记

#### 方法: Triton (TTIR→TTGIR→PTX)

**Triton IR**: `program_id(0)` × `program_id(1)` 定义2D grid——所有(pid_m, pid_e)对是独立并发单元 (每对映射到一个CTA)，GPU warp scheduler自动并发调度。这是Triton的核心并发优势：单kernel内所有CTA天然并发。

**编译链**: Python (@triton.jit)→TTIR (block-level, 保留并发信息: program_id沿grid轴, num_warps, num_stages)→TTGIR (thread-level: Layout Conversion→Loop Unrolling→Memory Coalescing→SMEM Allocation→Sync Insertion)→LLVM IR→PTX。

**TTGIR lowering关键**: layout conversion (block-level→per-thread register tiling, distributed layout: warp×lane mapping); memory coalescing (重组thread-to-address mapping→连续地址per warp); SMEM padding (加padding column使stride≠32×4B倍数→避免bank conflict); sync insertion (sm80 cp.async.commit_group+wait_group, sm90 TMA+mbarrier)。

**Autotuner**: `@triton.autotune`装饰器枚举20-50 candidates (tile sizes, num_warps, num_stages)→benchmark (warmup 25 iters + bench 100 iters + CUDA event timing)→cache[signature(M,N,K)]=(config, latency)。key=['M','N','K']：运行时参数变化时重新benchmark。

**硬件架构感知MMA selection**: sm80 (A100)→`mma.sync.aligned.m16n8k32`; sm90 (H100)→`wgmma.mma_async.sync.aligned`。TMA需`tl._experimental_descriptor_load` (实验性)，warp-specialized async pipeline无法直接在Triton表达——这是vs CUTLASS CuTe的核心差距。

**跨kernel编排缺失**: Triton无内置多kernel编排能力——同时执行attention_kernel和ffn_kernel需要外部调度器 (CUDA Stream/PyTorch scheduler/CUDA Graph)。

- **核心机制**: Block-level programming (开发者以Python DSL写tile级运算, 编译器自动处理thread调度/memory coalescing/SMEM管理) + Triton IR program_id grid天然CTA级并发 + Autotuner实测benchmark (key参数变化时重新测) + 架构感知MMA instruction selection
- **实现**: OpenAI Triton (MIT); 已集成vLLM/SGLang/PyTorch Inductor backend
- **实验环境**: NVIDIA (sm_70+, 含H100 wgmma), AMD CDNA3/4
- **来源**: Q3.1 (Triton IR 1762/213), Q3.2, Q3.4, Q3.6

#### 方法: PyTorch 2.0 (Dynamo+Inductor+CUDA Graph)

**TorchDynamo**: Python bytecode劫持→FX Graph capture (每个tensor op→FX Node: placeholder/call_function/call_module/output)。SSA形式天然编码数据依赖→无依赖node标记可并发。遇到不识别的op→graph break→回退eager。

**TorchInductor Fusion**: Fusion Group Formation基于buffer依赖+硬件约束:
- FG1: [ln1, qkv]→fused_layernorm_qkv (垂直融合, LN reduce→GEMM, 中间activation寄存器复用)
- FG2: [attention]→FlexAttention/FlashInfer (编译器注入用户Python callable score_mod/mask_mod为Triton代码块→预写手工attention kernel模板主循环)
- FG3: [gate, up]→fused_gateup_gemm (水平融合, [W_gate|W_up] concat, 单次GEMM双倍输出)
- FG4: [silu, mul, down]→fused_act_down (垂直融合, SiLU tile→pointwise_mul tile→GEMM tile, reg/SMEM保留中间tile)

**Tiling Heuristic**: candidate_tiles=enumerate({32,64,128,256}^3)→valid_tiles=filter(SMEM≤HW.SM.shared_mem_capacity & regs_per_thread×threads_per_block≤HW.SM.register_file_size)→best_tile=argmax(occupancy×compute_intensity)。

**CUDA Graph**: capture所有kernel launch+event sync→单次replay (~3μs total vs ~32μs传统4 kernel)。MagicDec论文为draft和verify各预编译独立CUDA Graph。

- **核心机制**: Dynamo bytecode劫持→FX Graph SSA (自动并发发现) + Inductor水平+垂直融合heuristic (hardware SM occupancy+SMEM constraint) + FlexAttention混合codegen (Python callable→Triton代码块注入手工kernel模板) + CUDA Graph静态化 (消除launch overhead)
- **实现**: PyTorch 2.0 (`torch.compile(mode="max-autotune")`); Triton backend + CUDA Graph
- **实验环境**: H100; DiT single denoising step; compile 120s cold/12s warm; exec: CUDA Graph replay (1 host→device submission)→GPU hardware scheduler dispatches (fused_layernorm_qkv∥fused_gateup_gemm不同stream)→flex_attention (全SM关键路径)→fused_act_down
- **来源**: Q3.4, GPT-Fast知识笔记

### 分类: Codegen — 并发Kernel的代码生成

#### 方法: CUDA Graph (编译时DAG静态化)

**三种构建方式**：
- Stream Capture: `cudaStreamBeginCapture→kernelA→kernelB→EndCapture` (运行时捕捉→编译时固化)
- Explicit API: `cudaGraphAddKernelNode→cudaGraphAddDependencies` (编译时显式构建DAG)
- Instance化: `cudaGraphInstantiate` → GPU驱动全局register allocation+SMEM预分配+依赖边编译为硬件同步原语+多kernel launch序列编译为GPU端单次提交
- Replay: `cudaGraphLaunch` → 仅一次CPU→GPU MMIO写入 (~3μs total)

**致命缺陷**：动态图每次需重建graph——ACS实验Brax仿真中DAG构造时间占程序总执行时间47%。适用静态推理图（固定DiT denoising loop、Video固定帧数pipeline）。

- **核心机制**: kernel launch序列编译为GPU端单次提交的静态执行图→eliminate launch overhead (5-20μs/kernel→~3μs total per replay)→Instance化全局资源静态划分
- **实现**: CUDA Driver API
- **实验环境**: NVIDIA GPU (A100/H100)
- **来源**: Q3.4, Q3.6, CUDA Graph知识笔记 (5770.5)

#### 方法: Hopper Warp Specialization + TMA

```
DMA Warp (tid 128-159, 1 warp, 32 threads):
  // 仅1线程(tid==128)发起TMA传输
  // TMA硬件独立于CUDA Core执行数据传输
  for k in range(0, K/T_K):
      if k>=PIPE: wait(cons[k%PIPE])
      if tid()==128:
          TMA_load(prod[k%PIPE], tile(gA)→sA[:,:,k%PIPE], tile(gB)→sB[:,:,k%PIPE])
      // TMA在后台burst copy (64-256KB/tile), CUDA Core线程立即继续循环(非阻塞)
  wait(copyout); TMA_store(sC→tile(gC))

Compute Warpgroup (tid 0-127, 4 warps×32=128线程):
  // 4 warps协作发射Tensor Core指令 (硬件要求)
  for k in range(0, K/T_K):
      wait(prod[k%PIPE])         // 等待DMA warp TMA完成
      warpgroup_sync()           // 128线程同步
      wgmma(accum, sA[:,:,k], sB[:,:,k])  // Tensor Core
      warpgroup_wait()           // 等待Tensor Core完成
      arrive(cons[k%PIPE])       // 通知DMA warp buffer可复用
  copy(accum, sC); syncthreads(); arrive(copyout)
```

**编译时pipeline深度选择**：pipeline_depth = min(max_pipe_from_hardware, ceil(TMA_latency/GEMM_latency), smem_size/(tile_A_size+tile_B_size))。通常PIPE=2/3。

**TMA vs cp.async**：cp.async (Ampere) 仍需线程显式发射指令并占用寄存器中转数据；TMA (Hopper) 完全由硬件管理 (从发出请求到数据到达SMEM)，无寄存器占用，支持多播 (同一数据→多个SM)。

- **核心机制**: DMA warp (TMA_load非阻塞)与Compute warpgroup (wgmma Tensor Core)在SM内部微算子并发→instruction-level计算-通信重叠；Warpgroup约束 (4 warp×32线程=128线程协作, Hopper硬件强制)；4-stage Ping-Pong pipeline消除TMA latency bubble
- **实现**: CUTLASS CuTe DSL→nvcc→PTX (wgmma, cp.async.bulk, mbarrier)→SASS
- **实验环境**: H100 SM90; SonicMoE varlen-M Grouped GEMM: 128×128×64 tile, 4 pipeline stages, SMEM ~180KB/CTA, 75-85% HBM BW utilization
- **来源**: Q3.6, Task-Based Tensor paper (1736.1), TMA知识笔记 (12311.5)

#### 方法: ACS Out-of-Order Kernel Scheduling

**编译时**：每个kernel标注read/write memory segments (__read_segments__={{(input,N*sizeof(float)),(weight,W*sizeof(float))}}, __write_segments__={{(output,N*sizeof(float))}})。metadata嵌入kernel launch包 (PTX无需修改)。

**运行时Scheduling Window (N=32)**：
```
Window Module (CPU线程):
  while kernel_stream.not_empty():
      k = pop(kernel_stream)
      upstream = []
      for each kk in SW:  // O(segments²)地址范围重叠检测
          for seg_w in k.write_segments:
              for seg_rw in kk.{read,write}_segments:
                  if overlap(seg_w, seg_rw):
                      upstream.add(kk.kid)
      SW.insert(k, upstream)

  on_kernel_complete(kid):
      for each k in SW:
          k.upstream.remove(kid)
          if k.upstream.empty(): k.status = READY

Scheduler Module (多CPU线程, 各绑1 CUDA Stream):
  while True:
      if SW.has_ready():
          k = SW.pop_ready()
          cudaLaunchKernel(k, stream_id)
          // 多个READY kernel可在不同stream并发
```

**硬件方案 ACS-HW**：1KB调度窗口SRAM集成在GPU Command Processor中，kernel完成后硬件自动更新upstream list (N-1 cycle)，无需CPU参与。效果：Brax Ant (RTX 3060, 28 SM) GPU occupancy 34%→~100%；ACS-SW 1.87×, ACS-HW 2.19× speedup。

- **核心机制**: 编译时RW-segments内存访问标注+运行时sliding window O(segments²)依赖检测 (实际410-1640ns, 远小于kernel launch 5-20μs)+ACS-HW GP CP硬件调度窗口 (kernel完成→硬件自动更新upstream, N-1 cycle)
- **实现**: ACS CUDA runtime (SW) / GPU Command Processor修改 (HW)
- **实验环境**: RTX 3060 28 SM; Brax Ant (DeepRL不规则图); Occ 34%→~100%; speedup 1.87× (SW)/2.19× (HW)
- **来源**: Q3.3, Q3.6, OoO知识笔记 (2407.6), ACS paper_secs (2060.4)

#### 方法: TPU VLIW指令编排与Scratchpad静态分配

**TPU VLIW bundle** (以TPUv4为例, 推测)：
```
VLIW[t]: {
  scalar:  address_calc_for_next_tile,     // 地址计算
  vector:  activation_fn(prev_tile_output), // VPU向量ALU
  matrix:  matmul(current_tile_A, current_tile_B), // MXU 128×128 systolic array
  dma:     prefetch(next_tile_A, next_tile_B)      // CMEM↔HBM传输
}
// → 4类操作在同一cycle内并发执行
// → 编译器负责数据依赖分析和指令调度，无运行时开销
```

**Scratchpad Memory (CMEM) 编译时静态分配**：
```
for tensor in topological_order(graph):
    live_tensors = {t: t.live_at(tensor.time)}
    for buf in cmem_buffers:
        if buf.last_user_time < tensor.first_use_time:
            reuse(buf, tensor); break
    if not reused:
        allocate_new_buffer(tensor.size)
// 编译时即可确定: 无运行时cache miss，延迟完全可预测
```

**TPU并发模型**：不依赖运行时多kernel并发（不像GPU CUDA stream/MPS）。并发表达为：(1) VLIW指令级并行 (同bundle内scalar/vector/matrix/DMA并发)；(2) SPMD数据并行 (多TPU chip执行相同program不同数据分片)；(3) 计算-通信Overlap (编译器插入异步collective-permute并调度计算与之重叠)。

- **核心机制**: VLIW bundle (4类操作同cycle并发) + scratchpad memory编译时全静态分配 (liveness analysis buffer sharing→零cache miss→确定性延迟) + SPMD collective-permute+compute overlap
- **实现**: XLA TPU backend; TPU ICI 2D/3D torus硬件加速通信
- **实验环境**: TPU v4; GSPMD 50-62% compute util on 2048 TPUv3; 关键约束: 所有tensor shape编译时确定 (MoE需padding expert capacity→token dropping)
- **来源**: Q3.6, GSPMD知识笔记 (41.8)

---

## 方法间关系

### 替代关系

- **FlashFuser ←→ MPK**: 两者都做GPU上的极致融合，但策略互补——FlashFuser聚焦GEMM链compute-intensive fusion (利用DSM扩展片上缓存, 五级存储层次建模)，MPK聚焦全模型mega-kernel fusion (tGraph SM级任务编排, 跨算子软件流水线)。FlashFuser适合MoE FFN/DiT backbone等GEMM密集型子图，MPK适合含跨算子通信(AllReduce)的端到端模型。
- **Triton ←→ CUTLASS CuTe**: Triton简化开发 (Python DSL, autotuner自动搜索, MLIR自动lowering) 但受限于MLIR中间表示——无法表达warp-specialized异步调度/Ping-Pong pipeline/cluster-level synchronization。CuTe直接控制异步硬件精确timing (TMA+wgmma+mbarrier) 但开发复杂度高 (C++ template, 编译时全静态配置)。选择取决于性能需求 vs 开发效率。
- **HyTiS ←→ Infera Zero-Tuning**: HyTiS通过实测profiling+两级tile scheduling达到最优 (但需~19min profiling per device)；Infera完全跳过profiling，静态资源约束推导 (编译时间-2-3数量级 vs Ansor) 但精度低于实测方法。对稳定硬件平台选HyTiS，对快速迭代或新硬件选Infera。
- **XLA/GSPMD ←→ MLIR/IREE**: 两者都做编译器级自动并行。GSPMD侧重SPMD大规模分布式 (TPU 512-device 2D torus)，sharding推导全自动但限于XLA生态；MLIR/IREE侧重多后端可移植 (GPU CUDA/Vulkan + NPU + PIM)，通过dialect分层提供更灵活的并发表达 (async.execute+scf.forall+gpu.launch) 但自动并行化不如GSPMD成熟。
- **CUDA Graph (静态DAG) ←→ ACS (动态OoO)**: CUDA Graph适合固定结构的推理图 (DiT denoising loop, static MoE routing)→静态化后eliminate launch overhead; ACS适合input-dependent不规则DAG (DeepRL物理仿真, 动态expert routing)→运行时滑动窗口自动发现并发。CUDA Graph在动态图中构建开销占47%总时间，此时ACS更优。

### 互补关系

- **FlashFuser + MPK**: 可在编译栈不同层次协同——FlashFuser在GEMM链层面做DSM-based fusion (中间58% memory access消除)，MPK在全模型层面做tGraph-based跨算子并发 (计算-通信重叠)。先FlashFuser融合局部GEMM链→再MPK编排融合后的fewer large kernels为mega-kernel。
- **Triton + CUDA Graph**: Triton提供单kernel内CTA级并发 (program_id grid) 但不能跨kernel编排；CUDA Graph将多个Triton kernel的launch序列静态化为单次GPU端提交。PyTorch 2.0 Inductor已整合两者。
- **HuntKTm Stream Scheduler + HyTiS Auto-tuning**: HuntKTm做编译时concurrency extraction (DFG→stream分配) 但不调优tile size；HyTiS做tile size调优但不考虑多kernel并发资源竞争。两者结合：先HuntKTm确定多stream调度计划→再HyTiS对每个stream内kernel做per-kernel tile tuning (作为joint tuning的approximation)。
- **MLIR async dialect + IREE HAL**: MLIR async.execute提供硬件无关的并发语义标注→IREE HAL将并发语义映射到具体硬件command buffer (CUDA stream/Vulkan async compute)。无barrier的dispatches由GPU硬件自动并发。
- **GSPMD + XLA TPU VLIW**: GSPMD在HLO层做SPMD partition (插入collective→各device独立program)→XLA TPU backend做VLIW指令编排 (scalar+vector+matrix+dma同bundle并发)。两层并发互补：device级 (SPMD) + 指令级 (VLIW)。

### 依赖关系

- **Warp Specialization + TMA → 依赖H100 SM90硬件**: TMA (Tensor Memory Accelerator) 和 wgmma (warpgroup matrix multiply-accumulate) 是Hopper架构独有特性。Ampere (A100) 只能用cp.async (仍需线程参与, 占用寄存器)。编译器codegen必须根据目标SM版本选择路径 (TileLang T.Pipelined的架构自适应策略)。
- **FlashFuser DSM Fusion → 依赖H100 Thread Block Cluster**: DSM是H100新引入的cluster内SM互联通道，A100无此硬件能力。FlashFuser的整个编译流水线 (dsm_comm原语+Dataflow Analyzer) 绑定H100 SM90架构。
- **GSPMD SPMD Partition → 依赖XLA HLO IR**: GSPMD的sharding propagation以HLO instruction semantics为基础 (每op的sharding推导规则硬编码)，不可脱离XLA生态。
- **CANN GE/IRFusion → 依赖Ascend Da Vinci架构**: Cube Unit 16×16脉动阵列固定数据流模式决定了融合形状的对齐约束；Vector Unit 32-lane SIMD决定了elementwise操作的并行度。CANN的融合规则与华为硬件深度绑定。
- **Groq/Cerebras确定性编译 → 依赖SRAM-only架构**: 消除HBM使编译器可全静态决定内存布局 (无cache miss→确定性延迟)。若引入HBM/DDR等不可预测延迟的存储层，确定性假设失效。
- **MegaScale-MoE Holistic Scheduling → 依赖手动算子拆解**: 当前调度为hand-tailored (不依赖torch.autograd monolithic反向)。自动化需要编译器级别的macro-module抽象 (论文§7未来方向)。

---

## 本层不确定性

1. **StableHLO while op的GPU异步支持**: XLA GPU backend上while body内部的异步算子 (async all-to-all) 支持状态笔记未明确说明。`xla_gpu_enable_latency_hiding_scheduler`为实验性flag，生产就绪度未知。

2. **NPU/Ascend编译框架细节缺失**: vault中CANN GE/IRFusion的具体融合算法 (边界决策规则、cost model)、IR dialect设计 (GE IR→Ascend指令lowering的pass pipeline)、AscendC精确ISA格式 (Cube Unit MAC数量、tile buffer大小) 笔记未覆盖。对Ascend NPU的代码生成分析基于架构推断。Ascend是否支持GPU式多kernel并发执行 (类似CUDA stream) 笔记未明确说明。

3. **TPU/加速器内部编译器架构**: Groq Compiler和Cerebras SDK的具体IR设计、编译pass pipeline、编译时间数据笔记未覆盖。TPU VLIW bundle的精确opcode格式和位宽属于Google闭源信息。关于确定性编译和空间映射的论述部分基于公开信息推断。

4. **DiT/Video模型的编译特殊性**: vault笔记聚焦MoE和多模态kernel优化。DiT的iterative denoising loop的编译优化 (Difflow以外的方法) 和Video模型temporal attention的communication-computation overlap自动生成能力笔记未充分覆盖。

5. **Joint multi-kernel auto-tuning**: HyTiS、Infera等方法针对单kernel tile tuning。多kernel并发的联合调优 (如何平衡各kernel的tile size、register分配以最大化总体throughput) 笔记未明确说明——这仍是open research problem。TVM AutoScheduler有初步的joint cost model但笔记未详细描述。

6. **跨框架benchmark数据**: 相同MoE/DiT/多模态/Video模型在XLA vs TVM vs Triton vs TorchInductor上的编译时间和执行延迟定量对比笔记未提供。实验笔记目录主要覆盖serving系统benchmark而非编译框架横向对比。

7. **多模态模型异构编码器的编译效果**: MLIR+IREE理论上支持GPU+NPU异构编译，但vault中缺乏LLaVA/BLIP-2在GPU+ANE上的编译实验数据和IR lowering细节。

8. **Video模型pmap/shard_map的详细知识笔记缺失**: vault中缺乏JAX pmap/shard_map在单GPU Video模型上的IR lowering细节。Q3.1的Video IR分析部分基于Difflow的video evaluation和MLIR通用parallel semantics推断。

9. **SambaNova/Groq/Cerebras编译时间**: 笔记未提供这些加速器编译器对大型模型 (如70B+) 的编译时间数据——预计可能为数十分钟到数小时。

10. **IREE HAL在GPU上的实际concurrent dispatch支持度**: 笔记未量化IREE有多少比例的实际dispatch可并发执行，以及硬件限制 (Vulkan async compute queue数量等)。

[HORIZON_SUMMARY_DONE] L3
