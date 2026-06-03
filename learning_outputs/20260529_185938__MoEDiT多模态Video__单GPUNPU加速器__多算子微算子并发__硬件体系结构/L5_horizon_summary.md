# L5: 硬件架构 — 水平分类总结

## 问题覆盖概览

| Q-ID | 覆盖方法数 | 关键方法 |
|------|-----------|----------|
| Q5.1 | 7 | SIMT+Tensor Core异构组织、Systolic Array Weight Stationary、3D Tensor PE Array (GyRot)、FEATHER NEST+BIRRD可重构Dataflow、Warp Specialization、MHE-TPE跨PE协同编码、CKE/ACS并发调度 |
| Q5.2 | 8 | GPU Memory Hierarchy四级数据流、Warp Scheduler推测调度、TMA异步DMA+mbarrier同步、三级异步数据供给流水线、双缓冲范式、Memory Coalescing、Warp-Group Specialization、FlashFuser DSM Fusion |
| Q5.3 | 10 | 2D Mesh NoC (RDN)、Crossbar NoC (MTIA 2i)、NVLink+NVSwitch全互联、AMD Infinity Fabric Full Mesh、Active Interposer+DFBM死锁自由、SoW晶圆级2D Mesh、MCM-GPU层次化Crossbar、CoCoTree、MixNet可重构光电互联、3D NMP 2D Mesh |
| Q5.4 | 9 | Dataflow Architecture (StreamTensor/Versal)、GPU Dataflow (Kitsune)、Ascend DaVinci Tile-based NPU、Focus Modular SA Add-on、IANUS NPU-PIM统一架构、CuTe DSL MoE kernel开发、Tandem Processor、DRX多加速器链式、RTL→FPGA→ASIC设计流程 |
| Q5.5 | 10 | GPGPU-Sim/Accel-Sim、gem5+DRAMsim3/Ramulator2、Mess Simulator分析型反馈、Timeloop+Accelergy、CACTI 7.0、BookSim 2.0、McPAT+GPUWattch+AccelWattch、自研MoE Multi-Chiplet Simulator、Roofline Model、Scale-Sim |
| Q5.6 | 13 | µShare intra-SM co-locating、Bullet prefill-decode时空并发、MuxWISE bubble-less multiplexing、Splitwise phase splitting、MoE-CAP评估框架、RPU低batch专用架构、QuantCache DiT视频生成加速、Q-VDiT/S²Q-VDiT量化、Tilus跨精度MatMul、KernelEvolve跨平台评估、PAPI PIM能耗建模、H2-LLM异构die stacking、Hardwired-Neuron LPU稀疏驱动低功耗 |

---

## 按实验环境分类

| 分类 | 方法 | 具体方法描述 | 硬件平台 | Benchmark | 实现框架 | 来源 |
|------|------|-------------|----------|-----------|----------|------|
| **GPU/NVIDIA** | SIMT+Tensor Core异构Pipeline | SM内4 sub-cores，每sub-core含Warp Scheduler+CUDA Core×16+Tensor Core×1+TMA engine；3-stage pipeline overlap（TMA load + CUDA dequant + TC MMA）并发；Warp Specialization实现producer/consumer warp分工 | H100 (Hopper SM90), A100 (Ampere SM80) | LiquidGEMM W4A8 GEMM kernel latency | CUTLASS 3.x, CuTe DSL, TileLang | Q5.1, Q5.2, Q5.4 |
| **GPU/NVIDIA** | Warp Specialization (Warp级微算子流水线) | DMA Warp（1 warp=32 threads，仅thread 0调用TMA）与Compute Warpgroup（4 warps=128 threads，执行WGMMA）通过mbarrier异步同步；warp scheduler time-multiplexing交替执行；Pipelining depth PIPE=3确保TMA延迟完全隐藏 | H100 (Hopper SM90) | FlashAttention-3 kernel latency | CUTLASS 3.x `sm90_mma_tma_gmma_rs_warpspecialized`, TileLang (P.Tiled调度原语) | Q5.1, Q5.2 |
| **GPU/NVIDIA** | Hierarchical Software Pipeline (三级异步数据供给) | Level 1 (Inter-tile): cp.async/TMA双缓冲搬运Global→SMEM；Level 2 (Intra-tile): CUDA Core dequant寄存器双缓冲；Level 3 (Compute): Tensor Core异步WGMMA；三级全重叠ALU利用率66%，TC利用率保持cuBLAS的71.6% | A100, H100 | W4A8 GEMM / FlashAttention kernel | CUTLASS, CuTe, CUTE | Q5.2 |
| **GPU/NVIDIA** | FlashFuser DSM Fusion | 利用H100 DSM（SM-to-SM Crossbar NoC, L1.5层, 0.2-3.6MB池）将fused GEMM chain中间数据路径从SMEM→HBM→HBM→SMEM改为SMEM→DSM→SMEM，减少58% global memory access | H100 (Hopper SM90) | fused GEMM chain latency | DSM API (dsm_shuffle, dsm_all_exchange) | Q5.2 |
| **GPU/NVIDIA** | GPU Dataflow Execution (Kitsune) | Inter-CTA Ring Queue（L2-resident ring buffer，通过GPU atomics同步）实现SM间producer-consumer数据传递；Modified Grid Scheduler增加SIMT/TENSOR双arbiter实现异构CTA SM co-residency；A100实测54 queue、128-256KB payload时aggregate BW达2 TB/s | A100 (108 SM) | 5个DL应用 | PyTorch Dynamo → CUDA spatial pipeline API | Q5.4 |
| **GPU/NVIDIA** | CKE (Concurrent Kernel Execution) | Thread Block Scheduler的Leftover Policy（仅队头kernel block可调度）+ Most-Room Policy（选容纳最多block的SM）；Concurrent-Isolated（两kernel block在不同SM，无竞争）vs Concurrent-Colocated（同SM混合，L1/functional unit竞争可致1.24×~96.1×退化） | A100, H100 | CUDA microbenchmarks + DL kernels | CUDA Streams, MPS, CUDA Graphs | Q5.1, Q5.2 |
| **GPU/NVIDIA** | µShare Intra-SM Scattered Co-locating | 通过blocksize参数间接影响GigaThread Engine的left-over调度实现SM内kernel交错执行；非侵入式（无需修改硬件或kernel代码）；10个MLPerf+PyTorch模型Azure production trace，SLO=200ms | A40 (84SM), A800 (108SM) | MLPerf 10-model, Azure INFless production trace | CUDA 11.8/12.1, PyTorch | Q5.6 |
| **GPU/NVIDIA** | Bullet Prefill-Decode动态SM分区 | 根据系统负载实时调整prefill SMs：burst时全GPU prefill，恢复后平衡分区；Nsight Systems验证并发时SM active cycles 86.2%（+11.2% vs SGLang），Tensor Core utilization +11.8%，memory-BW utilization +19.3% | A100, H100, H20; 8×A100 NVLink, 8×H100, 8×H200 | ShareGPT, Azure-Code, arXiv-Summary; Llama3.1-8B/70B + Qwen3-235B MoE | SGLang v0.4.6, FlashInfer v0.2.7, CUDA 12.4 | Q5.6 |
| **GPU/NVIDIA** | MuxWISE Bubble-less Prefill-Decode Multiplexing | Operator级intra-GPU multiplexing消除prefill-decode间的bubble；TBT SLO=50ms(Llama-8B)/100ms(Llama-70B)；对比SGLang Chunked-prefill、NanoFlow、LoongServe | 8×A100-80GB NVLink 600GB/s; 8×H100; 8×H200 | Llama-8B/70B + Qwen3-235B MoE | SGLang 0.4.10post2, CUDA 12.8 | Q5.6 |
| **GPU/NVIDIA** | TMA + mbarrier异步数据搬运与同步 | TMA（Hopper+）：单线程发起async bulk tensor传输(1D-5D)，支持multicast、descriptor-based addressing；mbarrier：硬件加速barrier替代__syncthreads()，producer arrive→consumer wait；对比cp.async (Ampere): 需warp内所有线程参与地址计算，经过寄存器增加register pressure | H100 (Hopper), A100 (Ampere) | FlashAttention / MoE GEMM kernel | CUDA `cuda::memcpy_async`, `<cuda/barrier>`, PTX `cp.async.bulk` | Q5.1, Q5.2 |
| **GPU/NVIDIA** | Thread Block Scheduler推测调度 | Leftover Policy（仅队头kernel block可调度，不可抢占）+ Most-Room Policy（选能容纳最多block的SM）；GSP firmware实现，用户不可直接编程 | 所有NVIDIA GPU | multi-kernel workloads | GSP firmware (闭源) | Q5.1, Q5.2 |
| **GPU/NVIDIA** | MIG (Multi-Instance GPU) | 将单GPU物理分区为最多7个独立GPU实例，各自有专用SM/L2 cache/内存带宽的硬件隔离；提供硬件级故障隔离和QoS保证 | A100, H100 | multi-tenant inference | GPU System Processor配置 | Q5.1, Q5.2 |
| **GPU/NVIDIA** | RPU低batch专用推理架构(提案) | Decode阶段TDP仅34%（vs prefill 90%），memory BW利用率仅32%；提出HBM-CO（容量优化型HBM）降低rank/bank/subarray冗余结构，保留内部BW架构同时提升bandwidth per dollar 2.4×；解耦memory/compute/network pipeline | H100 baseline + RPU sim | Llama3-405B dense-linear kernels | NVML功耗测量, Roofline建模 | Q5.6 |
| **GPU/NVIDIA** | Splitwise Phase Splitting | Prefill与Decode分离到不同GPU执行，通过KV-cache跨GPU传输；<7% KV transfer overhead；200/400Gbps互联带宽 | A100, H100 | Coding trace | CUDA, KV-cache transfer protocol | Q5.6 |
| **NPU/Ascend** | Ascend DaVinci Tile-based Architecture | AIC（AI Core: Cube Unit矩阵计算+Vector Unit+Scalar Unit）+ AIV（AI Vector: 元素操作）+ MTE（Memory Transfer Engine: 异步DMA）三单元独立调度并行；Tile-based控制粒度 vs GPU SIMT warp级；动态shape场景MFU从53%降至30-47% | Ascend 910B | LLM推理 | torch-npu (PyTorch backend) → CANN compiler stack | Q5.4 |
| **NPU/TPU** | Weight Stationary Systolic Array | 32×32 PE（FP16 multiply/FP32 accumulate），weights预加载到PE不动，inputs水平流经array，output-stationary外循环；GEMM tiling: m=1024, n=32, k=32 subtile | Focus baseline accelerator, TPU系列 | VLM推理 | SystemVerilog RTL, SCALEsim-v2 | Q5.1, Q5.4 |
| **加速器** | FEATHER NEST+BIRRD可重构Dataflow | 两阶段计算: Phase 1 Local Temporal Reduction（PE内local register）+ Phase 2 Interleaved Spatial Forwarding and Reduction（PE rows轮流空间归约）；BIRRD butterfly网络在reduction中重排序oActs到next-layer layout，支持(dataflow, layout) co-switching；RIR核心insight: 重排序post-reduction oActs而非直接转换iActs layout | FEATHER accelerator (TSMC 28nm) | GEMM + irregular-shaped workloads | SystemVerilog RTL, Layoutloop (增强Timeloop) | Q5.1 |
| **加速器** | GyRot 3D Tensor PE Array | 8×8×32 3D PE（2048 parallel ops/cycle）；Output-stationary systolic + per-PE 32-way INT4 dot product + adder tree；Inter-group accumulation流水线(G=32)；面积0.35mm²(16.6%)，功耗528.64mW(71.4%) | GyRot accelerator (TSMC工艺) | INT4 GEMM | 多bank memory设计（Input 64KB+8KB, Weight 64KB+4KB） | Q5.1 |
| **加速器** | MHE-TPE跨PE协同混合精度编码 | 三阶段统一计算范式: Bit-Slice Encoding → Vector PPs Generation → Cross-Dimensional Reduction；时域multiplicand映射+空域multiplier映射分解，跨PE共享vector PP lookup table消除PE间冗余PPs，PPs减少一半；支持INT2~INT32混合精度；operand bit-width减半→4× throughput | MHE-TPE (UMC 22nm, 1GHz) | 混合精度GEMM | Synopsys Design Compiler综合 | Q5.1 |
| **加速器** | Focus Modular SA Add-on | SEC (Semantic Concentrator): a-way parallel max units + a-way streaming bubble sorter做token-level top-k pruning，1.9% area；SIC (Similarity Concentrator): vector-level similarity compression，0.8% area；仅增加2.7%面积和0.9%功耗，换取4.47× vs dense SA、7.90× vs A100 | Custom 32×32 SA, TSMC 28nm, DDR4 64GB/s | VLM (cross-modal attention) | SystemVerilog RTL, Synopsys DC, SCALEsim-v2 | Q5.4 |
| **加速器** | IANUS NPU-PIM统一架构 | PIM Access Scheduling将NPU和PIM命令交错调度，有效内存带宽2.4 TB/s（9-10× GDDR6外部BW）；FPGA原型: Xilinx VCU118 + GDDR6-AiM PIM芯片通过FMC连接 + NPU cycle-accurate simulator | Xilinx VCU118 FPGA + real PIM chips | GPT-2推理（6.2× vs A100） | FPGA原型 + NPU simulator | Q5.4 |
| **加速器** | StreamTensor Dataflow Architecture | Linalg IR → dataflow conversion → dataflow kernel fusion → HLS synthesis → FPGA bitstream；多个kernel以空间流水线（spatial pipeline）方式并发，FIFO流式传递中间数据避免off-chip DRAM往返；Kernel→SRAM→PE数据流路径 | AMD Versal AI Engine, SambaNova SN40L, IBM AIU | LLM decode | StreamTensor Compiler (HLS backend) | Q5.4 |
| **加速器** | Tandem Processor (GEMM+可编程协处理器) | Systolic Array 32×32 + 32-lane Tandem SIMD协处理器；覆盖100%非GEMM算子，消除CPU fallback；Verilog RTL实现，Synopsys DC+IC Compiler，GF 65nm+FreePDK 15nm | Verilog RTL, 1GHz (FreePDK 15nm) | VGG-16, ResNet-50, BERT, GPT-2 | RTL → DC → IC Compiler | Q5.4 |
| **加速器** | H2-LLM Hybrid Bonding异构加速器 | HB-NMP PE + centralized processor的异构die stacking；通过hardware-dataflow co-exploration优化低batch LLM推理；Chisel RTL→Synopsys DC(40nm)，MAC能量0.974-1.365 pJ/MAC | Custom accelerator (40nm) | OPT 6.7B, LLaMA3 8B, PaLM 8B | Timeloop+Accelergy+Ramulator2扩展 | Q5.5, Q5.6 |
| **加速器** | Hardwired-Neuron LPU | HN Array通过4/128 MoE expert sparsity驱动极低电路活动实现超低功耗；大die area但通过稀疏性补偿功耗 | HN Array chip | MoE LLM (128 experts) | Metal embedding工艺 | Q5.6 |
| **互联系统** | NVLink+NVSwitch全互联交换网络 | NVLink 5th-gen (B200): 900 GB/s单向/link；NVSwitch (TSMC 4N, 64 port, 3.2 TB/s双向总带宽): all-to-all non-blocking switching，任意GPU pair单跳<1μs；SHARP in-network reduction (400 GFLOPS FP32)在switch内执行reduce，数据不往返GPU | H100/B200 DGX | MoE all-to-all token exchange, all-reduce | NCCL Ring/Tree/NVLS | Q5.3 |
| **互联系统** | AMD Infinity Fabric Full Mesh | 8 GPU全互联，每GPU 7条IF Link直连其他GPU（无需外部switch芯片）；任意pair单跳<1μs；编程透明（Iris框架device-side API）；缺点：无in-network reduction，O(N²) link数限制扩展到>8 GPU | MI300X (CDNA3) | GPU间通信 | Iris框架 (device-side load/store API) | Q5.3 |
| **互联系统** | SambaNova RDN 2D Mesh NoC | 三fabric分离（Vector packet-switched + Scalar packet-switched + Control circuit-switched）；credit-based per-hop flow control；Sequence ID重排序支持many-to-one乱序到达；动态2D dimension order routing + 软件配置static flow routing | SN40L RDU (TSMC 5nm) | 多kernel并发数据流 | 编译器Place-and-Route预计算 | Q5.3 |
| **互联系统** | Meta MTIA 2i Crossbar NoC | 64 PE(8×8)通过crossbar-based NoC连接SRAM/Memory/Host；Non-blocking架构保证任意initiator-target pair不被其他pair阻塞；Leaky-bucket traffic shaping + packet fragmentation做拥塞控制；BW提升3.3×（vs MTIA 1），die面积仅增1.13× | MTIA 2i (TSMC) | 推荐推理 | RISC-V Quad-Core Control | Q5.3 |
| **互联系统** | Active Interposer + DFBM死锁自由 | Chiplet内部4×4 mesh XY routing → Boundary Router → Active Interposer 4×4 mesh + 3VN + per-VN 2/4 VC + virtual cut-through；DFBM credit-aware admission control从根源消解跨chiplet循环通道依赖；支持"plug-and-play chiplet"标准化 | 学术方案 (gem5+Garnet cycle-accurate) | PARSEC benchmark suite, synthetic traffic | UCIe PHY, DFBM bridge module | Q5.3 |
| **互联系统** | SoW晶圆级2D Mesh | TSMC SoW: 24 compute dies + 96 HBM dies集成在单晶圆(>200,000mm²)；LSI垂直terabit-level带宽；XSR SerDes水平1.7 TB/s D2D, 200ns/hop；NUMA效应: local 300ns vs remote 3100ns (~10×差距) | TSMC SoW (晶圆级) | MoE serving (200B-1000B参数单晶圆容纳) | LSI + XSR SerDes | Q5.3 |
| **互联系统** | MCM-GPU层次化Crossbar | Concentrated hierarchical crossbar: 每chiplet与其他所有chiplet直连(单跳)；768 GB/s inter-chiplet BW, 32 cycles/hop；NUMA缓解: L1.5 cache专门缓存remote data + First-touch page allocation + Distributed CTA scheduling | 学术方案 (LRM-GPU) | GPU通用计算 | BookSim 2.0 + GPGPU-Sim联合仿真 | Q5.3 |
| **互联系统** | CoCoTree层次化二叉树 | Hierarchical binary tree interconnect + in-network computation支持集合通信（reduce/broadcast）；多root并行collective；延迟O(log N) | DIMM-PIM (UPMEM) | 集合通信 | PIM集成 | Q5.3 |
| **模拟器** | GPGPU-Sim / Accel-Sim生态 | Cycle-level GPU微架构模拟：SM/Warp/Thread级别建模，warp scheduler+scoreboard+SIMT stack+Tensor Core+L1/L2+HBM；支持SASS级别指令追踪；GPUWattch/AccelWattch集成功耗建模；Most-Room Policy可提升concurrent kernel模拟精度 | NVIDIA GPU (Pascal/Volta/Turing/Ampere-like) | CUDA microbenchmarks + DL kernels | 开源 (https://github.com/gpgpu-sim) | Q5.5 |
| **模拟器** | gem5 + DRAMsim3/Ramulator2 | 全系统cycle-accurate CPU+GPU+加速器模拟；多内存后端可选(Simple/Internal DDR/DRAMsim3/Ramulator2/Mess)；关键发现: 内置memory model严重低估延迟(实际89-109ns vs gem5 14-100ns)，Ramulator2误差高达52% | ARM/x86 CPU + GPU + Accelerator | STREAM, LMbench, Google multichase | 开源 | Q5.5 |
| **模拟器** | Mess Simulator分析型反馈 | 基于实测BW-Latency曲线的反馈控制替代时序模拟；每1000次内存操作检查模拟BW与延迟一致性并调整；ZSim+Mess误差1.3%，gem5+Mess误差3%，加速13-15× | 任何有BW-Latency曲线的系统 | DDR4/DDR5/HBM2/CXL | 私有 | Q5.5 |
| **模拟器** | Timeloop + Accelergy | Map Space Explorer（混合启发式+随机）搜索最优tiling/loop order/spatial mapping；Accelergy可插拔组件能量模型库（MAC/SRAM/互连能量）；联合输出Latency+Energy+Area | 通用DNN加速器 | Layer-level GEMM/Conv | 开源 | Q5.5 |
| **模拟器** | 自研MoE Multi-Chiplet Simulator | Python事件驱动：每die建模H100-like(1000 TFLOPS FP16, 80GB HBM, 3.35 TB/s local BW, 1.7 TB/s D2D BW)；Global CP维护Expert Distribution Table+Cross-token Heatmap Cache；PDU管理expert数据缓存；8×H100 DGX实测验证误差<5% | Wafer-scale multi-chiplet GPU (Dojo 5×5, TSMC SoW 8×3) | DeepSeek V3, Kimi K2, Llama4 Maverick, Qwen3-235B (MoE decode) | 开源 (Apache-2.0) | Q5.5 |
| **模拟器** | Roofline Model三维约束分析 | T=max(Traffic_Min/BW, FLOPs_MM/Peak_MM, FLOPs_Vec/Peak_Vec)；Peak Throughput%=T/t_measured；Trainium上峰值吞吐从49%→61%(v1)、45%→59%(v2) | AWS Trainium | 多引擎并发(Matrix+Vector+Scalar) | AccelOpt分析框架 | Q5.5 |
| **跨平台评估** | KernelEvolve 跨平台Kernel评估 | 统一operator specification→Triton kernel→FaaS dispatch到H100/A100/MI300/MI350/MTIA v2i/v3五平台→平台特定profiler(NCU/MTIA Insight/ROCm Profiler)采集hardware utilization；MapIdTransform实现3.48× speedup | H100, A100, MI300, MI350, MTIA v2i/v3 | KernelBench L1/L2/L3(250 kernel problems), TritonBench(160 ATen operators) | FaaS平台 + Triton + 平台特定profiler | Q5.6 |
| **跨平台评估** | MoE-CAP三维评估框架 | CAP(Cost-Accuracy-Performance) radar diagram；修正MBU/MFU(传统方法忽略sparse expert activation，高估1.5-3×)；引入GPU+CPU+通信+SSD全系统成本模型 | 多GPU+CPU+SSD | MLPerf, LLM-Perf, Open-LLM-Leaderboard | — | Q5.6 |

---

## 按方法类别分类

| 分类 | 方法 | 具体方法描述 | 核心机制 | 来源 |
|------|------|-------------|----------|------|
| **数据流设计** | Weight Stationary Systolic | Weights预加载PE不动→inputs流经array→partial sums驻留accumulator | Weight stationary dataflow消除weight重复读取 | Q5.1 |
| **数据流设计** | Output Stationary 3D PE Array (GyRot) | Per-PE 32-way INT4 dot product→partial sum留在PE accumulator→inter-group accumulation流水线 | 3D PE减少PE数(64 vs 1024)换取更高per-PE density | Q5.1 |
| **数据流设计** | FEATHER可重构Dataflow* | Temporal reduction(PE内)→Spatial reduction(BIRRD butterfly)→RIR重排序post-reduction oActs | **任意dataflow parallelism+per-layer (dataflow, layout) co-switching** | Q5.1, Q5.3 |
| **数据流设计** | StreamTensor Spatial Dataflow* | Linalg IR→dataflow circuit→FIFO流式传递中间数据→**多kernel空间流水线并发** | 空间计算替代时序复用，消除off-chip DRAM往返 | Q5.4 |
| **数据流设计** | GPU三级异步Pipeline* | Level 1(Global→SMEM cp.async/TMA双缓冲) + Level 2(SMEM→Reg+Dequant寄存器双缓冲) + Level 3(Tensor Core WGMMA异步) | Memory-Copy-Compute三级全重叠 | Q5.2 |
| **计算单元组织** | SIMT+Tensor Core异构* | SM内CUDA Core(60 TFLOPS FP32)+Tensor Core(989 TFLOPS FP16 INT8)吞吐比≈16.5×；Warp scheduler选择ready warp利用warp-level parallelism隐藏延迟 | 异构pipeline overlap: **三个硬件单元(TMA+CUDA+TC)并发工作** | Q5.1 |
| **计算单元组织** | GyRot 3D Tensor PE? | 8×8 2D systolic × 32-way INT4 dot product(第三维) = 2048 parallel ops/cycle | Per-PE 32路并行乘法器+adder tree+integer dequantization | Q5.1 |
| **计算单元组织** | MHE-TPE跨PE协同编码? | Bit-Slice Encoding→Vector PPs Generation→Cross-Dimensional Reduction统一计算引擎 | 跨PE共享vector PP lookup table消除PE间冗余PPs | Q5.1 |
| **计算单元组织** | Ascend DaVinci Tile-based* | AIC(Cube+Vector+Scalar)+AIV(Element-wise)+MTE(DMA)三单元并行 | Tile级数据并行+AIC/AIV/MTE计算-计算+计算-访存双重overlap | Q5.4 |
| **计算单元组织** | Focus Modular SA Add-on* | SEC(token pruning)+SIC(vector compression)嵌入SA memory interface侧 | **Streaming concentration完全与GEMM时间重叠** | Q5.4 |
| **控制与调度** | Warp Scheduler推测调度! | 每SM 4个warp scheduler，每周期从ready warp中选一个发射指令；greedy-then-oldest策略(推测)；零成本warp切换(per-warp独立PC+RF) | Scoreboard stall→自动切换到另一ready warp→latency hiding | Q5.1, Q5.2 |
| **控制与调度** | Thread Block Scheduler! | Leftover Policy(仅队头kernel block可调度)+Most-Room Policy(选容纳最多block的SM)；GSP firmware实现 | 多kernel并发通过spatial sharing(SM间)+colocation(SM内) | Q5.1, Q5.2 |
| **控制与调度** | CKE (Concurrent Kernel Execution)* | Concurrent-Isolated(两kernel block在不同SM)vs Concurrent-Colocated(同SM混合)两种模式 | 队头kernel所有block调度完后下一个kernel block才能调度 | Q5.1, Q5.2 |
| **控制与调度** | ACS Inter-Kernel Scheduling! | **运行时检测小窗口内kernel依赖关系**→out-of-order风格动态kernel发射→最高2.19×加速(avg 1.56×) | ACS-SW(纯软件开源)+ACS-HW(GPU硬件增加轻量依赖检测单元) | Q5.1 |
| **控制与调度** | GPU Dataflow (Kitsune)! | Inter-CTA Ring Queue(L2-resident ring buffer, atomics同步)+Modified Grid Scheduler(双arbiter) | 异构CTA类型SM co-residency→Tensor Core和SIMT Core同时被不同算子使用 | Q5.4 |
| **控制与调度** | µShare Intra-SM Co-locating* | 通过blocksize参数间接影响dispatch unit的left-over调度**实现SM内kernel交错** | 非侵入式(无需修改硬件或kernel代码) | Q5.2, Q5.6 |
| **控制与调度** | Bullet动态SM分区* | **根据系统负载实时调整prefill SM数**：burst时全GPU prefill→恢复后平衡分区 | Prefill-decode时空并发→SM active cycles 86.2%(+11.2%) | Q5.6 |
| **控制与调度** | MuxWISE Bubble-less Multiplexing* | **Operator级intra-GPU multiplexing**消除prefill-decode间bubble | TBT SLO=50-100ms约束下的最优multiplexing策略 | Q5.6 |
| **访存体系** | GPU四级Memory Hierarchy | HBM(3.35TB/s, 400-800 cycles)→L2(50MB, 7-12TB/s, ~200 cycles)→L1/SMEM(228KB/SM, ~19TB/s, 20-30 cycles)→RF(256KB/SM, ~40TB/s, 0 cycles) | 容量-带宽权衡金字塔：每层通过tiling适配working set | Q5.2 |
| **访存体系** | TMA异步DMA | 单线程发起bulk tensor传输(1D-5D)，硬件独立完成，不经过寄存器；TMA multicast单次HBM读取广播到cluster内多SM | 释放线程资源→producer warp仅需1 thread→其余127 thread可用于计算 | Q5.1, Q5.2 |
| **访存体系** | mbarrier硬件同步 | Producer arrive(硬件写入)→Consumer wait(按需等待)→支持expect_tx字节级计数；与__syncthreads()不同：不阻塞所有线程，仅阻塞等待方 | Warp-group specialization的核心同步原语 | Q5.1, Q5.2 |
| **访存体系** | 双缓冲范式 | Buffer A/B交替: read_buffer被计算时write_buffer被异步加载下一个tile；扩展到inter-tile(SMEM双缓冲)+intra-tile(Reg双缓冲) | 数据搬运与计算从串行依赖转化为流水线重叠 | Q5.2 |
| **访存体系** | Memory Coalescing | LSU→L1→L2的事务合并：warp内32线程连续对齐访问→1个128B transaction→100% BW效率；scatter访问→up to 32 transactions→3-10%效率 | Coalescing效率因子γ=A/(M×128)，多算子并发需注意不同访问模式兼容性 | Q5.2 |
| **访存体系** | FlashFuser DSM Fusion! | SMEM→DSM(L1.5层, 3.6MB池, 4-8TB/s via SM-to-SM Crossbar)→SMEM替代SMEM→HBM→HBM→SMEM | 减少58% global memory access，缓解Memory Wall | Q5.2 |
| **访存体系** | Expert Weight Prefetching (HtoD Overlap)! | MoE expert权重offload到CPU memory→GPU计算当前expert时**HtoD engine异步预取下一个expert权重到double buffer** | 前提: expert batch size > 2^11 tokens/expert→GPU计算时间≥HtoD copy时间 | Q5.2 |
| **访存体系** | RPU HBM-CO容量优化型HBM? | 减少rank/bank/subarray降低容量驱动结构，保留内部BW架构；bandwidth per dollar和energy efficiency up to 2.4× over conventional HBM | 低batch decode场景memory BW利用率仅32%→容量优化比带宽优化更关键 | Q5.6 |
| **片上网络** | RDN 2D Mesh of Non-Blocking Switches? | 三fabric分离(Vector/Scalar/Control)+credit-based per-hop flow control+many-to-one sequence ID重排序+multicast fan-out | 多PCU间并行packet传输，非阻塞架构消除head-of-line blocking | Q5.3 |
| **片上网络** | Crossbar NoC (MTIA 2i) | Non-blocking全互联+Leaky-bucket traffic shaping+packet fragmentation+source端flow control | 任意initiator-target pair不被其他pair阻塞 | Q5.3 |
| **片间互联** | NVLink+NVSwitch | Non-blocking crossbar switch+all-to-all单跳<1μs+SHARP in-network reduction | MoE all-to-all token exchange直接通过NVLink+NVSwitch完成 | Q5.3 |
| **片间互联** | AMD Infinity Fabric Full Mesh | 8 GPU全互联无外部switch+每GPU 7 links>50 GB/s/link+单跳<1μs+编程透明 | 无in-network reduction，O(N²) link数限制扩展到>8 | Q5.3 |
| **片间互联** | Active Interposer + DFBM | Credit-aware admission control+shared deadlock buffer+UCIe标准化die-to-die接口 | 从根源消解跨chiplet循环通道依赖→plug-and-play chiplet | Q5.3 |
| **片间互联** | SoW晶圆级2D Mesh | LSI垂直terabit-level+XSR SerDes水平1.7TB/s D2D 200ns/hop | 单晶圆容纳完整200B-1000B MoE模型，消除跨节点网络瓶颈 | Q5.3 |
| **片间互联** | MCM-GPU层次化Crossbar | Concentrated hierarchical crossbar单跳768GB/s 32 cycles/hop+L1.5 cache+First-touch page allocation | 多chiplet间低延迟高带宽全互联 | Q5.3 |
| **硬件实现流程** | RTL→FPGA→ASIC设计流程 | SystemVerilog/Verilog RTL→Xilinx Vivado FPGA原型(250MHz)→Synopsys DC综合→IC Compiler P&R→Tape-out (TSMC 28nm/GF 65nm/FreePDK 15nm) | 学术加速器主流流程 | Q5.4 |
| **硬件实现流程** | HLS设计方法 | Xilinx Vitis libraries提供FFT/SVM/AES-GCM等HLS实现；StreamTensor: Linalg IR→HLS synthesis→FPGA bitstream | 深度学习加速器核心路径用RTL(性能)，非关键路径用HLS(开发速度) | Q5.4 |
| **硬件实现流程** | CuTe DSL GPU Kernel开发? | CuTe C++ templates→NVCC→PTX→SASS→SM90/SM100 hardware；支持warp-specialized pipeline、Ping-Pong scheduling、cluster-level sync | Triton无法表达的底层硬件特性(warp-specialized异步调度/Ping-Pong pipeline) | Q5.4 |
| **模拟与评估** | 全系统Cycle-Accurate模拟? | GPGPU-Sim(cycle-level GPU微架构)+gem5(全系统CPU+内存)+DRAMsim3/Ramulator2(cycle-accurate DRAM) | SM/Warp/Thread级别建模到DRAM bank/row-buffer/时序参数 | Q5.5 |
| **模拟与评估** | 分析型设计空间探索? | Timeloop(Map Space Explorer混合启发式+随机)+Accelergy(可插拔组件能量模型)+Roofline Model(三维约束) | Map space搜索最优tiling/loop order+组件级能量累加+峰值性能上界 | Q5.5 |
| **模拟与评估** | 硬件功耗面积建模? | McPAT(transistor activity factor)+CACTI 7.0(SRAM bitcell/decoder/sense amplifier)+GPUWattch/AccelWattch(GPU专用)+Synopsys DC(RTL综合) | 从标准单元库参数→晶体管级activity factor→模块级功耗面积 | Q5.5 |
| **模拟与评估** | MoE专用自研模拟器? | Python事件驱动multi-chiplet GPU sim+Expert Distribution Table+Cross-token Heatmap Cache+PDU prediction table+8×H100 DGX验证<5%误差 | 现有工具(cycle-accurate太慢，ASTRA-sim不支持single-GPU-like编程模型)均不适用 | Q5.5 |

---

## 分类详细问答

### 分类: 数据流设计与空间架构

#### 方法: Weight Stationary Systolic Array Dataflow

来自 Q5.1 的 Focus baseline systolic array 数据流设计（笔记: `knowledge_notes/硬件知识笔记/Systolic-array Accelerator (for VLM Inference).md`, score: 7028.3）：

**GEMM tiling 执行流程**:
```
GEMM: C[M×N] = A[M×K] × B[K×N]
Tiling 参数:
  - Input tile:  m × K  (m=1024), 从 input buffer (128KB) 流式读取
  - Weight tile: K × n  (n=32),   预加载到 PE 的 weight buffer (78KB)
  - Output tile: m × n,           在 output buffer (512KB) 中累积
  - K dimension: 每 k=32 子tile 迭代一次

Weight Stationary 内循环 (per PE[i][j]):
  PE_ij.accum += Σ_k input[i][k] * weight[k][j]
  → Weights 预加载后驻留在 PE 中
  → Inputs 从上方水平流入 systolic array
  → Partial sums 垂直向下流动/驻留

外循环（output stationary）:
  Output tile 驻留在 on-chip 直到全部 K 迭代完成
```

- **核心机制**: Weight stationary dataflow — weights 预加载到 PE 不动，inputs 流过 array，partial sums 在 accumulator 中驻留
- **实现**: SystemVerilog RTL, TSMC 28nm HPC+, SCALEsim-v2 cycle-accurate 模拟
- **实验环境**: 32×32 PE array, FP16 mul/FP32 acc, DDR4 4ch×16 64GB/s, 734KB on-chip SRAM
- **来源**: Q5.1, Q5.1_L5_answer.md, `knowledge_notes/硬件知识笔记/Systolic-array Accelerator (for VLM Inference).md` (score: 7028.3)

#### 方法: GyRot 3D Tensor PE Array (Output-Stationary 3D Dataflow)

来自 Q5.1 的 GyRot 3D PE array 数据流设计（笔记: `knowledge_notes/硬件知识笔记/3D Tensor PE Array (8×8×32 _ 三维张量处理单元阵列).md`, score: 5182.3）：

```
GyRot PE Array 组织:
  2D systolic: 8 rows × 8 columns (64 PEs total)
  3D tensor:   每个PE内部执行32-way INT4 dot product (第三维)
  → 总计 64 × 32 = 2048 parallel operations/cycle

Systolic dataflow (output-stationary):
  每个cycle:
    - Activation X 从 input buffer 水平广播到整行8个PE
    - Weight W 从 weight buffer 垂直广播到整列8个PE
    - 每个PE: (32 activation × 32 weight) → 32-way dot product → 13-bit result
    - Partial sum 留在 PE accumulator 中 (output-stationary)

Inter-group accumulation (group size G=32):
  Group 0 (32 elements): PE执行dot product → dequantize → 32-bit accumulator
  Group 1 (32 elements): PE执行dot product → dequantize → 同一accumulator
  ...所有groups处理完 → FP16 conversion → output buffer

面积/功耗分解 (TSMC工艺):
  PE Array (INT Tensor):      0.26 mm² (12.4%), 410.24 mW (55.4%)
  PE Array (Dequant + Accum): 0.09 mm² (4.2%),  118.40 mW (16.0%)
  Total PE Array:              0.35 mm² (16.6%), 528.64 mW (71.4%)
```

- **核心机制**: Output-stationary systolic + per-PE 32-way INT4 dot product — 以更高 per-PE density 换取更少 PE 数（64 vs 1024），配合 integer dequantization 实现最高 area/energy efficiency
- **实现**: 多 bank memory 设计（Input buffer 64KB+8KB metadata, Weight buffer 64KB+4KB metadata），per-cycle feed 8×8 PE array 需 8×32 INT4 activations + 8×32 INT4 weights + metadata
- **实验环境**: TSMC 工艺, GyRot accelerator
- **来源**: Q5.1, Q5.1_L5_answer.md, `knowledge_notes/硬件知识笔记/3D Tensor PE Array...md` (score: 5182.3)

#### 方法: FEATHER NEST + BIRRD 可重构 Dataflow

来自 Q5.1 的 FEATHER 可重构 Dataflow 设计（笔记: `paper_secs/secs_2025/22-FEATHER_.../A.-FEATHERs-Neural-Engine--NEST.md`, score: 1736.5）：

```
FEATHER的NEST（Neural Engine with Spatial forwarding and Temporal reduction）引擎采用两阶段计算：

Phase 1: Local Temporal Reduction
  每个PE内部的local register执行partial sums的时域归约
  → PE独立完成AH次local reduction
  
Phase 2: Interleaved Spatial Forwarding and Reduction
  PE rows轮流进行空间归约，时间复用reduction network
  → PE row发送locally reduced results到BIRRD
  → 其他PE rows同时继续本地计算（pipelining保障）
  → BIRRD（Butterfly Interconnect for Reduction and Reordering in Dataflows）
    在reduction过程中重新排序oActs到next-layer所需layout

BIRRD拓扑:
  - 2×log(AW) stages, 每stage AW/2 switches
  - 每个switch (Egg): 2-input × 2-output + adder
  - 支持4种操作: Pass(=) / Swap(×) / Add-Left(∓) / Add-Right(±)
  
RIR (Reordering In Reduction) 核心insight:
  "重排序post-reduction oActs，而非直接转换iActs layout"
  → 多个iActs在reduction中自然归约到更少oActs
  → oActs数减少 → 写bank数减少 → 天然避免bank conflict
```

FEATHER per-layer dataflow 切换流程：对每个 layer，Layoutloop（增强 Timeloop）搜索最优 (dataflow, layout) pair（搜索空间: dataflow 10^36 × layout 10^8），离线生成 BIRRD switch 配置，运行时从 Instruction Buffer 加载——NEST 执行 Phase 1 temporal reduction → Phase 2 BIRRD spatial reduction + reorder，oActs 以 new layout 写入 StaB Pong，无需单独重排序步骤。

FEATHER vs Fixed-Dataflow Systolic Array 对比：Fixed SA 的 K 维度映射到单 PE 列，水平 rigid reuse link 无法为不规则 shape 达到 full utilization；FEATHER 将 K 维度映射到整个 2D array，各 column 可独立映射不同 dataflow，BIRRD flexible reduction 提升不规则 shape 利用率。

- **核心机制**: Temporal reduction + Spatial reduction 两阶段；BIRRD butterfly 网络在 reduction 中并发重排序 oActs，支持 (dataflow, layout) co-switching
- **实现**: SystemVerilog RTL, Layoutloop (增强 Timeloop) 离线搜索最优配置
- **实验环境**: FEATHER accelerator, TSMC 28nm, BIRRD 面积约 0.8% 总 area
- **来源**: Q5.1, Q5.3, Q5.1_L5_answer.md, `paper_secs/FEATHER` (score: 1736.5)

#### 方法: StreamTensor Spatial Dataflow Architecture

来自 Q5.4 的 StreamTensor 数据流架构（笔记: `paper_secs/secs_2025/11-StreamTensor.../1.1-Dataflow-Architecture.md`, score: 3044.6）：

数据流架构将算子映射为 **dataflow circuit**（空间硬件电路），通过 FIFO 流式传递中间数据，避免 off-chip DRAM 往返。多个 kernel 以空间流水线（spatial pipeline）方式并发执行，而非 GPU 的时序复用——在任意时刻不同硬件资源可以**同时**被不同算子使用。

与 GPU 的 BSP（Bulk Synchronous Parallel）模型不同，数据流架构以 **synchronous dataflow** 模型运行，CTA 间通过 on-chip queue 显式通信来触发和节流执行。

StreamTensor 编译框架：Linalg IR → dataflow conversion → dataflow kernel fusion → HLS synthesis → FPGA bitstream。

```
数据流加速器完整数据路径:
  HBM/DRAM → L2/Global Buffer (On-Chip SRAM)
  → PE Array (Spatial 2D Grid of Processing Elements)
  → Accumulator Buffer (Reduction Tree)
  → L2 → HBM (Write-back)

  Kernel: MatMul (Dataflow Circuit)
  → [Stream FIFO] → Kernel: Activation (Dataflow Circuit)
  → [Stream FIFO] → Kernel: LayerNorm (Dataflow Circuit)
```

商业平台对应：AMD Versal（AI Engine 阵列）、SambaNova SN40L（可重构数据流 RDU）、IBM AIU。

- **核心机制**: 空间计算（spatial computation）替代时序复用 — kernel 以 dataflow circuit 形式空间映射，FIFO 流式传递中间数据
- **实现**: StreamTensor Compiler (Linalg IR → HLS synthesis → FPGA bitstream), AMD Versal AI Engine, SambaNova SN40L
- **实验环境**: AMD Versal, SambaNova SN40L, IBM AIU; LLM decode 阶段 memory-efficient
- **来源**: Q5.4, Q5.4_L5_answer.md, `paper_secs/StreamTensor` (score: 3044.6)

---

### 分类: 计算单元组织

#### 方法: SIMT Core + Tensor Core 异构组织 (NVIDIA GPU)

来自 Q5.1 的 GPU SM 异构计算单元组织（笔记: `knowledge_notes/硬件知识笔记/CUDA Cores vs Tensor Cores...md`, score: 6090.6; `paper_secs/secs_2025/3-AMALI.../1-Introduction.md`, score: 1403.6）：

```
SM (Streaming Multiprocessor) 内部结构 (H100, Hopper架构):
┌─────────────────────────────────────────────────────┐
│  L1 Cache / Shared Memory (256KB, configurable)     │
│  ┌─────────────────────────────────────────────┐   │
│  │  Sub-core 0              Sub-core 1          │   │
│  │  ┌─────────────────┐   ┌─────────────────┐  │   │
│  │  │ Warp Scheduler   │   │ Warp Scheduler   │  │   │
│  │  │ (1 per sub-core) │   │                   │  │   │
│  │  ├─────────────────┤   ├─────────────────┤  │   │
│  │  │ Register File    │   │ Register File    │  │   │
│  │  │ (16384 x 32-bit) │   │                   │  │   │
│  │  ├─────────────────┤   ├─────────────────┤  │   │
│  │  │ CUDA Cores (×16) │   │ CUDA Cores (×16) │  │   │
│  │  │ FP32/INT32 ALU   │   │                   │  │   │
│  │  ├─────────────────┤   ├─────────────────┤  │   │
│  │  │ Tensor Core (×1) │   │ Tensor Core (×1) │  │   │
│  │  │ FP8/FP16/INT8 MMA│   │                   │  │   │
│  │  ├─────────────────┤   ├─────────────────┤  │   │
│  │  │ SFU (×4)         │   │ SFU (×4)         │  │   │
│  │  └─────────────────┘   └─────────────────┘  │   │
│  │  Sub-core 2              Sub-core 3          │   │
│  └─────────────────────────────────────────────┘   │
│  TMA Engine (异步DMA，不占用CUDA Cores)              │
│  mbarrier (硬件加速同步，Hopper新增)                  │
└─────────────────────────────────────────────────────┘
```

**多算子/微算子并发支持机制** — 异构 pipeline overlap：

```
LiquidGEMM ImFP pipeline (W4A8 GEMM kernel 的并发编排):
  时间轴 →
  ┌──────────────────────────────────────────────────────┐
  │ TMA load tile_k      │ TMA load tile_{k+1}           │  ← TMA (异步DMA引擎)
  ├──────────────────────┼───────────────────────────────┤
  │ CUDA Dequant tile_{k-1} │ CUDA Dequant tile_k        │  ← CUDA Cores (SIMT)
  ├──────────────────────┼───────────────────────────────┤
  │ TC MMA tile_{k-2}    │ TC MMA tile_{k-1}             │  ← Tensor Cores (MMA)
  └──────────────────────┴───────────────────────────────┘
  → 三个硬件单元并发工作，形成3-stage pipeline
  → dequantization瓶颈: Φ_CUDA (60 TFLOPS) << Φ_TC (990 TFLOPS, INT8)
  → 当 T_DQ > T_MMA 时，CUDA Cores成为瓶颈，Tensor Cores闲置
```

- **核心机制**: CUDA Cores 与 Tensor Cores 吞吐差距约 16.5×；warp scheduler 选择 ready warp 执行（LRR 或 GTO 策略），利用 warp-level parallelism 隐藏延迟；Tensor Core HMMA 指令 modifier（16816 vs 1688）决定 tensor size 直接影响 CPI（A100: 16816 需 8 cycles、1688 需 4 cycles）
- **实现**: CUTLASS 3.x, CuTe DSL, TileLang；LiquidGEMM W4A8 GEMM 在 CUDA/CUTLASS 中实现
- **实验环境**: H100 (FP8: 1979 TFLOPS, FP16: 989 TFLOPS), A100
- **来源**: Q5.1, Q5.2, Q5.4, `knowledge_notes/CUDA Cores vs Tensor Cores` (score: 6090.6), `paper_secs/AMALI` (score: 1403.6)

#### 方法: Ascend DaVinci Tile-based Architecture

来自 Q5.4 的华为 Ascend NPU DaVinci 架构（笔记: `paper_secs/secs_2026/6-XY-Serve.../2.2-Coarse-grained-Tile-based-Architecture.md`, score: 681.1）：

```
Ascend 910B NPU Micro-architecture:
  MTE (Memory Transfer Engine): Data Movement Manager (异步DMA)
  
  AIC (AI Core):                        AIV (AI Vector):
    Cube Unit (Matrix Compute,            Vector Engine (Element-wise Ops)
    类似Tensor Core)
    Vector Unit (Vector Operations)
    Scalar Unit (Control Flow)
  
  HBM ↔ MTE ↔ AIC / AIV (并行执行)
```

控制模块详解：
- **AIC (AI Core)**: 矩阵计算核心，类似 Tensor Core，执行 GEMM/Conv 等密集型算子；与 AIV 的 memory access unit 可并行操作，实现 compute 与 data movement 的 overlap
- **AIV (AI Vector)**: 向量操作单元，处理 element-wise 操作（ReLU/LayerNorm/Softmax）；与 AIC 独立调度，可同时执行不同算子
- **MTE (Memory Transfer Engine)**: 管理 HBM ↔ 片上 buffer 数据搬运，支持异步 DMA
- **Cube Unit**: 脉动阵列风格矩阵乘法引擎，固定 tile 宽度；动态 shape 场景 MFU 从 53% 降至 30-47%

Tile-based vs SIMT 对比：Tile-based 控制粒度为固定宽度 tile 级，计算密度更高（专用数据路径，无动态分支开销），但动态输入处理需 Padding/Unpadding + Mask（开销大），编程复杂度高（需手动处理不规则 pattern）。笔记指出 NVIDIA 也正向 Tile-based 方向演进（引入 TMA 和 tile-based 编程模型）。

- **核心机制**: AIC/AIV/MTE 三单元并行 — 计算-计算 + 计算-访存双重 overlap；Tile-based 数据并行 + 固定宽度 tile 控制
- **实现**: torch-npu (PyTorch backend) → CANN (Compute Architecture for Neural Networks) → Ascend 910B
- **实验环境**: Ascend 910B, HBM, LLM 推理; MFU 30-53%
- **来源**: Q5.4, Q5.4_L5_answer.md, `paper_secs/XY-Serve` (score: 681.1)

#### 方法: Focus Modular SA Add-on (SEC + SIC)

来自 Q5.4 的 Focus 架构 modular add-on 设计（笔记: `knowledge_notes/硬件知识笔记/Focus Unit.md`, score: 173.2; `knowledge_notes/硬件知识笔记/Semantic Concentrator (SEC).md`, score: 161.1）：

Focus 架构展示了一种 **modular add-on** 模式：不修改 core compute pipeline，而是在 systolic array 的 memory interface 侧插入专用硬件模块。

```
Focus Unit Internals (3.21mm², 736mW):
  SEC (Semantic Concentrator): Token-Level Pruning
    - Importance Analyzer: a-way parallel max units 处理 attention SoftMax cross-modal scores
    - a-way Streaming Bubble Sorter: 级联比较器链增量 top-k selection
    - Offset Encoder: sliding window 记录保留 token 空间坐标
    - 1.9% Area
  
  SIC (Similarity Concentrator): Vector-Level Compression
    - 0.8% Area
```

SEC 子模块详解：
- **Importance Analyzer**: 并行 max units 处理 attention SoftMax 输出的 T×M cross-modal attention scores；a 个并行 max unit（每 cycle 处理 a 个 attention score）；支持 Parallel (spatial) 和 Orthogonal (temporal) 两种 dataflow stream
- **a-way Streaming Bubble Sorter**: a-way bubble sorter 做增量 top-k selection；**完全与 image attention GEMM 重叠**——GEMM 耗时 M·(M+T)·h·n/(a·b) cycles >> sorter 耗时 M·a·k cycles
- **Offset Encoder**: sliding window 记录保留 token 空间坐标，流式输出到 SIC，不阻塞后续 pipeline

- **核心机制**: Modular add-on — SEC 的 top-k sorter 不占据 critical path，与 GEMM 计算**完全时间重叠**；仅增加 2.7% 面积和 0.9% 功耗，换取 4.47× vs dense SA、7.90× vs A100
- **实现**: SystemVerilog RTL, TSMC 28nm (N28HPC+), Synopsys Design Compiler 综合, SCALEsim-v2 cycle-accurate 模拟, Memory Compiler 生成 on-chip SRAM；开源: https://github.com/dubcyfor3/Focus
- **实验环境**: 32×32 PE SA, FP16 mul/FP32 acc, DDR4 64GB/s, 757 MHz (target), SS corner (0.81V, 125°C)
- **来源**: Q5.4, Q5.4_L5_answer.md, `knowledge_notes/Focus Unit` (score: 173.2)

#### 方法: MHE-TPE 跨 PE 协同编码

来自 Q5.1 的 MHE-TPE 混合精度映射方法（笔记: `paper_secs/secs_2025/87-MHE-TPE-.../87-MHE-TPE...md`, score: 1637.1）：

```
MHE-TPE 混合精度映射的两阶段分解:
  
  时域映射 (Temporal Mapping of Multiplicands):
    → 不同精度的multiplicand A在不同cycle中映射到同一PE
    → bit-sliced encoding: 将A拆为bit-slice groups
    → multi-operand high-radix encoder: 基于vector inner product减少PPs一半
  
  空域映射 (Spatial Mapping of Multipliers):
    → 不同精度的multiplier B在不同PE列中映射
    → 跨PE协同: PE之间共享vector partial product lookup table
    → 消除跨PE的冗余PPs reduction

  三阶段统一计算范式:
    Bit-Slice Encoding → Vector PPs Generation → Cross-Dimensional Reduction
    (合并multiplier微架构中的bit-sliced reduction维度
     与vector inner product的spatial reduction维度)
```

- **核心机制**: 打破传统 PE 间隔离计算——传统 dataflow 范式下每个 PE 独立执行所有 PPs 生成，PE 间仅传递 operand 不共享计算结果；MHE-TPE 利用 MBE 编码系数集合 {-2,-1,0,1,2} 的对称性导致不同 bit-slice 组合产生相同/线性相关的 PPs，通过共享 lookup table 消除冗余 PPs——PPs 减少一半
- **实现**: UMC 22nm, 1GHz 约束, Synopsys Design Compiler 综合面积和功耗
- **实验环境**: UMC 22nm; 对比 baseline 包括 IBM 7nm NPU fixed-point PE 和 Samsung 4nm mobile SoC PE；支持 INT2~INT32 混合精度；operand bit-width 减半→4× throughput
- **来源**: Q5.1, Q5.1_L5_answer.md, `paper_secs/MHE-TPE` (score: 1637.1)

#### 方法: IANUS NPU-PIM 统一架构

来自 Q5.4 的 IANUS 系统原型（笔记: `paper_secs/secs_2025/30-IANUS-.../7.3-IANUS-System-Prototyping.md`, score: 2713.6）：

IANUS 将 NPU 和 PIM（Processing-in-Memory）统一在一个内存系统中：
- **PIM Access Scheduling**: 在 NPU 和 PIM 间映射负载，调度 PIM 和普通内存命令的并发执行
- **FPGA 原型**: Xilinx VCU118 板卡，通过 FMC connector 连接真实 GDDR6-AiM PIM 芯片；NPU 因太大无法放入单 FPGA，使用功能+周期精确的 NPU simulator 替代
- **有效内存带宽**: 2.4 TB/s（单设备），是 GDDR6 外部带宽的 9-10×（PIM 内部带宽优势）
- **多设备扩展**: 2/4/8 IANUS 设备通过 PCIe 5.0 x16 互联，利用 intra-layer parallelism + attention head parallelism
- **关键结果**: vs A100 GPU — GPT-2 上 6.2× speedup，cost-efficiency (perf/TDP) 提升 2.1-3.9×

- **核心机制**: PIM Access Scheduling — 在共享内存系统中交错调度 NPU 计算命令和 PIM 内存命令，将计算 offload 到 PIM 芯片利用内部高带宽
- **实现**: FPGA 原型 (Xilinx VCU118) + cycle-accurate NPU simulator + real GDDR6-AiM PIM chips
- **实验环境**: Xilinx VCU118 FPGA, PCIe 5.0 x16, GPT-2 inference
- **来源**: Q5.4, Q5.4_L5_answer.md, `paper_secs/IANUS` (score: 2713.6)

---

### 分类: 控制与调度

#### 方法: Warp Specialization (Warp 级微算子流水线)

来自 Q5.1 和 Q5.2 的 Warp Specialization 机制（笔记: `knowledge_notes/硬件知识笔记/Warp Specialization (Warp 专业化).md`, score: 3300.9; `knowledge_notes/kernel知识笔记/Warp-Group Specialization (Hopper GPU).md`, score: 2669）：

```
SM 配置 (256 threads = 8 warps):
  ┌───────────────────┬───────────────────┐
  │ Producer Warp(s)   │ Consumer Warp(s)  │
  │ (TMA + CUDA Cores) │ (Tensor Cores)    │
  ├───────────────────┼───────────────────┤
  │ TMA load KV_tile[0]│                   │
  │ mbarrier.arrive    │ mbarrier.wait     │
  │                    │ wgmma(Q,KV[0],acc)│
  │ TMA load KV_tile[1]│ (与上面重叠)       │
  │ mbarrier.arrive    │ mbarrier.wait     │
  │                    │ wgmma(Q,KV[1],acc)│
  │ ...                │ ...               │
  └───────────────────┴───────────────────┘

关键硬件需求:
  - TMA: 单线程 async copy (硬件级异步)
  - wgmma.mma_async: 异步Tensor Core指令，发射后立即返回
  - mbarrier: 硬件加速barrier
  - 寄存器解耦: producer warp的寄存器可释放给consumer warps
```

DMA warp（1 warp=32 threads，仅 thread 0 调用 TMA）和 Compute Warpgroup（4 warps=128 threads，执行 WGMMA）通过 mbarrier 异步同步。Warp scheduler 的 time-multiplexing 交替执行：Compute warpgroup 等 Tensor Core 完成（`warpgroup_wait`）时→切换到 DMA warp；DMA warp 等 consumer 释放 buffer（`wait(cons)`）时→compute warpgroup 获得执行。Pipelining depth (PIPE=3) 确保 TMA 延迟被完全隐藏。

TileLang 编译器可自动分析 buffer 使用并插入 warp specialization 代码，其 FlashAttention 实现（~70 行 Python）达到手写 CUDA（FlashAttention-3）98% 的性能。

- **核心机制**: 将单一 kernel 内部 data movement 和 computation 解耦为不同 warp 角色，实现 fine-grained overlap；TMA+mbarrier+WGMMA 三个 Hopper 新特性是硬件基础
- **实现**: CUTLASS 3.x `sm90_mma_tma_gmma_rs_warpspecialized`, TileLang (P.Tiled 调度原语), ThunderKittens (LCSF 模板)
- **实验环境**: H100 (Hopper SM90), FlashAttention-3 kernel
- **来源**: Q5.1, Q5.2, `knowledge_notes/Warp Specialization` (score: 3300.9), `knowledge_notes/Warp-Group Specialization` (score: 2669)

#### 方法: GPU Dataflow Execution (Kitsune)

来自 Q5.4 的 Kitsune GPU 数据流执行（笔记: `paper_secs/secs_multimodal_kernel/Kitsune.../Kitsune-Enabling-Dataflow-Execution-on-GPUs.md`, score: 1662.6）：

Kitsune 通过两个关键原语在 NVIDIA GPU 上实现数据流执行：

**原语1 — Inter-CTA Ring Queue（软件实现）**:
```
Queue 结构（L2-resident ring buffer）:
┌─────────────────────────────────────────┐
│  Entry 0 (metadata + payload)            │  ← wr_acquire / rd_acquire
│  Entry 1 (metadata + payload)            │     通过 GPU atomics 同步
│  ...                                     │
│  Entry N-1 (double-buffering)            │
└─────────────────────────────────────────┘

Producer CTA (SM_i):                Consumer CTA (SM_j):
  wr_acquire() → 写入 payload        rd_acquire() → 自旋等待 metadata
  release()   → atomicAdd 更新 seq   release()   → atomicAdd 更新 seq
```

Ring queue 驻留在 L2 cache 中（通过 CUDA API pinning），A100 实测：54 个 queue、128-256KB payload 时 aggregate bandwidth 达 **2 TB/s**（37 GB/s/queue）。

**原语2 — Modified Grid Scheduler（硬件修改）**:
```
双 arbiter 设计:
  SIMT Arbiter (for Elementwise CTAs)
  TENSOR Arbiter (for GEMM CTAs)
  → SM Occupancy Check & Pairing
  → CTA Dispatch
```

双 arbiter 使 Grid Scheduler 能将不同类型的 CTA 配对到同一 SM，实现 Tensor Core 和 SIMT Core 同时被不同算子使用。GPU 硬件仅需"modest adjustment"——增加一个 arbiter 和一个 type-aware dispatch 逻辑。

- **核心机制**: Inter-CTA Ring Queue（L2-resident, atomics 同步）+ Modified Grid Scheduler（双 arbiter 支持异构 CTA 类型 SM co-residency）
- **实现**: PyTorch Dynamo → CUDA spatial pipeline API；A100 实测 aggregate BW 2 TB/s
- **实验环境**: NVIDIA A100 (108 SM)；5 个 DL 应用上 1.3×-2.4× 加速，off-chip traffic 减少 41%-98%
- **来源**: Q5.4, Q5.4_L5_answer.md, `paper_secs/Kitsune` (score: 1662.6)

#### 方法: µShare Intra-SM Scattered Kernel Co-locating

来自 Q5.6 的 µShare 非侵入式 SM 内并发（笔记: `paper_secs/secs_2026/66-μShare.../V.-EVALUATION.md`, score: 1928.1）：

µShare 通过修改 kernel launch 参数（blocksize）间接影响 GigaThread Engine 的 left-over 调度行为，在真实 GPU 上唯一可行的非侵入式方案：

- **实验平台**: 8 台服务器，Intel Xeon Gold 6338 CPU (128 逻辑核), A40 (84SM) + A800 (108SM)
- **负载**: 10 个 MLPerf + PyTorch benchmark 模型: Llama2-7b, GPT-2, Bert, ResNet50-v1.5, MobileNet_v2, Swin Transformer, Vision Transformer, Yolostiny, Resnet101, EfficientNet_B7
- **SLO**: 200ms（通用）, 400ms（Llama2-7b）
- **对比系统**: INFless (stacked co-location), Orion (compute/memory complementary co-location), Tacker (intra-SM kernel fusion), CUDA Graphs
- **关键结果**: 吞吐量提升 26.90%（vs INFless）、54.09%（vs Orion）；CUDA Graphs 对吞吐量提升仅 2.97%（因 co-location 场景中 kernel 执行时间主导 launch 时间）；µShare 仍保持 26.44% vs INFless + CUDA Graphs

- **核心机制**: 通过 blocksize 参数间接影响 dispatch unit 的 left-over 调度实现 SM 内 kernel 交错执行——非侵入式、无硬件修改
- **实现**: CUDA 11.8/12.1, PyTorch
- **实验环境**: A40 (84SM, 48GB GDDR6), A800 (108SM, 80GB HBM2e); Azure INFless production trace (Poisson arrival)
- **来源**: Q5.6, Q5.6_L5_answer.md, `paper_secs/µShare` (score: 1928.1)

#### 方法: Bullet 动态 SM 分区 Prefill-Decode 时空并发

来自 Q5.6 的 Bullet 动态 SM 分区方法（笔记: `paper_secs/secs_2026/2-Bullet.../4-Experimental-Evaluation.md`, score: 2831.6）：

- **动态 SM 分区**: 根据系统负载实时调整 prefill SM 数——burst 时全 GPU prefill，恢复后平衡分区
- **Nsight Systems 验证**: 并发执行时 SM active cycles 86.2%（+11.2% vs SGLang）、Tensor Core utilization +11.8%、memory-BW utilization +19.3%
- **固定 vs 动态**: 固定 SM 分区（SM-108/SM-84）导致 TTFT/TPOT 失衡；动态调整将两者同时优化

- **核心机制**: 运行时根据请求队列状态动态调整 SM 分配到 prefill 和 decode pool，最大化 SM active cycles
- **实现**: SGLang v0.4.6, FlashInfer v0.2.7, CUDA 12.4, PyTorch 2.6.0
- **实验环境**: A100/H100/H20 单 GPU + 8×A100 NVLink 600GB/s + 8×H100 + 8×H200；Llama3.1-8B/70B + Qwen3-235B-A22B FP8 MoE；ShareGPT/Azure-Code/arXiv-Summary
- **来源**: Q5.6, Q5.6_L5_answer.md, `paper_secs/Bullet` (score: 2831.6)

---

### 分类: 访存体系与数据供给

#### 方法: GPU Memory Hierarchy 四级数据流

来自 Q5.2 的 GPU 访存层次完整数据流（笔记: `knowledge_notes/硬件知识笔记/GPU Memory Hierarchy (HBM vs SRAM).md`, score: 10900; `knowledge_notes/系统知识笔记/Memory Wall.md`, score: 3878）：

```
HBM (Off-chip DRAM) 40-80GB / 1.5-3.35TB/s / Latency: 400-800 cycles
  → L2 Cache (On-chip SRAM shared by all SMs) 40-50MB / 7-12TB/s / ~200 cycles
    → L1/SMEM (per SM, configurable) 128-228KB / ~19TB/s / 20-30 cycles
      → RF (per SM) 256KB, 65536x32-bit regs / ~40TB/s / 0 cycles
        → Tensor Core / CUDA Core (Compute Units)

DSM (L1.5 Cache, H100 only): SM-to-SM Crossbar NoC, 0.2-3.6MB / 4-8TB/s
```

Memory Wall 定量：H100 FP16 峰值算力（~1000 TFLOPS）增速 3.3× 远超 HBM 带宽增速（1.5×），多算子并发加剧了这一带宽竞争——fusion 减少 HBM 往返是解决瓶颈的核心思路。

- **核心机制**: 容量-带宽权衡金字塔 — 每层通过 tiling 适配 working set（SMEM 适合 tile 级复用，RF 适合指令级复用），双缓冲/多级 pipeline 隐藏跨层级数据传输延迟
- **实现**: TMA/cp.async 硬件 DMA + 双缓冲 + CUDA Stream 异步
- **实验环境**: H100 (HBM3 80GB 3.35TB/s, L2 50MB, SMEM 228KB/SM, DSM 0.2-3.6MB), A100 (HBM2e 2.0TB/s, L2 40MB, SMEM 192KB/SM)
- **来源**: Q5.2, Q5.2_L5_answer.md, `knowledge_notes/GPU Memory Hierarchy` (score: 10900)

#### 方法: TMA + mbarrier 异步数据搬运与同步

来自 Q5.2 的 TMA 与 cp.async 两代异步搬运硬件对比（笔记: `knowledge_notes/硬件知识笔记/Hopper TMA (Tensor Memory Accelerator).md`, score: 8862; `knowledge_notes/硬件知识笔记/Tensor Memory Accelerator (TMA).md`, score: 8555）：

| 特性 | cp.async (Ampere A100) | TMA (Hopper H100) |
|------|------------------------|---------------------|
| 编程模型 | warp 内所有线程参与地址计算和发射 | 单线程发起，硬件完成全部传输 |
| 数据路径 | GMEM → L1/L2 → SMEM（16B 粒度） | GMEM → SMEM（bulk tensor，1D-5D） |
| 寄存器中转 | 经过寄存器（增加寄存器压力） | 不经过寄存器（直接 GMEM↔SMEM） |
| 同步机制 | `__pipeline_commit` + `__pipeline_wait_prior` | mbarrier（硬件自动 arrive on completion） |
| Multicast | 不支持 | 单次读取广播到 cluster 内多 SM 的 SMEM |
| 对齐要求 | 4/8/16B | 128B (SMEM destination), 16B (innermost coord) |

mbarrier 同步层次（从粗到细）：GPU 全局（CUDA Stream）→ Block 级（`__syncthreads()` ~20-50 cycles）→ Warpgroup 级（`warpgroup_sync` + `warpgroup_wait` ~10 cycles）→ Warp 级（`__shfl_sync` ~1-5 cycles）→ 异步（mbarrier, 硬件管理 ~0 cycle overhead）。

- **核心机制**: TMA 单线程异步 bulk tensor 传输释放 127 线程用于计算；mbarrier 硬件加速同步替代软件 spin-wait；TMA multicast 消除多 SM 对同一数据的冗余 HBM 读取
- **实现**: CUDA `cuda::memcpy_async`, PTX `cp.async.bulk`, `<cuda/barrier>`, `cuTensorMapEncodeTiled` 创建 TensorMap descriptor
- **实验环境**: Hopper H100 (SM90), cluster launch, DSM
- **来源**: Q5.2, Q5.2_L5_answer.md, `knowledge_notes/Hopper TMA` (score: 8862)

#### 方法: FlashFuser DSM Fusion

来自 Q5.2 的 FlashFuser 利用 H100 DSM 进行算子融合（笔记: `knowledge_notes/硬件知识笔记/Distributed Shared Memory (DSM) on NVIDIA GPU.md`, score: 3746）：

- **DSM (L1.5 层)**: 通过 SM-to-SM Crossbar NoC 将同一 cluster 内最多 16 个 SM 的 SMEM 互联为 3.6MB 池，带宽 4-8TB/s（介于 SMEM 和 L2 之间）
- **FlashFuser 优化**: 将 fused GEMM chain 的中间 tensor 数据路径从 `SMEM→HBM→HBM→SMEM` 改为 `SMEM→DSM→SMEM`，减少 58% global memory access
- **硬件基础**: SM-to-SM Crossbar NoC 提供低延迟高带宽的 SM 间直接通信通道

- **核心机制**: 利用 DSM 作为 inter-SM 片上数据通道 — 前一个算子的输出 tile 留在 SMEM→通过 DSM 直接传给后一个 SM→消除 HBM 往返
- **实现**: DSM API (`dsm_shuffle`, `dsm_all_exchange`), CUDA cluster launch
- **实验环境**: H100 (Hopper SM90) cluster 模式
- **来源**: Q5.2, Q5.2_L5_answer.md, `knowledge_notes/DSM` (score: 3746)

---

### 分类: 片上网络与互联架构

#### 方法: SambaNova RDN 2D Mesh NoC (三 Fabric 分离)

来自 Q5.3 的 RDN 互联架构（笔记: `knowledge_notes/硬件知识笔记/Reconfigurable Dataflow Network (RDN).md`, score: 10192.4）：

RDN 是 SambaNova SN40L RDU 的片上互联核心，连接 1040 个 PCU + 1040 个 PMU + AGCUs：

**三 fabric 分离设计**（关键创新）：
- **Vector fabric**: packet-switched, tensor 数据主通道，支持 one-to-many multicast 和 many-to-one 重排序
- **Scalar fabric**: packet-switched, metadata（地址/标量参数）
- **Control fabric**: circuit-switched, 单比特线束，coarse-grain flow control（Counter Done Events）

**流控机制**:
- Credit-based per-hop flow control：防止生产者溢出消费者
- End-to-end flow control：software tokens + hardware credits 双重保障
- Sequence ID metadata field：PMU 使用 sequence ID 计算写地址将乱序 packet 归位
- Packet Throttling Controller：可编程限速，平滑多算子并发流量

**路由模式**:
- Dynamic 2D dimension order routing：运行时自适应拥塞
- Static flow routing：编译器 Place-and-Route 预计算，确定性延迟
- Multicast：flow ID based fan-out

**控制模块详解**:
| 控制模块 | 功能 | 并发支持 |
|----------|------|----------|
| RDN Switch (Non-Blocking) | 2D mesh 交叉点交换节点，连接东西南北四邻居 | 多 PCU 间并行 packet 传输，非阻塞架构消除 head-of-line blocking |
| Credit Manager | 逐跳 credit 计数，发送方仅在接收方有 credit 时发包 | 防止多源 burst 导致的 buffer overflow |
| Flow ID Decoder | 解析 packet header flow ID，查表确定路由路径 | Static routing 支持确定性延迟，dynamic routing 适应拥塞 |
| Sequence ID Reorder Buffer (PMU) | 多源 many-to-one 按 sequence ID 恢复逻辑顺序 | 支持多个 producer 同时向同一 consumer 发送，无需全局 barrier |
| Packet Throttling Controller | 可编程限速，软件控制 packet 注入速率 | 减少 bursty traffic 瞬时拥塞 |
| Performance Counter | Switch 中计数 stall cycles，识别 RDN 拥塞热点 | 为编译器反馈优化提供硬件观测数据 |

- **核心机制**: 三 fabric 物理分离避免数据/控制/标量流量相互干扰；credit-based per-hop + end-to-end 双层流控；sequence ID 乱序重排序
- **实现**: SN40L RDU (TSMC 5nm), 编译器 Place-and-Route 预计算 static flow routing
- **实验环境**: SambaNova SN40L, 1040 PCU + 1040 PMU + AGCUs
- **来源**: Q5.3, Q5.3_L5_answer.md, `knowledge_notes/RDN` (score: 10192.4)

#### 方法: NVLink + NVSwitch 全互联交换网络

来自 Q5.3 的 NVLink/NVSwitch 互联架构（笔记: `knowledge_notes/硬件知识笔记/NVLink _ NVSwitch (GPU Interconnect).md`, score: 6941.0）：

```
NVSwitch 拓扑 (DGX H100 Node):
  GPU0 SM/TMA/CE
    → NVLink 4.0 (450 GB/s单向) → NVSwitch 0
    → NVLink → NVSwitch 1
    → NVLink → NVSwitch 2
    → NVLink → NVSwitch 3
  
  NVSwitch 0-3 (TSMC 4N, 64 port/颗, 3.2 TB/s双向总带宽/颗)
    → Crossbar Non-Blocking → GPU1-GPU7
  
  NVSwitch SHARP ALU (400 GFLOPS FP32)
    → In-Network Reduction → Reduced Result
```

**通信层次**: PCIe (64 GB/s, CPU↔GPU) < NVLink (450-900 GB/s, GPU↔GPU) < NVSwitch (3.2 TB/s aggregated, all-to-all switching) < NVSwitch SHARP (in-network reduction)

**MoE 场景使用模式**: Expert parallelism 的 all-to-all token exchange 直接通过 NVLink + NVSwitch 完成，每个 GPU 同时与所有其他 GPU 双向通信。

**关键瓶颈趋势**: 从 A100→B200：Tensor Core TFLOPS 提升 7.2×，HBM BW 提升 5.1×，但 NVLink BW 仅提升 3×——**通信成为瓶颈的硬件根源**。

- **核心机制**: Non-blocking crossbar switch 实现 all-to-all 单跳 <1μs；SHARP in-network reduction 在 switch 内部执行 reduce，数据不往返 GPU
- **实现**: NCCL Ring/Tree/NVLS (programming), NVSwitch (TSMC 4N 硬件)
- **实验环境**: H100 DGX (NVLink 4.0 450 GB/s), B200 (NVLink 5th-gen 900 GB/s); MoE all-to-all token exchange, all-reduce
- **来源**: Q5.3, Q5.3_L5_answer.md, `knowledge_notes/NVLink_NVSwitch` (score: 6941.0)

#### 方法: Active Interposer + DFBM 死锁自由 Chiplet 互联

来自 Q5.3 的死锁自由 chiplet 互联方案（笔记: `knowledge_notes/芯片知识笔记/Open Chiplet Ecosystem and Inter-Chiplet Deadlock.md`, score: 3299.6; `paper_secs/.../61-Deadlock-Free Bridge.../A.-Chiplet-Ecosystem.md`, score: 5540.0）：

**核心问题**: 即使每个 chiplet 内部 NoC 是 deadlock-free 的，跨 chiplet-interposer-chiplet 的循环通道依赖可导致死锁。

**DFBM 方案**: 将 deadlock 处理逻辑外置到 interposer 侧的 bridge module：
- Credit-based admission control 保证 chiplet→interposer 方向有足够吸收能力
- 从根源消解 CDG (Channel Dependency Graph) 环形成条件
- **标准化意义**: chiplet 供应商只需提供少量协议参数（coherence state machine 依赖、最大 outstanding request 数、VC 数量），无需暴露或修改内部 NoC 细节

**跨 Chiplet 死锁解决方案对比**:

| 方案 | 死锁处理 | 拓扑无关 | 低集成开销 | 高可移植性 | Chiplet 标准化 | 机制 |
|------|----------|----------|------------|------------|----------------|------|
| MTR | 避免（turn restriction） | ✗ | ✓ | ✗ | ✓ | 边界路由器禁止特定转向 |
| DeFT | 避免（VC isolation） | ✓ | ✗ | ✓ | ✓ | 上下行流量分配独立 VC |
| RC | 避免（injection control） | ✗ | ✗ | ✗ | ✗ | Chiplet 内专用 permission network |
| UPP | 恢复（允许死锁再恢复） | ✓ | ✗ | ✓ | ✗ | Escape channel + 死锁检测 |
| Steered Bubble | 恢复 | ✓ | ✓ | ✓ | ✗ | Directional bubble routing |
| **DFBM** | 避免（credit-aware admission） | ✓ | ✓ | ✓ | ✓ | Credit management + shared deadlock buffer |

- **核心机制**: Credit-aware admission control + shared deadlock buffer — 将 deadlock 处理外置到 interposer bridge，支持 "plug-and-play chiplet"
- **实现**: Active interposer (4×4 mesh + XY routing + 3 VN + per-VN 2/4 VC + virtual cut-through flow control), UCIe PHY
- **实验环境**: gem5 + Garnet cycle-accurate 模拟，4 homogeneous chiplets × shared active interposer，synthetic traffic (Uniform-Random/Transpose/Bit-Rotation), PARSEC benchmark
- **来源**: Q5.3, Q5.3_L5_answer.md, `knowledge_notes/Open Chiplet Ecosystem` (score: 3299.6), `paper_secs/DFBM` (score: 5540.0)

#### 方法: SoW 晶圆级 2D Mesh

来自 Q5.3 的 TSMC SoW 晶圆级互联（笔记: `knowledge_notes/芯片知识笔记/System-on-Wafer (SoW) Technology.md`, score: 797.1）：

- **规模**: 24 compute dies + 96 HBM dies 集成在单晶圆 (>200,000 mm²)，远超单 die photomask 限制 (800-1,000 mm²)
- **二維互聯分層**: 垂直方向 LSI (Local Silicon Interconnect) terabit-level 带宽；水平方向 XSR SerDes 1.7 TB/s D2D 带宽，延迟 200 ns/hop
- **NUMA 效应**: local HBM access 300 ns vs 远端 Die7 access 3100 ns (7 hops × 200 ns × 往返)，差距 ~10×（最远 ~15×）
- **MoE Serving 意义**: 可将完整 MoE 模型 (200B-1000B 参数) 容纳在单个晶圆上，消除跨机架/跨节点网络瓶颈

- **核心机制**: LSI (垂直 terabit 级) + XSR SerDes (水平 1.7 TB/s D2D) 二維互聯分層；8×3 矩形 2D Mesh 拓扑
- **实现**: TSMC SoW 晶圆级集成工艺
- **实验环境**: TSMC SoW (8×3 topology, 24 compute dies + 96 HBM dies)
- **来源**: Q5.3, Q5.3_L5_answer.md, `knowledge_notes/SoW` (score: 797.1)

---

### 分类: 硬件设计与实现流程

#### 方法: RTL → FPGA → ASIC 标准设计流程

来自 Q5.4 的硬件设计流程（综合自 DRX、Tandem Processor、Focus 三篇论文笔记）：

```
架构设计 → RTL 实现 (SystemVerilog/Verilog)
  → FPGA 原型 (Xilinx Vivado Synthesis + P&R)
    → FPGA Validation (Xilinx VCU118/VU9P, 250 MHz operation, 可选 real PIM chip connect via FMC)
  → ASIC 实现 (Synopsys Design Compiler Logic Synthesis → Synopsys IC Compiler Place & Route)
    → Tape-out (TSMC 28nm / GF 65nm / FreePDK 15nm 学术)
```

关键工具链与工艺节点：
- **综合工具**: Synopsys Design Compiler R-2020.09-SP4, Cadence Genus
- **布局布线**: Synopsys IC Compiler L-2016.03-SP1, Cadence Innovus
- **FPGA 平台**: Xilinx UltraScale+ VU9P (AWS F1 / VCU118), Vivado 2022.2
- **工艺节点**: TSMC 28nm (N28HPC+), GF 65nm, FreePDK 15nm, TSMC 22nm, UMC 22nm
- **功耗建模**: FreePDK 15nm logic cells + CACTI-P (on-chip memory), Cadence Spectre (TT corner, 25°C, 1GHz, 1.1V)
- **仿真**: Cycle-accurate simulator (自研，与 RTL 误差 ≤5%), SCALEsim-v2, Verilator

- **核心机制**: SystemVerilog/Verilog RTL→Synopsys DC 综合→IC Compiler P&R→Tape-out 的完整学术流片流程
- **实现**: SystemVerilog, Verilog, Xilinx Vivado, Synopsys DC/ICC, Cadence Genus/Innovus/Spectre
- **实验环境**: 多工艺节点 (TSMC 28nm/22nm, GF 65nm, FreePDK 15nm, UMC 22nm)；对比平台: NVIDIA Jetson Xavier NX, RTX 2080 Ti, A100 (iso-TOPs)
- **来源**: Q5.4, Q5.4_L5_answer.md, `paper_secs/DRX` (score: 496.6), `paper_secs/Tandem Processor` (score: 172.4), `knowledge_notes/Focus Unit` (score: 173.2)

#### 方法: CuTe DSL GPU Kernel 开发 (MoE 并发)

来自 Q5.4 的 CuTe-DSL GPU kernel 开发工具链（笔记: `knowledge_notes/编译知识笔记/CuTe-DSL (CUTLASS CuTe DSL).md`, score: 860.2）：

```
CuTe 编译链:
  CuTe C++ Template
    → nvcc -arch=sm_90a -std=c++20
    → PTX (Parallel Thread Execution)
    → ptxas
    → SASS (Native GPU ISA)
    → GPU Hardware (SM90/SM100)

核心硬件映射:
  TiledMMA  → WGMMA (SM90) / UMMA (SM100) — Tensor Core 异步指令
  TiledCopy → TMA load / cp.async           — 异步数据搬运
  Pipeline  → warp-specialized               — Producer/Consumer warp 分工
              producer: TMA load → smem
              consumer_0: WGMMA + epilogue
              consumer_1: TMA store → gmem
```

CuTe 能表达 Triton 无法表达的底层硬件特性：warp-specialized 异步调度、Ping-Pong pipeline、cluster-level synchronization（mbarrier cluster scope）。SonicMoE 完全基于 CuTe 编写 8 个 MoE kernel，利用 warp-specialized pipeline、TMA descriptor、Ping-Pong scheduling 和 persistent tile scheduler——这是 GPU 上实现多微算子并发的实际软件工具链。

- **核心机制**: C++ template-based DSL 直接生成 PTX/SASS，精确控制 TMA+WGMMA 异步 overlap timing、warp-specialized pipeline、cluster-level sync
- **实现**: CUTLASS 3.x, SonicMoE (基于 CuTe 编写 8 个 MoE kernel), NVCC → PTX → SASS
- **实验环境**: H100 (SM90), B200 (SM100)
- **来源**: Q5.4, Q5.4_L5_answer.md, `knowledge_notes/CuTe-DSL` (score: 860.2)

---

### 分类: 模拟与性能评估

#### 方法: GPGPU-Sim / Accel-Sim GPU 微架构模拟生态

来自 Q5.5 的 GPU 模拟器全景（笔记: `paper_secs/secs_2025/3-AMALI.../7.2-GPU-Simulator.md`, score: 2708.3）：

```
GPGPU-Sim 数据流:
  SASS/CUDA Kernel Binary
    → GPGPU-Sim Frontend (PTX/SASS Parser)
    → Warp Scheduler (Scoreboard + SIMT Stack)
    → SM (SIMT Core + Tensor Core)
    → L1 Cache + Shared Memory (128KB/SM, Configurable)
    → L2 Cache (Unified, Multi-bank)
    → Memory Controller (FR-FCFS / Adaptive)
    → HBM/DRAM (DRAMSim3 / Ramulator)

  Warp Scheduler → Most-Room Policy (Thread Block Placement)
  SM → GPUWattch / AccelWattch (Power Model)
  Most-Room Policy → Concurrent Kernel Simulation Accuracy
  GPUWattch → Energy per Kernel (mJ / Joules)
```

SM 建模粒度：warp scheduler（scoreboard、SIMT stack）、SIMT Core（ALU/FPU/SFU）、Tensor Core（MMA 指令级别）、L1/Shared Memory（bank conflict 建模）、L2 Cache（multi-bank + MSHR）。

- **核心机制**: Cycle-level GPU 微架构模拟，SASS 级别指令追踪；warp scheduler + scoreboard + SIMT stack + Tensor Core + L1/L2 + HBM 全流水线建模
- **实现**: 开源 (https://github.com/gpgpu-sim/gpgpu-sim_distribution)；Accel-Sim 扩展 SASS 支持
- **实验环境**: NVIDIA GPU (Pascal/Volta/Turing/Ampere-like)
- **来源**: Q5.5, Q5.5_L5_answer.md, `paper_secs/AMALI 7.2-GPU-Simulator` (score: 2708.3)

#### 方法: gem5 + DRAMsim3/Ramulator2 全系统模拟

来自 Q5.5 的全系统模拟平台（笔记: `paper_secs/.../IV.-PERFORMANCE-CHARACTERIZATION-MEMORY-SIMULATORS.md`, score: 1747.6）：

**内存模拟器精度关键发现**（笔记揭示重大偏差）：

| 模拟器 | 延迟精度 | 带宽精度 | 行缓冲建模 | 关键局限 |
|--------|----------|----------|------------|----------|
| DRAMsim3 | 起始 68 ns, 线性增长 | 行缓冲命中率 84-93%（偏差大） | 偏差大 | 无带宽饱和建模；高写入比延迟偏离 |
| Ramulator | 起始 25 ns（严重偏低） | 命中率趋势较好 | 饱和区偏离 | 起始延迟严重偏低 |
| Ramulator2 | 极低延迟 | 最大模拟 BW 仅 126 GB/s（实际 292 GB/s 的 43%） | — | 严重低估带宽 |
| Mess Simulator | 误差 1.3% (ZSim) / 3% (gem5) | 基于实测 BW-Latency 曲线 | 分析型反馈 | 需要目标系统 BW-Latency 曲线输入 |

- **核心机制**: 全系统 cycle-accurate CPU+GPU+加速器模拟；多内存后端可选；Mess Simulator 通过分析型反馈控制替代时序模拟实现 13-15× 加速 + 1.3-3% 误差
- **实现**: gem5 (开源) + DRAMsim3 (开源) + Ramulator2 (开源) + Mess Simulator (私有)
- **实验环境**: 64-core ARM/x86 CPU + GPU + Accelerator; DDR4/DDR5/HBM2/CXL
- **来源**: Q5.5, Q5.5_L5_answer.md, `paper_secs/IV.-PERFORMANCE-CHARACTERIZATION-MEMORY-SIMULATORS` (score: 1747.6)

#### 方法: Timeloop + Accelergy DNN 加速器设计空间探索

来自 Q5.5 的加速器评估框架（笔记: `paper_secs/.../H2-LLM...md §7.1`, score: 344.5）：

```
Timeloop 数据流:
  Workload Spec (Layer Shapes/Precision)
  Architecture Spec (PE Array/Memory Hierarchy)
  Mapping Constraints (Tiling/Loop Order/Spatial)
    → Map Space Explorer (Hybrid Heuristic + Random)
    → Analytical Performance Model (Latency + Energy)
    → Optimal Mapping Output

Accelergy:
  Component Energy Model (Plug-in Library)
  Area Model (CACTI + Custom)
    → Energy Estimate (pJ per Access/Operation)
    → 联合 Timeloop 输出 Latency + Energy + Area

后端工具链 (H2-LLM 实际使用):
  Chisel RTL → Synopsys DC (40nm) → MAC 能量 0.974-1.365 pJ/MAC
  SRAM Compiler → 密度 2.72mm²/MB, 访问能量 0.027 pJ/bit
  Ramulator2 扩展 → NMP PE 计算模拟
  Tileflow 性能模型 → centralized processor 算子性能
```

- **核心机制**: Map Space Explorer 搜索最优 tiling/loop order/spatial mapping + Accelergy 可插拔组件能量模型库 + Ramulator2 扩展 NMP PE 计算
- **实现**: Timeloop (开源) + Accelergy (开源) + Ramulator2 (开源扩展)
- **实验环境**: 通用 DNN 加速器; H2-LLM: OPT 6.7B, LLaMA3 8B, PaLM 8B; Chisel RTL → Synopsys DC (40nm)
- **来源**: Q5.5, Q5.5_L5_answer.md, `paper_secs/H2-LLM` (score: 344.5)

#### 方法: 自研 MoE Multi-Chiplet 事件驱动模拟器

来自 Q5.5 的 MoE 专用模拟器（笔记: `experiment_notes/硬件实验笔记/Orders in Chaos_...MoE...md`, score: 417.1）：

此模拟器专为 wafer-scale multi-chiplet GPU 上的 MoE decode 评估设计，因现有工具均不适用（Gem5/gpgpusim cycle-accurate 但太慢；ASTRA-sim 不支持 single-GPU-like programming model）：

```
Python Event-Driven Simulator:
  Die Object × N (LLC + HBM + Compute + D2D)
  Central Resource Manager (Bandwidth Contention + Congestion)
  Global CP (Expert Distribution Table + Cross-token Heatmap Cache 0.5MB)
  Local CP (Per-die SM Task Assignment + PDU Prediction Table)
  D2D Controller (ATU + PDU + XY Routing)
```

- **验证**: 8×H100 DGX server 实测数据验证，误差 < 5%
- **开源**: https://github.com/zhongkaiyu/waferscale_gpu_moe_sim (Apache-2.0)
- **关键结果**: 7.0× throughput + hop count 降低 213× (DeepSeek V3 on Dojo)；Allo Only 已降低 hop 142×

- **核心机制**: 事件驱动 + expert-aware task allocation + data movement prediction；Global CP 维护 Expert Distribution Table + Cross-token Heatmap Cache + PDU 管理 expert 数据缓存
- **实现**: Python 事件驱动；开源 (Apache-2.0)
- **实验环境**: Wafer-scale multi-chiplet GPU (Dojo 5×5, TSMC SoW 8×3)；DeepSeek V3, Kimi K2, Llama4 Maverick, Qwen3-235B MoE decode
- **来源**: Q5.5, Q5.5_L5_answer.md, `experiment_notes/Orders in Chaos` (score: 417.1)

#### 方法: Roofline Model 三维约束分析

来自 Q5.5 的 Roofline Model（笔记: `paper_secs/.../AccelOpt...md §3.3`, score: 1245.6）：

$$T = \max \left( \frac{\text{Traffic}_{\text{Min}}}{\text{Bandwidth}}, \frac{\text{FLOPs}_{\text{MM}}}{\text{Peak}_{\text{MM}}}, \frac{\text{FLOPs}_{\text{Vec}}}{\text{Peak}_{\text{Vec}}} \right)$$

$$\text{Peak Throughput \%} = \frac{T}{t_{\text{measured}}}$$

其中 $\text{Peak}_{\text{Vec}} = \text{Peak}_{\text{Vector Engine}} + \text{Peak}_{\text{Scalar Engine}}$（两者并行执行，取最优假设）。

- **核心机制**: 将硬件性能分解为三个约束维度：带宽瓶颈、矩阵计算瓶颈、向量/标量计算瓶颈；适用于多引擎并发场景（Tensor + Vector + Scalar 引擎同时运行）
- **实现**: AccelOpt 分析框架
- **实验环境**: AWS Trainium 1/2：峰值吞吐从 49%→61% (Trainium 1)、45%→59% (Trainium 2)
- **来源**: Q5.5, Q5.5_L5_answer.md, `paper_secs/AccelOpt` (score: 1245.6)

---

## 方法间关系

### 替代关系

1. **Weight Stationary Systolic Array ←→ Output Stationary 3D PE Array (GyRot)**:
   - Weight stationary: weights 预加载 PE 不动，减少 weight 重复读取但 partial sum 需在 PE 间移动
   - Output stationary (GyRot): partial sum 留在 PE accumulator，减少 data movement 但需要更宽 per-PE datapath 和 adder tree
   - 选择取决于 target workload 的矩阵形状和 bit-width

2. **Fixed-Dataflow Systolic Array ←→ FEATHER 可重构 Dataflow**:
   - Fixed SA: 水平 rigid reuse link，规则 shape 下效率高但 irregular shape 利用率下降明显
   - FEATHER: 各 column 独立映射 dataflow，BIRRD flexible reduction 提升 irregular shape 利用率；但 flexibility overhead（BIRRD 面积 ~0.8% total area）需要在灵活性和面积间 trade-off

3. **GPU SIMT 时序复用 ←→ Dataflow Architecture 空间计算 (StreamTensor)**:
   - GPU: BSP 模型，warp scheduler time-multiplexing，Tensor Core 和 SIMT 资源时序复用
   - Dataflow accelerator: synchronous dataflow 模型，多个 kernel 空间流水线并发，FIFO 流式传递中间数据
   - GPU 灵活但带宽利用率较低；Dataflow 加速器专用但对不规则 pattern 灵活性差

4. **NVLink+NVSwitch ←→ AMD Infinity Fabric Full Mesh**:
   - NVLink/NVSwitch: 通过外部 switch 芯片实现 all-to-all non-blocking，支持 SHARP in-network reduction，可扩展到更多 GPU
   - Infinity Fabric: GPU 间 full mesh 直连（无需外部 switch），单跳 <1μs 但无 in-network reduction，O(N²) link 数限制扩展到 >8 GPU

5. **2D Mesh NoC ←→ Crossbar NoC**:
   - 2D Mesh: 面积 O(N)，延迟 O(√N) Manhattan，适合 N=16-256 的分布式计算
   - Crossbar: 面积 O(N²)，延迟 O(1) 单跳，适合 N≤128 的全互联非阻塞交换

6. **MTR/DeFT ←→ DFBM (Chiplet 死锁解决方案)**:
   - MTR/DeFT: chiplet 内部需要修改路由策略（turn restriction 或 VC isolation），拓扑依赖性强
   - DFBM: deadlock 处理外置到 interposer bridge，chiplet 内部无修改 → 真正 "plug-and-play chiplet"

7. **Cycle-Accurate 模拟器 ←→ Mess Simulator 分析型反馈**:
   - GPGPU-Sim/gem5: cycle-accurate 但慢（小时-天），适合详细微架构评估
   - Mess: 分析型反馈控制 13-15× 加速，误差 1.3-3%，适合系统级快速权衡分析

### 互补关系

1. **Warp Specialization + TMA + mbarrier**: 三者是 Hopper 上高性能多算子并发的"三件套"——TMA 提供异步数据搬运硬件，mbarrier 提供异步同步原语，Warp Specialization 利用二者将 data movement 和 computation 解耦为不同 warp 角色实现 fine-grained overlap

2. **Hierarchical Software Pipeline + 双缓冲**: 三级 pipeline（inter-tile SMEM 双缓冲 + intra-tile Reg 双缓冲 + Tensor Core 异步 MMA）通过叠加多级双缓冲实现 Memory-Copy-Compute 三级全重叠

3. **TMA Multicast + DSM**: TMA multicast 单次 HBM 读取广播到 cluster 内多 SM + DSM (SM-to-SM Crossbar NoC) 提供 SM 间数据共享通道 → FlashFuser 利用二者将中间 tensor 数据保留在片上

4. **Kitsune Inter-CTA Ring Queue + Modified Grid Scheduler**: Ring Queue 提供 SM 间数据传递通道，Modified Grid Scheduler 将异构 CTA 配对到同一 SM → 共同实现 GPU 上的空间数据流执行

5. **Focus SEC + Focus SIC**: SEC (token-level pruning, 1.9% area) + SIC (vector-level compression, 0.8% area) → 两个 modular add-on 协同工作，均在 SA memory interface 侧操作，与 GEMM 计算完全时间重叠

6. **Timeloop + Accelergy**: Timeloop 搜索最优 mapping → Accelergy 提供组件级能量估算 → 联合完成 DNN 加速器 design space exploration

7. **MoE-CAP + Roofline Model**: MoE-CAP 修正 MBU/MFU（考虑 sparse expert activation）→ Roofline Model 提供三维性能约束分析 → 联合评估 MoE 硬件架构的真实效率

### 依赖关系

1. **Warp Specialization → TMA + mbarrier (Hopper 硬件特性)**: Warp Specialization 依赖 TMA 单线程异步搬运（释放线程资源）和 mbarrier 异步同步（不阻塞所有线程）——这两个 Hopper 新特性是 Warp Specialization 的硬件基础

2. **FlashFuser DSM Fusion → H100 DSM (L1.5) + SM-to-SM Crossbar**: FlashFuser 的 inter-SM 片上数据传递完全依赖 H100 的 DSM 硬件设施——不可移植到 pre-Hopper 架构

3. **Kitsune Modified Grid Scheduler → GPU Hardware Modification**: Kitsune 的空间数据流执行依赖 Grid Scheduler 的双 arbiter 修改——纯软件方案（仅 Inter-CTA Ring Queue）可实现部分数据流效果，但无法实现异构 CTA SM co-residency

4. **BIRRD 可重构 Dataflow → Layoutloop Offline Search**: FEATHER 的 per-layer (dataflow, layout) co-switching 依赖 Layoutloop（增强 Timeloop）离线搜索最优配置——运行时仅加载预计算配置，不可动态自适应未知 layer

5. **MHE-TPE 跨 PE 协同 → MBE 编码对称性**: MHE-TPE 的跨 PE PPs 共享机制依赖 MBE 编码系数集合 {-2,-1,0,1,2} 的对称性——该机制不可直接推广到其他编码方案

6. **DFBM 死锁自由 → Active Interposer + UCIe 标准化**: DFBM 的 credit-aware admission control 依赖 Active Interposer 提供的可编程 bridge module + UCIe 标准化接口参数

7. **自研 MoE Simulator → Expert Selection Traces (>150GB)**: Orders in Chaos 的模拟精度依赖 HuggingFace Dataset 提供的真实 expert selection traces——没有这些 traces 无法准确建模 MoE all-to-all 通信模式

8. **CuTe DSL Kernel → NVCC → PTX → SASS 编译链**: CuTe 的底层硬件特性表达依赖 NVCC 编译器将 C++ template 正确映射为 PTX/SASS 指令——编译链的优化质量直接影响最终 kernel 性能

---

## 本层不确定性

1. **MoE Expert Parallelism 的硬件级支持**: 笔记中 MoE 相关论文主要关注分布式调度和负载均衡层面，对单 GPU/NPU 内部硬件计算单元如何专门支持 expert parallelism（如硬件 router、expert-aware PE array 组织）缺乏详细说明。evidence 更多在软件调度层（expert parallelism mapping, all2all communication overlap via SM control）。

2. **DiT Diffusion Iterative Compute Mapping 的硬件细节**: 笔记显示 DSV、TetriServe 等论文主要关注训练稀疏化和 serving 调度，对"DiT 的 iterative denoising 如何映射到硬件 PE array/systolic array 的具体数据流"缺乏 evidence。"可推断"DiT 的 iterative nature 适合 streaming-based dataflow 架构，但无证据支持。

3. **多模态异构计算的硬件级融合**: 笔记中 Focus 论文（VLM accelerator）处理 cross-modal attention，但 evidence 集中在 systolic array + Focus Unit add-on 架构，对"多模态 encoder-decoder 在单加速器上的异构计算单元 partition 和协同"为「笔记未明确说明」。

4. **Video Temporal-Spatial Mapping 的硬件计算模块设计**: 笔记中 DSV 和 Video DiT 量化论文存在，但 evidence 集中在 sparsity/quantization 层面，对"video frame 的 temporal-spatial dimension 如何映射到硬件 PE array/计算单元"为「笔记证据不足」。

5. **NPU MAC 阵列的具体厂商微架构细节**: 笔记中提到 IBM 7nm NPU 和 Samsung 4nm NPU 的 PE 结构简化图，但华为 Da Vinci、寒武纪 MLU 等国产 NPU 的具体 MAC 阵列组织和并发机制为「笔记未明确说明」。仅 XY-Serve 论文涉及 Ascend 910B 的 DaVinci 高层架构。

6. **Command Processor / Warp Scheduler 内部设计**: 笔记中未找到命令处理器/指令分发器的详细硬件实现说明（NVIDIA 闭源）。Warp Scheduler 的 greedy-then-oldest 策略为逆向工程推测，NVIDIA 未正式文档化。不同 GPU 代际策略可能不同。

7. **DiT/Video 负载特定硬件控制/访存支持**: 笔记中缺乏 DiT denoising step memory footprint 在硬件控制/访存层面的专门研究。关于 DiT 和 Video 的 warp scheduler/DMA 优化，主要基于通用 GPU 机制推断。

8. **多模态 Cross-Modal Memory Access 硬件支持**: 笔记主要集中在算法层（cross-modal attention entropy、KV cache merging），缺乏硬件层面的 cross-modal memory access 设计方法。

9. **Chisel/Bluespec/SystemC 设计流程缺乏证据**: 笔记中所有硬件实现均使用 SystemVerilog 或 Verilog RTL。这可能是 vault 覆盖偏差（vault 主要收录 MICRO/ISCA/HPCA 等会议论文，学术加速器多用 Verilog）。仅 H2-LLM 论文提及 Chisel 综合但细节有限。

10. **Scale-Sim/MAESTRO/ZigZag 笔记证据有限**: 虽在预关键词中列出，但 vault 搜索中未发现含这些工具详细建模方法的论文笔记。

11. **FireSim FPGA 加速仿真证据不足**: vault 搜索未找到 FireSim 的详细笔记。

12. **MoE/DiT/多模态专用 Benchmark 标准缺失**: 各论文使用自研 benchmark，缺少类似 MLPerf 的统一标准——尤其是针对 MoE all-to-all + DiT diffusion step + multimodal pipeline 联合并发场景的 benchmark 套件。

13. **NPU 专用模拟器证据缺失**: vault 中未发现华为 Ascend、寒武纪 MLU 等 NPU 平台的模拟器/评估工具相关笔记。

14. **功耗建模精度验证不足**: McPAT/GPUWattch 的精度验证仅通过与实际 GPU 功耗对比（nvidia-smi），对于新型加速器（HB-NMP、PIM）缺乏硅验证的功耗模型精度数据。

15. **商业硬件平台 per-link 带宽和功耗数值缺失**: Meta MTIA 2i、SambaNova SN40L 的具体 NoC per-link 带宽和功耗数据笔记未明确记载（可能属于商业机密）。

16. **DRAMsim3/Ramulator 精度缺陷的根因**: 笔记揭示这些主流模拟器与实际系统存在重大偏差（Ramulator2 误差高达 52%），但"为什么 JEDEC 兼容的时序模拟仍无法正确模拟内存性能"的深层原因笔记未完全解释——实际平台可能存在未公开的控制器优化。

[HORIZON_SUMMARY_DONE] L5
