# GPU、TPU、专用加速器：动态/运行时资源调度方法与论文（2025至今）

> 导出时间：2026-07-14
> 来源：Obsidian vault 六大目录 omnisearch 检索 + 全文阅读
> 范围：仅 2025 年及之后的论文与方法

---

## 一、GPU —— 集群级调度

### 1. Wind: 预测性多维资源调度
- **方法思路**：将资源调度从"响应式"变为"预测式"。XGBoost 基于 15 维特征向量预测任务执行时间/CPU/内存/GPU 需求，Hilbert 曲线将 3-4D 资源向量映射到 1D 空间保持局部性，三级管道调度（快速筛选→精确匹配→容量最大化）实现 O(log n) 决策。支持隔离、共享、抢占、优先级四种策略的统一架构。
- **实现示例**：`hilbert_encode(cpu, mem, gpu, order=16)` → 在 1D Hilbert 坐标上找 θ 距离内的节点 → 三级管道选出最优节点。XGBoost 预测准确率 >87%，推理延迟 ~1.5ms。
- **论文标题**：*Bridging the GPU Utilization Gap: Predictive Multi-Dimensional Resource Scheduling for AI Workloads* (EuroSys '26)

---

## 二、GPU —— Serving 级调度

### 2. FinDEP: 解耦专家并行的细粒度任务调度
- **方法思路**：将 MoE 推理的 Attention 和 Expert 部署到独立 GPU 组（AG/EG），通过三项创新最大化任务重叠：(1) shared expert 分区计算（部分在 AG 复用 attention 中间激活、部分在 EG）；(2) AG 计算下一层 attention 与 EG 计算当前层 expert FFN 重叠；(3) token 级 shared expert 融合消除独立 kernel launch。
- **实现示例**：`partition_shared_expert(layer, ratio=0.4)` → AG 保留 40% shared expert 计算减少 token 传输 → `cudaLaunchKernel(attention[layer+1], stream_ag)` 与 `cudaLaunchKernel(expert_ffn[layer], stream_eg)` 并发执行。
- **论文标题**：*Efficient MoE Inference with Fine-Grained Scheduling of Disaggregated Expert Parallelism* (2025)

### 3. Bullet: GPU 内 Prefill-Decode 时空编排
- **方法思路**：在同一 GPU 内将 prefill 和 decode 拆成独立进程/worker，通过 SRM (SM-scaling Roofline Model) 预测 TTFT/TPOT，SLO-aware scheduler 周期性搜索最优 SM 分区，`libsmctrl_set_stream_mask` 实现微秒级 SM 重分区（avg 4.1μs）。prefill 以 layer 粒度执行，decode 以 CUDA Graph step 执行，CUDA IPC 共享 GPU memory pool。
- **实现示例**：`if predicted_ttft > TTFT_SLO: sm_prefill += 16` → `libsmctrl_set_stream_mask(prefill_stream, sm_mask[sm_prefill])` → SM mask 以 16 SM 为粒度，A100 上 6 种配置。
- **论文标题**：*Bullet: Boosting GPU Utilization for LLM Serving via Dynamic Spatial-Temporal Orchestration* (2025)

### 4. JANUS: Attention-MoE 解耦推理
- **方法思路**：将 Attention 层和 MoE 层部署到独立 GPU 子集群，各层类型独立配置并行度（不再被迫共享 TP/EP degree）。Adaptive Two-Phase Communication：Phase 1 同节点多实例通过 NVLink 聚合中间激活，Phase 2 聚合后大块数据通过 GPUDirect RDMA 跨节点传输。根据资源配置和流量负载自适应选择直接点对点传输或一对一中继+NVLink 多播模式。
- **实现示例**：Gating 放置在 MoE 侧简化通信 → Case-1 (直接 P2P) 或 Case-2 (中继+NVLink 多播) 自适应选择 → attention nodes 与 MoE nodes 独立 scale。
- **论文标题**：*JANUS: Disaggregating Attention and Experts for Scalable MoE Inference* (2025)

### 5. FlexPipe: 碎片化集群中的 Inflight Pipeline 重构
- **方法思路**：离线用约束动态规划在通信带宽、GPU 显存、计算-通信重叠目标下搜索最优 stage 切分边界；运行时持续监控请求 CV/队列长度/吞吐延迟，按多目标优化公式选择最优 pipeline granularity（CV 升高→拆细 stage→异步迁移 KV cache；CV 回落→合并相邻 stage→释放多余 GPU）。Hierarchical Resource Graph 在 server/rack/cluster 三级协调扩容资源。
- **实现示例**：`g* = argmax[α·T_k/T_max + (1-α)·L_min/L_k · exp(-|ν_t-ν_k|/σ)]` → CV=4 时选 16-stage → 通过 RDMA/sendfile 异步迁移 KV cache → 细粒度 stage per-stage compute 降至 18.67ms。
- **论文标题**：*FlexPipe: Adapting Dynamic LLM Serving Through Inflight Pipeline Refactoring in Fragmented Serverless Clusters* (2025)

### 6. Proteus: 精度可伸缩的高吞吐推理 Serving
- **方法思路**：通过 accuracy scaling 在不同负载下动态调整推理精度来维持吞吐。高负载时使用低精度快速路径保证 SLO，低负载时恢复高精度。
- **实现示例**：多模型 workload 场景下动态精度切换 + multi-tenancy 调度。
- **论文标题**：*Proteus: A High-Throughput Inference-Serving System with Accuracy Scaling* (2025)

---

## 三、GPU —— Kernel 级调度

### 7. ACS: 乱序 Kernel 发射调度
- **方法思路**：将 CPU Tomasulo 乱序执行思想搬到 GPU kernel 调度层。用固定大小调度窗口（N=32~64）+ 运行时内存地址重叠检测替代 DAG 全局静态构建。每个 kernel 标注 RW-segments（读/写 GPU 内存区域），进入窗口时做 O(segments²) 依赖检测，upstream 为空的 READY kernel 立即发射到独立 CUDA stream。ACS-HW 将窗口硬化到 GPU Command Processor SRAM（N=32 仅需 ~1KB），dispatch overhead 降至 ~50-100ns。
- **实现示例**：`check_dependency(new_kernel, window_kernels)` → `for wseg in K.write_segments: for rwseg in W: if overlap(wseg, rwseg): K.upstream.append(W)` → upstream 为空→READY→调度器发射。ACS-HW 2.19× 加速（RTX 3060, Brax RL）。
- **论文标题**：*ACS: Adaptive Concurrent Scheduling — Out-of-Order Kernel Dispatch on GPUs* (2025)

### 8. μShare: 非侵入式 Intra-SM Kernel Co-Locating
- **方法思路**：仅通过修改 kernel blocksize 间接操纵闭源 GPU 硬件调度器。Half-plus shaping：将 kernel blocksize 设为略超半 SM thread 容量（A40: 768+32=800），使同 kernel 两个 block 无法共存于同一 SM→被迫散布到不同 SM→每 SM 剩余 threads 容纳另一 kernel 的小 block→不同 hardware resource dominant 的 kernel 互补执行。1/3-plus shaping 适配 2048 threads/SM 新架构。Time-shifted launching 处理 cuBLAS/cuDNN 等不可修改 blocksize 的 kernel。
- **实现示例**：`blocksize = 1536/2 + 32 = 800` → vectorized kernel (LDST dominant) + roll kernel (INT32 dominant) 在同一 SM 内互补 → 6 种 HW avg utilization: 10.90%→15.10% → throughput +19.94%。完全 `LD_PRELOAD` 实现，零 kernel 代码修改。
- **论文标题**：*μShare: Non-Intrusive Kernel Co-Locating on NVIDIA GPUs* (2025)

### 9. HuntKTm: 混合 Kernel+Task 级调度
- **方法思路**：LLVM 编译框架内三个协同组件：(1) Stream Scheduler 编译期自动分析 kernel 间 RAW/WAR/WAW 数据依赖，构建 DFG 后用 PP-Set 启发式分配 kernel 到多 stream 实现 kernel-level 并发；(2) Task Scheduler 编译期 resource analyzer + 运行时 lazy engine 做三维度 SM 可用量评估的动态 task 放置；(3) Memory Manager 在 stream graph 上做 liveness 分析，推迟 allocation 到 live range 起点。
- **实现示例**：DFG constructor → kernel distributor → synchronization generator → memory manager (liveness 分析+延迟 allocation) → resource analyzer (nvcc 获取 register/shared memory) → function wrapper。M2 应用 peak memory 从 17.6GB→11.2GB (-36.4%)，FP32 utilization 提升 3.54×。
- **论文标题**：*HuntKTm: Hybrid Scheduling and Automatic Management for Efficient Kernel Execution on Modern GPUs* (2025)

### 10. Infera: Daemon Kernel Device-Side Launch
- **方法思路**：GPU 上常驻一个 persistent daemon kernel（独占 1 个 SM），通过 CUDA Dynamic Parallelism 的 `cudaLaunchDevice` 以 fire-and-forget 方式直接从 device 发射 fused kernel，无需 host 端参与每次 launch。daemon 维护 shared memory double-ended queue (DKQ)，从中取出 kernel 并发射，fire-and-forget launch latency <10μs。
- **实现示例**：daemon kernel spin-wait DKQ → DKQ 非空→取 kernel pointer+argument pointer+launch config → `cudaLaunchDevice(kernel, args, gridDim, blockDim, sharedMem, stream=0)` → GPU scheduler 立即调度 → daemon 执行 `cudaGetLastError()` 错误检查。
- **论文标题**：*Automated End-to-End Model Serving with Cooperative Compilation and Scheduling* (Infera, 2025)

### 11. Demystifying GPU Thread Block Scheduler
- **方法思路**：逆向工程 NVIDIA GPU thread block scheduler 在并发 kernel 场景下的真实调度策略。实验发现并非先前假设的 round-robin——实际是 left-over policy：SM 满足线程数要求即可入驻 block，导致同一 kernel blocks 倾向于堆叠在同一批 SM 上（stacked co-location），产生 "1 more, 5 less" 资源利用模式。
- **实现示例**：通过 micro-benchmark 系统测量 Pascal/Volta/Turing 三代 GPU 的 thread block placement → 揭示 left-over scheduling → 为 μShare 等上层优化提供硬件行为基础。
- **论文标题**：*Demystifying the Placement Policies of the NVIDIA GPU Thread Block Scheduler for Concurrent Kernels* (2025)

---

## 四、GPU —— MoE 调度

### 12. Dynamic Expert Placement (ES-MoE)
- **方法思路**：打破传统 EP 中 expert→GPU 的静态映射。借助 expert offloading（expert 从 CPU 按需加载），每 iteration 根据 gating network 输出的 token 数量，用 greedy scheduling (Graham 1969, 4/3-approximation) 动态决定 expert→GPU 映射，目标是最小化各 GPU 间处理时间差异（minimize makespan）。负载均衡决策与 token routing 决策完全解耦。
- **实现示例**：`sorted_experts = sort_by_processing_time(expert_loads, descending=True)` → `target_gpu = argmin(gpu_loads)` → `gpu_assignments[target_gpu].append(expert_id)`。GPU 间 token 数差异从 102% 降至 15%，消除 zero-padding。算法运行在 CPU 上 <2.69μs。
- **论文标题**：*Scaling Beyond the GPU Memory Limit for Large Mixture-of-Experts Model Training* (ES-MoE, ICML '24 / 持续影响至 2025+)

### 13. ScheMoE: EP 任务调度形式化
- **方法思路**：将 MoE layer forward/backward 的 7 类任务（compress、A2A dispatch、decompress、expert compute、compress、A2A combine、decompress）形式化为带数据依赖约束的调度问题，数学证明给定输入分区度 r 下的最优 CompTask 执行顺序（OptSche 算法）。提出 Pipe-A2A 通信算法——将 intra-node 和 inter-node A2A 分配到两个独立 CUDA stream 并发执行，同时利用两种带宽。
- **实现示例**：`OptSche(r)` → 7 类任务最优执行顺序 → Pipe-A2A: Stream a (intra-node SR) ∥ Stream b (inter-node SR) → AbsCompressor/AbsAlltoAll/AbsExpert 三层抽象接口使压缩和 A2A 算法可插拔替换。
- **论文标题**：*ScheMoE: An Extensible Mixture-of-Experts Distributed Training System with Tasks Scheduling* (EuroSys '24 / 持续影响至 2025+)

### 14. ScMoE: Shortcut-Connected Expert Parallelism
- **方法思路**：架构-调度协同设计——通过 shortcut 连接使 gating 和 All-to-All dispatch 可以基于前一层表示提前启动，与当前层的 attention+shared expert 计算重叠。当通信时间 ≤ overlap_window（约 50% 总 MoE 时间）时实现 100% 通信隐藏。
- **实现示例**：`shortcut_gate(prev_layer_hidden)` → 提前计算 gating → All-to-All 与 attention+shared expert 完全重叠 → 8×A30-PCIe（通信占 60%）下 1.49× 训练加速、1.82× 推理加速。
- **论文标题**：*Shortcut-connected Expert Parallelism for Accelerating Mixture of Experts* (ScMoE, ICML '25)

### 15. PopFetcher: Popularity-Based Expert-Wise Prefetch
- **方法思路**：利用滑动窗口（s=10 iterations）预测下一层热门 expert，在 Attention 层期间通过独立 CUDA stream 异步预取 remote expert 参数到本地 GPU。已预取的 expert 的 token 直接本地计算——消除该部分 token 的 All-to-All dispatch。Hybrid push-pull 范式：当 token 传输量 >2048 tokens 时 pull expert，否则 push token。
- **实现示例**：`sliding_window(s=10)` → 预测热门 expert → `cudaMemcpyAsync(expert_params, stream=prefetch_stream)` 与 attention compute 重叠 → 8×RTX 4090 (100Gbps IB) 上 token 传输量减 14.85%，per-iteration 加速 1.28-2.4×。
- **论文标题**：*PopFetcher: Towards Accelerated Mixture-of-Experts Training Via Popularity Based Expert-Wise Prefetch* (USENIX ATC '25)

### 16. Sem-MoE: Semantic-Aware Collaborative Scheduling
- **方法思路**：通过 semantic-aware model-data collaborative scheduling 提升 Local Activation Rate (LAR) 从 25% 到 62-68%。LAR 提升直接减少 EP 中 All-to-All 通信量 49-57%——不修改通信协议本身，而是从数据放置和调度层面使更多 token 路由到本地 expert。
- **实现示例**：Semantic clustering of tokens → 高 LAR 意味着更多 token 在本地 GPU 找到所需 expert → All-to-All 通信量直接减少。DeepSeek-V2-Lite MoE layer 中 All-to-All 占 forward latency 的 59.2%→被大幅削减。
- **论文标题**：*Sem-MoE: Semantic-Aware Model-Data Collaborative Scheduling for MoE Inference* (2025)

### 17. UCCL-EP: 可移植的 Expert-Parallel 通信
- **方法思路**：通过 CPU-proxy 架构解耦 GPU 通信发起与 NIC 通信执行——GPU 通过 FIFO channel 将 TransferCmd 传递给多线程 CPU proxy，CPU 通过 libibverbs（可移植 RDMA 库）执行所有 NIC 操作。仅需 O(m+n) 开发成本（vs 传统 GPU-initiated 通信的 O(m×n)）。在 EFA（无序传输、无硬件 atomics）和 Broadcom NIC 上首次实现 GPU-initiated token-level EP 通信。
- **实现示例**：GPU → FIFO channel → CPU proxy (libibverbs) → RDMA NIC。EFA 上优于 PPLX 2.1×，NVIDIA 上达到 DeepEP 可比水平。
- **论文标题**：*UCCL-EP: Portable Expert-Parallel Communication* (2025)

### 18. MoC-System: Fully Sharded Checkpointing for EP
- **方法思路**：解决 EP 中 checkpoint 保存的 bottleneck rank 问题——传统方案仅用 EP-Group-0 保存所有 expert checkpoint，导致该 rank 负载过高而其他 EP groups 闲置。Fully Sharded Checkpointing 将 expert checkpoint 按 expert 切分在所有 EP groups 间均分。
- **实现示例**：`shard_checkpoint_across_ep_groups()` → 每 EP group 仅保存本 group expert 的 checkpoint → bottleneck workload 降低 22%-29%。
- **论文标题**：*Partial Experts Checkpoint: Efficient Fault Tolerance for Sparse Mixture-of-Experts Model Training* (ASPLOS '25)

---

## 五、专用加速器调度

### 19. SCAR: Chiplet MCM 加速器多模型调度
- **方法思路**：异构 dataflow MCM AI 加速器上调度多模型 workload。搜索空间达 O(10^56)（2 模型在 6×6 chiplet 上），通过三组启发式导航：(1) Dataflow-Layer affinity matching——将各 layer 映射到与其 dataflow 偏好匹配的 chiplet；(2) Inter-chiplet pipelining——在 chiplet 间做流水线利用 package 内高带宽互连；(3) Multi-tenancy spatial partitioning——多模型共享 MCM 的资源隔离与利用率平衡。
- **实现示例**：`affinity_scores = compute_dataflow_affinity(layer, mcm_chiplets)` → WS dataflow chiplet 处理 Conv 层、OS dataflow chiplet 处理 FC 层 → inter-chiplet pipeline depth 优化（约束: interposer BW + chiplet SRAM）。
- **论文标题**：*SCAR: Scheduling Multi-Model AI Workloads on Heterogeneous Multi-Chiplet Module Accelerators* (2025)

### 20. ElasticMoE: NPU 零停机弹性垂直缩放
- **方法思路**：在华为 Ascend 910C NPU 上以 2 NPU 为粒度进行不中断服务的增量式垂直缩放。三个机制：(1) HBM 管理与推理执行解耦（HMM 以持久守护进程独立管理权重和 KV cache）；(2) Scale-while-serve 模型（旧实例持续服务，新实例后台准备，完成后无缝流量切换）；(3) 固定 TP 仅调 DP/EP（保证 attention 权重和 KV cache 布局不变，直接 zero-copy 复用）。
- **实现示例**：HMM 分析 4→6 NPU config → Attention 权重: NPU 0-3 zero-copy, NPU 4-5 HCCL P2P → Expert: 全局 remap → p2p-copy → vpage-remap → IMM pre-initialized 实例 zero-copy attach → Coordinator 无缝流量切换。Scale-up latency ~2.43s, 0 downtime。同时提供 CUDA 等效方案（CUDA IPC + virtual memory API）。
- **论文标题**：*ElasticMoE: An Efficient Auto Scaling Method for Mixture-of-Experts Models* (2025)

### 21. SADDLE: CI-Aware PIM-GPU 异构调度
- **方法思路**：在 HBM-PIM + GPU 异构系统上，每 iteration 计算每个 operator 的算术强度 CI = FLOPs/Bytes，与两设备的 ridge point 比较做 operator→device 映射决策。PIM ridge = 16.7 FLOP/Byte（高带宽低算力），GPU ridge = 208 FLOP/Byte（高算力低带宽）。CI < 16.7 → memory-bound → 去带宽高的 PIM；CI > 208 → compute-bound → 去算力强的 GPU。
- **实现示例**：`CI = FLOPs / Bytes` → `if CI < PIM_RIDGE: return "PIM" elif CI < GPU_RIDGE: return "GPU" else: return "GPU"`。仅需 O(1) 代数运算（一次除法），无需 profiling。动态 scheduling 在 static mapping 基础上再提升 1.13×，整体吞吐提升 1.21×。
- **论文标题**：*SADDLE: CI-Aware PIM-GPU Dispatch for Speculative Decoding* (2025)

### 22. KernelEvolve: 跨异构加速器的 Agentic Kernel Coding
- **方法思路**：LLM agent 驱动的自动 kernel 生成和优化，支持 NVIDIA/AMD/MTIA 多代硬件。将 kernel 优化建模为 LLM-driven iterative graph search——agent 迭代：生成 kernel 候选 → correctness 验证 + multi-level profiling → 根据执行反馈优化 prompt → 生成更优 kernel。通过 Meta's FaaS platform 异步 dispatch 到 remote accelerator pools。
- **实现示例**：Tree search engine (greedy/MCTS) → generation phase (prompt synthesis + KB retrieval + LLM) → evaluation phase (compilation + TritonBench + multi-GPU profiling) → 收敛到 expert-level 性能。支持 H100/A100/MI300X/MTIA v1/v2 五个平台。
- **论文标题**：*KernelEvolve: Scaling Agentic Kernel Coding for Heterogeneous AI Accelerators at Meta* (2025)

### 23. MixNet: 运行时可重配置光电混合互联
- **方法思路**：面向 MoE 训练的运行时可重配置光电混合互联架构。光电路交换机在运行时动态重构 GPU 间拓扑，使 expert-to-GPU 映射可根据 token 负载动态调整网络拓扑——hot expert 获得更高带宽路径，cold expert 共享剩余带宽。
- **实现示例**：Runtime reconfigurable OCS (Optical Circuit Switch) → 动态调整 GPU 间带宽分配 → hot expert path 获得专属高带宽 → 消除静态拓扑下的 All-to-All 拥塞。
- **论文标题**：*MixNet: A Runtime Reconfigurable Optical-Electrical Fabric for Distributed Mixture-of-Experts Training* (2025)

### 24. DFVG: FPGA+GPU 异构 Speculative Decoding
- **方法思路**：Draft model 部署在 FPGA（可重构逻辑实现低延迟 draft）、Verify model 部署在 GPU（高吞吐 tensor core 做 tree verification）。动态 draft budget——根据 confidence 自适应调整 draft length，高 confidence 位置增加 branch、低 confidence 位置减少。FPGA 与 GPU 通过 pipeline 重叠避免串行等待。
- **实现示例**：FPGA draft (low latency, dynamic budget) → GPU TreeSort-Verify (batch verify) → FPGA-GPU pipeline overlap → GPU 从不 idle。
- **论文标题**：*DFVG: A Heterogeneous Architecture for Speculative Decoding with Draft-on-FPGA and Verify-on-GPU* (2025)

### 25. TetriServe: 混合 DiT Workload 弹性调度
- **方法思路**：解决 Diffusion Transformer (DiT) serving 中不同分辨率/denoising steps 请求的 deadline-aware 调度。动态选择 sequence parallelism degree (SP=1/2/4/8)——低分辨率用低 SP 度避免通信开销、高分辨率用高 SP 度满足 deadline。请求级别可抢占，支持动态 SP 度切换。
- **实现示例**：`deadline_aware_sp_selection(resolution, remaining_steps)` → 256×256: SP=1/2 (通信占比<5%)、2048×2048: SP=4/8 (满足 deadline) → GPU 利用 MPS spatial sharing 混合不同 SP 度请求。
- **论文标题**：*TetriServe: Efficiently Serving Mixed DiT Workloads* (2025)

---

## 六、跨层/基础方法

### 26. Nimble AoT Multi-Stream Scheduling（持续影响 2025+）
- **方法思路**：在模型首次执行前完成全部 GPU task scheduling——operator dispatch、kernel selection、output shape inference、memory allocation、kernel argument preparation——将整个模型录制为 CUDA Graph。运行时仅需 cudaMemcpy + cudaGraphLaunch。Ford-Fulkerson 最大流算法分配 operator 到多 stream 实现最大并发（max concurrency=15）。
- **实现示例**：`traced_graph = torch.jit.trace(model, dummy_input)` → `stream_plan = ford_fulkerson_max_flow(dag, max_streams=15)` → `cudaGraphInstantiate(&exec_graph)` → 运行时 `cudaGraphLaunch(exec_graph)`。对 NASNet-A mobile 加速 22.34× vs PyTorch eager。AoT preparation 一次性 ~0.35s。
- **论文标题**：*Nimble: Lightweight and Parallel GPU Task Scheduling for Deep Learning*

### 27. SLO-Aware Load Estimator (ElasticMoE)
- **方法思路**：以 SLO 达标率（TTFT < α 且 TPOT < β 的请求比例）为自动扩缩容的决策依据，而非传统的 GPU utilization 等间接硬件指标。持续跟踪推理实例 SLO 达标率，SLO < 90% 持续 N 个窗口→触发 scale-up，SLO > 99% 且 util < 50% 持续 M 个窗口→触发 scale-down。
- **实现示例**：`slo_rate = count(TTFT<α AND TPOT<β) / total` → `if slo_rate < 90% for N windows: trigger scale-up(DP+1, EP+DP_increment)` → cooldown_period → resume monitoring。
- **论文标题**：*ElasticMoE: An Efficient Auto Scaling Method for Mixture-of-Experts Models* (2025)

### 28. PiLLM: Workload Prediction for Resource-Efficient LLM Inference
- **方法思路**：跨 GPU 和单 GPU 两个层面的资源预测。跨 GPU：预测请求的 token length 分布（而非依赖 GPU utilization 等滞后指标）来做 autoscaling 决策。单 GPU：基于 workload prediction 优化 KV cache eviction 策略（降低 68.39% eviction rate）。
- **实现示例**：Token length predictor → autoscaling 提前扩容（避免利用率的滞后反应）→ GPU 内预测型 KV cache management → GPU 资源节省 + SLO 保证。
- **论文标题**：*PiLLM: Resource-Efficient LLM Inference Using Workload Prediction* (2025)

### 29. Legion Runtime System（持续影响）
- **方法思路**：Stanford 的分布式 task-based runtime。用户声明 logical region（分布式数据集合）和 task（对 region 的操作），runtime 通过 dynamic dependence analysis 在运行时自动发现 task 间依赖、计算并调度所需通信、管理跨 processor 的数据一致性。Diffuse 在其上增加 task fusion + kernel fusion 中间层。
- **实现示例**：cuPyNumeric → Diffuse IR (task fusion + kernel fusion) → optimized task stream → Legion (dynamic dependence analysis + coherence + scheduling + memory) → GPU/CPU 执行。
- **论文标题**：*Composing Distributed Computations Through Task and Kernel Fusion* (Diffuse, 2025)

---

## 七、TPU —— 编译器驱动的调度（2025 相关）

> **注意**：TPU 调度主要在编译期完成（XLA/GSPMD），2025 年无新的 TPU 专属运行时动态调度论文进入 vault。以下为持续影响 2025 的 TPU 调度基础方法。

### 30. GSPMD: 编译器自动并行化（持续影响）
- **方法思路**：用户仅对 <1% 的图中 tensor 标注 sharding annotation，GSPMD 编译器自动推导全图分区方案并插入必要的 collective communication（AllReduce/AllGather/ReduceScatter/All-to-All/CollectivePermute），生成等价但已并行化的 SPMD 程序。支持 DP、In-layer MP、Spatial Partitioning、PP 全部并行范式。
- **实现示例**：`tensor_x = shard(input_tensor, mesh=['data','model'], sharding_spec={'batch':'data'})` → GSPMD propagation → auto-insert AllReduce → TPU executable。Mesh TensorFlow 进一步支持 einsum 自动编译为 all-to-all。
- **论文标题**：*GSPMD: General and Scalable Parallelization for ML Computation Graphs* (Xu et al., 2021 / 持续影响至 2025+)

---

## 八、关键 GPU 并发机制（调度基础设施）

### 31. GPU 并发机制: TS、MPS、MIG、vGPU
- **方法思路**：四种 GPU 并发机制形成从粗到细的隔离/共享谱系。(1) Time Slicing (TS): 不同应用轮流占用 GPU，时间片到期后 Ctx Drain 切换——隔离性最强但抢占延迟大。(2) MPS (Multi-Process Service): 多进程共享 GPU，CUDA stream 级别并发——利用率高但隔离性差。(3) MIG (Multi-Instance GPU): 物理隔离的 GPU 划分（A100 支持 7 个独立实例）——算力和显存完全隔离，partition 间互不影响。(4) vGPU: 关联物理资源的虚拟化封装，为不同粒度任务提供灵活资源分配。
- **实现示例**：TS: runlist 调度不同 channel，interleave frequency 控制优先级 → MPS: 多进程 CUDA stream 并发 → MIG: 物理切分 GPU (算力+显存隔离) → vGPU: 虚拟化封装。μGPU/SCG-SM 在此基础上进一步优化 Ctx 切换和 SM 级调度。
- **来源**：`human_notes/GPU架构笔记/GPU并发机制TS、MPS、MIG、vGPU...md` (score: 3607)

---

## 方法分类速查

| 调度层级 | 方法数量 | 代表论文 |
|---------|---------|---------|
| 集群级 | 2 | Wind, FlexPipe |
| Serving级 | 5 | FinDEP, Bullet, JANUS, Proteus, PiLLM |
| Kernel级 | 5 | ACS, μShare, HuntKTm, Infera, Nimble AoT |
| MoE调度 | 7 | ES-MoE, ScheMoE, ScMoE, PopFetcher, Sem-MoE, UCCL-EP, MoC-System |
| 专用加速器 | 7 | SCAR, ElasticMoE, SADDLE, KernelEvolve, MixNet, DFVG, TetriServe |
| TPU | 1 | GSPMD (持续影响) |
| 基础/跨层 | 3 | SLO-Aware Estimator, Legion/Diffuse, GPU并发机制 |
| **总计** | **30** | |

---

## 附录：2025 前后边界论文说明

以下论文发表于 2024 年会议但对 2025 方法有持续影响，在上文中已酌情保留：
- ES-MoE (ICML '24) —— Dynamic Expert Placement 奠定了 MoE 动态调度的基础范式
- ScheMoE (EuroSys '24) —— EP 任务调度形式化仍是最优调度顺序的理论基础
- GSPMD (2021) —— 作为 TPU 调度的基础框架持续影响
