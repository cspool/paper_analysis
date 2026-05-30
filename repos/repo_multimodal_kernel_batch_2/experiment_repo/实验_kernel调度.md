# 实验_kernel调度

## Tilus: A Tile-Level GPGPU Programming Language for Low-Precision Computation

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现是Tilus的thread-block级VM指令集（Table 1）对低精度tensor计算和内存传输的调度。核心kernel调度机制包括：(1) 软件流水线（software pipelining）——通过CopyAsync/CopyAsyncCommitGroup/CopyAsyncWaitGroup指令实现异步global→shared memory拷贝与计算的overlap，在decode stage batch>1时显著优于Ladder；(2) 显式内存层次调度——LoadGlobal/StoreGlobal操作global memory，LoadShared/StoreShared操作shared memory，CopyAsync异步global→shared拷贝，AllocateRegister管理register分配，开发者精确控制数据placement和movement；(3) 低精度weight loading pipeline——global memory layout预变换 + u8高效加载 + 零开销View reinterpret + 寄存器内PRMT/LOP3 vectorized casting，消除Triton的shared memory layout conversion瓶颈和Ladder的pipelining缺失；(4) auto-tuning——200配置per operator，auto-tune tile大小（BM, BN, BK参数）；(5) k-dimension parallelization支持。

  实验比较：operator级——vs Triton v3.1.0、Ladder (bitblas v0.0.1.dev15)、QuantLLM (commit 9802c5a)、Marlin v0.1.1，在batch size 1和16下评估低精度matmul（uint8, f6, uint4, int4, uint2, uint1）vs cuBLAS FP16的speedup。end-to-end级——将Tilus kernel集成至vLLM v0.5.3，在Gemma-2-9B、QWen2.5-32B、Llama-3.3-70B上评估prefill（2048 tokens）和decode（1/16 batch）延迟。跨batch size评估覆盖decode BS=1,4,8,16和prefill BS=4096,8192,12288。

- 后端平台是什么，配置是什么。
  NVIDIA L40S GPU (48 GiB, Ada Lovelace)，driver 565.57.01，CUDA 12.6.3。跨架构验证：NVIDIA A100 (Ampere, compute capability 8.0)、NVIDIA H100 (Hopper)。H100上Ladder产生非法指令（ERR），vLLM FP16在L40S上OOM（qwen2.5-30B超过48GB）。

- 评估性能的软件/脚本是什么。修改了什么。
  Tilus kernel通过单一参数化Python程序模板生成，支持所有低精度类型。集成至vLLM v0.5.3做end-to-end（artifact使用vLLM 0.7.3）。实验脚本bash run.sh自动拉取Docker镜像并顺序运行所有实验。Docker镜像预装PyTorch v2.5.1、Triton v3.1.0、BitBLAS v0.0.1.dev15、Marlin v0.1.1。kernel性能测量使用CUDA Events，50次执行取median latency，每次执行前清除L2 cache。低精度类型支持通过预处理kernel（图9）将权重变换为标准类型兼容的layout实现。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  开源地址：https://github.com/NVIDIA/tilus
  Artifact: https://github.com/yaoyaoding/tilus-artifacts (DOI: 10.5281/zenodo.16756859)
  Docker image (~21 GiB)，所有依赖预装。

  评估原理：Tilus将GPU kernel执行建模为thread-block级VM指令序列。每条指令（LoadGlobal, CopyAsync, View, Cast, Dot, StoreGlobal等）操作于整个thread block，编译时逐条生成低级GPU代码。性能优势来自：(1) 通过flattened layout预变换避免低精度加载的非连续内存访问；(2) 零开销View reinterpret消除Triton的shared memory layout conversion；(3) CopyAsync软件流水线解决Ladder的pipelining缺失；(4) 寄存器内PRMT/LOP3向量化casting避免shared memory往返。

  Kernel输入到性能输出全过程（FP16×INT4 decode matmul, BS=1, K=8192, N=57344）：
  1. 输入：A f16[BS, K] in global memory (activation cache)；B transformed u8[BK*BN*4/8 per tile] in global memory (权重预变换后的连续u8字节)
  2. 异步预取：CopyAsync将下一K-iteration的B tile从global memory拷贝到shared memory（pipelined with 当前iteration的computation）
  3. 同步：CopyAsyncCommitGroup() + CopyAsyncWaitGroup(0)确保预取完成
  4. 加载activation：LoadGlobal A tile [BM, BK] from global memory → registers, layout=m16n8k16 compatible
  5. 加载weight：LoadShared B tile from shared memory → registers, dtype=u8, layout=local(3).spatial(32)
  6. 零开销reinterpret：View(b_tile, dtype=i4, layout=spatial(8,4).repeat(1,4)) — 32 threads × (4×i4=16 bits) → reinterpret到Tensor Core兼容layout
  7. 向量化casting：Cast(b_tile, f16) — 使用PRMT permute bytes + LOP3 logical ops + bitwise指令在registers内完成
  8. Tensor Core计算：Dot(a_tile, b_tile, C_accum) → PTX mma.m16n8k16, 累加到f32 accumulator
  9. 循环K维：重复Steps 2-8直至K维完成
  10. 输出：Cast(C_accum, f16) → StoreGlobal → global memory 写出结果tile
  11. 测量：CUDA Event记录kernel start/stop → latency = stop - start。50次执行，clear L2 cache between runs，取median
  12. Speedup = cuBLAS FP16 kernel latency / Tilus kernel latency

## SageAttention3: Microscaling FP4 Attention for Inference and An Exploration of 8-Bit Training

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - SageAttention3 的 FP4 attention 使用 CUTLASS + CUDA 实现 kernel，包含三项关键硬件优化：(1) **K 的 Permutation**：FP4 MMA 的 FP32 accumulator 内存布局与 operand A 寄存器布局不匹配，通过重排 accumulator 布局（permute P tile 的列），并对应重排 K 的列（fuse 到量化 kernel 中），避免 thread shuffle 开销。(2) **Reuse Shuffle**：P̃ 的 micro-scaling 量化需要在 16 个连续行元素上找 max，但这 16 个元素分布在 4 个 thread 中。将量化与 online softmax 融合，复用 S 的 16 元素 max 给量化使用，减少 50% shuffle 和 max 操作，整体 kernel 加速约 10%。(3) **Producer Warp Epilogue**：传统 warp-specialized kernel 由 consumer warp 同时处理 MatMul 和 store，producer 只加载。由于寄存器约束，改为 producer warp 间 ping-pong：一个 producer 加载下一轮输入时，另一个 producer 存上一轮输出到 global memory。Consumer warp 只负责将 MatMul 结果从寄存器搬到 shared memory。实现 MatMul 与 global memory store 的 overlap。
  - SageBwd 使用 OpenAI Triton 实现 INT8 前向+反向 attention kernel。
  - 实验比较：与 FlashAttention2（CUDA）、xformers、FlashAttention2 Triton 版本对比 kernel speed（TOPS）和延迟。SageAttention3 在 RTX5090 上达到 1038 TOPS，是 FlashAttention2 的 5×。SageBwd 前向 2× 加速（最高），反向 1.2~1.6× 加速（最高），端到端 forward+backward 最高 1.67× 加速。

- 后端平台是什么，配置是什么。
  - SageAttention3 kernel：NVIDIA RTX5090 (Blackwell, FP4 Tensor Core)
  - SageBwd kernel：NVIDIA RTX4090 (INT8 Tensor Core)
  - 对比 head_dim=64 和 head_dim=128 两种配置

- 评估性能的软件/脚本是什么。修改了什么。
  - SageAttention3：基于 CUTLASS [22] 和 CUDA 自研 kernel，在 FlashAttention tiling 框架上替换 MatMul 为 FP4MMA 指令
  - SageBwd：基于 OpenAI Triton [23] 实现，修改了 attention 前向+反向的量化策略
  - 对比 baselines：FlashAttention2 (CUDA)、xformers、SageAttention、SageAttention2

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源地址：https://github.com/thu-ml/SageAttention
  - Kernel 执行流程：输入 FP16 Q, K, V 分块 → Quantization kernel（含 K transpose fuse 和 Smoothing）将 Q, K, V 量化为 NVFP4（E2M1 + E4M3 scale）→ FP4MMA 指令执行 QK^T → Online Softmax（含 two-level quantization for P，复用 rowmax 做 shuffle reduction）→ FP4MMA 指令执行 PV → Producer warp ping-pong store 输出 O。其中 permutation 优化在 K 量化阶段完成列重排；reuse shuffle 在 softmax 阶段将 max 值共享给 P 量化；producer warp epilogue 通过双 producer warp 交替完成 load 和 store。

## ParallelKittens: Systematic and Practical Simplification of Multi-GPU AI Kernels

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现是ParallelKittens (PK)，一个基于ThunderKittens扩展的C++ CUDA嵌入式编程原语集合，包含8个核心多GPU通信原语和一个统一的LCSC (Load-Compute-Store-Communicate) 编程模板。PK通过三大设计原则指导多GPU kernel开发：(1) 传输机制选择——根据工作负载特性从Copy Engine、TMA和寄存器级指令中选择最优传输机制，PK仅暴露每种功能最高效的传输机制（如TMA用于点对点通信，寄存器操作用于in-network加速）；(2) 调度策略——支持Intra-SM overlapping（同一SM内不同warp并发执行计算和通信）和Inter-SM overlapping（不同SM分别专用于计算和通信），通过LCSC模板统一实现两种调度；(3) 设计开销消除——使用预分配目标缓冲区实现单向传输，避免NCCL的双向同步和中间缓冲，寄存器保存peer地址避免NVSHMEM的重复load和group sync。实验在Data/Tensor Parallelism（AG+GEMM, GEMM+RS, GEMM+AR）、Sequence Parallelism（Ring Attention, DeepSpeed-Ulysses）和Expert Parallelism（MoE token dispatch+GEMM）三类负载上比较PK vs 非overlap基线(cuBLAS+NCCL)、编译器方法(Triton Distributed)、手写kernel(Flux, CUTLASS, Comet)、通信库方法(xDiT, YunChang)。

- 后端平台是什么，配置是什么。
  8×NVIDIA H100 80GB SXM GPU，4th-generation NVLink/NVSwitch (450 GB/s单向带宽)，CUDA 12.6，PyTorch 2.8.0。Blackwell验证平台：8×NVIDIA B200 GPU，5th-generation NVLink/NVSwitch (900 GB/s单向带宽)，CUDA 12.8，PyTorch 2.8.0。所有GEMM使用BF16元素类型和FP32累加器类型。

- 评估性能的软件/脚本是什么。修改了什么。
  PK通过LCSC模板定义了四个worker组件（loader, storer, consumer, communicator），用户只需实现这四个组件的per-tile逻辑，框架自动处理kernel配置、SMEM/TMA设置、barrier/synchronization管理、SM/warp分区调优。实现流程：(1) 定义globals struct（包含设备内存指针和参数）；(2) 定义LCSC template struct（实现loader/storer/consumer/communicator四个静态方法）；(3) 调用lcsc::launch_kernel<config, globals, lcsc_template>(G, stream)启动。每个kernel的通信相关device代码不超过50行。PK提供了8个原语：store_async（TMA异步存储tile到multicast memory）、store_add_async（TMA异步原子加）、reduce（multicast memory到local HBM的in-network reduction）、all_reduce（multicast memory上的in-network all-reduce）、signal（单设备barrier信号）、signal_all（广播barrier信号）、wait（等待barrier值）、barrier（全设备同步）。对比的baseline软件包括：cuBLAS+NCCL（非overlap基线）、Triton Distributed（编译器方法）、Flux/CUTLASS/Comet（手写kernel）、xDiT/YunChang（通信库方法）。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  开源地址：https://github.com/HazyResearch/ThunderKittens。PK作为ThunderKittens的扩展在该仓库中开源，包含所有kernel实现和工具代码。目前正在Cursor公司的in-house训练中被采用。
  
  评估原理：PK通过将通信和计算融合到单个kernel中消除kernel launch开销(T_launch)和非重叠时间(T_non-overlap)，目标是使总时间 T_kernel = T_launch + max(T_comp, T_mem, T_comm) + T_non-overlap + T_sync 中的max项主导。评估时测量观测到的平均计算吞吐量(FLOP/s)。
  
  kernel输入到性能输出全过程（以GEMM+RS fused kernel为例）：
  1. 输入：local GEMM shape M×N×K/8（8 GPU分担K维），输入矩阵A (M×K) 分片在本地HBM，B (K×N/8) 分片在各GPU，输出C需要reduce-scatter到各GPU持有N/8列。
  2. loader worker：使用TMA从本地HBM异步加载A_tile和B_tile到SMEM。Intra-SM overlapping：loader的单线程TMA调用不占用其他warp资源，consumer warp可同时执行MMA。
  3. consumer worker：warpgroup对加载的tile执行mma（tensor core GEMM），累积到寄存器C_accum中。
  4. storer worker：完成K维所有tile的累积后，将输出tile通过TMA store_async写入multicast memory（PGL），同时执行peer-to-peer传输。对于reduce-scatter：每个GPU将其计算结果tile通过store_add_async原子加到对应目标GPU的PGL区域。
  5. 对于inter-SM GEMM+AR：communicator worker在专用communication SM上等待所有compute SM完成本地写（通过barrier同步），然后执行all_reduce原语利用NVSwitch in-network reduction（multimem.ld_reduce）将各GPU的partial结果归约。
  6. 输出：各GPU获得最终结果矩阵的对应分片。最终以TFLOP/s报告compute吞吐量，non-overlapped communication ratio报告通信开销占比。
  
  PK通过num_comm_sms参数控制Inter-SM模式下的通信SM数量，运行时自动搜索最优分配。对于Intra-SM模式，所有SM都同时执行计算和通信，单线程TMA异步调用实现通信重叠而保持所有tensor core繁忙。

## MoDES: Accelerating Mixture-of-Experts Multimodal Large Language Models via Dynamic Expert Skipping

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现是自定义CUDA kernel用于MoE MLLM推理中的dual-modality thresholding和高效的expert计算执行。具体包括：(1) 在router kernel内部实现双模态阈值判定——在计算router logits和top-k后，使用branch-free masked comparison与modality-specific threshold比较，直接将跳过的expert路由设为sentinel expert ID（如M+1），不引入额外的kernel launch或独立的decision pass；(2) Sentinel-aware dispatch/gather——在MoE dispatch/gather阶段自动过滤sentinel entries，跳过专家加载和计算；(3) Group GEMM执行——使用Grouped General Matrix Multiplication将所有活跃experts的矩阵乘法合并到单个统一的kernel launch中并发执行，每个expert的计算为独立的sub-task；(4) Offline profiling + kernel tuning——对不同的代表性激活模式进行离线grid search，确定最优的kernel tile sizes，确保不同动态负载下的高计算吞吐。实验比较baseline kernel与原始模型（k=8/6/4 top-k routing）的prefill/decoding延迟和吞吐量（tokens/s）。

- 后端平台是什么，配置是什么。
  单张NVIDIA H200 GPU（用于inference speed测量），8×H200 GPU用于calibration、search和accuracy evaluation。Software: PyTorch、transformers库、flash-attention2。自定义CUDA kernels用于MoE层thresholding和Group GEMM。

- 评估性能的软件/脚本是什么。修改了什么。
  使用自定义CUDA kernels测量实际wall-clock inference speed。修改：(1) Router kernel——在router内部嵌入thresholding逻辑：router → top-k → apply modality-specific threshold via masked comparison → assign sentinel IDs to skipped routes；(2) MoE dispatch/gather kernel——增加sentinel filtering逻辑，通过检查expert ID是否等于M+1来过滤；(3) Group GEMM kernel——使用离线profiled的tile sizes，支持动态expert激活模式下的并发矩阵乘法。关键设计决策：thresholding逻辑嵌入现有kernel（不增加额外kernel launch），sentinel filtering仅需warp-level的少量元素操作（overhead最小）。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  开源链接：https://github.com/ModelTC/MoDES

  评估原理：
  1. Build：在8×H200 GPU环境中编译自定义CUDA extensions（router + MoE dispatch/gather + Group GEMM kernels）
  2. 加载pre-calibrated GMLG参数α̃^{(l)}和frontier-search找到的最优阈值(τ_t*, τ_v*)
  3. 使用prefill batch size=8、decode sequence length=1024进行inference
  4. 测量prefill time（ms，含所有token的首次forward pass）和decode time per iteration（ms，单token自回归生成）
  5. 计算speedup = original_time / MoDES_time

  全过程（以Qwen3-VL-MoE-30B-A3B-Instruct在单H200上，88% expert skipping ratio，decode阶段为例）：
  ```
  Host: 加载模型 + pre-computed (α̃^{(l)}, τ_t*, τ_v*)
  
  对于每个decode iteration（单个text token）:
  For each transformer layer l in 1..L:
    ┌─ Attention Layer (standard) ─────────────────────────────────┐
    │  Q = X @ W_Q; K = X @ W_K; V = X @ W_V                     │
    │  flash-attention2 → attention_output                         │
    │  X = RMSNorm(attention_output + residual)                    │
    └──────────────────────────────────────────────────────────────┘
    
    ┌─ MoE FFN Layer (MoDES customized) ──────────────────────────┐
    │                                                               │
    │  Router Kernel (fused with thresholding):                    │
    │    ① r = router(X)                        // 128 experts    │
    │    ② π = softmax(r)                                         │
    │    ③ topk = topk_indices(π, k=8)          // 8 candidates   │
    │    ④ for i in topk:                                         │
    │         s_i = α̃^{(l)} · π_i              // pre-computed α̃  │
    │    ⑤ τ = τ_t (text token)                                   │
    │    ⑥ mask = (s_i < τ)  // branch-free comparison            │
    │    ⑦ topk[i] = mask ? M+1(sentinel) : topk[i]              │
    │    → 输出: topk with sentinel entries for skipped experts    │
    │                                                               │
    │  MoE Dispatch/Gather:                                        │
    │    ① for expert_id in topk:                                  │
    │         if expert_id != M+1:                                 │
    │           dispatch token to expert_id's input buffer         │
    │    ② sentinel entries automatically filtered — no compute    │
    │                                                               │
    │  Group GEMM Kernel (single launch for all active experts):   │
    │    ① active_experts = unique(topk) - {M+1}                  │
    │    ② GroupedMatMul(                                         │
    │         X_inputs: [X_active1, X_active2, ...],              │
    │         weights: [W_expert_active1, W_expert_active2, ...]  │
    │       )                                                      │
    │       → Each expert as independent sub-task                 │
    │       → Tile sizes from offline profiling grid search       │
    │       → Single kernel launch, concurrent execution          │
    │    ③ weighted sum: y = Σ π_i · E_i(X)                       │
    └──────────────────────────────────────────────────────────────┘
  
  输出性能：
    - Prefill speedup: ~2.16× (batch=8, Kimi-VL-A3B-Instruct)
    - Decode speedup: ~1.26× (seq_len=1024, Kimi-VL-A3B-Instruct)
    - Qwen3-VL-MoE-30B: ~2.03× prefill, ~1.24× decode
    - 跳过88% expert仅保留小部分活跃expert计算，masked comparison和sentinel filtering开销<1%
    - Decode speedup小于prefill的原因是：(i) decode阶段为memory-bound，(ii) decode仅处理text token，跳过率低于prefill的vision+text混合
  ```

  关键kernel设计要点：
  - Branch-free masked comparison：避免warp divergence，所有threads执行相同操作
  - Sentinel filtering in dispatch：在expert输入buffer分配阶段即过滤，不浪费计算资源
  - Group GEMM offline profiling：由于expert activation pattern因token而异，通过pre-profile多种代表性pattern确定最优tile sizes，运行时按最接近pattern选择
  - Dequantization集成（与MC-MoE结合）：支持2.5-bit和1.5-bit权重，MoDES+量化可达到~10.67×压缩比

## Mirage Persistent Kernel: A Compiler and Runtime for Mega-Kernelizing Tensor Programs

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现是MPK in-kernel parallel runtime——完全嵌入在单个mega-kernel内部的并行运行时系统，包含以下核心组件：(1) SM分区：将GPU的SM物理划分为workers（每个SM一个worker，维护独立task queue）和schedulers（warp粒度，每个SM 4个scheduler warp，维护event queue）；(2) Event-driven execution model：tGraph从start event开始，scheduler dequeue event后dispatch依赖该event的所有tasks到workers，worker执行完成后notify triggering event，event接收足够trigger次数后激活并入队scheduler event queue，循环推进直至所有tasks完成；(3) Hybrid task launch：JIT（Just-In-Time）模式——scheduler在event激活后才assign task，适应workload imbalance但需要worker↔scheduler两次同步；AOT（Ahead-Of-Time）模式——预分配tasks到workers，worker仅需等event激活即可执行，仅需一次同步，消除dispatch开销但缺乏动态负载均衡；(4) Cross-task software pipelining：将每个task分解为pre-loading phase（TMA异步加载数据到shared memory）和compute phase（Tensor Cores/CUDA Cores计算），在当前task compute阶段未结束时即启动下一task的pre-loading，条件是当前task已发出所有data-transfer指令且有足够shared memory page；(5) Paged shared-memory abstraction：将shared memory分为固定大小pages（32KB），task需acquire/release pages，支持跨task的shared memory复用和数据prefetching；(6) Task description prefetching：每个task描述符352 bytes存储在device memory，worker prefetch到shared memory隐藏访问延迟。

  实验比较了MPK vs SGLang/vLLM/PyTorch在单GPU和多GPU（tensor parallelism）下的吞吐量。消融实验：(a) cross-task pipelining对Qwen3-8B final linear layer的影响（B200，1.2-1.3x加速）；(b) compute-communication overlap对Qwen3-1.7B在4×H100 TP下的per-iteration延迟影响（1.1x加速）。

- 后端平台是什么，配置是什么。
  NVIDIA A100（108 SMs, 104 workers + 4 schedulers [16 warps]）、H100（132 SMs, 128 workers + 4 schedulers）、B200（148 SMs, 144 workers + 4 schedulers）。Shared memory page size: 32KB → A100: 5 pages/SM, H100/B200: 7 pages/SM。多GPU：NVIDIA H100 DGX（8×H100, NVLink），tensor model parallelism。精度：bfloat16。Task描述符：352 bytes/task。Task队列：GPU device memory circular buffer，使用atomicAdd操作。Worker-scheduler同步：device memory semaphores。Inter-GPU通信：NVSHMEM nvshmem_signal_wait_until。

- 评估性能的软件/脚本是什么。修改了什么。
  评估在offline batched-inference设置下进行（消除request-arrival变异性），所有请求prompt_len=64、decode 1024 tokens，batch size 1-16。使用PyTorch + torch.compile(backend=MPK)生成mega-kernel并直接测量吞吐量。对比系统SGLang和vLLM使用各自默认配置（FlashInfer/FlashAttention + cuBLAS/cuTLASS + CUDA Graphs）。多GPU实验使用Megatron-LM tensor model parallelism，AllReduce由NVSHMEM实现。

  MPK修改/新增：
  (1) In-kernel page allocation和request scheduling——将传统CPU端的continuous batching逻辑全部移入mega-kernel内的单个task执行：在start event处理时，scheduler (a) 移除上一iteration完成的请求，(b) 接纳新到达请求，(c) 更新per-request KV-cache metadata。消除CPU-GPU同步延迟。
  (2) 支持dynamic batch sizes——编译器为2的幂次batch sizes（up to max_batch）分别生成专用tGraph，运行时按当前batch size选择。
  (3) Moe hybrid workload balancer——编译期静态分配expert-specific tasks，运行期利用topk-softmax产生的global metadata（activated experts数、per-expert token数）动态调整workload分配。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  开源链接：https://github.com/mirage-project/mirage

  评估原理（单GPU offline batched inference throughput测量）：
  1. 模型加载：通过HuggingFace Transformers加载模型架构，bfloat16精度
  2. Mega-kernel编译：torch.compile(backend=MPK) → 生成针对(batch_size, GPU_type)的专用tGraph和mega-kernel
  3. Page attention启用，continuous batching由mega-kernel内scheduler task处理
  4. 预热后测量decode 1024 tokens的总时间，计算throughput (tokens/s)
  5. Batch size = {1, 2, 4, 8, 16}，每个配置测量3次取平均

  全过程（以Qwen3-8B在H100上batch_size=1 decode为例）：
  ```
  Host: torch.compile(backend=MPK)(model) → 编译完成
  Host: mega_kernel() → 单次CUDA kernel launch

  GPU Mega-Kernel 内部执行 (持久运行至所有token decode完成):
  ┌─────────────────────────────────────────────────────────────┐
  │ SM Partitioning: 128 Workers + 4 Scheduler-SMs (16 warps)  │
  │                                                             │
  │ Scheduler (start event e0):                                │
  │   ① Remove completed requests                              │
  │   ② Admit new requests (batch_size=1)                      │
  │   ③ Update KV-cache metadata                               │
  │   ④ Dispatch Q_proj tasks → Workers                        │
  │                                                             │
  │ Worker SM_i (execute Q_proj task):                         │
  │   Pre-load phase: TMA load Q_weight tile → SMEM page       │
  │   Compute phase: Tensor Core MMA (input × Q_weight)        │
  │   └─ 同时: Pre-load K_proj weight tile → another SMEM page │
  │   → 完成, notify event e_Q                                 │
  │                                                             │
  │ Worker SM_j (execute K_proj task): 类似                    │
  │ Worker SM_k (execute V_proj task): 类似                    │
  │                                                             │
  │ Event e_QKV 激活: 所有 Q/K/V tasks 完成                    │
  │ Scheduler → dispatch Attention tasks                       │
  │                                                             │
  │ Worker SM (execute Attention task):                        │
  │   FlashAttention-style kernel on single SM                 │
  │   [JIT mode: 执行时间data-dependent (sequence length)]      │
  │   → 完成, notify event e_Attn                              │
  │                                                             │
  │ Event e_Attn 激活: 所有注意力tasks完成                      │
  │ Scheduler: JIT dispatch O_proj + RMSNorm tasks             │
  │   (workers更快完成attention的获得更多下游tasks → 负载均衡)    │
  │                                                             │
  │ [AOT mode] MLP tasks 已预分配到 workers:                    │
  │   Worker SM: check AOT queue → event已激活? → gate_proj    │
  │   Pre-load gate_weight → compute GEMM →                   │
  │   同时 pre-load up_weight →                                │
  │   SiLU activation → down_proj GEMM →                       │
  │   同时 pre-load 下一层 Q_weight... (cross-task pipelining)  │
  │                                                             │
  │ ... 循环处理所有Transformer layers ...                      │
  │ 直至 generate stop token → mega-kernel return              │
  └─────────────────────────────────────────────────────────────┘

  输出: per-token latency 12.5ms (vs vLLM/SGLang 14.5ms, 理论下限~10ms)
  Throughput: 80 tokens/s (batch=1, decode)
  ```
  对比kernel-per-operator系统（SGLang/vLLM）：每个operator为独立kernel launch → kernel barriers阻止跨算子pipelining和细粒度overlap → CPU端page allocation/scheduling产生额外CPU-GPU同步 → 数百次kernel launch/iteration的overhead在latency-critical场景不可忽略。

## LiquidGEMM: Hardware-Efficient W4A8 GEMM Kernel for High-Performance LLM Serving

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现是LiquidGEMM——一个硬件高效的W4A8 GEMM kernel，包含两项关键技术：(1) LiquidQuant (LQQ)——仅需两条32-bit硬件指令（IMAD + XOR）处理四个元素的overflow-safe dequantization算法；(2) Implicit Fine-Grained Pipeline (ImFP)——single-producer multiple-consumer执行模型，Load WG通过TMA加载weight到SMEM后切分为fine-grained tasks，多个Compute WG竞争获取task并各自完成dequantization+CUDA Core MMA，跨Compute WG实现dequantization与MMA的自然重叠，消除SMEM↔RF round-trip数据搬运和软件同步开销。还包含Dual-MMA packed layout——将两个连续MMA操作所需元素打包存储，每个线程用单条LDS.128指令加载32个UINT4元素。实现使用CUTLASS和Cute编程原语，WGMMA/barrier/TMA等用PTX包装，dequantization逻辑直接用CUDA实现。计算Y=(WX^T)^T替代Y=XW^T以利用WGMMA的m=64固定维度。

  实验比较的baseline kernels：QServe（W4A8，QoQ dequantization算法）、TRT-W4A16、TRT-W8A8、TRT-FP8、TRT-FP16。评估方式：(1) 系统级——LiquidServe vs QServe/TRT吞吐量和延迟；(2) kernel级——使用统一CUDA benchmark框架从各系统抽取GEMM kernel，隔离对比单层transformer所有GEMM（fused QKV projection、output projection、两个FFN GEMM）延迟，batch size 4-256。消融实验：逐步启用LQQ、ExCP（显式粗粒度pipeline）、ImFP（隐式细粒度pipeline），对比各组件贡献。

- 后端平台是什么，配置是什么。
  NVIDIA H800 GPU（80GB HBM, Hopper架构）。WGMMA指令支持INT8 MMA（m64nNk32/m64nNk64, N∈[8,256]）。TMA用于异步数据搬运。软件：PyTorch 2.4.0，CUDA 12.4，CUTLASS/Cute。

- 评估性能的软件/脚本是什么。修改了什么。
  使用内部统一CUDA benchmark框架（"An internal benchmarking tool used to evaluate GPU kernel performance before deployment"）对各系统抽取的GEMM kernel进行公平对比，支持灵活配置矩阵形状以模拟各种模型场景。每次测量5次取平均。修改：(1) 自研LiquidGEMM kernel——基于CUTLASS/Cute编程原语构建warp-specialized ping-pong kernel，fuse dequantization到MMA mainloop，实现Dual-MMA packed layout数据加载，ImFP pipeline替代ExCP；(2) kernel计算改写——从Y=XW^T改写为Y=(WX^T)^T以更好地利用WGMMA指令。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  开源情况：LiquidGEMM未提供开源代码。论文明确说明"LiquidGEMM is currently deployed as the primary GEMM kernel in our production LLM serving infrastructure"。评估kernel性能使用内部benchmark工具，外部不可获取。

  评估原理：
  1. 统一benchmark框架从各系统（LiquidServe, QServe, TRT）中抽取纯GEMM kernel，去除系统级因素（attention、KV cache管理等）干扰。
  2. 对每个模型（LLaMA2-7B/13B/70B, LLaMA3-8B, Mistral-7B, Mixtral-8×7B）的单层transformer所有GEMM（QKV fusion projection、output projection、两个FFN GEMM）分别测量延迟。
  3. Batch size从4到256遍历，每次5次运行取平均。
  4. 消融实验：从baseline（无LQQ/无pipeline）开始，逐步启用LQQ → ExCP/ImFP，测量每步加速比。

  全过程（以LiquidGEMM处理FFN层W4A8 GEMM为例，M=batch_size, N=hidden_dim×intermediate_factor, K=hidden_dim）：
  ```
  Host: 启动LiquidGEMM kernel(grid=(m×n thread blocks), block=(384 threads=3 WGs))
  
  Per Thread Block (处理Mt×Nt输出tile, 在K维度迭代):
    // ImFP: 1 Load WG + 2 Compute WGs, 共3 WGs
    
    Load WG (4 warps, TMA + CUDA Cores):
      for k_iter in 0..K/Kt:
        // 异步weight加载
        cp.async.bulk (TMA): GMEM[weight_tile_u4] → SMEM[buffer_ping]  // Dual-MMA packed layout
        cp.async.bulk.commit_group
        cp.async.bulk.wait_group
        
        // 将weight tile切分为fine-grained tasks写入SMEM task queue
        // 每个task = 一个WGMMA fragment所需weight（64×32 UINT4 elements）
        smem_task_queue.push(task_metadata)
        // 切换到pong buffer

    Compute WG_0 (4 warps, CUDA Cores + Tensor Cores):
      for task = smem_task_queue.pop():  // hardware-managed scheduling, no software sync
        // Step 1: 从SMEM加载weight到RF
        LDS.128: RF[0:31] = SMEM[task.weight_addr]  // 32 UINT4 elements, 1 instruction
        
        // Step 2: Unpack 4-bit → 8-bit (QServe method)
        // 8 × 4-bit elements in reg → 2 × 32-bit regs with 8-bit elements
        unpack_lo(reg_w0, w_packed)
        unpack_hi(reg_w1, w_packed)
        // (repeat for all 4 packed regs → 8 regs of 8-bit)
        
        // Step 3: Dequantization with LQQ (CUDA Cores)
        // Equation 12: Q_i8 = (Q_u4 * s_u8 + a) XOR 0x80
        // 2 instructions per 4 elements:
        r0 = IMAD(r0, s_broadcast, a_broadcast)  // multiply-add
        r0 = XOR(r0, 0x80808080)                 // flip MSB of each byte
        // (total: 7 instructions for 8 elements incl. unpack)
        
        // Step 4: MMA (Tensor Cores)
        warpgroup.mma.fence  // ensure dequantization results visible
        WGMMA.m64nNk32: C_frag += A_frag(INT8) × W_frag(INT8)
        // 使用dequantized weight作为INT8 MMA输入
        
    Compute WG_1: 同时竞争获取不同task，dequantization与MMA自然与WG_0重叠
      // WG_0做dequantization时, WG_1可能在做MMA, 反之亦然
      // 无需软件同步——由硬件task scheduling管理

    // Epilogue: 第一级dequantization (INT8→FP16) + 写回GMEM
    C_fp16 = C_int32 * s_i8 (per-channel scale)
    store GMEM[output_tile] = C_fp16
  ```

  性能输出（以LLaMA2-7B FFN GEMM为例，batch=256）：
  - LiquidGEMM: 2.90x speedup vs QServe W4A8 kernel
  - LQQ alone (memory-bound): limited benefit; LQQ alone (compute-bound): up to 1.29x speedup
  - ImFP vs ExCP: ImFP consistently better across all batch sizes;
    ExCP degrades at small batch due to round-trip traffic + sync overhead
  - LiquidGEMM vs TRT-FP8 on LLaMA2-7B: 1.12-1.58x speedup
  - LiquidGEMM vs TRT-W4A16 on Mixtral-8×7B: 1.12-2.53x speedup

## HipKittens: Fast and Furious AMD Kernels

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现是基于ThunderKittens的C++ embedded tile-based编程原语，移植并重新设计AMD GPU上的高性能AI kernel框架HipKittens (HK)。核心实现包括：(1) 8-WAVE PING-PONG和4-WAVE INTERLEAVE两种wave调度模式，替代NVIDIA的wave specialization模式；(2) 开发者可控的寄存器分配（pinned register tiles），绕过HIPCC编译器对AGPR寄存器使用的限制；(3) 针对AMD CDNA异构MFMA指令形状的共享内存swizzle优化，解决bank conflict；(4) chiplet感知的L2/LLC两级缓存grid调度算法（Algorithm 1: XCD swizzle for cache reuse）。实验比较的baseline包括：AMD AITER（手写汇编）、Composable Kernel (CK)、PyTorch SDPA/torch.compile、HipBLASLT、ROCm Triton、Mojo。评估的workload包括：BF16 GEMM、FP8 GEMM、FP6 GEMM（初步）、MHA/GQA Attention forward/backward（causal/non-causal, d=64/128）、fused dropout-residual-layernorm、RoPE。

- 后端平台是什么，配置是什么。
  AMD CDNA4 MI355X OAM GPU（8 XCD chiplet，256 CU，BF16 2.5 PFLOPs，288GB HBM，8.0 TB/s带宽）。AMD CDNA3 MI325X GPU。AMD MI350X GPU。对比平台：NVIDIA B200 SXM5（2.2 PFLOPs BF16，180GB，8.0 TB/s）。软件环境：ROCm 7.0 Docker (rocm/7.0-preview:rocm7.0_preview_pytorch_training_mi35x_beta)。

- 评估性能的软件/脚本是什么。修改了什么。
  自研HK C++ kernel通过Python bindings在Python脚本中benchmark。每个kernel 500次warmup + 100次测量取平均TFLOPs/s，输入为N(0,1)随机张量。AITER通过aiter.flash_attn_func调用，PyTorch通过torch.nn.functional.scaled_dot_product_attention，CK通过编译tile_example_gemm_basic/tile_example_fmha_fwd/tile_example_fmha_bwd二进制运行，HipBLASLT通过hipblaslt-bench命令行。修改：基于ThunderKittens框架，重写所有tile原语以包装AMD CDNA assembly/HIP（替代NVIDIA PTX/CUDA），新增pinned register tile接口、8-wave/4-wave调度模板、chiplet swizzle算法。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  开源链接：https://github.com/HazyResearch/HipKittens

  评估原理：
  1. HK以C++ header-only库形式提供，开发者使用tile原语（rt_bf/rt_fl/st_bf等）编写kernel，通过Python pybind11绑定调用。
  2. BF16 GEMM kernel（Fig. 21）以256×256输出tile/thread block、16×16×32 MFMA指令为基本单元，采用8-wave ping-pong调度：每SIMD 2个wave交替执行compute cluster（MFMA矩阵乘）和memory cluster（buffer_load_dword从HBM到LDS，ds_read从LDS到register）。
  3. Attention forward kernel（Fig. 23）以32×128 tile/wave为输出单元，同样采用8-wave ping-pong，在compute cluster内交替执行online-softmax vector ops（max/subtract/exp2/accumulate）和MFMA指令，通过sched_barrier hints指导LLVM编译器精确调度vector和matrix指令的交错。
  4. Cache优化：Algorithm 1在kernel启动前remap block indices，将连续C个block分配给同一XCD（L2复用），以W高度的垂直窗口遍历输出矩阵（LLC复用）。

  全过程（以BF16 GEMM为例）：
  ```
  用户调用HK GEMM kernel(D=AB+C, M=N=K=8192, dtype=BF16)
    → Algorithm 1: 根据M/N tile数、XCD数(8)、W和C参数，计算remap后的block坐标(row, col)
    → 每个thread block负责256×256输出子矩阵
    → Prologue: 8 waves协作preload A/B tile从HBM到shared memory (buffer_load_dword)
    → Conditional barrier: 4个leader wave继续preload，4个follower wave等待
    → Hotloop: leader和follower交替执行
        Cluster 0: load B_tile_0从shared到register (ds_read_b128) → load A_tile → G::load next As → s_barrier
        Cluster 1: __builtin_amdgcn_s_setprio(1) → mma_ABt(C[0][0], A, B_tile_0) → s_setprio(0) → s_barrier
        Cluster 2: load B_tile_1 → G::load next Bs → s_barrier
        Cluster 3: mma_ABt(C[0][1], A, B_tile_1) → s_barrier
        Cluster 4-7: 对称处理C[1][0]/C[1][1]，完成compute和memory的overlap
    → Epilogue: store C_accum tiles回HBM
    → 测量：rocm-profiler或wall-clock计时，500 warmup + 100 iters → 报告TFLOPs/s
  ```

## GPU Scheduling on the NVIDIA TX2: Hidden Details Revealed

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现是通过黑盒实验和公开文档，逆向工程 NVIDIA TX2 GPU 的内部调度行为。具体包括：通过合成 benchmark（可配置 block 资源需求、kernel 持续时间、copy 操作）测量 GPU 上每个 thread block 的 start/end 时间（使用 globaltimer 寄存器），推导出 TX2 GPU 调度器在多 stream、多 task 场景下对 kernel dispatcher 和 copy engine 的调度规则（G1-G4, X1, R1-R3, C1-C4, N1-N2, A1-A2）。实验比较了：(1) task 共享地址空间 vs process 独立地址空间的调度行为差异；(2) NULL stream 对并发的阻塞影响；(3) stream priority 的抢占和饥饿行为；(4) 多 process 下 time-slicing/preemption 的开销。

- 后端平台是什么，配置是什么。
  NVIDIA Jetson TX2 嵌入式开发板。SoC 设计：四核 2.0GHz 64-bit ARMv8 A57 + 双核 2.0GHz superscalar ARMv8 Denver + 集成 Pascal GPU（2 个 SM，每个 SM 128 核 @ 1.3GHz，共享 512KB L2 cache）。6 核 CPU 与 GPU 共享 8GB 1.866GHz DRAM。GPU 有 1 个 Copy Engine (CE)。

- 评估性能的软件/脚本是什么。修改了什么。
  自研合成 benchmark（synthetic workload），可配置：每个 block 的线程数、shared memory 用量、kernel 持续时间（通过 spin loop）、copy 操作大小。使用 CUDA 8.0.62 的 globaltimer 寄存器在 GPU 端记录每个 block 的 start/end 时间戳，CPU 端记录 kernel launch 时间。可视化工具绘制 GPU timeline（每 block 的起止时间、SM 分配、thread 数）。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  开源链接：https://github.com/yalue/cuda_scheduling_examiner_mirror

  评估原理：
  1. 创建合成 CUDA kernel，每个 block 执行固定时长的 spin loop（如 1 秒），block 配置指定线程数（768/512/1024/256 等）和 shared memory（0/32KB）。
  2. 多个 kernel 按实验设计通过不同 stream（含 NULL stream、不同 priority）在不同 CPU task/process 上 launch。
  3. 每个 block 在 GPU 端通过 `clock64()` 或 `globaltimer` 记录自己的 start 和 end 时间戳，写入全局内存数组。
  4. CPU 端在 kernel 完成后读取时间戳数据，绘制 Gantt-chart 式 GPU timeline（x 轴为时间，y 轴为 SM，每个矩形为一个 block）。
  5. 从 timeline 中观察：kernel 何时 dispatch、block 在哪个 SM 执行、copy 操作与 kernel 执行的重叠关系、优先级抢占行为。
  6. 对比不同实验条件下的 timeline，推导调度规则。

  全过程：
  ```
  CPU task/process 调用 CUDA API launch kernel K(k) with N blocks, T threads/block, S shared_mem/block
    → CUDA runtime 将 GPU operation 入队到对应 stream queue (Rule G1)
    → 当 kernel 到达 stream queue 头部，入队到 EE queue (Rule G2)
    → GPU scheduler 检查 EE queue 头部 kernel 的 block 是否满足资源条件 (Rules R1-R3, X1)
    → 若有 SM 满足 threads ≤ 2048, shared_mem ≤ 64KB, registers ≤ 65536，则分配 block 到该 SM
    → Block 在 SM 上执行 spin loop → 记录 start/end 时间戳
    → 所有 block 完成后，kernel 从 EE queue 出队 (Rule G3)，从 stream queue 出队 (Rule G4)
    → CPU 端读取时间戳 → 生成 timeline 可视化
  ```

## Memory-Efficient Acceleration of Block Low-Rank Foundation Models on Resource Constrained GPUs

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现是五套针对Monarch和BLAST矩阵乘法的自定义Triton kernel优化，解决多token推理场景下BLR方法因中间数据移动和内存访问模式差导致的memory-bound瓶颈：
  
  **Monarch优化①②③（联合使用，提供累加效率增益）：**
  - ① V矩阵重排布（Re-layout of V）：将V的存储格式从(b₁, r'b₂, p)改为r'先连续再b₂（原来b₂先连续再r'），消除推理时的r'↔b₂ permutation。此优化在离线阶段对静态权重执行一次。
  - ② 排列融合（Permutation fusion）：将b₂↔b₁ permutation与第一个bmm融合为单个Triton kernel。Kernel计算b₁×t_n×t_r个输出tiles，permutation通过计算b₂索引→调整r'偏移→用swapped indices写出来实现（Fig. 5 pseudo-code）。
  - ③ 避免最终permutation：当Monarch线性层输出立即与静态权重相乘时，离线pre-permute该权重的行，消除推理时的(b₂,n,q)→(n,q,b₂) kernel launch。
  
  **BLAST优化④和⑤（分别应用，代表不同策略）：**
  - ④ bmm部分融合（Partial fusion of bmm）：消除V和S之间的中间permutation和第一个bmm输出在global memory中的物化。每个thread block在内部循环b₁维度，加载S的(b₂,t_r) tile并广播与第一个bmm的(1,t_n,t_r)输出做累加batched outer product。牺牲tensor core利用率（第二个bmm跑CUDA cores），但避免了大中间张量(b₁,b₂,n,r)。
  - ⑤ 仅排列融合+Tensor Core优化（Permutation-only fusion with tensor core optimization）：转置S和U的第一和最后一维（S^T, U^T），从左侧乘，在每个kernel内transpose中间输出tiles。保持n连续，r/b₁/b₂依次作为三个kernel的batch维度，每个实现transposed bmm with outer-dimension reordering。消除permutation开销同时保持高tensor core利用率（via Triton dot()）。

  实验比较的baseline：BLAST repository (Lee et al. 2024)和Monarch repository (Dao et al. 2022)的PyTorch实现，均使用Triton auto-tuner和torch.compile()。评估方式：(1) layer-wise speedup——对每个模型的所有(B)LR替换层单独benchmark延迟；(2) end-to-end throughput——整个模型用torch.compile() + CUDA graph后测量prefill throughput（语言模型）、单步inference（扩散模型）、标准前向（视觉模型）。消融实验：BLAST ④ vs ⑤ 的tradeoff（④用CUDA cores做第二个bmm牺牲tensor core throughput，⑤用transpose保持tensor core但引入额外transpose开销）。

- 后端平台是什么，配置是什么。
  NVIDIA A40（40GB显存，6MB L2 cache，BF16 tensor core支持，HBM2e带宽约696 GB/s）。NVIDIA Jetson Orin Nano 8GB（边缘GPU，4-6MB L2 cache，DDR DRAM带宽约68 GB/s，2048 CUDA cores + 64 tensor cores）。软件：A40用Python 3.12.8、PyTorch 2.8.0、Triton 3.4.0、CUDA 12.6.3；Jetson用JetPack 6.2、L4T 36.4.3、CUDA 12.6.11、PyTorch 2.6.0、Triton 3.2.0。Triton autotuner sweep tile sizes（32-256 powers of two）、threads per block、pipelining stages。

- 评估性能的软件/脚本是什么。修改了什么。
  使用Triton do_bench() utility对每个layer单独benchmark：多次执行取平均延迟，warm-up消除cold-start（kernel compilation + cache population），torch.no_grad() + torch.cuda.synchronize()确保异步CUDA完成。端到端用torch.utils.benchmark.Timer()多次迭代forward pass。所有benchmark使用torch.compile() + CUDA graph capture。Baseline实现来自BLAST和Monarch开源repository，已包含Triton autotuner和torch.compile()优化。
  
  修改：(1) Monarch kernel——重写V存储layout + 融合permutation到bmm + 可选pre-permute下游权重；(2) BLAST kernel——两个变体（④partial fusion用CUDA cores batched outer product、⑤permutation-only fusion用tensor core transpose）；(3) 全部使用Triton编写，利用dot()算子做tensor core MMA，自定义tile sizes和内存布局。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  开源链接：https://github.com/pabillam/mem-efficient-blr

  评估原理：
  1. Layer-wise benchmark：对每个模型的每个线性层类型（QKVproj、Oproj、gate/upproj、downproj、c_attn、c_fc、c_proj、fc1、fc2等），用各方法（Dense/LR/Monarch/BLAST × baseline/optimized kernel）分别执行多token前向。Triton do_bench()自动多次warmup+计时取平均。输入为随机生成的tensor，shape由模型层配置（Table 3）和序列长度n决定（语言模型n=1024-2048, DiT n=16K, ViT n=197）。
  2. End-to-end benchmark：整个模型加载压缩权重后执行完整前向推理，torch.compile() wrapping整个网络以启用CUDA graph，torch.utils.benchmark.Timer()计数迭代取平均。语音模型报告prefill throughput（tokens/s），扩散模型报告单步latency，视觉模型报告标准forward latency。
  3. 消融：BLAST ④ vs ⑤分别对每种层和模型benchmark，分析tensor core vs CUDA core的tradeoff。

  全过程（以Llama-7B QKVproj层，n=1024, i=o=4096, r=1024, b=16在A40上，Monarch ②优化为例）：

  ```
  Host: 调用 Monarch 线性层 forward(X: [1024, 4096])
    # V ∈ R^{16 × 1024 × 256}（已重排布为 r'=64 先连续）
    # U ∈ R^{16 × 256 × 1024}

  Step 1: 输入reshape
    X_blocks = X.view(1024, 16, 256)   # [n, b₁, p]

  Step 2: Fused perm+bmm kernel (优化②, Fig. 5 pseudo-code)
    Triton Kernel Launch: grid=(b₁, ceil(n/t_n), ceil(r/t_r))
    # b₁=16 块, t_n=64, t_r=128 (通过autotuner选择)

    For each thread block (b_1, n_tile, r_tile):
      # 计算permutation target indices
      b_2 = (r_tile * t_r + [0:t_r-1]) // r'   # ★ target b_2 index
      r'_offset = (r_tile * t_r + [0:t_r-1]) % r' + b_1 * r'  # ★ adjusted r' offset

      acc = zeros(t_n, t_r)   # accumulator tile in registers
      for p_tile in range(0, p, t_p):   # p=256, t_p=64
        x = X_blocks[b_1, n_tile*t_n:(n_tile+1)*t_n, p_tile*t_p:(p_tile+1)*t_p]
        v = V[b_1, p_tile*t_p:(p_tile+1)*t_p, r_tile*t_r:(r_tile+1)*t_r]
        acc += triton.dot(x, v)   # Tensor Core MMA, t_n×t_p @ t_p×t_r → t_n×t_r

      # 写入输出，使用swapped indices完成permutation ★
      Z_out[n_tile*t_n:(n_tile+1)*t_n, b_2 * n * r' + r'_offset] = acc

  Step 3: 第二批bmm（U）
    # Z_out shape after kernel: [n, b₂·b₁·r'] effectively [n, b₂, b₁·r']
    Triton Kernel Launch for batch matmul: Z_out × U
    # 产生Y: (b₂, n, q) → 若输出连residual/add，则需最终permutation

  Step 4 (可选, 优化③): 若Y随后与静态W_down相乘
    # 离线已pre-permute W_down行 → 跳过在线permutation kernel

  性能输出（A40上）：
    - Monarch ② vs baseline Monarch: Qproj 1.46× speedup, Kproj 1.58×, Vproj 1.62×, Oproj 1.37×
    - Monarch ①+②+③ vs baseline: 综合 1.46-2.37× speedup across layers
    - BLAST ⑤ vs baseline BLAST: DiT-XL/2 QKVproj up to 7.15× on Jetson
    - BLAST ⑤ vs dense: up to 3.76× (GPT2-S c_fc layer on Jetson)
    - End-to-end BLAST ⑤: 1.13-1.48× over dense across all models
    - BLAST ④ < BLAST ⑤ consistently（因为④的CUDA core batched outer product < tensor core throughput 16×）
  ```

## HuntKTm: Hybrid Scheduling and Automatic Management for Efficient Kernel Execution on Modern GPUs

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现是stream scheduler，它自动将multi-kernel程序中的kernel按数据依赖关系分配到多个CUDA stream（即多个hardware queue）上并发执行。核心包括三部分：(1) DFG constructor——通过轻量级代码标注（在kernel参数列表开头插入writable参数个数常量）自动分析kernel间的RAW/WAR/WAW数据依赖，构建数据流图（DFG）；(2) kernel distributor——将DFG分层（同一层内kernel无数据依赖），按PP-Set（preferred predecessor set）大小排序，通过三条规则（无前驱的round-robin分配、单前驱的同stream放置、多前驱的选最少未调度后继的前驱所在stream）将kernels分配到多stream；(3) synchronization generator——基于依赖传递性和同stream内串行执行的隐式同步，剪除冗余同步屏障。实验比较的baseline包括：Serial（串行执行）、Taskflow（静态调度）、GrSched（动态调度）。单任务场景下评估kernel执行加速比，多任务场景下评估system throughput和硬件资源利用率（SM occupancy, FP32 utilization, memory bandwidth utilization via DCGM）。

- 后端平台是什么，配置是什么。
  NVIDIA A100 (40GB HBM, 6912 CUDA cores) 和 NVIDIA RTX 4090 (24GB)。实验在配备4× NVIDIA A100 GPU、2× AMD EPYC 7742 64核处理器、256 GB DDR4的服务器上进行。操作系统Debian 10.2.1，NVIDIA driver 555.42.06。另一平台配备4× NVIDIA RTX 4090 24GB GPU、2× Intel Xeon Gold 6338N CPU、1024 GB DRAM。

- 评估性能的软件/脚本是什么。修改了什么。
  使用NVIDIA DCGM (Data Center GPU Manager) 低开销GPU系统监控工具周期性采集硬件指标（SM occupancy、FP32 utilization、memory bandwidth utilization）。benchmark包括7个代表性应用：VEC (Vector Square, DFG width=2), B&S (Black & Scholes, width=10), ML (Machine Learning, width=2), IMG (Image Processing, width=3), DL (Deep Learning, width=2), M1 (Micro-1, width=8), M2 (Micro-2, width=6)，其中M1/M2的kernel来自NVIDIA FasterTransformer。修改：HuntKTm通过LLVM pass自动将串行CUDA代码转换为多stream并发代码，仅需开发者在kernel参数列表添加一个常量标注writable参数个数（每kernel一行LoC），无需手动编写任何CUDA stream/event管理代码。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  开源链接：https://github.com/Gemini321/HuntKTm

  评估原理：
  1. HuntKTm以LLVM pass形式编译时自动分析CUDA host IR，识别`__cudaPushCallConfiguration`调用点来定位kernel launch。
  2. DFG constructor通过遍历kernel参数中标注的writable参数信息，逆序遍历kernel调用序列，用BFS识别每个kernel的直接前驱（基于同数据对象的读写冲突判断RAW/WAR/WAW依赖）。
  3. kernel distributor对DFG分层后逐层分配kernel到stream，核心理念是最小化跨stream同步数量。
  4. synchronization generator遍历每个stream中的kernel，三步剪除冗余屏障：(i)为不在同stream的前驱创建屏障；(ii)每个stream仅保留来自最后前驱的同步；(iii)同stream内核执行顺序带来的隐式同步消除冗余屏障。
  5. 最终输出stream graph（含多stream执行信息和最小同步指令集），编译为可执行程序。

  全过程（以M2为例，包含多个activation和reduction kernel）：
  ```
  用户编写串行CUDA程序，在kernel定义处添加writable参数个数标注
    → LLVM pass: DFG constructor 自动识别kernel间数据依赖，构建DFG（M2的DFG宽度=6）
    → LLVM pass: kernel distributor 将kernel分层并分配到6个stream
       Level 1: round-robin分配无依赖kernel到stream 1-6
       Level 2-N: 按PP-Set大小排序，单前驱kernel跟随后继同stream，多前驱选最少未调度后继的前驱
    → LLVM pass: synchronization generator 创建跨stream event同步，剪除冗余barrier
    → LLVM pass: 编译生成多stream可执行程序
    → 运行时: 程序在多stream上并发执行kernel → NVIDIA MPS支持跨进程kernel在同一GPU上space-sharing
    → DCGM周期采集SM occupancy/FP32 utilization/memory bandwidth → 计算throughput和加速比
  ```

## KernelEvolve: Scaling Agentic Kernel Coding for Heterogeneous AI Accelerators at Meta

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现是KernelEvolve通过LLM agent驱动的图搜索自动生成针对多后端平台的优化Triton kernel。具体包括：(1) 跨平台kernel合成——从统一operator specification自动生成NVIDIA GPU (CUDA/Tensor Cores/TMA)、AMD GPU (Infinity Cache/ROCm)和MTIA (SFU/PE array/inter-PE communication)的优化kernel实现；(2) 算子融合（Operator Fusion）——如conv1d将5个kernel融合为2个（消除NCHW↔NHWC layout转换和中间tensor materialization），WuKong Optimized FM将2次bmm融合为单kernel（消除intermediate HBM round-trip），InterFormer PFFN将FFN+GELU+RMSNorm融合为single-pass kernel（从3次memory pass减少到1次），MapIdTransform将4个PyTorch算子（bucketize+clamp+gather+where）融合为单kernel（消除3个中间tensor materialization）；(3) Shape-specific tiling——针对生产shape定制tile大小以保持SRAM residency；(4) 硬件特定优化——NVIDIA H100上利用TMA+Warp Specialization+Tensor Cores+double-buffering+differentiated cache modifiers（.ca/.cg），AMD上利用Infinity Cache-aware tiling，MTIA上利用SFU LUT operations+inter-PE broadcast/reduction+runtime barriers+compile-time loop unrolling+SIMD-vectorized counting+adaptive block sizing+cb_multiplier/use_dual_core compilation options；(5) Cross-operation tile reuse——PFFN中同一tile加载后完成全部算子链（matmul+bias+GELU+RMSNorm）再写回HBM；(6) Batched parallel execution——Batch Event Truncate中将多feature并行处理替代sequential per-feature loop；(7) Register-resident computation——中间结果保持在寄存器中，无intermediate tensor allocation。

  实验比较：
  - OSS Operator：160个ATen operators，对比KernelEvolve-generated Triton kernel vs PyTorch torch.compile baseline
  - KernelBench Level 1/2/3：共250个problems
  - Convolutional Transformer conv1d：Triton kernel vs PyTorch conv1d vs PyTorch conv2d workaround，FP16/FP32，多种batch sizes (32-2048) 和 shapes
  - Cross-platform conv1d：5个硬件平台 (H100, A100, MI300, MI350, MTIA v3)
  - WuKong Optimized FM：Fused Triton kernel vs PyTorch torch.compile (two extern_kernels.bmm)，多种 (B,N,D,K) production shapes
  - InterFormer PFFN：Fused Triton kernel vs PyTorch torch.compile (extern_kernels.bmm + triton_per_fused_rms_norm_add_gelu)，多种 (B,N,D,K) shapes
  - MapIdTransform：Fused Triton kernel vs PyTorch (bucketize+clamp+gather+where)，MTIA v2i/v3，多种 (UniqueIDs × Batch) 配置
  - MBDT：Fused Triton kernel vs PyTorch torch.compile，MTIA v2i/v3，多种 Batch × Features × Borders 配置
  - Batch Event Truncate：Batched Triton kernel vs non-batched PyTorch sequential，多种 feature counts (1/5/9/32) 和 event lengths

- 后端平台是什么，配置是什么。
  NVIDIA H100 GPU（Hopper, TMA, WGMMA, mbarriers, 多级cache hierarchy），NVIDIA A100 GPU（Ampere），AMD MI300 GPU（Infinity Cache, CDNA架构），AMD MI350 GPU（CDNA4），MTIA v2i（8×8 PE array, per-PE: dual RISC-V cores + SFU/DPE/RE/SIMD/MLU/CP fixed-function units, on-chip SRAM, custom inter-PE communication），MTIA v3（next-gen, improved compute throughput and native operator coverage）。LLM backends：Claude 4.5, GPT-5, Meta CWM, Llama on Twine（64K-1M token context windows）。Profiling工具：TritonBench, Triton MPP (NVIDIA), Triton Proton (intra-kernel), NCU (NVIDIA kernel-level), Torch Profiler (system-level), MTIA Insight (MTIA-specific: PE utilization, DPE/SFU/MLU utilization, cache hit rates, memory bandwidth, per-PE read/write counters)。

- 评估性能的软件/脚本是什么。修改了什么。
  使用TritonBench (https://github.com/meta-pytorch/tritonbench) 作为主要benchmark框架，通过BenchmarkOperator wrapper进行correctness验证（torch.allclose with precision-dependent tolerances）和speedup测量。Torch Profiler用于system-level timeline capture（CPU/GPU time, kernel launch overhead）。NCU用于NVIDIA kernel-level hardware metrics（occupancy, memory throughput, instruction mix）。Triton Proton用于intra-kernel instruction-level profiling。Triton MPP (Multi-Pass Profiler) 统一跨工具profiling数据采集。MTIA Insight用于MTIA-specific metrics（PE utilization, fixed-function accelerator metrics, cache analysis, memory bandwidth, load-store throughput）。Evaluation Code Generator自动将LLM生成的kernel artifact转换为各profiling tool的instrumented evaluation script，通过FaaS平台dispatched到remote hardware执行。
  
  修改：KernelEvolve的核心修改是构建了完整的agentic kernel优化pipeline，而非修改单个profiling工具。它通过Evaluation Code Generator自动合成evaluation harness（输入：标准化的PytorchModel+TritonModel+get_inputs() artifact；输出：各profiling tool的executable Python script），将profiling从手动操作转化为图搜索反馈信号。关键架构决定：evaluation代码是deterministically generated（保证reproducibility），而kernel逻辑是LLM-generated（允许创造性优化），两者通过标准接口解耦。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  KernelEvolve框架是Meta内部生产系统，论文未提供完整开源链接。系统依赖的开源组件：
  - TritonBench: https://github.com/meta-pytorch/tritonbench（kernel benchmark框架）
  - KernelBench: LLM kernel generation benchmark
  - KernelAgent: https://github.com/meta-pytorch/KernelAgent
  - KernelLLM: https://huggingface.co/facebook/KernelLLM
  - TLX: https://github.com/facebookexperimental/triton

  评估原理：
  1. KernelEvolve的Evaluation Framework将kernel评估抽象为多维度fitness signal采集：correctness（binary pass/fail）、performance（speedup ratio）、profiling metrics（occupancy, memory throughput, instruction mix, intra-kernel pipeline behavior）。这些信号综合决定图搜索节点的fitness score和下一轮优化方向。
  2. Evaluation Code Generator（deterministic）将标准化的dual-implementation artifact（PytorchModel + TritonModel + get_inputs()）转换为各tool的instrumented script，确保across-kernel-variant的reproducible profiling。
  3. FaaS-based remote evaluation将kernel评估dispatch到对应硬件平台的remote worker（NVIDIA/AMD/MTIA），利用pre-deployed Bento interpreter environments消除per-kernel compilation overhead。
  4. Fitness Function: F(v) = t_pytorch / t_triton，correctness失败或compilation/runtime错误的kernel F(v)=0。

  全过程（以MapIdTransform kernel在MTIA v2i上为例）：
  ```
  输入：KernelEvolve生成的Triton kernel artifact
    kernel_n.py:
      class PytorchModel(nn.Module):  # PyTorch reference: bucketize+clamp+gather+where
      @triton.jit def mapid_kernel(...):  # Fused Triton: binary search + clamp + match
      class TritonModel(nn.Module):  # Grid launch wrapper
      def get_inputs():  # Test cases: various (UniqueIDs, Batch) sizes

  Step 1 - Evaluation Code Generation (Section 3.4.3):
    → Deterministic code generator解析artifact
    → 生成 TritonBench harness:
        class MyOperator(BenchmarkOperator):
          def get_input_iter(self): return get_inputs()
          def run(self, inputs): return TritonModel()(*inputs), PytorchModel()(*inputs)
    → 配置: baseline=True (correctness), speedup measurement enabled
    → 生成 MTIA Insight instrumentation:
        mtia_insight.start(); kernel[grid](x, ...); metrics = mtia_insight.stop()
    → 生成 Torch Profiler script: torch.profiler.profile() around kernel launches

  Step 2 - FaaS Remote Evaluation (Section 3.4.6):
    → Dispatch evaluation request to FaaS endpoint: meta_kernel_mtia_interpreter
    → Remote MTIA worker: load pre-deployed Bento interpreter (Triton-MTIA compiler + MTIA Insight + runtime)
    → 执行evaluation script:
        a) TritonBench:
           - PyTorch baseline执行: torch.bucketize → torch.clamp → torch.gather → torch.where
             (部分ATen ops在MTIA v2i上缺少native支持 → CPU fallback + host-device sync)
             执行时间: t_pytorch
           - Triton kernel执行: 单次launch, fused binary search in registers + coalesced block-parallel
             执行时间: t_triton
           - Correctness验证: torch.allclose(pytorch_output, triton_output, atol=..., rtol=...)
        b) MTIA Insight profiling:
           - PE utilization (%), SFU LUT utilization (%)
           - Memory bandwidth: DRAM read/write bytes per PE
           - Per-PE CPU runtime, cache hit rates (I/D-cache, LLC)
           - Load-store throughput per PE
        c) 返回结构化结果:
           {correctness: true, speedup: 3.48, pytorch_ms: 1.623, triton_ms: 0.466, ...}

  Step 3 - Feedback Loop (Section 3.1):
    → Fitness score: F(v) = 1.623 / 0.466 = 3.48
    → Context Memory Sub-Agent分析:
        - 诊断: 4 PyTorch ops → 1 fused kernel, 4× memory traffic reduction
        - 发现: speedup随batch size增大（0.78× at 2000 → 4.07× at 50000）
        - 建议: 小batch时kernel launch overhead主导, 可探索persistent kernel
    → Metadata store更新: (id, pid, score=3.48, is_buggy=false, path_ref)
    → 分析报告写入: object store overview.md

  输出（记录在metadata store中）：
    - correctness: 100% pass across all test shapes
    - speedup: 3.48× on MTIA v2i (100 × 10000), 4.07× peak (10000 × 50000)
    - 跨平台: 1.05-1.36× on MTIA v3（baseline更强）
  ```

## Stream-K: Work-centric Parallel Decomposition for Dense Matrix-Matrix Multiplication on the GPU

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现是Stream-K——一种work-centric的GEMM并行分解策略，将MAC-loop迭代（而非输出tile）作为跨SM的workload量子化单元。核心实现包括：(1) 将GEMM的总MAC-loop迭代次数（total_iters = ⌈m/BLK_M⌉ × ⌈n/BLK_N⌉ × ⌈k/BLK_K⌉）均匀分配给g个CTA，每个CTA执行iters_per_cta = ⌈total_iters/g⌉个MAC-loop迭代；(2) CTA的迭代范围沿m→n→k线性化连续映射，可跨越output tile边界；(3) 当一个CTA的起始/结束迭代不与tile边界对齐时，通过temporary global storage交换partial sums，由执行该tile的k=0 MAC-loop迭代的CTA负责累积并写出最终结果；(4) "two-tile Stream-K + data-parallel"混合调度——限制iteration balancing仅在最后部分数据并行wave的tile范围内，确保每个输出tile最多被两个CTA覆盖，隐藏inter-CTA同步延迟并改善cache locality；(5) 基于解析模型的grid size选择——建模CTA运行时间为固定开销a + partial sum输出开销b + 每迭代MAC开销c + 每协作CTA的partial sum累积开销d，最小化该模型以选择最优g。

  实验比较：(1) vs 同blocking factor的data-parallel CUTLASS kernel（衡量量化效率提升）；(2) vs cuBLAS ensemble (CUDA 11.6)；(3) vs CUTLASS oracle（始终选择最优data-parallel blocking factor的理想化ensemble）。FP64 oracle从5种blocking中选择，FP16→32 oracle从4种blocking中选择。评估覆盖32,824个GEMM shapes（m,n,k ∈ {128...8192}，对数采样，计算体积跨越六个数量级）。

- 后端平台是什么，配置是什么。
  NVIDIA A100 GPU（108 SM cores），功率锁定400W，SM时钟锁定1005 MHz（~71%动态峰值）。FP64 Tensor Core峰值吞吐13.9 TFLOP/s，FP16→32 Tensor Core峰值吞吐222.3 TFLOP/s。Software: CUDA 11.8, CUTLASS 2.11。

- 评估性能的软件/脚本是什么。修改了什么。
  CUTLASS library（https://github.com/NVIDIA/cutlass）。修改：(1) 在CUTLASS的CTA-wide MacLoop() subroutine之上实现Stream-K grid-level decomposition（Algorithm 5），将total_iters均匀分配到g个CTA，每个CTA沿m→n→k线性化执行其迭代范围；(2) 实现partial sum的temporary global storage交换和accumulation逻辑——使用StorePartials/LoadPartials/Signal/Wait原语进行inter-CTA同步；(3) 实现"two-tile Stream-K + data-parallel"混合调度策略；(4) 实现基于解析模型的grid size选择启发式——参数a/b/c/d通过微基准（microbenchmark）每target architecture一次性经验测量后静态编译入库。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  开源地址：https://github.com/NVIDIA/cutlass（自CUTLASS 2.11起包含Stream-K实现）。使用CUDA 11.8编译可复现论文性能。

  评估原理：
  1. 为每种精度（FP64, FP16→32）构建单个Stream-K kernel（使用"two-tile Stream-K + data-parallel"混合分解）。
  2. 对32,824个GEMM shapes逐一执行warmup + timing测量，计算GFLOPS/s和处理器利用率百分比。
  3. CUTLASS data-parallel baseline使用相同blocking factor的data-parallel kernel；cuBLAS使用CUDA 11.6的ensemble；CUTLASS oracle对每个shape枚举所有候选blocking factor取最优。
  4. 绘制roofline图（利用率% vs 计算强度ops/byte）展示性能响应的一致性（narrow band = better）。

  全过程（以FP16→32 GEMM, m×n×k = 384×384×128, BLK_M=128, BLK_N=128, BLK_K=32, g=4 CTAs为例）：
  ```
  Host: 调用Stream-K GEMM kernel
  
  Step 1: Grid size selection (启发式模型)
    total_tiles = ⌈384/128⌉ × ⌈384/128⌉ = 3 × 3 = 9
    iters_per_tile = ⌈128/32⌉ = 4
    total_iters = 9 × 4 = 36
    → 选择 g = 4 CTAs（最优由解析模型确定）
    → iters_per_cta = ⌈36/4⌉ = 9

  Step 2: Grid launch (4 CTAs on 4 SMs)
    CTA_0: iter ∈ [0, 9)   → tile 0 (iter 0-3) + tile 1 (iter 4-7) + tile 2 start (iter 8)
    CTA_1: iter ∈ [9, 18)  → tile 2 end (iter 12-13) + tile 3 (iter 12-15) + tile 4 (iter 16-17) + ...
    CTA_2: iter ∈ [18, 27) → ...
    CTA_3: iter ∈ [27, 36) → ...

  Step 3: Per-CTA execution (以CTA_0为例)
    while iter < 9:
      tile_idx = iter / 4 = 0
      local_iter = 0, local_iter_end = min(9, 4) - 0 = 4
      
      MacLoop(tile_0, iter=0, iter_end=4): 完整执行tile 0的4个MAC-loop迭代
        for iter in 0..4:
          kk = iter × 32
          frag_a = LoadFragment(A, mm=0, kk)  // 128×32
          frag_b = LoadFragment(B, kk, nn=0)  // 32×128
          MMA: accum += frag_a × frag_b        // 128×128×32 MACs per iter
      
      tile_started=true → 不写partials（k=0被CTA_0覆盖）
      tile_ended=true → 直接StoreTile(C, tile_0, accum)
      
      iter = 4
      → tile_idx = 1, local_iter = 0, local_iter_end = min(9, 8) - 4 = 4
      MacLoop(tile_1, 0, 4): 完整执行tile 1
      → StoreTile(C, tile_1, accum)
      
      iter = 8
      → tile_idx = 2, local_iter = 0, local_iter_end = min(9, 12) - 8 = 1
      MacLoop(tile_2, 0, 1): 仅执行tile 2的第1个MAC-loop迭代
      
      tile_started=true → 不写partials（CTA_0覆盖了k=0）
      tile_ended=false → 等待CTA_1完成tile 2的剩余iterations
        Wait(flags[1])
        accum += LoadPartials(partials[1])
      → StoreTile(C, tile_2, accum)

  Step 4: Inter-CTA partial sum consolidation
    CTA_0执行tile_2的k=0迭代时仅做了1个MAC-loop迭代
    CTA_1执行了tile_2的后3个MAC-loop迭代（iter 9-11, local=1-3）
      → CTA_1: StorePartials(partials[1], accum) + Signal(flags[1])
      → CTA_0: Wait(flags[1]) + accum += LoadPartials(partials[1])

  Step 5: 性能输出
    → 100% quantization efficiency（所有4个SM执行相同9个MAC iterations）
    → vs data-parallel (9 CTAs for 9 tiles on 4 SMs = 3 waves, last wave 1/4 active = 75% utilization):
      4/3× 理论利用率提升
    → 实测FP16→32平均1.63× vs CUTLASS data-parallel, 最大14.7×（极端强伸缩scenario m×n小k大）
    → vs cuBLAS FP16→32: 平均1.13×, 最大6.74×
    → vs CUTLASS oracle: 平均1.12× (FP16→32), 证明Stream-K达到tile-based方法无法企及的利用率
  ```

  关键设计要点：
  - Stream-K的单位workload（1个MAC-loop iteration = BLK_M×BLK_N×BLK_K MACs）比data-parallel的单位（1个output tile = ⌈k/BLK_K⌉ MAC-loop iterations）小32×，因此量化效率远优于tile-based方法
  - Communication/synchronization/global storage overheads与问题规模无关，仅与CTA数g（处理器宽度）成正比，即O(p)
  - 当tile数 > CTA数时，每个tile最多被2个CTA覆盖，synchronization-waiting因producer-consumer时间偏移自然隐藏
  - 单kernel、单tile size配置即可超越需要复杂heuristics的20+ kernel ensemble（cuBLAS），可执行代码体积减少20×

## Nimble: Lightweight and Parallel GPU Task Scheduling for Deep Learning

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现是Nimble DL执行引擎，包含两个核心技术：(1) Ahead-of-Time (AoT) Scheduling——在模型执行前，通过CUDA Stream Capture API预运行模型一次（使用dummy input），拦截所有GPU kernel调用和内存分配，生成CUDA Graph（包含dispatched GPU kernels、函数参数、提交顺序和task-to-stream assignment的记录）。运行时通过CUDA Graph Launch API重放，完全绕开PyTorch框架的runtime scheduling overhead。(2) Automatic Multi-Stream Execution——自动将算子分配到多个CUDA stream在单GPU上并行执行。核心是stream assignment算法：计算DAG的Minimum Equivalent Graph (MEG)，从MEG构建bipartite graph，通过Ford-Fulkerson算法寻找maximum matching，将独立节点分配到不同stream同时最小化跨stream同步。理论证明该算法实现最大逻辑并发度+最小同步数。

  实验比较：(a) vs PyTorch inference speedup；(b) vs TorchScript speedup；(c) vs Caffe2 speedup；(d) vs TensorRT v7.1 inference speedup；(e) vs TVM v0.6.1 inference speedup；(f) vs PyTorch training speedup；(g) Nimble single-stream vs multi-stream 消融实验。

- 后端平台是什么，配置是什么。
  NVIDIA V100 GPU + 2.10GHz Intel Xeon CPU E5-2695 v4。Software: PyTorch v1.4, CUDA 10.2, cuDNN 8.0.2。使用TorchScript graph作为输入。

- 评估性能的软件/脚本是什么。修改了什么。
  Nimble基于PyTorch构建，使用TorchScript graph作为输入。用户只需将PyTorch模型包装进Nimble对象（两行额外代码）：`nimble_model = nimble.Nimble(model)`，然后对`nimble_model`执行inference或training。修改：(1) Graph Rewriter —— 执行stream assignment算法，将TorchScript graph中的算子分配到多个CUDA stream；(2) AoT Scheduler —— 使用CUDA Stream Capture API记录GPU kernel call trace（包括kernel launches和memory allocations）并生成CUDA Graph；(3) 实现部分operator fusion（不如TensorRT aggressive）和Conv算子的basic kernel selection（cuDNN vs PyTorch native）。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  开源链接：https://github.com/snuspl/nimble

  评估原理：
  1. 将PyTorch模型包装为Nimble对象（`nimble.Nimble(model)`）
  2. AoT preparation阶段：使用dummy input预运行模型，Nimble通过CUDA Stream Capture API记录所有GPU kernel调用和内存分配，生成优化的CUDA Graph（含多stream分配）。AoT preparation平均耗时0.35s，最大1.07s（NASNet-A large），这是一次性开销。
  3. 运行时：对每个新输入，Nimble直接通过CUDA Graph Launch API重放录制的trace，完全绕开PyTorch的operator dispatch、shape inference、kernel selection和argument preparation等框架开销。
  4. 多stream并行：重放时自动在多个CUDA stream上并发执行独立kernel，跨stream同步点由stream assignment算法最小化。
  5. 测量inference latency（batch size 1）和training throughput。

  全过程（以NASNet-A mobile在V100上的inference为例）：
  ```
  Step 1: 用户代码
    import nimble
    model = torchvision.models.nasnetamobile()  # 加载PyTorch模型
    nimble_model = nimble.Nimble(model)          # 包装为Nimble对象
  
  Step 2: AoT Preparation (执行一次)
    dummy_input = torch.randn(1, 3, 224, 224)
    nimble_model.prepare(dummy_input)
    
    Nimble内部流程:
    ┌─ TorchScript Trace ───────────────────────────────────────┐
    │ torch.jit.trace(model, dummy_input)                       │
    │ → 生成TorchScript graph（包含所有算子及其依赖）             │
    └───────────────────────────────────────────────────────────┘
    
    ┌─ Graph Rewriter (Stream Assignment) ──────────────────────┐
    │ ① Build computation DAG from TorchScript graph            │
    │ ② Compute Minimum Equivalent Graph (MEG)                  │
    │ ③ Construct bipartite graph from MEG                      │
    │ ④ Ford-Fulkerson maximum matching → stream assignment     │
    │ ⑤ 每个算子被分配到一个CUDA stream，独立算子分配到不同      │
    │    stream，有依赖的算子通过CUDA event同步                   │
    │                                                             │
    │ Example: NASNet-A cell (branch structure):                 │
    │   sep_conv1 (stream 0)  ∥ sep_conv3 (stream 1)           │
    │   sep_conv2 (stream 0)  ∥ sep_conv4 (stream 1)           │
    │   sep_conv5 (stream 0)  ∥ sep_conv6 (stream 1)           │
    │   → concat (stream 0, sync across streams)                │
    └───────────────────────────────────────────────────────────┘
    
    ┌─ AoT Scheduler (CUDA Graph Capture) ─────────────────────┐
    │ ① CUDA Stream Capture API: cudaStreamBeginCapture(stream) │
    │ ② 在多个stream上执行模型forward pass:                    │
    │    - 每个stream按分配执行其算子                            │
    │    - Memory allocation被拦截和记录                         │
    │    - CUDA kernel launches、arguments全部被record           │
    │ ③ cudaStreamEndCapture(stream, &graph)                    │
    │ ④ CUDA Graph Instantiation: cudaGraphInstantiate(&exec)   │
    │ ⑤ 输出的CUDA Graph包含:                                   │
    │    - 所有GPU kernel调用序列                                │
    │    - 多stream执行的拓扑                                    │
    │    - 跨stream同步点                                       │
    │    - Pre-allocated memory buffers                         │
    │ AoT preparation time: 0.35s (mean), 1.07s (NASNet-A large)│
    └───────────────────────────────────────────────────────────┘

  Step 3: Runtime Inference (每次新输入)
    output = nimble_model(new_input)
    
    执行流程:
    ┌─ CUDA Graph Replay ──────────────────────────────────────┐
    │ cudaGraphLaunch(exec, stream)                             │
    │                                                             │
    │ GPU端直接执行预录制的kernel序列（无PyTorch框架参与）:       │
    │                                                             │
    │ Stream 0: sep_conv1 → sep_conv2 → concat_wait             │
    │ Stream 1: sep_conv3 → sep_conv4 → concat_signal           │
    │ ↓ cudaEventSynchronize (cross-stream barrier)              │
    │ Stream 0: concat → avg_pool → fc → softmax                │
    │                                                             │
    │ 完全绕过PyTorch的runtime overhead:                        │
    │   ✗ operator dispatch (no Python/C++ operator lookup)     │
    │   ✗ output shape inference                                │
    │   ✗ GPU kernel selection (already recorded)               │
    │   ✗ kernel argument preparation (already recorded)        │
    │   ✗ memory allocation (pre-allocated in AoT phase)        │
    └───────────────────────────────────────────────────────────┘
  
  输出性能：
    - vs PyTorch inference: up to 22.34× speedup (NASNet-A mobile, batch_size=1)
    - vs TensorRT v7.1: up to 2.81× speedup (NASNet-A mobile)
    - vs TVM v0.6.1: up to 1.70× speedup (EfficientNet-B5)
    - Multi-stream vs single-stream (Nimble内部): up to 1.88× speedup
    - Training speedup (CIFAR-10 small models): up to 3.61×
    - GPU idle time in baseline: PyTorch up to 91%, TensorFlow up to 71%
    - Large model training (BERT, ResNet-50 ImageNet): limited speedup（kernel本身计算量大，框架overhead占比小）
  ```

  关键设计要点：
  - CUDA Graph Capture使AoT preparation仅需一次dummy forward pass，之后所有执行跳过框架runtime
  - Stream assignment algorithm的理论保证：最大并发度+最小同步数，通过MEG+Bipartite Matching实现
  - 限制：仅支持static neural network models（不支持dynamic control flow），与TensorRT类似
  - 框架overhead来源：output shape inference, kernel dispatch, kernel argument preparation, memory allocation等多个方面，非仅memory allocation一项
  - Nimble与TensorRT/TVM的关系：正交优化。TensorRT/TVM做graph optimization和kernel tuning，Nimble做runtime scheduling overhead消除。Nimble叠加operator fusion后超越TensorRT（除MobileNet V2的TVM特例：TVM花>1天tuning Conv kernel找到了比cuDNN更高效的实现）

## Twilight: Adaptive Attention Sparsity with Hierarchical Top-p Pruning

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现是Twilight的多项GPU kernel优化，基于FlashInfer构建：(1) **Efficient SpGEMV with INT4 K cache** —— 使用per-head asymmetric INT4 quantization存储额外K cache（在shared memory中解包+dequantize），cp.async异步加载实现2-stage pipeline（加载+计算overlap），将SpGEMV的global memory access降至1/4。Dequantization参考QServe采用per-head动态量化（FP16 scale+zero），利用FasterTransformer的PTX汇编实现INT4→FP16快速类型转换；(2) **Top-p via Binary Search kernel** —— 修改FlashInfer的top-p sampling kernel用于attention weights。采用parallel-friendly binary search（Algorithm 1），element-wise操作(max/where/sum)融合为单次tensorized GPU循环，不物化中间变量W0，避免O(N log N)排序；(3) **Head-wise varlen attention** —— 支持MHA的head级动态budget和GQA的group级动态budget。GQA下每query group取各head选择token的union，使用flattened paged KV cache layout，复用FlashInfer的load balancing算法（flatten head dim）处理head间不平衡；(4) **SpGEMV kernel优化** —— INT4 K元素bit-packed到uint8_t buffer（2×4-bit per byte），地址计算remap到4-bit granularity（halving effective byte offset），interleaved packing简化dequantization。

  实验比较：(a) Self-attention operator —— FlashInfer-Twi vs FlashInfer, Quest-Twi vs Quest, vs FlashAttention2, vs PyTorch SDPA (Memory-Efficient Attention backend)，batch=32-256, seq_len=10k-30k, 测量latency和speedup；(b) End-to-end decoding —— Quest-Twi vs Quest vs FlashInfer, batch=32-256, 测量TPOT (Time-Per-Output-Token)；(c) Ablation —— time breakdown (TokenSel+SpGEMV+Top-p+Attention), quantization bits vs compute time (Figure 12); (d) Offloading scenarios —— Quest vs Quest-Twi, tokens loaded from CPU memory。

- 后端平台是什么，配置是什么。
  单张NVIDIA A100 GPU。Software: PyTorch, CUDA, OpenAI Triton, FlashInfer (https://github.com/flashinfer-ai/flashinfer)。SpGEMV kernel基于FlashInfer的attention decoding kernel修改，top-p kernel修改自FlashInfer的top-p sampling kernel。Per-head动态量化参数（FP16 scale+zero）使用paged memory layout存储。

- 评估性能的软件/脚本是什么。修改了什么。
  使用FlashInfer作为基础kernel库。修改：(1) 新增SpGEMV kernel——将FlashInfer的decode attention kernel修改为sparse GEMV（q_fp16 @ K_int4），加入INT4 dequantization逻辑，使用cp.async + 2-stage software pipeline，FP16 dequantized K cache（非FP32以优化计算）；(2) 新增top-p binary search kernel——修改FlashInfer的top-p sampling kernel（原用于LLM token sampling）用于attention weight累积概率计算；(3) 修改attention kernel——支持head-wise/group-wise varlen attention with flattened paged KV cache；(4) Quest kernels修改——支持batch inference（原始Quest仅支持batch=1）。

  对比baselines：FlashInfer (原始attention kernel), Quest (SOTA sparse attention runtime), FlashAttention2 (xformers Memory-Efficient Attention backend), PyTorch SDPA。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  开源地址：https://github.com/tsinghua-ideal/Twilight

  评估原理：
  1. 使用Longbench中三种类型任务(Qasper/GovReport/LCC)的10k-30k prompts进行batch inference
  2. 自研kernel通过CUDA/Triton实现，集成至FlashInfer的attention pipeline
  3. 测量self-attention operator latency (CUDA Events) 和 end-to-end TPOT (wall-clock)
  4. 每配置warmup后多次测量取平均

  Kernel输入到性能输出全过程（以Quest-Twi, batch=64, seq_len=32k, LLaMA-3.1-8B-Instruct decode step为例）：
  ```
  Host: 启动Twilight attention pipeline (3个kernel launch)
  
  Kernel 1: Token Selector (Quest SpGEMV with FP16 K cache)
    Input: q ∈ R^{BS×H×d}, K_paged ∈ R^{N×d}
    ① Quest: max_pool K to page granularity (16 tokens/page)
    ② SpGEMV: q @ K_pooled^T → approximate scores
    ③ Top-k: select top B0/16 pages → expand to B0 tokens
    Output: I0 indices (B0 tokens), mask_0
  
  Kernel 2: Twilight Pruner (SpGEMV + Top-p)
    Input: q, K_int4 ∈ R^{N×d/2} (paged, per-head dynamic quantized)
    ① SpGEMV with INT4:
       - Load: cp.async from GMEM[K_int4] → SMEM (2-stage pipeline)
         Thread 0-31: async load next K_int4 tile while computing current tile
       - Dequantize in SMEM: unpack UINT4 → apply (K_int4 - zero) * scale → FP16
         Use FasterTransformer-style PTX asm for INT4→FP16 fast conversion
       - Dot product: q_fp16 @ K_fp16 in registers → W_approx[I0]
    ② Softmax: W_norm = softmax(W_approx[I0])
    ③ Top-p Binary Search (Algorithm 1, fully tensorized):
       - l=0, r=max(W_norm), B1=0
       - Loop (typically 8-12 iterations for ε=0.01):
         a. mask = (W_norm >= (l+r)/2)  // where op
         b. cumsum = sum(mask * W_norm)  // fused where+sum
         c. if cumsum >= p: l=(l+r)/2  // threshold too low → raise
            else: r=(l+r)/2              // threshold too high → lower
       - M = (W_norm >= l), B1 = count(M)
       - 所有element-wise/mask/sum操作在单次register循环中完成
    Output: I1 indices (B1 tokens, B1 << B0), M mask
  
  Kernel 3: Sparse Attention (varlen attention)
    Input: q, K[I1], V[I1] (仅加载B1个token的FP16 KV cache)
    ① GQA group union (for LLaMA-3.1):
       - For each query group (e.g., 4 Q heads → 1 KV head):
         I_group = union(I_head1, I_head2, I_head3, I_head4)
         B_group = |I_group|
    ② Flatten head dimension:
       - Concat all group token sets → [total_pruned_tokens]
       - Load balance: FlashInfer scatter-arrange by token count per group
    ③ Sparse FlashAttention:
       - Load Q tile [BM, d], K tile [B_group, d], V tile [B_group, d]
       - Online softmax: S = Q @ K^T / sqrt(d)
       - P = exp(S - rowmax), l = rowsum(P)
       - O += P @ V, update m, l
       - Finalize: O = diag(l)^{-1} @ O
    Output: O ∈ R^{BS×H×d}
  
  性能测量:
    - Self-attn latency (μs): CUDA Event record start/stop per kernel
    - Speedup = FlashAttention2 latency / Twilight latency
    - FlashInfer-Twi: 6.5× vs FA2, 2.4× vs FlashInfer (seq_len=32k, batch=64)
    - Quest-Twi: 15.8× vs FA2, 1.4× vs Quest (seq_len=32k)
    - End-to-end TPOT: Quest-Twi 3.9× vs FlashInfer, 1.35× vs Quest
    - Time breakdown (batch=64, 32k):
      TokenSel ~15%, Pruner(SpGEMV+Top-p) ~20%, SparseAttn ~65%
    - Offloading: Quest-Twi 7.2-16.1× vs Quest (10k-30k)
  ```

  关键kernel设计要点：
  - INT4 SpGEMV的memory access降至1/4 → memory-bound kernel直接受益于量化
  - Top-p binary search避免sorting：O(log(range/ε)) vs O(N log N)，且tensorized on GPU
  - Head-wise dynamism → load balancing：将不同head的不同budget展开为flat load，消除padding waste
  - GQA group union：trade-off accuracy vs repeated loading，实测group varlen优于head-wise varlen（重复加载）和padded（浪费计算）
  - 2-bit K cache精度不足（累积注意力权重显著下降），8-bit浪费带宽，4-bit是最优trade-off

## Kitsune: Enabling Dataflow Execution on GPUs

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现包括两部分：(1) 基于L2 cache的inter-CTA ring buffer queue——通过global atomics实现CTA间同步数据传递，使用CUDA API将queue pin在L2 cache中。Queue为双buffer（两个entry）设计，使用sequence number在producer/consumer间同步，acquire/release API自动处理sequencing。每个entry的metadata由atomic操作保护，synchronization变量全部cache line对齐避免false sharing。Queue操作仅由CTA内一个线程（threadid==0）执行，release需CTA级barrier `__syncthreads()`。(2) Modified GPU Grid Scheduler——将原有的单round-robin arbiter扩展为两个arbiter（SIMT和Tensor各一个），通过cudaPipeline API指定spatial pipeline内每个kernel的primary resource type（SIMT或TENSOR），scheduler按类型选择对应arbiter进行CTA dispatch，确保同一SM上同时有不同类型的CTA colocated以实现资源互补。

  实验比较：(1) Queue性能微基准——测量无同步 vs 有同步的inter-CTA通信带宽（54 queues/108 CTAs对应A100的108 SMs），payload 1KB-2048KB，测出aggregate bandwidth达2 TB/s（37 GB/s/queue）；(2) 端到端应用加速比——5应用×inference/training vs BSP baseline和vertical fusion；(3) SM/DRAM利用率对比——Kitsune vs BSP vs TensorRT在4种utilization组合下的runtime占比；(4) 硬件敏感性——2× SM, 2× L2 bandwidth, 2× DRAM bandwidth下的加速比变化。

- 后端平台是什么，配置是什么。
  NVIDIA A100 GPU（108 SMs, 192 KB shared memory/SM, L2 cache bandwidth ≈ 3× HBM bandwidth）。Queue微基准在真实A100硅片上测量。端到端应用性能通过NVArchSim (NVAS) GPU simulator评估——NVAS是NVIDIA内部的混合trace/execution-driven GPU simulator，已针对Ampere架构验证。硬件敏感性实验通过修改NVAS的machine parameters（SM count、DRAM bandwidth、L2/crossbar bandwidth）进行。

- 评估性能的软件/脚本是什么。修改了什么。
  自研C++ queue library + PyTorch Dynamo compiler backend + NVAS GPU simulator。修改：(1) Queue library——纯软件实现，基于CUDA atomics（`atomicAdd`, `atomicCAS`等）实现inter-CTA的ring buffer queue，提供acquire/release API；(2) Modified NVAS——将单arbiter grid scheduler改为双arbiter（SIMT/Tensor），并在kernel call header中增加type metadata，修改CTA dispatch逻辑使其优先将不同类型的CTA配对到同一SM；(3) CUDA kernel改写——每个融合的DL算子kernel约8人时的手动改写，将10-40行代码从global memory读写改为queue读写。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  论文为NVIDIA研究团队发表，论文未提供开源链接。Queue library可在真实GPU上运行（已在A100硅片验证），端到端评估依赖NVAS（NVIDIA内部simulator，外部不可获取）。

  评估原理：
  1. Queue性能评估：创建54个queue（对应A100的54对SM），每个queue连接一对producer/consumer CTA（共108 CTA对应108 SM）。Producer CTA向queue写入指定大小的payload，consumer CTA从queue读取。通过CUDA event timer测量传输总时间，计算aggregate bandwidth。对照组：将queue的atomic同步操作禁用，仅测试raw L2 bandwidth，差异即为同步overhead。
  2. 端到端性能评估：PyTorch Dynamo compiler → 生成包含spatial pipeline的可执行文件 → NVAS simulator加载执行 → 产出cycle-accurate的performance counters和timeline → 报告runtime speedup和DRAM traffic reduction。
  3. 利用率分析：NVAS输出每个cycle的SM utilization和DRAM utilization → 与NSIGHT Compute在BSP/TensorRT下的实测数据对比。

  全过程（以Kitsune queue执行一个GEMM→Elementwise→GEMM的spatial pipeline为例）：
  ```
  GPU启动cudaPipeline，包含3个kernel（各带TENSOR/SIMT/TENSOR type标注）
    → Modified Grid Scheduler分配CTA:
      SM_0: Linear_1_CTA_0 (Tensor) + ReLU_CTA_0 (SIMT) ← 双arbiter确保co-location
      SM_1: Linear_1_CTA_1 (Tensor) + ReLU_CTA_1 (SIMT)
      ...
      SM_N: Linear_2_CTA_* (Tensor) + (may overlap with other stage CTAs)

  Queue操作过程（以Linear_1 → queue_0 → ReLU为例）：
    Producer CTA (Linear_1):
      for each output tile:
        wr_acquire(queue_0, tile_id):  // 原子操作获取write entry
          while true:
            seq = atomicAdd(queue.seq, 0)  // 读取当前sequence number
            if seq == tile_id: break        // entry可用则跳出spin
            // 否则spin wait
        write tile data to queue.entries[wr_idx].data  // 写入tile数据（64-256KB）
        wr_release(queue_0):              // 原子操作释放entry
          atomicAdd(queue.seq, 1)         // 递增sequence number
          __syncthreads()                 // CTA级barrier确保所有线程完成写入

    Consumer CTA (ReLU):
      for each input tile:
        rd_acquire(queue_0, tile_id):  // 原子操作获取read entry
          while true:
            seq = atomicAdd(queue.seq, 0)
            if seq == tile_id + 1: break  // producer已完成此tile
        read tile data from queue.entries[rd_idx].data
        // 执行Elementwise ReLU on tile
        rd_release(queue_0):              // 释放entry供producer重用
          atomicAdd(queue.consumed, 1)
          __syncthreads()

    ReLU CTA完成后通过queue_1写入结果给Linear_2 consumer

  性能输出：
    → Queue bandwidth: 37 GB/s/queue @ 128-256KB payload, 2 TB/s aggregate
    → 同步overhead: 12× @ 1KB, <63% @ ≥64KB
    → SM utilization: Kitsune仅15% runtime在"both low utilization" vs BSP的26% (inference)
    → DRAM traffic: Kitsune减少41-98% (inference), 16-42% (training)
    → 端到端加速比: 1.3×-2.3× (inference), 1.1×-2.4× (training)
  ```

## SLA: Beyond Sparsity in Diffusion Transformers via Fine-Tunable Sparse-Linear Attention

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现是SLA的自定义GPU kernel，将稀疏注意力和线性注意力融合到单个CUDA kernel中执行前向和反向pass。核心kernel设计包括：(1) Fused Sparse+Linear Attention Forward Kernel——将critical块（FlashAttention-style O(N²)）、marginal块（O(N)线性注意力，预计算h_j/z_j后仅需矩阵加法）和negligible块（跳过）三种不同计算复杂度的操作融合在同一kernel内执行；(2) Fused Backward Kernel——同时反传稀疏注意力梯度（遵循FlashAttention backward公式）和线性注意力梯度（dH_i/dZ_i预计算后矩阵加法聚合），融合在同一kernel内避免额外launch和中间数据物化；(3) 额外效率优化——Lookup table（sparsity>90%时预处理非零mask位置减少内存流量）、Pre-aggregation（预计算全局行/列和再用减法替代90%加法）、Method of Four Russians（将marginal块分组预计算2^g子集和，用查表替代在线求和）。

  实验比较的baseline kernels：FlashAttention2（完整O(N²) attention）、VSA（89% sparsity, trainable sparse attention kernel）、VMoBa（85% sparsity, Mixture-of-block attention kernel）。评估指标：FLOPS = O(full attention)/t（kernel效率），end-to-end video generation latency（秒）。消融实验包括：Linear Only、Sparse Only、L+S（无Proj的直接相加）、不同激活函数φ（softmax/elu+1/hedgehog）、不同k_h参数（5%/10%/20%）。

- 后端平台是什么，配置是什么。
  NVIDIA RTX 5090 GPU。Attention kernel对比FlashAttention2（RTX 5090上最快可用版本）。软件：PyTorch + 自定义CUDA kernel。使用Block size b_q=b_{kv}=64。激活函数φ默认使用softmax。

- 评估性能的软件/脚本是什么。修改了什么。
  自定义CUDA kernel实现SLA的所有计算逻辑。修改：(1) 编写了SLA forward kernel——融合sparse FlashAttention critical块计算 + linear attention marginal块计算（预计算h_j = φ(K_j)^T V_j和z_j后用矩阵加法聚合）+ negligible块skip；(2) 编写了SLA backward kernel——融合sparse attention backward（dO^s → dQ/dK/dV via FlashAttention公式）+ linear attention backward（dO^l → dQ^φ/dK^φ/dV via chain rule），dH_i和dZ_i预计算后梯度聚合仅需加法；(3) 实现了额外效率优化：Lookup table存储非零mask位置、Pre-aggregation用减法替代加法、Method of Four Russians分组预计算。使用PyTorch autograd integration将自定义kernel集成到Wan2.1和LightningDiT模型中。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  开源链接：https://github.com/thu-ml/SLA（论文声明代码即将公开）。

  评估原理：
  1. Kernel级速度：测量SLA forward/backward kernel vs FlashAttention2 forward/backward的wall-clock时间，计算FLOPS = O(full attention) / t。O(full attention)是理论完整注意力FLOPs，t是实测延迟。
  2. End-to-end延迟：在Wan2.1-1.3B视频生成流程中替换所有注意力层为SLA，测量完整生成（所有denoising步骤）的wall-clock时间，包括attention和非attention（MLP/RMSNorm/Conv等）时间。

  Kernel输入到性能输出全过程（以Wan2.1-1.3B单层attention forward，N=30K tokens, d=head_dim, b_q=b_{kv}=64, k_h=5%, k_l=10%）：

  ```
  Host: 加载Wan2.1模型 + 替换注意力层为SLA attention
  Host: 对于每个denoising step t ∈ {T, T-1, ..., 1}:

  ┌─ SLA Forward Kernel (单次CUDA launch) ──────────────────────────┐
  │ Input: Q, K, V ∈ R^{N×d} (bfloat16)                              │
  │                                                                   │
  │ Step 1: Compressed mask prediction (GPU, inline):                │
  │   Q_pool = mean_pool(Q → blocks of 64)  // R^{469 × d}          │
  │   K_pool = mean_pool(K → blocks of 64)                           │
  │   P_c = Softmax(Q_pool @ K_pool^T / sqrt(d))  // R^{469 × 469} │
  │   for each row i of P_c:                                         │
  │     M_c[i, :] = classify(P_c[i,:], k_h=5%, k_l=10%)             │
  │     // 5% = 1 (critical), next 85% = 0 (marginal), 10% = -1     │
  │                                                                   │
  │ Step 2: Precompute linear attention components (line 4):         │
  │   for j in 0..T_n-1:                                            │
  │     K_phi_j = softmax(K_j)  // activation φ=softmax             │
  │     h_j = K_phi_j^T @ V_j    // R^{d × d}                      │
  │     z_j = rowsum(K_phi_j^T)   // R^{d × 1}                      │
  │                                                                   │
  │ Step 3: Main computation loop (lines 7-17):                      │
  │   for i in 0..T_m-1:  // each Q block                           │
  │     O_i_s, O_i_l: accumulators init to zero                      │
  │     for j in 0..T_n-1:  // each K,V block                       │
  │       if M_c[i,j] == 1:  // CRITICAL (~5% of blocks)            │
  │         // Full FlashAttention on this block pair               │
  │         S_ij = Q_i @ K_j^T / sqrt(d)    // b_q × b_kv GEMM     │
  │         OnlineSoftmax update:                                    │
  │           m_curr = max(m_prev, rowmax(S_ij))                    │
  │           P_ij = exp(S_ij - m_curr)                              │
  │           l_curr = exp(m_prev-m_curr) * l_prev + rowsum(P_ij)   │
  │           O_i_s = exp(m_prev-m_curr) * O_i_s + P_ij @ V_j       │
  │       elif M_c[i,j] == 0:  // MARGINAL (~85% of blocks)         │
  │         // Linear attention: single matrix addition             │
  │         H_i += h_j    // d × d addition                         │
  │         Z_i += z_j    // d × 1 addition                         │
  │       // else: NEGLIGIBLE → skip entirely                        │
  │     O_i_s = O_i_s / l_curr  // normalize sparse output          │
  │     O_i_l = (softmax(Q_i) @ H_i) / (softmax(Q_i) @ Z_i)         │
  │                                                                   │
  │ Step 4: Fusion output (Eq.6):                                    │
  │   O = O_s + Proj(O_l)  // learnable linear projection            │
  │   return O                                                        │
  └───────────────────────────────────────────────────────────────────┘

  ┌─ SLA Backward Kernel (单次CUDA launch) ──────────────────────────┐
  │ Input: dO (gradient of loss w.r.t. output), Q,K,V, M_c, L_i,    │
  │        H_i, Z_i, O_s, O_l from forward pass                      │
  │                                                                   │
  │ Step 1: Precompute linear attention gradients per Q block:       │
  │   for i in 0..T_m-1:                                            │
  │     D_i_s = rowsum(dO_i_s ⊙ O_i_s)  // for softmax backward     │
  │     D_i_l = rowsum(dO_i_l ⊙ O_i_l)                              │
  │     dH_i = (Q_phi_i / (Q_phi_i @ Z_i))^T @ dO_i_l               │
  │     dZ_i = -(Q_phi_i / (Q_phi_i @ Z_i))^T @ D_i_l               │
  │     dQ_phi_i = (dO_i_l @ H_i^T - D_i_l @ Z_i^T) / (Q_phi_i @ Z_i)│
  │                                                                   │
  │ Step 2: Aggregate gradients per K,V block:                       │
  │   for j in 0..T_n-1:                                            │
  │     dH_agg = 0; dZ_agg = 0                                       │
  │     for i in 0..T_m-1:                                          │
  │       if M_c[i,j] == 1:  // CRITICAL: FlashAttention backward   │
  │         S_ij = Q_i @ K_j^T / sqrt(d)                            │
  │         P_ij = exp(S_ij - L_i)                                   │
  │         dV_j += P_ij^T @ dO_i_s                                 │
  │         dP_ij = dO_i_j_s @ V_j^T                                │
  │         dS_ij = P_ij ⊙ (dP_ij - D_i_s)                          │
  │         dQ_i += dS_ij @ K_j                                     │
  │         dK_j += dS_ij^T @ Q_i                                   │
  │       elif M_c[i,j] == 0:  // MARGINAL: aggregate precomputed   │
  │         dH_agg += dH_i    // matrix addition                    │
  │         dZ_agg += dZ_i    // vector addition                    │
  │     dK_phi_j = V_j @ dH_agg^T + dZ_agg^T                        │
  │     dV_j += K_phi_j @ dH_agg                                     │
  │                                                                   │
  │ return dQ, dK, dV, dQ_phi, dK_phi                                │
  └───────────────────────────────────────────────────────────────────┘

  性能输出（RTX 5090, Wan2.1-1.3B, 30K tokens）：
    - SLA Forward Kernel: 13.7× speedup vs FlashAttention2 forward
    - SLA Forward Kernel: 1.93× faster than VSA @ 95% sparsity
    - SLA Forward Kernel: 3.36× faster than VMoBa @ 95% sparsity
    - SLA Backward Kernel: 6.8× speedup vs FlashAttention2 backward
    - Attention latency reduction: 97s → 11s (8.8×)
    - End-to-end video generation: 2.2× speedup
    - SLA @ 95% sparsity (1-5%): ~3× more efficient than Sparse Only @ 85% sparsity
    - Fine-tuning overhead: 2000 steps × batch 64 << 0.1% pretraining cost
  ```

  额外效率优化（Appendix A.3）：
  - Lookup table: sparsity>90%时，扫描整行/列读取mask值产生显著内存开销 → 预处理每行/列的非零位置存为lookup table → 计算时直接查表，减少内存流量
  - Pre-aggregation for linear attention: 虽然每次矩阵加法开销极小（Line 13），但M_c中>90%为0时重复加法累积 → 预计算全局∑h_j和∑z_j → 然后减去M_c[i,j]≠0的贡献 → 90%加法被10%减法替代
  - Method of Four Russians: 边际块数量既不很小也不很大（~50%）时 → 将h_j和z_j分组为g个连续块 → 预计算每组2^g个子集和 → 前向/反向时单次查表替代on-the-fly求和 → 理论计算量减少1/g
  ```

## Task-Based Tensor Computations on Modern GPUs

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现是Cypress编译器生成的warp-specialized CUDA kernel，在NVIDIA Hopper GPU上编排TMA（异步数据搬运）和Tensor Core（异步矩阵乘）之间的producer-consumer pipeline。编译器自动将用户的任务描述转换为warp-specialized代码：(1) DMA warp专门执行TMA异步copy（单线程调用TMA_load/TMA_store），通过completion barriers通知compute warpgroup数据就绪；(2) Compute warpgroup（128线程，4 warps）专门执行WGMMA指令驱动Tensor Core；(3) 软件pipeline（PIPE=3）使DMA warp预取PIPE步后的数据，隐藏global memory访问延迟；(4) Named barriers（prod/cons）管理DMA↔Compute之间的producer-consumer同步；(5) Backwards anti-dependency edges保证pipeline correctness（防止覆盖消费者尚未用完的buffer）；(6) 对于Flash Attention，编译器推断并插入TMA和Tensor Core之间的interleaved communication和synchronization。

  实验比较：(a) GEMM/Batched-GEMM vs cuBLAS（手写汇编/CUTLASS优化）和Triton；(b) Dual-GEMM（fused A·B₁+A·B₂）vs Triton；(c) GEMM+Reduction（fused GEMM + row-wise sum reduction）vs Triton；(d) Flash Attention 2 vs cuDNN/ThunderKittens/Triton；(e) Flash Attention 3 vs Flash Attention 3参考实现/cuDNN/ThunderKittens/Triton。

- 后端平台是什么，配置是什么。
  NVIDIA H100 80GB SXM5 GPU（Hopper架构，Tensor Cores支持wgmma 64×256×16 MMA指令、TMA支持异步burst copy和multicast、named barriers用于warp间同步、warpgroup概念——128线程协同启动Tensor Core操作）。CUDA 12.5.1（多数实验），Flash Attention实验部分系统用CUDA 12.3.1。Triton nightly 3.0.0.post20240716052845。

- 评估性能的软件/脚本是什么。修改了什么。
  使用Cypress compiler生成的CUDA C++ kernel（warp-specialized, 包含TMA loads, WGMMA instructions, shared memory barriers）直接benchmark。对比的baseline系统：
  - cuBLAS/cuDNN: NVIDIA vendor libraries
  - CUTLASS: 开源模板库，参考实现在CUTLASS Hopper GEMM main loop (sm90_mma_tma_gmma_rs_warpspecialized.hpp)
  - ThunderKittens: 最新Hopper kernel库
  - Triton: 公开示例程序，部分kernel需手动修改启用实验性TMA操作
  - Flash Attention 3: 参考实现[37]

  修改：Cypress compiler生成代码使用CuTe dispatch到PTX WGMMA指令，kernel组织为DMA warp (TMA) + compute warpgroup (Tensor Core)的warp-specialized结构。Flash Attention 3中，用户重写main loop为pipelined方式后，Cypress编译器自动推断所有interleaved通信和同步（原Flash Attention 3需手动标注位置）。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  论文未明确提供Cypress开源链接。Cypress为Stanford/NVIDIA合作研究原型。

  评估原理：对每个kernel和问题size，100次迭代warmup 5次后取平均throughput (TFLOPs/s)。GEMM-like计算使用相同随机分布矩阵元素normalize power throttling影响。Triton kernel直接使用或适配公开示例程序。

  全过程（以H100 Hopper GEMM kernel执行为例）：
  ```
  输入：C=m×n矩阵，A=m×k矩阵，B=k×n矩阵 (FP16)

  Kernel: warp-specialized GEMM, grid=(ceil(m/T_M), ceil(n/T_N)), block=(128+32 threads, 1 DMA warp + 4 compute warps)

  SMEM: sA[T_M, T_K, 3], sB[T_K, T_N, 3], sC[T_M, T_N]
  Barriers: prod[3], cons[3], copyout

  ┌─ DMA Warp (32 threads, only thread 128 used for TMA) ───────────┐
  │ for k in range(0, K/T_K):                                       │
  │   if k >= PIPE: wait(cons[k % PIPE])    // wait consumer done   │
  │   if tid == 128:                                                │
  │     TMA_load(prod[k%PIPE],                                      │
  │       tile(gA, (blk_x, k)) → sA[:, :, k%PIPE],                 │
  │       tile(gB, (k, blk_y)) → sB[:, :, k%PIPE])                 │
  │                                                                  │
  │ wait(copyout)                                                   │
  │ if tid == 128:                                                  │
  │   TMA_store(sC → tile(gC, blk_x, blk_y))                       │
  └──────────────────────────────────────────────────────────────────┘

  ┌─ Compute Warpgroup (128 threads, 4 warps) ──────────────────────┐
  │ for k in range(0, K/T_K):                                       │
  │   wait(prod[k % PIPE])    // wait TMA完成数据加载               │
  │   warpgroup_sync()        // 128线程对齐                         │
  │   wgmma(accum, sA[:,:,k], sB[:,:,k])  // 异步发起Tensor Core    │
  │   warpgroup_wait()        // 等待Tensor Core完成                │
  │   arrive(cons[k % PIPE])  // notify DMA warp buffer可用         │
  │                                                                  │
  │ copy(accum, sC)           // 寄存器accum→shared memory staging  │
  │ syncthreads()                                                   │
  │ arrive(copyout)           // notify DMA warp可写出              │
  └──────────────────────────────────────────────────────────────────┘

  输出：GEMM throughput on H100
    - M=N=K=8192: ~980 TFLOPs/s (0.97x cuBLAS)
    - vs Triton: 1.05-1.11x speedup
    - Dual-GEMM: ~970 TFLOPs/s (与GEMM接近, vs Triton 1.36-1.40x)
    - GEMM+Reduction: 2.02-2.18x vs Triton (Triton未overlap GEMM与reduction,
      且heuristic将reduction accumulator放在SMEM而非register file)
  ```

  关键kernel设计要点：
  - DMA warp不参与compute——释放其registers给compute warpgroup存储更大accumulator
  - Pipelining (PIPE=3): DMA warp跑PIPE步领先compute warps，TMA延迟被完全隐藏
  - Backwards dependencies: DMA warp必须先等consumer用完buffer（cons barrier）才能写入新数据
  - Flash Attention 3 pipelining: 用户显式重写loop body做pipelining后，
    Cypress编译器自动推断所有interleaved TMA→Tensor Core同步和通信位置
  - 持久kernel优化未实现（影响小sequence length下Flash Attention 3的performance gap）

## ThunderKittens: Simple, Fast, and Adorable Kernels

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现是ThunderKittens (TK)，一个C++ embedded AI kernel编程框架，通过三层GPU抽象简化高性能kernel开发：(1) warp级——16×16矩阵tile作为基本数据结构，提供PyTorch风格的操作（mma, exp, cumsum等），自动选择最小化bank conflict的shared memory布局（32/64/128字节swizzle）；(2) block级——LCSF (Load-Compute-Store-Finish) 异步编程模板，基于生产者-消费者范式协调load/store worker与compute worker的异步overlap执行，支持多级pipeline buffer隐藏HBM延迟；(3) grid级——persistent grid减少block launch/setup开销，block launch order调度提升L2 cache复用率。实验比较：(a) GEMM vs CuBLAS、CUTLASS；(b) Attention forward/backward（causal/non-causal, d=64/128）vs FlashAttention-3；(c) Linear attention（polynomial-based特征图和learned特征图）vs Flash Linear Attention (FLA, Triton)；(d) State space models long convolution (FFT-based) vs FlashFFTConv；(e) Mamba-2 vs Triton kernels from Dao & Gu 2024；(f) Rotary positional encoding、fused dropout-residual-layernorm vs popular Triton kernels。

- 后端平台是什么，配置是什么。
  NVIDIA H100 80GB SXM5 GPU（Hopper架构，132 SM，tensor cores支持wgmma指令，TMA异步数据搬运）。扩展测试：NVIDIA RTX 4090（consumer GPU），Apple M2 Pro（personal hardware，Metal API）。CUDA 12.6，Triton 3.00，PyTorch 2.4。

- 评估性能的软件/脚本是什么。修改了什么。
  使用NVIDIA Nsight Compute (NCU) 进行kernel profiling，分析tensor core利用率、issue slot利用率、HBM bandwidth/stalls、shared memory stalls。性能测量：10次warmup + 10次benchmark iteration取平均，通过cudaEvents计时。TK本身不修改现有软件，而是提供一个全新的C++ embedded框架替代CUTLASS/CuTe进行kernel开发。baseline GEMM通过CuBLASLt的auto-tuning获取最优性能，Triton kernel通过triton.autotune调优。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  开源地址：https://github.com/HazyResearch/ThunderKittens。TK以header-only C++ library形式发布（include/目录<1MB），用户编写kernel时include TK头文件，使用kittens命名空间。评估原理：NCU通过硬件性能计数器采集SM内部的tensor core pipelines利用率、issue slot active cycles、HBM读/写bytes及stall cycles、shared memory bank conflict计数。以attention kernel为例：(1) 用户用TK的tile类型(rt_bf, st_bf)声明register/shared memory tiles和global layout descriptors (gl<bf16, -1, -1, -1, D>)描述HBM tensor；(2) 在LCSF模板中编写load函数（TMA异步load K、V tiles到shared memory pipeline buffer）、compute函数（warpgroup::mm_ABt计算Q@K^T → softmax via sub_row/exp/row_sum/div_row → copy转为bf16 → warpgroup::mma_AB计算att@V）、store函数（TMA异步写回output）、finish函数；(3) TK自动选择shared memory swizzle布局消除bank conflict，自动生成TMA descriptor，管理barrier同步；(4) 编译为CUDA binary后在H100上执行，NCU monitor采集性能计数器。TK GEMM kernel仅40行device code即与CuBLAS竞争。整个TK attention kernel约217行LoC，对比FlashAttention-3的CUTLASS实现约2325行LoC。

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

## Iris: First-Class Multi-GPU Programming Experience in Triton

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现是Iris——一个纯Python+Triton实现的多GPU通信库，提供tile级device-side API(load/store/get/put/copy/atomic_add/atomic_cas等)，使开发者能在单个Triton kernel内无缝交织计算和通信。核心kernel调度实现包括：(1) 指针翻译机制——通过__translate函数将本地symmetric heap指针转换为远程GPU地址(偏移计算+heap base加法+类型cast)，实现跨GPU内存访问；(2) 融合kernel模式分类——Unfused Bulk-Synchronous（先后两个kernel，中间global barrier）、Unfused Producer-Consumer（两个kernel在不同CU异步stream上执行，通过CU分区和atomic锁同步）、Fused Sequential（单kernel内GEMM tile产出后立即iris.store到远程GPU，顺序依赖）、Fused Workgroup Specialization（单persistent kernel内按pid划分workgroup：前256个workgroup做GEMM计算并atomic_cas发信号，后48个workgroup等待信号执行iris.put通信）；(3) tile级同步——使用gpu-scoped atomic_cas(acquire/release)替代kernel级barrier，实现fine-grained overlap；(4) cache感知调度——cache_modifier(".wt"等)控制写策略，chiplet_swizzle映射workgroup到XCD分组优化LLC locality，GROUP_SIZE_M做L2 spatial swizzle。

  实验比较：(a) Microbenchmarks——load/store/atomic point-to-point操作带宽利用率(heatmap)，all-load/all-store多GPU同时操作带宽利用率；(b) GEMM+All-Scatter workload——Iris Unfused Bulk-Synchronous、Unfused Producer-Consumer、Fused Sequential、Fused Workgroup Specialization四种overlap模式 vs PyTorch torch.matmul + RCCL AllGather baseline，6种problem shape(M=8192固定，N×K变化)，2/4/8 GPU配置。

- 后端平台是什么，配置是什么。
  8×AMD Instinct MI300X GPU，全连接Infinity Fabric拓扑(7条Infinity Fabric Link/GPU)，NPS1/SPX memory和compute partition模式，ROCm 6.3.1。MI300X每GPU 304 Compute Units。

- 评估性能的软件/脚本是什么。修改了什么。
  自研Iris库(Python+Triton)，使用PyTorch Distributed初始化rank、HIP IPC(hipIpcGetMemHandle/hipIpcOpenMemHandle)建立symmetric heap。修改：(1) 新增device-side API——load/store(值语义，register↔remote memory)、get/put/copy(指针语义，buffer↔buffer)、atomic_*系列操作(算术/位运算/交换/比较)，所有操作遵循acquire/release memory ordering + block/gpu/sys scope；(2) 实现多种融合GEMM+All-Scatter kernel变体(对应Listings 3-5)；(3) GEMM loop复用统一gemm_loop模板。Baseline使用torch.matmul(PyTorch GEMM) + RCCL AllGather。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  开源地址：https://github.com/ROCm/iris

  评估原理：
  1. Microbenchmarks：各GPU同时执行load/store/atomic操作测量remote memory access带宽，归一化到理论可达带宽，绘制heatmap。All-load/store benchmark：所有GPU同时跨所有链路执行load/store，不同buffer size测量带宽利用率。
  2. 应用级评估：对每种problem shape和GPU数，分别用Iris的四种overlap pattern和PyTorch+RCCL baseline执行GEMM+All-Scatter，测量wall-clock时间计算speedup。
  3. Deep-dive分析：分解GEMM(深色区域)和Communication(浅色区域)时间占比，展示overlap效果。

  Kernel输入到性能输出全过程（以Fused Workgroup Specialization GEMM+All-Scatter，8 GPU，M=8192，N=3584，K=14336为例）：

  ```
  Host: iris.init()初始化——分配symmetric heap，IPC handle exchange
  Host: 输入矩阵A[M,K]分片在各GPU本地，B[K,N/8]各GPU持有N维1/8
  Host: launch wg_specialized_gemm_all_scatter[(304,)] (304 = MI300X CU数)

  GPU Mega-Kernel内部:
  ┌──────────────────────────────────────────────────────────────┐
  │ Workgroup分配: 256 GEMM workers (pid 0-255), 48 COMM workers (pid 256-303) │
  │                                                              │
  │ GEMM Worker (pid 0-255, persistent for-loop):               │
  │   for tile_id in range(pid, total_tiles, 256):               │
  │     ① gemm_loop(A, B, C):                                    │
  │       for k in range(0, K, BLOCK_SIZE_K):                    │
  │         a = tl.load(A_tile)   // global→register             │
  │         b = tl.load(B_tile)                                   │
  │         acc += tl.dot(a, b)   // Tensor Core MMA              │
  │       → 产出C_tile [BLOCK_M, BLOCK_N] in registers           │
  │     ② tl.store(C_local + offset, c, cache_modifier=".wt")    │
  │       → 写本地GPU memory（write-through for coherence）      │
  │     ③ tl.atomic_cas(locks + tile_id, 0, 1, sem="release",    │
  │                      scope="gpu")                            │
  │       → 发信号: tile_id已就绪                                 │
  │                                                              │
  │ COMM Worker (pid 256-303, persistent for-loop):              │
  │   for tile_id in range(pid-256, total_tiles, 48):            │
  │     ① spin-lock:                                            │
  │       while atomic_cas(locks+tile_id, 1, 0, sem="acquire")==0│
  │         pass  // 等待GEMM worker完成                          │
  │     ② for remote_rank in range(8):                           │
  │         if remote_rank != cur_rank:                          │
  │           iris.put(C_local+offset, C_local+offset,           │
  │                    cur_rank, remote_rank, heap_bases)         │
  │           → translate: offset_in_heap = ptr - local_heap_base│
  │           → remote_ptr = remote_heap_base + offset_in_heap   │
  │           → tl.store(remote_ptr, data) // 跨GPU写             │
  └──────────────────────────────────────────────────────────────┘

  输出性能：
    - Unfused Bulk-Synchronous (Iris baseline): 与PyTorch+RCCL性能相当，验证无抽象开销
    - Unfused Producer-Consumer: up to 2.5× speedup (8192×3584×14336, 8 GPU)
      → 小N(被8分割后)+大K使通信完全隐藏在GEMM后面
    - Fused Sequential: up to 1.79× speedup (8192×4608×36864, 4 GPU), 1.5× (8 GPU)
    - Fused Workgroup Specialization: 通信近乎100%隐藏于GEMM
    - 跨所有配置平均1.21× speedup vs PyTorch+RCCL
    - Microbenchmarks: near-optimal bandwidth utilization

  关键kernel调度优势：
  - tile级同步替代kernel barrier：消除bulk-synchronous模式的"bubble"
  - GEMM tile产出后立即scatter：无intermediate global memory write→read往返
  - Workgroup specialization使GEMM和通信使用不同CU子集并发执行
  - 值语义(iris.store)直接从register scatter到remote GPU，无需本地buffer中转

## TileLang: A Composable Tiled Programming Model for AI Systems

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现是TileLang的调度空间（Scheduling Space）系统，由四种核心调度机制组成，均通过解耦dataflow与scheduling实现：(1) **Thread Binding**——通过Layout Inference Pass按三层优先级（GEMM > Element-wise > Copy）自动推断buffer layout和thread binding。LayoutMap记录所有buffer的layout信息，从高到低优先级逐层推断，直至无更多buffer可推断。Fragment Layout支持repeat/repeat_on_thread/replicate四种组合操作构造复杂block级layout。对于GEMM后的element-wise操作（如bias add），自动推断bias buffer的replication策略以匹配GEMM output的thread分布。(2) **Memory Layout Composition**——基于IterVar的composable Layout抽象（f: K^n → K^m），支持swizzle layout（避免shared memory bank conflict）、padding layout（优化access pattern）、Fragment Layout（f: K^n → K²，输出thread index和register index）。Layout通过forward_index表达式和arithmetic analyzer推断buffer shape和访问边界。(3) **Pipeline**——T.Pipelined自动推导pipeline schedule：分析loop body各语句的buffer使用，确定Copy和GEMM的依赖关系，生成interleaved schedule（Copy→GEMM与其他copy重叠）。Ampere: 自动插入cp.async/cp.async.commit_group/cp.async.wait_group。Hopper: 自动TMA + mbarrier + warp specialization（通过Live Variable Analysis确定同步点，生产者/消费者根据threadIdx分入不同执行路径）。AMD CDNA: 利用s_waitcnt lgkmcnt和buffer_load_dword lds指令。(4) **Intrinsic Tensorization**——两种硬件指令利用方式：Tile Library-based (CUTLASS cute / AMD CK; 默认方案，自动选择最优指令) 和 Direct PTX/C++ source injection (T.ptx + T.import_source + T.call_extern)。

  实验比较：FlashAttention (H100, vs FA3 1.36×/Triton 1.41×/PyTorch 1.70×)，Linear Attention (H100, vs Triton 平均1.77×和2.10×)，MLA (H100 1075.9× vs Torch, 98% of FlashMLA; MI300X 129.2× vs Torch, 95% of AITER)，GEMM (4 GPU type × vendor libs/Triton)，Dequantized Matmul (A100, INT2 7.65× vs cuBLAS, INT4 1.04× vs Marlin, NF4 1.62× vs BitsandBytes)。

- 后端平台是什么，配置是什么。
  NVIDIA H100 (80 GB, CUDA 12.4)，NVIDIA A100 (80 GB, CUDA 12.4)，AMD Instinct MI300X (192 GB, ROCm 6.1.0)，RTX 4090。所有平台Ubuntu 20.04。

- 评估性能的软件/脚本是什么。修改了什么。
  TileLang kernel通过@tilelang.jit decorator编译，tilelang.compile(program, target="cuda"/"hip")生成可执行kernel。FlashAttention benchmark使用Table 3的5种shape配置（batch=1, nheads=32, seq_len=512/1024/4096, head_dim=128, causal/non-causal）；Linear Attention使用Mamba-2的chunk-scan和chunk-state函数，Table 4的12种shape；MLA在H100和MI300X上对比；GEMM使用Table 2的16种矩阵shape (M 1-8192, N 9216-57344, K 9216-57344)；Dequantized Matmul基于BitBLAS-TileLang后端，覆盖W_INT2A_INT8 / W_INT4A_FP16 / W_NF4A_FP16 format。Baselines: FlashAttention-3(手写CUDA), Triton(开源框架), cuBLAS(NVIDIA), rocBLAS(AMD), PyTorch(手写FA2 kernel), BitsandBytes(NF4 kernel), Marlin(INT4 kernel), FlashInfer, FlashMLA(手写), AITER(手写AMD kernel)。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  开源地址：https://github.com/tile-ai/tilelang

  评估原理：TileLang将kernel执行建模为dataflow-centric tile operators + scheduling annotations的组合。关键调度优化如何转化为性能增益：

  1) **Layout Swizzling去除Bank Conflict**：T.gemm默认对A_shared和B_shared应用MakeSwizzleLayout。swizzle通过异或位操作重排shared memory地址，使得warp内不同thread的shared memory访问映射到不同bank。无swizzle时bank conflict导致shared memory bandwidth下降，GEMM性能损失可达20-30%。Layout swizzling确保所有测试设备上zero bank conflict。

  2) **Pipeline Overlap**：T.Pipelined(K // block_K, num_stages=2)自动推导Copy-GEMM overlap。对每个k-iteration i，编译器分析(i+1)轮Copy与(i)轮GEMM无依赖，生成interleaved schedule。在Ampere，cp.async用于异步global→shared copy，与Tensor Core GEMM计算overlap。在Hopper，TMA hardware unit接管copy，warp specialization将线程分为producer(TMA copy)和consumer(wgmma.mma_async)，通过mbarrier同步。与Triton的num_stages参数不同，TileLang允许用户通过自定义pipeline order实现更复杂的overlap pattern。

  3) **Thread Binding + Vectorization**：T.copy在Layout Inference Pass后自动parallelize和vectorize（图8）。以(8,32)的2D copy为例：Pass推断loop axes → 自动分配thread binding（如threadIdx.x映射到i轴, vectorize 4 elements along j轴） → 应用SwizzleLayout。生成代码使用128-bit vectorized load/store (uint4/float4)，最大化memory bandwidth利用率。

  4) **Warp Specialization (Hopper独占)**：TileLang自动分析buffer usage确定各语句的producer/consumer角色 → 按threadIdx分离执行路径 → Live Variable Analysis确定同步点 → 插入mbarrier。这使得在FlashAttention实现中达到与FlashAttention-3手写kernel相当的pipeline复杂度。

  Kernel输入到性能输出全过程（以H100 FlashAttention为例）：

  输入: Q[batch, heads, dim] f16, KV[batch, seq_kv, kv_heads, dim] f16
  1. T.Kernel(batch, heads // min(B_H, kv_group), threads=256) → grid_size, block_size
  2. T.alloc_shared + T.alloc_fragment: 分配Q_shared, KV_shared, S_shared共享内存; acc_s, acc_o, scores_max等register files
  3. T.copy(Q → Q_shared): Layout Inference → thread binding + vectorized load (128-bit)
  4. T.Pipelined loop over KV tiles:
     a) Producer threads: TMA async copy KV[bx, k*BN:(k+1)*BN, ...] → KV_shared (global→shared via TMA hardware)
     b) mbarrier arrive (producer signals data ready)
     c) Consumer threads: mbarrier wait → wgmma.mma_async(Q_shared, KV_shared, acc_s) (async Tensor Core matmul)
     d) T.reduce_max(acc_s, scores_max, dim=1) → online softmax rescaling
     e) T.gemm(S_shared, KV_shared, acc_o) → output accumulation
     f) Producer continues TMA copy for next KV tile (overlapped with consumer compute)
  5. T.copy(acc_o → Output): register → shared → global store with thread binding + vectorization
  6. 评估: tilelang.compile返回的kernel函数，通过CUDA events测量wall-clock latency。与FlashAttention-3比 speedup = latency_FA3 / latency_TileLang

