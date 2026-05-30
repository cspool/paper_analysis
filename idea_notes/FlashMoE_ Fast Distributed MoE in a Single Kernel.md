## FlashMoE: Fast Distributed MoE in a Single Kernel

- baseline方法是什么？
  **标准分布式MoE执行（以Megatron-LM + DeepEP / DeepSpeedMoE / FasterMoE / COMET为代表）**：MoE layer分为多个独立kernel——Gate kernel（计算routing）+ Dispatch AlltoAll collective（token重排）+ Expert FFN kernels（逐expert GEMM，2层）+ Combine AlltoAll collective（token恢复原始顺序）。所有kernel由CPU逐个launch，AlltoAll为NCCL同步collective。关键性能瓶颈：
  - (a) **同步AlltoAll collective**：所有GPU必须同时参与AlltoAll，straggler GPU卡住全部GPU的进展（P95 delay 1.32× on supercomputer, 11.4× on VM），通信占MoE layer总运行时间最高68%
  - (b) **大量kernel launch overhead**：单层MoE forward pass需33-550个GPU kernel（Table 1），每次launch产生CPU-GPU同步、CUDA API overhead、kernel start time non-determinism，造成GPU idle gap
  - (c) **CPU-managed调度缺乏work-conserving**：CPU串行launch kernel，无法根据GPU内部readiness动态分配计算任务
  - (d) **Token padding浪费通信带宽**：不对称routing导致GPU收到的token数不足expert capacity时，DeepSpeed等框架在通信buffer中补零传输，浪费带宽和算力

  全栈执行例子（Megatron-LM + DeepEP, 8×H100, S=16K tokens, E=128 experts, top-2 routing, FP16）：
  - **模型推理算法层**：Standard top-2 gating。Gate function G(x) = softmax(x·W_g) → select top-2 experts per token → 每个token需routing到2个expert。routing table T_φ[e][c] = (token_idx, combine_weight)。Expert capacity C = 256（假定uniform distribution），超过capacity的token丢弃（capacity factor=1.0）。
  - **系统框架层**：Megatron-LM distributed MoE layer实现。使用PyTorch distributed + NCCL backend。MoE layer forward: Gate → token permutation (AlltoAll dispatch) → expert FFN (各GPU上的local experts的serial或batched GEMM via cuBLAS/Transformer Engine) → token unpermutation (AlltoAll combine)。DeepEP提供NCCL + NVSHMEM混合的优化AlltoAll实现。
  - **编译框架层**：论文未明确说明。Megatron-LM使用nvcc编译手写CUDA kernel，Transformer Engine（TE）使用fp8精度GEMM。
  - **kernel调度层（关键瓶颈）**：
    - Gate kernel (1 launch): CUDA kernel计算routing
    - Dispatch AlltoAll (NCCL collective): 同步barrier，所有GPU等待最慢GPU的token buffer准备好 → GPU在等待期间SM idle。AlltoAll底层为多次P2P send/recv或NVLink copy + NVSwitch routing
    - Expert FFN (expert_cnt × 2 launches): 每个expert的GEMM0和GEMM1各一次cuBLAS kernel launch（或TE fused kernel），若16 local experts=32次launch。每次launch：CPU enqueue → CUDA API overhead → kernel start → global memory load weights → GEMM → store
    - Combine AlltoAll (NCCL collective): 再次同步barrier
    - 总kernel launch数：432（Megatron-LM+DeepEP，Table 1）。CUDA API trace（Figure 4b）显示大量small CUDA API calls，kernel间显著stall gap。SM utilization仅13.55%（DeepEP+Megatron-LM）、9.67%（FasterMoE）。
    - Overlap方案（COMET、FasterMoE）：将部分expert computation与AlltoAll pipeline重叠，但受限于collective的同步barrier——先dispatch完成才能开始全部FFN，先全部FFN完成才能开始combine，overlap有限
  - **硬件架构层**：8×H100 GPU NVSwitch全互联。NVLink 900 GB/s unidirectional per GPU。NCCL AlltoAll通过NVSwitch可实现full bisection bandwidth，但同步barrier导致collective latency由最慢参与者决定。Tensor core算力充足但大量时间idle等data。

  Baseline缺陷：
  - (a) **同步通信straggler问题**：AlltoAll collective barrier要求全部GPU参与，最慢GPU决定整体延迟。straggler cause包括kernel scheduling jitter、OS interference、NVLink congestion——尤其expert分布不均衡时更严重
  - (b) **Kernel launch overhead**：432个kernel launch per MoE layer（vs FlashMoE 1个），产生~90% GPU idle time（Figure 4a）。每次launch对应CUDA API call (~10μs)、CPU-GPU synchronization、kernel cold-start L1/L2 cache miss、global memory round-trip for intermediate data
  - (c) **Non-work-conserving execution**：CPU按静态顺序launch kernel（Gate→AlltoAll→expert0_GEMM0→...→AlltoAll），无法根据dynamic availability（某GPU已完成expert计算但另一GPU的FFN仍在执行）进行动态调度。GPU SM在等待远端token期间完全空闲
  - (d) **通信payload冗余**：expert capacity padding——若某些expert收到少于C个token，剩余slot补零，这些零值通过NVLink传输再被GEMM计算（结果为零），浪费通信带宽和计算资源

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **FlashMoE：单个持久化GPU kernel融合全部分布式MoE计算与通信**。核心创新是将MoE operator从CPU协调的多kernel架构彻底重构为GPU-resident的单kernel actor-based并发系统。

  核心设计一（解决缺陷b,c）：**Actor-based单kernel架构 + Warp Specialization**。将N个thread block特化为N-1个Processor + 1个OS block。OS block内：1个Scheduler warp（work-conserving多线程调度器，Algorithm 3）+ 3个Subscriber warps（解码远端tile packet，Algorithm 4）。Processor持续loop等待Scheduler signal分配task（Algorithm 2）。整个MoE layer生命周期内仅1次kernel launch，消除所有kernel launch overhead（从432→1）。Scheduler根据task readiness动态分配——只要有Processor idle且task queue非空→立即schedule——实现work-conserving。

  核心设计二（解决缺陷a）：**Device-initiated one-sided (R)DMA替代同步AlltoAll collective**。使用NVSHMEM PGAS编程模型——GPU kernel内直接通过nvshmem_putmem发起跨GPU内存写——无需远端GPU CPU参与，无需collective barrier。关键insight：token dispatch从"pull model"（AlltoAll，所有GPU同步exchange）变为"push model"（Processor主动push tile到远端GPU memory，Subscriber被动接收）。远端GPU的Subscriber通过检测NVSHMEM signal flag异步消费数据。配合Symmetric Tensor Layout L（Theorem 3.1 write-write conflict-free），所有one-sided write无需锁或同步——每个(p_s, r, b=1, e, c) index唯一确定目标位置。

  核心设计三（解决缺陷d）：**In-place padding + Payload-efficient communication**。Token在dispatch前在本地symmetric tensor buffer L内padding到expert capacity对齐（divisible by tile height bM=128），网络仅传输实际token tile——无null payload在网络传输。相比DeepSpeed/DeepEP在通信buffer中补零传输的做法，节省通信带宽（极端不对称routing case下可节省数倍payload）。

  核心设计四（实现fine-grained overlap）：**Tile-level parallelism with (128, 64) tile**。将MoE计算和通信分解为tile粒度——每个tile映射为task descriptor t = (M, ⋆, φ)。GEMM0、GEMM1、combine、通信均为独立task，可并发执行：Processor A执行expert_i的GEMM0时，Processor B可同时dispatch expert_j token到远端，Subscriber可解码已到达tile的GEMM1 task，Scheduler并行调度。三个actor通过shared memory（Subscriber↔Scheduler）和global memory（Scheduler↔Processor，inter-GPU signal flag）异步通信，形成reactive、非阻塞的pipeline（Figure 6）。

  全栈执行例子（FlashMoE, 8×H100, same config S=16K, E=128, FP32）：
  - **模型推理算法层**：Same top-2 gating, standard FFN, expert-combine（Equation 2,3）。算法本身不变——创新在执行方式。FusedGate in-kernel（Algorithm 1 line 2）——Gate计算与routing table构建在一个函数内完成，无需写回HBM。
  - **系统框架层**：FlashMoE Python/C++ library（6820行代码）。`flashmoe::forward(A, X, O)` 一次调用完成全部MoE layer。无PyTorch NCCL backend依赖——通信全部通过NVSHMEM kernel内完成。框架不参与dispatch/combine调度——CPU仅做一次CUDA kernel launch。
  - **编译框架层**：CMake + nvcc编译。CUTLASS作为submodule提供device-side BLAS。NVSHMEM提供device-side通信API。无JIT compilation——tile size (128,64) compile-time固定。Binary size 29 MB, compilation time 53s。
  - **kernel调度层（关键创新，全覆盖）**：
    - FusedGate（Algorithm 1 line 2）: 所有block并行计算Gate → T_φ, G_φ写入shared memory + global memory
    - 角色分化（Algorithm 1 lines 6-11）: N-1 blocks → Processor (processor::start()); 1 OS block → warp 0 Scheduler + warps 1-3 Subscriber
    - Dispatch: Processor按T_φ将(128,64) tile通过NVSHMEM put写入远端GPU的L → write flag通知远端Subscriber。In-place padding确保无null传输
    - Subscriber poll flags → atomic retrieve → memory fence → 解码tile为GEMM0 task → write tQ → doorbell Scheduler
    - Scheduler sweep doorbells → WarpInclusiveSum → 从ready queue取idle Processor → signal。Work-conserving: 任一Processor空闲即分配，任一tile就绪即调度
    - Processor: awaitTaskFromScheduler → warp broadcast task → switch(type): GEMM0(fused GEMM+epilogue via CUTLASS device-side) → notify completion → GEMM1(same, result可能NVSHMEM put到远端combine buffer) → Subscriber解combine signal → Scheduler调度combine → Processor: Hadamard product + accumulate to O
    - Kill: Scheduler counted taskBound → interrupt subscribers → interrupt processors → kernel return
    - 结果: 93.17% SM utilization (9× higher than FasterMoE), 6× latency speedup, 4× overlap efficiency vs baselines (at 8 GPUs, 128 experts, 16K tokens)
    - 注意: FlashMoE在FP32下取得这些结果，baseline在FP16——通信量（4B vs 2B per element）和计算量（FP32 vs FP16 GEMM）均为double
  - **硬件架构层**：H100 GPU + NVLink。NVSHMEM nvshmem_putmem通过NVLink RDMA直接写远端HBM——利用H100的NVLink interconnect和NVSwitch实现cross-GPU data path。Symmetric tensor layout L overprovision 4× memory（2 rounds × 2 staging buffers = 4× token buffer），H100 80GB完全可容纳（≤2% overhead for popular models）。CUTLASS device-side GEMM utilize Tensor Cores through MMA instructions called from within persistent kernel。

  关键设计选择与Baseline缺陷的对应：
  - **defect (a): 同步AlltoAll straggler** → 方案：Device-initiated push-model one-sided (R)DMA替代pull-model synchronous collective。每个GPU独立push token到目标GPU，无需等待该GPU也完成同一round的push——消除barrier和straggler effect。Theorem 3.1证明write-write conflict-free保证无需同步。Temporal buffering (2 staging buffers) 隔离dispatch和combine的并发访问。
  - **defect (b): 432次kernel launch** → 方案：单persistent kernel融合全部操作。Actor-based concurrency模型允许同一kernel内执行gate、FFN GEMM0、FFN GEMM1、combine、dispatch通信、combine通信、调度——所有逻辑在while(!interrupt) loop的CUDA thread内完成。Kernel launch overhead从~3.6ms（432次，估算）降至~1μs（1次launch）。
  - **defect (c): Non-work-conserving CPU调度** → 方案：In-kernel work-conserving Scheduler（Algorithm 3）。多线程（1 warp=32 threads）并行sweep doorbells→aggregate→schedule，保证只要Processor空闲且task就绪→立即分配。Scheduler持续atomic poll taskBound确保不漏调度。Processor-Subscriber-Scheduler三者通过shared memory/global memory异步通信，形成reactive event-driven pipeline。
  - **defect (d): Token padding浪费带宽** → 方案：In-place padding + payload-efficient通信。Padding在本地L buffer内完成，网络传输仅包含实际token tile——NVSHMEM put的size = actual_tokens × H × sizeof(float)，而非 padded_capacity × H × sizeof(float)。尤其expert分布高度skewed时收益最大。
