## DSV: Exploiting Dynamic Sparsity to Accelerate Large-Scale Video DiT Training

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  DSV实现三个核心kernel/运行时计算组件：
  
  (i) **Fused Critical KV Estimation Kernel（CUDA）**：将低秩矩阵乘法（Q_lr·K_lr^T）与top-K选择融合为单一CUDA kernel，避免物化完整的[S, S]大小的attention score矩阵。采用两阶段方案：第一阶段计算各query的top-K阈值，第二阶段基于阈值选择indices。由于矩阵乘法shape为"slim"型（低秩维度d_lr ≪ S），在CUDA cores上而非Tensor cores上执行更高效（memory-bound特性）。每个SM对多个完整行执行乘法和Bitonic Select进行在线top-K选择。大top-K场景下split为两阶段以避免shared memory瓶颈。
  
  (ii) **Sparse Attention Kernel with Query Grouping（Triton）**：基于Observation 5（相邻tokens的critical KV pairs高度重叠，2×2×2 3D cube内重叠率>92.4%），将相邻queries按3D voxel分组（如2×2×2或2×4×4），共享critical KV pair集合，减少estimation overhead和memory traffic。支持自适应grouping机制，根据输入video scene动态调整group size以保证overlap ratio >80%。
  
  (iii) **Hybrid Sparsity-Aware Context Parallelism**：建模和分析Head-wise CP（HCP）和Sequence-wise CP（SCP）在稀疏场景下的trade-off，将CP配置问题形式化为min-max优化问题（最小化任意GPU的T_comm + T_comp），使用Gurobi求解器周期性求解最优HCP/SCP group size组合。HCP组优先部署在节点内（利用高带宽NVLink执行All-to-All），SCP组用于跨节点通信（仅传输critical KV）。
  
  实验比较：对比Vanilla 3D Full Attention（FlashAttention-2）和Window-based Sparse Attention（WA），测量forward/backward kernel speedup、不同sparsity水平下的加速比、end-to-end训练吞吐。

- 后端平台是什么，配置是什么。
  最多64张NVIDIA H100 GPU。节点内：900 GB/s NVLink双向互联。节点间：InfiniBand with RoCE（200 Gbps per cross-node GPU pair）。内存：GPU HBM，大KV indices场景使用asynchronous CPU offloading（如50K tokens, 80% sparsity产生约2GB KV index tensor）。计算精度：BF16。

- 评估性能的软件/脚本是什么。修改了什么。
  自定义CUDA kernel（fused MM + top-K）和Triton kernel（sparse attention with query grouping）。修改PyTorch FSDP框架扩展tensor和context parallelism。仅需替换原模型的attention API即可集成。
  
  评估原理：kernel级——测量不同输入长度（10K-200K tokens）和sparsity水平（90%-98%）下sparse attention module的forward/backward耗时，与FlashAttention-2 baseline对比。系统级——在32 GPU（CP=4, DP=8）和64 GPU（CP=8, DP=8）配置下测量end-to-end训练吞吐。

- 开源情况。论文未提供开源代码仓库链接。kernel实现描述：CUDA kernel在CUDA cores上（非Tensor cores）执行矩阵乘法，通过Bitonic Select进行在线top-K；Triton kernel实现稀疏attention的query grouping优化。截至分析时未在公开平台找到代码发布。

  Kernel执行流程说明：
  ```
  输入: H ∈ R^{S×d}, W_Q^lr, W_K^lr ∈ R^{d×d_lr}, sparsity_level
  
  Step 1: Fused Critical KV Estimation (CUDA kernel)
  - 输入: Q_lr = H @ W_Q^lr, K_lr = H @ W_K^lr  (shape: [H, S, d_lr])
  - 每个SM负责多个query行的完整计算:
      for each query q in assigned_queries:
          scores_q = []  # 保持在寄存器中
          for each tile of K_lr:
              partial_score = q_tile @ K_lr_tile^T  # CUDA core MM
              scores_q = BitonicSelect(scores_q, partial_score, K)
          # scores_q最终保留top-K score的indices
  - 输出: crit_indices ∈ Z^{H×S×K}  (K = ceil((1-sparsity)×S))
  
  Step 2: Query Grouping (on CPU or lightweight GPU kernel)
  - 将3D latent space划分为voxel groups (e.g., 2×2×2)
  - 每个group内所有queries共享proxy query的critical KV indices
  - 输出: grouped_crit_indices (减少的K集)
  
  Step 3: Sparse Attention (Triton kernel)
  - 输入: Q, K, V (完整), grouped_crit_indices
  - 对每个query group:
      gathered_K = gather(K, crit_indices[group])
      gathered_V = gather(V, crit_indices[group])
      O[group] = softmax(Q[group] @ gathered_K^T / sqrt(d_k)) @ gathered_V
  - 输出: O ∈ R^{S×d}
  
  Step 4: Context Parallelism (HCP + SCP hybrid)
  - HCP组内: All-to-All重分布head子集 → 每GPU独立计算 → All-to-All恢复
  - SCP组内: Ring-style KV gathering（仅传输critical KV）
  - 配置由Gurobi求解器周期性优化求解
  ```

## Demystifying NVIDIA GPU Internals to Enable Reliable GPU Management

> **近似层次匹配说明**：本文的核心实验通过自研的nvdebug内核模块和gpu-microbench微基准测试套件，在kernel/OS级别研究NVIDIA GPU的硬件调度行为。实验直接测量compute kernel和copy操作在不同runlist、channel配置下的timeslicing、并行度、互斥和干扰模式，属于kernel调度层面的运行时计算行为研究。

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  论文实现了两个核心工具套件用于kernel调度行为研究：
  (i) **nvdebug**：Linux内核模块，通过MMIO直接访问GPU寄存器（绕过GPU驱动），暴露GPU调度状态的检查和修改接口（/proc/gpuX/）。关键能力：查看runlist内容、禁用/启用channel、查看device_info（engine-runlist拓扑）、查看PCE-LCE映射。
  (ii) **gpu-microbench**：微基准测试库，包含exec_logger（持续执行compute kernel并以微秒精度记录每次执行时间戳）和copy_monitor（类似但仅使用copy engine，不触发compute工作），用于精准测量各引擎的调度行为。

  实验比较的核心是GPU在不同硬件配置下的kernel调度行为模式：
  - **Channel级别（R1-R2）**：验证所有GPU操作必须经过channel（禁用channel后kernel/copy/device-mapped memory allocation无法完成）；channel数限制intra-task并行度——x86_64默认8 compute channel，9个stream时第9个stream产生false dependency；增加CUDA_DEVICE_MAX_CONNECTIONS可消除
  - **Runlist级别（R3-R5）**：验证channel必须在runlist中；单runlist上每约2ms compute timeslice互斥、约1ms copy timeslice互斥；多runlist支持compute/copy独立调度无干扰；单runlist（Jetson TX2）上compute和copy共享导致copy被compute干扰（中断间隔1024µs=compute timeslice而非copy的1049µs）
  - **Engine映射级别（R6-R8）**：发现所有GPU的Runlist 0同时绑定Compute和GRCE；每个engine仅绑一个runlist（PTOP寄存器约束）；GRCE可通过共享底层PCE干扰独立runlist上的LCE——RTX 6000 Ada上OpenGL texture upload使CUDA GPU→CPU copy减速约2×

  还使用已有工具cuda_scheduling_examiner（Otterness et al. [17]）进行部分辅助实验。

- 后端平台是什么，配置是什么。
  9款NVIDIA GPU（覆盖5代架构，2016-2022）：
  GTX 1060 3GB (Pascal, CC 6.1)、GTX 1080 Ti (Pascal, CC 6.1)、Jetson TX2 (Pascal embedded, CC 6.2)、Titan V (Volta, CC 7.0)、Jetson Xavier (Volta embedded, CC 7.2)、RTX 2080 Ti (Turing, CC 7.5)、A100 40GB (Ampere, CC 8.0)、Jetson Orin (Ampere embedded, CC 8.7)、RTX 6000 Ada (Ada Lovelace, CC 8.9)。OS：x86_64和aarch64 Linux。所有实验前禁用后台GPU任务（Jetson TX2额外通过nvdebug清空runlist残留条目）。

- 评估性能的软件/脚本是什么。修改了什么。
  - **自研**：nvdebug（内核模块，~1500行C代码，MMIO寄存器访问 + GPU页表解析）、gpu-microbench（exec_logger记录compute kernel每次迭代的开始/结束时间戳和SM编号，微秒精度；copy_monitor仅使用copy engine记录copy进度）、多个实验编排脚本（同时运行多个微基准测试实例，交叉验证调度行为）
  - **已有工具**：cuda_scheduling_examiner（Otterness et al. [17]，部分实验使用）
  - **修改/新增**：exec_logger和copy_monitor为全新开发（需多年专家级调优和bug修复）；nvdebug为全新开发（需解决GPU页表访问/解析/遍历、多代GPU寄存器地址兼容性等问题）；实验脚本组合使用以上工具进行cross-validation

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - nvdebug: http://rtsrv.cs.unc.edu/cgit/cgit.cgi/nvdebug.git/
  - gpu-microbench及实验脚本: https://www.cs.unc.edu/~jbakita/rtas24-ae/

  评估原理与流程（以验证R2——channel数限制并行度实验，Fig.5，GTX 1060 3GB为例）：

  1. **环境**：加载nvdebug内核模块 → cat /proc/gpu0/device_info确认GPU拓扑 → 确认默认compute channel=8（CUDA 12.2 on x86_64）→ 禁用后台GPU任务

  2. **实验**：创建9个CUDA stream（Stream 1-9），每个stream内顺序launch 4个kernel（K1, K2短kernel, K3长kernel, K4短kernel）。长kernel K3含大量thread blocks以延长dispatch时间。

  3. **数据采集（exec_logger）**：每个kernel的每个thread block在开始/结束时通过CUDA event记录时间戳 → 记录(kernel_name, thread_block_index, start_time, end_time, SM_id) → 微秒精度时间线

  4. **分析（Fig.5 top）**：Stream 1-8的头kernel几乎立即开始执行 → Stream 9的头kernel等到t≈0.7s（Stream 1的K3所有blocks dispatch完毕释放channel）→ 验证false dependency：Stream 9需等待任意channel释放

  5. **对照（Fig.5 bottom）**：CUDA_DEVICE_MAX_CONNECTIONS增加channel至≥9 → 所有9个stream头kernel同时开始执行，无false dependency → 确认R2

  6. **跨GPU验证**：9款GPU上重复实验，确认x86_64默认=8，Jetson默认=2-4

  验证R4（单runlist互斥timeslicing）流程（Fig.6, 7）：
  - exec_logger双实例 → 时间线显示严格互斥（Fig.6 inset），约2ms timeslice切换
  - copy_monitor双实例 → copy进度交替推进（Fig.7），约1ms timeslice切换

  验证R5（多/单runlist调度独立性）流程（Fig.8, 9）：
  - GTX 1080 Ti（多runlist）：exec_logger + 2×copy_monitor + copy-and-compute task → compute和copy时间线非同步、不相关（Fig.8）→ 多runlist独立调度
  - Jetson TX2（单runlist）：exec_logger + copy_monitor → copy出现1024µs周期性中断（=compute timeslice，非copy的1049µs）（Fig.9）→ 单runlist导致跨引擎干扰

  验证R8（GRCE→LCE PCE共享干扰）流程（Fig.10, 11）：
  - GTX 1080 Ti vs RTX 6000 Ada → 各运行CUDA GPU→CPU copy + 并发OpenGL texture upload → RTX 6000 Ada copy被减速约2×（因GRCE映射到同一LCE→共享PCE），GTX 1080 Ti几乎不减速
  - cat /proc/gpu0/lce_for_pce0和shared_lce_for_grce0确认映射关系

## Characterizing Concurrency Mechanisms for NVIDIA GPUs under Deep Learning Workloads

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  论文对NVIDIA Ampere GPU上的三种并发机制（priority streams、time-slicing、MPS）在DL训练+推理并发workload下的kernel级调度行为进行了完整的实验表征。实现是使用CUDA API直接控制并发策略，通过NVIDIA GPU profiling工具（NSight Systems、nvidia-smi/NVML API、global timer register）在microarchitectural层面测量thread block scheduler的行为（leftover policy、most-room policy）、SM资源分配（threads、registers、shared memory、L1/L2 cache）、warp scheduler调度策略、时间片长度、上下文切换开销等。实验比较的是三种机制在inference turnarround time、variance（predictability）和training execution time（proxy for utilization）三个指标上的表现，并提出了fine-grained block-level preemption的必要性。

- 后端平台是什么，配置是什么。
  NVIDIA GeForce RTX 3090 GPU（Ampere microarchitecture）：82 SMs，每SM限制1536 threads、16 thread blocks、64KB registers、1024KB shared memory；全局24GB GDDR6X DRAM、6144KB L2 cache、936 GB/s memory bandwidth。

- 评估性能的软件/脚本是什么。修改了什么。
  - **PyTorch examples**（github.com/pytorch/examples）：ResNet-50、ResNet-152、AlexNet、VGG-19、DenseNet-201，用于training（最大batch size，避免OOM）和inference（batch size=1）两类任务。为测试priority streams，做了少量修改使training和inference task从同一进程的不同CUDA stream启动。
  - **MLPerf Inference v1.0**（git commit 8b58587c93af）和**MLPerf Training v0.7**（git commit 96ef5cabfc）：ResNet-34 inference（batch size=1）、BERT inference（batch size=1）、RNNT training（batch size=1024）。MLPerf模型未做任何修改（保持benchmark完整性），因此未测试priority streams。
  - **CUDA工具链**：NSight Systems用于profiling kernel执行时间和memory transfer时间；nvidia-smi/NVML API用于utilization测量；GPU global timer register（通过PTX内联汇编读取）用于测量time slice间隔和上下文切换时间。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  论文未提供独立开源代码仓库。所有实验基于公开可获取的PyTorch examples和MLPerf benchmarks。评估原理和全过程如下：

  **评估原理**：单GPU上同时运行一个training task（best-effort，持续运行整个实验期间）和一个inference task（latency-sensitive，500或5000个请求），通过三种CUDA并发机制调度。测量inference请求的平均turnaround time、variance（predictability），以及training task的执行时间（proxy for utilization）。

  **kernel输入到性能输出全过程**：
  1. Host端PyTorch/TensorFlow模型被框架编译为CUDA kernel序列（如convolutional implicit SGEMM kernel、FFT kernel等），每个kernel有自己的grid size（thread block数量）、block size（每block threads数）、register和shared memory需求。
  2. 每个kernel通过CUDA runtime API被dispatch到对应CUDA stream（或MPS client queue），由GPU application-level scheduler决定哪个stream/process的kernel入队。
  3. Kernel到达GPU后，thread block scheduler（采用leftover policy + most-room placement policy）将thread blocks分配到82个SM上。分配受限于每SM的thread/register/shared memory/block数量上限。
  4. 每SM内的4个warp scheduler单元以greedy-then-oldest或loose round-robin策略从就绪warps中选择下一条指令发射（每两周期一条warp指令）。
  5. Priority streams机制：thread block scheduler始终优先从高优先级stream取blocks调度，但不抢占已执行的blocks。Training kernel的long-running blocks造成"compounded delay"——inference kernel到达后需等待已执行的training blocks完成。
  6. Time-slicing机制：application-level scheduler以约2ms固定时间片轮转，整个GPU交替分配给两个进程。时间片之间约145μs的切换开销（通过global timer register测量）。
  7. MPS机制：MPS server调度来自不同CUDA context的kernels，允许blocks在同一SM上co-locate。可设置per-client thread limit（实验中设为100%），但无优先级概念，采用FCFS + leftover policy。
  8. 性能输出：NSight Systems记录每个kernel的执行时间和memory transfer时间；nvidia-smi轮询GPU utilization；application层记录inference请求完成时间以计算turnaround time和variance。

## A Survey of Resource-efficient LLM and Multimodal Foundation Models

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  本文为综述论文，无原创kernel实验。§5.3.1 "Inference Accelerating" 系统梳理了LLM推理kernel优化技术：
  (i) **FlashAttention [76]**：IO-aware fused attention kernel，利用tiling和recomputation避免具体化完整N×N attention矩阵，将HBM读写从O(N²)降至O(N)，实现prefill阶段加速。FlashAttention-2 [75]通过改进work partitioning减少非MatMul FLOPs并增加并行度。
  (ii) **Flash-Decoding [78]**：针对decoding阶段batch size大、seqlen短的特点，在seqlen维度上额外并行化，设计专门CUDA kernel加速decode。
  (iii) **FlashDecoding++ [146]**：在Flash-Decoding基础上进一步优化softmax操作和flat GEMM，并增加AMD GPU支持。
  (iv) **DeepSpeed-Inference [21]**：针对小batch size场景（FM serving常见但FM training罕见）的GPU kernel优化。
  (v) **ByteTransformer [468]**、Google PaLM serving system [314]也提供了GPU/TPU的小batch优化kernel。
  (vi) 论文表4给出了多种attention变体的时间复杂度与空间复杂度对比（Transformer O(T²d)、Reformer O(T log T d)、Linear Transformers O(T d²)、RetNet O(T d)、RWKV O(d)等）。

- 后端平台是什么，配置是什么。
  被引述kernel的硬件平台：NVIDIA A100/H100 GPU、AMD GPU、Google TPU v4。综述未进行统一实验。

- 评估性能的软件/脚本是什么。修改了什么。
  论文使用flops-profiler（https://pypi.org/project/flops-profiler/）对GPT-2及Stable Diffusion 2.1进行FLOPs分析（§2.1.3、§2.3.3），而非性能benchmark。该工具为现有工具，综述未修改其实现。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  综述全部材料开源：https://github.com/UbiquitousLearning/Efficient_Foundation_Model_Survey。以下以FlashAttention为例说明kernel输入到性能输出的全过程：

  传统attention vs FlashAttention的kernel计算流程：
  ```
  // 传统Attention (Standard)
  // 瓶颈：N×N attention matrix具体化到HBM
  Load Q, K from HBM           // [N, d] each
  S = Q @ K^T                  // [N, N], 写入HBM
  P = softmax(S)               // [N, N], 读写HBM
  O = P @ V                    // [N, d], 读写HBM
  // HBM访问量: O(N²) >> N
    
  // FlashAttention (IO-Aware Fused Kernel)
  // 分块计算，避免具体化完整N×N矩阵
  for j in 0..T_c-1:           // K,V blocks loaded once per outer loop
      Load K_j, V_j from HBM to SRAM   // [B_c, d] each
      for i in 0..T_r-1:       // Q, O blocks
          Load Q_i, O_i, l_i, m_i from HBM to SRAM
          S_ij = Q_i @ K_j^T            // [B_r, B_c] on-chip
          m_ij = rowmax(S_ij)           // local softmax rescaling
          P_ij = exp(S_ij - m_ij)       // safe softmax numerator
          l_ij = rowsum(P_ij)           // local denominator
          // 在线更新running statistics
          m_new = max(m_i, m_ij)
          l_new = exp(m_i - m_new)*l_i + exp(m_ij - m_new)*l_ij
          O_i = diag(exp(m_i - m_new)) * O_i + exp(m_ij - m_new) * P_ij @ V_j
          m_i = m_new; l_i = l_new
          Store O_i, l_i, m_i to HBM
  // SRAM: ~20TB/s vs HBM: ~1.5-2TB/s (A100)
  // 结果：2-4× 加速，10-20× 内存节省
  ```

## HLX: A Unified Pipelined Architecture for Optimized Performance of Hybrid Transformer-Mamba Language Models

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  HLX 在 kernel 调度/数据流层面的核心实现包括两个新的细粒度流水线数据流：(i) **PipeFlash**——在 FA-2 的块级计算基础上进一步细粒度划分，每次处理 Q block 中的两行，将 QK^T、local softmax、PV、update O 四个步骤以流水线方式执行，其中 softmax 和 update O 与 MatMul（QK^T 和 PV）并发执行，实现非 MatMul 延迟隐藏。中间数据量从 FA-2 的 128KB（score/probability 矩阵）降至 1KB，减少 4.8×；(ii) **PipeSSD**——首次提出 SSD 的融合流水线执行，将 fused SSD 分为三个流水线阶段：第1阶段为 dA 相关预处理，第2阶段为 CB^T、CB^TLdt、Y_Diag，第3阶段为 Y_Off 与 states_N 并发计算 + Y_Final 与 update states。PipeSSD 减少 DRAM 访问 6.8×，中间数据从 642KB 降至 58.5KB（11×）。两个数据流通过控制每引擎处理的行数实现流水线阶段平衡：PipeFlash 中 QK^T 与 PV 按 `⌈block_size/d_head⌉` 调整行数；PipeSSD 中 Y_Off/states_N 阶段通过平衡总计算周期而非严格匹配行数实现灵活平衡。

  实验比较：(1) FA-2 和 SSD 的 compute utilization 对比 GPU (A100, H100) 和 TPUv3 baseline（序列长度 1K-128K）；(2) FA-2/SSD kernel 延迟加速比 vs GPU 和 TPU；(3) FA-3 on H100 vs PipeFlash on HLX^60（序列长度和 batch size 扫描）；(4) 端到端 Hybrid-2.7B 模型延迟加速比；(5) batch size 1-128 下的 compute utilization 和加速比变化（固定 seqlen=1K）；(6) PipeFlash vs FA-3 on H100 的单独比较。结果：FA-2 compute utilization 97.5%@128K（A100 native 61%），SSD compute utilization 78.4%（A100 26.9%，H100 <40%），平均加速 FA-2 1.75×/2.78×（vs A100/H100），SSD 2.91×/4.95×（vs A100/H100），端到端 1.56×/2.08×（vs A100/H100）。

- 后端平台是什么，配置是什么。
  GPU baseline: NVIDIA A100 80GB (312 TFLOPS, 1935 GB/s BW, 7nm, 826mm², 300W), NVIDIA H100 80GB (756 TFLOPS, 2000 GB/s BW, 4nm, 814mm², 350W)。TPU baseline: TPUv3 (61.5 TFLOPS, 450 GB/s BW, 16nm, 324mm², 225W, 16MB on-chip SRAM, 16GB DRAM)。HLX 配置三档：HLX^60 (614.4 TFLOPS, 30.4MB SRAM, 14nm, 475mm², 358W, 对标 H100)；HLX^30 (307.2 TFLOPS, 15.2MB SRAM, 14nm, 235.8mm², 174.64W, 对标 A100)；HLX^6 (61.44 TFLOPS, 3.04MB SRAM, 14nm, 47.16mm², 35.06W, 对标 TPUv3)。模型：Hybrid-2.7B (Mamba2attn-2.7B)，注意力层 30 head × d_head=128，SSD 80 head × d_head=64，d_state=128，block_size=256。数据类型 FP16。

- 评估性能的软件/脚本是什么。修改了什么。
  GPU baseline 使用 NVIDIA Nsight Systems 和 Nsight Compute 测量执行时间和 compute utilization。GPU 端运行来自 Mamba-2 GitHub 仓库 (https://github.com/state-spaces/mamba) [11] 的 CUDA 优化 FA-2/FA-3/SSD kernel。HLX 使用自研 cycle-level simulator，实现了 PipeFlash 和 PipeSSD 数据流的逐 cycle 模拟，以及完整 Hybrid 模型的端到端评估（含 FFN、conv1D、RMSNorm）。

  修改/新增内容：
  - PipeFlash 数据流：FA-2 的 4 步计算（QK^T → softmax → PV → update O）从块级改为更细粒度的行级（每次 2 行 Q）流水线，使 softmax/update O 与 MatMul 并发，score/probability 矩阵从 128KB 降为 1KB
  - PipeSSD 数据流：首次将 SSD 的 5 个分离 kernel（chunk cumsum, chunk state, state passing, BMM chunk, chunk scan）融合为单 kernel 的三阶段流水线（预处理 → Y_Diag → Y_Off∥states_N + Y_Final∥update states），中间数据 642KB→58.5KB
  - 流水线阶段平衡策略：根据 bottleneck（DPE MatMul 计算周期）控制每引擎处理行数，当 block_size=d_head=d_state 时可达近 100% compute utilization
  - 自定义 cycle-level simulator：模拟 URSC 内 DPE/RVPE/UpE 的流水线执行、NoC 数据转发、GS 暂存、DRAM 访问

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  论文未提供 HLX simulator 的开源链接（2026年5月检索未找到公开代码仓库）。GPU baseline 使用开源 Mamba-2 仓库 [11] (https://github.com/state-spaces/mamba) 的 CUDA kernel。HLX 自研 cycle-level simulator 的评估原理：以 Hybrid-2.7B 模型的计算图作为输入，将 FA-2 部分映射为 PipeFlash 数据流（Q block 内逐行流水线 QK^T→softmax→PV→update O），SSD 部分映射为 PipeSSD 数据流（三阶段流水线预处理→Y_Diag→Y_Off/states_N/Y_Final/update states），在三个 HLX 配置（HLX^6/30/60）下模拟 URSC 流水线执行周期，计算 DRAM 访问延迟（根据配置的 HBM2E/HBM2 带宽），输出每层/每 kernel 的 compute utilization 和延迟。GPU 端同理，使用 Nsight Compute/Systems 测量实际 CUDA kernel 执行时间，读取 SM 利用率。compute utilization 定义为实际达到的 TFLOPS / 理论峰值 TFLOPS。

## Flex Attention: A Programming Model for Generating Optimized Attention Kernels

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  FlexAttention 在 kernel 调度层面的核心实现包括：(i) BlockMask 数据结构——将 score 矩阵按 block（默认 128）分割，通过 kv_num_blocks [B,H,Num_Row] 和 kv_indices [B,H,Num_Row,Num_Col] 两个紧凑张量编码 block 级别稀疏性，内存开销 O(⌈Q_LEN/BS⌉ × ⌈KV_LEN/BS⌉) 远小于完整 score 矩阵 O(M×N)；(ii) Full/Partial Block 优化——区分 Full Blocks（无 score 被 mask，可跳过 mask_mod）和 Partial Blocks（部分被 mask，需逐元素执行 mask_mod），对 causal mask 等模式获得约 15% 性能提升；(iii) BlockMask 引导的间接内存访问——通过 kv_indices 映射跳过完全 masked 的 block，无需修改 kernel 即可支持 sliding window、local-global attention 等多种稀疏模式；(iv) 数据预取 Pipeline——SM 沿 Q_LEN tile 并行，每 SM 沿 KV_LEN 迭代，当前 score block 计算时预取下一 KV tile（HBM→SRAM），BlockMask 消除了条件分支检查从而允许高效 pipeline；(v) Paged Attention 支持——通过 BlockMask 转换合并 page table 的间接内存访问与 BlockMask 的稀疏跳过访问，无需重写 kernel，overhead <1%。

  实验比较：(1) 7 种 attention 变体的 kernel 速度 vs FAv2/FAv3/FAKV/SDPA，训练和推理；(2) block sparsity 的加速效果（proportional to sparsity）；(3) Paged Attention overhead vs FlashAttn-v2；(4) 端到端 torchtune 训练（Llama3-8B on Alpaca）和 gpt-fast 推理（Llama3.1-8B/70B）。结果：training forward 0.68×-1.43× vs FAv2, backward 0.86×-1.05×；inference 0.93×-1.45× vs FAKV；GQA+alibi 场景 5.37× vs FAKV；Paged Attention overhead <1%。

- 后端平台是什么，配置是什么。
  Nvidia H100 GPU（功率限制 650W，内存带宽限制 2.4TB/s），Nvidia A100 GPU（功率限制 330W），Nvidia A6000 GPU。KV size 固定 256 MiB，head dimension 64，数据类型 bfloat16。BlockMask 默认 block size=128。

- 评估性能的软件/脚本是什么。修改了什么。
  基于 PyTorch 核心 + Triton（https://github.com/triton-lang/triton）的 attention kernel。评估脚本使用 PyTorch benchmark 工具测量 kernel wall-clock time。
  
  修改/新增内容：
  - BlockMask 数据结构：两个张量 kv_num_blocks（每行非零 block 数）和 kv_indices（非零 block 列索引），由 create_block_mask() 通过 torch.vmap 自动生成
  - Full/Partial Block 分类逻辑：编译时判定 block 是否全部可见（full）、部分可见（partial）或全部 masked（oblivious），运行时对 full block 跳过 mask_mod 仅执行 score_mod
  - 间接内存访问策略：GPU block 的 workload 根据 kv_num_blocks 调整，通过 kv_indices 映射到下一个 block（可不连续，支持非连续 token 访问）
  - 数据预取 pipeline：在 Triton kernel 模板中插入预取逻辑，当前 block 计算时预取下一 KV tile
  - Paged Attention 集成：通过将 page table 的物理-逻辑映射融入 BlockMask 的 kv_indices，实现 fused indirect memory access

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  FlexAttention 已集成至 PyTorch 2.5+（https://github.com/pytorch/pytorch），attention-gym（https://github.com/pytorch-labs/attention-gym）提供工具。

  Kernel 调度评估原理与流程（以 H100 causal mask, training forward, QKV_LEN=16k, head_dim=64, bf16 为例）：

  1. **BlockMask 生成（编译时）**：
     - 输入：causal_mask_mod(b, h, q_idx, kv_idx) → return q_idx >= kv_idx
     - create_block_mask 用 torch.vmap 对 Q_LEN=16384, KV_LEN=16384, block_size=128 批量评估 mask_mod
     - 输出：score 矩阵被切分为 (16384/128)×(16384/128) = 128×128 个 block
     - Full blocks：对角线上方，全部 kv_idx ≤ q_idx 的 block（约 50% 总 block 数，对这些 block 运行时跳过 mask_mod）
     - Partial blocks：对角线上的 block（少量，需逐元素 mask_mod）
     - Oblivious blocks：对角线下方，全部 kv_idx > q_idx，完全跳过计算
     - kv_num_blocks: [B,H,128]，每行的非 oblivious block 数从 1 到 128
     - kv_indices: [B,H,128,128]，记录每行非 oblivious block 的列索引

  2. **GPU 调度**：
     - Q_LEN=16384, 每个 SM 处理一个 Q tile（Q_BLOCK_SIZE tokens，由模板决定）
     - 每个 SM 沿 KV_LEN 维度迭代处理一"行"block
     - SM 读取 kv_num_blocks[row] 确定该行需处理的 block 数
     - SM 通过 kv_indices[row, :] 获取非 oblivious block 的索引
     - 当前 block 计算时，通过预取管线加载下一 KV tile（HBM→SRAM）

  3. **Per-Block 计算**：
     - Full block：加载 Q tile + K tile → QK^T GEMM → score_mod（如 Alibi bias） → softmax（online rescaling） → PV GEMM。**不执行 mask_mod**。
     - Partial block：加载 Q tile + K tile → QK^T GEMM → mask_mod 逐元素 mask（设为 -inf）→ score_mod → softmax → PV GEMM。
     - Oblivious block：完全跳过（通过 kv_indices 自动排除）。

  4. **数据预取 Pipeline**：
     - 时间线：while iterating blocks: prefetch(KV_tile[i+1]) || compute_score(KV_tile[i]) → score_mod → online_softmax_update
     - 因为 BlockMask 消除了条件分支（不需要逐元素检查是否 masked），pipeline 可以高效流水线化

  5. **性能测量**：
     - CUDA event timing 测量 kernel wall-clock time
     - 吞吐量 = (effective FLOPs) / time，其中 effective FLOPs 仅计算非 masked block 的 FLOPs
     - Speedup = FlexAttention time / baseline（FAv2/FAKV）time
     - 对于 causal mask（50% sparsity）：forward 1.00×-1.22× vs FAv2
     - 对于 sliding window（更高 sparsity）：speedup 更显著

  6. **Paged Attention overhead 测量**：
     - 对比 FlexAttention with paged attention vs FlexAttention without paged attention vs FlashAttn-v2 without paged attention
     - Batch size=32, head_dim=64, num_heads=16, 变化 seq_len 和 page size
     - 通过将 page table 映射融入 kv_indices 实现 fused indirect memory access
     - 结果：平均 overhead <1%，远低于 vLLM 报告的 20-26% attention kernel overhead

## FlashInfer Efficient and Customizable Attention Engine for LLM Inference Serving

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现FlashInfer v0.2：基于FlashAttention-2（Turing/Ampere/Ada架构）和FlashAttention-3（Hopper架构）算法的CUDA/CUTLASS可定制attention kernel模板系统。核心kernel调度设计包括：(i) **Block-Sparse Row (BSR) attention kernels**——将KV-cache统一为block-sparse矩阵格式，支持任意block size $(B_r, B_c)$，通过从分散的global memory加载sparse tile到contiguous shared memory再调用dense tensor core MMA（使用LDGSTS 128B异步拷贝指令），支持vector-sparse（fine-grained sparsity with $B_c$=1）；(ii) **多tile size microkernel选择**——FA2 kernel提供tile sizes $(1,16,32,64,128) \times (32,64,128)$，FA3提供64倍数row tile sizes对齐WGMMA要求，根据硬件资源和workload特征（平均query长度、GQA group size $g$）的heuristic自动选择最优tile size；(iii) **JIT编译的attention variant kernel生成**——通过CUDA代码字符串定义的variant functors（QueryTransform, KeyTransform, ValueTransform, OutputTransform, LogitsTransform, LogitsMask）填充CUDA模板，用PyTorch JIT compiler编译并注册为custom operator，支持fused RoPE、soft-cap、sliding window、FlashSigmoid（无softmax）等变体；(iv) **Load-balanced persistent kernel调度**（Algorithm 1）——CPU端planning阶段根据query/KV长度信息计算CTAs间workload分配（类似Stream-K但保证deterministic aggregation order），生成work queue和partial-to-final output index mapping，GPU端persistent attention/contraction kernel按plan执行，兼容CUDAGraph的static grid size要求。

  实验比较：(i) Kernel bandwidth和FLOPs utilization vs FlashAttention main branch（含FA2和FA3），decode和prefill两种模式，batch size 16，3种sequence length分布——constant (1024)、uniform (512-1024)、skewed (Zipf, avg 1024)，A100 40GB和H100 80GB，f16精度；(ii) Fused RoPE+attention kernel vs FlashAttention unfused kernel（RoPE + attention分离），kernel bandwidth utilization对比，Vicuna-13B Streaming-LLM on MT-Bench，recent window size变化（changing window sizes）。

- 后端平台是什么，配置是什么。
  - NVIDIA A100 40GB SXM（Ampere SM80, server-class）：Tensor Core FP16 GEMM throughput约312 TFLOPS
  - NVIDIA H100 80GB SXM（Hopper SM90A, server-class）：支持TMA、WGMMA异步指令，FP16 GEMM throughput约989 TFLOPS
  - CUDA 12.4 + PyTorch 2.4.0，存储和计算均使用FP16精度
  - FlashInfer kernel支持Turing (sm75)到Hopper (sm90a)全系列GPU架构：Ampere/Ada(sm89)使用FA2 algorithm + LDGSTS异步拷贝，Hopper使用FA3 algorithm + WGMMA + TMA（dense contiguous KV-cache）/ LDGSTS fallback（sparse non-affine KV-cache）

- 评估性能的软件/脚本是什么。修改了什么。
  基于FlashAttention-2（Dao, 2023）和FlashAttention-3（Shah et al., 2024）算法的CUDA/CUTLASS模板实现。修改内容包括：
  (1) **Sparse tile loading module**：非contiguous KV-cache地址通过BSR indices数组计算→使用`cp.async` (LDGSTS, 128B width)将分散global memory数据gather到contiguous shared memory→shared memory内数据变为dense tile，后续dense MMA路径与FlashAttention一致；
  (2) **Multi-tile-size microkernel generation**：在CUDA template中参数化query tile size（$T_q \in \{1,16,32,64,128\}$）和K/V tile size（$T_{kv} \in \{32,64,128\}$），compile-time resolve register和shared memory constraint，优先maximize SM occupancy。$T_q=1$使用CUDA core路径（因mma指令min row=16），$T_q \geq 16$使用Tensor Core路径；
  (3) **JIT compiler pipeline**：variant specification（CUDA code定义functors + additional tensors + data types）→ template population → PyTorch `torch.utils.cpp_extension.load_inline` 编译→ DLPack framework-agnostic interface注册custom operator；
  (4) **Load-balanced scheduler**：CPU端$\{l_{qo}(i), l_{kv}(i)\}$输入→计算max KV chunk size $L_{kv}$→split query tiles into KV chunks→sort by length descending→greedy min-cost CTA assignment (Algorithm 1)→输出plan info（CTA work queue + partial/final output index mapping）→async copy to GPU workspace buffer；
  (5) **Persistent kernel设计**：单persistent kernel合并attention+contraction两阶段，fixed grid size兼容CUDAGraph，workspace buffer fixed offset确保CUDAGraph capture指针不变。

  评估脚本：使用CUDA event timing测量kernel wall-clock time，bandwidth utilization = achieved bytes / peak bandwidth，FLOPs utilization = achieved FLOPs / peak FLOPs。Benchmark中decode核心理念——输出是$O(l_{qo})$密集的，FLOPs较低时bandwidth为主要限制。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  已开源：https://github.com/flashinfer-ai/flashinfer (Apache-2.0)。已集成进SGLang、vLLM、MLC-Engine等主流LLM serving框架。

  评估原理与流程（以H100 decode kernel, uniform seqlen 512-1024, batch=16, GQA group=4, head_dim=128, f16）：
  1. **Input准备**：Ragged tensors Q/O（shape `[total_tokens, nheads, head_dim]`，各请求token数通过`indptr`数组索引），KV-cache BSR matrix（shape `[total_blocks, B_r, nheads_kv, head_dim]`，`B_r=T_q`matched query tile，`B_c`=1 for page-level sparsity）。对应metadata：`kv_indptr`（row pointers for BSR）、`kv_indices`（column indices for non-zero blocks）。
  2. **Compile-time tile selection**：平均query长度（GQA fused）= $l_{qo} \times g$。若avg≥128选$T_q=128$，avg∈[64,128)选$T_q=64$，依此类推。Register constraint: $T_q \times T_{kv}$ 需与shared memory size constraint同时满足，求解max occupancy tile size。
  3. **CPU scheduler plan**（Algorithm 1）：给定$\{l_{qo}(i), l_{kv}(i)\}_{i=1}^{16}$，hyperparameters α=1, β=1（default cost function $cost(l_q,l_{kv})=\alpha l_q + \beta l_{kv}$）。Compute $L_{kv}= \sum \lceil l_{qo}(i)/T_q \rceil \cdot l_{kv}(i) / \#CTA$。每个query tile $(T_q)$ 的KV split为chunks of max $L_{kv}$，得work queue $W$。Sort $W$ descending→greedy assign to CTAs with min current cost→输出plan info（CTA-wise chunk assignment + partial output aggregation mapping）。
  4. **Persistent attention kernel**：Grid size = compiled constant。每个CTA读取plan info中自己的work queue→逐chunk处理：(a) Load sparse KV tile: compute block indices from BSR metadata→`cp.async` LDGSTS从HBM搬移到SMEM；(b) Load Q tile: dense affine addressing→LDGSTS；(c) $S_{ij}=Q_iK_j^T$ by WGMMA (Hopper) 或 HMMA (Ampere)；(d) Online softmax: rowmax (CUDA core REDUX)→exp (MUFU.EX2)→rowsum→rescale running O and l→$\tilde{P}_{ij}V_j$ by WGMMA/HMMA→accumulate partial O and l。
  5. **Persistent contraction kernel**（与attention合并入同一persistent kernel）：各CTA的attention partial outputs（Attention State: `(O_partial, LSE_partial)`）按plan info index mapping进行$\oplus$ composition（attention compose operator, equation in Section 2.2）→final O。$\oplus$操作：$O_{final} = (\exp(LSE_1)O_1 + \exp(LSE_2)O_2)/(\exp(LSE_1)+\exp(LSE_2))$，$LSE_{final}=\log(\exp(LSE_1)+\exp(LSE_2))$。
  6. **Performance measurement**：CUDA event timing测kernel wall-clock time。Bandwidth = (Q size + KV size + O size read/written bytes + partial O bytes) / time。FLOPs = $(2 \times total\_kv\_tokens \times head\_dim \times nheads_{qo})$ (QK^T + PV GEMM, 2× for MAC) / time。
  7. **Output**：Figure 8——decode kernel bandwidth utilization（FlashInfer显著高于FlashAttention on uniform and skewed distributions因load-balanced scheduler + multi-tile-size选择）；prefill kernel FLOPs utilization（similar but FlashInfer stable across distributions）。Fused RoPE+attention kernel: Figure 9——FlashInfer fused kernel bandwidth util vs FlashAttention unfused kernel (1.6-3.7× higher)。

## FlashAttention-T: Towards Fully Tensorized Attention by Exploiting Tensor-Vector Parallelism

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现FlashAttention-T：基于FlashAttention-2/3的fused attention kernel，通过repurposing tensor MMA指令（HMMA.1688.F32.TF32 / HGMMA.64x8x8.F32.TF32）将softmax关键原语（element-wise scaling、fused multiply-add、row-sum reduction）offload到idle tensor unit上执行，并通过架构感知调度技术（Ampere的ILP interleaving + Hopper的TLP跨warpgroup并行）将tensorized和vectorized softmax计算并行化，消除vector interval bottleneck。
  实验比较：(i) Attention throughput vs FlashAttention-2、FlashAttention-3、FlashInfer、Triton fused attention kernel，固定总token数16384、总hidden dimension 2048，变化sequence length（512-8192）和head dimension（64/128），FP16-FP32和FP8-FP32两种精度，causal和non-causal两种mask模式；(ii) Vector interval ratio对比baseline（FlashAttention-2在A100上29.8%，FlashAttention-3在H100上36.3%）；(iii) Ablation study——FA2 baseline、FA2+Max16（仅加入surrogate maximum）、AllTensor（全部tensorized）、ILP FA-T（完整ILP调度）的throughput对比；(iv) Synthetic attention accuracy——随机Q/K/V输入下RMSE vs FP64 reference，测试不同outlier variance τ²；(v) Generative benchmark——HumanEval Pass@10和MMLU score在Llama3.1 8B、Ministral 8B、Qwen3 8B、Llama2 13B、Mistral NeMo、Qwen3 14B上的功能正确性。

- 后端平台是什么，配置是什么。
  - NVIDIA A100 80GB SXM4（Ampere SM80, server-class）：FP16-FP32 GEMM吞吐312 TFLOPS，HBM带宽约2.0TB/s
  - NVIDIA Jetson AGX Orin 64GB（Ampere SM87, edge-class）：集成GPU，支持FP16-FP32 tensor MMA指令
  - NVIDIA H100 80GB PCIe（Hopper SM90A, server-class）：FP16-FP32 GEMM吞吐约624 TFLOPS（2× A100），FP8-FP32吞吐约989 TFLOPS，支持异步WGMMA指令
  - CUDA环境：基于FlashAttention-2（优化Ampere）和FlashAttention-3（优化Hopper）的代码库修改，nvcc编译器

- 评估性能的软件/脚本是什么。修改了什么。
  FlashAttention-T基于FlashAttention-2（针对Ampere GPUs）和FlashAttention-3（针对Hopper GPUs）开源代码修改实现。修改内容包括：
  (1) **Repurposed tensor MMA instructions**：在fused attention kernel内插入自定义MMA调用——通过特殊operand value assignment使tensor unit执行softmax原语：
    - Element-wise scaling：设置fragment B为包含scaling factor α的pattern（图5a），使D(v,t)=α·A(σ(v),t)，permutation σ通过Cayley distance最小化swap开销
    - Fused multiply-add：同上B赋值 + 设置C fragment为offset值，使D(v,t)=α·A(σ(v),t)+C(v,t)
    - Row-sum reduction：设置B为全1 pattern（图5b），使D'(v,t)=∑_{t∈κ_i} A(σ(v),t)，再加intra-thread additions得row sum
    - 关键设计：repurposed instructions直接在GEMM MMA的output fragment上操作（同一register space），零copy overhead
  (2) **Tensorized online softmax**（Algorithm 1）：用X-row tile surrogate maximum m̂[i]替代逐行maximum m[i]，满足tensor MMA uniform scaling factor约束，支持HMMA.1688(X=16)和HGMMA.64x8x8(X=64)
  (3) **ILP scheduling for Ampere**：horizontal/vertical split策略将softmax计算分为tensorized和vectorized部分，在warp内interleave tensor和vector指令
  (4) **TLP scheduling for Hopper**：仅tensorize P̃ row-summation（leaf-stage最小化register dependency），将repurposed WGMMA指令加入下一迭代的QK^T+PV WGMMA batch，与另一warpgroup的vector softmax并行

  评估脚本：cycle-level profiling使用NVIDIA cycle-counting routines（如`clock64()`）测量t_vec、t_iter、t_softmax；throughput使用CUDA event timing；accuracy使用RMSE vs FP64 reference

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  开源artifact在Zenodo：https://doi.org/10.5281/zenodo.17673796（v1, 2025-11-22），基于FlashAttention-2和FlashAttention-3代码库构建。

  评估原理与流程（以Ampere A100 ILP FlashAttention-T为例，h=128, s=4096, FP16-FP32, non-causal）：
  1. **Input准备**：Q/K/V tensors in FP16，shape `[batch, seq_len=4096, nheads, head_dim=128]`，在GPU HBM中
  2. **Warpgroup tile划分**：每个warpgroup计算O的一个row block。Attention row block进一步按16-row tile分给每个warp。K/V tile沿sequence维度划分为s/b_N个block（b_N为sequence维度的tile size）
  3. **ILP FlashAttention-T单次iteration执行流程**（每个iteration处理一个KV block，参考图7a）：
     a. QK^T GEMM on tensor unit（HMMA.1688.F32.TF32指令，FP16→TF32 accumulator）
     b. **Softmax阶段（tensor+vector并行）**：
        - Step ① Compute 16-row surrogate m̂ via warp all-reduce REDUX（vector）
        - **Step ②③ Tensorized rescale**：repurposed HMMA.1688.F32.TF32 for attention output O rescaling and attention logits S rescaling——element-wise scaling with uniform m̂（tensor unit，与vector指令ILP-interleaved）
        - Step ④ Vector exponentiation exp₂(Z)（vector unit，MUFU.EX2或等价的快速exp近似）
        - **Step ⑤ Tensorized row-sum reduction**：repurposed HMMA.1688.F32.TF32 for P̃ row sum（tensor unit，与vector指令ILP-interleaved）
        - Vector max/exp/fma等non-tensorizable操作（vector unit）
     c. PV GEMM on tensor unit（HMMA.1688.F32.TF32）
     d. Write O_i, ℓ_i, m_i to HBM
  4. **Timing测量**：使用CUDA event timing测总iteration time t'_iter；使用cycle-level counter（clock64()）分段测量softmax time t'_softmax；vector interval ratio = (t'_softmax - (t_vec - t'_softmax))/t'_iter
  5. **Throughput计算**：TFLOPs/s = 总compute FLOPs / runtime(ms)，其中GEMM FLOPs = 2s²h（QK^T）+ 2s²h（PV），softmax FLOPs根据2.1节公式计算

## FlashAttention Fast and Memory-Efficient Exact Attention with IO-Awareness

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现一个fused CUDA kernel，将attention的全部操作（QK^T矩阵乘 → softmax含masking和dropout → PV矩阵乘）融合为单个GPU kernel，避免中间$N \times N$ attention矩阵在HBM中的materialization。核心设计：(i) **Tiling with online softmax**——将Q/K/V分块加载到SRAM，沿KV block维度（外循环）和Q block维度（内循环）做block-wise计算，通过running max $m$和running sum $\ell$维护正确归一化；(ii) **Recomputation for backward**——前向仅保存输出O和softmax统计量$(m,\ell)$（$O(N)$内存），反向在SRAM中重计算S和P，比标准方法（从HBM读取$O(N^2)$中间值）更快（即使FLOPs增加）；(iii) **Block-sparse variant**——跳过预定义稀疏mask中零值block的全部计算（BMM1除外）。
  实验比较：(i) Forward+backward runtime vs standard attention（seq length 1024, head dim 64, 16 heads, batch 64, A100 GPU），测量GFLOPs、HBM R/W(GB)、Runtime(ms)；(ii) Block size消融——$B_c$从64到512下forward runtime变化，验证HBM accesses是runtime主导因素；(iii) Block-sparse FLASHATTENTION runtime vs sparsity比例（seq length 4K）；(iv) Full benchmark——forward+backward runtime和attention memory usage随sequence length（128-64K）变化，vs PyTorch exact attention和多种approximate/sparse attention（Linformer, Linear Attention, Performer, Reformer, Smyrf, Local Attention）。

- 后端平台是什么，配置是什么。
  - NVIDIA A100 GPU (Ampere架构)：40GB HBM，带宽1.5-2.0TB/s，108 SMs，每SM 192KB on-chip SRAM（带宽约19TB/s）
  - Benchmark配置：seq length 128-64K，head dim 64，16 heads，batch size 64，key-padding mask + dropout，单卡A100 40GB
  - CUDA环境：自编CUDA kernel，基于NVIDIA Apex FMHA代码（https://github.com/NVIDIA/apex/tree/master/apex/contrib/csrc/fmha）

- 评估性能的软件/脚本是什么。修改了什么。
  - 自编CUDA kernel实现FLASHATTENTION的forward和backward pass，以及block-sparse FLASHATTENTION variant
  - 修改：(i) 替代标准PyTorch attention实现（`torch.nn.functional.scaled_dot_product_attention`或等效的手写attention），将attention计算替换为单个fused CUDA kernel调用；(ii) Forward kernel中内循环结构——对每个KV block $j$，内循环遍历Q blocks $i$：计算$S_{ij}=Q_iK_j^T$（Tensor core MMA）→ online softmax（CUDA core: rowmax + EXP MUFU.EX2 + rowsum + rescale + combine）→ 累加$O_i$（Tensor core MMA: $\tilde{P}_{ij}V_j$ + rescale）→ write $O_i, \ell_i, m_i$ to HBM。中间$S_{ij}, \tilde{P}_{ij}$仅驻留SRAM；(iii) Backward kernel——利用保存的统计量$(m,\ell)$和输入Q/K/V/O重计算$S_{ij}, P_{ij}$ in SRAM，计算$dQ, dK, dV$梯度；(iv) Block-sparse kernel——仅在内循环的$M_{ij}=0$时跳过softmax+PV计算，其余逻辑相同。
  - Benchmark脚本：测量CUDA kernel wall-clock time（CUDA event timing）、HBM读写量（通过理论分析验证）、peak memory allocation（`torch.cuda.max_memory_allocated()`）

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  已开源：https://github.com/HazyResearch/flash-attention（BSD许可证）。安装：`pip install flash-attn`。Python接口：`from flash_attn import flash_attn_func; output = flash_attn_func(q, k, v, causal=False, dropout_p=0.0)`。

  评估原理与流程（以单卡A100 forward+backward benchmark为例，N=1024, d=64, 16 heads, batch=64）：
  1. **Input准备**：在PyTorch中创建Q/K/V tensors in FP16/BF16：`[batch=64, seq_len=1024, nheads=16, head_dim=64]`，位于GPU HBM。应用key-padding mask和optional dropout mask。
  2. **Forward kernel launch**（单次CUDA kernel调用）：
     a. Compute block sizes: $B_c = \lceil 192\text{KB} / (4 \times 64 \times 2\text{B})\rceil = \lceil 12288 / 512\rceil \approx 384$，$B_r = \min(384, 64) = 64$（head_dim bound）。
     b. 划分：$T_c = \lceil 1024/384 \rceil = 3$个KV blocks（大小384），$T_r = \lceil 1024/64 \rceil = 16$个Q blocks（大小64）。
     c. 外循环（KV blocks, j=1..3）：从HBM加载$K_j(384\times64), V_j(384\times64)$到SRAM（约49KB per matrix）。
       内循环（Q blocks, i=1..16）：从HBM加载$Q_i(64\times64)$到SRAM（约8KB）。On-chip: $S_{ij}=Q_iK_j^T$（64×384, FP16 GEMM on Tensor core）→ rowmax（CUDA core reduction: 64 values per row）→ $\exp(S_{ij}-\tilde{m}_{ij})$（MUFU.EX2）→ rowsum → rescale running $m_i,\ell_i$ → $\tilde{P}_{ij}V_j$（Tensor core MMA）→ rescale and accumulate $O_i$ → write $O_i,\ell_i,m_i$ to HBM。
     d. 中间$S_{ij}$和$\tilde{P}_{ij}$（64×384 each）驻留在SRAM中，不写入HBM。
  3. **Backward kernel launch**（单次CUDA kernel调用）：
     a. 加载$O, dO, \ell, m$和Q/K/V from HBM。
     b. 对每个(i,j) block pair在SRAM中重计算$S_{ij}, P_{ij}$。
     c. 计算$\partial Q_i = dO \cdot V_j^T \cdot \text{diag}(P_{ij})$等梯度（由softmax反向公式展开）。
     d. Write $dQ, dK, dV$ to HBM。
  4. **Performance measurement**：CUDA events记录`flash_attn_func`的forward+backward总时间。FLASHATTENTION: GFLOPs=75.2（高于standard的66.6，因recomputation），HBM R/W=4.4GB（远低于standard的35.3GB），Runtime=11.7ms（vs standard 35.1ms，3× faster）。
  5. **Memory measurement**：Standard attention memory = $O(batch \times heads \times N^2)$ = 64×16×1024²×2B ≈ 2.1GB（仅S和P）。FLASHATTENTION memory = $O(N)$额外 = 64×16×1024×4B ≈ 4.2MB（仅$m,\ell$）。Memory减少约500×。
  6. **Scaling验证**（Figure 3）：runtime和memory随N从128到64K变化。FLASHATTENTION runtime grows quadratically（FLOPs $O(N^2)$），但constant factor远小于baseline；memory grows linearly。Block-sparse FLASHATTENTION runtime在64K时比所有approximate attention方法都更快。
  7. **Output**：Figure 2（left: standard vs FLASHATTENTION的runtime/HBM breakdown；middle: block size对runtime的影响——block越大HBM accesses越少，直到arithmetic瓶颈；right: block-sparse sparsity vs runtime）。

## Flash Multi-Head Feed-Forward Network

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现SRAMFFN——I/O-aware fused kernel用于FlashMHF的multi-head FFN计算。Analogous to FlashAttention的online softmax，SRAMFFN通过blockwise computation（沿着d_ff维度分块）避免在HBM中materialize大的intermediate activation tensor (SiLU(QK^T) ⊙ (QU^T))。两个实现版本：（1）Triton版本（Algorithm 1-3，forward+backward two-pass DK/DU/DV），可在consumer GPU（RTX3090等）上高效运行；（2）ThunderKittens/TK版本（Algorithm 4-5，forward+backward），针对Hopper架构，利用asynchronous data movement（TMA）、warp-group specialization（consumer/producer分工）、stage/ring buffer。Forward pass中producer warpgroup预取K/U/V tiles到SRAM stage buffer，多个consumer warpgroup并行处理不同x-block（sequence tile），各自独立迭代所有inter-tiles，仅在每个sub-network开始时同步router R。
  实验比较：FlashMHF vs SwiGLU FFN vs MH-FFN的（i）peak memory consumption（MB）across sequence lengths 192-16128（Figure 8a, Table 5）；（ii）latency（ms）across sequence lengths（Figure 8b, Table 5）。FlashMHF vs SwiGLU达到3-5x memory reduction和1.00x-1.08x inference speedup。

- 后端平台是什么，配置是什么。
  - NVIDIA H100 GPU (Hopper架构)，单卡benchmark，batch size=8
  - 配置：d_e=384, H=16, E=22, d_h=128，sequence length从192到16128（Table 5）
  - Triton kernel版本可在consumer GPU（RTX3090）上运行但不在Hopper上高效
  - 对比baseline：cuBLAS优化的标准SwiGLU FFN kernel

- 评估性能的软件/脚本是什么。修改了什么。
  - 自编benchmark脚本测量memory和latency，对比FlashMHF kernel vs SwiGLU FFN vs MH-FFN
  - Triton实现（Algorithm 1-3）：编写SRAMFFN-FORWARD-TRITON、SRAMFFN-BACKWARD-TRITON(DQ,DR)、SRAMFFN-BACKWARD-TRITON(DK,DU,DV)三个Triton kernel。Grid并行化over batch/head/sequence blocks。Forward pass在inner loop中blockwise累积输出O_acc（公式15）；backward pass分两轮：第一轮计算dQ和dR，第二轮计算dK/dU/dV
  - ThunderKittens/TK实现（Algorithm 4-5）：Hopper架构上使用warp-group specialization——1个producer warpgroup异步预取Q/R/K/U/V tiles，CON_WARPGRPS≥2个consumer warpgroup并行计算。Producer维护NUM_STAGES ring buffer实现流水线预取；consumer在stage间wait/signal实现producer-consumer同步
  - 核心优化：kernel避免materialize intermediate activation (SiLU(QK^T)⊙(QU^T)) in HBM，改为blockwise SRAM累加；output在单个fused kernel中直接生成

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  论文声明代码将公开于 https://anonymous.4open.science/r/FlashMHF-9395（当前匿名审阅）。将包含Triton和ThunderKittens两套kernel实现，可直接替换标准FFN module。

  评估原理与流程（以Hopper上FlashMHF单层memory/latency benchmark为例，d_e=384, H=16, E=22, d_h=128, L=4096, bs=8）：

  1. **Input**：Q ∈ R^{B×H×L×d_h}（输入query tensor），K/U/V ∈ R^{H×E×d_e×d_h}（sub-network参数），R ∈ R^{B×H×L×E}（router gating weights）。这些均在GPU HBM中。

  2. **SRAMFFN-FORWARD-TK kernel launch**（Algorithm 4）：
     Grid配置：x = ceil(L / (BLOCK_SEQ · CON_WARPGRPS)), y = H, z = B
     a. **Warmup (producer)**：prefetch所有consumer的Q tiles → prefetch sub-network e=0的router R → prefetch首批NUM_STAGES (K,U,V) inter-tiles到SRAM stage buffer
     b. **Producer loop**：遍历所有inter_tile = NUM_STAGES ... E·(d_e/BLOCK_INTER)-1。每个iteration先wait consumer完成当前stage（释放buffer），再prefetch下一个(K,U,V) tile。若inter_tile是new sub-network的第一tile，额外prefetch router R rows
     c. **Consumer warpgroup c ∈ {0,...,CON_WARPGRPS-1}**（独立并行）：load自己的Q tile（x-block）→ O_acc=0 → 遍历所有inter_tiles → wait producer填充当前stage → M=Q_blk·K_tile^T; N=Q_blk·U_tile^T → S=SiLU(M)⊙N → S=S⊙r（apply router row）→ O_acc+=S·V_tile → signal producer → 最后store O_acc to global memory
     d. **关键**：中间tensor M/N/S仅驻留在SRAM中，不写入HBM。O_acc在SRAM中累积，最终才write to HBM

  3. **Memory profiling**：torch.cuda.max_memory_allocated()测量peak memory。FlashMHF peak memory = O(d_model·L) vs SwiGLU = O((d_ff+d_model)·L) vs MH-FFN = O((d_ff·H+d_model)·L)。L=16128时FlashMHF≈3016MB, SwiGLU≈9966MB, MH-FFN→OOM

  4. **Latency profiling**：CUDA event记录kernel execution wall time。FlashMHF vs SwiGLU latency对比（Table 5）。L=4032时FlashMHF=126.40ms vs SwiGLU=127.44ms (~1.01x)；L=16128时FlashMHF=497.40ms vs SwiGLU=535.20ms (1.08x)。Speedup源于消除HBM↔SRAM的intermediate activation读写

  5. **Output**：Memory (MB)和Latency (ms)随L变化的对比表/图。FlashMHF在所有L下memory均优于SwiGLU（3-5x），latency从略慢（短序列cuBLAS更优）到略快（长序列消除I/O瓶颈）

  6. **Backward pass**（Algorithm 5, SRAMFFN-BACKWARD-TK）：producer prefetch Q/dO/R for sequence tiles → 两个consumer warpgroups各自处理不同的inter-tile (A/B)，并行计算dK/dU/dV → 每个sequence tile t处交换并累加dQ^(A)+dQ^(B)和dR^(A)+dR^(B) → store到global memory

## Composing Distributed Computations Through Task and Kernel Fusion

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  Diffuse 的 kernel fusion 部分属于 kernel 调度/运行时计算优化。实现包括：(i) 融合 task 内部的 MLIR-based kernel 生成与 fusion —— 将顺序的多个 task body 组合为单个 fused kernel，通过 polyhedral 优化融合嵌套循环、消除临时分配（memref dialect allocation）、并行化循环；(ii) 将优化后的 MLIR lowered 为 GPU kernel launch（via MLIR GPU dialect）或 CPU OpenMP parallel region。Kernel fusion 使得原本需要多次 pass over data 的多个独立 kernel 合并为一次 pass，提升 arithmetic intensity 并大幅减少 memory traffic。
  实验比较：Diffuse fused vs unfused 在 7 个应用的 weak-scaling throughput。额外比较 vs MPI-based PETSc 和 vs 手工优化版本。特别关注 kernel fusion 的效果——论文明确指出"task fusion alone can only reduce runtime overhead... did not yield speedups"，kernel fusion 才是加速的主要来源。

- 后端平台是什么，配置是什么。
  NVIDIA A100 DGX SuperPOD：每节点 8×A100 80GB（NVLink+NVSwitch），双路 128 核 AMD 7742 Rome，2TB 内存，InfiniBand 互联（每节点 8 NICs），最多 128 GPUs。

- 评估性能的软件/脚本是什么。修改了什么。
  评估应用（Figure 9）：Black-Scholes（67 data-parallel 操作，全部 fusible），Dense Jacobi Iteration（密集矩阵-向量乘 + 2 个 fusible 向量操作），Sparse Krylov Solvers CG 和 BiCGSTAB（cuPyNumeric/Legate Sparse 实现），Geometric Multi-Grid GMG（CG-based + V-cycle preconditioner），Computational Fluid Dynamics CFD（Navier-Stokes 2D channel flow），Shallow Water Equation TorchSWE（浅水方程求解器）。
  修改：无需修改用户应用代码。库开发者需为每类操作注册 MLIR generator function（每个操作约 50–100 行 C++ 代码）。Diffuse 自动选择 window size（通过逐步增大 window size 直到所有 task 都被融合的过程）。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  Diffuse 的 kernel fusion 编译 pipeline 在 CPU 上运行（JIT compilation），生成优化后的 MLIR module，通过 MLIR GPU lowering pass 生成 CUDA kernel，在 A100 GPU 上执行。评估通过 12 次 run 取平均（排除最快和最慢），warmup 迭代不计入 steady-state timing。

  Kernel fusion 评估原理（以 Black-Scholes 为例，67 个 data-parallel 操作，window size=70）：
  1. 输入：cuPyNumeric Black-Scholes 应用生成 67 个 element-wise index task。每个 task body 由 MLIR generator 生成 fragment（Figure 8a 示例：memref<?xf64> 参数 + affine.for 循环 + arith.addf/mulf 计算）。
  2. Task fusion analysis：所有 67 个 task 满足全部 4 个 fusion constraints（相同 launch domain，相同 partition，无 aliasing，无 reduction conflict），全部融合为单个 index task。
  3. MLIR kernel 组合与优化（Figure 8 pipeline）：67 个 task body 顺序组合为初始 fused kernel → temporary store elimination 消除 64 个中间 store（降级为 task-local memref.alloc）→ polyhedral 循环融合将 67 个独立 affine.for 合并为单个 affine.par 循环 → memref.alloca 消除（因 temporary 被 inlined）→ 生成最优 single-pass kernel。
  4. GPU lowering：MLIR affine → GPU dialect lowering pass，将 affine.par 映射到 CUDA grid/block/thread launch configuration。
  5. 执行：在 A100 上以单 CUDA kernel 一次 pass 完成全部 67 个 element-wise 操作的计算。Arithmetic intensity 从 67 次 HBM read/write per element 降至 1 次 read + 1 次 write per element。
  6. 输出：128 GPUs 实现 10.7× speedup vs unfused。Unfused 版本每迭代 67 个 task → fused 后仅 1 个 task（Figure 9: Tasks per Iteration 67→1）。

  Compilation overhead 评估（Figure 13）：测量 warmup 时间（含 compilation）vs unfused warmup。Black-Scholes 编译 0.06s vs unfused 0.38s（更快因为只需编译 1 个 kernel），其他应用需 25–119 次迭代 amortize 编译开销。论文认为这些开销合理，因为科学计算应用通常运行数千至数百万次迭代。

## BLASST Dynamic BLocked Attention Sparsity via Softmax Thresholding

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现两套专用CUDA kernel：prefill kernel（compute-bound优化）和decode kernel（memory-bound优化），均基于FlashAttention-3/4的tiled online softmax pipeline修改。Prefill kernel跳过softmax指数计算和attention-value MMA操作（tensor core），decode kernel跳过Value matrix的HBM加载（memory bandwidth节省）。两个kernel的skip decision通过复用online softmax已有的running maximum和local maximum统计量，仅增加少量指令（warp-level VOTE + ATOMIC to shared memory），零额外开销。
  实验比较：(i) BLASST prefill kernel vs FlashAttention-3 BF16 baseline在B200和H200上的speedup，sparsity从0%到~94%；(ii) BLASST decode kernel vs FlashAttention-3 BF16 baseline在B200和H200上的speedup；(iii) 0% sparsity时的overhead验证（kernel overhead被pipeline隐藏）。

- 后端平台是什么，配置是什么。
  - NVIDIA Blackwell B200 GPU：prefill batch=1, 64K seq len; decode batch=148, 32K seq len
  - NVIDIA Hopper H200 GPU：prefill batch=1, 64K seq len; decode batch=128, 16K seq len
  - 容器化环境：Docker（NVIDIA Container Toolkit）或Singularity，基于nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc6 容器镜像
  - 编译：CUDA nvcc，target sm90a（Hopper）和 sm100（Blackwell）

- 评估性能的软件/脚本是什么。修改了什么。
  - 框架：TensorRT-LLM和FlashInfer中集成的BLASST CUDA kernel
  - 评估脚本：自动sweep不同threshold scale factor，测量sparsity、执行时间、memory bandwidth、speedup vs dense baseline
  - 修改：(i) prefill kernel pipeline（Figure 3）：在softmax warpgroup中增加skip check逻辑（predicate + VOTE + ATOMIC），当block被跳过时消除BMM2（MMA）和exp计算，压缩pipeline时间线；(ii) decode kernel pipeline（Figure 4）：改为batched load scheduling——连续发射多个K^TQ计算（back-to-back BMM1），再根据skip check结果批量发射仅需要的V_j加载，消除pipeline bubble；(iii) 两类kernel均支持MHA/GQA/MQA/MLA attention变体。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  已开源集成到TensorRT-LLM（https://github.com/NVIDIA/TensorRT-LLM）和FlashInfer。Artifact repo: https://github.com/cameronshinn/blasst-ae-mlsys26.git（Apache 2.0）。

  评估原理和流程（以Hopper prefill kernel benchmark为例）：
  1. 容器启动：`./start_docker.sh` → 自动pull TensorRT-LLM release容器并挂载repo到/workspace
  2. 进入目录：`cd /workspace/hopper_prefill/` → 阅读README.md中的具体编译和运行指令
  3. 编译kernel：nvcc将BLASST CUDA kernel模板编译为sm90a target的二进制
  4. 自动benchmark脚本：对一系列threshold scale factor进行sweep，对每个factor：
     a. 创建随机初始化tensor（Q/K/V，模拟64K sequence length, batch=1）
     b. 执行BLASST kernel → 收集execution time和memory bandwidth
     c. 执行FlashAttention-3 baseline → 收集baseline执行时间
     d. 执行closed sm100 binary → 测量实际达到的exact sparsity percentage
     e. 计算speedup = baseline_time / BLASST_time
  5. 结果输出到stdout：sparsity vs speedup表（对应论文Table 5）

  Prefill kernel pipeline详解（Figure 3）：
  - FlashAttention: 每轮mainloop迭代包含BMM1(QK^T) → softmax(EX2) → BMM2(PV)，顺序执行，18个time units完成4轮
  - BLASST prefill: 同样执行全部BMM1，但在skip check通过的block中跳过softmax和BMM2。例如loop 1和loop 3被跳过时，pipeline被压缩到14个time units。Schedule中不同tile row（T0/T1）的运算用不同色调标注，skip check、rowsum和softmax scaling被隐藏在BMM1之后。

  Decode kernel pipeline详解（Figure 4）：
  - FlashAttention decode: V load → BMM1(QK^T) → BMM2(PV)，顺序流水线，38个time units完成所有V loads
  - BLASST decode (batched load scheduling): 连续K1^TQ, K2^TQ, ..., K_B^TQ背靠背（BMM1s）→ 批量skip check → 仅发射通过检查的V_j加载。需维护B个S_j的shared memory buffer（因query len=1，开销小）。消除pipeline bubble，31个time units完成。Arrows指示scoreboard dependency：skip check完成后才能决定加载哪些V blocks。

## ACS Concurrent Kernel Execution on Irregular, Input-Dependent Computational Graphs

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  ACS（Automatic Concurrent Scheduling）在运行时对顺序发射的 GPU kernel 进行乱序调度，通过在固定大小的调度窗口内执行依赖检查和状态跟踪来识别独立 kernel 并并发执行。实现分为两部分：(i) ACS-SW：纯软件实现，使用用户态运行时系统维护调度窗口，通过 CUDA stream 并发发射独立 kernel；(ii) ACS-HW：硬件-软件协同实现，在 GPU 硬件中实现调度窗口以消除 CPU-GPU 同步开销。
  实验比较了四种配置：(i) Baseline（单 CUDA stream 串行执行，cuDNN/JAX 实现）；(ii) ACS-SW（真实硬件上评估，仅 Deep RL 仿真 workload）；(iii) ACS-SW-Sim（模拟器上评估 ACS-SW，用于与 ACS-HW 对比）；(iv) ACS-HW（模拟器上评估硬件-软件协同机制）；(v) CUDAGraph（将核间依赖构建为 DAG 后提前发送给 GPU）。评估指标为运行时加速比和 GPU 达到的 occupancy。

- 后端平台是什么，配置是什么。
  - **真实硬件（ACS-SW）**：Intel Core i7-11700K CPU @ 3.6GHz，4-wide OOO dispatch，32-entry LSQ，L1D+L1I 32KB 4-way LRU，L2 256KB 8-way LRU，L3 1MB 16-way LRU，2-channel DDR4 DRAM（4GB / 12GB variant）；NVIDIA RTX 3060 GPU，28 SMs @ 1.3GHz，每 SM 2 个 scheduler，32768 registers，32KB shared memory，128KB L1D，12GB DDR4。
  - **GPU 模拟器（ACS-HW）**：Accel-Sim 模拟器，配置为 RTX 3070 参数：46 SMs @ 1.4GHz，每 SM 4 个 scheduler，32768 registers，32KB shared memory，128KB L1D，16GB DDR4。功耗建模使用 AccelWattch。调度窗口大小=32。

- 评估性能的软件/脚本是什么。修改了什么。
  - Deep RL 物理仿真：Brax 框架（JAX 实现），5 个 MuJoCo 环境（Ant, Grasp, Humanoid, Cheetah, Walker2d）。
  - 动态 DNN：InstaNAS-A（CIFAR10）、Dynamic Routing Dynamic-A 16-layer（Cityscapes）、Conditional Convolution 4-experts + EfficientNet-B4 backbone，PyTorch 实现，batch size=1。
  - 静态 DNN（NAS 优化 CNN）：NASNet、AmoebaNet、SqueezeNet、RandomWire，PyTorch 实现，batch size=1，CIFAR10。
  修改：通过 ACS_wrapper 为每个 kernel 定义 `__read_segments__` 和 `__write_segments__`（起始地址+大小列表），以及 `get_addresses()` 函数在 kernel launch 前解析虚拟地址。ACS-SW 运行时系统由 window module 线程和 scheduler module 线程实现。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  论文声明将提供 ACS-SW 开源实现（"We will provide an open-source software-only implementation of ACS"），但当前未找到公开开源链接（2026年5月检索 arXiv abs/2401.12377 及相关页面均未列出代码仓库 URL）。论文使用的评估框架均为已有开源项目（Brax、PyTorch、Accel-Sim）。

  评估原理：ACS 运行时系统在 CPU 端维护一个输入 FIFO 队列和调度窗口。应用线程调用 kernel 时，先通过 `get_addresses()` 解析 read/write segments（起始虚拟地址+大小），kernel 及其 segments 元数据进入输入 FIFO。Window module 线程将 kernel 插入调度窗口时，比较新 kernel 的 write segments 与窗口内所有 kernel 的 read+write segments 是否重叠，若有重叠则标记为 upstream kernel。Scheduler module 线程（可配置数量，每个对应一个 CUDA stream）轮询调度窗口，找到 upstream list 为空的 ready kernel，将其发射到自己的 CUDA stream 中，然后调用 cudaStreamSynchronize 等待完成。完成后通知 window module 更新所有 kernel 的 upstream list。

  ACS-HW 流程：软件端维护输入 FIFO 和 scheduled_list（允许 stale），硬件端（GPU 命令处理器内的调度窗口 SRAM）管理 kernel 依赖和状态（ready/pending/executing）。CPU 将 kernel 发送到 CUDA stream → GPU 硬件 upstream load module 修正 stale upstream list → 调度窗口跟踪各 kernel 的 upstream kernel ID（每个 slot 8-bit ID × (N-1) 个）→ kernel 完成时硬件自动更新所有 slot 的 upstream list → ready kernel 被硬件 dispatch 到 GPU 的 kernel dispatch unit 执行。

  硬件面积开销：调度窗口 N=32 时需要约 1KB SRAM；调度窗口 N=64 时插入延迟约 64 cycles（约 50-100ns）。依赖检查延迟约 410ns（窗口 16，6 个 RW-segment）到 1640ns（窗口 32，10 个 RW-segment）。

## A Framework for Fine-Grained Synchronization of Dependent GPU Kernels

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  cuSync 是一个 header-only 的 CUDA 库，实现四个核心机制来实现依赖 GPU kernel 的细粒度同步：(i) 将依赖 kernel 分别发射到不同 CUDA stream 上消除 stream synchronization；(ii) wait-kernel 机制确保 producer kernel 的 thread block 先于 consumer kernel 被调度到 SM 上；(iii) 通过 atomic global counter 实现自定义 tile 处理顺序（如 RowMajor）以最小化 consumer 的等待时间；(iv) 使用 global memory semaphore 数组 + memory fence 实现 post/wait 的 tile 级依赖同步。
  实验比较：cuSync 的多种同步策略（TileSync、RowSync、StridedSync、Conv2DTileSync 及带 W/R/T 优化的变体）vs. CUDA Stream Synchronization（StreamSync）和 Stream-K。评估指标是各 ML 模型端到端推理延迟的减少百分比。

- 后端平台是什么，配置是什么。
  NVIDIA DGX-2 系统，含 8 块 NVIDIA Tesla V100 32GB GPU，通过 NVLINK 互联。CPU 为 2.60GHz 12-core Intel Xeon E5-2690，448GB RAM。CUDA 12.2。

- 评估性能的软件/脚本是什么。修改了什么。
  基于 NVIDIA CUTLASS 3.1 的 GeMM 和 Conv2D CUDA kernel。修改包括：在 GeMM kernel 中添加 cuSync 的 stage.start()、stage.tile()、stage.wait()、stage.post() 调用（约 25 行，占 CUTLASS GeMM 代码的 0.5%）；在 Conv2D kernel 中添加类似调用（约 22 行，占 0.6%）；自研 Softmax-Dropout 融合 kernel 中添加同步调用（约 5 行，占 1%）。评估脚本位于 `src/ml-bench/volta_transformer/eval_llm.py`（LLM）和 `src/ml-bench/volta_conv2d/eval_conv.py`（CV），运行 20 次取平均，warmup 5 次。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  开源地址：https://github.com/microsoft/cusync，CGO 2024 Artifact Evaluation 分支 `cgo-24-ae`。

  评估原理：cuSync 在每个 tile-based kernel 内部插入同步点。Producer kernel 计算完一个 tile 后调用 `stage.post(row, col)`，通过 `__threadfence_system()` 保证 global memory 写入可见后，对 policy 指定的 semaphore 做 `atomicAdd`。Consumer kernel 加载 tile 前调用 `stage.wait(tile, grid)`，第一个线程在 global memory semaphore 上 busy-wait 直到值达到预期，其余线程被 `__syncthreads` 阻塞。这样 consumer 的 tile 只需等待依赖的 producer tile 完成，而非等待整个 producer kernel。

  执行流程：
  1. 用户用 cuSyncGen DSL 描述 kernel 间的 tile 依赖关系（如 MLP 中第二个 GeMM 的每个 consumer tile 依赖第一个 GeMM 同行所有列 tile）
  2. cuSyncGen 生成 policy 类（sem/value 方法）和 tile 处理顺序函数
  3. 用户修改 CUDA kernel，在加载前调用 wait，计算后调用 post，使用 CuStage 实例化 policy
  4. 主函数创建 CuStage 对象，声明依赖关系（CuSync::dependency），在独立 stream 上发射 kernel
  5. consumer stream 先发射 wait-kernel（单线程 busy-wait 等待 producer 开始），然后发射 consumer kernel
  6. 测量 kernel 执行时间，比较不同 policy（TileSync/RowSync/StridedSync）和优化组合（+W 去 wait-kernel, +R 重排 tile load, +T 去自定义 tile order）的性能

## AccelOpt: A Self-Improving LLM Agentic System for AI Accelerator Kernel Optimization

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  AccelOpt 使用 LLM agentic workflow（Planner-Executor-Summarizer）+ beam search + optimization memory，在 Trainium 加速器上自动生成和优化 NKI kernel。优化技术包括：loop invariant code motion（循环不变量外提）、loop fusion（循环融合）、tile size 增大（256→512 以利用硬件 optimal throughput 配置 128×128 stationary + 128×512 moving）、algebraic simplification（如 θ − γλθ → (1 − γλ)θ）、intrinsic fusion（如 reciprocal(sqrt) → rsqrt，x/(1+e^(-x)) → x·sigmoid(x)）、memory spilling 消除（通过 recomputation 减少 off-chip memory 访问）等。
  实验比较：(1) AccelOpt vs Claude Sonnet 4 重复采样（peak throughput percentage），(2) beam search vs 重复采样，(3) beam search + optimization memory vs beam search only，(4) 不同 executor 模型（Qwen3-Coder-30B、gpt-oss-120b、Qwen3-Coder-480B）及 model ensemble，(5) 不同 memory 配置（TopK, ExpN）的 cost-benefit trade-off，(6) Reflexion-style baseline，(7) AccelOpt vs human experts（Mamba 和 RoPE kernel），(8) AccelOpt 在 H100 GPU Triton kernel 上的泛化实验。

- 后端平台是什么，配置是什么。
  Amazon Trainium 1 (trn1.32xlarge) 和 Trainium 2 (trn2.48xlarge) EC2 实例。Trainium 芯片包含 Tensor Engine、Vector Engine、Scalar Engine（三者并发运行），通过 kernel-managed on-chip memory（SBUF 和 PSUM）与 HBM 通信。NKI（Neuron Kernel Interface）是 Python-embedded kernel 编程语言。GPU 泛化实验使用 NVIDIA H100 GPU。

- 评估性能的软件/脚本是什么。修改了什么。
  自建 NKIBench benchmark suite（14 个 NKI kernel，来自 Qwen3、DeepSeek-V2.5/V3/MoE、Falcon-40B 等真实 LLM workload），分布式 profiling service 基于 Neuron Profile 工具。AccelOpt 系统实现了 beam search 算法 + optimization memory curation（Algorithm 1 & 2）。评估使用 Roofline 模型计算 peak throughput percentage: T = max(Traffic_Min/Bandwidth, FLOPs_MM/Peak_MM, FLOPs_Vec/Peak_Vec)，性能指标为 T/t（百分比）。GPU 泛化验证使用 FlashInfer-Bench 的 24 个 Triton kernel。Agent 请求通过 vLLM 服务 open-source 模型，使用 Logfire 记录 LLM query 信息。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  开源地址: https://github.com/zhang677/AccelOpt
  评估原理:
  1. NKIBench 提供 14 个 baseline NKI kernel 及其对应 ML operator 问题描述（Matmul、BatchMatmul+Softmax、Group Query Attention、Mamba block、LoRA、RoPE、SiLU、SwiGLU、AdamW 等），每个 kernel 关联 config（shape）、profiling 数据和 peak throughput 计算。
  2. AccelOpt 每轮迭代: Planner 为每个 candidate kernel 生成 N 个优化计划 → Executor 每个计划实现 K 次，生成 B×N×K 个新 kernel → 分布式 profiling service 在 Trainium 硬件上运行所有 kernel，收集 latency、HBM read/write bytes、tensor/vector/scalar engine utilization、spill bytes 等指标 → Summarizer 从超过 speedup 阈值的 slow-fast kernel pairs 中提炼 experience items → 更新 optimization memory → Beam search candidate selection function β 选择 Top-B kernels 进入下一轮。
  3. 最终评估以 peak throughput percentage（= 理论最优时延 T / 实测时延 t）衡量 kernel 质量，T 基于 Roofline 模型取 memory bandwidth bound、tensor engine bound、vector engine bound 三者的最大值。Traffic_Min 为所有输入+输出 tensor 的 byte 总量。
  4. 从 baseline kernel 输入到优化后 kernel 性能输出全过程: NKI kernel 源码 → Planner 分析 profile 瓶颈（如低 HFU、高 spill、高 memory write）→ Executor 实现 loop transformation / tiling / memory layout 优化 → Neuron Compiler 编译 → Trainium hardware 执行 → Neuron Profile 采集性能数据 → Summarizer 提炼通用优化策略 → optimization memory 积累经验 → 下一轮迭代。共运行 T=16 轮，最终输出最优 kernel 和 peak throughput percentage。

## Efficient and Adaptable Overlapping for Computation and Communication via Signaling and Reordering

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  FlashOverlap 属于 kernel 调度/运行时计算层优化。实现包括：(i) 在 CUTLASS 模板 GEMM kernel 的 epilogue 中融合 signaling 机制——通过原子计数器追踪 tile 完成进度，当 wave group 内所有 tile 完成后发送信号触发通信；(ii) wave-based signaling timing——利用 GEMM 执行的 wave pattern（多个 tile 几乎同时完成），以 wave 而非单个 tile 为重叠单位提升带宽利用率；(iii) 可调 wave grouping——将多个 wave 组合为 group，在重叠机会和通信分段之间权衡；(iv) pre/post-communication reordering——pre-communication reordering 将非连续地址的 tile 重排为连续地址以直接调用 NCCL API，post-communication reordering 在通信后恢复数据顺序；(v) predictive search tuning——通过剪枝+延迟预测器实时搜索最优 wave group partition。
  实验比较：(i) 算子级：FlashOverlap vs non-overlap baseline、decomposition-based baseline（Async-TP、VanillaDecomposition）、fusion-based baseline（FLUX），覆盖 GEMM+AR、GEMM+RS、GEMM+A2A 三种通信原语，每种在 50+ GEMM sizes 下测试；(ii) 端到端：LLM 推理（Llama3-70B TP=8 on vLLM）、LLM 训练（Mixtral-8x7B EP=4/TP=2、Llama3-70B TP=8 on Megatron-LM）、T2V 生成（Step-Video-T2V TP=4 on xDiT）；(iii) 消融实验：固定大小 grouping vs 等大小 grouping vs FlashOverlap 搜索；(iv) 预测搜索准确性 vs 穷举搜索；(v) 华为 Ascend 910B NPU 跨平台验证。

- 后端平台是什么，配置是什么。
  - NVIDIA A800 GPU（NVLink 互联，pairwise 连接，1935GB/s HBM 带宽，312 TFLOPS FP16）——主要用于端到端评估
  - NVIDIA RTX 4090 GPU（PCIe 互联，跨 NUMA 节点，1008GB/s HBM 带宽，330 TFLOPS FP16）——用于算子级评估
  - 华为 Ascend 910B NPU——用于跨平台适配验证
  - 软件环境：CUDA 12.1、NCCL 2.19.3、PyTorch 2.5.1、CUTLASS 3.6.0
  - A800 服务器 pairwise NVLink；RTX 4090 服务器 PCIe 4.0 穿越 NUMA

- 评估性能的软件/脚本是什么。修改了什么。
  基于 CUTLASS 3.6.0 的模板 GEMM 实现。修改包括：(i) GEMM kernel epilogue 中融合 pre-communication reordering——将 tile 输出按执行顺序重排为连续地址（tile/subtile/subtoken 级粒度）；(ii) 添加 counting table（原子计数器，size=P 对应 P 个 group）追踪 tile 完成；(iii) signaling kernel——独立于 GEMM 在另一 CUDA stream 上运行，周期性查询 counting table，达到 group size 后触发 NCCL 通信；(iv) post-communication reordering 融合到后续 element-wise kernel（如 RMSNorm）中恢复数据顺序；(v) CUDA stream 管理——GEMM 在 stream A，signaling + 通信在 stream B，实现并发执行。
  评估脚本：Artifact repo 提供 e1_correctness.py（正确性验证）、e1_speedup.py（加速比测量）、e1_compare.py（与 SOTA 对比）、e2_predictive_search.py（预测搜索准确性）、e3_rmsnorm_overhead.py/e3_gemm_overhead.py（开销测量）。端到端评估通过替换 vLLM/Megatron-LM/xDiT 中的原始 linear layer + 通信为 FlashOverlap 实现完成。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  已开源：https://github.com/infinigence/FlashOverlap（ae 分支），Zenodo DOI: 10.5281/zenodo.17201530。

  评估原理与流程（以 GEMM+AllReduce on 4×RTX 4090 为例）：
  1. **Preparation**：运行 evaluation/preparation.py → profiler 对目标 GEMM size (M×N×K) 和 GPU 执行 CUTLASS profiler，获取最优 GEMM 配置（tile size、swizzling pattern、wave 数、duration）；通过多轮通信采样构建 (data_size, bandwidth) 曲线；确定通信原语占用的 SM 数 → 更新 wave 数 T。
  2. **Online Tuning**：对 GEMM size 生成 wave group partition 候选（剪枝后设计空间），对每个 candidate 运行 Alg.1 的 latency predictor：遍历每个 group G_i → 计算 computation latency t_p = GEMM_duration/T × |G_i| → 根据 data_size 插值 bandwidth curve 得到 communication latency t_m → 累加 t_m^acc = max(t_p^acc, t_m^acc) + t_m → 选择最小 t_m^acc 的 partition。
  3. **Execution**：创建两个 CUDA stream。Stream A 发射 GEMM kernel（含 fused pre-communication reordering 在 epilogue 中）。GEMM kernel 每个 tile 完成时通过 atomicAdd 更新 counting table。Stream B 发射 signaling kernel——以 busy-wait 周期性读取 counting table，当 group G_j 计数达到 |G_j| 时，调用 NCCL API（如 ncclAllReduce）对已重排的连续数据 buffer 执行通信。通信完成后，post-communication reordering kernel 根据 mapping table 将数据恢复为原始顺序。
  4. **Measurement**：CUDA event 记录 GEMM launch 到通信完成的 total latency。Speedup = non_overlap_latency / overlap_latency。Non-overlap baseline 为顺序执行 cuBLAS GEMM + NCCL 通信。
  5. **Post-communication reordering 硬件流**：对于 AllReduce，tile 在 wave 内任意顺序均可（只需所有 GPU 保持一致）；对于 ReduceScatter，tile 按 row 拆分为 subtile（每个 GPU 对应一个 subtile），通信后通过 local row exchange 纠正顺序；对于 All-to-All，tile 按 token(row) 拆分为 subtoken，各 destination GPU 有独立 memory pool，subtoken 在 pool 内按执行顺序重排。
  6. **Output**：terminal 输出 speedup table（最多 1.65× on RTX 4090, 1.30× on A800）。

## FastTree Optimizing Attention Kernel and Runtime for Tree-Structured LLM Inference

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  FastTree 的 GPU attention kernel 和 tree structure-adaptive runtime 属于 kernel 调度/运行时计算优化。实现包括：(i) tree-structured attention kernel——将按 radix tree 共享 KV cache 的 queries 聚合为 context-queries groups，在单个 kernel 中并行处理不同 group，每个 group 内部以 Flash-Attn 风格 tile-by-tile 执行 attention，Q tile 跨 block 并行化、KV tile 在 block 内串行迭代，利用 online softmax 存储中间结果于 shared memory；(ii) greedy heuristic runtime——以 BFS 遍历 radix tree，对每条边比较 SplitKVCost（分离）和 SplitQCost（拼接）的估计开销（padding cost + intermediate result cost），贪心选择开销更小的边赋值，生成 virtual tree 后做 node-centric query aggregation；(iii) multi-phase tiling——靠近 root 的节点聚合 query 多，用大 tile size 最大化 KV 复用；靠近 leaf 的节点 query 少，用小 tile size 避免 shared memory 浪费、提升 SM occupancy；(iv) GPU-efficient long context splitting——当 group 级并行度不足或存在极长 tail context 时，split 超阈值长度的 context 提升 GPU occupancy，虽引入 intermediate result reduction overhead 但被 SM utilization 提升所抵消；(v) 最后 launch 一个 lightweight reduce kernel，利用 LogSumExp vectors 做跨 group 的中间结果 rescaling。
  实验比较：(i) kernel benchmark——FastTree vs FlashAttention v2.6.3、FlashInfer v0.1.6、SGLang Triton kernels、DeFT、Multi-Level Cascade Attention（CascadeAttn），测量 attention kernel execution time，覆盖多种 tree shape（N=node number per level, C=context length per level）和 GQA ratio（1/4/16），head dim=128；(ii) end-to-end——FastTree+SGLang vs SGLang-Triton vs SGLang-FlashInfer，在 Llama-2-7B 和 Mistral-7B 上测量四个 benchmark（multi-level system prompt、multiple few-shot learning、multi-chain reasoning、multi-document QA）的 throughput；(iii) 消融——无 greedy heuristic 的 naive aggregation vs full FastTree、long context splitting 的单独影响。

- 后端平台是什么，配置是什么。
  NVIDIA H100 GPU (80GB)，CUDA 12.2。论文当前实现未使用 Hopper 特有特性（如 TMA）。Triton 3.0.0。FlashInfer v0.1.6。SGLang v0.2.13。

- 评估性能的软件/脚本是什么。修改了什么。
  - Kernel benchmark：自编 Python 脚本，用 N（每层 node 数）和 C（每层 per-node context length）两个数组生成不同 tree shape 的 KV cache，如 N=[1,2,4], C=[128,32,32] 表示 3 层共 7 个 node 的 tree。测量各配置下 attention kernel 执行时间（ms），结果以 normalized speedup 展示。
  - End-to-end：在 SGLang v0.2.13 上集成 FastTree，替换原有 attention backend（Triton/FlashInfer），测量 batch=128、gen=256 tokens 的 throughput（tokens/s），breakdown 分析 decoding latency 和 GPU kernel execution time。
  修改：(i) 实现 FastTree attention kernel（Triton），支持 Flash-Attn 风格的 tiled attention + query aggregation；(ii) 实现 tree structure-adaptive runtime（Python/C++），包括 BFS greedy heuristic 和 long context splitting；(iii) 集成到 SGLang 的 attention 路径，decode 阶段使用 FastTree kernel，prefill 阶段沿用 FlashInfer。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  已开源：https://github.com/PanZaifeng/FastTree-Artifact（Apache-2.0）。提供 Docker 环境，含 CUDA 12.2、PyTorch 2.4.0、Triton 3.0.0、FlashAttention 2.6.3、FlashInfer 0.1.6、SGLang 0.2.13。运行 `kernel_bench/run.sh` 可一键复现 kernel benchmark 结果，生成 `kernel_bench/norm_perf.pdf`。

  评估原理与流程（以 N=[1,2,4], C=[128,32,32], GQA=1 的 kernel benchmark 为例）：
  1. **输入**：Tree shape (N, C) 定义 radix tree 结构——N=[1,2,4] 表示 3 层分别有 1、2、4 个 node；C=[128,32,32] 表示各层 context length。总 node 数=1+2+4=7。每个 leaf node 关联若干 queries（模拟不同请求），leaf 间共享路径上的祖先节点 KV cache。
  2. **Runtime 处理**：FastTree runtime 读取 radix tree → BFS 遍历各节点 → 对每条 parent→child 边贪心决策：
     a. SplitKVCost (edge=0, 分离): 计算 parent node v 的 padding cost C_P(nQ_curr, len_v) + child node l 的 padding cost C_P(nQ_l, len_l) + intermediate result cost γ·nQ_l·d
     b. SplitQCost (edge=1, 拼接): 计算分离后 v 的 padding cost C_P(nQ_curr - nQ_l, len_v) + 拼接后 l 的 padding cost C_P(nQ_l, len_v + len_l)
     c. 比较两 cost，选更小的赋值。若拼接则更新 nQ_curr -= nQ_l 和 L[l] += len_v。
  3. **Virtual tree 生成**：根据边赋值生成 virtual tree。拼接边对应的 node 被"复制"后拼接到不同 leaf。最终通过 node-centric query aggregation 生成 (context, {queries}) groups list。
  4. **Attention kernel 执行**：单 kernel launch 并行处理所有 groups。每个 group 分配到一个 block set：
     a. Tile division：Q 矩阵按 tile size 沿 query dim 划分 → 不同 block
     b. For each block：copy Q tile + 首 KV tile from HBM → shared memory
     c. Loop over KV tiles：每次 load 下一个 KV tile → BMM1(QK^T) on tensor core → online softmax (max update + exp + rowsum) → BMM2(PV) on tensor core → 更新 partial O + L (in shared memory)
     d. Write partial results (O, L) to HBM
  5. **Reduction kernel**：对每个 query，跨 groups 的 partial O_i 用 L_i rescale 后累加得到 final attention output。
  6. **Performance measurement**：CUDA event 记录 kernel launch 到 completion 的 wall time。Compare FastTree vs baselines（FlashAttention/FlashInfer/SGLang Triton/DeFT/CascadeAttn）。FlashAttention 因 decode 阶段 GEMV 仅 <1% effective computation；FastTree 通过 query aggregation 将 GEMV 转 GEMM，tensor core utilization 大幅提升，且共享 KV tile 仅在 shared memory 加载一次（非 HBM 重复加载）。
  7. **输出**：Normalized speedup plot（Figure 9），FastTree 平均 5.1× over FlashAttention, 4.2× over FlashInfer, 10.6× over DeFT。

  Long context splitting 评估原理：在 N=[1,10], C=[4000,400] 等 GPU 欠饱和配置下，开启 splitting 可获 up to 1.9× speedup。Splitting 将超长 context node 沿 context dim 切分，增加 group 级并行度以填满 SM，虽引入中间结果 reduction 开销但被 occupancy 改善抵消。

## UltraAttn: Efficiently Parallelizing Attention through Hierarchical Context-Tiling

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  UltraAttn 在 kernel 调度/运行时计算层面的核心实现包括：(i) **Device-Level Context-Tiling**——将 attention workload 沿 Q 和 KV 两个维度划分为 $P \times P$ grid，通过 ILP 求解器（Gurobi）在计算负载均衡约束（$DLI_{P,CP} \le \theta_{DLI}$）下最小化每个 GPU 的通信量（MCV），输出每个 block $B_{r,c}$ 到设备 $U_g$ 的分配方案 $x_{r,c,g}$，同时推导出 Q/KV/O 的入站/出站流量（变量 A/B/C/D），生成 peer-to-peer 通信计划；(ii) **Node-Level Context-Tiling**——将跨节点通信异构性问题解耦，先在节点级别做 context-tiling（将每个节点视为集成设备，节点间使用 groupwise peer-to-peer 充分利用所有 NIC 带宽），再在设备级别做 context-tiling（节点内 NVLink peer-to-peer），两层方法类比统一；(iii) **Kernel-Level Context-Tiling**——在 node/device-level tiling 生成的 parallel dependency graph（DAG）上通过三种 substitution（computation kernel batching、peer-to-peer communication batching、collective communication batching）进行图变换，使用贪心策略按 transformation gain 降序逐个应用变换，通过 ILP runtime 评估执行时间决定保留与否，自适应选择最优 kernel granularity 以平衡 kernel overlap 和单 kernel device utilization；(iv) **ILP-based Runtime**——将 parallel dependency graph 的 kernel 调度形式化为 ILP：每个 kernel v 的 start time $S_v$（实变量）+ duration $D_v$（profiling 获取），同一 CUDA stream 内的 kernel 通过 $Order_{uv}$ 布尔变量控制互斥执行（Stream Exclusivity Constraints），kernel 依赖通过 DAG 边约束 $S_u + D_u \le S_v$ 保证，目标最小化 $End\_Time$。求解后按 $S_v$ 排序得到每 CUDA stream 的最优 kernel 执行顺序。Context remap 技术（$\phi: T \to CR$ 映射）作为离线预处理步骤，用于增强 workload locality（如 strided attention 用 $\phi(t_i) = \lfloor i \cdot 16/S \rfloor \mod 4$），仅在 attention 模块内影响性能，不影响其他 LLM 模块。

  实验比较：(1) 端到端训练/推理速度对比（Llama2-7B, S=512K, CP=8/64, Nh=1/32, 64 GPUs）vs ring attention、striped attention、zigzag ring attention baseline，UltraAttn 平均 $2.2\times$（Nh=1）和 $3.4\times$（Nh=32）端到端加速；(2) 分布式 attention 模块加速比——6 种 attention pattern：dense（full/causal）+ block sparse（strided/global+local/star/streaming），CP=2-64, Nh=1/32，平均加速从 $10.2\times$（strided training, Nh=1, CP=64）到 $1.9\times$（streaming inference, Nh=32, CP=8）；(3) 消融实验——逐步叠加 Node Tile → Node+Device Tile → Node+Device+Kernel Tile → +ILP Runtime，每步独立性能增益；(4) 强可扩展性——固定 S=512K，CP 从 16 到 64 的 MFU 变化，对比 baseline；(5) 性能预测准确度——intra-node $R^2=0.9932$，inter-node $R^2=0.9181$；(6) ILP 搜索开销——node/device-level tiling ILP 0.07ms（strided, CP=16）到 3672ms（causal, CP=64），runtime ILP 0.10ms（strided, CP=16）到 1073ms（causal, CP=32）。

- 后端平台是什么，配置是什么。
  8 节点集群，每节点 8× NVIDIA H100-NVLink-80GB GPU（共 64 GPUs），96 CPU cores，2 CPU sockets。节点内 NVLink 双向带宽 450GB/s。节点间 8× 400Gb/s InfiniBand EDR，每 GPU 与 1 NIC 有 affinity，PCIe-5.0 连接。软件：PyTorch 2.6.0、NCCL 2.21、FlashAttn 2.5.7、Gurobi ILP solver。模型：Llama2-7B，batch_size=1，FP16/BF16。Attention patterns：full attention、causal attention、strided attention、global+local attention、star attention、streaming attention。

- 评估性能的软件/脚本是什么。修改了什么。
  基于 PyTorch + NCCL + FlashAttn 的自研 context parallelism 系统（~10K 行 Python）。直接使用 NCCL C-level API（非 torch.distributed）以获得更精细的通信 kernel 调度。修改/新增内容：
  - **ILP context-tiling 模块**：将 attention workload 建模为 ILP（9 个变量类型，10 个约束组），通过 Gurobi 求解器输出最优 workload-to-device 分配。变量包括 $x_{r,c,g}$（block 分配）、$H_{g,r}$（是否需要 $Q_r$）、$V_{g,c}$（是否需要 $KV_c$）、A/B/C/D（四种流量方向）、$Cin_g/Cout_g$（总通信量）、MCV（最大通信量）。10 个约束组：Allocate Uniqueness、Definition of H/V/A/B/C/D、Inbound/Outbound Traffic、Computation Balance、Minimization Objective
  - **Context remap 预处理**：对某些 pattern（如 strided attention）手动定义 $\phi$ 映射增强 locality。$\phi(t_i) = \lfloor i \cdot CP/S \rfloor$ 为默认序贯映射，$\phi(t_i) = \lfloor i \cdot 16/S \rfloor \mod 4$ 用于 strided attention 增强 node-level locality
  - **Parallel dependency graph 构建**：从 tiling 结果生成 DAG，节点为 computation kernel（矩形）、receive kernel（椭圆）、send kernel（菱形），边表示依赖
  - **Greedy kernel-level tiling**：在图变换候选集（三种 substitution）中按 gain（融合后减少的时间）降序排序，贪心选择
  - **ILP runtime**：将 kernel 调度建模为 ILP，stream exclusivity + dependency constraints + end time minimization
  - **CUDA graph 集成**：cudagraph 消除 CPU kernel launch overhead

  评估脚本 workflow：
  1. 编译 NCCL from source（`third_party/comm_test/third_party/nccl`, target sm_90）: `make -j src.build NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90"`
  2. 集群 profiling：`third_party/kernel_profiler/scripts/bench_ops_m2_py.sh` 采集 FlashAttn kernel 性能 → `third_party/comm_test/scripts/wrapper_conflict_bench_hamming.sh 8` 和 `16` 分别采集 intra-node 和 inter-node NCCL 通信性能
  3. 创建 database 并复制 profiling 数据到 `database/m_configs/`
  4. 分布式 attention 评估：`scripts/schedule/task1_BSA_hamming.sh bsa_train`（单节点 8 GPU 生成执行计划 + 评估 intra-node）→ `scripts/schedule/task2_BSA_hamming.sh bsa_train`（8 节点评估 inter-node），结果缓存到 `database/inter_bsa_exe_plans_profile.json`
  5. Baseline：`third_party/UltraAttn_baseline/scripts/runtime/run_exp.sh`，结果手动复制到 database
  6. 端到端：Megatron-LM 上 `scripts/ultraattn_e2e.sh`（Llama2-7B），结果存 `results/UltraAttn_E2E/hamming/`
  7. 绘图：`python plot/da_bsa_training_pick.py` 等生成 Figure 7-11

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  已开源：https://github.com/oliverYoung2001/UltraAttn，Zenodo DOI: 10.5281/zenodo.15301789。Artifact 提供 Makefile 一键复现：`make spack_packages` → `make prepare_conda_env` → `make compile` → `make cluster_profile` → `make figure7`（约 6h）/ `make figure8`（约 9h）/ `make figure9`（约 10min）/ `make figure6`（约 1h）/ `make figure10` / `make figure11`。编译 NCCL 需 GCC，建议使用 Spack 管理依赖。

  评估原理与流程（以 strided attention training, CP=64, S=512K, Nh=1, 64 GPU 为例）：

  1. **Cluster Profiling（前置步骤）**：
     - Attention kernel profiler：在不同 (M, N, K) shape 组合下测量 FlashAttn kernel 执行时间 → 构建 kernel 性能 lookup table（$D_v$ duration 的来源）
     - Communication profiler：测量 intra-node NVLink 和 inter-node InfiniBand peer-to-peer 在不同 message size 下的带宽/延迟 → 构建通信性能 lookup table
     - 这些数据构成 ILP 性能预测模型的底层输入

  2. **Adaptive Workload Partition**：
     - 输入：strided attention pattern（Figure 2(c) 的 diagonal stripes），Q/KV context length = 512K
     - 计算 partition degree P：从公式 $DLI_{P,CP} = \lceil COMP/CP \rceil / (COMP/CP) - 1$ 递增 P 直到 DLI 低于阈值 $\theta_{DLI}$
     - 将 attention workload 划分为 $P \times P$ grid，每 cell 标注 Full/Causal/Empty（strided pattern 仅 diagonal band 内 cell 非空）

  3. **Node-Level Context-Tiling ILP**（CP > 8 时启用）：
     - 输入：$P \times P$ grid block table，FB/CB/EB 集合，CP=8（8 nodes），profiling 数据
     - ILP 变量：$x_{r,c,g}$（binary, $B_{r,c}$ 是否分配给 node g），$H_{g,r}$（binary, node g 是否需要 $Q_r$），$V_{g,c}$（binary, node g 是否需要 $KV_c$），A/B/C/D（integer, 四种流量），$Cin_g/Cout_g$（integer, 总通信量），MCV（integer, 最大通信量）
     - 约束：Allocate Uniqueness（$\sum_g x_{r,c,g}=1$）+ H/V definition（$H_{g,r} \ge x_{r,c,g}$）+ A/B/C/D definition（含 Cmap 映射）+ Inbound/Outbound Traffic（$Cin_g = A_g \times 1 + B_g \times 2 + C_g \times 1$，系数为 Q/KV/O 的 per-token 数据量比）+ Computation Balance（$\sum_{FB} x_{r,c,g} \times 1 + \sum_{CB} x_{r,c,g} \times 0.5 \le \tau$，$\tau = \lceil (|FB| \times 1 + |CB| \times 0.5) / CP \rceil$）
     - 目标：minimize MCV（$\ge \max\{Cin_g, Cout_g\}; \forall g$）
     - 输出：每个 node 分配的 workload blocks + node 间 groupwise peer-to-peer 通信计划（需 Q/KV/O 的 source→destination）

  4. **Device-Level Context-Tiling ILP**：
     - 在每个 node 内（8 GPU），对分配的 block subset 再次求解相同 formulation 的 ILP（CP=8, peer-to-peer 替代 groupwise peer-to-peer）
     - 输出：每个 GPU 分配的 workload blocks + GPU 间 peer-to-peer 通信计划
     - ILP 时间随 attention pattern 密度和 P 增长：strided P=2 仅 0.07ms，causal P=8 达 3672ms

  5. **Parallel Dependency Graph Construction**：
     - 从 device-level tiling 结果生成 DAG：每个 GPU 的 computation kernel（FlashAttn 对分配的 blocks 执行 attention）+ receive kernel（NCCL recv: 接收来自其他 GPU 的 Q/KV）+ send kernel（NCCL send: 发送本地 Q/KV 到其他 GPU）
     - 依赖关系：receive → computation → send（数据流依赖），以及跨 GPU 的通信依赖
     - 示例（Figure 5b, GPU1）：A0 → Q3(send to GPU3) → KV3(recv from GPU3) → A1 → A2

  6. **Greedy Kernel-Level Tiling**：
     - 生成三种 substitution 的 transformation candidates：
       a. Computation kernel batching：相邻 computation kernel 合并为大 kernel（受 FlashAttn backend 的 attention shape 支持范围限制）
       b. Peer-to-peer communication batching：同一 (src, dst) 对的多个 send/recv 合并
       c. Collective communication batching：多个 peer-to-peer 合并为 collective（如 all-to-all）
     - 按 transformation gain（融合后减少的时间）降序排序
     - 贪心遍历：检查 candidate 的 kernel 是否未被之前变换修改 → 应用到 DAG → 通过 ILP runtime 评估执行时间 → 若改善则保留

  7. **ILP Runtime Kernel Scheduling**：
     - 将 DAG kernel 按共享带宽分组：同一输出带宽的 send kernel → 同一 CUDA stream；同一输入带宽的 recv kernel → 同一 stream
     - 在每 stream 内求解 ILP：
       - 变量：$S_v$（start time），$Order_{uv}$（boolean, u 是否在 v 之前），$End\_Time$
       - Stream Exclusivity：$(S_u + D_u \le S_v + (1 - Order_{uv})Ub) \land (S_v + D_v \le S_u + Order_{uv}Ub)$
       - Dependency：$S_u + D_u \le S_v$ for $\forall (u,v) \in E$
       - 目标：minimize $End\_Time$
     - 求解后按 $S_v$ 排序得最优 kernel 执行顺序 → 转换为 CUDA stream graph
     - ILP 时间：从 0.10ms（strided, CP=16）到 1073ms（causal, CP=32）

  8. **CUDA Graph 执行**：
     - 将 CUDA stream graph 编译为 CUDA graph
     - GPU 执行：各 CUDA stream 并行执行，stream 内按 ILP 顺序串行
     - 实际时间线：computation kernel（FlashAttn forward）与 communication kernel（NCCL send/recv）交错，最大化 computation-communication overlap

  9. **Performance Measurement**：
     - CUDA event timing 测量 distributed attention 模块 wall-clock time
     - Speedup = baseline_time / UltraAttn_time
     - MFU = achieved TFLOPS / peak TFLOPS（H100 FP16: 989 TFLOPS × 64 = 63,296 TFLOPS 理论峰值）
     - 性能预测准确度：对比 predicted time vs actual time，计算 $R^2$ 和 relative error
     - Results：intra-node 仅 3.0% cases 超过 30% relative error，inter-node 约 5.8% 超过 50%

## Demystifying the Placement Policies of the NVIDIA GPU Thread Block Scheduler for Concurrent Kernels

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  本论文是一个测量/表征研究（measurement/characterization study），而非提出新系统。实现分为两部分：(i) 通过设计特定的两 kernel 实验 workload（Kernel X + Kernel Y），利用 `smid` 寄存器和 `globaltimer` 计时代码，实证推导 NVIDIA GPU thread block scheduler 在并发 kernel 下的调度策略——"most-room policy"；(ii) 设计四类 purpose-built kernel（L1-cache-dependent、compute-intensive、memory-intensive、PCIe-transfer-bandwidth-dependent），测量 most-room policy 对并发 kernel 性能的影响。
  实验比较了三种场景：(i) Serial（串行，baseline——每个 kernel 独立运行，无并发）；(ii) Concurrent-Isolated（并发但 Kernel B 的所有 block 被分配到与 Kernel A 不同的 SM 上）；(iii) Concurrent-Colocated（并发且 Kernel A 和 Kernel B 的 block 被分配到同一个 SM 上）。指标为各 kernel 的 execution time（ms），通过 nvprof 测量。

- 后端平台是什么，配置是什么。
  - **Pascal**: GeForce GTX 1080, Compute Capability 6.0, 5 SMs, 2048 threads/SM, 1024 max threads/block, 32 max blocks/SM, 64 max warps/SM
  - **Volta**: Tesla V100, Compute Capability 7.0, 80 SMs, 2048 threads/SM, 1024 max threads/block, 32 max blocks/SM, 64 max warps/SM
  - **Turing**: GeForce RTX 2080 Ti, Compute Capability 7.5, 68 SMs, 1024 threads/SM, 1024 max threads/block, 16 max blocks/SM, 32 max warps/SM

- 评估性能的软件/脚本是什么。修改了什么。
  论文使用自编 CUDA kernel 而非标准 benchmark（如 Rodinia），以精确控制调度结果和特定资源的竞争。使用 NVIDIA nvprof 作为性能 profiling 工具测量 kernel execution time。
  - **L1-cache-dependent kernel**：每个线程反复访问 texture memory，利用各 GPU 的 L1 cache 大小和 set-associativity 信息使访问高 cacheable 但易被替换。测量到 serial case 下 L1 cache hit rate 平均 90%（75%-95%）。
  - **compute-intensive kernel**：反复执行浮点运算占用 functional units，避免 global memory 访问以防止内存竞争影响结果。
  - **memory-intensive kernel**：反复写入 global memory 中的大数组（使用写操作防止 L1/texture cache 缓存），线程间访问地址间隔拉开以避免 coalescing。
  - **transfer-bandwidth-dependent kernel**：利用 UVM（Unified Virtual Memory）触发大量 page fault，通过 PCIe 异步传输数据。线程内 block 的地址靠近以允许 coalescing，block 间地址远离以减少 global memory contention 对 PCIe 竞争的影响。
  - 调度策略推导 kernel：使用 `globaltimer` 寄存器让每个 block spin 与 SM id 成比例的时间（B0 最短，Bn 最长），精确控制 block 完成顺序，暴露 scheduler 的 placement 决策。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  论文未提供开源链接。这是一篇 2020 年左右的工作（发表于 SIGMETRICS 相关 venue），作者来自 Worcester Polytechnic Institute。论文使用 Google Cloud Platform 运行实验，使用 NVIDIA profiler nvprof。

  评估原理（以 Turing GPU L1-cache-dependent kernel 的 most-room policy 性能影响实验为例）：
  1. **输入设计**：Kernel A 发射 n−1=67 个 block（Turing 有 68 SM），每个 block 512 threads，先发射到第一个 CUDA stream。保证所有 67 block 各自占据一个 SM（SM0-SM66），SM67 空置。
  2. **Kernel B 变体设计**：Kernel B 有两个版本——version1 每 block 33 threads（concurrent-isolated 条件），version2 每 block 32 threads（concurrent-colocated 条件）。两者均 8 blocks，从第二个 CUDA stream 晚于 Kernel A 发射。
  3. **Most-room policy 触发**：33-thread 版本以 threads 为 limiting resource → SM67（空 SM）可容纳最多 Kernel B block → 全部 8 block 均分配到 SM67（concurrent-isolated）。32-thread 版本以 blocks/SM 为 limiting resource → SM67 分配第 1 个 block 后，所有 SM 可容纳的 block 数相同 → 按 tie-breaking 顺序（even-then-odds: 0,2,4,...,66,1,3,...,67）分配剩余 block → 部分 Kernel B block 与 Kernel A block colocate（concurrent-colocated）。
  4. **Kernel 执行**：Concurrent-isolated 时 Kernel A 和 Kernel B 在完全独立的 SM 上执行，各自独占 L1 cache。Concurrent-colocated 时部分 SM 上两个 kernel 的 block 共享 L1 cache，产生 cache contention。
  5. **性能测量**：nvprof 测量各 kernel execution time。每秒测量 30 次取平均，coefficient of variation < 3%。比较 serial（各 kernel 独立运行无并发）vs concurrent-isolated vs concurrent-colocated。
  6. **输出**：Concurrent-isolated 下 Kernel A=85ms, Kernel B=79ms（与 serial 一致）。Concurrent-colocated 下 Kernel A=105ms (1.24X), Kernel B=105ms (1.33X)。Total time: serial=164ms, isolated=85ms (0.52X), colocated=105ms (0.64X)。

  Most-room policy 推导实验流程（Pascal GPU 示例，Figure 2）：
  1. Kernel X: 5 blocks（Pascal 有 5 SM），256 threads/block。通过 `globaltimer` spin 保证 B0 最先完成（SM0），B4 最后完成（SM4）。
  2. Kernel Y: 3 blocks，160 threads/block。发射时机使得 B0 已完成（SM0 空），B1-B4 仍在运行（SM1-SM4 各有一个 Kernel X block）。
  3. Scheduler 决策：SM0（空, 2048 free threads = 12 blocks of Y）> SM1-SM4（1792 free threads = 11 blocks of Y）→ Y0→SM0。Y0 占 160 threads 后 SM0 剩 1888 threads = 11 blocks → 与 SM1-SM4 平票 → tie-breaking SM0 → Y1→SM0。Y1 占后 SM0 剩 1728 threads = 10 blocks < SM1-SM4 的 11 blocks → Y2→SM1。
  4. 如果 scheduler 是 round-robin，Y0→SM0, Y1→SM1, Y2→SM2。实际观察到 Y0,Y1→SM0, Y2→SM1，证明 most-room 而非 round-robin。
  5. Tie-breaking ordering: Pascal = ascending (0,1,2,3,4); Turing = even-then-odds (0,2,4,...,66,1,3,...,67); Volta 也有独立的 fixed ordering。

  论文还通过修改 limiting resource 的变体实验验证了 other limiting factors：将 Kernel Y threads 降至 32（limiting factor 变为 blocks/SM），观察 block distribution；将 Kernel Y threads 升至 33（limiting factor 变为 warps/SM），观察不同的 placement 行为——均与 most-room policy 预测一致。

## FlashAttention-2 Faster Attention with Better Parallelism and Work Partitioning

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  基于FlashAttention v1，重写CUDA kernel实现三方面改进：(i) **算法tweak**——前向不再对output两项都做`diag(ℓ)^{-1}` rescale，改维护"un-scaled" output并在最终一次性rescale；反向只存储logsumexp L = m + log(ℓ)而非同时存m和ℓ，减少non-matmul FLOPs；(ii) **并行化**——除batch和head维度外，增加sequence length维度的并行。前向：外循环（over K/V blocks）embarrassingly parallel，不同thread block处理不同row block，无需通信；反向：不同thread block处理不同column block，用atomic add更新dQ；(iii) **Warp间工作划分**——前向改为split Q across warps（保持K/V所有warp可访问），避免FlashAttention v1的"split-K"方案（split K/V across warps）带来的shared memory通信开销；反向同样避免split-K。实验比较：(a) Forward+backward runtime vs FlashAttention v1、FlashAttention Triton、xformers cutlass实现、PyTorch标准attention，seq length 512-16K，head dim 64/128，causal/non-causal，A100 80GB SXM4；(b) Decoding阶段attention kernel vs PyTorch naive、FasterTransformer，MQA setting，batch size 1-1024；(c) End-to-end GPT训练吞吐（1.3B/2.7B参数，2k/8k context，8×A100）vs 无FlashAttention baseline和FlashAttention v1。

- 后端平台是什么，配置是什么。
  - NVIDIA A100 80GB SXM4 GPU：108 SMs，每SM 192KB on-chip SRAM，HBM带宽1.5-2.0TB/s，SRAM带宽约19TB/s，FP16/BF16 matmul峰值312 TFLOPs/s，non-matmul FP32峰值19.5 TFLOPs/s
  - NVIDIA H100 GPU（仅forward+backward runtime benchmark，未使用TMA和4th-gen Tensor Cores特殊指令）
  - 端到端训练：8×A100 80GB SXM4
  - Benchmark：seq length 512/1k/2k/4k/8k/16k，batch size使总tokens=16k，hidden dim=2048，head dim=64（32 heads）或128（16 heads），FP16/BF16

- 评估性能的软件/脚本是什么。修改了什么。
  - 基于CUTLASS 3.x库手写CUDA kernel实现FlashAttention-2 forward和backward pass
  - 修改：(i) Forward kernel——外循环并行化：对每个row block（Q block i），启动一个thread block独立处理所有KV blocks；thread block内部warp split Q而非K/V，每个warp持有Q的一个slice，K/V由所有warp共享，每个warp计算其Q slice对应的局部S=QK^T，直接乘V得到output slice，无需warp间通信。最终一次性rescale output by `diag(ℓ)^{-1}`并存储L=m+log(ℓ)。causal mask优化：跳过row indices < column indices的blocks（约一半blocks），每行仅需对1个block施加causal mask。(ii) Backward kernel——列block并行化：每个column block j由一个thread block处理，遍历所有row blocks i，在SRAM中重计算S_ij和P_ij，累加dK_j和dV_j，用atomicAdd更新dQ_i。同样避免split-K warp划分。(iii) Decoding kernel——将KV cache加载split到不同thread blocks以saturate HBM bandwidth，写中间结果到HBM后再用separate reduce kernel合并。
  - Benchmark方法：CUDA event timing测kernel wall-clock time。FLOPs计算：forward = 4·seqlen²·head_dim·num_heads（causal mask时÷2），backward = forward FLOPs × 2.5。TFLOPs/s = FLOPs / runtime。
  - 端到端训练FLOPs公式（Megatron-LM）：6·seqlen·num_params + 12·num_layers·hidden_dim·seqlen²

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  已开源：https://github.com/Dao-AILab/flash-attention（BSD许可证）。安装：`pip install flash-attn`。Python接口兼容FlashAttention v1：`from flash_attn import flash_attn_func; output = flash_attn_func(q, k, v, causal=True)`，内部自动使用FlashAttention-2 kernel。底层CUTLASS 3.x实现，build system基于PyTorch extension。

  **Kernel输入到性能输出全过程**（以A100 forward pass, N=8192, d=128, 16 heads, batch=2, causal=False为例）：

  1. **Tensor分配**：Q/K/V tensors `[2, 8192, 16, 128]` FP16/BF16 in HBM。Block sizes: `B_r=128, B_c=128`（tuned for head_dim=128）。划分：`T_r = ceil(8192/128) = 64` row blocks，`T_c = 64` column blocks。Output O `[2, 8192, 16, 128]`，logsumexp L `[2, 8192, 16]`。

  2. **Thread block调度**（forward，parallelism over sequence length）：对每个(head, row_block_i)组合，launch 1个thread block。共batch×heads×T_r = 2×16×64 = 2048个thread blocks。每个thread block独立处理其row block的所有64个KV column blocks，无需与其他thread block通信。2048 >> 108 SMs，occupancy接近满载。

  3. **Thread block内部forward执行**（单个row block i, B_r=128 rows）：
     - 从HBM加载Q_i `[128, 128]` 到SRAM（~32KB FP16）。初始化 `O_i = 0 [128, 128]`, `ℓ_i = 0 [128]`, `m_i = -inf [128]`。
     - **外循环** j=1..64（KV column blocks）：
       a. 从HBM加载K_j `[128, 128]`、V_j `[128, 128]` 到SRAM（~32KB each, 64KB total）。
       b. **Warp内计算**（4 warps per thread block, split Q row-wise, 各warp持有Q_i的32 rows）：
          - Each warp: `S_warp = Q_warp × K_j^T` (32×128, Tensor Core MMA, FP16→FP32 accumulate)
          - `m_new = max(m_old, rowmax(S_warp))` (CUDA core reduction, 128 elements per row × 32 rows)
          - `P_warp = exp(S_warp - m_new)` (MUFU.EX2 instruction)
          - `ℓ_new = exp(m_old - m_new)·ℓ_old + rowsum(P_warp)` (CUDA core)
          - `O_warp = diag(exp(m_old - m_new))·O_old + P_warp × V_j` (Tensor Core MMA, 32×128 × 128×128)
          - **无需warp间通信**：K_j和V_j由所有warp共享，每个warp独立计算其output slice。
       c. SRAM中仅保留最新O_i、ℓ_i、m_i。中间S_warp (32×128×2B=8KB/warp) 和P_warp存于register/SRAM，不写HBM。
     - **循环结束**：`O_i = diag(ℓ_final)^{-1}·O_final` (一次性rescale)，`L_i = m_final + log(ℓ_final)`。
     - 写O_i和L_i到HBM对应位置。

  4. **Backward kernel**（column-parallel）：对每个(head, col_block_j) launch 1个thread block（共batch×heads×T_c=2048 blocks）。Load K_j, V_j to SRAM。Initialize dK_j=0, dV_j=0。内循环i遍历T_r row blocks：load Q_i, O_i, dO_i, L_i, D_i → recompute S_ij, P_ij in SRAM → compute dV_j += P_ij^T·dO_i, dS_ij → dK_j += dS_ij^T·Q_i → atomicAdd dQ_i += dS_ij·K_j to HBM。Write dK_j, dV_j to HBM。

  5. **性能评估**：CUDA event timing测量kernel wall-clock time（ns）。TFLOPs/s = (4×8192²×128×16) / 1e12 / runtime_seconds ≈ 5.5e11 FLOPs / runtime。A100 FP16 matmul peak: 312 TFLOPs/s。FlashAttention-2 forward实测~210 TFLOPs/s（~67% peak），FlashAttention v1实测~105 TFLOPs/s（~34% peak），standard PyTorch attention ~21 TFLOPs/s（~7% peak）。

## FlashAttention-3 Fast and Accurate Attention with Asynchrony and Low-precision

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现基于Hopper H100 GPU的warp-specialized异步attention kernel，使用CUTLASS primitives（WGMMA、TMA、setmaxnreg）构建。核心kernel实现包括：(i) **Producer-consumer warp specialization**：producer warpgroup使用TMA从HBM异步加载Q/K/V tiles到circular SMEM buffer，consumer warpgroup使用WGMMA执行QK^T和PV矩阵乘法；(ii) **setmaxnreg动态寄存器分配**：producer warp释放registers（仅需1 thread for TMA），consumer warp获取更多registers用于WGMMA；(iii) **2-stage GEMM-softmax pipelining**：在consumer warpgroup内通过寄存器缓冲$\mathbf{S}_{\text{next}}$实现跨迭代流水线——WGMMA(QK^T) of iter j+1 与 softmax of iter j 重叠，WGMMA(PV) of iter j 与 softmax of iter j+1 重叠；(iv) **FP8 WGMMA支持**：Q/K以k-major布局存储（contiguous in head dim），V需in-kernel transpose（LDSM + byte_perm + STSM）转为m-major布局，FP32 accumulator通过byte_perm + shfl_sync转换为FP8 operand register layout以conform to FP8 WGMMA要求；(v) **Persistent kernel**：launch与SM数量相等的threadblocks（132 on H100），每个threadblock处理多个tiles，重叠epilogue和prologue以减少tensor core空闲。
  实验比较：(i) Forward speed (TFLOPs/s) vs FlashAttention-2 (CUDA)、FlashAttention-2 in Triton、cuDNN attention、standard PyTorch attention；(ii) Backward speed vs FlashAttention-2、FlashAttention-2 in Triton；(iii) FP8 forward speed vs BF16 baselines；(iv) 消融：warp-specialization vs GEMM-softmax pipelining各自贡献（batch=4, seqlen=8448, nheads=16, hdim=128 forward）；(v) 2-stage vs 3-stage pipelining效果（3-stage理论上更多重叠但register pressure导致更小block size）。

- 后端平台是什么，配置是什么。
  - NVIDIA H100 80GB SXM5 GPU (Hopper architecture, 700W)：132 SMs，80 GiB HBM @ 3.35 TB/s，228 KiB SMEM per SM，boost clock 1830 MHz
  - FP16/BF16 Tensor Core理论峰值：989 TFLOPs/s；FP8 Tensor Core理论峰值：~1978 TFLOPs/s (2× FP16)
  - Special functions throughput：3.9 TFLOPs/s（16 ops/SM/clock × 132 SMs × 1830 MHz）
  - 软件栈：CUDA 12.3, CUTLASS 3.6 (WGMMA and TMA abstractions), cuDNN 9.5.0.50, Triton 3.1, PyTorch 2.5.0

- 评估性能的软件/脚本是什么。修改了什么。
  - 自编CUDA C++ kernel，基于CUTLASS primitives实现：(i) WGMMA for asynchronous tensor core matrix multiplication（SS prefix: first operand from SMEM; RS prefix: first operand from register file）；(ii) TMA for asynchronous HBM↔SMEM data movement；(iii) setmaxnreg for dynamic register (de)allocation between warpgroup roles；(iv) bar.sync for inter-warpgroup synchronization and barrier-based pipeline management；(v) circular SMEM buffer (s-stage) with producer-consumer commit/wait protocol。
  - 修改：(i) 替换FlashAttention-2的同步模型——原FlashAttention-2内循环中BMM1→wait→softmax→BMM2→wait为全同步，FlashAttention-3改为2-stage流水线：BMM1(iter j+1, commit no wait) → softmax(iter j, overlapping with BMM1) → BMM2(iter j, commit no wait) → wait both；(ii) Warp-specialization替代原统一warp模型——producer warp仅执行TMA load + commit，consumer warpgroup仅执行WGMMA + softmax；(iii) FP8 precision——添加FP32→FP8 operand layout转换（byte_perm + shfl_sync），in-kernel V transpose（LDSM→byte_perm→STSM），k-major/m-major layout constraints处理；(iv) Inference优化——split-KV（Flash-Decoding）+ GQA packing + PagedAttention with TMA block table。
  - 评估脚本：CUDA event timing测量kernel wall-clock time，重复10次取平均，固定GPU clock 1830MHz。FLOPs计算：forward = 4 × seqlen² × head_dim × nheads（causal时/2），backward = forward × 2.5（2 forward matmuls + 5 backward matmuls via recomputation）。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  已开源：https://github.com/Dao-AILab/flash-attention（BSD license），集成到PyTorch。
  
  评估原理与流程（以单卡H100 SXM5 BF16 forward benchmark, seqlen=8192, head_dim=128, nheads=16为例）：
  1. **Input准备**：Q/K/V tensors in BF16/FP16 `[batch=2, seqlen=8192, nheads=16, head_dim=128]`（total tokens=16K），位于H100 HBM。
  2. **Kernel launch configuration**：Grid = 132 threadblocks（persistent kernel，对应132 SMs）。每个threadblock处理一个Q tile ($B_r$)。CTA内3个warpgroups：(a) producer warpgroup（1 warp, register-deallocated）执行TMA loads；(b) consumer warpgroup 1（2 warps, register-reallocated）；(c) consumer warpgroup 2（2 warps）用于pingpong scheduling。
  3. **Kernel执行（per CTA, single Q tile $\mathbf{Q}_i \in \mathbb{R}^{B_r \times 128}$）**：
     a. Producer warpgroup TMA load $\mathbf{Q}_i$ → SMEM → commit。进入主循环：for j=0..T_c-1，wait for stage consumed → TMA load $\mathbf{K}_j$, $\mathbf{V}_j$ → SMEM at stage j%s → commit。
     b. Consumer warpgroup 1 执行 Algorithm 2 主循环（2-stage GEMM-softmax pipelining）：
        - SS-WGMMA: $\mathbf{S}_{ij} = \alpha \mathbf{Q}_i \mathbf{K}_j^T$ (64×128×128, BF16 accum in FP32, ~128K MACs per tile)
        - softmax: FMNMX.FTZ (rowmax) → SHFL.BFLY (warp-level reduction) → MUFU.EX2 (exponential) → FADD + FMUL (rowsum + rescale)
        - RS-WGMMA: $\mathbf{O}_i \mathrel{+}= \tilde{\mathbf{P}}_{ij} \mathbf{V}_j$ (64×128 tensor core MMA)
        - 关键：WGMMA(QK^T) of iter j+1 异步发射后，softmax of iter j 在另一 warpgroup 的 tensor core 空闲间隙被调度执行
     c. Epilogue：$\mathbf{O}_i = \operatorname{diag}(\ell_i)^{-1} \mathbf{O}_i$ → TMA store $\mathbf{O}_i$, $L_i$ → HBM。
  4. **Performance measurement**：CUDA event记录kernel launch→completion时间。FLOPs = 4 × 8192² × 128 × 16 = ~550B FLOPs (forward)。TFLOPs/s = FLOPs / runtime。达到840 TFLOPs/s即85% H100 FP16理论峰值利用。
  5. **FP8 variant额外步骤**：(a) Q/K量化为FP8 e4m3 with per-block scaling；(b) in-kernel V transpose via LDSM + byte_perm + STSM转置in SMEM；(c) FP32 accumulator → FP8 operand register exchange via byte_perm + shfl_sync；(d) FP8 WGMMA: 2× throughput of BF16 WGMMA，达到1.3 PFLOPs/s。

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

## FlashMoE: Fast Distributed MoE in a Single Kernel

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现FlashMoE：首个将完整分布式MoE算子（Gate → Dispatch → Expert FFN → Combine）融合为单个持久化GPU kernel（megakernel）的系统。核心kernel调度设计包括：(i) **Actor-based并发模型**——将GPU thread block和warp特化为三种角色：Processor（N-1个block，执行GEMM和element-wise操作）、Subscriber（OS block内3个warp，解码远端tile packet为task descriptor）、Scheduler（OS block内1个warp，多线程work-conserving调度器，将计算任务动态分配给空闲Processor），实现fine-grained parallelism和loosely coupled non-blocking execution；(ii) **Tile-level parallelism**——将输入token矩阵划分为(128, 64) tile，每个tile映射为一个task descriptor，通过Scheduler根据readiness动态分配，确保无SM idle；(iii) **Device-initiated one-sided (R)DMA通信**——使用NVSHMEM建立跨GPU全局地址空间（PGAS模型），用DMA/RDMA替代bulk-synchronous AlltoAll collective，消除straggler延迟；(iv) **Symmetric tensor layout L**——维度为(P, R, B, E, C, H)的tensor，overprovision memory by 2r倍以消除one-sided write冲突，实现fully non-blocking memory access（Theorem 3.1可证write-write conflict-free），内存开销约占total inference memory的2%；(v) **In-place padding for payload efficiency**——在本地symmetric tensor buffer内直接padding而非网络传输null token，消除不对称routing导致的冗余通信payload；(vi) **In-device BLAS**——基于CUTLASS实现自定义high-performance GEMM，tile内FFN的两个GEMM（GEMM0 + GEMM1）与activation融合为单一fused __device__函数。

  实验比较：(i) Forward Latency——变化sequence length（2K-16K tokens per GPU），4 GPU和8 GPU配置，FlashMoE vs Comet、FasterMoE、Megatron-CUTLASS、Megatron-TE；(ii) GPU Utilization——SM utilization平均over 100 iterations，FlashMoE 93.17% vs FasterMoE 9.67%、DeepEP+Megatron-LM 13.55%、Megatron-TE 59.11%、Comet 42.31%；(iii) Throughput——scaling GPU count 2→8，MTokens/s，FlashMoE 17.7 MTokens/s at 8 GPUs；(iv) Overlap Efficiency（Weak Scaling）——定义O_e = T(2)/T(N_G)，2→8 GPU weak scaling，FlashMoE 4× higher at 8 GPUs vs baselines；(v) Expert Scalability——固定sequence length 16K，变化experts数8→128（total across all GPUs），4和8 H100。关键说明：FlashMoE使用FP32精度而所有baseline使用FP16，FlashMoE在通信量和计算量double的情况下仍大幅领先——up to 6× latency speedup, 9× higher GPU utilization, 5.7× throughput。

- 后端平台是什么，配置是什么。
  - 8× NVIDIA H100 80GB SXM GPU，NVLink互联
  - 125 GB RAM，20 vCPUs
  - PyTorch 2.6.0，CUDA 12.8，Ubuntu 22.04
  - NVSHMEM v3.2.5 用于device-initiated one-sided通信（替代NCCL AlltoAll）
  - CUTLASS 用于in-device BLAS/GEMM（替代cuBLAS host-launched kernel）
  - 2× A100 GPU用于kernel launch profiling（Table 1，Nsight Systems CUDA API trace）
  - MoE Transformer模型配置：16 attention heads，embedding dim=2048，FFN intermediate dim=2048，top-2 routing，capacity factor=1.0
  - 所有实验使用Distributed Data Parallelism (DDP) + Expert Parallelism (EP)，仅forward pass单层MoE

- 评估性能的软件/脚本是什么。修改了什么。
  FlashMoE为全新实现（6820行CUDA/C++），非修改现有框架。核心性能评估涉及以下组件：
  (1) **NVSHMEM替代NCCL**：将传统MoE框架中的NCCL AlltoAll collective（同步、barrier-based）替换为NVSHMEM device-initiated one-sided DMA/RDMA。NVSHMEM提供PGAS（Partitioned Global Address Space）编程模型，允许GPU kernel内直接发起跨GPU内存访问（nvshmem_put/get），无需CPU参与或collective barrier。FlashMoE在kernel内通过Subscriber actor检测NVSHMEM signal flag → 消费远端tile packet → 解码为task descriptor。关键修改：传统框架的host端NCCL调用（ncclAlltoAll）被替换为kernel内nvshmem_putmem + signal write。
  (2) **CUTLASS替代cuBLAS**：在persistent kernel内使用CUTLASS模板库实现device-side GEMM（GEMM0和GEMM1对应FFN两层），通过fused __device__ function将GEMM + epilogue activation（GELU/ReLU） + bias addition合并为单次调用。关键修改：传统框架每个GEMM是独立的cuBLAS kernel launch（需host→device dispatch），FlashMoE的GEMM在已运行的persistent kernel内通过CUTLASS device-side API直接执行。
  (3) **Warp specialization代替CPU launch**：传统MoE需要CPU逐kernel launch（Gate→Dispatch→Expert→Combine，DeepSpeedMoE多达550个kernel，Megatron-LM+DeepEP多达432个），FlashMoE的OS block内Scheduler warp（1个）和Subscriber warps（3个）在kernel内持续运行，通过shared memory doorbell和global memory flag进行actor间通信。关键修改：CPU从MoE layer的control plane移除，仅需一次kernel launch。
  (4) **Nsight Systems profiling**：用于统计kernel launch次数（Table 1，CUDA API trace）和SM utilization（Figure 9，SM Active metric）。FlashMoE仅1个persistent kernel vs baselines 33-550个short-lived kernels。SM utilization = ratio of cycles with ≥1 active warp to total cycles。

  评估方法：仅执行单层MoE forward pass，32 warmup + 32 measured iterations取平均值（减少cold-start和transient variance）。Latency通过CUDA event timing测量kernel wall-clock time。SM utilization通过Nsight Systems SM Active metric。Throughput = (Tokens × N_GPUs) / latency。Weak scaling efficiency O_e = T(2)/T(N_G)。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  已开源：https://github.com/osayamenja/FlashMoE (BSD-3-Clause)。提供pip安装（`pip install flashmoe-py[cu12]`，支持CUDA 12/13）和CMake C++构建两种方式。依赖：CUDA toolkit、C++20、ninja、CMake≥3.28、cuBLASDx、NVSHMEM。要求GPU SM70+且有P2P interconnect（NVLink、PCIe或GPUDirect RDMA）。C++测试：`mpirun -n <world> ./testFlashMoE <args>`。Python quickstart：`python quickstart.py`。

  评估原理与流程（以8×H100, S=16K tokens, E=128 experts, top-2 routing, FP32为例）：
  1. **Input准备**：MoE Transformer模型配置——16 attention heads, H=2048 embedding, I=2048 FFN intermediate。输入token矩阵A ∈ R^{S×H} (S=16K per GPU)，expert权重X ∈ R^{E×H×I}（128 experts分布在8 GPU，每GPU 16 local experts），capacity factor=1.0 → expert capacity C = (S × top_k × capacity_factor) / E = (16K × 2 × 1.0) / 128 = 256。
  2. **Single kernel launch**：`flashmoe::forward(A, X, O)` → CUDA kernel<<<grid, block, smem>>>启动单一persistent kernel。Grid配置：每个GPU上N个thread blocks（N-1个Processor + 1个OS block），block size=128 threads（Processor和OS block相同），shared memory=46KB/block，registers=255/thread，max active blocks per SM=2，0 spill stores/loads per thread，0 B kernel stack frame。
  3. **Kernel内FusedGate阶段**（所有block执行）：输入A → Gate function计算routing table T_φ ∈ (N×R)^{E×C}（记录每个expert slot对应哪个token及combine weight w）和affinity scores G_φ ∈ R^{S×E}。
  4. **Kernel内角色分化**（Algorithm 1）：blockId < N-1 → Processor role（调用processor::start()，进入while(!interrupt) loop等待Scheduler分配task）；blockId == N-1（OS block）中 warp 0 → scheduler::start()（Algorithm 3），warp 1-3 → subscriber::start(T_φ, G_φ, O, X)（Algorithm 4）。
  5. **Dispatch阶段**（Processor执行）：Processor按T_φ将token按(128,64) tile粒度从本地symmetric tensor copy到远端GPU的symmetric tensor layout L对应位置（Figure 7a）。使用NVSHMEM `nvshmem_putmem`（DMA intra-node through NVLink）或`nvshmem_putmem_nbi`（RDMA inter-node），一次transfer一个tile的连续内存。Transfer完成后write远端GPU的signal flag到L的flag区域通知Subscriber。In-place padding：token在本地写入L前已在本地补齐到expert capacity对齐（divisible by tile height bM=128），因此网络仅传输实际token tile。
  6. **Subscriber解码**（Algorithm 4）：Subscriber warps持续poll dispatch flag → 检测到signal set → atomic retrieve signal → memory fence确保数据可见 → 从L读取tile packet → 解码为GEMM0 task descriptor t_1 = (M, ·, φ_1) → 写入task queue tQ（global memory circular buffer）→ 通过shared memory doorbell notify Scheduler。
  7. **Scheduler调度**（Algorithm 3）：Scheduler warp sweep Processor doorbells + Subscriber doorbells → populate task count per queue into tqState → WarpInclusiveSum计算全局task count和inclusive scan output qS → 从ready queue rQ取idle Processor ID → 以qS为起始索引signal Processor处理tqState中对应task。Scheduler是多线程且work-conserving的——只要有task且有空闲Processor就立即调度，不等待batch。
  8. **Processor GEMM0执行**（Algorithm 2）：Processor warp 0 thread 0 awaitTaskFromScheduler → warp broadcast task descriptor到shared memory → switch(task.Type==GEMM0) → 调用fused GEMM kernel fGET(GEMM0, task)：CUTLASS device-side GEMM A×B_1 → tensor core MMA → activation φ_1 (GELU/ReLU) applied in-register → accumulate bias D_1 → write intermediate C_1 → notify tile completion → NotifySchedulerNextGEMM（将GEMM1 task加入调度）。
  9. **Processor GEMM1执行**：Scheduler再次schedule同一tile的GEMM1 task → Processor执行fGET(GEMM1, task)：C_1×B_2 (tensor core MMA) → epilogue (identity φ_2 in-register) → 若tile目标GPU为远端GPU，通过NVSHMEM put directly write result到远端GPU的combine buffer L[src_gpu][COMBINE][INCOMING][dst_gpu][expert_idx][slot] → 远端Subscriber检测combine signal → 解码为combine task。
  10. **Combine阶段**：Subscriber检测combine signal on flags → atomic retrieve → memory fence → 解码tile packet为combine task t_3 = (M, ⊙, φ_2) → Scheduler调度 → Processor执行combine：F_{t_3}(A, S, C, C) = C ← (A ⊙ S + C)，即对多个expert输出的同一token做Hadamard product with affinity scores G_φ后的weighted accumulation → 最终写回output O。
  11. **Kernel exit**：Scheduler检测scheduled count == taskBound（所有task完成）→ InterruptSubscribers() → InterruptProcessors() → Processor和Subscriber退出while loop → persistent kernel返回。Timer stop（CUDA event record）。Output O ∈ R^{S×H}。
  12. **Performance关键**：Symmetric tensor layout L的write-write conflict-free性质（Theorem 3.1）——每个NVSHMEM put的index (p_s, r, b=1, e, c)唯一确定目标内存位置，p_s即source GPU rank作为L第一维索引，不同GPU写入不同位置→无需同步。Temporal buffering（B=2 staging buffers × R=2 rounds = 4× token buffer size）提供double-buffering隔离dispatch和combine的并发访问。Overlap efficiency来自Processor在执行GEMM的同时Subscriber可解码下一个tile packet、Scheduler可调度已就绪task——三者通过shared memory/global memory异步signal通信，无barrier。

## HyTiS: Hybrid Tile Scheduling for GPU GEMM with Enhanced Wave Utilization and Cache Locality

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现是HyTiS，一个两层级联的混合 tile 调度框架。第一级使用吞吐量导向的 micro-kernel（large tiles）处理 full waves，最大化 SM 利用率；第二级使用延迟导向的 micro-kernel（fine-grained tiles）处理 partial wave，最小化残余 wave 的延迟。此外，HyTiS 实现自适应 tile layout 选择（group-M vs group-N），通过分析模型最小化 wave 粒度的 DRAM 到 L2 cache 数据流量。实验比较 HyTiS 与 cuBLAS、Inductor-Triton、Split-K（CUTLASS）、Stream-K（CUTLASS），以及两个消融变体 HyTiS(L1)（仅单级调度）和 HyTiS(STL)（静态 tile layout）。评估指标包括执行延迟（speedup over cuBLAS）、SM 负载均衡度 B=(max-min)/avg（通过 NSight Compute 采集 sm_cycles_active 指标）和 DRAM read 数据量（NSight Compute dram_bytes_read.sum）。

- 后端平台是什么，配置是什么。
  NVIDIA H100-PCIE (80GB, Hopper 架构, compute capability sm_90) 和 NVIDIA A100-PCIE (40GB, Ampere 架构, compute capability sm_80)。

- 评估性能的软件/脚本是什么。修改了什么。
  基于 Triton 3.2.0 实现 HyTiS 调度框架。主要修改包括：
  1. 实现两层级联 GEMM kernel（Algorithm 1）：第一级循环以 TO micro-kernel K1 执行 n1_wave 个 full waves；第二级以 LO micro-kernel K2 处理剩余 n2_tiles。每个级别的 tile-to-output-offset 映射由 HyTiScheduler 生成的 l1_offset_fn 和 l2_offset_fn 函数控制。
  2. 离线 profiling 阶段：在目标 GPU 上对单个 data layout 执行一次 GEMM operator profiling（H100 ~19 min，A100 ~36 min），收集 SMEM 使用量、register spill 情况和执行延迟，构建 TO 候选集 S^TO 和 LO 候选集 S^LO。
  3. 自适应 tile layout：运行时根据问题形状和 tile 配置，利用分析模型计算最优 group-M/group-N 布局及 group size s_opt，选择使 wave 粒度 DRAM→L2 流量 V_tol 最小的布局。
  4. Hopper 架构上使用 persistent kernel + TMA 指令；Ampere 架构上使用传统 data-parallel launch（因 TMA 不支持且 persistent kernel 导致 register 压力过大）。
  5. 自动调优：在运行时对 TO-LO 组合搜索空间执行 auto-tuning，缓存结果以消除重复开销。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  开源，Zenodo DOI: 10.5281/zenodo.15244191（论文 AE 版本），另有 Zenodo DOI: 10.5281/zenodo.16674739。评估原理和流程如下：
  1. **安装**：`src/install.sh` 安装 PyTorch 2.3.1、Triton（从源码 build，应用 `patchs/triton-patchs/` 下两个 patch 以启用 HyTiS 调度）、CUTLASS 3.4.1（用于 Stream-K/Split-K baseline）、HyTiS（pip install -e .）。
  2. **整体性能基准测试**：`python exps-1.0/run_tasks.py 0`（无 NCU profiling）运行全部 3600+1024=4624 个 GEMM 测试用例，对每个 (M,N,K, layout) 组合分别跑 cuBLAS、Inductor-Triton、Split-K、Stream-K、HyTiS(L1)、HyTiS(STL) 和 HyTiS，记录执行延迟，计算 speedup 并输出到 `checkpoints/cache/`。生成 Figure 9：`python exps-1.0/fig9.py`。
  3. **Breakdown 分析**：`python exps-1.0/run_tasks.py 1`（带 NCU profiling）采集 SM balance 指标（sm_cycles_active.avg/.max/.min）和 DRAM read 量（dram_bytes_read.sum），按 low/mid/high 三区归一化到 cuBLAS。生成 Table 3：`python exps-1.0/table3.py`。
  4. **Wave quantization 专项测试**：固定 N,K 变 M（如 M ∈ [512,8192] step 64, N=1024, K=4096），在量化显著区（orange highlight）和非显著区比较 HyTiS 与 cuBLAS/Inductor-Triton 的延迟。通过 `MNs=0 MNe=120 TASK_ID=3 python run_tasks.py` 可快速评估 120 个代表性 case（~1 小时）。生成 Figure 10：`python exps-1.0/fig10.py`。
  5. **Hyperparameter 分析**：`MNs=0 MNe=1000 L1_THRES=1.3 L2_THRES=1.4 python run_tasks.py` 测试不同 virtual tile 数量下 l1、l2 阈值对搜索空间和性能的影响。生成 Figure 11：`python exps-1.0/fig11.py`。
  6. **kernel 输入到性能输出全流程**：用户调用 `hytis.matmul(a, b)` → HyTiScheduler 接收问题形状 (M,N,K) → 检查 tuning cache → 若无缓存命中，对 TO×LO 搜索空间内每个有效组合（first level tile 覆盖 full waves，second level tile 覆盖 partial wave 且 tile 数 ≤ N_SM）进行 auto-tuning → 选择最优 (K1, K2, layout) 组合 → HyTiS_GEMM kernel launch：grid_size 个 CTA 并发执行，前 n1_wave×k_tiles 个 iteration 以 K1 处理 full waves（TMA load on H100），后续以 K2 处理 n2_tiles 个 partial wave tiles → 结果写回 output tensor C。性能指标通过 CUDA event timer 采集 kernel 执行时间，NSight Compute 采集 SM 利用率和内存流量。

## MetaAttention: A Unified and Performant Attention Framework across Hardware Backends

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  MetaAttention 的 kernel 调度核心是 **IntermediateTensor-based scheduling** + **two-layer scheduling policy**。实现包括：(i) **IntermediateTensor 抽象**——将 attention 计算中所有 transient tensors（Q/K/V/scores/weights/output 及 customizable function 内部中间结果）统一建模为 IntermediateTensor 对象，每个对象携带三个可配置属性：TileShape（tile size, 通过 computation graph 传播推导所有 tensors 的 tile shape）、MemoryLocation（Register/Shared Memory/Global Memory, 逐级权衡延迟与容量）、PipelineStage（memory copy 与 computation 的重叠阶段数，决定 buffer 需求和调度灵活性）；(ii) **DeviceConfig 抽象**——封装 hardware-specific 约束：BaseTileShape（硬件对齐的 tile shape，如 H100 wgmma MMA instruction tile 和 memory transaction alignment）和 MemoryInfo（各 memory tier 容量：Register 256KB/SM, Shared Memory 228KB/SM, Global Memory 80GB）；(iii) **Tile Config Scheduling（外层）**——枚举所有合法 output tile sizes（对齐 basetile），沿 computation graph 传播 tile shape 约束到所有 IntermediateTensors（相邻 tensors 必须共享相同 tile size），生成候选 tile graphs；(iv) **Tile Resource Scheduling（内层）**——对每个 tile graph，初始化所有 tensors 到最高 memory tier (Register)，按 `(use_count, tile_size)` 排序逐步降级 memory location（Register→Shared Memory→Global Memory），在每级枚举 PipelineStage 并检查 memory constraint，返回所有合法 execution plans；(v) **Profiling-based Selection**——对所有合法 plans 通过 profiling 选 latency 最优者；(vi) **Attention Runtime Dispatch**——根据 scheduling plan 选择 Parallel 或 Recurrent kernel template，通过 code inlining 将 customizable function 的 hardware-mapped code 直接 fused 到 attention execution loop，实现零额外 kernel launch overhead。NVIDIA backend 利用 TMA (Tensor Memory Accelerator) 异步数据加载 + Tensor Core MMA；AMD backend 利用 Matrix Core + async copy。

  实验比较：(1) H100 上 10 种 attention 变体的 kernel 延迟 vs 手写 library（FA2/FA3/FlashMLA/FlashSigmoid/Mamba2/FLA）和编程模型方案（FlexAttention/FlashInfer），涵盖 Parallel、Recurrent、Customized、MLA、Sparse GQA 五类；(2) MI250 上 4 种 attention 变体（Softmax/ReLU/Mamba2/RetNet Recurrent）vs baselines，验证跨硬件调度能力；(3) 编译时间——分钟级（46-89s），significantly shorter than traditional DL compilers；(4) 开发工作量——22-90 LoC vs 手写 library 400-3000 LoC。

- 后端平台是什么，配置是什么。
  NVIDIA H100 SXM5 (132 SMs, 80GB HBM @ 3.35TB/s, 989 TFLOPS FP16, CUDA 12.4, Triton 2.3.1)，AMD Instinct MI250 (ROCm 6.2.4, Triton 3.1.0)。NVIDIA backend 使用 TMA (cp.async.bulk) + Tensor Core wgmma；AMD backend 使用 Matrix Core + async copy。DataType FP16。

- 评估性能的软件/脚本是什么。修改了什么。
  MetaAttention 为全新实现的框架（7.3k lines C++/Python），非修改现有软件。评估使用自编 benchmark 脚本测量 attention kernel wall-clock time，baseline 使用 FlashAttention-2 v2.7.4、FlashAttention-3、FlashMLA（blockSize=64）、FlashSigmoid、Mamba2 chunk kernel、Flash-Linear-Attention v0.2.0、FlexAttention、FlashInfer、PyTorch native。

  核心 kernel 调度实现（Algorithm Fig.10 伪代码对应）：
  ```
  // 外层: Tile Config Scheduling
  tiles = EnumerateTiles(g.output_shape, D.basetile)  // 对齐 MMA tile 枚举
  tensor_tile_graphs = PropagateTileGraphs(g, tiles)   // 沿 computation graph 传播 tile shape
  for tile_graph in tensor_tile_graphs:
    plans += TileResourceScheduling(tile_graph, D)    // 内层: memory + pipeline 调度
  for plan in plans:
    if Profile(plan) < best_latency:                  // profiling-based 选最优
      best_plan = plan
  
  // 内层: Tile Resource Scheduling
  tensor_list = GetIntermediateTensors(g)
  SetTile(tensor_list, g.tiles)
  SetMem(tensor_list, "L0")                           // 初始全放 Register
  tensor_list_sorted = sort(tensor_list, key=(use_count, tile_size))
  for tensor_i in tensor_list_sorted:
    plans = EnumerateUnsetAttributes(tensor_list)      // 枚举 pipelineStage
    for plan in plans:
      if not MeetMemoryConstraint(plan, D.memoryInfo): // 检查 Register/SMEM 容量
        plans.remove(plan)
    if not plans.isEmpty():
      return plans
    LowerMemLocation(tensor_i.mem)                     // Register→SMEM→Global 降级
  ```

  与 Handwritten Kernel Scheduler 的关键差异：
  - Baseline（FA2/FA3/FlashMLA/Mamba2）：kernel 内 tile size、memory placement、pipeline stage 全部 hardcode（如 FA2 固定 B_r=128/B_c=128，SMEM 分配固定），不同 attention 变体或不同 GPU 需重新手写
  - MetaAttention：scheduling 由 IntermediateTensor attributes + DeviceConfig constraints 自动推导；同一套 scheduling policy 适用于所有 attention variants（通过 computation graph 自动传播 tile shape）和所有 hardware backends（通过 DeviceConfig 控制约束）；tile size 非固定——对 dimqk≠dimv 的配置（如 Diff-Transformer-3B: dimqk=128, dimv=256）自动选择不等长的 tile sizes 避免 zero-padding waste

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  已开源：https://github.com/SJTU-IPADS/MetaAttention (MIT License)。Docker 环境：CUDA backend (`Dockerfile.cu128`, ~50 min build)，ROCm backend (`Dockerfile.rocm`, ~80 min build)。Performance test 复现 Figure 11 (H100, ~90 min) 和 Figure 14 (MI250X, ~20 min)。

  **Kernel 调度评估原理与流程**（以 H100 Softmax Attention (head=32, dimqk=128, dimv=128), batch=1, seqlen=2048 为例）：

  1. **Input 准备**：用户定义 Parallel Pattern attention template → MetaAttention 解析为 computation graph G = {Q, K, scores, weights, V, output} 六个 IntermediateTensors + customizable functions DAG（无自定义函数则为 identity）。DeviceConfig: H100 BaseTileShape={M=64/128, N=128/64} (wgmma MMA tile), MemoryInfo={RF:256KB, SMEM:228KB, GMEM:80GB}。

  2. **Tile Config Scheduling（外层）**：
     - EnumerateTiles: 枚举所有合法 output tile shapes——对 dimv=128, 可能的 output tile = {(Br,128) | Br ∈ {64,128,256,...}, Br × 128 × 2B ≤ SMEM free}，生成 ~10-30 个候选
     - PropagateTileGraphs: 沿 G 传播 tile shape——
       - output[B_r, d_v] → weights[B_r, B_c]（需 B_c 对应 V 的 seqlen tile）→ scores[B_r, B_c]（与 weights 同 shape）→ Q[B_r, d_qk] 和 K[B_c, d_qk]（与 scores 的首/末维度对应）
       - 约束：相邻 tensors 共享对应维度 → 所有合法 tile graphs 约 5-15 个

  3. **Tile Resource Scheduling（内层）对每个 tile graph**：
     - 初始 memory location: Q=RF, K=RF, scores=RF, weights=RF, V=RF, output=RF
     - Sorted by (use_count, tile_size): scores(used 2×: weights calc + online norm) > Q(1×) ≈ K(1×) ≈ V(1×) ≈ output(1×)
     - 枚举 pipelineStage: 对 MMA-heavy patterns, pipeline stages=2（async TMA load + compute 重叠）；对 memory-bound patterns, stages=1
     - Memory constraint check: Σ(tile_size × 2B per tensor) ≤ SMEM 228KB, register pressure ≤ 255/SM thread
     - 若 scores[B_r,B_c] + Q[B_r,d] + K[B_c,d] + V[B_c,d] 超 SMEM → 降级 scores 到 Global Memory → re-check → 若仍超 → 降级 V 到 Global Memory → ... → 找到满足约束的最优 placement

  4. **Profiling-based Selection**：
     - 对每个合法 plan，构造 execution time estimate（通过 lightweight microbenchmark 或 analytic cost model）→ 选 latency 最小的 plan
     - 可选实际 kernel launch profiling（更精确但更慢）

  5. **Attention Runtime Code Generation & Execution**：
     - 选择 Parallel Pattern kernel template → 根据 plan 配置 tile sizes, memory buffers, pipeline stages
     - Code inlining: 将 hardware-mapped customizable functions（若有）直接 inline 到 kernel mainloop
     - NVIDIA CUTE backend 生成：TMA cp.async.bulk 异步 load Q/K/V tiles (HBM→SMEM) → wgmma QK^T (SMEM→RF) → CUDA core online softmax (max/exp/rowsum in RF) → wgmma PV (RF accum) → TMA store output (RF→HBM)
     - Pipeline: producer warp (TMA load next K/V tile) ∥ consumer warp (compute current tile)，通过 mbarrier 同步

  6. **Performance measurement**：
     - CUDA event 记录 kernel wall-clock time → TFLOPs = attention FLOPs / time
     - 对比 FA3 (FlashAttention-3 handcrafted CUDA kernel with hardcoded scheduling)
     - 结果：MetaAttention achieves comparable or up to 1.61× (Diff-Transformer-3B forward) speedup over FA3，因 scheduler 可为 dimqk≠dimv 配置自动选择 non-padded tile sizes（FA3 固定 pad 到相同维度）

  7. **Cross-backend 验证（MI250）**：
     - 同 attention template → DeviceConfig 切换为 MI250（BaseTileShape 适应 AMD Matrix Core, MemoryInfo 适应 MI250 hierarchy）→ scheduling policy 自动生成 ROCm-optimized plan → TileLang backend lowering → ROCm kernel
     - MI250 上平均 3.3× forward / 2.0× backward speedup over baselines

  8. **Scheduling Time**：外层枚举 + 内层贪心 → 每次编译 46-89 秒（Table 4），远快于传统 auto-scheduling compiler (Ansor: minutes to hours)。Scheduling result 可 cache 复用（相同 attention config + device → 直接加载 plan，跳过 scheduling）。
