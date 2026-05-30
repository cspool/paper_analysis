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
