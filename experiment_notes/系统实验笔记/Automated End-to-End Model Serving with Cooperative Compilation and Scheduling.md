## Automated End-to-End Model Serving with Cooperative Compilation and Scheduling

- 属于Serving调度的实现是什么？实验比较什么？
  提出Infera inference server（约17k LoC C++ kernel-space module），通过cooperative compilation and scheduling实现端到端DNN inference serving。核心Serving调度实现包含：(1) Job Dispatch Unit (JDU)：从inference job queue中按FCFS dequeue jobs，基于GPU可用显存和estimated remaining time（通过#inst/IPC估算）选择目标GPU，支持overload throttling和periodic task migration；(2) Task Schedule Unit (TSU)：任务5状态机（New→Blocked↔Ready→Running→Exit），内存管理通过mm_wq异步swap-in/out（LRU后台回收，4个CUstream处理分配/释放/双向传输）。Priority scheduling：64个runqueue（priority 0-63），ddl_rq(0)用EDF、rt_rq(1-39)用FIFO、gcfs_rq(40-63)用GCFS按nice value分配instruction budget。Aging mechanism逐步提升长期未选normal任务优先级。Preemption：高优先级任务到达→发送preemption signal→TSU暂停调度保存状态→TEU响应保存HKQ/DKQ/shared memory queue；(3) Task Execution Unit (TEU)：三阶段pipeline——SelectKernels（两阶段：先选data block最大化asynchronous wavefront，再按#inst/IPC+TLP≥4+hazard分析在线回归模型选kernel）→FuseKernels（warp-level binary fusion：prologue恢复special registers+shared memory offset+barrier重组+preemption/locking/progress flags）→LaunchKernel（HKQ→GDRCopy→DKQ→daemon kernel CDP fire-and-forget launch，覆盖placeholder kernel slots实现device-side launch）；(4) Daemon Kernel：常驻一个SM，用CUDA Dynamic Parallelism device-side发射kernel，维护shared-memory double-ended queue，fire-and-forget launch消除stream tracking overhead和HoL问题（launch latency <10μs）。Preemption：host端暂停HKQ→DKQ传输、保存HKQ kernel，device端保存DKQ和shared memory kernel；in-flight kernel通过flag快速终止，kernel idempotent execution保证安全。实验对比Stream、MPS、Triton、Paella，single-model serving speedup 1.14×–1.40×（平均1.28×），multi-model serving speedup至少1.6×（uniform requests/uniform models）最高3.5×（lognormal requests/lognormal models）。

- 硬件平台是什么，配置是什么。
  Intel Xeon Gold 6330 CPU, 512 GB RAM, NVIDIA A100-PCIE-40GB GPU, Linux 6.1.0, CUDA 12.0。Latency-sensitive CPU线程使用real-time scheduling或绑定isolated CPU core并禁用中断；GPU daemon kernel独占一个SM。

- 开源Serving框架是什么。修改了什么。
  Infera inference server从零开发（C++ kernel-space module，约17k LoC），不基于现有开源serving框架。实现包括：JDU（GPU dispatch+load balancing）、TSU（priority scheduling+task state machine+memory management）、TEU（kernel select/fuse/launch pipeline）、daemon kernel（CDP-based device-side launch）、preemption机制、GDRCopy-based low-latency data transfer、driver-level placeholder kernel slot覆盖。

- 开源情况。Serving框架如何使用？作用是什么？基于开源文档和论文，使用例子解释。
  未找到明确开源仓库（EuroSys 2026, DOI: 10.1145/3767295.3769392）。以multi-model mixed inference为例：
  1. 部署：启动Infera inference server（加载编译好的模型库和权重到model pool，host memory标记为cudaHostAllocWriteCombined/cudaHostAllocPortable）→ JDU初始化GPU dispatch表→ TSU初始化per-GPU task scheduler（64 runqueues）→ TEU初始化kernel fuser thread pool和daemon kernel
  2. 请求到达：用户提交inference job（如normal priority, nice=0）→ JDU检查GPU可用显存，选择estimated remaining time最小的GPU→ job在目标GPU上创建task（New状态）
  3. 内存管理：Task检查显存需求→若权重/input tensor不在GPU memory→进入Blocked状态→mm_wq等待异步swap-in（优先级排序，同优先级FIFO）→LRU后台回收旧模型腾空间→数据就绪→进入Ready状态→加入gcfs_rq
  4. 调度周期：TSU开始scheduling cycle→从gcfs_rq选择tasks组成VT→按nice value分配instruction budget→生成VTB→TEU执行VTB
  5. TEU三阶段：(a) SelectKernels：扫描data dependency DAG选择zero in-degree且maximize asynchronous wavefront G(u)的data blocks→在线回归模型估计IPC→选#inst/IPC最小且TLP≥4的kernel；(b) FuseKernels：多个CUDA binary kernel warp-level fusion→prologue恢复%tid等special registers→shared memory offset添加→barrier重组→insert preemption/locking/progress flags；(c) LaunchKernel：fused kernel入HKQ→GDRCopy copy code+args到device→覆盖placeholder kernel slot→入DKQ→daemon kernel CDP fire-and-forget launch
  6. Multi-model scenario：BERT+ViT+Inception混合请求→bursty lognormal arrival→Infera tile-level调度允许cross-model micro-kernel colocation→warp-level fusion保证SM内空间共享→比baseline快1.6×-3.5×。Infera-P preemption约10μs，Infera-R约5μs，Infera-P比REEF-N快约2.5×，比EffiSha快超一个数量级
