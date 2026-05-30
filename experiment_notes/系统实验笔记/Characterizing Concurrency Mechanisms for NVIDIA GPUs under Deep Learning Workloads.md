## Characterizing Concurrency Mechanisms for NVIDIA GPUs under Deep Learning Workloads

> **近似层次匹配说明**：本文并非修改开源Serving框架，而是直接使用CUDA API（priority streams、time-slicing、MPS）在单GPU上执行concurrent training+inference workload，以评估不同并发机制对inference serving性能的影响。这与Serving调度的核心关注点（多请求调度、SLO、吞吐量）紧密相关，但实现层面在CUDA/kernel级而非Serving框架级。

- 属于Serving调度的实现是什么？实验比较什么？
  论文实现了在单GPU上同时运行latency-sensitive inference serving和best-effort training的并发workload，通过三种NVIDIA GPU并发机制（priority streams、time-slicing、MPS）进行调度。实验比较各机制下inference请求的turnaround time（平均延迟）、variance（可预测性），以及training execution time（资源利用率proxy）。Inference请求使用两种模式：MLPerf server mode（Poisson过程到达，500请求）和MLPerf single-stream mode（连续请求，5000请求）。

- 硬件平台是什么，配置是什么。
  NVIDIA GeForce RTX 3090（Ampere microarchitecture）：82 SMs、1536 threads/SM、64KB registers/SM、1024KB shared memory/SM、24GB GDDR6X、6144KB L2 cache。

- 开源Serving框架是什么。修改了什么。
  论文未使用或修改开源Serving框架。所有并发调度通过CUDA runtime API直接控制：
  - **Priority streams**：将training和inference task置于同一OS进程的不同CUDA stream，inference stream设高优先级（-2到0三级），CUDA thread block scheduler优先从高优先级stream取blocks。
  - **Time-slicing**：两task作为独立进程运行，由CUDA application-level scheduler以约2ms固定时间片轮转调度。
  - **MPS**：启动MPS server，两task作为独立MPS client进程提交kernels，MPS server调度来自不同CUDA context的kernel blocks，允许spatial sharing（同一SM colocation），可设置per-client thread limit。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  论文未提供独立开源代码。以下是基于论文描述的各机制的调度全过程：

  **Priority Streams 全过程**（以ResNet-50 training + inference为例）：
  1. Host端：单进程内创建两个CUDA stream——stream_high（priority=0，inference kernels）和stream_low（priority=-2，training kernels）。
  2. Training task持续向stream_low提交kernel序列（如convolution、batch norm、ReLU等），Inference task根据请求模式向stream_high提交kernel序列。
  3. GPU端Thread Block Scheduler：当两种stream都有pending blocks时，总是优先从stream_high取blocks分配到SM。但已分配的low-priority blocks不会被抢占。
  4. **Compounded Delay问题**：当inference kernel执行完毕，下一inference kernel到达前有一段时间窗口。在此期间，training kernel抢占了GPU所有SM资源并填满blocks。下一inference kernel到达后必须等待当前执行中的training blocks完成才能被调度，造成约2-4×的turnaround time增加。
  5. 结果输出：每个inference kernel完成后记录timestamp，计算端到端turnaround time。

  **Time-Slicing 全过程**：
  1. Host端：两个独立进程各自创建CUDA context并提交kernels。
  2. GPU Application-Level Scheduler：以约2ms固定时间片round-robin分配GPU。每时间片内，整个GPU（所有82 SMs）专属于一个进程。
  3. 时间片切换：约145μs切换开销（通过global timer register测量，推测一半保存context、一半恢复context），但register和shared memory似乎不传输（推测为避免高开销）。
  4. 限制：两进程的kernel资源需求总和不能超过GPU硬件上限（即使不同时执行），否则第二进程OOM。
  5. 结果特点：turnaround time低且可预测（2ms延迟 + 切换开销），但utilization差（resource在时间片内空闲时无法被另一进程使用）。

  **MPS 全过程**：
  1. Host端：启动MPS control daemon和MPS server，两task作为MPS client进程连接server。
  2. MPS Server：接收来自两个CUDA context的kernel dispatch请求。调度策略为FCFS + leftover policy（优先调度最近到达kernel的所有blocks）。
  3. SM Spatial Sharing：两个进程的thread blocks可以在同一SM上colocated，只要两者thread总和不超过SM limit。
  4. 实验结果：MPS utilization最好（training execution time增加通常仅20-30秒），但inference degradation因无优先级而显著（如ResNet-152 turnarround time 2×）。

  **性能对比总结**（基于论文Figure 1/3）：
  - ResNet-50: Priority Streams +103% TT, MPS +78% TT, Time-Slicing +18% TT（但training time +90s）
  - VGG-19: Priority Streams worst（training含大量long-running kernels），Time-Slicing TT best
  - RNNT+BERT: Time-Slicing表现差（因memory transfer contention）
