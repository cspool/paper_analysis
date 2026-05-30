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
