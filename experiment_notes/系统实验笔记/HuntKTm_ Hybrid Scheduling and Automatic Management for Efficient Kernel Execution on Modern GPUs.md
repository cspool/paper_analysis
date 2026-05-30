## HuntKTm: Hybrid Scheduling and Automatic Management for Efficient Kernel Execution on Modern GPUs

- 属于Serving调度的实现是什么？实验比较什么？
  实现是task scheduler，在multi-GPU系统中自动将多个任务（每个任务是包含多个kernel的独立GPU程序）调度到合适的GPU设备上，实现task-level并发执行。task scheduler包含三部分：(1) resource analyzer——编译期分析每个kernel的launch配置（thread数、register数、shared memory量）和memory object大小，汇总每个task的计算和内存资源需求；(2) lazy engine——运行时拦截CUDA API调用，延迟GPU相关操作（memory allocation/deallocation/data transfer），收集编译期无法确定的动态资源信息，精确预测task资源需求后发送给task dispatcher；(3) task dispatcher（Algorithm 1）——遍历可用GPU列表，基于三个维度（threads、registers、shared memory）评估SM可用量，选择拥有最多available SMs且内存和hardware queue充足的GPU。若所有GPU都无法满足需求，task挂起到pending queue等待资源释放。

  实验比较的task-level scheduling baseline包括：SA (Single-Assignment，每个GPU一次分配一个任务)、CASE (compiler-assisted scheduling framework，自动分析资源需求并调度)、HuntK (仅stream scheduler + SA)、HuntKT (stream scheduler + task scheduler，无memory management)。也对比了单任务kernel concurrency baseline：Taskflow（静态）、GrSched（动态）、Serial。评估指标包括system throughput（多GPU系统）、硬件资源利用率（DCGM采集的SM occupancy/FP32 utilization/memory bandwidth utilization）、memory reduction ratio、task-level kernel execution speedup。

- 硬件平台是什么，配置是什么。
  服务器配备4× NVIDIA A100 GPU (40GB HBM each, 6912 CUDA cores each)、2× AMD EPYC 7742 64核处理器、256 GB DDR4内存。操作系统Debian 10.2.1，NVIDIA driver 555.42.06。另一平台：4× NVIDIA RTX 4090 24GB GPU、2× Intel Xeon Gold 6338N CPU、1024 GB DRAM。NVIDIA MPS启用以实现跨进程space-sharing并发，NVIDIA persistence mode启用以减少GPU初始化开销。

- 开源Serving框架是什么。修改了什么。
  未使用现有开源Serving框架。HuntKTm自建task scheduler系统，基于CUDA Runtime和LLVM Compiler Infrastructure实现。修改/自建内容包括：
  - 通过function wrapper拦截所有CUDA runtime调用（cudaMallocAsync, cudaFreeAsync, kernel launch等），收集resource信息。
  - lazy engine维护deferred CUDA operation queue，在task被dispatch到具体GPU前暂缓所有GPU操作。
  - resource analyzer通过nvcc获取每个kernel的register和shared memory使用量。
  - task dispatcher实现Algorithm 1的资源感知调度算法，通过shared memory与lazy engine通信。
  - 调度后调用cudaSetDevice绑定task到目标GPU，使用cudaDeviceGetDefaultMemPool获取默认memory pool并用cudaMemPoolSetAttribute设置release threshold为预测的memory footprint。
  - 每个GPU的可用hardware queue数上限设为32（匹配CUDA runtime最大连接数）。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  开源链接：https://github.com/Gemini321/HuntKTm

  作用：HuntKTm的task scheduler解决的是"多个无协作关系的GPU程序（tasks）如何自动在multi-GPU系统上高效共置"的问题。传统方式需要用户手动为每个程序指定目标GPU并确保不超出内存容量，HuntKTm通过编译期资源分析和运行时lazy execution实现全自动、内存安全的task-to-GPU调度。

  全过程（以W4 workload为例，包含约16个混合大小的task同时到达的场景）：
  ```
  用户编译多个multi-kernel CUDA程序（已通过stream scheduler自动转化为多stream版本）
    → LLVM pass (resource analyzer): 分析每个kernel的launch配置(num_threads, registers, shared_mem)
      和memory object大小，插入cudaTaskSchedule调用点在资源需求完全确定的位置
    → 编译生成带resource metadata的可执行程序
    → 运行时：W4的所有16个task同时到达，每个task启动lazy engine
    → lazy engine拦截所有CUDA API调用并存入deferred operation queue，暂不执行
    → 当程序执行到cudaTaskSchedule点：lazy engine聚合所有stream的首kernel computing需求
      和所有memory object大小，通过shared memory发送给task dispatcher
    → task dispatcher执行Algorithm 1:
      for each GPU g in 4×A100:
        if g.free_memory >= task.memory and g.free_hw_queues > 0:
          score = g.available_SMs - min(thread_score, reg_score, shmem_score)
          选择score最大的GPU
    → 若GPU可用：lazy engine → cudaSetDevice(target_gpu_id)
      → 初始化memory pool (size = predicted_peak_memory)
      → 顺序执行deferred operations (cudaMallocAsync → cudaMemcpyAsync → kernel launches)
      → 多stream的kernels并发在MPS环境下与同GPU其他task的kernels space-sharing执行
    → 若GPU不可用：task进入pending queue，等待其他task释放资源后被唤醒重试
    → DCGM监控每个GPU的硬件指标 → 计算system throughput = 总完成task数 / 总耗时
  ```

  HuntKTm通过该流程，相比CASE在多GPU A100系统上实现平均33.2% throughput提升，在4090系统上平均52.5%提升（因4090内存更紧张，memory management效果更显著）。
