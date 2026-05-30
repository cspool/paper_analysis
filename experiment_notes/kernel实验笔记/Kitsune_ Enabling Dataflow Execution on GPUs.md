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
