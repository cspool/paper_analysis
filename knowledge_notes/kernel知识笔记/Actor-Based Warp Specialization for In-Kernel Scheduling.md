## Actor-Based Warp Specialization for In-Kernel Scheduling

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Actor-Based Warp Specialization 是 FlashMoE 提出的 GPU kernel 内并发编程模型：将 thread block 内的 warp 按功能角色 (actor) 特化——每个 actor 独立异步执行特定职责，通过 shared memory/global memory signal (doorbell/flag) 进行 loose coupling 的 event-driven 通信。不同于 Hopper 硬件 warp-group specialization (producer/consumer + TMA + mbarrier)，FlashMoE 是纯软件 actor 模型，可在 SM70+ GPU 运行。三种角色: Processor (N-1 blocks, 执行 GEMM + combine + dispatch), Subscriber (OS block 内 3 warps, 解码远端 tile packet → task descriptor → enqueue), Scheduler (OS block 内 1 warp, 多线程 work-conserving 调度)。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Actor 交互: Subscriber poll NVSHMEM flag → atomic retrieve → memory fence → 从 L 读 tile → 解码为 GEMM0 task → write task queue (GMEM circular buffer) → atomicAdd doorbell (SMEM) notify Scheduler。Scheduler sweep all doorbells → WarpInclusiveSum → 从 ready queue 取 idle Processor → signal Processor (GMEM)。Processor await_scheduler_signal → warp broadcast task → switch(type): GEMM0 (fGET fused CUTLASS GEMM+GELU) → notify completion → schedule next GEMM1 → GEMM1 (fGET + NVSHMEM put result) → combine (Hadamard + accumulate)。各 actor 无 barrier——通过 poll + signal pattern 实现 event-driven。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现: (1) CUDA SIMT 模型自然支持同一 block 内不同 warp 执行不同代码路径 (通过 warpId 分支)，无需 __syncthreads() 全局同步；(2) Non-blocking communication——所有角色通过 memory-based signal 通信，不 block 等待；(3) 资源分配——N-1 blocks (~97% SM) 用于 Processor 最大化计算吞吐，仅 1 block 做 administrative tasks。对比 Hopper 硬件 warp-group spec: producer/consumer 基于 TMA+mbarrier 强耦合，FlashMoE actor model 纯软件 loose coupling，更灵活但需更精细 memory ordering。

涉及论文标题：
- FlashMoE: Fast Distributed MoE in a Single Kernel
