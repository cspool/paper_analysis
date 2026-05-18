## Semantic Gap in GPU Scheduling (GPU调度语义鸿沟)

术语是什么？

Semantic Gap in GPU Scheduling（GPU调度语义鸿沟）是μShare定义的核心问题概念：指kernel的实际硬件资源需求与GPU硬件调度器的资源分配决策之间存在信息不对称——kernel知道自己的计算特性（偏向何种硬件资源），但硬件调度器仅根据简单的资源容量条件（thread/smem/reg数量）做block placement决策，不具备理解kernel micro-architectural resource demand的能力。这一semantic gap导致调度器无法将资源需求互补的kernel block分配到同一SM（如LDST-heavy + INT32-heavy），只能按顺序调度同kernel blocks（stacked co-location），造成intra-SM硬件资源浪费。在NVIDIA GPU硬件闭源的约束下，μShare选择在kernel launch stage通过参数修改间接"传达"资源信息给调度器，而非直接打开调度器接口。

从系统架构角度拆解术语：

Semantic gap在三阶段GPU执行流程中的体现：

```
Information flow across the three stages:

Stage 1: Launch (CPU-side, full information available)
  Kernel knows:  blocksize, gridDim, sharedMem, registers_per_thread
  Profiler knows: rFP32, rFP64, rINT32, rLDST, rSFU, rTensor, tLaunch
  μShare adds:   half-plus blocksize, time-shifted relaunch
  ↓ Messages passed to Stage 2: ONLY blocksize, gridDim, sharedMem, stream
  ↓ SEMANTIC GAP: 6-way resource utilization profile NOT passed

Stage 2: Schedule (GPU hardware, limited information)
  Dispatch unit knows: blocksize, SM.available_threads, SM.available_smem
  Dispatch unit DOES NOT know: kernel's dominant resource type
  Scheduling decision: SM.available_threads >= blocksize → assign
  ↓ Consequence: blocks with identical demand stacked → resource waste

Stage 3: Execute (GPU hardware)
  SM executes blocks independently
  SM DOES NOT know: what other blocks are co-scheduled
  Result: "1 more, 5 less" utilization pattern within each SM

μShare's approach to bridging the gap:
  - CANNOT modify Stage 2 or Stage 3 (closed-source hardware)
  - CAN modify Stage 1 parameters (blocksize, launch timing)
  - Strategy: make blocksize "large enough" to prevent same-kernel stacking
    → indirect signal to dispatch unit: "this SM can't fit another block like me"
    → remaining threads become available for complementary kernel blocks
```

术语一般如何实现？如何使用？

Semantic gap的解决方案对比：
1. **Open-sourcing the scheduler (侵入式硬件修改)**：CCWS/Prema/PriorityRR在simulator层面修改GPU scheduler，使scheduler感知kernel resource demand并做resource-coupled scheduling。问题：无法在真实NVIDIA GPU上部署
2. **Kernel fusion (侵入式kernel修改)**：Tacker/T3/Rammer/COMBO将互补kernel合并为一个，消除semantic gap（因为fused kernel内部已包含多个resource type）。问题：需要重写kernel代码、增加开发复杂度、不适用于闭源kernel（cuDNN/cuBLAS）
3. **μShare的indirect bridging (非侵入式)**：在launch stage通过blocksize修改间接编码资源信息——大的blocksize意味着"我是heavy user of resource X"，dispatch unit的left-over调度自然产生scattered co-location。信息通道：blocksize参数 → dispatch unit left-over check → SM资源分配 → co-location outcome

涉及论文标题：
- μShare: Non-Intrusive Kernel Co-Locating on NVIDIA GPUs
