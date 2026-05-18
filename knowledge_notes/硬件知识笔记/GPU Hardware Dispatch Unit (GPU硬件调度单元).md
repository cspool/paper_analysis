## GPU Hardware Dispatch Unit (GPU硬件调度单元)

术语是什么？

GPU Hardware Dispatch Unit（GPU硬件调度单元）是NVIDIA GPU中负责block-to-SM调度的专用硬件模块，实现在GPU的GigaThread Engine或equivalent硬件中。它接收来自CUDA streams的kernel block队列，根据每个SM的可用资源（thread capacity、shared memory、registers）和block的资源需求，将block分配到合适的SM执行。Dispatch unit采用left-over scheduling策略：遍历SM列表，将block分配到第一个满足资源条件的SM。该模块是NVIDIA GPU中完全闭源的硬件组件，不暴露调度策略配置接口，因此无法直接修改其行为。μShare的核心挑战正是在dispatch unit闭源的前提下，通过修改kernel launch参数间接影响其调度决策。

从硬件架构角度拆解术语：

Dispatch unit在GPU执行流程中的位置和作用：

```
Kernel Lifecycle三阶段中的Dispatch Unit位置：

Stage 1: Launch (CPU side, 可控)
  PyTorch/TensorRT → CUDA Driver API → Command Processor
  ↓
Stage 2: Schedule (GPU hardware, 闭源) ← Dispatch Unit在此
  GigaThread Engine:
    ├── 接收CUDA stream中的kernel blocks队列
    ├── 读取每个SM的available resources (threads/smem/regs)
    ├── 执行left-over scheduling:
    │   for each pending block:
    │     for each SM (0..83 for A40):
    │       if SM.free_threads >= block.blocksize:
    │         dispatch(block, SM)
    │         SM.free_threads -= block.blocksize
    │         break
    ├── 当kernel线程数 > GPU总capacity: blocks排队pending queue
    └── 当SM资源释放(block完成): retry pending blocks
  ↓
Stage 3: Execute (GPU hardware)
  SM:
    ├── Warp Scheduler (per-SM): 选择warps from dispatched blocks
    ├── Dispatch Unit (per-SM): 发射warp指令到执行单元
    └── Execution Units: FP32/FP64/INT32/Tensor/SFU/LDST
```

μShare利用dispatch unit的left-over策略来实现scattered co-location：
- Half-plus blocksize > SM_thread_capacity/2 → dispatch unit无法将同kernel两个block放入同一SM
- 剩余threads仍满足left-over条件 → dispatch unit自然选择不同kernel的小block
- 整个过程不需要修改dispatch unit硬件逻辑

术语一般如何实现？如何使用？

NVIDIA GPU dispatch unit的硬件实现细节未公开，但可通过行为观察推断：
1. **GigaThread Engine**：NVIDIA GPU的全局调度硬件，从Fermi架构开始引入，管理跨SM的block分发
2. **Left-over scheduling**：μShare通过CUDA inline PTX读取SM ID register和clock counter register，实际测量6,802次kernel执行的block placement和timing，确认了left-over调度行为
3. **Warp Scheduler vs Dispatch Unit**：注意区分——global dispatch unit（论文讨论的，block→SM）和per-SM warp scheduler（warp→execution units）。μShare只影响前者（通过blocksize），不涉及后者（纯硬件，无法从launch stage影响）
4. **闭源限制**：由于dispatch unit闭源，现有intra-SM sharing work（CCWS/Prema/PriorityRR）只能通过simulator验证修改，无法在真实NVIDIA GPU上部署。μShare的non-intrusive approach是唯一在真实GPU上可行的方案

涉及论文标题：
- μShare: Non-Intrusive Kernel Co-Locating on NVIDIA GPUs

