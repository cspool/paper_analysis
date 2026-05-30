## GPU Grid Scheduler

术语是什么？
GPU Grid Scheduler 是 GPU 硬件中负责将 kernel 的 CTA（thread block）dispatch 到 SM 上执行的控制单元。在 NVIDIA GPU 中，Grid Scheduler 维护 SM 的 occupancy 信息（已分配多少线程/shared memory/寄存器），通过 round-robin arbiter 在 SM 间分配 CTA。当前 GPU 的 Grid Scheduler 采用 FIFO 非抢占策略：仅当前 kernel 的所有 CTA dispatch 完成后，才处理下一个 kernel——这导致不同 kernel 的 CTA 间几乎没有执行重叠。

从硬件架构角度拆解术语：
在 NVIDIA A100 中，Grid Scheduler 的工作流程：

```
Grid Scheduler 硬件结构:
  ┌─────────────────────────────────┐
  │  SM Occupancy Table              │
  │  SM_0: [threads=1024/2048,       │
  │         shmem=32KB/192KB,        │
  │         regs=16384/65536]        │
  │  SM_1: [...]                     │
  │  ...                             │
  │  SM_107: [...]                   │
  └─────────────────────────────────┘
         ↑
  ┌─────────────────────────────────┐
  │  Round-Robin Arbiter             │
  │  current_SM = (last_SM + 1) % N  │
  │  检查occupancy → dispatch CTA    │
  │  (FIFO: 仅处理当前kernel)       │
  └─────────────────────────────────┘
         ↑
  ┌─────────────────────────────────┐
  │  Kernel Queue (FIFO)             │
  │  K1(100 CTAs) → K2(50 CTAs) → ...│
  └─────────────────────────────────┘

Dispatch流程:
1. 从Kernel Queue头部取kernel K1
2. for each CTA in K1:
     round-robin遍历SM
     if SM有足够资源 (threads, shmem, regs):
        assign CTA to SM
     else: 尝试下一个SM
3. K1全部CTA dispatch后，从Kernel Queue出队
4. 返回步骤1处理K2
```

Kitsune 的 Modified Grid Scheduler（§4.2）：

```
修改后的Grid Scheduler:
  ┌─────────────────────────────────┐
  │  SM Occupancy Table              │
  │  (same as baseline)              │
  └─────────────────────────────────┘
         ↑                    ↑
  ┌──────────────────┐ ┌──────────────────┐
  │ SIMT Arbiter     │ │ Tensor Arbiter   │
  │ (round-robin)    │ │ (round-robin)    │
  │ selects SM for   │ │ selects SM for   │
  │ SIMT-heavy CTAs  │ │ TensorCore CTAs  │
  └──────────────────┘ └──────────────────┘
         ↑                    ↑
       根据kernel call header中的type metadata选择arbiter

  cudaPipeline API → spatial pipeline kernels带type标注
  kernel_Linear1: type=TENSOR → 通过Tensor Arbiter dispatch
  kernel_ReLU:   type=SIMT   → 通过SIMT Arbiter dispatch
  效果: 两个arbiter独立地将不同类型CTA配对到同一SM
```

关键修改：(a) 单个 arbiter → 两个 arbiter（SIMT/Tensor）；(b) kernel call header 新增 type metadata 字段；(c) CTA dispatch 逻辑不变（仍检查 occupancy），但通过分 arbiter 实现类型感知的 SM 分配。

术语一般如何实现？如何使用？
Grid Scheduler 是 GPU 硬件实现，对 CUDA 开发者透明。Kitsune 提出的是 concept-level modification——论文通过 NVArchSim (NVAS) simulator 模拟修改后的调度器行为，未做 RTL 实现。修改是 "modest" 的：(a) 复制现有 arbiter 逻辑；(b) 新增 type field 到 kernel call header；(c) dispatch 逻辑不变。Kitsune compiler 通过 cudaPipeline API 向调度器传递 type info。

涉及论文标题：
- Kitsune: Enabling Dataflow Execution on GPUs

---
