## CUDA Context

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

CUDA Context 是 NVIDIA CUDA 编程模型中每个 GPU-using task 的虚拟地址空间容器，类似于 CPU 进程的地址空间概念。一个 context 封装了该 task 在 GPU 上的所有资源：GPU 内存分配（device memory）、CUDA module（编译后的 GPU 代码）、stream 和 event 等。Context 的创建是开销较大的操作（Bakita & Anderson 实验显示 context 初始化产生约 100ms 的 compute engine 干扰，Fig.6），因此通常每个 task 创建一个 context 并在整个 task 生命周期内复用。Context 在 GPU 硬件调度管线中映射为一个 TSG（Time-Slice Group），TSG 内的所有 channel 共享该 context 的虚拟地址空间。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Context 在 GPU 调度中的角色：

```
CPU进程/Task → CUDA Context (per-task, per-GPU) → TSG
  ├── GPU Memory Allocations (cudaMalloc)
  ├── Streams (cudaStreamCreate)
  │   └── Kernel Launches + Memory Copies
  ├── Events (cudaEventCreate)
  └── Modules (loaded GPU code)

GPU HW调度视角:
  Context = TSG on runlist → Round-robin timeslicing between contexts
  Context内的所有streams → 共享TSG的channel pool
  Context初始化 → 产生compute engine干扰(~100ms, Fig.6)
                  → 影响co-running task的实时性
```

关键约束和特性（Bakita & Anderson 发现）：
- **One-to-one task-to-context mapping**：标准使用模式，多 context per task 可能但 discouraged
- **Context 初始化干扰**：创建 CUDA context 会产生约 100ms 的 compute engine 干扰（Fig.6），影响 co-running 实时 task
- **MPS 特例**：NVIDIA Multi-Process Service (MPS) 使多应用共享一个 MPS-created context 作为 subcontext，自 Volta 架构起改变 context 语义

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Context 通过 CUDA runtime API 隐式创建（首次调用任何 CUDA API 时自动创建 primary context）或显式通过 CUDA Driver API 的 cuCtxCreate() 创建。在实时系统中，推荐在 task 初始化阶段创建 context（而非运行时临界路径），以避免 context 初始化对 co-running task 的干扰。MPS 在 GPU serving 场景中广泛使用（多 client 共享 GPU），但其 context/subcontext 架构改变了 Bakita & Anderson 规则的适用方式。

涉及论文标题：
- Demystifying NVIDIA GPU Internals to Enable Reliable GPU Management
