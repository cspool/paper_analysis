## Amazon Trainium AI Accelerator

术语是什么？
Amazon Trainium 是 AWS 为机器学习训练和推理工作负载设计的第一代和第二代 AI 加速器芯片。Trainium 1 和 Trainium 2 分别搭载于 trn1.32xlarge 和 trn2.48xlarge EC2 实例，每个芯片包含多个计算核心（cores），每核心内部包含三类并发执行的硬件计算引擎：Tensor Engine（执行矩阵乘法，Trainium 1: 23.75 TFLOPS/core, Trainium 2: 19.75 TFLOPS/core）、Vector Engine（执行向量运算如激活函数、element-wise 操作，Trainium 1: 286.8 GFLOPS/core, Trainium 2: 550.0 GFLOPS/core）和 Scalar Engine（执行标量运算和控制流）。芯片的软件管理片上内存层次结构包括 SBUF（State Buffer，每 partition 限 192KB，kernel 可显式管理 tile 的存放位置）和 PSUM（Partial Sum Buffer，free dimension 限 512 elements），通过 DMA 引擎与 HBM（Trainium 1: 440.2 GB/s/core, Trainium 2: 640.0 GB/s/core）交换数据。

从硬件架构角度拆解术语：
Trainium 单核的硬件执行模型：
```
                 ┌──────────────────────────────────────┐
                 │           Trainium Core               │
                 │                                      │
  HBM ──DMA──────┤  ┌──────────┐  ┌──────────┐         │
  (High BW       │  │   SBUF   │  │  PSUM    │         │
   Memory)       │  │(State    │  │(Partial  │         │
                 │  │ Buffer)  │  │Sum Buffer│         │
                 │  └────┬─────┘  └────┬─────┘         │
                 │       │             │                │
                 │  ┌────▼─────────────▼───────────┐    │
                 │  │    Tensor Engine              │    │
                 │  │  - nc_matmul(stationary, moving)│  │
                 │  │  - 128x128 stationary +        │    │
                 │  │    128x512 moving = optimal    │    │
                 │  │  - reads SBUF, writes PSUM     │    │
                 │  └────────────────────────────────┘    │
                 │  ┌────────────────────────────────┐    │
                 │  │    Vector Engine                │    │
                 │  │  - activation functions          │    │
                 │  │  - element-wise ops              │    │
                 │  │  - transpose, copy               │    │
                 │  └────────────────────────────────┘    │
                 │  ┌────────────────────────────────┐    │
                 │  │    Scalar Engine                │    │
                 │  │  - loop control, address calc   │    │
                 │  │  - semaphore operations          │    │
                 │  └────────────────────────────────┘    │
                 └──────────────────────────────────────┘

  关键约束:
  - Tensor/Vector/Scalar engines 并发运行
  - nc_matmul 的 stationary operand: partition_dim <= 128, free_dim <= 128
  - nc_matmul 的 moving operand: partition_dim <= 128, free_dim <= 512
  - Tensor 默认以第一维为 partition dimension (可用 par_dim 显式标注)
  - Free dimension (partition dimension 右侧) 上的元素顺序读写
```

Roofline 性能分析模型基于三类 engine 的并发性：最优 latency T = max(Traffic_Min / PeakBW, FLOPs_MM / Peak_MM, FLOPs_Vec / Peak_Vec)，分别对应 memory bandwidth bound、tensor engine compute bound、vector+scalar engine compute bound 三种场景的瓶颈。

术语一般如何实现？如何使用？
AWS 通过 EC2 实例提供 Trainium：trn1.32xlarge（16 个 Trainium 1 芯片，共 32 cores per chip，512 cores total）、trn2.48xlarge（24 个 Trainium 2 芯片）。开发者通过 Neuron SDK（包含 NKI、Neuron Compiler、Neuron Profile 等工具）编程和使用。Neuron Profile 提供详细的硬件级性能指标（HBM read/write bytes, SBUF/PSUM access bytes, spill bytes, engine utilization, hardware FLOPs 等）。Trainium 被认为是 "emerging AI accelerator" 的代表——其架构与 GPU 显著不同，导致开发者缺乏优化直觉和成熟的优化 recipe，这正是 AccelOpt 试图解决的问题。

涉及论文标题：
- AccelOpt: A Self-Improving LLM Agentic System for AI Accelerator Kernel Optimization
