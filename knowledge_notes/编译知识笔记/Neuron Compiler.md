## Neuron Compiler

术语是什么？
Neuron Compiler 是 AWS 为 Trainium/Inferentia AI 加速器开发的编译器工具链，负责将高层 ML 框架（PyTorch/JAX）的 operator 图或 NKI kernel 源码编译为 Trainium 硬件可执行的 NEFF（Neuron Executable File Format）文件。编译器执行多项关键优化：(1) 自动将高层 operator 映射为 NKI kernel 调用序列；(2) 编译 NKI Python 源码为 Trainium ISA 指令；(3) 自动插入 transpose 和 partition broadcast 操作以解决 memory layout 冲突（这些操作使用 Tensor Engine 的 matmul 指令实现数据搬移）；(4) 管理 SBUF 和 PSUM 片上内存分配；(5) 调度 DMA 引擎在 HBM 和片上内存之间传输数据。在 AccelOpt 的上下文中，Neuron Compiler 作为 "执行后端"——generated kernel 需通过它编译才能在 Trainium 上运行，且其自动插入的 layout 变换会影响 profiling 数据。

从编译框架角度拆解术语：
Neuron Compiler 在 AccelOpt 优化循环中的位置：
```
NKI Kernel 源码 (由 AccelOpt Executor 生成)
    │
    ▼
Neuron Compiler (nki compiler frontend + backend)
    │  1. 解析 Python/NKI 语法
    │  2. 类型推断 + tensor shape/layout 推导
    │  3. 自动插入 transpose/partition broadcast
    │     (可能在 profile 中引入 hardware_flops 包含 transpose_flops)
    │  4. 分配 SBUF/PSUM —— 若分配失败则触发 spilling
    │  5. 生成 NEFF
    ▼
NEFF ──► Trainium Hardware ──► Neuron Profile 数据
    ▲                               │
    │  编译时间 1.59-31.29s           │
    └───────────────────────────────┘
    (仅测量执行时间, 不包含编译时间)
```

Neuron Compiler 的一个关键行为：它自动插入的 data movement matmul 指令（transpose/partition broadcast）会计入 `hardware_flops` 指标。因此 `mm_arithmetic_intensity = (hardware_flops - transpose_flops) / (hbm_read + hbm_write)` 需要减去 transpose_flops 才能准确反映真正的 matmul compute intensity。这也是 AccelOpt prompt 中需要解释 profiling 术语的原因。

术语一般如何实现？如何使用？
Neuron Compiler 通过 `torch-neuronx`（PyTorch 集成）或 `neuronx-cc`（直接编译）使用。对于 NKI kernel，通过 `@nki.jit` 装饰器触发 JIT 编译。编译时间：AccelOpt 优化后 kernel 编译时间 1.59-31.29s（mean 7.38s），baseline kernel 编译时间 1.77-30.61s（mean 8.13s）。由于 kernel 在 ML pipeline 中一次编译多次复用，编译时间不影响 kernel 质量评估。当前版本（v2.20）的 Neuron Compiler 尚不支持所有 NKI API 特性（处于 beta 阶段）。

涉及论文标题：
- AccelOpt: A Self-Improving LLM Agentic System for AI Accelerator Kernel Optimization
