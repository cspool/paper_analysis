## NKI (Neuron Kernel Interface)

术语是什么？
NKI（Neuron Kernel Interface）是 Amazon AWS 为 Trainium/Inferentia AI 加速器设计的 Python-embedded kernel 编程语言。它允许开发者直接编写底层计算 kernel，精确控制数据在 HBM（High Bandwidth Memory）、SBUF（State Buffer）、PSUM（Partial Sum Buffer）之间的移动，以及 Tensor Engine、Vector Engine、Scalar Engine 上的指令调度。NKI 的核心编程模型基于 tile-based 计算：kernel 以 tile 为基本数据单元，使用 `nl.ndarray` 定义多维张量并标注 buffer 位置（`nl.sbuf`、`nl.psum`、`nl.hbm`），通过 `nl.affine_range` 创建无依赖并行循环（free dimension 上的迭代可被 hardware parallel execution），通过 `nl.sequential_range` 创建串行循环。关键 API 包括 `nki.isa.nc_matmul`（Tensor Engine 矩阵乘法，计算 `transpose(stationary) @ moving`）、`nki.isa.nc_transpose`（利用 Tensor Engine 做转置）、`nki.isa.activation`（支持 sigmoid/rsqrt/silu 等激活函数融合）以及 `nl.load`/`nl.store`（SBUF 与 HBM 之间 DMA 数据传输）。

从编译框架角度拆解术语：
NKI 在 AWS Neuron 软件栈中的位置和编译流程：
```
高层 ML Framework (PyTorch/JAX)
    │
    ▼
Neuron Compiler (torch-neuronx / neuronx-cc)
    │  将 ML operator 图编译为 NKI kernel 调用序列
    │  或直接生成 NKI kernel 源码
    ▼
NKI Kernel (Python-embedded 源码)
    │  AccelOpt 在此层介入：生成优化的 NKI kernel
    │
    ▼
Neuron Compiler (nki compiler pass)
    │  将 NKI kernel 编译为 Trainium ISA
    │  - 自动插入 transpose/partition broadcast (解决 memory layout 冲突)
    │  - 分配 SBUF/PSUM 空间
    │  - 调度 DMA 传输
    ▼
Trainium Hardware 可执行文件 (NEFF)
```

NKI 的编程约束（编译期检查）：(1) `affine_range` 循环的各迭代间不能写同一内存位置（output dependency 检查），除非用 `sequential_range`；(2) SBUF 每 partition 不超过 192KB；(3) PSUM free dimension 不超过 512；(4) partition 维度的 partition 数不超过 128；(5) tensor indexing 只能纯 basic indexing 或纯 advanced indexing，不能混用；(6) if/else/for 控制块内定义的 tensor 不能被外部引用。

术语一般如何实现？如何使用？
NKI 以 Python 包形式提供（`nki` 和 `nki.isa`），开发者通过 `import nki.language as nl` 和 `from nki import isa as nisa` 使用。Kernel 函数用 `@nki.jit` 装饰器标注以触发 JIT 编译。典型 NKI kernel 结构：(1) 用 `nl.ndarray` 分配 SBUF tile，(2) 用 `nl.load` 从 HBM 加载数据到 SBUF，(3) 用 `nisa.nc_matmul` 等指令在 Tensor/Vector Engine 上计算，(4) 用 `nl.store` 将结果写回 HBM。NKI 是 beta 阶段的相对新语言（v2.20），开发者缺乏成熟的优化 recipe——这正是 AccelOpt 试图解决的问题。

涉及论文标题：
- AccelOpt: A Self-Improving LLM Agentic System for AI Accelerator Kernel Optimization
