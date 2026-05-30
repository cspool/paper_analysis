## Ascend NPU Architecture (AI Core / AI Vector / Da Vinci)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Ascend NPU 是华为自主研发的 AI 加速处理器，基于 Da Vinci（达芬奇）架构。新一代 A2 系列（如 EPD-Serve 使用的 Atlas 800I A2 服务器）采用分离式 AI Core 设计，核心计算单元分为三类：（1）**AI Core（AIC/矩阵核/Cube Unit）**：负责大规模矩阵乘法运算（16×16×16 的 MAC 阵列），是主要算力来源，执行 MatMul 等算子；（2）**AI Vector（AIV/向量核/Vector Unit）**：负责向量运算（激活函数、Element-wise 操作、AllReduce 通信等），256 位 SIMD 向量运算，比 Cube Unit 灵活但算力低；（3）**Scalar Unit**：负责程序流程控制和标量数据运算。三者采用顺序取指令、并行执行的调度方式，各自拥有独立的指令队列（S/V/M/MTE），支持并发执行。存储层次包括 Unified Buffer（UB, 256KB）和 L1/L2 Cache，HBM 片上内存（Atlas 800I A2 为每 NPU 64 GB）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

EPD-Serve 利用 Ascend NPU 的 AI Core/AI Vector 异构计算单元实现 operator-level 空间复用（Figure 6）：

```
            Ascend NPU (Atlas 800I A2, 64GB HBM)
┌─────────────────────────────────────────────────────┐
│                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────┐  │
│  │  AI Core      │  │  AI Vector   │  │  Scalar   │  │
│  │  (Cube Unit)  │  │  (Vector)    │  │  Unit     │  │
│  │               │  │              │  │           │  │
│  │  MatMul       │  │  AllReduce   │  │  Control  │  │
│  │  Attention    │  │  LayerNorm   │  │  Flow     │  │
│  │  Conv2D       │  │  Activation  │  │           │  │
│  │  256 MACs/    │  │  256-bit     │  │  1-2 ops/ │  │
│  │  cycle        │  │  SIMD        │  │  cycle    │  │
│  └──────┬───────┘  └──────┬───────┘  └────┬─────┘  │
│         │                 │               │        │
│         └─────────────────┴───────────────┘        │
│                           │                         │
│                    ┌──────▼──────┐                  │
│                    │  Unified    │                  │
│                    │  Buffer     │                  │
│                    │  (256KB)    │                  │
│                    └──────┬──────┘                  │
│                           │                         │
│                    ┌──────▼──────┐                  │
│                    │    HBM      │                  │
│                    │   (64 GB)   │                  │
│                    └─────────────┘                  │
└─────────────────────────────────────────────────────┘

算子互补示例（Figure 6 左子图）:
  MatMul:    AI Core 密集型, 低数据搬运
  AllReduce: AI Vector 密集型, 高数据搬运
  → 当 MatMul 在执行时, AllReduce 可以用 AI Core 空闲周期
  → 当一个阶段等待通信(P-D传输), 另一阶段利用空闲单元执行算子
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Ascend NPU 的软件栈：CANN（Compute Architecture for Neural Networks）——对标 NVIDIA CUDA 的异构计算架构，包含 Ascend C 编程语言（SPMD 并行范式）、图编译器（GE/Graph Engine）、算子库（Ascend Operator Library）、Runtime 和调优工具链。开发者通过 PyTorch Ascend Adapter 或 TensorFlow Ascend Adapter 使用 NPU，无需直接编写 Ascend C。实际使用：华为 Atlas 系列服务器（800I A2 用于训练和推理）、Ascend 910 系列芯片广泛部署于华为云和私有数据中心。EPD-Serve 在 Atlas 800I A2 上运行，通过标准 PyTorch/Ascend 接口执行模型推理，不直接修改底层 CANN 或算子。

涉及论文标题：
- EPD-Serve A Flexible Multimodal EPD Disaggregation Inference Serving System On Ascend
