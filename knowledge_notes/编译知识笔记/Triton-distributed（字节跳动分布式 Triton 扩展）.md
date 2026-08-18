## Triton-distributed（字节跳动分布式 Triton 扩展）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Triton-distributed 是字节跳动（ByteDance）对 Triton 语言的分布式扩展（arXiv:2504.19442），把通信（collective）提升为 Triton 语言的一等抽象，使开发者能在单个 Triton kernel 中声明式地编写跨设备（分布式）计算并自动生成重叠的 kernel：分布式张量（distributed tensor）+ 通信原语（如 distributed_linear、collective）由编译器自动插入通信调度（overlapping kernels），避免手写 NCCL/NVSHMEM 拼接。Tetris（ISCA'26）的推理后端构建于 PyTorch 与 Triton-distributed 之上（并复用部分 vLLM 组件），用于实现其 CDSP 推理引擎的分布式计算（prefill ring attention 等跨实例算子的 kernel 级表达与通信-计算重叠）。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
```
# Triton-distributed 的编译-执行流程（Tetris 推理后端视角）
源码: 用 @triton.jit 写 kernel，把跨设备算子声明为 distributed 张量操作
  → Triton-distributed 编译器分析分布式布局（shard 维度、group）与通信依赖
  → 自动插入/调度 collective（与本地计算重叠），生成各 rank 的 CUDA kernel
  → GPU 执行：每个 rank 跑本地 kernel 片段 + 编译器生成的通信 kernel（重叠执行）
```
Annotations: 编译器承担"何时通信、与哪段计算重叠"的调度，替代手写 NVSHMEM/NCCL 拼接；用户只写单卡风格 kernel + 分布式张量声明。
在 Tetris 中：与 Flash Attention zigzag ring 扩展、NVSHMEM/NCCL 传输配合构成推理后端；论文未明确说明 Triton-distributed 具体负责哪些算子（论文只列为其后端基础之一）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：Triton-distributed 开源扩展（ByteDance 团队，arXiv:2504.19442，基于 Triton 编译器，支持 NVIDIA GPU 与 NVLink/InfiniBand）；使用方式——加载 distributed runtime 后以分布式张量编写 kernel，编译器生成带通信的 kernel 并支持通信-计算重叠。在 Tetris 中作为推理后端（与 PyTorch、vLLM 组件、Flash Attention、NVSHMEM、Flash Decoding、CUDAGraph、NCCL 共同构成 ~17.5K 行实现）。Web 证据：Triton-distributed 论文（arXiv:2504.19442）描述语言扩展与 overlapping kernel 生成。

涉及论文标题：
- Tetris: Efficient Long-context LLM Serving with Chunkwise Dynamic Sequence Parallelism
