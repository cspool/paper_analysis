## TileLang（可组合的 Tiled 编程模型）

术语是什么？通过联网搜索让回答具体和精准。

TileLang 是一种面向 AI 系统的 composable tiled programming model，通过将调度空间（thread binding、layout、tensorize、pipeline）与数据流解耦，让用户专注于 kernel 数据流描述而编译器处理硬件优化。它提供类 Python 的 tile-level 编程语法，按用户层次分为三级：Beginner（高级表达式+自动调度）、Developer（tile-level 编程，类似 Triton）、Expert（low-level CUDA-like 控制）。TileLang 内置 Layout Inference Engine（自动推导内存布局：Strict→Common→Free 三级推断）和 Pipeline Inference（用户仅指定 num_stage，编译器自动分析依赖并调度 stage）。性能方面可达 SOTA kernel 水平（如 ~50 行 TileLang 代码实现 DeepSeek MLA，达到手写 CUDA 95%+ 性能），支持 CUDA GPUs（H100/A100/V100）、ROCm GPUs（MI300/MI250）和国产加速器。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。

TileLang 在编译框架中的作用类似于 "high-level IR + backend code generator"：(1) 用户用 TileLang 描述 kernel dataflow（tile-level 操作，如 `T.load(A[ti, tj])`、`T.gemm(A, B)`）；(2) TileLang 的 Layout Inference 自动推导各 tensor 的最优内存布局（row-major/column-major/swizzled）；(3) Pipeline Inference 根据 num_stage 和 dependency 自动插入 async copy 和 barrier；(4) TileLang 编译器将 tiled program lowering 为目标后端代码（CUDA C++/PTX via CUTE、ROCm HIP、TPU/NPU 指令等）。在 MetaAttention 中，TileLang 作为 NVIDIA 和 AMD 后端的 backend framework 之一。MetaAttention runtime 的 kernel template 用 TileLang 编写，scheduling plan 中的 tile/memory/pipeline 配置注入 TileLang program 的参数，TileLang 编译器负责生成最终的可执行 kernel。MetaAttention 也使用 CUTE（CUTLASS 的 CuTe 抽象库）作为 NVIDIA 的另一个后端，两个 backend 在不同 operator pattern 下提供性能选择。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

TileLang 的开源生态包括：(1) 核心项目 TileLang（https://github.com/tile-ai/tilelang，arXiv:2504.17577）；(2) TileScale（https://github.com/tile-ai/tilescale）——分布式扩展，通过 Hierarchical Distributed Architecture (HDA) 抽象虚拟化多 GPU/多节点/跨芯片系统为统一 "mega-device"；(3) TileOPs（https://github.com/tile-ai/TileOPs）——TileLang 算子库。TileLang 与 Triton 的区别：Triton 通过 block-level programming + JIT compilation 降低 CUDA 门槛但 loss 部分细粒度控制，TileLang 则通过三层用户接口在易用性和可控性之间提供更宽的谱系，且通过 Layout/Pipeline Inference 减少手动优化工作。

涉及论文标题：
- MetaAttention: A Unified and Performant Attention Framework Across Hardware Backends
- Accelerating Sparse Transformer Inference on GPU

---
