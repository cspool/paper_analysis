## MindSpore

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

MindSpore 是华为开源的深度学习框架（https://github.com/mindspore-ai/mindspore），支持 AI 模型的全场景开发、训练和推理。提供自动并行（auto-parallel）、动态图/静态图混合执行、以及针对 Ascend NPU 的原生优化。MindSpore 2.0 引入统一的计算图表达和融合优化能力。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。

在 LocMoE 的 PanGu-Σ MoE 训练中，MindSpore 提供以下编译和运行时能力：

1. **自动并行**：PanGu-Σ 基于 MindSpore 的 auto-parallel 框架，自动将模型分布在多 Ascend NPU 上，支持 TP (Tensor Parallel)、EP (Expert Parallel)、DP (Data Parallel) 等多维并行策略的组合。
2. **Group-wise All-to-All**：MindSpore 内置的 group-wise exchange 算法，将 All-to-All 通信按 TP 域和 EP 域拆分：每个 device 在 EP 域内负责部分 All-to-All 传输，然后通过 TP 域 All-Gather 同步。这将部分通信从低带宽的跨节点 EP 域转移到高带宽的 TP 域（HCCS）。
3. **通信-计算重叠**：MindSpore 支持 FFN 计算与 All-to-All 通信的切片重叠（slice-and-overlap），通过流水线化执行掩盖通信延迟。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

MindSpore 提供 Python API 和 C++ 后端。通过 CANN 异构计算架构对接 Ascend NPU 硬件。其图编译流程：Python 定义模型 → MindSpore 图 IR 构建 → 图优化（算子融合、内存复用、并行策略插入）→ 后端代码生成（Ascend AI Core 指令）。LocMoE 使用 MindSpore 2.0.0 + CANN 5.1.RC2.1 进行训练。

涉及论文标题：
- LocMoE: A Low-overhead MoE for Large Language Model Training
