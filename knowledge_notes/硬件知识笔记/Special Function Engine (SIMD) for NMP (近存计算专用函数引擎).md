## Special Function Engine (SIMD) for NMP (近存计算专用函数引擎)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Special Function Engine 是 Stratum NMP 中每个 PU 集成的 256-way SIMD 处理单元，负责执行 LLM 推理中 GPU 不擅长的非线性算子（Softmax, SiLU, GeLU 等），避免这些 element-wise 操作成为 NMP pipeline 的瓶颈。引擎包含 vector register file（存储向量操作数）、scalar register file（存储标量中间结果）和 multiple arithmetic units（支持指数、除法、乘法等原语）。通过将复杂函数（如 Softmax）分解为简单的基本操作序列（max reduction, exp, sum, division），在 SIMD 引擎内部最大化数据复用——操作数和中间结果在 vector/scalar register files 内循环，无需频繁访问 shared memory 或 DRAM。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
Special Function Engine 在两个场景中的执行：
1. **Expert Processing**（SiLU + Hadamard）：Tensor Core 输出 Z_1（GeMM1 result）和 Z_2（GeMM2 result）→ Special Function Engine 执行 SiLU(Z_1) = Z_1 ⊗ sigmoid(Z_1) → Hadamard product: SiLU(Z_1) ⊙ Z_2 → 输出 X_2 作为 GeMM3 输入。与 GeMM2 的 tensor core 计算重叠执行（因无数据依赖）。
2. **Attention Processing**（Softmax）：Head-level parallelism 下，每个 PU 独立计算 local Softmax：(a) local max = row_max(Scores_slice) → ring scalar exchange → global max；(b) local exp_sum = Σ exp(Scores_slice - global max) → ring scalar exchange → global exp_sum；(c) Softmax(Scores) = exp(Scores - global max) / global exp_sum。Softmax 分解为 3 步，Scalar exchange 仅传输标量（每 PU 2 个值），与另一 head 的 MatMul 交错执行隐藏延迟。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Stratum 的 Special Function Engine 区别于 AttAcc 和 Duplex 的 dedicated Softmax unit——Stratum 采用通用 SIMD 可编程架构，支持 instruction-level 编程而非固定函数硬件。这使得它不仅可以执行 Softmax，还能执行 SiLU, GeLU, weighted sum 等 MoE 特有的非线性操作。实现层面：SystemVerilog 实现 → Cadence Genus synthesis → 7nm ASAP7 PDK。关键设计权衡：通用 SIMD 的面积/功耗略高于专用硬件，但在 MoE serving 场景中提供了更好的灵活性（如支持 expert output weighted sum 的 on-the-fly 执行）。

涉及论文标题：
- Stratum: System-Hardware Co-Design with Tiered Monolithic 3D-Stackable DRAM for Efficient MoE Serving
