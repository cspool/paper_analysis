## Template-Based Lowering

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Template-based lowering 是一种编译器代码生成策略，结合了手写高性能 kernel 模板的性能优势和编译器的灵活性。核心思路是：预先手写经过手工优化的 kernel 模板（包含通用的计算模式如 tiling、online softmax、内存管理），然后在模板中预留代码注入点（code injection points），编译器自动将用户定义的高层操作（如 score modification 函数）翻译为低级代码块并注入到模板的预留位置。这避免了全自动编译器（如 TVM、Mirage）在生成 competitive fused attention kernel 时的主要困难——online softmax 支持、双 GEMM fusion（QK^T + PV）、block sparsity 处理——同时保留了手写 kernel 的核心性能优化。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
在 FlexAttention 中，template-based lowering 的具体流程：

1. **模板定义**：手写 3 个 Triton attention kernel 模板——forward、backward、decoding。每个模板内包含：
   - 在线 softmax（online rescaling logic：m = max(m, m_new); O = diag(exp(m_old - m)) * O + diag(exp(m_new - m)) * P_new @ V_new）
   - Tiled QK^T GEMM 和 PV GEMM（双重矩阵乘法融合）
   - GPU occupancy 管理（register/shared memory 预算分配）
   - GQA（Grouped Query Attention）支持

2. **注入点标记**：模板内标记两个代码注入点——
   - mask_mod 注入点：在 QK^T GEMM 后、softmax 前，对 Partial Block 逐元素执行 mask（设为 -inf）
   - score_mod 注入点：在 QK^T GEMM 后、softmax 前，对所有 visible block 逐元素执行 score modification

3. **代码生成与注入**：TorchInductor 将捕获的 score_mod/mask_mod PyTorch 子图翻译为 Triton kernel 代码片段，运行时动态拼接到模板的注入点。

4. **优势**：模板捕获了 attention kernel 的通用优化（online softmax、tiling、GQA），而注入的代码仅处理 attention 变体独有的 element-wise 差异，因此性能接近手写 kernel。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Template-based lowering 的关键实现要素：
- 模板用 Triton 语言编写（Python-embedded DSL for GPU programming），编译到 NVIDIA GPU 的 PTX
- TorchInductor 作为 lowering 引擎，负责 PyTorch FX graph → Triton IR → Triton code 的翻译
- 模板中的注入点通过 Python string formatting 或 Triton JIT compilation 实现动态拼接
- backward pass 的模板额外通过 torch.autograd 自动推导 score_mod/mask_mod 的梯度

涉及论文标题：
- Flex Attention: A Programming Model for Generating Optimized Attention Kernels
