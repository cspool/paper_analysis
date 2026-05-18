## Asynchronous Data Property Inference with Tensor Fingerprint（异步数据属性推断与张量指纹）

术语是什么？通过联网搜索让回答具体和精准。

Asynchronous Data Property Inference是Difflow提出的用于隐藏调度overhead的技术：在dTask尚未执行时，通过符号化数据属性表达式和tensor fingerprint技术推断每个dTask的输入/输出数据属性（如哪些tensor冗余、哪些维度ragged），使scheduler能够在计算结果实际产生之前就进行后续dTask的batching planning，从而实现调度与执行的异步重叠。核心洞察：dGraph执行确定性计算，输出完全由输入决定——因此可以从输入指纹直接推导输出指纹，无需operator-by-operator重算。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

Tensor fingerprint与异步推断运转流程：

```
// === Tensor Fingerprint计算 ===
1. Function ComputeFingerprint(tensor):
2.     return lightweight_hash(tensor_elements)  // O(n) time, n = num_elements

// === 输出指纹传播 (利用dGraph DFG确定性) ===
3. // 编译时: 通过data-flow equation确定每个output依赖哪些input
4. // 运行时: 从input fingerprints直接推导output fingerprint:
5.     FP[output_i] = Φ(FP_g, FP[input_dep_{i,0}], ..., FP[input_dep_{i,n}])
6.     // FP_g: dGraph唯一标识符
7.     // Φ: operand-commutative hash function
8.     // input_dep: 影响output_i的输入集合

// === 属性表达式求值 ===
9. Function InferOutputProperties(dGraph, input_properties):
10.    return evaluate(property_expressions[dGraph], input_properties)
11.    // e.g., output_redundancy = α ∧ β (α=prompt redundant, β=image redundant)

// === 异步调度流程 ===
12. Scheduler thread:
13.    while requests arrive:
14.        infer properties for dTasks whose inputs are ready
15.        generate batching plan via DP
16.        dispatch batches to executor queue (non-blocking)
17. Executor thread:
18.    while queue not empty:
19.        execute dEngine batch → output properties written to shared state
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Difflow中实现：(1) 轻量hash函数（复杂度与tensor元素数线性）；(2) 应用/用户可直接标记correlative requests中的duplicate inputs（避免大tensor hashing开销）；(3) 利用dGraph DFG的data-flow equation从输入指纹直接推导输出指纹——因为dGraph是确定性计算；Φ使用operand-commutative hash确保相同计算结果有相同指纹。大调度窗口下GPU idle time <5%（含cold start），调度开销<10% dEngine runtime。

涉及论文标题：
- Difflow: A Data-Characteristic-Aware Serving System for Diffusion Models
