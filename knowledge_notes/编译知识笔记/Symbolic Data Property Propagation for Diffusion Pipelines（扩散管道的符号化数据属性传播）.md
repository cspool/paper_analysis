## Symbolic Data Property Propagation for Diffusion Pipelines（扩散管道的符号化数据属性传播）

术语是什么？通过联网搜索让回答具体和精准。

Symbolic Data Property Propagation是Difflow(ChituDiffusion)编译器中的核心技术：用symbolic boolean variables表示扩散pipeline中每个tensor的per-dimension数据属性（如某维度是否redundant），通过operator-specific propagation rules将属性沿DFG传播，最终推导每个operator的输出属性表达式。这使得编译器可以：识别哪些连续operators共享相同优化条件（→dGraph decomposition）、枚举dGraph可能的输入属性组合（→dEngine compilation）、剪枝不可满足或低收益的优化场景。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。

Symbolic propagation rules (Table 1) 示例：

```
// 输入tensor A: redundancy vector [a1, a2]  (per-dimension boolean)
// 输入tensor B: redundancy vector [b1, b2]

// Unary elementwise (ReLU, Tanh): output redundancy = [a1, a2]
// Binary elementwise (+, -, ×, ÷): output redundancy = [a1∧b1, a2∧b2]
// Batch Matmul [NHW]: output redundancy = [a1∧b1, a2, b3]
// Conv2D w/o padding [NCHW]: output redundancy = [a1, b1, a3∧a4∧b3∧b4, a3∧a4∧b3∧b4]
// Conv2D w/ padding [NCHW]: output redundancy = [a1, b1, N, N]  // padding destroys spatial redundancy

// ∧ = logical AND: 仅当两个输入在该维度都冗余时输出才冗余
```

完整传播流程：
```
// === 初始化 ===
1. for each pipeline input:
2.     if input is application-fixed (always same across requests): property = T (redundant)
3.     elif input varies per request: property = F (non-redundant)
4.     elif input may or may not be same (determined at runtime): property = symbolic variable (α, β, γ...)

// === Propagation ===
5. for each operator in topological order:
6.     output_property[op] = apply_propagation_rule(op.type, input_properties[op])
7.     // 例如 Conv2D: output[a3∧a4∧b3∧b4] 如果input spatial dims都冗余则output spatial dims冗余

// === Loop handling ===
8. for each denoising loop:
9.     unroll iterations until loop inputs stabilize
10.    // Diffusion loops converge within few steps (no nested/overlapping loops)

// === dGraph decomposition ===
11. group consecutive operators with identical output property expressions into dGraphs
12. // 相同output property → shared optimization conditions
```

以SDXL prompt embedding为例：prompt对于correlative requests冗余(α=T)→CLIP text encoder输出冗余→U-Net cross-attention的K/V projection输出冗余→整个cross-attention block的K/V参与计算可被识别为冗余。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Difflow实现中propagation rules以operator category为粒度定义（Table 1），支持用户为new operators自定义propagation rules。Application characteristics由开发者在pipeline定义时标注——哪些输入always same (fixed T)、哪些potentially same (symbolic variable)、哪些always vary (fixed F)。Symbolic variables用boolean algebra expression维护，输出属性expression可能包含AND组合（如α∧β）。Loop unrolling到stabilization通常只需few steps（diffusion loops不是嵌套的）。

涉及论文标题：
- Difflow: A Data-Characteristic-Aware Serving System for Diffusion Models

---
