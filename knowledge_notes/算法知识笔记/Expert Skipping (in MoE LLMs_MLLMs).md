## Expert Skipping (in MoE LLMs/MLLMs)

术语是什么？
Expert Skipping是一种训练无关的MoE模型推理加速技术：在推理时动态跳过对当前token贡献不足的冗余expert，仅激活真正重要的expert子集执行计算。与expert pruning（永久移除）不同，expert skipping是per-token的动态决策。核心机制：(1) 计算top-k候选expert对当前token的重要性分数；(2) 将分数低于阈值的expert标记为跳过；(3) 仅对保留的active expert执行FFN计算和加权聚合。MoDES识别出MLLMs场景下两个关键因素：层间贡献不均（shallow layer experts更重要，因error在后续层被放大）和模态行为差异（vision token的expert冗余度更高）。

从算法pipeline角度拆解术语：
以Qwen3-VL-MoE-30B-A3B-Instruct的l-th MoE层为例（128 experts, k=8）：
```
r = Router(x)                                    // [128] routing logits
π = softmax(r)                                   // routing probabilities
S = topk_indices(π, k=8)                         // 8 candidate experts

for each i in S:
    s_i = α̃^{(l)} · π_i                          // GMLG: global × local importance
τ = is_text(x) ? τ_t : τ_v                      // DMT: modality-specific threshold
active = {i ∈ S : s_i ≥ τ}                       // keep only important experts

y = Σ_{m ∈ active} π_m · Expert_m(x)             // weighted aggregation
```
跳过比例skip_ratio = 1 - |active|/k。MoDES在跳过88% expert时仍保持97.33%原始性能。

术语一般如何实现？如何使用？
现有方法：NAEE（routing probability阈值判定）、MC-MoE（attention-aware protection）、DiEP（differentiable pruning with expert similarity）。MoDES在此基础上引入GMLG（全局层重要性×局部routing概率）和DMT（text/vision分别设阈值），配合Frontier Search找最优阈值。Custom CUDA kernel内嵌branch-free masked comparison实现，跳过expert路由为sentinel ID并在dispatch/gather阶段过滤。适用所有MoE架构的LLMs/MLLMs推理加速，尤其在高跳过率（>80%）下优势显著。

涉及论文标题：
- MoDES: Accelerating Mixture-of-Experts Multimodal Large Language Models via Dynamic Expert Skipping

---
