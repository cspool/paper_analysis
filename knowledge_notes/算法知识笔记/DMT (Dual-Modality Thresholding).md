## DMT (Dual-Modality Thresholding)

术语是什么？
DMT是MoDES的模态感知expert skipping策略。基于发现：(1) vision token在FFN前后的余弦相似度更高（FFN对其更新幅度更小）；(2) 降低vision token的top-k对performance影响更小（vision expert冗余度更高）。DMT为text和vision token分别设置阈值τ_t和τ_v（τ_v < τ_t），对vision token更激进跳过expert。这是text-only LLM expert skipping工作未曾考虑的因素。

从算法pipeline角度拆解术语：
```
// Per token, per MoE layer:
τ = (token is text) ? τ_t : τ_v   // τ_v < τ_t
for each expert i in topk:
    if s_i < τ: skip Expert_i
```
效果：vision token的跳过率显著高于text token；深层跳过率显著高于浅层（对应α^{(l)}小的深层）。最佳阈值(τ_t*, τ_v*)通过Frontier Search在O(ND)时间找到，约束target skipping ratio ρ。

术语一般如何实现？如何使用？
τ_t和τ_v离线搜索确定。推理时在custom CUDA router kernel内通过branch-free masked comparison实现：`mask = (s_i < τ); topk[i] = mask ? M+1 : topk[i]`。DMT与GMLG叠加使用效果最强——单一组件在低跳过率时差异不大，但在高跳过率（>80%）下两者叠加的非线性增益显著（DMT+GMLG比仅Thresholding高~10%）。

涉及论文标题：
- MoDES: Accelerating Mixture-of-Experts Multimodal Large Language Models via Dynamic Expert Skipping

---
