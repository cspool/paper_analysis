## Cross-modal Fusion in LLM Layers (LLM 层的跨模态融合)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Cross-modal Fusion in LLM Layers 是多模态 LLM 中 visual tokens 和 text tokens 在 Transformer 层中通过 Self-Attention 进行信息交互的过程。AIM 通过消融实验发现：LLM 的不同层对 cross-modal fusion 的需求差异显著——早期层依赖 visual tokens 建立视觉-文本对齐（跨模态融合阶段），后期层主要进行 text-only reasoning（文本推理阶段），visual tokens 可被安全移除。

这一发现直接指导了 AIM 的 Scheduler 设计：l₁~l₂ 之间的层是 fusion→text-only 的过渡带。默认配置 l₁=14, l₂=22 对应 Qwen2-7B 的 28 层中前 50% 全保留 visual tokens，50%~79% 线性递减，79% 后全部移除。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**Cross-modal Fusion 在各层的角色变化**：

```
LLM Layer 1~13 (l < l₁): Cross-modal Fusion 阶段
    x = [v; t]  // visual + text tokens 拼接
    A = softmax(Q@K^T / sqrt(d_k))  // Attention 包含 visual↔text 交互
    // visual tokens 通过 Attention 将视觉信息传递给 text tokens
    // 结论：此阶段 visual tokens 必须全保留

LLM Layer 14~21 (l₁ ≤ l ≤ l₂): 过渡阶段
    // visual↔text 交互逐渐减少
    // 每次剪除 1/(l₂-l₁) 比例的 visual tokens
    // 关键 visual tokens 在 PageRank 中得分高，被保留到最后

LLM Layer 22~28 (l > l₂): Text-only 推理阶段
    x = [t]  // 仅 text tokens
    // Self-Attention 和 FFN 全部在 text tokens 上
    // visual tokens = 0，计算量大幅降低
```

**消融证据**：
- l₂=8（第 8 层后移除全部 visual tokens）：VideoMME 从 58.0 暴跌至 41.9
- l₂=15（第 15 层后移除）：54.3（部分恢复，但仍低）
- l₂=22（第 22 层后移除）：58.1（几乎无损）
- l₂=29（不移除）：58.0

结论：前 14 层（50%）需要 visual tokens 做 fusion；22 层（79%）后 visual tokens 完全无用。

术语一般如何实现？如何使用？

Cross-modal fusion 是 MLLM 架构的内在属性，无需额外实现。AIM 通过分析各层的 Attention 行为（PageRank 分数分布、visual↔text attention 比例）和消融实验（在不同层剪枝 visual tokens 观察性能影响）来量化 fusion→text-only 的转变点。这些发现可用于指导其他 MLLM 的效率优化。

涉及论文标题：
- AIM: Adaptive Inference of Multi-Modal LLMs via Token Merging and Pruning

---
