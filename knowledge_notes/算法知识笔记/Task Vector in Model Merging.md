## Task Vector in Model Merging

术语是什么？
Task Vector（任务向量）定义为 fine-tuned expert 模型参数与 base 模型参数之间的差值：τᵢ = θ_b − θᵢ，表示从 base 到领域特化 expert 的参数空间位移方向。在模型合并中，通过组合多个 task vector 并加回 base：θ_m = θ_b + λ · Σ τᵢ，可获得多领域能力。MergeME 在 MoE 场景中使用 task vector：(a) 计算各 expert 的 τᵢ → Dare/Ties 处理 → τ_m → θ_m；(b) Task Vector Routing（附录 C）——计算输入梯度 g_inf 与 τᵢ 的余弦相似度作为路由决策。

从算法pipeline角度拆解术语：
```
τ_i = θ_b - θ_i                                // task vector 定义
τ_m = Σ τ_i                                     // 合并
θ_m = θ_b + λ · τ_m                            // 加回 base
// MergeME MoE: 仅非 FFN 层参与，FFN 层保持独立
```

术语一般如何实现？如何使用？
- 开源：mergekit（https://github.com/arcee-ai/mergekit）提供 Ties/Dare/Task Arithmetic。
- Task Vector Routing：g_inf = ∇_{θ_b} L(x_inf)，路由权重 = SoftMax(top-K(Sim(g_inf, τᵢ)))。实验显示 PPL 路由优于 Task Vector Routing（Table 3: 8.08 vs 7.05）。

涉及论文标题：
- MergeME: Model Merging Techniques for Homogeneous and Heterogeneous MoEs

---
