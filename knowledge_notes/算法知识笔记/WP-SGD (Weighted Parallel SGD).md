## WP-SGD (Weighted Parallel SGD)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
WP-SGD（Weighted Parallel SGD，Cheng et al., 2020）是 SimulParallel SGD 在数据不均衡场景下的扩展。当各 worker 分配的数据子集大小不一致时，简单参数平均产生有偏梯度估计。WP-SGD 引入样本数加权系数 γ_k = |D_k|/|D|，对各 worker 参数做加权平均以保持梯度无偏性。MoE-DisCo 将其用于共享 backbone 参数的融合阶段：θ_shared* = Σ γ_k · θ_shared^(k)。当 K-Means 产生平衡簇时 γ_k ≈ 1/E，退化为简单平均。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# WP-SGD 在 MoE-DisCo Reintegration 阶段
total = Σ_{k=1}^{E} |D_k|
for k in 1..E:
    γ_k = |D_k| / total

θ_shared* = Σ_{k=1}^{E} γ_k · θ_shared^(k)   # 加权平均
θ_exp* = Concat(θ_1, ..., θ_E)                # expert 直接拼接
Θ = (θ_shared*, θ_exp*)                       # 组装完整 MoE
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
WP-SGD 在 Reintegration 阶段执行一次（离线），权重 γ_k 由 K-Means 聚类后自然得到。在 MoE-DisCo 实验中，K-Means 通常产生大小相近的簇（γ_k ≈ 1/E），但 WP-SGD 作为理论保障确保聚类不均衡时不引入偏差。该框架可推广到任意分布式训练中数据量不一致的场景。

涉及论文标题：
- MoE-DisCo: Low Economy Cost Training Mixture-of-Experts Models
