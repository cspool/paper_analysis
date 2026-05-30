## Heterogeneous MoE Merging with Projectors

术语是什么？
Heterogeneous MoE Merging with Projectors 是 MergeME 首个支持不同架构 expert 合并为 MoE 的方法。核心组件：(1) 共享 Embedding/Head：各 expert embedding/head padding 零对齐到 d_m 后平均；(2) Proj-in/Proj-out：每个 expert 配备一对随机初始化 MLP——Proj-in: R^{d_m}→R^{dᵢ}, Proj-out: R^{dᵢ}→R^{d_m}；(3) Sequence-level Router：因异构 attention 不兼容，整句 token embedding 平均后通过 MLP router 做序列级路由。

从算法pipeline角度拆解术语：
```
d_m = max(d₁,...,dₗ)
emb_shared = avg(padded(emb_i, d_m)), head_shared = avg(padded(head_i, d_m))
Proj_in[i]: d_m → d_i, Proj_out[i]: d_i → d_m  // 随机初始化
// Forward:
avg_e = mean(e₁,...,e_t), α = SoftMax(top-K(θ_r · avg_e))
for selected expert k:
    r_k = Proj_out[k](Expert_k(Proj_in[k](e)))
output = head_shared(Σ α_k · r_k)
```

术语一般如何实现？如何使用？
- 局限：(a) 不合并 attention → 参数多（~4B vs ~3.7B）；(b) 合并 embedding 可能导致 router 偏好同架构 expert（Figure 6）。
- MergeME Table 4: MoE w/ Math TinyLlama avg 13.34 vs 3-expert MoE 10.54。

涉及论文标题：
- MergeME: Model Merging Techniques for Homogeneous and Heterogeneous MoEs

---
