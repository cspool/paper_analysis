## Hardware-Aware Cost Function（硬件感知成本函数）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Hardware-Aware Cost Function（Z）是本文定义的统一推理成本指标，将 memory（M_infer）和 computational（C_infer）两种不同量纲的成本合并为单一数值，用于比较不同 GQA 配置的成本效率。公式为：

$$Z = \lambda \cdot M_{\text{infer}}^{\alpha} + (1-\lambda) \cdot C_{\text{infer}}^{\beta}$$

其中 λ 控制 memory vs compute 的相对重要性，α 和 β 控制成本增长的非线性惩罚。默认参数 λ=0.9, α=1/2, β=1/3 由作者环境的硬件利用率测试确定，反映 memory 通常是长上下文推理的主要瓶颈（偏重 memory）。

从算法pipeline角度拆解术语：

**参数含义**：
- λ=0.9：memory 占 90% 权重，反映长上下文下 memory bandwidth 为主要瓶颈
- α=1/2：memory 成本以平方根增长——边际成本递减（DDR 带宽利用率在大 memory footprint 时更高效）
- β=1/3：compute 成本以立方根增长——计算可被 Tensor Cores 更好地并行化
- λ=0 最小化纯 FLOPs；λ=1 最小化纯 memory

**在 Cost-Optimal GQA Search 中的使用**：
```
for each candidate H=(nh, nkv):
    N* = solve L(N;H) = L*
    C = 2N* + 4TL·d_h·nh
    M = N* + 2TL·d_h·nkv
    Z = 0.9 * sqrt(M) + 0.1 * cbrt(C)
H* = argmin Z
```

术语一般如何实现？如何使用？

参数 (λ, α, β) 可根据具体部署硬件调整——memory bandwidth 瓶颈更严重的硬件（如 edge devices）应增大 λ；compute-bound 场景（如 prefill 为主的 serving）可减小 λ。通过调整 λ 可实现 Pareto-optimal 的 memory-compute tradeoff。代码开源：https://github.com/THUNLP/cost-optimal-gqa。

涉及论文标题：
- Cost-Optimal Grouped-Query Attention for Long-Context LLMs

---
