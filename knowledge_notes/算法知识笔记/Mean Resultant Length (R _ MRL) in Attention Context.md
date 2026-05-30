## Mean Resultant Length (R / MRL) in Attention Context

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Mean Resultant Length (MRL/R) 是方向统计学（Directional Statistics, Mardia & Jupp, 1999）中的标准度量，TriAttention 将其引入注意力分析，用于量化 pre-RoPE 空间中 Q/K 向量围绕其均值方向的聚集程度。对频段 f 的 Q 向量：$R_f = \|\mathbb{E}[q_f]\|/\mathbb{E}[\|q_f\|]$。R_f = 1 表示所有向量指向完全相同方向（完美聚集，三角函数级数精确）；R_f = 0 表示向量均匀分布（无聚集，三角函数级数不可用）。

MRL 在 TriAttention 中的双重角色：
(1) 量化聚集强度——判断三角函数级数对每个频段的可靠性；
(2) 自适应加权因子——在 S_norm 中 (1-R_f) 决定了 norm-based 信号的贡献：R_f 高时 (1-R_f) 小，S_trig 主导；R_f 低时，S_norm 贡献更大。

从算法pipeline角度拆解术语：
```
for each head h, frequency band f:
    E_q_f = mean(calib_Q[h, :, f])     # Q 中心（复数）
    R_f = |E_q_f| / mean(|q_f|)       # MRL: 0 ≤ R_f ≤ 1
    S_norm 中的自适应权重 = (1 - R_f)  # 聚集强 → 权重小；聚集弱 → 权重大
```

术语一般如何实现？如何使用？

实现：离线校准阶段计算，与 Q/K 中心同时完成。在 Qwen3-8B 上典型 MRL 约 0.98，约 90% heads 的 R > 0.95。MRL 跨领域数据（Math/Coding/Chat）几乎相同，证明其为模型内在属性。使用方式：(1) 诊断哪些 head 适合纯 S_trig；(2) 自适应平衡两个评分组件；(3) 跨架构验证聚集现象的普遍性。

涉及论文标题：
- TriAttention: Efficient Long Reasoning with Trigonometric KV Compression

---
