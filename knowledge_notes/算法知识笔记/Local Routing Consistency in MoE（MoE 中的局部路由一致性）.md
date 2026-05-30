## Local Routing Consistency in MoE（MoE 中的局部路由一致性）

术语是什么？
Local Routing Consistency（局部路由一致性）是 MoE 模型中连续 token 倾向于激活相同或相似 experts 的程度属性。当 MoE 模型具有高局部路由一致性时，在一定长度的 token segment 内 router 选择的 expert 集合保持相对稳定。该属性由论文 "Not All Models Suit Expert Offloading" (ICLR 2026) 首次系统定义。高局部路由一致性使 expert offloading 系统的 GPU expert cache 获得更高命中率，减少从 CPU 加载 expert 的慢路径。大多数模型在短 segment（m=4）内展示相似的短期一致性，但长期（m≥16）差异显著——仅 Group 1 模型（LLaMA-MoE-v2, OLMoE 等）维持高 SRP (>0.5)。

从算法pipeline角度拆解术语：
```
# 局部路由一致性分析 pipeline
# Phase 1: 收集路由决策
for each sequence T, layer l, token t:
    A[T][l][t] = Router_l.top_k(hidden_state)  # 激活专家索引

# Phase 2: 统计 per-segment 专家激活频率
for each expert e, segment length m, start position p:
    f[e,p,m] = Σ A[T][p:p+m] where e is activated

# Phase 3: SRP 计算 (Eq.4,6 — 专家固有属性，无参数)
for α in [0,m]:
    F1[α] = 2*Σ_{f>=α} f / Σ[m·I(f>=α) + f]
SRP = max_α F1[α]

# Phase 4: SCH 计算 (带 cache 容量约束)
for scenario with cache ratio ρ:
    simulate oracle cache evicting least-future-used experts
    SCH = hit_count / total_accesses
```

论文关键发现：(1) 局部路由一致性与局部负载均衡存在 trade-off——高一致性模型路由更集中（expert activation SD 大），但全局负载均衡可通过 domain-specialized experts 与局部一致性共存；(2) Shared experts 通过减小 expert combination space 降低局部一致性；(3) Domain-specialized experts 对局部一致性的贡献大于 vocabulary-specialized ones；(4) Cache size ≈ 2× active experts 在大多数模型上取得最佳性价比。

术语一般如何实现？如何使用？
部署前评估：对候选 MoE 模型调用论文代码 (https://github.com/ljcleo/moe-lrc) 计算 SRP/SCH，选择高一致性模型部署到 memory-constrained 设备。架构设计指导：避免 shared experts、增大 expert combination space（更多 total experts 和适当 k）、重视 domain specialization 的训练。

涉及论文标题：
- Not All Models Suit Expert Offloading: On Local Routing Consistency of Mixture-of-Expert Models
