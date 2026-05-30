## Fine-Grained MoE Scaling Law / Granularity（细粒度 MoE 缩放定律 / 粒度）

术语是什么？
Fine-grained MoE Scaling Law 由 Krajewski et al. (2024) 提出，将 MoE 模型性能建模为总参数 P、训练 token 数 D 和粒度 G（active expert 数量）的函数：L(P, D, G) = c + (g/G^γ + a)/P^α + b/D^β。其中 G = P_active / P_expert，即每 token 激活的 expert 数量。与 Clark et al. (2022) 的早期 MoE scaling law 不同，该公式引入了粒度 G 作为独立 scaling axis，并证明在 compute-optimal 设置下，更高粒度（更多更小的 expert）一致优于低粒度（更少更大的 expert）。外推预测：持续提升模型容量最终需要极高粒度的大型模型，对应极大量极小专家的架构。PEER 直接将此理论推到极致：d_expert = 1 → G = hk，N = P/G（百万级）。

从算法pipeline角度拆解术语：
Scaling law 的推导逻辑：
```
给定: P (总参数), P_active (每 token 激活参数), P_expert (单个 expert 大小)
G = P_active / P_expert    (粒度 = active expert 数量)
N = P / P_expert = P × G / P_active   (总 expert 数量)

目标: 降低 loss L → 增大 P, D, G
约束: 限制 P_active (控制计算和内存开销)
策略: G ↑ → P_expert = P_active/G ↓ → N = P/P_expert ↑
结论: 需要大量小 expert 而非少量大 expert
```
PEER 的参数化：d_expert=1 → P_expert ≈ 2×d_model, P_active = hk × 2×d_model, G = hk。通过增加 N（product key 规模）仅增加总参数存储，不增加 P_active（每 token FLOPs 不变）。

术语一般如何实现？
PEER 直接基于此 scaling law 设计：(1) d_expert=1 最小化 P_expert；(2) Product Key 支持 N ≥ 10⁶；(3) hk 控制 P_active（固定计算预算）。Ablation 验证了预测：增加 N（128² → 1024²）单调改善 perplexity；增加 hk（32 → 512）改善性能但渐趋饱和，需权衡性能和资源。当前此 scaling law 在 Krajewski et al. (2024) 中为经验性发现，理论上的最优 G 与 P 的关系仍为开放问题。

涉及论文标题：
- Mixture of A Million Experts
