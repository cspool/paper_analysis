## Expert Locality in MoE

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Expert Locality 是 MoE 语言模型中观察到的一种 token 序列模式：在处理连续 token 时，模型倾向于复用部分 expert，而非每 token 随机选择全新 expert。论文对 Mixtral-8x7B-Instruct 的分析（图 1）发现两类局部性：(1) 某些 expert 在 2-4 个连续 token 上持续激活（连续复用模式）；(2) 另一些 expert 以"间隔"方式复用——在非相邻 token 之间反复出现。这种局部性是 MoE offloading 中 LRU cache 策略有效的基础——若无局部性，cache hit rate 将接近随机水平。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Expert locality 示例 (Mixtral-8x7B, 某层 8 experts, top-2 routing):
# Token:    t0    t1    t2    t3    t4    t5    t6    t7
# Experts:  E2,E5 E2,E5 E2,E7 E7,E3 E3,E0 E0,E1 E1,E6 E6,E2
# 
# 观察:
# - E2: t0,t1,t2 → 3 连续 token (连续复用)
# - E5: t0,t1 → 2 连续 token
# - E7: t2,t3 → 2 连续 token, 间隔复用 (t2 和 t3)
# - E3: t3,t4 → 2 连续 token
# - E0: t4,t5 → 2 连续 token
#
# LRU cache (k=2):
# t0: cache=[E2,E5], miss=2, load E2,E5
# t1: cache=[E2,E5], hit=2, 直接使用
# t2: cache=[E2,E5], hit=1 (E2), miss=1 (E7), load E7, evict=无(k=2, 仅 E2/E5 在 cache)
#     实际实现中 t2 需加载 E7 并 evict less recently used
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- 论文在 OpenAssistant 对话数据上评测了不同 cache size k 的 LRU cache hit ratio（图 2 left panel）
- Cache hit ratio 随 k 增大而单调增长但边际递减
- MoE 的 expert 局部性源于专家专业化——某些 expert 学习特定语言模式（如介词、概念表达），在相关主题的连续 token 上被反复激活
- 该模式最早由 Shazeer et al. (2017) 观察到 interpretable expert specializations，论文首次将其用于 offloading 优化

涉及论文标题：
- Fast Inference of Mixture-of-Experts Language Models with Offloading
- LocMoE: A Low-overhead MoE for Large Language Model Training

**LocMoE 的 Locality Loss (局部性损失)**：

LocMoE 将 Expert Locality 从一种观察现象提升为主动优化的训练目标。其 Locality Loss $L_{loc}$ 鼓励 token 优先路由到同节点（本地）的 expert：

$$L_{loc} = \mu \cdot KL(D_c || D_l) = -\mu \int D_c(x) \ln[\frac{D_l(x)}{D_c(x)}] dx$$

其中 $D_c$ 为当前 batch 中 token 在各节点各 expert 的实际分配分布，$D_l$ 为完全局部化的理想分布（token 仅分配给本地 expert），$\mu$ 为超参数。

Localitiy Loss 与 Auxiliary Load Balance Loss ($L_{aux}$) 联合作为软约束：

$$L_{task} = L_{aux} + L_{loc} + L_{cross}$$

作用机制：
- Load balance ($L_{aux}$) 保证 token 在各 expert 间均匀分配（统计均衡）
- Locality ($L_{loc}$) 在负载均衡前提下，将跨节点 All-to-All 通信转为节点内高带宽通信（如 HCCS 256GB/s），降低 All-to-All 时间 5.13%
- 同时 locality 软约束避免 SwitchMoE 的 "winner-take-all"——更多 expert 参与早期训练

局限性：当节点数 > expert 数时（如 256N 下 16 experts 分布在 32 节点），部分节点无本地 expert，locality 策略失效，性能不如单纯负载均衡的 HashMoE。
