## Graph Distribution Shift

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Graph Distribution Shift 是指图数据中训练分布（源分布 D_s）与测试分布（目标分布 D_t）之间的差异，源于图结构、节点特征、边特征或其组合的自然变化。与标准 ML 中的 covariate shift 不同，图分布偏移具有独特性质：(1) **多维性**——偏移可来自图大小变化、节点度变化、特征噪声、边密度变化、子图结构变化等多种维度，且可组合形成复合偏移；(2) **实例级异质性**——同一目标分布中不同节点/图实例可能经历不同程度和类型的偏移；(3) **非 IID 传播**——偏移通过邻居关系和消息传递机制在图结构中传播。GraphMETRO 将这些偏移建模为多个 shift component 的混合（Assumption 1: 任意分布偏移可建模为 ≤k 个 transform classes 的混合），通过 gating model 识别成分、expert models 缓解各成分的影响。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# GraphMETRO 中分布偏移的分解与处理
stochastic_transforms = {
    'drop_edge':    随机删除边 (p ∈ [0.3, 0.5]),
    'add_edge':     随机添加边 (p),
    'drop_node':    随机删除节点 (p),
    'noisy_node_feat': 向节点特征加 Gaussian noise,
    'random_subgraph': 随机 k-hop 子图采样,
}

# τ^{(k)} = τ_{i1} ∘ τ_{i2} ∘ ... ∘ τ_{ik} 模拟 G 在 D_t 中的表现
# Gating: w = ϕ(G) → 识别各 τ_i 对当前 instance 的贡献
# Expert: ξ_i(τ_i(G)) ≈ ξ_0(G) → 消除对应 τ_i 的影响
# Aggregate: h = Σ Softmax(w)[i] · ξ_i(G) → 对组合偏移不变
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实际使用：(1) 根据领域知识选择 transform 函数集合——通用图变换覆盖多数场景，特定领域（分子图）需定制 transforms；(2) 可通过目标域样本在 embedding 空间中测量变换后数据集与目标样本的距离来筛选相关 transforms；(3) transform 数量影响性能——过多引入噪声降低性能，过少无法充分覆盖偏移空间。GraphMETRO 代码：https://github.com/Wuyxin/GraphMETRO。

涉及论文标题：
- GraphMETRO Mitigating Complex Graph Distribution Shifts via Mixture of Aligned Experts
