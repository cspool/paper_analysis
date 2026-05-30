## Instance-wise Heterogeneity in Graph Distribution Shifts

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Instance-wise Heterogeneity 是图分布偏移中的关键现象：同一目标分布的图/节点实例经历不同类型和程度的偏移。在 WebKB 数据集中，两个不同网页节点在目标域的特征变化程度截然不同——尽管都经历了 source→target 的偏移，各自的 shift pattern 不同。标准 invariant learning 关注 group-level patterns，缺乏对实例间差异的建模。GraphMETRO 通过 gating model ϕ 对每个 instance 输出个性化权重 w ∈ R^{K+1} 来编码该 instance 的偏移成分分布——w 连续且 instance-dependent，支持无限种偏移组合，实现 instance-adaptive 处理。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Instance-wise 异质性处理
# 数学表达: h(G) = Σ_{i=0}^K Softmax(ϕ(G))[i] · ξ_i(G)
# w 为 instance-dependent → 不同 instance 得到不同 expert 组合

# WebKB 例子:
# Node u¹ (内容大变化/结构不变): w=[0.05,0.05,0.5,0.1,0.2,0.1]
#  → noisy_node_feat expert (idx 2) 主导
# Node u² (内容小变化/结构变化): w=[0.05,0.4,0.1,0.3,0.1,0.05]
#  → add_edge (idx 1) + drop_node (idx 3) 主导

# 连续 w → 无限种 expert 组合 → 任意粒度偏移自适应
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现依赖 MoE 的 softmax routing——gating model 使用 BCE loss（多标签二分类）训练，使 ϕ 对每个 τ_i 的敏感性独立于其他 τ_j，确保 w 各分量独立反映对应 shift component 的存在与否。Distribution shift discovery（Figure 4b）验证了 gating model 能准确识别目标分布的全局偏移类型（WebKB: add_edge 主导，Twitch: noisy_node_feat+drop_node 主导），gating accuracy 达 92.4%/93.8%。

涉及论文标题：
- GraphMETRO Mitigating Complex Graph Distribution Shifts via Mixture of Aligned Experts
