## Knee Truncation（膝点截断）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Knee Truncation 是一种动态选择 top-k 候选 token 数量 k* 的方法。与固定 k 的 top-k sampling 不同，knee truncation 通过检测排序概率分布中的"膝点"（elbow/knee point）——即相邻排序概率之间差值最大的位置——来自适应地确定 k*：k* = argmax_k (p_{(k)} - p_{(k+1)})。其中 p_{(1)} ≥ p_{(2)} ≥ ... 为降序排列的 token 概率。直觉上，膝点之前是概率密集区（多个竞争 token），膝点之后是概率长尾区（token 概率均为极小值）。在幻觉步中，分布往往更平坦、k* 更大；在非幻觉步中，分布尖锐、k*≈1。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Knee Truncation 计算
p_sorted = sort(p_i, descending=True)
diffs[k] = p_sorted[k] - p_sorted[k+1]
k* = argmax_k(diffs)
C_i = top_k(p_i, k*)

# 数值示例:
# 幻觉步: p = [0.30, 0.28, 0.15, 0.12, 0.08, 0.04, 0.02, 0.01]
#   diffs = [0.02, 0.13, 0.03, 0.04, 0.04, 0.02, 0.01], k*=2
# 自信步: p = [0.95, 0.02, 0.01, 0.007, ...]
#   diffs = [0.93, 0.01, 0.003, ...], k*=1
```

k* 的双重作用：(a) 确定候选集大小——只有 C_i 内的 token 参与协商混合；(b) 参与不确定性检测——k*>1 是触发 visual decider 的必要条件之一（另一条件是 margin≤δ）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
膝点检测是一种广泛使用的自适应阈值选择方法。在 LLM 解码中，knee truncation 相比固定 k 的优势在于：(a) 不需要手动调参——k* 根据每步的概率分布形状自动确定；(b) 在自信步时 k*≈1 保持 greedy 行为；(c) 在不确定步时 k* 较大，给证据更多操作空间。论文实证（Ghosh et al., [5]）支持：幻觉步倾向于有更大的 k* 和更小的 variance。

涉及论文标题：
- See It, Say It, Sorted: An Iterative Training-Free Framework for Visually-Grounded Multimodal Reasoning in LVLMs
