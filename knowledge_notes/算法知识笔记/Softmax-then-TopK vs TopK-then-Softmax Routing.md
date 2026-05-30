## Softmax-then-TopK vs TopK-then-Softmax Routing

术语是什么？

在 MoE Router 中，将 Router logits 映射为 expert 选择有两种顺序：(1) **Softmax-then-TopK**：先对整个 logit 向量做 softmax 得到概率分布，再从概率分布中选 top-K（标准 MoE 做法，Shazeer et al., 2017）；(2) **TopK-then-Softmax**：先从 logit 中选 top-K，仅对这 K 个 logit 做 softmax（Mixtral 8x7B 的做法，Jiang et al., 2024）。

两者的本质区别在于：Softmax-then-TopK 给所有 E 个 expert 都分配了非零概率（虽然只有 top-K 被激活），保留了"非 top-K 的 logit 有多接近被选中"的信息；TopK-then-Softmax 则完全丢弃了非 top-K logits 的绝对值信息，因为 softmax 仅作用于 top-K 个 logit，对于 topK=1 的特殊情况，softmax of single element = 1（常数），梯度为零。

从算法pipeline角度拆解：

```
# 方法 1: Softmax-then-TopK (本文推荐)
r = x @ W_r               # (E,) Router logits
s = softmax(r)            # (E,) probability distribution
[p1, p2], [e1, e2] = topk(s, 2)  # select top-2
# 输出: p1*Expert(e1) + p2*Expert(e2)
# p1, p2 是原始 softmax 值，通常不归一化到和为1

# 方法 2: TopK-then-Softmax (Mixtral 风格)
r = x @ W_r               # (E,) Router logits
[val1, val2], [e1, e2] = topk(r, 2)  # select top-2 logits
[s1, s2] = softmax([val1, val2])  # 仅对选中的 logit 做 softmax
# 输出: s1*Expert(e1) + s2*Expert(e2)
# s1 + s2 = 1 (保证)
```

本文发现 softmax-then-topK 在 upcycling 场景下一致优于 topK-then-softmax。推测原因是保留所有 expert 的相对信息有助于 Router 梯度更丰富。但 softmax-then-topK 也有缺点：upcycling 初始阶段 MoE 输出与 dense 模型不等价（而 topK-then-softmax 在 topK > 1 时可以使输出 sum to 1），这一缺点被 Weight Scaling 方法弥补。

术语一般如何实现？

在 Megatron-LM 中默认使用 softmax-then-topK。在 NeMo 中可配置。切换方式：修改 Router 模块中 softmax 和 topK 的调用顺序。

涉及论文标题：
- Upcycling Large Language Models into Mixture of Experts
