## Attention Entropy (注意力熵)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Attention Entropy 是衡量单个 query 的 attention score 分布在所有 key 上集中程度的信息论度量。定义为对 softmax 后的 attention scores 计算 Shannon entropy：
$$E = -\sum_{i=1}^{N} a_i \log(a_i)$$
其中 $(a_1, \ldots, a_N) = \operatorname{Softmax}(qK^T/\sqrt{d_k})$ 是 attention score 分布。Attention entropy 越低表示 attention 越集中（sparse——少数 token 占据绝大部分 attention mass），越高表示 attention 越均匀分散（dense——所有 token 贡献大致相等）。最大熵发生在 uniform 分布时：$E_{\max} = -\log(1/N) = \log(N)$。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**Attention Entropy 计算流程**：
```
# 输入: attention scores S = qK^T ∈ R^{N}
# 输出: entropy E ∈ R (标量)

def compute_attention_entropy(S):
    a = softmax(S / sqrt(d_k))     # (N,) 概率分布
    E = -sum(a_i * log(a_i) for i in 1..N)
    return E

# E ∈ [0, log(N)]
# E → 0: 极度集中 (one-hot-like, 所有 mass 在单个 token)
# E → log(N): 完全均匀 (所有 token 同等重要)
```

**跨层 Attention Entropy 分析**（Exploiting Sparsity 论文 Fig.5）：
```
for each layer ℓ in 1..L:
    # 对 50 个 1000-token samples 的最后一个 token 计算
    E_layer[ℓ] = mean_over_samples(compute_attention_entropy(S_last))
    # 聚合 all heads 的 attention score

# 观察: Layer 1 的 entropy 显著高于后续 layers
# Layer 2-32 的 entropy 迅速下降并保持低位
# → 深层 attention 天然更稀疏，可以更激进地压缩
```

术语一般如何实现？如何使用？

Attention entropy 在长上下文推理中有三个核心用途：(1) **稀疏度预测**——低 entropy 表示高稀疏度，可以用较少 k 恢复 dense attention 性能；(2) **任务难度预估**——不同的下游 task 的 attention entropy 有显著差异（Needle In A Haystack: entropy 1.93 vs Word Counting: entropy 2.68），高 entropy 任务需更多 k；(3) **跨层 budget 分配指导**——第一层 entropy 最高，后续层迅速下降，指导 layer-wise adaptive k 分配策略（给第一层更多 k budget，后续层逐渐减少）。

Exploiting Sparsity 论文发现：attention entropy 与 "达到 95% dense attention 性能所需的最小 k%" 之间的 Pearson correlation 达到 0.847，表明 entropy 是预测 k 需求的可靠指标。

涉及论文标题：
- Exploiting Sparsity for Long Context Inference: Million Token Contexts on Commodity GPUs

---
