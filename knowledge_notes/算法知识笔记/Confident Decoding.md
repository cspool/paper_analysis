## Confident Decoding

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Confident Decoding是Dimple提出的动态token选择策略，用于改进离散扩散模型的推理效率。与标准MaskGIT每步解码固定数量token（由schedule决定）不同，Confident Decoding基于绝对置信度阈值$\gamma \in (0,1)$动态决定每步解码的token数量。核心motivation：文本不同位置的token可预测性差异大——固定短语很早就可高置信预测，而复杂推理位置需要更多上下文。Fixed schedule忽略这种异质性。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
Input: x_t, logits z_t, τ, γ, fallback K

p_t = softmax(z_t)              # pre-revision (用于confidence)
p̃_t = softmax(z_t / τ)          # post-revision (用于采样)

For each masked i:
  c^(i) = max(p_t^(i))          # confidence
  x̃^(i) ~ Categorical(p̃_t^(i))  # 候选token

If ∃i, c^(i) ≥ γ:
  I = {i | c^(i) ≥ γ}           # 批量更新高置信位置
  For i in I: x_{t+1}^(i) = x̃^(i)
Else:
  I = RandomSample({1..N}, K)   # fallback: 随机选K个
  For i in I: x_{t+1}^(i) = x̃^(i)

Return x_{t+1}
```

Annotations: γ=0.7为Dimple经验值；confidence使用pre-revision概率（不受temperature影响，保留位置间相对关系）；Fallback保证即使无高置信度位置也能推进生成。典型效果：22 token仅需7次迭代完成（~1/3 response_length）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Confidence计算使用pre-revision概率——因为temperature/top-p等revision以position-wise方式应用，破坏跨位置relative confidence ranking。Confidence函数可选max probability（最简单）、entropy、或概率margin。阈值γ需调优：太高→频繁fallback（低效）；太低→低质量token过早解码（质量差）。可与任何decoding algorithm组合（MaskGIT、随机选择等）。预期加速：Dimple上将迭代数压缩到response_length的1/3至1/2。

**Fast-dLLM的理论扩展**：Fast-dLLM对confidence-aware parallel decoding进行了严格的理论分析（Theorem 1）。当n个token的边际置信度均满足p_j(X_{i_j}=x_{i_j}|E) > 1-ε，且(n+1)ε ≤ 1时，greedy parallel decoding（乘积边际分布的argmax）等价于greedy sequential decoding（真实联合分布的argmax）。该定理同时给出了L_p距离上界D_TV < (3n-1)ε/2和前向KL散度上界D_KL < (n-1)[H_b(ε) + ε·ln(|V|-1)]，量化了乘积分布对真实联合分布的逼近程度。

基于此定理，Fast-dLLM提出两种实用策略：(1) **Threshold策略**：仅解码c_i > τ的token，始终保底解码max confidence token以避免死循环；(2) **Factor策略**：排序置信度后找最大n使(n+1)(1-c^(n)) < f，动态控制并行度。Factor策略通常比threshold策略提供1.4-1.5×更高吞吐量（代价约1-3%准确率）。Fast-dLLM在LLaDA上实现置信度感知并行解码单独加速13.3×（8-shot, gen_len=1024），与KV Cache结合达27.6×。

涉及论文标题：
- Dimple Discrete Diffusion Multimodal Large Language Model with Parallel Decoding
- Fast-dLLM Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding
