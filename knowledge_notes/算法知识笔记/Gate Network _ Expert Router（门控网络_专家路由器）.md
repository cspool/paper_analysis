## Gate Network / Expert Router（门控网络/专家路由器）

术语是什么？
Gate Network（Expert Router）是MoE架构中决定每个token由哪些expert处理的小型可学习线性层（W_g ∈ R^{d_model × num_experts}）。输入token hidden state h，输出经Softmax归一化后得到每个expert的概率分数，Top-k选择后加权累加选中expert输出。FineMoE创新地使用gate完整probability distribution（不仅top-k选中集）——概率值表达gate对每个expert的相对置信度，驱动similarity-aware expert selection和probability-aware cache management。

从算法pipeline角度拆解术语：
```
logits = h @ W_g          # [d_model] @ [d_model, num_experts] → [num_experts]
probs = Softmax(logits)   # 对所有expert的概率分布，sum(probs)=1
selected = ArgTopK(probs, k)
norm_probs = probs[selected] / sum(probs[selected])  # 重新归一化选中expert概率
output = Σ norm_probs[i] * ExpertFFN_i(h)
```

术语一般如何实现？
标准实现为`nn.Linear(d_model, num_experts)` + Softmax。Mixtral-8x7B用top-2 gating，DeepSeek-V3用Sigmoid替代Softmax扩展至256 experts。FineMoE将完整probability distribution存入Expert Map Store，用于cosine similarity检索和probability-aware prefetch priority（p/(l-l_now)）及eviction priority（1/(p×freq)）。

涉及论文标题：
- Taming Latency-Memory Trade-Off in MoE-Based LLM Serving via Fine-Grained Expert Offloading

