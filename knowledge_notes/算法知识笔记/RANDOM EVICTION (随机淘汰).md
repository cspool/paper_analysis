## RANDOM EVICTION (随机淘汰)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

RANDOM EVICTION 是 NACL 引入的基于概率采样的 KV Cache 淘汰策略。将 PROXY-TOKENS EVICTION 的 F_score 经 Softmax 归一化为概率分布 P_prompt = Softmax(F_score)，从该分布中采样 C_r 个 token 保留。每个 attention head 和每层使用不同随机种子实现 head-wise/layer-wise 多样化。

直觉：确定性 top-K 淘汰一旦丢弃关键 token 无法恢复。head-wise 随机采样确保每个 token 在多个 head 中有独立保留机会。LLaMA-7B 32层×32头 budget=20% 时保留概率 > 99.92%。

从算法pipeline角度拆解术语：

```
输入: F_score ∈ R^{p}, C_r, seed_h

Step 1: P_prompt = Softmax(F_score)                   # attention-guided prob distribution
Step 2: for head h: S_random^h ~ Multinomial(P_prompt, C_r, seed=seed_h)
Step 3: S_keep^h = S_proxy ∪ S_random^h               # total C = C_p + C_r
```

消融：移除 RANDOM EVICTION → short-text -1.2%, long-text -9.2%。Uniform 采样替代 → long-text -1.1%。随机 budget 10%→70% 性能提升 2.25%→8.17%，>90% 时下降（需 attention 引导）。

术语一般如何实现？如何使用？

PyTorch `torch.multinomial`，每 head 用 `head_idx * layer_idx * large_prime` 作为 seed。budget 分配：total 20% = 6% proxy + 12% random + 2% protect proxy（Table 4）。

涉及论文标题：
- NACL: A General and Effective KV Cache Eviction Framework for LLMs at Inference Time

---
