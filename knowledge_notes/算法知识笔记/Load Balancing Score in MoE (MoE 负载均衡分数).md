## Load Balancing Score in MoE (MoE 负载均衡分数)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Load Balancing Score 是 MoE-Pruner 提出的量化 MoE expert 激活均衡度的指标：s = σ/μ = sqrt(Σ(f_i - μ)²/n) / μ，f_i 为 batch 内 routed 到 expert i 的 token 数。所有 layer 取平均得模型整体分数。分数越低 → expert 越均衡；越高 → 越不均衡。

从算法pipeline角度拆解术语：
```
for layer in 1..l:
    for token in batch: f[TopK(Softmax(x @ W_g), k=2)] += 1
    s[layer] = std(f) / mean(f)  # 变异系数
S = mean(s)
```
发现：(a) Upcycling 模型（Mixtral, Qwen1.5-MoE, MiniCPM-MoE）分数低 → expert 均衡 → 适合 weight-level pruning；(b) Train-from-scratch 模型（DeepSeek-V2, OLMoE）分数高 → cold expert 可被安全 expert-level prune；(c) Qwen1.5-MoE 例外——虽用 upcycling 但打乱 expert 参数，行为似 train-from-scratch。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- C4 子集一次前向 hook 收集 topk_indices。用于指导 MoE 剪枝策略选择和评估训练质量。局限：仅计数未考虑 router 权重幅度。

涉及论文标题：
- MoE-Pruner: Pruning Mixture-of-Experts Large Language Model using the Hints from Its Router
