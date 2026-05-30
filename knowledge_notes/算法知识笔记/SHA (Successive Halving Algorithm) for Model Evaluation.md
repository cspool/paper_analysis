## SHA (Successive Halving Algorithm) for Model Evaluation

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Successive Halving Algorithm (SHA) 是 Jamieson & Talwalkar (AISTATS 2016) 的超参数优化 bandit 算法。将有限计算预算均分给所有候选，通过多轮淘汰赛（每轮保留 top 1/η）快速筛选最优候选。每轮预算递增（×η），survivor 获得更充分评估。在 Mordal 中，SHA 用作 inter-cluster evaluation 的 early stopping——快速淘汰差 cluster，资源集中于有潜力的 cluster。

从算法pipeline角度拆解术语，给出具体例子。
Mordal 中 SHA 流程：
```
representatives = [cluster_medoids]; budget = b = 0.03
while len(representatives) > top_k_inter:
    for each rep: train(rep, data_ratio=budget); score = evaluate(rep)
    representatives = top_k(score, k = ceil(len/eta))
    budget *= eta
// Example: 7 reps, eta=2
// Rung0: budget=3%, 7→4; Rung1: budget=6%, 4→3 (≤top_k_inter=3, done)
// Total: 7×3%+4×6% ≈ 0.81× one full training
```

术语一般如何实现？如何使用？
Mordal 限制 SHA 于 inter-cluster evaluation（rough filtering）。η=2, R=0.125, b=0.03。消融实验：SHA 显著减少搜索时间但 aggressive elimination 可能误淘汰有潜力候选（如 AI2D 的 SigLIP-Qwen 在早期表现不佳）。因此 Mordal 在 intra-cluster 阶段使用 scaling prediction 替代 SHA 做精细评估。

涉及论文标题：
- Mordal: Automated Pretrained Model Selection for Vision Language Models

---
