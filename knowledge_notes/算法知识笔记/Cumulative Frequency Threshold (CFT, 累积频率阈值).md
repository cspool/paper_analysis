## Cumulative Frequency Threshold (CFT, 累积频率阈值)

术语解释
Cumulative Frequency Threshold (CFT) 是 BuddyMoE 中从共激活数据构建紧凑 buddy expert 列表的算法参数。对于 pivot expert i，按 q_{j|i} 降序排列 peers，选最小前缀 t 使累积覆盖 ≥ α，构成 buddy list B_ℓ(i;α)。α∈(0,1] 是唯一超参数：α 越大 → 更大 buddy list → 更高 GPU 命中率；α 越小 → 更紧相似性 → 更小 buddy list。

术语是什么？
CFT 公式：t_i(α) = min{t | Σ_{r=1..t} q_{π_i(r)|i} ≥ α}，B_ℓ(i;α) = {π_i(1), ..., π_i(t_i(α))}。Capped at K_max（如 16）以控制 metadata。支持 per-layer α_ℓ 或 monotone schedule 适应不同层的冗余模式差异。Stabilization techniques：同时累积 binary + probability-weighted co-activation；Laplace smoothing M←M+ε；可选 down-weight early warm-up steps。

从算法pipeline角度拆解术语：
```
buddy_list = []
cumsum = 0
for peer in argsort_descending(q[i]):
    cumsum += q[i][peer]
    buddy_list.append(peer)
    if cumsum >= alpha: break
return buddy_list[:K_max]
```

术语一般如何实现？如何使用？
- 离线一次性计算，无运行时开销
- Profiling data 需匹配部署领域以确保 co-activation 模式代表性
- Verifying compactness：报告 |B_ℓ(i;α)| 分布确保 buddy lists 紧凑

涉及论文标题：
- BuddyMoE Exploiting Expert Redundancy to Accelerate Memory-Constrained Mixture-of-Experts Inference
