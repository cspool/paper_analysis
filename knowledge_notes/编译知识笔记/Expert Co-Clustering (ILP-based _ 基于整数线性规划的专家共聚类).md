## Expert Co-Clustering (ILP-based / 基于整数线性规划的专家共聚类)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Expert Co-Clustering 是 Sem-MoE 提出的离线优化方法，将 MoE 推理中的 expert placement 和 token routing 建模为联合优化问题（0-1 ILP），通过交替优化算法求解。目标函数为 min θ·load_imbalance + (1-θ)·remote_activation_volume，受限于：(1) 每个 token 唯一分配到某个 cluster/device；(2) 每个 expert 唯一分配到某个 cluster/device；(3) 每个 cluster/device 的 expert 数量相等（保证 EP 下的内存均衡）。决策变量：R_{ij} ∈ {0,1}（token j 是否分配到 cluster i）、C_{ij} ∈ {0,1}（expert j 是否分配到 cluster i）。左半部分 Σ|Σ(R_{ij}·a_j) - S/E| 最小化各 cluster 的 token 频率差异（load balance），右半部分 Σ_{i1≠i2} Σ_{j,k} R_{i1j}·C_{i2k}·C_{p,jk}·a_j 最小化跨 cluster 的远程激活（通信量）。θ ∈ (0,1) 控制权衡。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。

属于 offline optimization（部署前求解），非 online compiler pass。求解流程（Algorithm 1）：

```
Input: C_p (token-expert confidence table), a (token frequency), E (cluster count)
Output: E (expert labels), T (token labels), Tp (confidence)

1. Initialize: p_matrix_ep ← zeros(N, E)/E, p_matrix_req ← zeros(K, E)/E

2. Alternating Optimization (n_epochs):
   Repeat:
     # Phase 1: Expert Placement
     Sort experts by load (hotness) descending
     For each expert e:
       Compute EAfE[e] = expert-expert affinity within cluster
       Compute EAfR[e] = req-expert affinity within cluster
       aff_score = α_e·EAfE[e] + β_e·EAfR[e] - γ_e·load_cluster
       Place e to cluster with max aff_score (mask saturated clusters)
     Fine-tune: randomly swap experts between clusters if improves score (f_t steps)

     # Phase 2: Request Scheduling
     Sort requests by length
     For each request r:
       Compute RAfR[r] = req-req affinity within cluster
       Compute RAfE[r] = req-expert affinity within cluster
       aff_score = α_r·RAfR[r] + β_r·RAfE[r]
       Schedule r to cluster with max score (mask full clusters)

     Score current solution → keep best

3. Output: E = argmax(p_matrix_ep_opt, axis=1)
           T, Tp = argmax_with_values(p_matrix_tk_opt, axis=1)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Python 实现，在 CPU 上以 alternating optimization 近似求解（非精确 ILP solver，避免大量中间变量），部署前离线完成（一次性）。负载均衡约束（每 cluster expert 数相等）和掩码机制（saturated cluster 不再分配）确保每个 EP rank 获得相同数量的 expert。与 MoETuner（Go & Mahajan, 2025）的 ILP 相比：MoETuner 优化 end-to-end execution time，需精确的 per-expert 计算时间估算；Sem-MoE 优化 LAR + load balance，基于 profiling data 的统计信息。Sem-MoE 交替优化实测比 SGLang vanilla 提升 LAR 15.4%，比 MoETuner 提升 36.7%。

涉及论文标题：
- Speculative MoE: Communication Efficient Parallel MoE Inference with Speculative Token and Expert Pre-scheduling
