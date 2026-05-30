## Deployment Plan Search for Disaggregated MoE Inference

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Deployment Plan Search（部署计划搜索）是 MegaScale-Infer 中自动确定最优 disaggreated expert parallelism 配置的优化算法（Algorithm 1）。给定 MoE 模型、硬件约束（C_a, C_e, M_a, M_e）和 SLO（TBT latency requirement），搜索目标为最大化 throughput per unit cost。搜索空间包含：(1) attention tensor parallelism size (tp_a)；(2) expert tensor parallelism size (tp_e)；(3) attention node 数量 (n_a)；(4) micro-batch 数量 (m)；(5) global batch size (B)。这是一个约束优化问题，包含 GPU memory capacity constraint、computation balance constraint (T_a ≈ T_e)、communication hiding constraints (T_c < T_f, m ≥ 2(1+T_c/T_f)) 和 SLO constraint (T_iter ≤ SLO)。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
Algorithm 1 伪代码：
```
Input: MoE model G, C_a, C_e, N_m, M_a, M_e
Output: optimal deployment plan plan*

plan* ← ∅
for tp_e ∈ {1, 2, ..., M_e}:
    for tp_a ∈ {1, 2, ..., M_a}:
        if tp_a × C_a > P_a and tp_e × C_e > P_e:  // Memory check
            n_a ← BALANCE(G, tp_a, tp_e)           // Eq. T_a ≈ T_e
            for m ∈ {3, 4, ..., N_m}:
                plan ← {(tp_e, E), (tp_a, n_a), m}
                B, tpuc ← SIMULATE(G, plan, SLO)  // Binary search max B
                plan ← plan ∪ {B, tpuc}
                if plan*.tpuc < plan.tpuc:
                    plan* ← plan
return plan*
```

SIMULATE 函数核心——性能模型：
- T_a 建模（memory-intensive attention）：T_a = k_1·b_a + k_2，其中 k_1 涵盖 KV cache 访问时间和 GEMM 时间，k_2 涵盖 TP synchronization 时间。k_i 通过 profiling + interpolation 获得。
- T_e 建模（compute-intensive expert FFN）：T_e = k_3·b_e + k_4，类比，通过 profiling GEMM 在不同 batch size 下的延迟获得。
- BALANCE 函数：由 T_a ≈ T_e 推导 n_a = (b_e·E)/(b_a·K) = (k_1·E)/(k_3·K)。
- T_c 建模（M2N 通信）：T_c = max{b_a·h·K/(tp_a·W_a·Util(.)), b_e·h/(tp_e·W_e·Util(.))}，network bandwidth utilization 函数 Util(msg_size) 通过 profiling 获得。
- T_iter ≤ SLO 检查和 binary search B 的最大值。
- Objective：tpuc = (B/T_total) / (tp_a·n_a·Cost_a + tp_e·E·Cost_e)

复杂度：O(M²·N_m)，M 为 GPU per server 上限（通常 4: {1,2,4,8}），N_m 为 max micro-batches（通常设 4），因此搜索空间很小（≤ 64 次 SIMULATE）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 搜索在 deployment 前离线执行一次（或硬件/模型变更时重新执行），推理时不引入 overhead。
- BALANCE 函数假设 T_a 和 T_e 的 linear model coefficients 在 batch size 范围内稳定（通过 profiling 验证）。若 profiling 显示非线性（如 batch size 跨越 GPU utilization knee point），可用 piecewise linear model。
- 效果验证（Figure 15 with DBRX）：DP degree 过低（n_a=1-4）→ attention 瓶颈；DP=8 → 平衡 → peak throughput；DP=16 → expert 瓶颈 → throughput 下降。仅特定 deploy plan 能最小化 idle 并最大化 GPU 利用率。
- 扩展：论文提到可扩展至考虑异构 GPU 类型选型（在枚举中增加 hardware type dimension），但未在 Algorithm 1 中显式实现。

涉及论文标题：
- MegaScale-Infer: Serving Mixture-of-Experts at Scale with Disaggregated Expert Parallelism

---
