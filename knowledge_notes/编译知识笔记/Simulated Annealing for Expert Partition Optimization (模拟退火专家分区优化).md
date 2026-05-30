## Simulated Annealing for Expert Partition Optimization (模拟退火专家分区优化)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Simulated Annealing (SA) 是一种用于解决组合优化问题的元启发式算法，模拟冶金退火过程。在 MoE-Prism 中，SA 被用作 Partitioning Optimization Solver 的第二阶段，对贪心初始化产生的初始 expert 分区进行迭代精炼。算法在解空间中进行随机游走（交换不同 sub-expert 中的 neuron），始终接受更优解（更低 cost），并以概率 exp((current_cost - new_cost) / T) 接受更差解（避免陷入局部最优），温度 T 从 T0=100 按冷却率 α=0.995 指数衰减。优化目标为最小化所有 batch 上被 deactivated sub-experts 的 L1 范数之和：P* = argmin Σ_b Σ_{l∈top-K(L_b)} l。

从编译框架角度拆解术语：
SA solver 接收从 Neuron Activation Profiler 输出的激活矩阵 M(B×C)，输出最优分区映射 P*。其作为 offline "编译器"将 MoE 模型从粗粒度结构"编译"为细粒度弹性结构：
```
Input: M (B×C activation matrix), N (num sub-experts), T0=100, α=0.995, I=100000
Output: P* = {S_1: [neuron_ids], ..., S_N: [neuron_ids]}

# Phase 1: Greedy Initialization
neuron_impact = sum(|M[:, c]| for c in range(C))  # per-neuron L1 norm
sort neurons by impact descending
P = {S_1:[], ..., S_N:[]}
for each neuron n in sorted_neurons:
    assign n to sub-expert S_j with minimum current cumulative_impact[j]

# Phase 2: SA Refinement
current_cost = compute_cost(P, M)
best_P, best_cost = P, current_cost
T = T0
for i in 1..I:
    # Neighbor generation: swap two random neurons
    S_src, S_dst = random_sub_experts(P)
    n_src = random_neuron(S_src)
    n_dst = random_neuron(S_dst)
    P_new = P with n_src↔n_dst swapped
    
    cost_new = compute_cost(P_new, M)
    Δ = cost_new - current_cost
    
    if Δ < 0:  # Better solution (always accept)
        P = P_new; current_cost = cost_new
        if current_cost < best_cost:
            best_P = P_new; best_cost = current_cost
    else:  # Worse solution (accept with probability)
        if random() < exp(-Δ / T):
            P = P_new; current_cost = cost_new
    
    T *= α  # Cool down

return best_P
```
关键设计选择：(1) 贪心初始化提供高质量起点（O(C log C) 复杂度），减少 SA 所需迭代；(2) SA 的随机接受机制使求解器能跳出局部最优——当温度高时接受较差解概率大（探索），温度低时趋向于只接受更优解（利用）；(3) cost function 直接对"停用 sub-expert 的 L1 norm"求和，使优化目标与运行时的动态去激活行为对齐。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- MoE-Prism 用 Python 实现 SA solver，配置 T0=100, α=0.995, I=100K。每个 expert（C 从 256 到 1536 不等）独立求解。
- SA 是通用优化方法，在 ML 系统中常见于硬件-算法协同设计中的离散优化问题（如 model placement、tile sizing、scheduling order）。在 compiler/框架上下文中，类似优化还有整数线性规划（ILP）、多面体优化、遗传算法等，用于解决算子融合、内存分配、loop tiling 等问题。
- 论文未说明开源链接。SA 的收敛性取决于 cooling schedule——过快冷却导致早熟收敛，过慢冷却浪费计算。MoE-Prism 的 α=0.995 使温度从 100 到约 100×0.995^100000 ≈ 100×e^{-501} ≈ 0，确保充分探索后收敛。

涉及论文标题：
- MoE-Prism: Disentangling Monolithic Experts for Elastic MoE Services via Model-System Co-Designs

---
