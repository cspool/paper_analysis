## Sub-Expert Decomposition (Expert Partitioning for MoE Elasticity / 子专家分解)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Sub-Expert Decomposition 是 MoE-Prism 提出的后训练（post-training）模型重构技术，将预训练 MoE 模型中每个 monolithic expert（完整 FFN）分解为 N 个细粒度、功能内聚的 "sub-expert"。核心思想基于两个观察：(1) MoE 中每个 monolithic expert 内部存在显著的激活稀疏性——对任意输入 token，expert 内 50% 的 neuron 激活幅度低于 0.0167，75% 低于 0.0391；(2) SwiGLU FFN 中不同列（即不同 neuron）的计算是独立的，因此可以将 neuron 重新分组。分解过程分三步：Neuron Activation Profiler 从校准数据集收集激活矩阵 M(B×C)；Partitioning Optimization Solver 以最小化被停用 sub-expert 的 L1 范数之和为目标，用贪心初始化 + Simulated Annealing (T0=100, α=0.995, 100K 迭代) 求解最优分区 P*；Gating Mechanism Reconstructor 构建新的细粒度路由机制。每个原 expert 划分为 N=4 个子 expert 后，激活控制粒度提升 4 倍，将 MoE 的 "Quality Cliff" 转化为平滑的 cost-quality 权衡曲线。

从算法pipeline角度拆解术语：
MoE-Prism 对每个 MoE layer 中每个 expert 的分解流程：
```
# Step 1: Profiling
for each expert e in MoE_layer:
    M_e = []  # B x C activation matrix
    for token batch in calibration_dataset:
        H = input_hidden_states  # [B, d_model]
        A_gate = SiLU(H @ W_gate)  # [B, C]
        A_up = H @ W_up            # [B, C]
        A = A_gate * A_up           # [B, C], element-wise
        M_e.append(A)

# Step 2: Partition Optimization (SA solver)
def simulated_annealing_partition(M, N_sub_experts, T0=100, alpha=0.995, I=100000):
    P = greedy_init(M, N_sub_experts)  # 按impact降序贪心分配, 维护负载均衡
    T = T0
    best_P, best_cost = P, compute_cost(P, M)
    for i in range(I):
        P_new = swap_random_neurons(P)  # 随机交换两neuron所属sub-expert
        cost_new = compute_cost(P_new, M)
        if cost_new < best_cost or random() < exp((best_cost - cost_new) / T):
            P = P_new
            if cost_new < best_cost:
                best_P, best_cost = P_new, cost_new
        T *= alpha
    return best_P  # {S_1: [neuron_ids], ..., S_N: [neuron_ids]}

# Cost function: sum of L1 norms of K deactivated sub-experts
def compute_cost(P, M):
    cost = 0
    for b in range(B):
        L_b = [||M[b, S_n]||_1 for S_n in P]  # per-sub-expert L1 norms
        cost += sum(smallest_K(L_b))  # top-K smallest norms
    return cost

# Step 3: Gating Reconstruction
C_co = B.T @ B  # co-activation matrix, B = binary_top_k(M)
for S_n in P:  # for each sub-expert
    centrality[n] = sum(C_co[n, j] for j in S_n)  # neuron centrality
    gate_neurons[S_n] = top_r(centrality[S_n])  # r=4 representative neurons
```
流程：校准数据前向传播收集激活→SA 求解器将每个 expert 的 C 个 neuron 划分到 N 个子 expert→选择每个子 expert 中 centrality 最高的 gate neurons→可选微调 router。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- MoE-Prism 论文在 PyTorch 2.7.0 + CUDA 12.6 上实现。校准使用 Wikitext-2-raw-v1，SA 参数 T0=100, α=0.995, I=100K 迭代。每个 expert 分为 N=4 sub-experts。
- 相关方法：DualSparse-MoE (2025) 也使用 post-training expert partitioning，但侧重于 tensor-level 和 neuron-level 双重稀疏性，在 ~25% drop rate 下仅损失 0.08%-0.28% 准确率。DERN (2025) 通过 expert 剪枝后分解为 neuron-level expert segment 再合并，在 50% expert sparsity 下提升 5% 推理性能。
- 核心价值：使 MoE 模型从 coarse-grained（如 k 只能选 1-2 个整数）升级为 fine-grained（如 k' 可以是 9-32，等效于原模型的 2.25-8 个 expert），提供 4 倍以上可区分操作点。

涉及论文标题：
- MoE-Prism: Disentangling Monolithic Experts for Elastic MoE Services via Model-System Co-Designs

---
