## Expert Merging

术语解释
专家合并是将MoE模型中功能相似的多个专家合并为一个专家的技术，以减少专家总数、降低参数量和推理计算量，同时尽量保持模型性能。

术语是什么？
专家合并基于观察：MoE中不同专家可能学到相似的功能或特征，可以被合并而不显著损害性能。
- **Branch-Train-Merge**：在不同数据子集上独立训练模型的不同部分，避免了传统大模型训练中的大规模多节点同步
- **Branch-Train-Mix**：异步并行训练多个种子LLM以专精于不同领域，然后合并MoE层的参数创建统一模型，经过二次微调提升性能
- **MC-SMoE**：基于路由策略将专家分组，每组合并为一个专家（加权和），然后对合并后专家使用低秩分解
- **HC-SMoE**：层次聚类合并，无需重训练，任务无关
- **MEO**：drop-in replacement算法——先合并选中的专家参数，再高效计算
- **DEK**：在特征空间识别并分组相似专家，在权重空间合并
- **C-PRUNE Parameterized Expert Merging**：在 hierarchical clustering 分组后，每个 cluster 内专家通过 affinity-weighted averaging 合并为一个。权重 ω_i = exp(γ·A_ik) / Σ exp(γ·A_jk)，其中 A_ik 是 expert i 与 cluster center 的 cosine affinity score，温度 γ 控制融合锐度（γ 越大越接近 hard selection）。同时路由权重通过均值 + exploration noise 更新：Ŵ_k = mean(W_i) + ε·N(0,I)
- **HyperMoE**：利用未选中专家的上下文信息补偿迁移到特定专家的性能损失
- **LiteMoE**：基于应用特征保留最关键的专家，合并次要专家，获得最终稀疏模型——适用于移动设备

从算法pipeline角度拆解术语。
```
# Expert Merging on Forward Pass (MEO drop-in replacement)
def moe_forward_with_merge(x, experts, router, K):
    # 标准Top-K选择
    θ = Softmax(R(x))
    selected = TopK(θ, K)
    
    # 合并选中的专家（而非分别计算）
    W_merged = sum(θ[i]/sum(θ[j] for j in selected) * experts[i].weight 
                   for i in selected)
    # 单次FFN计算（替代K次）
    y = FFN_with_weight(x, W_merged)  # σ(x @ W_1) @ W_2
    return y

# 静态合并（预处理阶段）
def static_expert_merging(experts, similarity_threshold):
    groups = hierarchical_clustering(experts, 
              distance=cosine_similarity(weight_flattened))
    merged_experts = []
    for group in groups:
        W_merged = weighted_average([experts[i].weight for i in group])
        merged_experts.append(W_merged)
    return merged_experts
```

术语一般如何实现？如何使用？
- 静态合并：离线完成，推理时直接使用合并后的模型
- 动态合并（MEO）：推理时在线合并选中的专家，节省K次FFN计算的kernel启动开销
- 合并可在权重空间（参数平均）或特征空间（聚类）进行
- 通常需要微调恢复精度

涉及论文标题：
- A Survey on Inference Optimization Techniques for Mixture of Experts Models
- A Survey on Mixture of Experts in Large Language Models
- BTS Harmonizing Specialized Experts into a Generalist LLM
- Cluster-Driven Expert Pruning for Mixture-of-Experts Large Language Models

**BTS 论文中的 "Expert Merging" 范畴**：BTS 论文将 Expert Merging 定义为一类方法的统称（与 Expert Upcycling 相对）：在合并阶段 Seed 和 Expert 模型参数**保持冻结**，仅训练少量新参数（如 stitch layers 或 adapters）。具体包括以下变体：

- **Model Soup (Wortsman et al., 2022)**：均匀平均 Seed 和所有 Expert 的权重，无需任何训练。形式：$\theta_{\text{soup}} = \frac{1}{n+1}\sum_{i=0}^n \theta_i$
- **BTM (Li et al., 2022)**：对 Seed 和 Expert 的输出 logits 使用 Bayes 规则加权 ensemble。无需训练。每位专家的权重 $w_i \propto P(\text{input} | \text{expert}_i)$ 由均匀先验下的 Bayes 规则估计。
- **Expert Routing**：训练一个线性路由器 $R \in \mathbb{R}^{\dim \times n}$，基于 prompt 平均 embedding 选择单个模型处理整个序列。路由器决策即模型选择，所有后续 token 路由到同一模型。
- **BAM with Adapters (Zhang et al., 2024)**：在 MoE/MoA 架构中，每个 Attention/FFN Expert 输出后插入线性 adapter $W_{\text{proj}_i} \in \mathbb{R}^{\dim \times \dim}$，仅训练 router 和 adapters。
- **BTS**：通过插入 Stitch Layer 在 Seed（Hub）和 Expert（Spoke）之间建立双向可学习连接（详见 Stitch Layer 条目）。

BTS 论文的关键对比维度：Expert Merging（264M 可训练参数、11B 总参数）vs Expert Upcycling（7.2B+ 可训练参数），前者保持模块性和可解释性。

**补充（来自 EEP 的 Continuous Expert Merging via Evolutionary Strategy）**：EEP 将 Expert Merging 作为一种无梯度 post-pruning knowledge recovery 方法，与 pruning phase 组合为一个统一的进化搜索框架。

- **连续合并矩阵**：Expert Merging Matrix WEM ∈ R^{E'×E} 的元素从 Pruning Phase 的 one-hot (0/1) 过渡到 Merging Phase 的连续实数值。第 j 个新 expert: θ'_j = {Σ_i ω_ji W₁i, Σ_i ω_ji W₂i, Σ_i ω_ji W₃i}，其中 ω_ji 来自 WEM 第 j 行。
- **与 Router 解耦**：Merging Phase 中 WRM ≠ WEM（不再相等），允许路由权重和 expert 权重独立优化。WRM 也变为连续值，实现了更灵活的路由映射。
- **与 Model Soup 的区别**：Model Soup 对所有模型做均匀平均，EEP 通过进化搜索学习最优的非均匀加权系数，可包含负值（负系数表示某些 expert 的知识对下游任务无益）。
- **进化参数**：160 iterations，Mutation=element-wise Gaussian noise added to merging coefficients，Crossover=沿 retained expert 维度组合父代 merging coefficients。
- **关键结果**：Merging 在 Pruning 基础上额外提升 5%-7% 准确率（如 SQuAD: 75.2%→80.6%），整个过程不需要梯度计算，可在仅支持推理的设备上完成。
- **权重分组**：为减少优化参数数量，expert weights 按深度均匀分为 4 groups（或 32 groups per dataset），组内共享 merging coefficients。

涉及论文标题：
- A Survey on Inference Optimization Techniques for Mixture of Experts Models
- A Survey on Mixture of Experts in Large Language Models
- BTS Harmonizing Specialized Experts into a Generalist LLM
- Cluster-Driven Expert Pruning for Mixture-of-Experts Large Language Models
- Efficient Expert Pruning for Sparse Mixture-of-Experts Language Models: Enhancing Performance and Reducing Inference Costs

---
