## Expert Pruning (MoE)

术语解释
MoE专家剪枝是减少MoE模型中专家参数量的技术，通过移除不重要的专家（structured pruning）或专家权重（unstructured pruning）来降低模型大小和计算量。

术语是什么？
由于专家占MoE模型参数的绝大部分（如Mixtral-8x7B中占96%），专家剪枝是MoE压缩的首要目标：
- **结构化剪枝（Structured Pruning）**：直接移除整个专家，减少专家数量
  - TSEP：针对下游任务移除非专业专家，只保留和微调专业专家
  - NAEE：在小校准集上评估专家组合，最小化精度损失
  - SEER-MoE：使用heavy-hitters计数法进行专家剪枝
- **非结构化剪枝（Unstructured Pruning）**：剪除专家内部的权重
  - MoE-Pruner：基于magnitude × activation × router weight的剪枝准则
- **专家合并（Expert Merging）**：将多个相似专家合并为一个
  - MC-SMoE：基于路由策略分组后合并
  - HC-SMoE：层次聚类合并，无需重训练
- 混合方法：STUN结合结构化+非结构化剪枝；MoE-Compression统一框架
- **C-PRUNE (Cluster-driven Expert Pruning)**：两阶段自适应剪枝框架。Phase 1 在每层内用 hierarchical agglomerative clustering 按 expert 参数相似度（cosine affinity）将功能冗余的 expert 分组成 cluster；Phase 2 跨所有层建立 unified importance score 进行全局剪枝，考虑深层 expert 更同质的趋势（depth penalty）。剪枝后 expert 通过 **parameterized expert merging**（affinity-weighted averaging）合并，路由权重同步更新。20% pruning rate 下 DeepSeek-V2-Lite 15.7B→13.0B，MMLU 仅降 1.4%。
- **DiEP (Differentiable Expert Pruning)**：将专家选择从离散搜索转化为连续优化。定义 intra-layer importance α 和 inter-layer importance β，通过交替梯度优化（3:1 比例）学习 per-expert 和 per-layer 的重要性权重。全局排序 s_i^(l) = α·β 后统一剪枝，自动实现 non-uniform pruning。仅需 128 calibration samples（C4），额外参数 ~0.01%。Mixtral 8×7B 50% sparsity 下保留 92% 性能，pruning time 0.23h（vs NAEE 1.31h exhaustive search）。NeurIPS 2025。

从算法pipeline角度拆解术语。
```
# Structured Expert Pruning (TSEP-style)
def prune_experts(model, task_dataset, target_sparsity):
    # 1. 评估每个expert对下游任务的重要性
    for each expert e in MoE layers:
        importance[e] = evaluate_on_task(model \ {e}, task_dataset)
    # 2. 移除最不重要的expert
    keep = topk(importance, ceil(N * (1-sparsity)))
    # 3. 微调保留的expert
    finetune(model, task_dataset, keep)
    return model

# Unstructured Expert Pruning (MoE-Pruner)
def moe_pruner_weight_score(W_expert, x_activation, router_weight):
    # W: [d_ffn, d_model]
    # score = |W| ⊙ mean(|x|) ⊙ router_weight (per output neuron)
    score = abs(W_expert) * mean(abs(x_activation), dim=0) * router_weight
    mask = score > percentile(score, sparsity)
    return W_expert * mask

# C-PRUNE: Cluster-Driven Expert Pruning
def c_prune(model, D_calib, K_layer, K_global, R_target):
    # Phase 1: Layerwise Expert Clustering
    for layer l in model.moe_layers:
        # Step 1: Compute expert embeddings φ(f_i)
        phi = []  # shape: [N, d]
        for expert f_i in layer.experts:
            # Average expert output over calibration samples
            phi_i = mean([f_i(x_k)/K for x_k in D_calib], dim=0)
            phi.append(phi_i)
        # Step 2: Affinity matrix A (cosine similarity)
        A = zeros(N, N)
        for i, j in pairs:
            A[i,j] = sigmoid(alpha * cos_sim(phi[i], phi[j]))
        # Step 3: Hierarchical agglomerative clustering
        clusters = agglomerative_cluster(A, n_clusters=K_layer)
        # Step 4: Parameterized merging within clusters
        for cluster C_k in clusters:
            omega = softmax([gamma * A[i, center] for i in C_k])
            theta_merged = sum(omega_i * theta_i)
    # Phase 2: Global Cluster Pruning
    scores = []
    for layer l, cluster c:
        # Depth penalty: deeper layers get lower scores
        score = importance(c) / (1 + beta * depth_penalty(l))
        scores.append((l, c, score))
    prune = bottom_k_percentile(scores, R_target)
    for (l, c) in prune:
        remove_cluster_from_layer(l, c)
    return model
```
结果：C-PRUNE 在 20% pruning rate 下显著优于 Random（MMLU avg 16.28→44.94）、Seer Prune（28.76）和 Group&Merge（32.03），接近 Base 模型（45.58）。50%-96.875%的稀疏率，结构化剪枝可直接减少expert数量从而减少激活参数。

术语一般如何实现？如何使用？
- 通常需要校准数据集或下游任务数据指导剪枝
- 部分方法需要微调恢复精度（如TSEP），部分无需微调（如EEP、HC-SMoE）
- C-PRUNE 使用 task-specific calibration data 计算 expert embedding（φ(f_i)），剪枝后可选择 task-specific fine-tuning 恢复精度
- 剪枝后的MoE模型可直接在标准框架中推理（structured pruning减少expert数，unnstructured需稀疏kernel支持）
- C-PRUNE 开源：https://github.com/Fighoture/MoE_unsupervised_pruning

涉及论文标题：
- A Survey on Inference Optimization Techniques for Mixture of Experts Models
- A Survey on Mixture of Experts in Large Language Models
- Cluster-Driven Expert Pruning for Mixture-of-Experts Large Language Models
- DiEP: Adaptive Mixture-of-Experts Compression through Differentiable Expert Pruning
- EAC-MoE: Expert-Selection Aware Compressor for Mixture-of-Experts Large Language Models

**补充（来自 EAC-MoE 的 PESF）**：PESF 是一种在线动态 expert 剪枝方法，从 expert 粒度（而非 token 粒度）剪枝。推理时统计当前序列的 expert 选择频率，将选择次数低于阈值 (l×K/N)×α 的 expert 完全跳过。α=0.3 时几乎无损（<0.5% 准确率），α=0.7 时加速 1.3-1.47×。关键创新：基于"同一任务类别内 expert 偏好高度相似"的观察动态统计，而非使用静态先验。仅适用 prefill 阶段。详见 PESF 和 Expert Selection Frequency 词条。

**补充（来自 EEP 的 Gradient-Free Evolutionary Expert Pruning）**：EEP (Efficient Expert Pruning) 引入了一种全新的 expert 剪枝范式——使用无梯度进化策略搜索最优剪枝模式，无需任何参数更新即可在仅支持推理的设备上执行。核心创新包括：

- **参数化搜索空间**：引入 Router Mapping 矩阵 WRM ∈ R^{E'×E} 和 Expert Merging 矩阵 WEM ∈ R^{E'×E}，将剪枝决策转化为矩阵元素搜索问题。两矩阵在 Pruning Phase 中约束为 one-hot rows（每行仅一个元素为1），且 WRM = WEM，确保仅选择/保留 experts。Router 变换：G' = WRM·softmax(ZW_G)（E→E' 维路由权重降维）。
- **进化搜索流程**：(a) 随机初始化 one-hot 矩阵构成初始种群 P；(b) 每轮按 fitness F(W·Θ)（下游任务准确率）排名，选 Top M_CP 个体进入候选父代集 CP；(c) Crossover：随机组合两个父代的 expert 维度；(d) Mutation：随机替换 pruned expert 为其他 expert（Pruning Phase）或加 Gaussian noise（Merging Phase）；(e) 迭代 Epochs 轮。
- **两阶段设计**：Pruning Phase（40 iterations，仅选择最佳 expert 子集，不更新参数）→ Expert Merging Phase（160 iterations，WRM/WEM 解耦后从离散 0/1 过渡到连续值，通过 weighted sum 合并 knowledge）。
- **反直觉发现**：剪枝 50% experts 在 SQuAD 上准确率从 53.4% 升至 75.4%（不做任何参数更新）。原因：Router 从 8 个 experts 的复杂划分简化为 4/2 个 experts 的决策，re-normalized routing weights 使路由更精准。
- **双重使用场景**：减少 total experts（节省显存：8→4 减 47%，8→2 减 71%）和减少 active experts（Top-2→Top-1：prefill 加速 1.63×）。
- 代码开源：https://github.com/imagination-research/EEP

涉及论文标题：
- A Survey on Inference Optimization Techniques for Mixture of Experts Models
- A Survey on Mixture of Experts in Large Language Models
- Cluster-Driven Expert Pruning for Mixture-of-Experts Large Language Models
- DiEP: Adaptive Mixture-of-Experts Compression through Differentiable Expert Pruning
- EAC-MoE: Expert-Selection Aware Compressor for Mixture-of-Experts Large Language Models
- Efficient Expert Pruning for Sparse Mixture-of-Experts Language Models: Enhancing Performance and Reducing Inference Costs

---
