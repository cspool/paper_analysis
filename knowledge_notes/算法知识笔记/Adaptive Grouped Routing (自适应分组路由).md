## Adaptive Grouped Routing (自适应分组路由)

术语解释
Adaptive Grouped Routing 是 AT-MoE 提出的两层层次化 MoE 路由机制，先用群组级路由（Group-level Routing）在专家类别间分配全局权重，再用组内路由（Within-group Routing）在组内进行局部专家权重归一化，实现对复杂多意图复合指令的层次化、可解释的专家权重分配。

术语是什么？
标准 MoE 路由仅使用单层 top-K 门控（Linear→Softmax→TopK），无法区分复合指令中不同子任务的重要性。Adaptive Grouped Routing 通过两层矩阵解决此问题：

**第一层 - 群组路由（Group Routing）**：使用群组路由向量 W_G ∈ R^{N_dim × N_G}，将输入嵌入 x 映射为跨组权重向量：
```
M_G = x @ W_G                       # [1, N_G]
W'_G = SoftMax(M_G / τ_g)          # 温度 SoftMax，得各组权重
```
N_G 为专家组的数量（如医学场景分为功能类、领域知识类、风格类三个组）。

**第二层 - 组内路由（Within-group Routing）**：使用组内路由矩阵 W_D ∈ R^{N_G × N_M}，其中 N_M 为每组最多专家数。对组内专家做逐列 SoftMax：
```
M_D = W'_G @ W_D                    # 用群组权重加权
W'_D = col_wise_SoftMax(M_D / τ_d)  # 每列（每组）独立 SoftMax
```
不足 N_M 专家的组用 -inf padding，使其不参与 SoftMax 计算。

最终路由函数 F_G(W̄_e) = Σ_j W'_D[j] · LoRA_j(x)，各任务特定 LoRA 专家按层次化权重加权求和。

从算法pipeline角度拆解术语。
```
# Adaptive Grouped Routing Forward Pass
def adaptive_grouped_routing(x, W_G, W_D, lora_experts, group_ids, tau_g, tau_d):
    """
    x: token embedding [d]
    W_G: group routing vector [d, N_G]
    W_D: within-group routing matrix [N_G, N_M]
    lora_experts: list of LoRA modules, each has group_id and expert_id
    """
    # Step 1: Group-level routing
    M_G = x @ W_G                                    # [N_G]
    W_prime_G = softmax(M_G / tau_g)                 # [N_G], 跨组权重

    # Step 2: Within-group routing
    M_D = W_prime_G @ W_D                            # [N_M], 组内专家logits
    # 对不足N_M的组pad -inf
    for g in range(N_G):
        actual_experts = count_experts_in_group(g)
        M_D[actual_experts:] = -inf
    W_prime_D = softmax(M_D / tau_d)                 # [N_M], 组内专家权重

    # Step 3: 将所有专家按层次权重加权
    expert_out = 0
    for j, (lora, group_id) in enumerate(zip(lora_experts, group_ids)):
        weight = W_prime_G[group_id] * W_prime_D[j]  # 层次权重 = 群组权重 × 组内权重
        expert_out += weight * lora(x)
    
    return expert_out
```

以医学复合查询"四肢无力+开中药方"为例，路由过程：
1. 群组路由 → 功能组 0.6, 领域组 0.3, 风格组 0.1
2. 功能组内路由 → 诊断 0.5, 处方 0.4, 分诊 0.1
3. 领域组内路由 → 消化内科 0.5, 中医 0.4, 放射科 0.1
4. 风格组内路由 → 严谨型 0.8
5. 最终诊断 LoRA 权重 = 0.6 × 0.5 = 0.3（可追溯、可解释）

术语一般如何实现？如何使用？
- 训练阶段：先在各任务数据上训练 LoRA 专家（冻结 LLM + 路由器未训练），再冻结所有 LoRA 专家，训练 W_G 和 W_D 路由矩阵
- 推理阶段：路由模块根据输入动态计算 W'_G 和 W'_D，加权融合多个 LoRA 的输出
- 不同 Transformer 层有独立的 W_G^(l) 和 W_D^(l)（layer-wise routing）
- 通过预合并通用 LoRA W_p 和平衡参数 λ 融合任务特定专家与通用专家
- 适用于需要可解释性和可控性的多任务场景（如医学诊断、法律咨询）
- 目前无开源代码，论文未提供实验验证

涉及论文标题：
- AT-MoE: Adaptive Task-planning Mixture of Experts via LoRA Approach
