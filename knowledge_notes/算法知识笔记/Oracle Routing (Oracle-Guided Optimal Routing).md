## Oracle Routing (Oracle-Guided Optimal Routing)

术语解释
Oracle Routing 是 Duo-LLM 框架中提出的最优路由发现方法：对每条输入序列，穷举所有可能的 per-layer 路由选择（big/small/skip），在给定计算预算约束下选择最小化 perplexity 的路由路径，作为自适应计算的理论性能上界。

术语是什么？
给定 L 层模型，当仅考虑 big vs small 二选一时有 2^L 条可能路径，加入 skip 选项时有 3^L 条。Oracle 对每条路径执行完整 forward pass 并计算 cross-entropy loss，选择 loss 最低的路径。对于计算预算约束的场景（如仅允许 4/12 层使用 big module），仅搜索满足预算的路径子集。

Oracle 的发现揭示了现有 MoE router 训练的次优性：
- 最优 big layer 数量为 6/12（非 12/12），因为 C(12,6)=924 候选路径最多，增大了选到优质路径的概率
- 仅用 1 个 big layer 的 oracle 路由 perplexity 低于所有层都用 big module
- 预算有限（4 big layers）时优先将 big 分配给后层；预算充足（8 big layers）时优先给前层
- 后层存在"容量阈值"——满足后才值得给前层增加计算

从算法pipeline角度拆解术语：
```
# Oracle: Exhaustive Enumeration with Budget Constraint
# Model: L layers, each with big FFN and small FFN
# Budget B: max number of big layers per token

def oracle_optimal_route(x, labels, budget_B):
    L = 12  # number of layers
    best_loss = inf
    best_route = None
    
    # Enumerate all routes with exactly B big layers
    for route in combinations(range(L), budget_B):
        # route: indices of layers using big module
        mask = [1 if l in route else 0 for l in range(L)]
        
        # Forward pass with this route
        h = x
        for l in range(L):
            h_attn = Attention_l(h)
            if mask[l] == 1:
                h_ffn = BigFFN_l(h_attn)    # inner_dim=10240
            else:
                h_ffn = SmallFFN_l(h_attn)  # inner_dim=640
            h = h + h_ffn
        
        loss = CrossEntropy(h @ W_vocab, labels)
        if loss < best_loss:
            best_loss = loss
            best_route = route
    
    return best_route, best_loss
```

术语一般如何实现？如何使用？
- Oracle 需要 ground truth labels（计算 CE loss），因此仅适用于 holdout 评估，无法在生产环境使用
- 需要穷举所有可能路径（2^L 或 3^L），计算复杂度为 O(2^L × L)，实际仅适用于小规模研究（L=12 时约 4096 条路径）
- Oracle 的核心价值是作为理论上界：衡量 learned router 与最优路由之间的差距
- Duo-LLM 发现 trained router perplexity 更接近 fixed pattern 而非 oracle，gap 巨大
- 未来方向：训练 surrogate model 近似 oracle 的决策，避免穷举和 ground truth 依赖

涉及论文标题：
- Duo-LLM: A Framework for Studying Adaptive Computation in Large Language Models

---
