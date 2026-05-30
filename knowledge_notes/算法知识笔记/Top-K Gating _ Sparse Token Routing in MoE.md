## Top-K Gating / Sparse Token Routing in MoE

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Top-K Gating (Token Routing) 是 MoE 架构中决定每个 token 由哪些 experts 处理的核心机制。给定输入 token x ∈ R^H，gate function 计算该 token 对所有 E 个 experts 的 affinity scores（通过线性投影 x·W_g + optional noise），然后选择 top-k 个最高分的 experts 作为该 token 的"激活 experts"。未被选中的 experts 不参与该 token 的计算——这就是"稀疏激活"（sparse activation）的来源。

形式化 (FlashMoE 使用 top-2 routing, capacity factor=1.0):
- Gate logits: $l = xW_g \in \mathbb{R}^E$
- Affinity scores: $g_i = \text{softmax}(l)_i$ 或直接 $g_i = \text{softmax}(\text{topk}(l, k))_i$（仅对 top-k 做 softmax，其他为 0）
- Top-K indices: $E_i = \{e_1, e_2, ..., e_k\}$ 其中 $g_{e_1} \ge g_{e_2} \ge ... \ge g_{e_k}$ 且对任意 $e \notin E_i$, $g_e = 0$
- Expert capacity 限制: 若某 expert 已收到 C 个 token，则此后即使被选为 top-k 也跳过该 token（token 被"丢弃"）

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# ===== Top-K Gating Pseudocode (标准 MoE) =====
def topk_gating(x, W_g, k=2, capacity_factor=1.0, noise_std=0.0):
    """
    x: [S, H] input tokens
    W_g: [H, E] gate weight matrix
    Returns: routing_table mapping expert→[(token_idx, weight)]
    """
    S, H = x.shape
    E = W_g.shape[1]
    C = int(S * k * capacity_factor / E)
    
    # 1. Gate logits
    logits = x @ W_g  # [S, E]
    
    # 2. Optional: add noise (for exploration during training)
    if noise_std > 0:
        noise = randn(S, E) * noise_std
        logits = logits + noise
    
    # 3. Softmax to get affinity scores
    gate_scores = softmax(logits, dim=-1)  # [S, E]
    
    # 4. Top-K selection: 每 token 选 k 个最高分 expert
    # Using topk: returns top k values and indices along dim=-1
    topk_scores, topk_experts = topk(gate_scores, k, dim=-1)
    # topk_scores:   [S, k] → 归一化的 gating weights
    # topk_experts:  [S, k] → expert indices
    
    # 5. Build routing table T_φ
    # T_φ[e][c] = (token_idx, combine_weight) 或 (token_idx, weight)
    T_phi = {e: [] for e in range(E)}
    
    for token_idx in range(S):
        for j in range(k):
            e = topk_experts[token_idx, j]
            w = topk_scores[token_idx, j]
            if len(T_phi[e]) < C:  # Expert not yet full
                T_phi[e].append((token_idx, w))
            # else: token overflow → dropped for this expert
            # (auxiliary loss encourages balanced routing to minimize drops)
    
    return T_phi, gate_scores
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Top-K Gating 的变体：(1) **Top-1 routing** (Switch Transformer)——仅选 1 个 expert，最稀疏但需更多 experts 保持模型质量；(2) **Top-2 routing** (GShard, FlashMoE)——最广泛使用，平衡稀疏性和模型容量；(3) **Top-8 routing** (DeepSeek-V3)——极高稀疏度 (8/256) 需配合 shared experts 保持质量；(4) **Expert choice routing**——反向：expert 选 top tokens 而非 token 选 experts，保证 load balance 无需 auxiliary loss。

Auxiliary Load Balancing Loss: $L_{aux} = E \cdot \sum_{e=1}^E f_e \cdot P_e$，其中 $f_e = \frac{1}{S} \sum_{i=1}^S \mathbb{1}[e \in E_i]$ 为 expert e 的实际 token 比例，$P_e = \frac{1}{S} \sum_{i=1}^S g_{i,e}$ 为 gate 平均分配给 expert e 的概率。当 f_e 和 P_e 均为 1/E 时 loss 最小（完全均匀）。实际训练中 this loss 乘以 small coefficient (0.01) 添加到主 loss。

涉及论文标题：
- FlashMoE: Fast Distributed MoE in a Single Kernel
