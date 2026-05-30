## Token Condensation (令牌凝聚)

术语解释
Token Condensation 是 LUFFY 分布式 MoE 训练系统中提出的通信优化技术：在 MoE 的 dispatch phase 中，识别被路由到同一 expert 的高度相似 token，仅传输 representative token，其余 token 在 combine phase 使用 representative 的 expert 输出替代，从而消除冗余的跨 GPU 通信。

术语是什么？
Token Condensation 基于两个关键观察：(1) 在 MoE 训练中，被路由到同一 expert 的 token 之间存在显著相似性——MoE-TransformerXL 中约 62% 的 token 对相似度超过 0.75，MoE-BERT-Large 第六 block 中约 57% 的 token 对相似度超过 0.55；(2) token 相似度在通过 expert 后高度保留——约 95% 的 token 对在 expert 计算后相似度变化小于 0.2。

核心流程：
1. 将 token 建模为全连接图（node=token, edge weight=similarity）
2. 通过 Fast Similarity Measurement 快速计算边权重
3. 根据 Adaptive Threshold h_t 删除低相似度边
4. 在每个连通分量中保留 degree 最高的代表性 token
5. Dispatch 阶段仅传输 representative tokens
6. Expert 计算仅处理 representative tokens（减少计算量）
7. Combine 阶段使用 representative 的输出替代被凝聚 token 的输出

从算法pipeline角度拆解术语：
Token Condensation 在 MoE 训练 pipeline 中的位置和执行流程：

```
=== MoE Layer Forward Pass with Token Condensation ===

输入: token embeddings X [N, d] after self-attention

Step 1 - Gate Routing (标准):
    gate_logits = X @ W_gate        # [N, num_experts]
    gate_probs = SoftMax(gate_logits)
    expert_ids = TopK(gate_probs, k=2)  # 每 token 选 top-2 experts

Step 2 - Token Graph Construction:
    G = (V, E) where V = {tokens}, E = all pairs
    for each edge (u, v):
        # 2a: 不同 expert → 边权重=0 (直接跳过)
        if expert_ids[u] != expert_ids[v]:
            weight[(u,v)] = 0
        
        # 2b: 历史相似度查找 (O(1))
        elif s_prev[(u,v)] > S1: weight[(u,v)] = 1  # 极端相似
        elif s_prev[(u,v)] < S2: weight[(u,v)] = 0  # 极端不相似
        
        # 2c: 计算真实余弦相似度 (仅不确定的少量对)
        else:
            weight[(u,v)] = cosine(X[u], X[v])

Step 3 - Adaptive Threshold:
    l_norm = (loss_ini - loss_prev) / loss_ini
    h_t = 1.0 / (1.0 + exp(l_norm))
    # 早期: h_t ≈ 0.73 (保留大部分 token)
    # 后期: h_t ≈ 0.27 (凝聚更多 token)

Step 4 - Graph Pruning & Component Selection:
    删除 weight < h_t 的边 → 稀疏图
    对每个连通分量:
        rep = argmax(degree(node))  # 保留连接最多的 token
        token_to_token[node] = rep  # 其他 token 映射到代表

Step 5 - Condensed Dispatch:
    for expert in experts:
        tokens_to_send = {rep}  # 仅 representative tokens
        all_to_all_send(tokens_to_send, target_gpu)

Step 6 - Reduced Expert Computation:
    # 更少的 tokens → 更少的 FLOPs
    expert_out = expert_ffn(received_tokens)  # [N' << N, d]

Step 7 - Expanded Combine:
    for token in all_tokens:
        if token in token_to_token:
            output[token] = expert_out[token_to_token[token]]
        else:
            output[token] = expert_out[token]
```

术语一般如何实现？如何使用？
- 实现基于 DGL (Deep Graph Library) 构建 token 图，利用 GPU 加速图操作
- Fast Similarity Measurement 将 O(N²·d) 的 naive pairwise 计算降低到仅少量不确定对需真实余弦计算
- Adaptive Threshold 通过 sigmoid 函数将 loss 下降量映射为 [0, 1] 区间的阈值
- 参数 S₁ 和 S₂ 控制历史相似度判断的激进程度：减小 S₁ 增加凝聚率但可能影响收敛，增大 S₂ 保留更多 token 但减少通信节省
- 局限性：仅适用于训练阶段（需访问 token embeddings 和 gate 输出）；token 相似度假设在推理时可能不成立

涉及论文标题：
- Communication-Efficient Sparsely-Activated Model Training via Sequence Migration and Token Condensation

---
