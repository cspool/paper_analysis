## Token Importance Prediction via Attention Distillation

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Token Importance Prediction via Attention Distillation 是一种通过知识蒸馏训练轻量级预测器来估计 LLM 解码过程中每个 token 对当前 query 重要性的方法。核心思想：冻结预训练 LLM 作为 teacher，用一个极小的外部 MLP 预测器（<1% LLM 参数量）作为 student，蒸馏 teacher 每层每个 head 的 masked causal attention distribution。训练时，teacher 产出每层每 head 的真实注意力分布 A_true，student 预测低维 importance queries 并与降维后的真实 KV-cache keys 做点积得到 A_pred，最小化 softmax 化后的 cross-entropy loss：L_CE = -E[Σ P_k log(Q_k)]，其中 P = softmax(A_true), Q = softmax(A_pred)。推理时，预测器输出 token 重要性分数，在固定 budget 下选择 top-k token 参与注意力计算。训练数据仅需 1K 长度的通用语料（C4、FineWeb-Edu、CodeParrot、BABILong），预测器通过 key-cache 投影机制泛化到 64K 长上下文。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Token Importance Prediction 插入 LLM decode pipeline 中，位于每层 attention 计算之前：

```
# 训练阶段
For each training sequence (length ≤ 1K):
    # 冻结 LLM 前向，收集 teacher attention
    with torch.no_grad():
        for layer in LLM.layers:
            Q, K, V = project(layer.input)
            A_true[layer] = masked_attention(Q, K)  # teacher logits
    
    # 仅训练预测器参数
    for producer_layer in {0, G, 2G, ...}:
        H = hidden_states[producer_layer]             # [B, L, E]
        Q_imp = MLP(LayerNorm(H))                      # [B*H, G, L, d']
        
        for consumer_layer in [producer+1, producer+G]:
            slot = (consumer_layer - 1) % G
            K_proj = K_cache[consumer_layer] @ W_K[l]  # [B, H_kv, L, d']
            A_pred = Q_imp[:, slot] @ K_proj.transpose  # [B*H, L, L]
            
            # 蒸馏 loss：teacher-student 交叉熵
            loss += CE(softmax(A_true[consumer_layer] + mask),
                       softmax(A_pred + mask))
    
    loss.backward()  # 仅更新 MLP 和 W_K 参数

# 推理阶段（decode step t）
if t % prediction_interval == 0:
    H = hidden_states[producer_layer][:, -1:, :]  # 仅最新 token
    Q_imp = MLP(LayerNorm(H))                       # [B*H, G, 1, d']
    for consumer_layer in consumer_layers:
        K_proj = K_cache[consumer_layer] @ W_K
        scores = Q_imp[:, slot, 0, :] @ K_proj.T    # [H, L_kv]
        # 排除 sink + window tokens，取 top-B
        selected = topk(scores[candidate_mask], B)
        migrate_to_important_buffer(selected)

# Attention: 拼接 [Sink | Important | Local_Window]
attn_out = FlashAttention(Q, K[selected_all], V[selected_all])
```

Key dimensions: d'=16 (interaction dimension), G=4 (producer frequency), MLP hidden=512. Predictor params: 29.4M for Llama-8B (0.368%), 20.9M for Qwen2.5-7B (0.299%). Training cost: ~9h on single A6000 for Llama-8B.

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现方式：在 HuggingFace Transformers 模型上挂载外部预测器模块。预测器结构为 LayerNorm → Linear(E→512) → GELU → Linear(512→B*H*G*d') → Reshape。K-cache 投影矩阵为每层独立参数 W_K^(l) ∈ R^{D×d'}（使用 GQA 时 H_kv < H，需 broadcast）。训练时使用 row-subsampling 加速：仅对序列尾部 R 个位置的 query 计算 loss（R << L），将 O(L²) 降为 O(RL)。推理时预测器在 producer layer 处每 prediction_interval 步触发一次，中间步复用上次选择。可与 FlashAttention 标准 kernel 无缝集成。代码开源：https://github.com/abdelfattah-lab/TokenButler。

涉及论文标题：
- TokenButler: Token Importance is Predictable
