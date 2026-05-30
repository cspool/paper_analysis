## Oracle-Space-Based Routing (Oracle-MoE 路由)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Oracle-Space-Based Routing 是 Oracle-MoE (ICML 2025) 提出的 MoE routing 机制，用基于高层语义的"Oracle Space"替代传统的 token embedding 空间做专家路由。其核心洞察：连续 token 具有语义局部性（semantic locality），但 token-level embedding 被 token-identity 特征主导（如"the" vs "cat" 的 token ID 差异淹没语义相似性），导致相邻 token 被路由到不同专家，产生频繁的 expert swapping。Oracle-MoE 利用注意力分数（Q·K^T 内积）挖掘高层语义相关性：attn 分数高的 token 共享相似高层语义 → 归入同一语义组（Semantic Group） → 计算组嵌入作为 token 的语义表示 → 在 Oracle Space 上用 K-means 聚类（k=专家数）→ 每个聚类中心对应一个专家 → 同一组内所有 token 路由到同一专家。由于语义组嵌入比 token embedding 方差低得多（理论证明 Var(z_S) = (Σ_s + Σ_j)/n < Var(t_t)），连续 token 路由变化极小，CSD_oracle << CSD_token。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

Oracle-Space-Based Routing 的完整 pipeline（5 阶段）：

```
# ===== Stage 1: Warm-up Training =====
# 对 token-level MoE 进行短期预训练，获取合理的 token embeddings

# ===== Stage 2: Oracle Space Initialization =====
oracle_space = []
for each sampled_sequence in N_samples:  # N=8192
    # 2a: 计算注意力分数矩阵（使用 KV cache 中的 Q,K）
    A = Q @ K.T / sqrt(d)  # [T, T] lower-triangular
    A = softmax(A, dim=-1)

    # 2b: 贪心语义组划分 (Minimum Clique Cover on DAG, polynomial-time greedy)
    semantic_groups = []
    for t in range(T):
        merged = False
        for group in reversed(semantic_groups):
            if all(A[t][k] > epsilon for k in group):
                group.append(t)
                merged = True
                break
        if not merged:
            semantic_groups.append([t])

    # 2c: 计算语义组嵌入（组内 token embedding 均值）
    for group in semantic_groups:
        z_S = mean(token_embeddings[group])  # z_S ∈ R^d
        oracle_space.append(z_S)

# 2d: SVD 降维（保留 top-r 奇异值，提高计算效率）
U, S, Vt = SVD(oracle_space)
W_svd = Vt[:r, :]  # 降维变换矩阵 r << d

# 2e: K-means 聚类（k = num_experts）
reduced_embeddings = [W_svd @ z for z in oracle_space]
cluster_centers = KMeans(reduced_embeddings, k=num_experts)

# ===== Stage 3: Training/Prefill Routing =====
def oracle_moe_forward(token_embeddings, attention_scores):
    groups = partition_semantic_groups(attention_scores, epsilon)
    for group in groups:
        z_S = mean(token_embeddings[group])
        z_reduced = W_svd @ z_S
        expert_id = argmin(||z_reduced - c_k|| for k in range(num_experts))
        for token in group:
            output += expert_ffn[expert_id](token_embeddings[token])
    return output

# ===== Stage 4: Decode Routing =====
def oracle_moe_decode(new_token_embedding, kv_cache):
    q = W_Q @ new_token_embedding
    attn_scores = [q @ k_i / sqrt(d) for k_i in kv_cache.K]
    for group in existing_groups:
        if all(attn_scores[0][k] > epsilon for k in group):
            group.append(new_token_idx)
            z_S = mean(token_embeddings[group])
            z_reduced = W_svd @ z_S
            return argmin(||z_reduced - c_k||)
    new_group = [new_token_idx]
    z_reduced = W_svd @ new_token_embedding
    return argmin(||z_reduced - c_k||)

# ===== Stage 5: Expert Prediction Optimization =====
for layer in range(1, num_layers):
    pred_expert[layer] = expert_predictor[layer](hidden_states[0])
# 预加载预测的专家以隐藏 I/O 延迟（准确率 85%-95%，减少 10%-15% latency）
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 计算开销：路由计算从 token-level 的 W_g · t_t（矩阵乘法, 1e-4s）变为降维后的欧氏距离计算 ||W_svd · z_S - c_k||（2.5e-4s），相比单次 forward-backward pass（3.5s）可忽略。语义组划分利用 KV cache（decode 时已存在），无额外内存开销。
- 使用场景：(1) 内存受限的边缘设备（手机、Jetson 等）上部署 MoE LLM 推理；(2) 单用户 batch_size=1 场景（连续 token 语义局部性强）；(3) 需要减少 GPU memory footprint 但不可接受 token-level MoE 的高 swapping latency 的场景。
- 限制：(1) 需要 warm-up 阶段建立 Oracle Space（每层聚类 ~4 min，相对 tens of hours 预训练可忽略）；(2) 语义局部性假设在极端随机 token 序列中减弱（但实验显示即使跨数据集拼接的 diverse data，Oracle-MoE 仍每 100 token 仅换 12.2 次 vs Switch 90.54 次）；(3) 当前仅验证于 GPT-2 架构 MoE。
- 开源：论文未明确说明（ICML 2025 proceedings 无 GitHub 链接）。

涉及论文标题：
- Oracle-MoE: Locality-preserving Routing in the Oracle Space for Memory-constrained Large Language Model Inference

---
