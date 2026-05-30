## Oracle-MoE: Locality-preserving Routing in the Oracle Space for Memory-constrained Large Language Model Inference

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：Oracle-MoE 提出了一种全新的 MoE 路由机制，用基于 Oracle Space（语义组嵌入空间）的路由替代传统的 token-level 路由。核心流程分为以下阶段：
    - **Warm-up 阶段**：先对 token-level MoE 模型进行短期预训练，随机采样 N 条数据提取每层的语义组嵌入（Semantic Group Embedding），构成初始 Oracle Space。语义组的划分基于因果注意力分数矩阵：对连续 token，若注意力分数 a_ij > ε（阈值），则归入同一语义组。语义组嵌入为该组内所有 token 嵌入的平均值 z_S = (1/|S|) * Σ t_j。为提高计算效率，对 Oracle Space 中的嵌入做 SVD 降维。
    - **预训练/推理 Prefill 阶段路由**：在 Oracle Space 上运行 K-means 聚类（k = 专家数量），每个聚类中心对应一个专家。对每条新数据，先根据注意力分数划分语义组，计算语义组嵌入（用相同的 SVD 变换矩阵降维），计算该语义组嵌入到各聚类中心的欧氏距离，将距离最近的聚类对应的专家分配给该语义组的所有 token：e_t = argmin_k ||z_S(t) - c_k||。
    - **推理 Decode 阶段路由**：新 token 到来时，根据其与已缓存 token 的注意力分数决定所属语义组，更新该语义组嵌入，路由到该语义组对应聚类中心的专家。语义组变化缓慢，因此连续 token 往往路由到相同专家，大幅减少 expert swapping。
    - **Expert Prediction 优化（可选）**：用第一层的 embedding 预测后续层的专家激活，预测准确率达 85%-95%，进一步减少 10%-15% 的 expert loading 延迟。
  - 实验比较：(1) Expert Activation 模式对比——Oracle-MoE vs Switch Transformer，可视化连续 token 生成时的专家激活变化；(2) Memory-Latency 曲线——四种模型规模（195M/295M/729M/2.06B）在不同 memory budget 下对比 FIFO/LRU/SwapMoE 策略的每样本处理延迟；(3) First Token Latency——765M 模型在 50% 内存预算下对比各策略的首 token 延迟；(4) Downstream Task 性能——TriviaQA (F1)、GLUE (Acc)、MAG (Acc)、Sci-Cite (Acc)、XSum (Rouge-1) 上的零样本性能对比；(5) 激活不一致性——DeepSeekMoE-16B、Qwen1.5-MoE-A2.7B、Switch Transformer 和 Oracle-MoE 各层的激活不一致性对比；(6) 细粒度专家 MoE 扩展实验（3B 参数，12 MoE 层，64 experts，top-6 激活）。

- 硬件平台是什么，配置是什么。
  - NVIDIA Jetson Xavier NX（边缘设备）：384 核 NVIDIA Volta 架构 GPU，8 GiB GPU 内存，约 21 TOPS AI 算力。

- 模型是什么。数据集和bench分别是什么。
  - 模型：基于 GPT-2 架构的 MoE 模型，四种规模：
    - 2*4(195M)：12 层 Transformer，2 个 MoE 层，每层 4 个专家（top-1），hidden dim=768
    - 4*8(295M)：12 层 Transformer，4 个 MoE 层，每层 8 个专家（top-1），hidden dim=768
    - 8*16(729M)：12 层 Transformer，8 个 MoE 层，每层 16 个专家（top-1），hidden dim=768
    - 9*24(2.06B)：24 层 Transformer，9 个 MoE 层，每层 32 个专家（top-1），hidden dim=1024
    - 扩展实验：3B 模型，12 MoE 层，64 experts，top-6 激活，hidden dim=1536，expert intermediate dim=1024（仿 DeepSeekMoE 设计）
  - Baseline：Switch Transformer（token-level MoE routing）
  - 数据集：OpenWeb-Text（预训练）；下游任务——Trivia QA（问答）、GLUE（分类）、MAG（分类）、Sci-Cite（分类）、XSum（摘要）
  - Benchmark 指标：Expert Activation Variation、Memory-Latency Curve、First Token Latency、下游任务性能指标（F1/Accuracy/Rouge-1）

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 论文未明确说明开源链接，ICML 2025 论文页面无 GitHub 仓库链接。
  - 算法 Pipeline 伪代码：

    ```
    # === Oracle Space Initialization (Warm-up Phase) ===
    # 在 token-level MoE 短期预训练后执行一次

    oracle_space = []  # 收集所有语义组嵌入
    for each sampled_sequence in N_samples:
        # Step 1: 计算注意力分数矩阵
        A = compute_attention_scores(sequence)  # [T, T] lower-triangular

        # Step 2: 贪心划分语义组
        semantic_groups = []
        for t in range(T):
            # 找到最大的 j < t 使得 A[t][j] > epsilon
            # 且 token j 所在组内所有 token k 都有 A[t][k] > epsilon
            merged = False
            for group in reversed(semantic_groups):
                if all(A[t][k] > epsilon for k in group):
                    group.append(t)
                    merged = True
                    break
            if not merged:
                semantic_groups.append([t])

        # Step 3: 计算语义组嵌入
        for group in semantic_groups:
            z = mean(token_embeddings[group])  # z ∈ R^d
            oracle_space.append(z)

    # Step 4: SVD 降维
    U, S, Vt = SVD(oracle_space)  # 保留 top-r 奇异值
    W_svd = Vt[:r, :]  # 降维变换矩阵

    # Step 5: K-means 聚类
    reduced_embeddings = oracle_space @ W_svd.T
    cluster_centers = KMeans(reduced_embeddings, k=num_experts)

    # === Oracle-MoE Routing (Training & Prefill) ===
    # 对每个输入序列：
    def oracle_moe_forward(token_embeddings):
        # Step 1: 划分语义组（同上述贪心算法）
        groups = partition_semantic_groups(attention_scores, epsilon)

        # Step 2: 计算语义组嵌入并降维
        for group in groups:
            z = mean(token_embeddings[group])      # 组嵌入
            z_reduced = W_svd @ z                  # SVD 降维

            # Step 3: 最近邻聚类中心 -> 专家选择
            expert_id = argmin(||z_reduced - cluster_centers[k]|| for k in range(num_experts))

            # Step 4: 组内所有 token 路由到同一专家
            for token in group:
                router_probs[token] = one_hot(expert_id, num_experts)

        # 标准 MoE FFN 计算
        output = sum(router_probs[i] * expert_ffn_i(token_embeddings[i])
                     for i in range(seq_len))
        return output

    # === Decode Stage Routing ===
    def oracle_moe_decode(new_token, kv_cache):
        # Step 1: 计算新 token 与缓存 token 的注意力分数
        attn_scores = compute_new_token_attention(new_token, kv_cache)  # [1, len(cache)]

        # Step 2: 决定语义组归属
        assigned_group = None
        for group in existing_groups:
            if all(attn_scores[0][k] > epsilon for k in group):
                assigned_group = group
                break
        if assigned_group is None:
            assigned_group = create_new_group([new_token_idx])

        # Step 3: 更新语义组嵌入
        assigned_group.append(new_token_idx)
        z = mean(token_embeddings[assigned_group])
        z_reduced = W_svd @ z

        # Step 4: 路由到最接近聚类中心的专家
        expert_id = argmin(||z_reduced - cluster_centers[k]||)
        return expert_id

    # === Expert Prediction Optimization ===
    # 用第一层 embedding 预测深层专家激活
    def predict_deep_experts(first_layer_hidden):
        pred_experts = []
        for layer in range(1, num_layers):
            expert_pred = expert_predictor[layer](first_layer_hidden)
            pred_experts.append(expert_pred)
        return pred_experts  # 准确率 85%-95%
    ```

  - 张量计算核心：传统 token-level MoE 的 gate 为 g(t) = softmax(W_g * t) ∈ R^N，选择 top-k；Oracle-MoE 替换为 z_S(t) = (1/|S(t)|) * Σ t_j（语义组内平均），e_t = argmin_k ||W_svd * z_S(t) - c_k||（最近聚类中心）。对比之下，Oracle-MoE 的路由输入从 per-token embedding（受 token-identity 主导，高方差）变为 per-semantic-group embedding（保留高层语义，低方差），使得连续 token 的 CSD_oracle << CSD_token，从而减少 expert swapping。
