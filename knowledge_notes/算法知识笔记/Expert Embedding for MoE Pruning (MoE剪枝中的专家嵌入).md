## Expert Embedding for MoE Pruning (MoE剪枝中的专家嵌入)

术语解释
Expert Embedding for MoE Pruning 是将每个 MoE expert 映射为一个固定维度的特征向量的技术，通过将 expert 的功能行为编码为可比较的向量表示，使相似度计算和聚类成为可能。核心思想是用 expert 的实际输出（而非 router logits 或参数值）作为其"功能签名（functional signature）"。

术语是什么？
C-PRUNE 提出的 expert embedding 计算方法：φ(f_i) = E_{x~D}[1/K Σ_{k=1}^K f_i(x_k)] ∈ R^d。具体而言：在 task-specific calibration 数据集 D_calib 上对每个 sample 做 forward pass，对每个 expert f_i，取 K 个 token 的输出向量的平均值作为该 expert 的功能嵌入。维度 d = hidden_dim（如 DeepSeek-V2-Lite 的 d=2048）。这一定义基于如下观察：两个 expert 如果对相同输入产生相似的输出分布，则它们在功能上是冗余的，可以被合并或剪除。

与 router-based 方法（如 Seer Prune 使用 gate activation frequency）的区别：router-based 方法间接通过"哪些 expert 被选中"来判断重要性，而 expert embedding 直接衡量 expert 的"计算行为"——两个不同 expert 可能都被频繁激活（高 gate frequency），但计算输出几乎相同（功能冗余），router-based 方法无法检测这种情况。

从算法pipeline角度拆解术语。
```
# Expert Embedding Computation
Input: MoE model M, calibration dataset D_calib
       samples_per_expert K (e.g., K=100)

expert_embeddings = {}  # layer_id -> list of phi vectors

for each MoE layer l in range(L):
    phi_list = []
    for each expert f_i in layer l (i in 1..N):
        outputs = []
        for each batch x in D_calib:
            # Forward pass through this expert only
            h = f_i(x)  # expert FFN output: [batch, seq, d_model]
            # Average over K randomly selected token positions
            indices = random.sample(range(seq_len * batch), K)
            outputs.append(mean(h.view(-1, d_model)[indices], dim=0))
        
        # phi_i: average expert output over all calibration samples
        phi_i = mean(outputs, dim=0)  # shape: [d_model]
        phi_list.append(phi_i)
    
    expert_embeddings[l] = phi_list  # shape: [N, d_model]

# Use: pairwise cosine similarity -> affinity matrix
for layer l:
    phi = expert_embeddings[l]  # [N, d]
    for i, j in pairs:
        sim = phi[i] @ phi[j] / (norm(phi[i]) * norm(phi[j]))
        A[i,j] = sigmoid(alpha * sim)
```
注解：
- φ(f_i) 的维度 = d_model（expert 输出维度），与 expert 内部 FFN 维度无关
- K 是超参数，控制采样的 token 数量；论文中 K 的值"论文未明确说明"
- Calibration 数据集 D_calib 使用 task-specific samples
- expert embedding 仅需一次 offline computation，计算量 O(L × N × |D_calib| × K × d_ffn²)

术语一般如何实现？如何使用？
- **Offline Computation**：在剪枝前一次性计算所有 expert 的 embedding，存储为 [L, N, d] 张量
- **相似度计算**：基于 cosine similarity 或 Euclidean distance。Cosine similarity 对输出 scale 不敏感，更适合
- **聚类输入**：embedding 矩阵直接作为 hierarchical/k-means clustering 的输入特征
- **与其他方法的关系**：HC-SMoE (ICML 2025) 也使用 expert output-based similarity；Mosaic Pruning 使用 expert performance profile 作为 embedding
- 限制：(1) embedding 质量依赖 calibration 数据的 representativeness；(2) 大模型时 embedding 计算开销显著；(3) 不同 task domain 可能需要不同的 embedding

涉及论文标题：
- Cluster-Driven Expert Pruning for Mixture-of-Experts Large Language Models

---
