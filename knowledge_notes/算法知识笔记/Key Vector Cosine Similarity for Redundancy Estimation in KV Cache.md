## Key Vector Cosine Similarity for Redundancy Estimation in KV Cache

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

R-KV 提出的 redundancy estimation 机制：通过对 key vectors 做 L2 归一化后计算余弦相似度矩阵 S = K̄·K̄^T∈R^{n×n}（Eq.5），显式测量 token 间的语义冗余。核心洞察：冗余 token 的 key vectors 在向量空间中高度聚集（因为解码时 model 对重复内容产生相似的 key 表示），通过余弦相似度可以在向量空间层面（而非 token 表面）捕捉语义冗余，无需分析文本内容本身。

具体流程（Eq. 5-6）：(1) L2 归一化：K̄_i = K_i / (||K_i||₂ + ε)，去量纲化；(2) 相似度矩阵：S = K̄·K̄^T，S_{i,j}∈[-1,1] 表示 token i 和 j 的 key vector 方向夹角余弦；(3) 对角线置零：S_{i,i}=0，防止 token 被标记为"与自身冗余"；(4) 保留最近 β 个高相似 token：对每个 token i，找到 S_{:,i} > T（T 为相似度阈值）的高相似 token 集合 I_i，保留其中 β 个最新位置（largest indices）的 token 不被标记冗余——因为即使内容重复，最新出现的变体离当前解码位置最近，contextual relevance 最高；(5) 平均相似度：S̄_i = (1/n)·Σ_j S_{j,i}，衡量 token i 与多少其他 token 相似；(6) Softmax 归一化：R = softmax(S̄)，得到 per-token redundancy score ∈ [0,1]，总和为 1。

计算复杂度：O(n²·d) for similarity matrix computation（n 个 key vectors 两两内积），O(n²) for similarity matrix 处理。总 overhead 为 O(B_budget²)，在 B_budget=1536 时约 2.4M 元素相似度矩阵，相比 attention 计算 O(B_budget·B_buffer·d) 量级仍较小。

从算法pipeline角度拆解：

```python
def redundancy_estimation(K_cand, T, beta, eps=1e-8):
    """
    K_cand: [n, d_head] (n个候选token的key vectors)
    T: 相似度阈值 (论文未明确给出具体值)
    beta: 最近保留的高相似token数 (论文未明确给出具体值)
    """
    n, d = K_cand.shape
    
    # Step 1: L2归一化
    K_norm = K_cand / (K_cand.norm(dim=-1, keepdim=True) + eps)  # [n, d]
    
    # Step 2: 余弦相似度矩阵
    S = K_norm @ K_norm.T  # [n, n], S_ij = cos(k_i, k_j)
    S.fill_diagonal_(0)    # 抑制自相似
    
    # Step 3: 标记高相似pair并保留最近β个
    B = (S > T).float()  # [n, n], 二值化
    for i in range(n):
        similar_j = B[:, i].nonzero().squeeze(-1)  # 与i高相似的token索引
        if len(similar_j) <= beta:
            continue  # 不够β个，全保留
        
        # 保留最近β个（largest indices → 最新的token）
        recent_beta = similar_j.topk(k=beta, largest=True).values
        S[recent_beta, i] = 0  # 不标记为冗余
    
    # Step 4: 平均相似度
    S_bar = S.mean(dim=0)  # [n], 每个token被多少token"相似于"
    
    # Step 5: Softmax归一化
    R = torch.softmax(S_bar, dim=0)  # [n], Σ R_i = 1
    
    return R  # 高R_i → token更冗余
```

关键参数：T（similarity threshold）——太低会导致几乎所有 token 对被视为相似，太高会导致无 token 被标记冗余。R-KV 论文未明确给出 T 的具体值（仅说明为"fixed hyperparameter"）。β——控制即使 token 高度重复，最新出现的变体仍被保留。论文也未给出具体值。实际使用时可能需要 calibration。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

该 redundancy estimation 是纯 PyTorch 实现，无需特殊硬件或 kernel。核心操作为矩阵乘法（K_norm @ K_norm.T），与标准 attention 的 Q@K^T 类似但 size 为 n×n（n 为 candidate token 数，如 B_budget=1536）。该操作在 GPU 上高效，n=1536 时 ~2.4M 元素矩阵乘法仅需 ~10ms（A100）。主要使用场景：针对推理模型（如 DeepSeek-R1）的长 CoT 解码——这些模型产生的输出含大量重复内容，redundancy estimation 是识别并淘汰冗余 token 的关键。尤其适合数学推理任务（如 MATH-500, AIME）中常见的反复自我验证模式。

局限性：(1) 对 key vector 高度依赖——若模型训练时未产生明显的 key vector 聚集（如短输出任务），redundancy estimation 可能不必要；(2) similarity threshold T 和 beta 的选择影响性能，需针对不同模型/任务 calibrate；(3) O(n²) 的相似度矩阵计算在极端大的 B_budget（如 >10K）时可能成为 bottleneck。

涉及论文标题：
- R-KV: Redundancy-aware KV Cache Compression for Training-Free Reasoning Models Acceleration
