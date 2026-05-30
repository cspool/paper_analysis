## Q-Filters

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Q-Filters 是一种训练无关（training-free）的 KV Cache 压缩方法，由 Godey et al. (2025) 提出，核心思想是利用 Query 和 Key 向量的几何特性——特别是 Query 分布的各向异性（anisotropy）——来估计 KV pair 的重要性，而无需访问注意力权重矩阵。方法分为两个阶段：

**离线校准阶段**：(1) 从校准数据集（如 Pile 子集，~3000 样本）前向传播，收集各层各注意力头的 Query 表示 $Q^h$；(2) 对每个头的 Query 矩阵进行 SVD 分解 $Q^h = U \Sigma V^\top$；(3) 取第一右奇异向量 $v_1$ 作为该头的 Q-Filter，并进行符号规范化 $v_1^+ = \operatorname{sgn}(\mathbf{1}u_1^\top)v_1$ 以确保正期望投影。对于 GQA，对每组共享 KV head 的 Query head 的 Q-Filters 取平均。

**推理阶段**：对每个注意力头，计算所有已存储 Key 向量在 Q-Filter 上的投影 $\langle K_t^h, v_1^+ \rangle$ 作为重要性得分，保留得分最高的 KV pairs，丢弃得分最低的。

理论基础是论文的定理 3.3：$\mathbb{E}_{Q_i^h}(\langle Q_i^h, K_j^h \rangle) \approx \kappa^h \langle K_j^h, u^h \rangle$，其中 $u^h$ 是 Query 分布的主方向（即 Q-Filter 方向），$\kappa^h > 0$ 为常数。该定理表明 Key 在 Query 主方向上的投影可近似期望注意力 logits，因此可作为 KV pair 重要性的有效估计。与 K-Norm（仅用 L2 范数）相比，Q-Filters 额外捕捉了 Key 向量在 Query 主方向上的角度分量 $\cos(K_j^h, u^h)$，Spearman 相关性显著更高。

关键特性：(1) 训练无关——无需参数更新；(2) FlashAttention 兼容——不访问注意力权重矩阵，仅需一次标量积投影；(3) 上下文无关——Q-Filters 仅依赖模型固有几何特性，不同校准数据集的 Q-Filters 高度一致（余弦相似度 > 0.9）；(4) 校准成本极低——Llama-3.2-70B 上 < 3 分钟（2×A100-80GB GPU）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**Q-Filters 完整 pipeline 伪代码**：

```
// ========== 离线校准阶段（仅执行一次） ==========
// 输入：预训练模型 M，校准数据集 D（如 Pile 子集, ~3000 样本）
// 输出：每层每头的 Q-Filter v_1^+

Q_bank = {}  // Q_bank[layer][head] = list of Query activations

for batch in D:  // 前向传播，收集 Query 表示
    for layer in M.layers:
        for head in layer.heads:
            Q_activations = get_query_activations(layer, head, batch)
            // Q_activations: [batch_size, seq_len, d_head]
            Q_bank[layer][head].append(Q_activations)

q_filters = {}
for layer in M.layers:
    for head in layer.heads:
        // Step 1: 随机采样并拼接
        Q_samples = random_sample(Q_bank[layer][head], 3000)
        Q_matrix = flatten(Q_samples)  // [N*samples, d_head]

        // Step 2: SVD 分解
        U, S, Vt = SVD(Q_matrix, full_matrices=False)
        v1 = Vt[0, :]  // 第一右奇异向量, shape: [d_head]

        // Step 3: 符号规范化（保证正期望投影）
        sign = sign(mean(U[:, 0]))  // 基于第一左奇异向量的均值符号
        q_filters[layer][head] = sign * v1  // v_1^+

// GQA 处理
if model_uses_GQA:
    for kv_head in range(num_kv_heads):
        q_start = kv_head * heads_per_group
        q_end = q_start + heads_per_group
        // 组内 Q-Filters 取平均
        q_filters[kv_head] = mean(q_filters[q_start:q_end])

// ========== 推理阶段 ==========
// 输入：KV Cache，最大容量 max_size，Q-Filters
// 输出：压缩后的 KV Cache

def q_filters_compress(kv_cache, max_size):
    for layer in M.layers:
        for head in layer.heads:
            K = kv_cache[layer][head].keys  // [seq_len, d_head]
            V = kv_cache[layer][head].values  // [seq_len, d_head]

            // 标量积投影：计算每个 Key 的重要性得分
            scores = K @ q_filters[layer][head]  // [seq_len]

            // Top-k 选择：保留得分最高的 KV pairs
            if seq_len > max_size:
                keep_indices = topk_indices(scores, max_size)
                kv_cache[layer][head] = (K[keep_indices], V[keep_indices])
```

**张量计算流程**：
给定 Key 矩阵 $K^h \in \mathbb{R}^{L \times d_H}$ 和 Q-Filter $v_1^+ \in \mathbb{R}^{d_H}$，重要性得分 $s = K^h \cdot v_1^+ \in \mathbb{R}^L$。保留 $s$ 最大的 $k$ 个 KV pairs。该操作仅涉及一次矩阵-向量乘法和一次 top-k 选择，计算复杂度 $O(L \times d_H)$，与 FlashAttention 完全兼容（无需物化注意力矩阵）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

开源实现：https://github.com/NathanGodey/qfilters ，基于 KVPress 库（https://github.com/kvpress）和 HuggingFace Transformers。使用方式：(1) 调用 KVPress 的 Q-Filters compressor 加载预计算的 Q-Filters；(2) 在 HuggingFace 推理 pipeline 中插入 compressor hook，每次 KV Cache 更新后自动执行 top-k 筛选。校准数据集推荐使用 Pile 或多域混合数据（Q-Filters 对数据域不敏感，跨域余弦相似度 > 0.9）。校准样本数建议 ~3000（边际收益在 1000 样本后递减）。压缩比支持 2× 到 64×，在 32× 压缩比下 NIAH 仍保持 99% 准确率。已知局限：对使用 QK-normalization（如 Olmo-2）或 attention bias（如 Qwen-2.5）的模型效果减弱，需适配分析。

涉及论文标题：
- Q-Filters: Leveraging QK Geometry for Efficient KV Cache Compression

---
