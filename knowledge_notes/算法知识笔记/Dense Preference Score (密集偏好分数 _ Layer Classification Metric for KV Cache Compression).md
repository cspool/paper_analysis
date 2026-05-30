## Dense Preference Score (密集偏好分数 / Layer Classification Metric for KV Cache Compression)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Dense Preference Score (P) 是 TailorKV 提出的离线度量指标，用于判断 Transformer 每一层注意力应该采用量化还是稀疏选择的压缩策略。核心公式：使用最近的 n_q 个 query 向量 Q_{last_q} ∈ R^{n_q × d_h} 与全部 key 向量 K ∈ R^{n × d_h} 计算完整 attention score 矩阵，取每行 Top-k attention scores 之和的补数。即 P = n_q - Σ_{(i,j)∈Î} Â_{i,j}，其中 Î 是每 query 行的 Top-k 位置集合。P 值高 → 注意力分布均匀（密集） → quantization-friendly；P 值低 → 注意力集中在少量 token → sparsity-friendly。阈值 τ=0.2 通过 synthetic LongBench 实验确定，该 metric 跨数据集一致（同一模型的 P 分布在 different datasets 下几乎相同），因此可离线一次标定。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
// === 离线标定阶段（仅执行一次） ===
// 输入：校准数据集 prompt，模型权重
// 超参：n_q（最近query数），k（top-k数），τ（阈值=0.2）

for each Transformer layer l in 0..L-1:
    // Step 1: 收集 prefilling 阶段的 attention 信息
    Q_last_q = recent_n_q_query_vectors(l)    // shape: (n_q, d_h)
    K_all = all_key_vectors(l)                 // shape: (n, d_h)

    // Step 2: 计算完整 attention score
    A_hat = Softmax(Q_last_q @ K_all.T / sqrt(d_h))  // shape: (n_q, n)

    // Step 3: 计算 Dense Preference Score
    for i in 1..n_q:
        topk_vals[i] = Top_k(A_hat[i, :], k)
    P_l = n_q - sum(topk_vals)                 // Eq.(8)

    // Step 4: 层分类
    if P_l > τ:
        layer_type[l] = "Quantization-Friendly"   // 浅层（layer 0, 有时 layer 1）
    else:
        layer_type[l] = "Sparsity-Friendly"       // 深层

// 结果示例（Llama-3.1-8B, L=32）:
// Q = {0} → 仅 layer 0 是 quantization-friendly
// Llama-2-7B / Yi-6B / Yi-9B: Q = {0, 1} → layer 0 和 1 是 quantization-friendly
```

直觉解释：P 捕捉了"有多少 attention mass 不在 top-k 中"——密集层中 attention 分散，top-k 只覆盖少量 mass → P 大；稀疏层中 attention 集中在 few tokens → P 小。

术语一般如何实现？如何使用？

实现：(1) 使用校准数据集（如 synthetic LongBench 的一个子集）运行一次完整 prefill → 记录每层最近 n_q 个 query 和全部 key → 计算 P；(2) τ 通过 grid search 在 LongBench 验证集上确定（TailorKV 发现 τ=0.2 对所有模型通用）；(3) 离线标定结果（即每层的类型 label）在 serving 时作为静态配置使用，不需要在线重新计算。

适用场景：任何需要对 Transformer 层进行差异化 KV cache 压缩策略的方法（不仅限于量化+稀疏，也可扩展到不同剪枝率、不同卸载策略等）。与 PyramidKV 的"金字塔信息漏斗"假设互补——PyramidKV 假设信息从浅层向深层集中（所有层用同一策略只是不同预算），TailorKV 发现浅层和深层需要根本不同的策略（浅层适合保留全部信息的量化、深层适合只保留关键 token 的稀疏）。

涉及论文标题：
- TailorKV: A Hybrid Framework for Long-Context Inference via Tailored KV Cache Optimization

---
