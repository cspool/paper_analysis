## Alternative Optimization for LLM-Guided MoE

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Alternative Optimization（交替优化）是 LEGO 提出的联合训练 LLM routing weights 和 graph expert 参数的优化策略。由于 LLM（Llama 3.1 8B）的推理成本远高于 GNN expert 的梯度下降更新，LEGO 不每步都更新 routing weights，而是每隔若干 epoch（E 个 epoch）才调用 LLM 重新评估 routing，在此期间固定 routing weights 仅优化 GNN expert 参数。此设计将 LLM 调用次数从 per-batch 降低到 per-epoch-interval，大幅降低训练开销。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
输入: 训练集 D, 预训练 LLM (Llama 3.1 8B), K 个 GNN experts {θ^1,...,θ^K}
输出: 训练好的 expert 参数

初始化所有 expert 参数
while not converged:                        // 外层循环
    // === Step A: 更新 Routing Weights（LLM 推理，低频）===
    for each sample in D (or subset):
        提取 hierarchical prompt
        各 expert 前向生成 candidate predictions
        LLM Judge 推理 → 选择最佳 expert
        // Label Smoothing (Eq.7)
        ω(k) = α if k == chosen else (1-α)/(K-1)
    
    // === Step B: 优化 Expert 参数（梯度下降，高频）===
    for epoch in 1..E:
        for each batch:
            使用当前 routing weights ω 组合 experts
            计算 ℒ = ℒ_mse + ℒ_div
            梯度下降更新 {θ^1,...,θ^K}（Adam, lr=0.0005）
```
关键设计：
- LLM 推理仅在 Step A 执行（低频），Step B 不涉及 LLM
- 间隔 E 的选择：E 过小 → LLM 调用过多成本高；E 过大 → routing weights 过时影响训练
- LLM 不需要微调（保持零样本能力），仅用作 routing function

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现方式：(a) LLM 调用通过 API（如 HuggingFace transformers 加载 Llama 3.1 8B 本地推理）；(b) Routing weights 在每个交替周期存储为哈希表（sample_id → one-hot expert choice）；(c) Step B 中固定 routing weights 使得 expert 训练等价于标准多任务学习
- 计算开销：LLM 推理成本 vs GNN 训练成本的比例决定了最优交替频率
- 类似方法：(a) EM（Expectation-Maximization）算法（固定 routing 优化 expert → 固定 expert 优化 routing）；(b) K-Means 的交替优化（分配聚类 → 更新中心）；(c) GAN 的交替训练（生成器/判别器）
- 优势：(a) 避免 LLM per-step fine-tuning 的高成本；(b) LLM 保持原始的零样本泛化能力（不被动态系统数据 overfit）；(c) 训练稳定性好
- 局限：(a) 论文未具体说明交替间隔 E 的值；(b) Step A 可能需要 sub-sampling 来处理大数据集

涉及论文标题：
- Marrying LLMs with Dynamic Forecasting A Graph Mixture-of-expert Perspective

---
