## Expert Selection Frequency

术语解释
Expert Selection Frequency 是 MoE 模型推理期间统计的各 expert 被 router 选中的频率分布，用于分析 MoE 的任务偏好、sparsity 特性，以及指导 expert 压缩策略（量化位宽分配、剪枝决策）。

术语是什么？
EAC-MoE 论文的核心发现：
1. **任务内相似性**：同一任务类别（QA/CR、Math、Code、特定语言）的不同数据集，expert 选择频率 pairwise cosine similarity >0.8
2. **任务间差异性**：不同任务类别之间 expert 选择频率 cosine similarity 显著较低
3. **稀疏性**：少数 expert 被高频选中（>30%），多数 expert 很少被选（<1%），但"重要 expert"的身份因任务而异
4. **结论**：不能用单一静态校准集确定 expert 重要性——对 QA/CR 重要的 expert 可能对 Code 不重要

计算方式：对 MoE layer m，数据集 d，统计 expert i 被选中次数 C(m,d,i)，归一化后展平为向量 P(d)，计算 cosine similarity。

从算法pipeline角度拆解术语：
```
=== Expert Selection Frequency 统计 ===
输入: 数据集 D, MoE 模型 (L 个 MoE layer, N 个 expert/layer)
输出: 每层每个 expert 的选择频率分布

For each layer m in [0..L-1]:
    C[m] = [0] * N                              # selection counts
    For each sequence in D:
        For each token t:
            logits = router_W @ h_t
            selected = TopK(Softmax(logits), K)  # K = top-K per token
            for expert_id in selected:
                C[m][expert_id] += 1
    # 归一化
    total = sum(C[m])
    P[m] = C[m] / total                         # P[m][i] = freq of expert i

# 展平所有层为一维向量用于 similarity 计算
P_flat(D) = concat([P[0], P[1], ..., P[L-1]])
Sim(D_i, D_j) = cosine_similarity(P_flat(D_i), P_flat(D_j))
```

术语一般如何实现？如何使用？
- 在 MoE 推理时自然产生（只需记录 router 的 TopK 选择），无额外计算开销
- 应用：(1) 指导 PESF 动态剪枝阈值；(2) 分析模型对任务的 specialization 程度；(3) 检测 expert 负载均衡情况
- PMQ/BSP 用此频率决定混合精度位宽分配，但 EAC-MoE 证明了这会导致跨任务过拟合
- Mixtral-8x7B 的 expert 选择分布更均匀（稀疏性弱），Deepseek-moe-16b-base 有 64 expert 则稀疏性更强

涉及论文标题：
- EAC-MoE: Expert-Selection Aware Compressor for Mixture-of-Experts Large Language Models
