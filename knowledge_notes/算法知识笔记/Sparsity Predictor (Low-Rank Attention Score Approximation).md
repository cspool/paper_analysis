## Sparsity Predictor (Low-Rank Attention Score Approximation)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Sparsity Predictor是DSV中用于在线预测attention score分布并识别critical KV pairs的轻量级组件。每个self-attention模块配备两个额外的低秩可训练矩阵W_Q^lr和W_K^lr（形状d × d_lr，其中d_lr ≪ d_k，如d_k=128时d_lr=16），将输入投影到远小于原始Q/K的低维空间：Q_lr = H @ W_Q^lr, K_lr = H @ W_K^lr。用低秩乘积Q_lr·K_lr^T近似原始attention score分布QK^T（注意：近似pre-softmax的QK^T而非softmax后的值，因为softmax单调，pre-softmax的相对顺序和softmax后的相对顺序一致）。预测器参数量极小（3B模型<10M参数）。

从算法pipeline角度拆解，predictor的训练和使用：
```
# Predictor 结构
W_Q_lr ∈ R^{d × d_lr}    # low-rank query projection, d_lr << d_k
W_K_lr ∈ R^{d × d_lr}    # low-rank key projection

# Forward: predict attention scores
Q_lr = H @ W_Q_lr        # [S, d_lr]
K_lr = H @ W_K_lr        # [S, d_lr]
approx_scores = Q_lr @ K_lr^T   # [S, S] - 近似QK^T分布

# Predictor training loss (detached from main graph!)
L_approx = 0.95 * CosLoss(approx_scores, Q @ K^T) \
         + 0.05 * NormLoss(approx_scores, Q @ K^T)
# CosLoss: cosine similarity, 保持相对大小关系
# NormLoss: L2 norm difference, 保持整体scale一致
# 从主计算图detached → predictor gradient不影响DiT参数

# Stage 2: 使用predictor识别critical KV
K_per_query = ceil((1 - sparsity_head) * S)
crit_indices = FusedTopK(approx_scores, k=K_per_query)  # [H, S, K]
O = SparseAttention(Q, K, V, crit_indices)
```

设计动机：(1) 避免物化完整attention score矩阵（O(S²)内存）；(2) 在fused kernel中完成低秩MM和top-K选择，中间不物化完整矩阵；(3) 不影响fused attention kernel（FlashAttention）的优化路径；(4) 低秩近似带来了约O(S·d_lr)的额外计算开销，远小于O(S²)的attention计算本身。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现：每个attention module增加两个nn.Linear(d, d_lr)（无bias），predictor参数独立于主模型参数（不同步梯度），需手动管理predictor的parameter replication和gradient synchronization（因为FSDP可能切分不同参数）。训练策略采用sample-based方法减少query计算量（随机采样部分query进行loss计算）。Cosine loss + Norm loss组合相比MSE对attention score的scale变化更robust。训练到avg(L_approx) < 0.01后进入Stage 2。

涉及论文标题：
- DSV: Exploiting Dynamic Sparsity to Accelerate Large-Scale Video DiT Training
