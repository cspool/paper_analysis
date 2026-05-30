## Wanda Pruning（Wanda 剪枝 / 权重-激活联合幅度剪枝）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Wanda（Pruning by Weights and Activations, Sun et al., 2023）是一种简单高效的 LLM 后训练 one-shot 剪枝方法。核心思想：weight importance score = |W_ij| × ||X_j||_2，即每个权重的幅度乘以其对应输入特征（列）的 L2 激活范数，按行（output neuron）独立比较并剪除 score 最低的权重。不需要权重更新（zero update）、不需要二阶 Hessian 信息、不需要反向传播。仅需一次前向传播收集激活统计 + 逐行 score 排序，计算复杂度远低于 SparseGPT。

数学表达：S_ij = |W_ij| · ||X_j||_2，对第 i 行保留 S_i 最高的 k%（或满足 N:M 模式）。等价于 SparseGPT 中 Hessian 逆的对角近似（忽略 Hessian 非对角元）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Wanda 剪枝伪代码（per layer）
# 输入: W [d_out, d_in], X [d_in, N_samples], sparsity_ratio s

# Step 1: 收集激活统计
X_norm = ||X||_2  # L2 norm per input channel [d_in]

# Step 2: 计算 importance score
for i in range(d_out):  # per output neuron
    for j in range(d_in):
        S[i][j] = |W[i][j]| * X_norm[j]

# Step 3: 逐行剪枝 (per-output comparison)
for i in range(d_out):
    k = (1-s) * d_in  # 保留的权重数量
    threshold = top_k(S[i], k)  # 第k大的score
    mask[i] = (S[i] >= threshold)
    W_pruned[i] = W[i] * mask[i]  # 零值更新，不调整剩余权重

# 2:4 结构化稀疏变体:
# 将 d_in 分成 d_in/4 个连续组，每组保留score最高的2个
for i in range(d_out):
    for g in range(0, d_in, 4):
        top2_idx = argsort(S[i][g:g+4])[-2:]  # 每组保留2个
        mask[i][g:g+4] = 0
        mask[i][g + top2_idx] = 1
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源：https://github.com/locuslab/wanda。支持 OPT、LLaMA、LLaMA-2 模型系列。使用 HuggingFace Transformers 加载模型，128 条 C4 校准样本，逐层前向传播收集激活统计后立即剪枝该层并释放激活，GPU 显存需求 = 单层权重 + 单层激活。在 SLiM 中是默认剪枝方法，作用于量化后的权重 W^Q 上，进一步施加误差 E_S。支持 unstructured (50%) 和 2:4 semi-structured (NVIDIA Sparse Tensor Core 兼容) 两种模式。

涉及论文标题：
- SLiM One-shot Quantization and Sparsity with Low-rank Approximation for LLM Weight Compression

---
