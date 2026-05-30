## State-Aware Weight Sorting for SSMs (状态感知SSM权重排序)

术语是什么？
State-Aware Weight Sorting 是 UniQL 针对 Mamba（State Space Model）块提出的一种结构化权重排序策略。与 Transformer 的 attention 机制不同，SSM 的计算核心是线性递归 $(h_t = \Delta_t A_t h_{t-1} + \Delta_t B_t x_t,\ y_t = C_t h_t)$，其内部状态 $h_t$ 携带长序列信息。UniQL 将 Mamba 块的计算拆分为两个子公式进行权重排序：

1. **SSM 输入掩码 $\mathcal{M}$（B-C 排序）**：$\mathcal{M} = \phi(X\mathbf{W}_C) (\Delta \odot \phi(X\mathbf{W}_B))^{\top}$。排序 $\mathbf{W}_B^g$ 和 $\mathbf{W}_C^g$ 的列，考虑输入依赖的离散化变量 $\Delta^g$ 通过广播外积对 $\mathbf{B}^g$ 的调制作用。
2. **SSM 状态 $\mathcal{H}$（z-x-o 排序）**：$\mathcal{H} = \Delta A \mathcal{H}(h_0) + \Delta B X_\phi$。从已计算出的 SSM 状态 $\mathcal{H}^i$ 收集相关性矩阵 $C_{\mathcal{H}} = \mathcal{H}^{i\top} \mathcal{H}^i$，计算 ridge leverage scores 来排序 $\mathbf{W}_z^i$、$\mathbf{W}_x^i$ 和 $\mathbf{W}_o^i$。

从算法pipeline角度拆解：
B-C 排序的完整流程（Algorithm 4）：
```
# 输入: W_B^g, W_C^g ∈ R^{D_h × D_s}, 校准激活 X_h, 输入依赖步长 Δ^g
For each SSM group g = 1, ..., G_s:        # Mamba2: G_s=1
    B = conv1d(X_h @ W_B^g)                 # [T, D_s], 1D causal conv + SiLU
    C = conv1d(X_h @ W_C^g)                 # [T, D_s]
    ΔB = Δ^g ⊗ B                            # [H_m^g, T, D_s], 广播外积
    
    # 计算状态维度 D_s 的相关性
    ΔC_B = mean_i((ΔB^i)^T @ (ΔB^i))       # [H_m^g, D_s, D_s]
    C_C = mean_i((C^i)^T @ (C^i))           # [D_s, D_s]
    
    # 多SSM头的范数得分汇总
    s = zeros(D_s)
    For k = 1, ..., H_m^g:
        s += ||(ΔC_B^k)^{1/2}|| ⊙ ||C_C^{1/2}||
    
    S_BC = I[:, argsort(s)]                # 排序矩阵
    W_B^g, W_C^g = W_B^g @ S_BC, W_C^g @ S_BC  # 列重排
```

z-x-o 排序（Algorithm 5）：从 SSM 状态 $\mathcal{H}^i \in \mathbb{R}^{H_m \times (T \times D_s) \times D_{hd}}$ 收集 $C = \frac{1}{N}\sum_{i=1}^N \mathcal{H}^{i,j\top}\mathcal{H}^{i,j}$，然后计算 ridge leverage scores 排序 $\mathbf{W}_z$、$\mathbf{W}_x$、$\mathbf{W}_o$。

与 MLP 的 ridge leverage score 排序的关键区别：B-C 排序的得分是 B 和 C 两组相关性范数的逐元素乘积（考虑了 SSM 的双路径特性），而 z-x-o 排序直接从 SSM 状态相关性开始，称为"状态感知"排序。

术语一般如何实现？如何使用？
该策略专门为 Mamba 架构设计，需要理解 SSM 的线性递归方程和输入依赖离散化。实现要求：能前向传播 Mamba 块（如 Mamba2）以收集 $\mathbf{B}$、$\mathbf{C}$ 的卷积输出和 $\mathcal{H}$ 状态，并实现广播外积和相关性矩阵计算。对 Mamba2-8B，B-C 排序处理 1 个 SSM group（$G_s=1$），$H_m=64$ 个 SSM head，$D_s=128$ 状态维度。整个排序在 A6000 上耗时 16 分钟。

涉及论文标题：
- UniQL: Unified Quantization and Low-rank Compression for Adaptive Edge LLMs
