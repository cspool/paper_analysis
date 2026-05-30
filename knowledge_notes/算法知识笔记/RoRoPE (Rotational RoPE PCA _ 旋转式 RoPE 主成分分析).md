## RoRoPE (Rotational RoPE PCA / 旋转式 RoPE 主成分分析)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

RoRoPE 是 TransMLA 论文提出的将 GQA 模型中分散在多个 KV head 的 RoPE 位置信息集中到第一个 attention head 的技术，是实现 GQA→MLA 转换中解耦 RoPE 的关键步骤。核心原理：当多个 KV head 合并为一个 latent head 后，每个 head 的同一 RoPE 频率维度在各自 head 内独立旋转。RoRoPE 利用正交旋转在 RoPE 内积下的不变性（Theorem/Equation 19）：对于第 l 个 RoPE 频率对应的 2D 子空间（real + imaginary），将各 head 中该子空间的分量拼接为 g 维向量，用正交矩阵 U_l ∈ R^{g×g} 旋转。因为 U_l^T U_l = I 且 real/imag 分量使用相同的 U_l，内积不变。选择 U_l 使得旋转后第一 head 捕获最大方差（PCA），其余 head 的位置信息可忽略，从而安全移除其 RoPE。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**RoRoPE 计算流程**（g 个 KV head，每 head d 维）：
```
// Step 1: 合并所有 KV head 为一个 latent head
// 引入 W_i^{UK} 作为 selector 矩阵（等价变换）

// Step 2: 对每个 RoPE 频率 l ∈ {1,...,d/2}：
For l = 1 to d/2:
    // 从所有 g 个 head 收集第 l 个 RoPE 子空间
    K_x_real = []  // shape: [N, g], real 分量
    K_y_imag = []  // shape: [N, g], imag 分量
    For head_idx in 0..g-1:
        K_x_real[:, head_idx] = key_activations[head_idx, 2l-1, :]  // 第 (2l-1) 维
        K_y_imag[:, head_idx] = key_activations[head_idx, 2l, :]    // 第 (2l) 维

    // 构建联合协方差矩阵
    Σ_l = K_x_real^T @ K_x_real + K_y_imag^T @ K_y_imag  // [g, g]

    // 特征分解得到最优正交旋转矩阵 U_l
    eigenvalues, U_l = eig(Σ_l)  // 按特征值降序排列

    // 旋转 W^K 和 W^{UK}（等价变换，不改变 attention 输出）
    // 旋转后第一 head 捕获 max variance → K_rope
    // 其余 head 位置信息可忽略 → K_nope

// Step 3: 移除 K_nope 的 RoPE
// K_rope（第 1 head）保留 RoPE；K_nope（第 2~g head）去除 RoPE
```

**内积不变性证明（Equation 19 核心）**：
```
S_l = cos((t-j)θ_l) · (q_x^T k_x + q_y^T k_y) + sin((t-j)θ_l) · (q_x^T k_y - q_y^T k_x)
S'_l = cos((t-j)θ_l) · ((U_l q_x)^T (U_l k_x) + (U_l q_y)^T (U_l k_y)) 
     + sin((t-j)θ_l) · ((U_l q_x)^T (U_l k_y) - (U_l q_y)^T (U_l k_x))
     = cos((t-j)θ_l) · (q_x^T U_l^T U_l k_x + q_y^T U_l^T U_l k_y) + ...
     = cos((t-j)θ_l) · (q_x^T k_x + q_y^T k_y) + sin((t-j)θ_l) · (q_x^T k_y - q_y^T k_x)
     = S_l  // 因 U_l^T U_l = I
```

术语一般如何实现？如何使用？

RoRoPE 在校准数据集（如 WikiText-2 子集）上离线执行。收集每层 key 激活值 → 构建联合协方差矩阵 → 特征分解得到 U_l → 旋转 W^K 和 W^{UK}。整个过程为等价变换（不改变 attention 输出），training-free。关键约束：同一 RoPE 子空间的 real 和 imag 分量必须使用相同的 U_l（否则内积不保持不变）。选择保留的主成分数 m：m=1 表示仅第一 head 保留 RoPE（最激进），m>1 表示更多 head 共享 RoPE 信息。TransMLA 实验证明 LLaMA-3-8B 上 RoRoPE 在 90% RoPE 去除率下仍保持 log-perplexity ≈ 2，而 MHA2MLA 方法升至约 6。

涉及论文标题：
- TransMLA: Multi-Head Latent Attention Is All You Need

---
