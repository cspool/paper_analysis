## SLiM-LoRA（显著性可逆可加低秩适配器）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SLiM-LoRA 是 SLiM 论文提出的 one-shot 低秩适配方法，利用具有可逆性和可加性的显著性函数（F(W) = diag(x)W，x 为校准集激活平均绝对值），通过 SVD 数学推导出低秩适配器的闭式解，无需任何迭代训练。核心创新：(1) 显著性函数 F 的可加性允许将压缩误差的显著性从原权重中隔离——F(-(E_Q+E_S)) = F(W^C - W)，再对显著性误差矩阵做 SVD；(2) 逆显著性变换 L = diag(1/x)·L̃, R = R̃ 将显著性空间的适配器转换回权重空间；(3) 显著性加权确保适配器优先修正对模型输出影响最大的权重通道（对比 Naive-LoRA 均匀最小化 Frobenius 范数）。

数学推导：目标 max_{L,R} ||F(W^C + LR)||² = min_{L,R} ||F(-(E_Q+E_S)) - F(LR)||²。由于 F 可加：F(LR) = diag(x)·L·R。SVD 分解：diag(x)·(-E_Q-E_S) = UΣV^T，取 rank r（论文默认 r=0.1d），得到 diag(x)·L = U_r·Σ_r^{1/2}, R = Σ_r^{1/2}·V_r^T，最终 L = diag(1/x)·U_r·Σ_r^{1/2}。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# SLiM-LoRA 算法（Algorithm 2 from paper）
# 输入: 原权重W, 压缩权重W^C, 校准输入X
# 输出: 低秩适配器 L, R

# Step 1: 计算总压缩误差
E_C = E_Q + E_S = W^C - W  # d_in × d_out

# Step 2: 构建显著性向量
x_tilde = mean(X, dim=batch)  # [d_in]
x = x_tilde + min(|x_tilde|)  # 避免零元素，保证F可逆

# Step 3: 计算误差显著性
S_C = diag(x) @ E_C  # [d_in, d_out]

# Step 4: SVD 低秩近似
U, Σ, V^T = SVD(S_C)
k = r × d  # r=0.1, rank = 10% of hidden dim
L_tilde = U[:, :k] @ sqrt(Σ[:k, :k])   # [d_in, k]
R = sqrt(Σ[:k, :k]) @ V^T[:k, :]       # [k, d_out]

# Step 5: 逆显著性变换
L = diag(1/x) @ L_tilde   # [d_in, k]

# 最终: W ≈ W^C + L @ R
```

对比 Naive-LoRA：Naive-LoRA 直接对 E_C 做 SVD——L,R = SVD(E_C) 截断——忽略不同权重通道对输出的差异化影响。SLiM-LoRA 的显著性加权使 top 通道（激活幅度大的通道）的误差被优先修正，提升 1-3% 准确率。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现集成于 SLiM 开源库。低秩适配器 rank 默认为 hidden_dim × 0.1。可选对适配器做 AbsMax group quantization（group size 128, 4-bit）减少 4× 存储/计算开销。推理时：主权重矩阵使用 Sparse Marlin kernel（2:4 稀疏 × 4-bit 量化），低秩适配器使用 Dense Quantized Marlin 或标准 PyTorch GEMM。内存开销分析（rank r=0.1d）：适配器存储 O(2rd²)，原始权重 O(d²)，当 r≪1 时开销可忽略。FLOP 分析类似。

涉及论文标题：
- SLiM One-shot Quantization and Sparsity with Low-rank Approximation for LLM Weight Compression

---
