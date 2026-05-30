## Kronecker Product Affine Transformation（Kronecker 仿射变换）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Kronecker Product Affine Transformation 是 FlatQuant 提出的核心技术，使用 Kronecker 乘积将大矩阵分解为两个小矩阵的乘积来构造高效的仿射变换。给定需要变换的维度 n，将其分解为 n = n₁ × n₂（取 n₁+n₂ 最小化以最小化参数量），构造两个轻量矩阵 P₁∈R^{n₁×n₁}、P₂∈R^{n₂×n₂}，总变换为 P = P₁ ⊗ P₂。利用 Kronecker 乘积的向量化性质：vec(V)(P₁⊗P₂) = vec(P₁^T V P₂)，将原本需要 O(n²) 的大矩阵乘法转化为两次小矩阵乘法。参数从 n² 降至 n₁²+n₂²（≤ n/2 倍节省），计算量节省 √n/2 倍（取 n₁=n₂=√n 时最优）。例如 LLaMA-2-7B 的 hidden_dim=4096 分解为 n₁=n₂=64，intermediate_dim=11008 分解为 n₁=64, n₂=172。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以 LLaMA-2-7B hidden_dim=4096、n₁=n₂=64、单个 token 为例：

```
输入: X ∈ R^{1×4096}, P₁ ∈ R^{64×64}, P₂ ∈ R^{64×64}
W ∈ R^{4096×4096}

// 变换矩阵构造
P = P₁ ⊗ P₂  // 显式 Kronecker 乘积（仅在概念层面）

// 激活侧在线仿射变换
X̃ = X.reshape(1, 64, 64)               // [k, n₁, n₂]
X' = P₁^T @ X̃[0] @ P₂                   // [64×64 matmul] × 2 = 2×64³ ops
// 对比：使用完整 P 的 X' = X @ P 需要 4096² = 16.8M ops

// 权重侧逆变换（离线融合到量化权重）
P₁_inv = V₁ @ Σ₁^{-1} @ U₁^T            // SVD 分解求逆
P₂_inv = V₂ @ Σ₂^{-1} @ U₂^T
W̃ = W.reshape(4096, 64, 64)
W' = P₁_inv^T @ W̃[i] @ P₂_inv            // [per output channel]
W_q = per_channel_quantize(W')          // INT4
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
FlatQuant 中 Kronecker 仿射变换在 PyTorch 中实现：(1) P₁、P₂ 通过 Cayley 参数化（torch.nn.utils.parametrizations.orthogonal）保证正交性和可逆性；(2) 使用 SVD 分解在 FP16+AMP 下稳定计算逆矩阵（P^{-1}=VΣ^{-1}U^T，非对角元素误差 1×10^{-6} vs 直接求逆的 1×10^{-3}）；(3) 最优分解取 n₁,n₂ = arg min(n₁+n₂) s.t. n₁n₂=n, n₁≤n₂；(4) P₁、P₂ 通过 AdamW 优化器在逐 block MSE 损失上学习，随机初始化。LLaMA-2-7B 所有在线变换的总 FLOPs 仅占 FP16 模型的 2.61%，总内存开销仅 3.41MB。

涉及论文标题：
- FlatQuant: Flatness Matters for LLM Quantization

---
