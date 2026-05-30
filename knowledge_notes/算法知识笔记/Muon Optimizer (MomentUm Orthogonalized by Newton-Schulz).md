## Muon Optimizer (MomentUm Orthogonalized by Newton-Schulz)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Muon (MomentUm Orthogonalized by Newton-Schulz) 是由 Keller Jordan et al. (2024) 提出的一种专用于神经网络中 2D 矩阵参数（如 attention 投影矩阵、FFN 权重矩阵）的优化器。其核心思想：将标准 SGD momentum 累积的梯度动量矩阵通过 Newton-Schulz 迭代进行近似正交化（≈ (M M^T)^(-1/2) M = U V^T，即用 momentum 矩阵的左右奇异向量之积替代逐元素更新），使得每次更新的奇异值全部近似为 1，消除更新在少数主导方向上的过拟合，迫使参数在所有奇异向量方向上等强度学习。非矩阵参数（如 RMSNorm 的 gamma/bias、embedding、LM head）仍用 AdamW 处理。Muon 仅维护 1 个动量 buffer（vs AdamW 的 m 和 v 两个），内存开销减半。Moonshot AI (Liu et al. 2025) 将 Muon 扩展到大规模 LLM 训练，提出三项关键技术：weight decay、Consistent Update RMS 和 Distributed Muon，训练了 16B MoE 模型 Moonlight，证明 Muon 在 compute-optimal 设置下仅需 AdamW 约 52% 的训练 FLOPs 即可达到相同性能。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Muon 优化器在 LLM 训练中的完整 pipeline（结合 Liu et al. 2025 的扩展）：

```
# 每个训练 step，对每个矩阵参数 W ∈ R^{A×B}：
# Step 1: 计算梯度 G = ∇L(W_{t-1})  (标准反向传播)
# Step 2: Nesterov-style momentum
M_ext = μ * M_{t-1} + G              # M_{t-1} 来自上一步的动量

# Step 3: 准备 Newton-Schulz 输入 (Nesterov 外推)
X = μ * M_ext + G                     # 注意: M_ext = μ*M_{t-1} + G 后再加 G
X = X / ||X||_F                       # Frobenius norm 归一化，确保 |X|_F = 1

# Step 4: Newton-Schulz 迭代 (N=5, 系数 a=3.4445, b=-4.7750, c=2.0315)
for k = 1 to 5:                       # 在 bf16 精度下执行
    X_tmp = X @ X^T                   # [A, B] × [B, A] → [A, A]
    X = a*X + b*(X_tmp @ X) + c*(X_tmp @ X_tmp @ X)
    # 等价于 f(x) = ax + bx³ + cx⁵ 作用于奇异值
    # 结果 X ≈ U V^T (SVD 中 M = U Σ V^T 的左右奇异向量乘积)

O_t = X                               # 正交化后的更新方向

# Step 5: Consistent Update RMS + Weight Decay
update = 0.2 * O_t * sqrt(max(A,B))   # 缩放因子匹配 AdamW 的 update RMS ~0.2
                                       # sqrt(max(A,B)) 抵消 Lemma 1 的 shape 效应
W_t = W_{t-1} - lr * (update + λ * W_{t-1})  # λ = 0.1

# Step 6: 保存动量用于下一步
M_t = M_ext                           # 注意：存的是不带 Nesterov 外推的动量
```

关键超参数：lr 复用 AdamW 的 optimal lr（因 update RMS 已匹配），μ = 0.95，λ = 0.1，N=5。对于非矩阵参数（RMSNorm gamma/bias、embedding table、LM head），直接使用 AdamW 更新，共享相同的 lr 和 λ。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现方式：
- 原始实现：Keller Jordan 的 GitHub 仓库 [KellerJordan/Muon](https://github.com/KellerJordan/Muon)，提供 `zeropower_via_newtonschulz5` 函数，系数 a=3.4445, b=-4.7750, c=2.0315 由手工调优得到，确保多项式在 [0.5, 1.5] 范围内有界且零点处导数最大
- Moonshot AI 扩展：分布式 Muon 实现将以 PR 形式贡献给 Megatron-LM 开源项目；预训练 checkpoint、SFT checkpoint 已发布
- HuggingFace 社区实现：`Motif-Technologies/optimizer` 仓库的 `torch-ext/optimizer/muon.py` 提供了完整可复现代码；`bird-of-paradise/muon-distributed` 提供了带注释的 CPU 友好版本
- 使用时需注意：Muon 仅用于矩阵参数（≥2D），非矩阵参数（bias、norm、embedding）必须用 AdamW；Newton-Schulz 迭代在 bf16 下计算以利用 GPU tensor core，通信开销 <1% of total training FLOPs
- Newton-Schulz 系数可通过 Chebyshev-type 多项式加速（CANS, arXiv:2506.10935），或使用 AuON (arXiv:2509.24320) 以 O(n) 替代 O(n²) 的 Newton-Schulz

涉及论文标题：
- Muon is Scalable for LLM Training
