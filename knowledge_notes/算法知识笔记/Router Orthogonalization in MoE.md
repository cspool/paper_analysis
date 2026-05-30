## Router Orthogonalization in MoE

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Router 正交化是指对 MoE 模型的 Router 权重矩阵施加正交约束（R^T R ≈ I），使 Router 的前向映射 x → xR 成为近似等距变换（保留 token 间的成对角度/点积）。在 MoE 中，这意味着相似 token 被 Router 映射后仍保持相似的 routing scores，从而获得相似的 expert 分配。正交化可以通过两种方式实现：

1. **显式参数化（Parametric）**：使用 QR 分解或 Cayley 变换将 Router 权重约束在 Stiefel 流形上。PyTorch 提供 `torch.nn.utils.parametrizations.orthogonal`。缺陷：需要 float32 计算，bfloat16 下数值不稳定；频繁重正交化开销大；不适用于大规模训练。

2. **损失-based 软约束（Loss-based）**：SIMBAL 采用的方法——将 L_orth = ||R^T R - I||_1 作为辅助损失项，通过梯度下降软性地将 R 推向正交。可在 bfloat16 中直接训练。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

正交 Router 如何保持 token 间关系：

给定两个 token: x1, x2 ∈ R^{D_M}
Router 输出: s1 = x1·R, s2 = x2·R ∈ R^E

若 R 是正交矩阵 (R^T R = I):
  s1·s2 = (x1·R)·(x2·R) = x1·(R·R^T)·x2 ≈ x1·x2
  
因此 cos(x1, x2) ≈ cos(s1, s2) → 相似 token 获得相似 routing scores

SIMBAL 论文 Table 2 验证：在 1536×32 的 Router 上 100 步优化后，Gram matrix 与 I 的 L1 distance 为 ~1×10^-5（vs QR 参数化的 ~2×10^-4），即 loss-based 方法比显式参数化更接近正交。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- **初始化**：使用 Saxe et al. 2014 正交初始化（对权重矩阵做 SVD 后替换奇异值为 1），或仅训练 Router 数步即可达到接近正交
- **loss coefficient**：SIMBAL 论文 lambda=0.1，但 0.01-1.0 均有效
- **训练稳定性**：正交 Router 对输入扰动更鲁棒（角度保持），训练早期不会出现频繁的 routing shift
- **与前人工作的区别**：OMoE (Liu et al. 2024) 正交化 Expert 权重（在 optimizer 中更新方向正交），MOORE (Hendawy et al. 2024) 正交化 Expert 表示（Gram-Schmidt）。SIMBAL 是首个对 Router 做正交化的方法——Router 参数极少（<0.02% total params）但编排 billions of parameters

涉及论文标题：
- Load Balancing Mixture of Experts with Similarity Preserving Routers
