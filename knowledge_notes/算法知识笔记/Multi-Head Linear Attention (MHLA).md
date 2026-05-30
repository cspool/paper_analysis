## Multi-Head Linear Attention (MHLA)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Multi-Head Linear Attention (MHLA) 是一种新型线性注意力机制，由 Zhang et al. (ICLR 2026) 提出。核心创新在于：沿 token 维度（而非 channel/head 维度）将序列划分为 M 个 non-overlapping blocks（"token-level heads"），为每个 block 独立计算局部 KV summary，再通过可学习的系数矩阵 Mc 使每个 query block 生成专属的混合 summary，从而恢复 standard linear attention 中丧失的 query-conditioned 选择性和 token 级别多样性。

MHLA 的关键操作流程：
1. 输入 X ∈ R^(N×d)，投影得到 Q, K, V
2. 应用 kernelized feature map φ(·)：Q̃ = φ(Q), K̃ = φ(K)
3. 将序列沿 spatial（2D）或 spatiotemporal（3D）维度分为 M 个 blocks，每 block 含 N_b 个 token
4. 每 block b 计算局部 KV summary：S_b = Σ_{j∈b} K̃_j^T V_j ∈ R^(d×d)，z_b = Σ_{j∈b} K̃_j ∈ R^d
5. 通过可学习系数矩阵 Mc ∈ R^(M×M)，query block i 的混合 summary：S̃_i = Σ_{b=1}^M m_{i,b} S_b，z̃_i = Σ_{b=1}^M m_{i,b} z_b
6. 输出：o = (q̃^T S̃_i) / (q̃^T z̃_i) = Σ_{b} m_{i,b} (q̃^T S_b) / Σ_{b} m_{i,b} (q̃^T z_b)

复杂度 O(Nd² + M²d²)。当 M² ≤ N 时，主导项 O(Nd²) 与 standard linear attention 相同。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**MHLA forward pass 伪代码**：

```
def mhla_forward(X, W_Q, W_K, W_V, M_c, block_ids):
    # X: [N, d], M_c: [M, M] learnable coefficient matrix
    Q, K, V = X @ W_Q, X @ W_K, X @ W_V                # [N, d]
    Q_tilde = phi(Q)                                     # kernelized, e.g., ReLU
    K_tilde = phi(K)
    
    # Step 1: Compute local KV summaries per block
    S = []  # list of [d, d] summaries
    z = []  # list of [d] normalizers
    for b in range(M):
        mask_b = (block_ids == b)                        # [N_b]
        S_b = K_tilde[mask_b].T @ V[mask_b]              # [d, d]
        z_b = K_tilde[mask_b].sum(dim=0)                 # [d]
        S.append(S_b)
        z.append(z_b)
    
    # Step 2: Multi-Head Mixing — query-conditioned per block
    O = zeros(N, d)
    for i in range(M):
        mask_i = (block_ids == i)
        q_i = Q_tilde[mask_i]                            # [N_i, d]
        S_mixed_i = sum(M_c[i, b] * S[b] for b in range(M))  # [d, d]
        z_mixed_i = sum(M_c[i, b] * z[b] for b in range(M))  # [d]
        O[mask_i] = (q_i @ S_mixed_i) / (q_i @ z_mixed_i)    # [N_i, d]
    return O
```

**实际实现优化**：所有 block summaries 堆叠为 [M, d, d] tensor，通过 batched GEMM 一次性计算所有混合 summaries：S_all = einsum('ij,jkl->ikl', M_c, S_stacked)。

术语一般如何实现？如何使用？

MHLA 通过标准 PyTorch 和 GEMM 操作实现，无自定义 CUDA kernel，可直接替换任何 Transformer 架构中的 attention 模块。初始化策略：Mc 按 locality-biased 初始化 m_{i,j}^(0) ∝ 1 - dist(i,j)/max_k(dist(i,k))，训练过程中 clip 到 (0,1) 保持非负。开源实现：https://github.com/DAGroup-PKU/MHLA（MIT license），含五个子项目覆盖图像分类（DeiT/VLT）、图像生成（DiT/DiG）、T2I（SANA）、视频生成（Wan2.1）、NLP。

涉及论文标题：
- MHLA: Restoring Expressivity of Linear Attention via Token-Level Multi-Head

---
