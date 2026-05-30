## Fused Sparse-Linear Attention GPU Kernel

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Fused Sparse-Linear Attention Kernel是SLA的核心实现——将稀疏FlashAttention（O(N²) per critical block）、线性注意力（O(1) per marginal block via precomputation）和negligible block skipping三种不同计算模式融合到单个CUDA kernel中，支持完整的前向和反向pass。融合的关键优势：(1) 单次kernel launch消除多次launch overhead；(2) 所有block计算共享同一Q/K/V数据加载，避免重复HBM访问；(3) 前向和反向对称设计，反向也融合sparse gradients和linear gradients。

从kernel调度角度拆解：
```
Fused SLA Forward Kernel (Algorithm 1, single CUDA launch):

// Phase 1: Precompute for linear attention (GPU parallelized over j)
parfor j in 0..T_n-1:
    K_phi_j = softmax(K_j)           // per-block activation
    h_j = matmul(K_phi_j^T, V_j)     // [d,d] — stored in HBM for all Q blocks
    z_j = rowsum(K_phi_j^T)          // [d,1]

// Phase 2: Main fused loop (GPU parallelized over i)
parfor i in 0..T_m-1:
    O_i_s, H_i, Z_i = 0, 0, 0
    m_prev, l_prev = -inf, 0
    
    for j in 0..T_n-1:
        if M_c[i,j] == 1:   // CRITICAL → Tensor Core GEMM pipeline
            S_ij = wgmma(Q_i, K_j^T) / sqrt(d)     // Tensor Cores
            m_curr, l_curr = online_softmax_update(S_ij, m_prev, l_prev)
            O_i_s = rescale_and_accumulate(O_i_s, S_ij, V_j, m_prev, m_curr)
            m_prev, l_prev = m_curr, l_curr
        elif M_c[i,j] == 0: // MARGINAL → CUDA Core addition
            H_i += h_j      // d×d matrix addition
            Z_i += z_j      // d×1 vector addition
        // else NEGLIGIBLE → no operation

    O_i_s = diag(1/l_prev) @ O_i_s       // normalize sparse output
    Q_phi_i = softmax(Q_i)
    O_i_l = (Q_phi_i @ H_i) / (Q_phi_i @ Z_i)  // linear output

// Phase 3: Fusion output
O = O_s + Proj(O_l)  // learnable projection
```

反向kernel（Algorithm 2）对称融合：sparse gradients遵循FlashAttention backward公式（dO^s → dS_{ij} → dQ_i, dK_j, dV_j），linear gradients通过chain rule（dO^l → dH_i/dZ_i → dQ_i^φ, dK_j^φ, dV_j），marginal块的梯度聚合也为矩阵加法（dH_agg += dH_i, dZ_agg += dZ_i）。

性能特征：Forward 13.7× vs FlashAttention2 (RTX 5090, Wan2.1-1.3B)；Backward 6.8× vs FlashAttention2。marginal块（~85% of blocks）仅占<0.5% full attention cost，使得critical块的Tensor Core GEMM主导执行时间，GPU利用率接近dense attention但计算量仅5%。

术语一般如何实现？如何使用？
实现为CUDA kernel（论文未开源实际kernel代码，仅开源高层接口 https://github.com/thu-ml/SLA）。使用WGMMA (warp group matrix multiply-accumulate) for Tensor Core GEMM on critical blocks, TMA for async data loading, CUDA cores for linear attention additions。额外优化（Appendix A.3）：Lookup table存储非零mask位置（sparsity>90%）、Pre-aggregation（用∑h_j - ∑_{M_c≠0}h_j替代逐个加法）、Method of Four Russians（分组预计算子集和）。块大小b_q=b_{kv}=64（已在线性注意力预计算开销和分类粒度间平衡优化）。

涉及论文标题：
- SLA: Beyond Sparsity in Diffusion Transformers via Fine-Tunable Sparse-Linear Attention
