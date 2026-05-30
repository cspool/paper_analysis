## SLA: Beyond Sparsity in Diffusion Transformers via Fine-Tunable Sparse-Linear Attention

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现是SLA的自定义GPU kernel，将稀疏注意力和线性注意力融合到单个CUDA kernel中执行前向和反向pass。核心kernel设计包括：(1) Fused Sparse+Linear Attention Forward Kernel——将critical块（FlashAttention-style O(N²)）、marginal块（O(N)线性注意力，预计算h_j/z_j后仅需矩阵加法）和negligible块（跳过）三种不同计算复杂度的操作融合在同一kernel内执行；(2) Fused Backward Kernel——同时反传稀疏注意力梯度（遵循FlashAttention backward公式）和线性注意力梯度（dH_i/dZ_i预计算后矩阵加法聚合），融合在同一kernel内避免额外launch和中间数据物化；(3) 额外效率优化——Lookup table（sparsity>90%时预处理非零mask位置减少内存流量）、Pre-aggregation（预计算全局行/列和再用减法替代90%加法）、Method of Four Russians（将marginal块分组预计算2^g子集和，用查表替代在线求和）。

  实验比较的baseline kernels：FlashAttention2（完整O(N²) attention）、VSA（89% sparsity, trainable sparse attention kernel）、VMoBa（85% sparsity, Mixture-of-block attention kernel）。评估指标：FLOPS = O(full attention)/t（kernel效率），end-to-end video generation latency（秒）。消融实验包括：Linear Only、Sparse Only、L+S（无Proj的直接相加）、不同激活函数φ（softmax/elu+1/hedgehog）、不同k_h参数（5%/10%/20%）。

- 后端平台是什么，配置是什么。
  NVIDIA RTX 5090 GPU。Attention kernel对比FlashAttention2（RTX 5090上最快可用版本）。软件：PyTorch + 自定义CUDA kernel。使用Block size b_q=b_{kv}=64。激活函数φ默认使用softmax。

- 评估性能的软件/脚本是什么。修改了什么。
  自定义CUDA kernel实现SLA的所有计算逻辑。修改：(1) 编写了SLA forward kernel——融合sparse FlashAttention critical块计算 + linear attention marginal块计算（预计算h_j = φ(K_j)^T V_j和z_j后用矩阵加法聚合）+ negligible块skip；(2) 编写了SLA backward kernel——融合sparse attention backward（dO^s → dQ/dK/dV via FlashAttention公式）+ linear attention backward（dO^l → dQ^φ/dK^φ/dV via chain rule），dH_i和dZ_i预计算后梯度聚合仅需加法；(3) 实现了额外效率优化：Lookup table存储非零mask位置、Pre-aggregation用减法替代加法、Method of Four Russians分组预计算。使用PyTorch autograd integration将自定义kernel集成到Wan2.1和LightningDiT模型中。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  开源链接：https://github.com/thu-ml/SLA（论文声明代码即将公开）。

  评估原理：
  1. Kernel级速度：测量SLA forward/backward kernel vs FlashAttention2 forward/backward的wall-clock时间，计算FLOPS = O(full attention) / t。O(full attention)是理论完整注意力FLOPs，t是实测延迟。
  2. End-to-end延迟：在Wan2.1-1.3B视频生成流程中替换所有注意力层为SLA，测量完整生成（所有denoising步骤）的wall-clock时间，包括attention和非attention（MLP/RMSNorm/Conv等）时间。

  Kernel输入到性能输出全过程（以Wan2.1-1.3B单层attention forward，N=30K tokens, d=head_dim, b_q=b_{kv}=64, k_h=5%, k_l=10%）：

  ```
  Host: 加载Wan2.1模型 + 替换注意力层为SLA attention
  Host: 对于每个denoising step t ∈ {T, T-1, ..., 1}:

  ┌─ SLA Forward Kernel (单次CUDA launch) ──────────────────────────┐
  │ Input: Q, K, V ∈ R^{N×d} (bfloat16)                              │
  │                                                                   │
  │ Step 1: Compressed mask prediction (GPU, inline):                │
  │   Q_pool = mean_pool(Q → blocks of 64)  // R^{469 × d}          │
  │   K_pool = mean_pool(K → blocks of 64)                           │
  │   P_c = Softmax(Q_pool @ K_pool^T / sqrt(d))  // R^{469 × 469} │
  │   for each row i of P_c:                                         │
  │     M_c[i, :] = classify(P_c[i,:], k_h=5%, k_l=10%)             │
  │     // 5% = 1 (critical), next 85% = 0 (marginal), 10% = -1     │
  │                                                                   │
  │ Step 2: Precompute linear attention components (line 4):         │
  │   for j in 0..T_n-1:                                            │
  │     K_phi_j = softmax(K_j)  // activation φ=softmax             │
  │     h_j = K_phi_j^T @ V_j    // R^{d × d}                      │
  │     z_j = rowsum(K_phi_j^T)   // R^{d × 1}                      │
  │                                                                   │
  │ Step 3: Main computation loop (lines 7-17):                      │
  │   for i in 0..T_m-1:  // each Q block                           │
  │     O_i_s, O_i_l: accumulators init to zero                      │
  │     for j in 0..T_n-1:  // each K,V block                       │
  │       if M_c[i,j] == 1:  // CRITICAL (~5% of blocks)            │
  │         // Full FlashAttention on this block pair               │
  │         S_ij = Q_i @ K_j^T / sqrt(d)    // b_q × b_kv GEMM     │
  │         OnlineSoftmax update:                                    │
  │           m_curr = max(m_prev, rowmax(S_ij))                    │
  │           P_ij = exp(S_ij - m_curr)                              │
  │           l_curr = exp(m_prev-m_curr) * l_prev + rowsum(P_ij)   │
  │           O_i_s = exp(m_prev-m_curr) * O_i_s + P_ij @ V_j       │
  │       elif M_c[i,j] == 0:  // MARGINAL (~85% of blocks)         │
  │         // Linear attention: single matrix addition             │
  │         H_i += h_j    // d × d addition                         │
  │         Z_i += z_j    // d × 1 addition                         │
  │       // else: NEGLIGIBLE → skip entirely                        │
  │     O_i_s = O_i_s / l_curr  // normalize sparse output          │
  │     O_i_l = (softmax(Q_i) @ H_i) / (softmax(Q_i) @ Z_i)         │
  │                                                                   │
  │ Step 4: Fusion output (Eq.6):                                    │
  │   O = O_s + Proj(O_l)  // learnable linear projection            │
  │   return O                                                        │
  └───────────────────────────────────────────────────────────────────┘

  ┌─ SLA Backward Kernel (单次CUDA launch) ──────────────────────────┐
  │ Input: dO (gradient of loss w.r.t. output), Q,K,V, M_c, L_i,    │
  │        H_i, Z_i, O_s, O_l from forward pass                      │
  │                                                                   │
  │ Step 1: Precompute linear attention gradients per Q block:       │
  │   for i in 0..T_m-1:                                            │
  │     D_i_s = rowsum(dO_i_s ⊙ O_i_s)  // for softmax backward     │
  │     D_i_l = rowsum(dO_i_l ⊙ O_i_l)                              │
  │     dH_i = (Q_phi_i / (Q_phi_i @ Z_i))^T @ dO_i_l               │
  │     dZ_i = -(Q_phi_i / (Q_phi_i @ Z_i))^T @ D_i_l               │
  │     dQ_phi_i = (dO_i_l @ H_i^T - D_i_l @ Z_i^T) / (Q_phi_i @ Z_i)│
  │                                                                   │
  │ Step 2: Aggregate gradients per K,V block:                       │
  │   for j in 0..T_n-1:                                            │
  │     dH_agg = 0; dZ_agg = 0                                       │
  │     for i in 0..T_m-1:                                          │
  │       if M_c[i,j] == 1:  // CRITICAL: FlashAttention backward   │
  │         S_ij = Q_i @ K_j^T / sqrt(d)                            │
  │         P_ij = exp(S_ij - L_i)                                   │
  │         dV_j += P_ij^T @ dO_i_s                                 │
  │         dP_ij = dO_i_j_s @ V_j^T                                │
  │         dS_ij = P_ij ⊙ (dP_ij - D_i_s)                          │
  │         dQ_i += dS_ij @ K_j                                     │
  │         dK_j += dS_ij^T @ Q_i                                   │
  │       elif M_c[i,j] == 0:  // MARGINAL: aggregate precomputed   │
  │         dH_agg += dH_i    // matrix addition                    │
  │         dZ_agg += dZ_i    // vector addition                    │
  │     dK_phi_j = V_j @ dH_agg^T + dZ_agg^T                        │
  │     dV_j += K_phi_j @ dH_agg                                     │
  │                                                                   │
  │ return dQ, dK, dV, dQ_phi, dK_phi                                │
  └───────────────────────────────────────────────────────────────────┘

  性能输出（RTX 5090, Wan2.1-1.3B, 30K tokens）：
    - SLA Forward Kernel: 13.7× speedup vs FlashAttention2 forward
    - SLA Forward Kernel: 1.93× faster than VSA @ 95% sparsity
    - SLA Forward Kernel: 3.36× faster than VMoBa @ 95% sparsity
    - SLA Backward Kernel: 6.8× speedup vs FlashAttention2 backward
    - Attention latency reduction: 97s → 11s (8.8×)
    - End-to-end video generation: 2.2× speedup
    - SLA @ 95% sparsity (1-5%): ~3× more efficient than Sparse Only @ 85% sparsity
    - Fine-tuning overhead: 2000 steps × batch 64 << 0.1% pretraining cost
  ```

  额外效率优化（Appendix A.3）：
  - Lookup table: sparsity>90%时，扫描整行/列读取mask值产生显著内存开销 → 预处理每行/列的非零位置存为lookup table → 计算时直接查表，减少内存流量
  - Pre-aggregation for linear attention: 虽然每次矩阵加法开销极小（Line 13），但M_c中>90%为0时重复加法累积 → 预计算全局∑h_j和∑z_j → 然后减去M_c[i,j]≠0的贡献 → 90%加法被10%减法替代
  - Method of Four Russians: 边际块数量既不很小也不很大（~50%）时 → 将h_j和z_j分组为g个连续块 → 预计算每组2^g个子集和 → 前向/反向时单次查表替代on-the-fly求和 → 理论计算量减少1/g
  ```
