## SLA: Beyond Sparsity in Diffusion Transformers via Fine-Tunable Sparse-Linear Attention

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是SLA（Sparse-Linear Attention），一个可训练的混合稀疏+线性注意力方法，用于加速Diffusion Transformer（DiT）模型，特别是视频生成场景。核心算法包括三个关键设计：(1) 三级注意力权重分类——通过压缩注意力权重矩阵P_c = Softmax(pool(Q)pool(K)^T/√d)将注意力权重块动态分为critical（top k_h%）、marginal（中间k_h%~k_l%）和negligible（bottom k_l%）三类，对应应用O(N²)稀疏FlashAttention、O(N)线性注意力和跳过三种策略；(2) 统一融合GPU kernel——将稀疏注意力和线性注意力的前向和反向计算融合到单个GPU kernel中，预计算h_j = φ(K_j)^T V_j和z_j = rowsum(φ(K_j)^T)使marginal块仅需单次矩阵加法；(3) 可学习投影层Proj(O^l)——对线性注意力输出O^l应用可学习线性变换R^d→R^d，减少softmax注意力和线性注意力之间的分布不匹配。仅需少量fine-tuning步骤（2000步，<0.1% pretraining cost），SLA即可将注意力计算减少95%而不损失生成质量。

  实验比较的baseline包括：(1) VSA——训练式稀疏注意力（89% sparsity）；(2) VMoBa——训练式MoE block注意力（85% sparsity）；(3) SpargeAttn-F——训练无关稀疏注意力（85% sparsity）；(4) SpargeAttn-T——训练式稀疏注意力（84% sparsity）；(5) Linear Only——仅线性注意力；(6) Sparse Only——仅SLA的稀疏组件；(7) L+S——稀疏和线性注意力的简单输出相加（无Proj层）。视频质量用VBench的IQ/OC/AQ/SC + Vision Reward + Aesthetic/Technical Video Quality评估，效率用FLOPs和FLOPS评估。

- 硬件平台是什么，配置是什么。
  NVIDIA RTX 5090 GPU用于kernel速度和端到端延迟评估。FlashAttention2作为参考attention实现。训练使用batch size=64，fine-tune 2000 steps。软件：PyTorch，自定义CUDA kernel实现SLA前向和反向pass。

- 模型是什么。数据集和bench分别是什么。
  模型：Wan2.1-1.3B（视频生成，30K sequence length用于视频生成）为主要实验模型；LightningDiT-1p0B/1（1.03B参数，图像生成）用于补充实验。数据集：私有数据集（来自Pexels和Common Crawl，20,000个5秒480p视频）用于视频fine-tuning；ImageNet 512×512用于图像实验。Benchmarks：VBench（IQ/OC/AQ/SC四个维度）、Vision Reward（人类偏好）、Aesthetic Video Quality (VA)、Technical Video Quality (VT)；图像用FID。FLOPs和FLOPS用于效率评估。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源链接：https://github.com/thu-ml/SLA（论文声明代码即将公开）。

  SLA算法pipeline（以Wan2.1-1.3B注意力层，block size b_q=b_{kv}=64, k_h=5%, k_l=10%, φ=softmax为例）：

  **压缩mask预测（离线/在线都执行）：**
  ```
  # Step 1: 预测压缩注意力权重矩阵 (Eq.2)
  Q_pool = mean_pool(Q, block_size=b_q)      # R^{N/b_q × d}
  K_pool = mean_pool(K, block_size=b_{kv})    # R^{N/b_{kv} × d}
  P_c = Softmax(Q_pool @ K_pool^T / sqrt(d))  # R^{N/b_q × N/b_{kv}}

  # Step 2: 三级分类生成压缩mask M_c (Eq.3)
  for each row i in P_c:
      M_c[i, :] = -1  # 初始化为negligible
      top_kh_indices = TopK(P_c[i,:], k_h=5%)   # critical: top 5%
      M_c[i, top_kh_indices] = 1
      bottom_kl_indices = BottomK(P_c[i,:], k_l=10%)  # negligible: bottom 10%
      # 剩余15%保持marginal (M_c=0)
  ```

  **SLA前向pass（Algorithm 1，单kernel执行）：**
  ```
  # Precompute for linear attention (line 4)
  for j in range(T_n):
      K_phi_j = softmax(K_j)  # activation function φ
      h_j = K_phi_j^T @ V_j           # R^{d × d}
      z_j = rowsum(K_phi_j^T)          # R^{d × 1}

  # Main loop over query blocks (line 7-17)
  for i in range(T_m):  # T_m = N/b_q
      O_i_s = 0  # sparse output accumulator
      H_i = 0; Z_i = 0  # linear attention accumulators
      m_prev = -inf; l_prev = 0  # OnlineSoftmax state

      for j in range(T_n):  # T_n = N/b_{kv}
          if M_c[i,j] == 1:  # CRITICAL → O(N²) sparse FlashAttention
              S_ij = Q_i @ K_j^T / sqrt(d)
              m_curr = max(m_prev, rowmax(S_ij))
              P_ij = exp(S_ij - m_curr)
              l_curr = exp(m_prev - m_curr) * l_prev + rowsum(P_ij)
              O_i_s = diag(exp(m_prev - m_curr)) @ O_i_s + P_ij @ V_j
              m_prev = m_curr; l_prev = l_curr

          elif M_c[i,j] == 0:  # MARGINAL → O(N) linear attention
              H_i += h_j    # 仅矩阵加法（已预计算）
              Z_i += z_j    # 仅向量加法（已预计算）

          # else M_c[i,j] == -1: NEGLIGIBLE → skip

      # Finalize outputs (line 16)
      O_i_s = diag(l_prev)^{-1} @ O_i_s        # sparse output normalization
      Q_phi_i = softmax(Q_i)
      O_i_l = (Q_phi_i @ H_i) / (Q_phi_i @ Z_i)  # linear attention output
      L_i = m_prev + log(l_prev)                # log-sum-exp for backward

  # Final output (Eq.6)
  O = O_s + Proj(O_l)  # Proj: learnable ℝ^d → ℝ^d linear
  ```

  **反向pass关键设计（Algorithm 2）：**
  - 稀疏attention梯度：复用FlashAttention的backward公式，dO^s → dS_ij → dQ_i/dK_j/dV_j，使用D_i^s = rowsum(dO_i^s ⊙ O_i^s)进行softmax梯度计算
  - 线性attention梯度：dO^l → dH_i/dZ_i → dQ_i^φ/dK_j^φ/dV_j，dH_i和dZ_i预计算后对每个marginal块仅需矩阵加法
  - 稀疏和线性组件的梯度融合在同一kernel内执行

  关键设计要点：
  - 压缩mask P_c的分辨率为N/b_q × N/b_{kv}（而非N×N），预测开销可忽略
  - 线性注意力在Wan2.1中仅占full attention的<0.5%，因此marginal块的线性注意力替代是"几乎免费"的
  - 三级分类中仅critical块（5%）执行完整O(N²)计算，marginal块（~85%）用O(N)线性注意力，negligible块（10%）跳过
  - Proj层解决softmax和线性注意力的分布不匹配，使线性注意力作为"learnable compensation"而非直接近似
  - 额外效率优化（Appendix A.3）：Lookup table（sparsity>90%时预处理非零位置）、Pre-aggregation（用减法替代90%加法）、Method of Four Russians（group预计算2^g子集和）
