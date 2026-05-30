## Block Sparse Attention (GPU Block-Level)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Block Sparse Attention是将注意力稀疏性从element-wise提升到block level的实现策略，使稀疏注意力在现代GPU上高效执行。动机：element-wise sparse attention（在单个Q_i·K_j dot-product级别做mask）在GPU上效率低——不规则的稀疏pattern导致warp divergence、uncoalesced memory access和低tensor core利用率。Block-level sparse attention将Q、K、V、S、P、M都划分成blocks {Q_i}、{K_j}、{V_j}，每个block mask M_{ij} ∈ {0,1}^{b_q×b_{kv}}完全填0或1，跳过M_{ij}=0的整块Q_i K_j^T和P_{ij} V_j计算。

在SLA中，block sparse attention用于critical块（M_c[i,j]=1, ~5% blocks）。每对critical (Q_i, K_j, V_j)执行完整FlashAttention：S_{ij} = Q_i K_j^T/√d, OnlineSoftmax normalization, O_i^s += P_{ij} V_j。block size b_q=b_{kv}=64是在GPU效率和分类粒度间的平衡——太小的block导致mask预测开销增大，太大的block使分类粗糙。

从kernel调度角度拆解：
```
Block Sparse Attention Forward (per Q block Q_i):
  for j in 0..T_n-1:
      if M_c[i,j] == 1:   // critical block pair
          // Full FlashAttention on this block:
          S_ij = Q_i @ K_j^T / sqrt(d)    // [b_q, b_{kv}] GEMM, Tensor Cores
          // OnlineSoftmax rescaling:
          m_new = max(m_prev, rowmax(S_ij))
          P_ij = exp(S_ij - m_new)        // [b_q, b_{kv}]
          l_new = exp(m_prev - m_new)*l_prev + rowsum(P_ij)
          O_i_s = exp(m_prev - m_new)*O_i_s + P_ij @ V_j  // [b_q, d] GEMM
          m_prev = m_new; l_prev = l_new
      // else: skip entire block computation
  O_i_s = diag(1/l_prev) @ O_i_s  // final normalization
```

关键效率考量：block-level sparsity使tensor core GEMM操作均在规则的[b_q, b_{kv}]或[b_q, d] tile上执行，无warp divergence。SLA的M_c在block级别分类（分辨率为N/b_q × N/b_{kv} = 469×469 for N=30K），而非元素级别（N×N = 30K×30K），使mask预测和存储开销可忽略。

术语一般如何实现？如何使用？
Block sparse attention的GPU实现通常基于FlashAttention框架（https://github.com/Dao-AILab/flash-attention），在tiling outer loop中插入block mask检查。SLA将block sparse attention与linear attention融合在单个kernel中。Block size选择：b_q=b_{kv}=64是FlashAttention的典型block大小，平衡SRAM使用和并行度。VSA、VMoBa等方法也使用类似的block-level稀疏策略，但分类粒度（block vs element）和mask预测方法（训练式 vs 训练无关）不同。

涉及论文标题：
- SLA: Beyond Sparsity in Diffusion Transformers via Fine-Tunable Sparse-Linear Attention
