## OnlineSoftmax (Block-Wise Incremental Softmax)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
OnlineSoftmax (Milakov & Gimelshein, 2018) 是一种在矩阵分块场景下增量计算softmax的算法，是FlashAttention实现的核心技术之一。标准softmax需要三次pass over the entire row：一次求max（numerical stability）、一次求exp sum、一次求normalized values。OnlineSoftmax将各行分块，逐block增量更新running max和running sum，每块仅需一次前向pass，最后做一次归一化。核心更新公式：

$$m_{new} = \max(m_{old}, \operatorname{rowmax}(S_{block})), \quad l_{new} = e^{m_{old}-m_{new}} \cdot l_{old} + \operatorname{rowsum}(e^{S_{block}-m_{new}})$$

$$O_{new} = \operatorname{diag}(e^{m_{old}-m_{new}}) \cdot O_{old} + e^{S_{block}-m_{new}} \cdot V_{block}$$

在SLA中，OnlineSoftmax用于critical块（M_c[i,j]=1）的稀疏FlashAttention计算。由于critical块在K维度是非连续的（被marginal和negligible块间隔），OnlineSoftmax的增量更新特性允许在遍历K块时自然地跳过非critical块（无需重新归一化已完成的部分）。

从kernel调度角度拆解：
```
OnlineSoftmax in SLA critical block processing:
  m_prev = [-inf, -inf, ..., -inf]  // per-row running max, [b_q]
  l_prev = [0, 0, ..., 0]          // per-row running sum, [b_q]
  O_i_s = zeros(b_q, d)            // running weighted output

  for j where M_c[i,j] == 1:       // only critical K,V blocks
      S_ij = Q_i @ K_j^T / sqrt(d)  // [b_q, b_{kv}]
      m_curr = elementwise_max(m_prev, rowmax(S_ij))
      
      // Rescale old accumulators to new max:
      scale = exp(m_prev - m_curr)  // [b_q]
      l_curr = scale * l_prev + rowsum(exp(S_ij - m_curr))
      O_i_s = diag(scale) @ O_i_s + exp(S_ij - m_curr) @ V_j
      
      m_prev = m_curr
      l_prev = l_curr

  // After ALL critical blocks processed:
  O_i_s = diag(1/l_prev) @ O_i_s   // final normalization
```

关键特性：marginal块的线性注意力（H_i += h_j）不参与softmax归一化——marginal块用独立的线性注意力路径，critical块用OnlineSoftmax归一化，两者在最终输出时通过Proj融合。这使得critical块间的非连续遍历不影响OnlineSoftmax的正确性。

术语一般如何实现？如何使用？
OnlineSoftmax是FlashAttention (Dao et al., 2022; Dao, 2023)的标准实现组件。在CUDA kernel中以register-resident的m和l向量实现（每行一个scalar，存储在warp-level registers）。SLA复用FlashAttention的OnlineSoftmax实现用于critical块，并在同一kernel中添加线性注意力路径。FlashAttention 2 (Dao, 2023) 进一步优化了OnlineSoftmax的rescaling开销（减少非matmul FLOPs）。

涉及论文标题：
- SLA: Beyond Sparsity in Diffusion Transformers via Fine-Tunable Sparse-Linear Attention
