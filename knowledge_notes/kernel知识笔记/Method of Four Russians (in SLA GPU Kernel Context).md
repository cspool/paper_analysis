## Method of Four Russians (in SLA GPU Kernel Context)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Method of Four Russians (Arlazarov et al., 1970) 是布尔矩阵乘法和传递闭包计算的经典算法，核心理念是通过预计算和查表来减少在线计算量。SLA在Appendix A.3中将其适配到marginal块的线性注意力聚合场景：当M_c中标记为0（marginal）的块数量既不太小也不太大（~50%）时，将连续的g个h_j和z_j分组，预计算每组内所有2^g种可能的子集和存入查找表，前向/反向时任何子集的聚合结果通过单次查表获取，而非on-the-fly逐个求和。

从kernel调度角度拆解：
```
Standard marginal block aggregation (line 13 in Algorithm 1):
  for each Q block i:
      H_i = 0
      for j where M_c[i,j]==0:
          H_i += h_j    // one d×d addition per marginal block
  // Cost: (#marginal_blocks) × (d×d addition)

Method of Four Russians optimization:
  // Offline/precompute: group h_j into segments of g consecutive blocks
  for each segment s of g blocks:
      precompute all 2^g subset-sums of h_j in segment s
      store in lookup table LUT_s[bitmask]  // 2^g entries, each d×d

  // Online: use lookup table
  for each Q block i:
      H_i = 0
      for each segment s:
          bitmask = extract_g_bits(M_c[i, segment_range])
          if bitmask != 0:  // not all negligible/critical
              H_i += LUT_s[bitmask]    // single lookup + addition
  // Cost: (#segments) × (one lookup + one addition)
  // Theoretical reduction: 1/g
```

适用条件：marginal块比例~50%时最优。sparsity极高（>90%）时用Pre-aggregation更优（∑_all - ∑_non_marginal），sparsity极低时直接加法即可。SLA中默认使用直接加法（85% marginal块），Method of Four Russians作为备选优化。

术语一般如何实现？如何使用？
经典实现用于布尔矩阵乘法和transitive closure。在SLA的GPU kernel中：预计算的lookup table存储在GPU global memory或shared memory中（取决于g大小），g的选择平衡查找表大小（2^g × d×d，exponential in g）和计算节省（1/g reduction）。SLA论文未报告此优化的独立ablation结果，将其列在Appendix A.3作为supplementary efficiency optimization。

涉及论文标题：
- SLA: Beyond Sparsity in Diffusion Transformers via Fine-Tunable Sparse-Linear Attention
