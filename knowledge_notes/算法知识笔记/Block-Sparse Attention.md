## Block-Sparse Attention

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Block-Sparse Attention是FlashAttention的扩展变体，通过在预定义的block级稀疏mask约束下跳过零值block的attention计算来加速推理。给定block sparsity mask $\mathbf{M} \in \{0,1\}^{N/B_r \times N/B_c}$（其中$B_r, B_c$为block sizes），block-sparse attention仅计算$M_{ij}=1$的(i,j) block对：$\mathbf{S}_{ij} = \mathbf{Q}_i \mathbf{K}_j^T$仅在$M_{ij}=1$时计算，softmax和$\tilde{\mathbf{P}}_{ij}\mathbf{V}_j$同理。其IO复杂度为$\Theta(Nd + N^2d^2M^{-1}s)$（s为non-zero block比例），比dense FlashAttention减少sparsity倍。与一般稀疏attention不同，block-sparse要求稀疏模式在block边界对齐——这一约束恰好与FlashAttention的tiling自然吻合，因为tiling本身就在block粒度上操作。论文使用固定butterfly sparsity pattern（Dao et al., 2022），这种模式被证明可以逼近任意稀疏矩阵。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Block-sparse FlashAttention (Algorithm 5)的核心计算流程：
```
# 与FlashAttention Algorithm 1的区别仅在于内循环条件：
for j = 1 to T_c:
    load K_j, V_j from HBM to SRAM
    for i = 1 to T_r:
        if M[i][j] == 0:           # ← 唯一区别：跳过零值block
            continue                  # 省softmax + PV计算 + V加载
        # 其余完全同FlashAttention:
        load Q_i from HBM to SRAM
        S_ij = Q_i @ K_j.T         # BMM1（compute-bound，可能仍需计算...实际上FlashAttention的block-sparse同样跳过S_ij计算）
        # online softmax...
        # accumulate O_i...
```
注意：在FlashAttention的block-sparse实现中，即使$M_{ij}=1$的block在BMM1（Q_i @ K_j.T）步骤也不计算——算法直接跳过整个内循环迭代，从而实现与sparsity s成比例的runtime减少。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Block-sparse attention在FlashAttention代码库（https://github.com/HazyResearch/flash-attention）中以`BlockSparseAttention`接口提供。使用：指定block sparsity mask作为`(N/B_r, N/B_c)`的二进制矩阵。butterfly pattern是常用选择：对序列中相距较远的token pair赋予1（长程依赖），对相距近的token pair也赋予1。在LRA benchmark上，block-sparse FlashAttention达到2.8× speedup（vs dense，seq length 1K-4K），同时accuracy与dense attention持平（LRA平均59.6 vs 59.8）。在Path-256（seq length 64K）上，block-sparse使Transformer首次达到63.1%准确率（dense FlashAttention因memory限制无法扩展到64K）。block-sparse的sparsity pattern选择对accuracy影响较大——论文使用预定义的butterfly pattern而非learned sparsity。

涉及论文标题：
- FlashAttention Fast and Memory-Efficient Exact Attention with IO-Awareness
- Flex Attention: A Programming Model for Generating Optimized Attention Kernels
