## VENOM V:N:M Sparse Format

术语是什么？
VENOM（Vectorized N:M format）是 Castro et al.（SC '23）提出的一种灵活结构化稀疏数据格式，旨在利用 NVIDIA SpTC 硬件加速稀疏-密集矩阵乘法，同时突破 cuSPARSELt 固定 50%（2:4）稀疏比的限制。VENOM 的 V:N:M 格式中：M 是 block 大小（列维度），N 是在 block 中保留的 vector 数量，V 是每个 vector 的 element 数量。通过调整 N 和 V，VENOM 支持从 50% 到 90%+ 的灵活稀疏比，而仍然利用 SpTC 的 2:4 硬件加速。其核心技巧是将不同 N:M 比例的向量组合映射到 2:4 pattern，在 metadata 层面"欺骗" SpTC 选择器以正确处理非标准稀疏模式。

从 kernel 调度角度拆解术语：
VENOM 的 V:N:M 编码和 SpTC 映射机制：

```
// VENOM 编码示例: V=2, N=1, M=4 → 75% sparsity
// 原始矩阵 (M=4 columns per block, V=2 elements per vector):
// Column:   0  1  2  3
// Row 0:    a  .  .  .   (非零值 a 在 col 0)
// Row 1:    b  .  .  .   (非零值 b 在 col 0)
// Row 2:    .  .  c  .   (非零值 c 在 col 2)
// ...
// 每 2 行 × 4 列 block 中仅保留 1 个 vector（N=1），每个 vector 含 2 个元素（V=2）
// 有效稀疏比 = 1 - (1×2)/(2×4) = 1 - 2/8 = 75%

// VENOM 编码为 2:4 compatible 格式：
// 将多个 V:N:M block 的 non-zero vectors 拼接成符合 2:4 的 dense rows
// metadata 记录 "原始 column → packed column" 的映射
// SpTC 执行时通过 metadata selection 从 dense B 中选择正确的列参与计算

// VENOM sparse-dense matmul 伪代码：
for each thread block (tile of C[m_b][n_b]):
    for k_tile = 0 to K step K_b:
        load VENOM_A[m_b][K_b] from GMEM → SMEM → register
        load dense_B[K_b][n_b] from GMEM → SMEM → register
        
        for each V:N:M block in A:
            sel_cols = metadata.indices  // 哪些列是活跃的
            
        // Execute via mma.sp (SpTC 利用 metadata 选择 B 列)
        mma.sp(C_tile, A_packed, B_dense[sel_cols], metadata)
```

**VENOM 的关键局限（Samoyeds 论文 Figure 6 揭示）**：
当输入矩阵 B 也是稀疏的（如 MoE token routing），VENOM 的 sparse-dense 设计暴露出三类问题：
1. **I/O amplification（格式②③）**：跳过稀疏 weight column 时，若该 column 对应的 input row 也是稀疏列，则可能加载了不需要的 input 数据或跳过了需要的 weight 行的 input 数据。
2. **Uncoalesced memory access（格式④）**：稀疏 column 导致数据在内存中不连续，GPU 无法 coalesce memory transaction，带宽利用率下降。
3. **Small tile fragmentation**：稀疏 pattern 将数据打散为小 tile，降低 warp 利用率。

这正是 Samoyeds 提出 dual-side sparse format 的动机——解决 VENOM 在"权重稀疏 + 输入稀疏"场景下的退化问题。

术语一般如何实现？如何使用？
- VENOM 代码开源（SC '23 artifact），与 cuSPARSELt 对比，支持灵活 sparse ratio（50%~90%+），在 SC '23 基准上取得 1.38× 加速 over cuSPARSELt。
- VENOM 的适用场景：单端权重稀疏的推理（输入 dense），如传统 LLM 的 FP16 推理。但在 MoE（输入天然稀疏）下性能退化。
- Samoyeds 的改进：(1) 双端稀疏格式——输入端也采用 vector-wise 稀疏 by SEL array；(2) 专门的 sparse-sparse kernel 而非 sparse-dense；(3) customized packing 和 data stationary 避免 VENOM 的 I/O amplification 问题。Samoyeds 在 kernel 级 up to 1.99× vs VENOM，模型级 up to 1.58× vs vLLM。

涉及论文标题：
- Samoyeds: Accelerating MoE Models with Structured Sparsity Leveraging Sparse Tensor Cores
