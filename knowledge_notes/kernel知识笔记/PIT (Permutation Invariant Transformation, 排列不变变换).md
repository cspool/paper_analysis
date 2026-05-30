## PIT (Permutation Invariant Transformation, 排列不变变换)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

PIT（Permutation Invariant Transformation，排列不变变换）是一种动态稀疏编译器技术，由 Zheng et al. (SOSP 2023) 提出，用于将稀疏数据高效加载到 GPU 的 dense compute blocks 中。核心思想是：在不改变计算结果的前提下，通过数学上可证明的排列不变变换，将多个空间上非连续的稀疏微 tile 重组为连续 dense tile，从而利用高效的 dense GEMM（Tensor Core）进行计算，避免稀疏格式的低效 irregular memory access。

"Permutation Invariant" 的含义：变换（对输入数据的行列重排）不改变最终计算结果，因为：(1) 加法的交换律保证重排不改变累加结果；(2) softmax 的归一化在重排后依然正确。

在 MInference 中，PIT 被用于 Vertical-Slash FlashAttention kernel 的 column part——当垂直线是非连续的 column indices 时（如 column [100, 5230, 8100, ...]），PIT 将这些非连续的 column data 加载到同一个 dense compute block 中，通过索引重映射实现正确的注意力计算。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。

**PIT 在 Vertical-Slash Attention 中的应用**：

```
# 问题：垂直线列的 K, V 是非连续的（如 cols = [100, 5230, 8100, 12000, ...]）
# PIT 解决方案：将 B 个非连续列的数据加载为一个 dense tile

# Step 1: 收集非连续列为一组（group by B=64）
for j ← 0 to c_col step B:
    cols = i_col[index, j:j+B]                    # [B] — B 个 column indices
    # cols 中元素不连续，不能直接作为 GEMM 输入

# Step 2: PIT 加载（通过 shared memory 重排）
    K_chip = Load_Scattered(K, cols)               # 从 HBM 加载 B 行非连续 K
    V_chip = Load_Scattered(V, cols)               # 从 HBM 加载 B 行非连续 V
    # K_chip, V_chip 现在在 shared memory 中，视为连续 dense tile [B, d_h]

# Step 3: Dense GEMM（利用 Tensor Core）
    S = τ × Q_chip @ K_chip^T                      # [B, B] — 标准 dense matmul
    S = causal_mask(S)                              # 应用 causal mask
    # softmax, exp, 累加 (标准 FlashAttention 流程)
    O_chip = α × O_chip + P @ V_chip                # 标准 dense matmul

# PIT 的正确性保证：
# 假设 cols = [a, b, c, ...], dense GEMM 计算的是:
#   S[i] = Q_row · K[a]^T  (对 cols[0])
#   S[i] = Q_row · K[b]^T  (对 cols[1])
#   这些恰好是我们想要的稀疏注意力值（仅仅是索引不连续）
#   因为加法交换律，最终 O = Σ P_i × V_i 的结果与按原始索引计算一致
```

术语一般如何实现？如何使用？

PIT 在 GPU 上的实现：
1. **Shared Memory 重排**：使用 warp-level 的 `__shfl_sync` 或 shared memory 的 coalesced load，将 scattered global memory data 重排为连续布局
2. **Index Remapping**：在计算 softmax 和 write back 时，需要将 PIT tile 的内部索引映射回原始 sequence 中的位置
3. **与 FlashAttention 集成**：PIT part 作为 FlashAttention kernel 的第二个循环（第一个循环处理 block-sparse 斜线部分），共享相同的 online softmax 状态（m, l 向量）

使用场景：适用于任何需要处理非连续内存访问的稀疏计算场景，如：
- MoE 中的 token-to-expert dispatch（多个 token 的 FFN 输入被 gather 到同一 expert）
- 稀疏 attention 中的 column-level sparse patterns
- 任何需要将 irregular sparse data 转换为 regular dense compute 的场景

PIT 的开源实现整合在 MInference 代码库中（https://aka.ms/MInference），原始 PIT 编译器（SOSP '23）地址论文未明确给出。

涉及论文标题：
- MInference 1.0: Accelerating Pre-filling for Long-Context LLMs via Dynamic Sparse Attention
