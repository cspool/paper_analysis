## PIT (Permutation Invariant Transformation) for MoE

术语解释
由 Zheng et al. (2023) 提出，利用 Permutation Invariant Transformation（f(P·x) = P·f(x)，P 为置换矩阵）将 MoE 的稀疏微 tile 重组为 GPU 高效 dense tile，在不改变计算结果的前提下提升 GPU 利用率。

从kernel调度角度拆解术语。
```
# PIT Tiling: sparse micro-tiles -> dense tile -> GEMM -> inverse
micro_tiles = extract_per_expert_tiles(input_tokens, tile_size)
dense_tile = pit_rearrange(micro_tiles)      # 置换不变性保证等价
result = dense_gemm(dense_tile, merged_W)     # high GPU utilization
output = pit_inverse_rearrange(result)         # 恢复原始顺序
```

术语一般如何实现？如何使用？
- 作为深度学习编译器图优化 pass
- 需要形式化证明 operator 满足 PIT 性质
- 开源情况：论文未明确给出独立开源链接

涉及论文标题：
- A Survey on Mixture of Experts in Large Language Models
