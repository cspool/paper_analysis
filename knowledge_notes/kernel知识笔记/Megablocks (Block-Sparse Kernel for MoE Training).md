## Megablocks (Block-Sparse Kernel for MoE Training)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Megablocks (Gale et al. 2022) 是斯坦福大学提出的面向 MoE 高效训练的 block-sparse 矩阵乘法系统。核心贡献是实现了高效的 block-sparse 矩阵乘法 CUDA kernel，使得不同 expert 可以拥有不同的尺寸（不同 hidden dimension），仍能在一个 kernel 中批量计算，而无需传统的 token padding/dropping。传统 MoE 训练使用 expert parallelism + all-to-all dispatch/combine，各 expert 计算统一使用 GEMM——这要求所有 expert 接收的 token batch 大小相同（通过 capacity factor padding），引入大量冗余 FLOPs。Megablocks 通过两级 block-sparse 设计消除这一限制：(1) 外层 block-sparse 矩阵高效定位每个 token 对应的 expert；(2) 内层使用高速 CUDA kernel (CUTLASS/cuBLAS) 执行各 expert 的局部密集矩阵乘法，但所有 expert 的计算融合在单一 kernel 中，无需多次 kernel launch。

在 HMoE 中，Megablocks 的关键意义在于：异构 MoE 的不同 expert 具有不同的 FFN hidden dimension（如 2304→5888），无法使用统一 shape 的 GEMM。Megablocks 的 block-sparse kernel 原生支持混合形状的批量矩阵乘法，使 HMoE 的训练从工程层面成为可能。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Megablocks 的核心 kernel 设计（两级稀疏）：

```python
# Megablocks 处理 MoE 层 forward 的简化流程

# 输入: tokens [N, D]  按 expert assignment 排序后
# expert_offsets: [E+1]  各 expert 的 token 范围
#   expert_i tokens = tokens[expert_offsets[i] : expert_offsets[i+1]]

# Step 1: 构建 Block-Sparse 矩阵
# 将 E 个 expert 的 GEMM 表达为 Block Diagonal 矩阵:
# B = [[W_1,   0,   0]
#      [  0, W_2,   0]
#      [  0,   0, W_3]]
# 其中各 W_i 尺寸不同: W_1 [D, H_1], W_2 [D, H_2], ...

# Step 2: Block-Sparse GEMM (单 CUDA kernel)
# 使用 dCUDA block-sparse matrix multiply
# 各 CUDA block 负责一个 expert 的计算:
for each expert e with token range [start, end]:
    n_e = end - start
    if n_e > 0:
        # 局部 dense GEMM: [n_e, D] @ [D, H_e] → [n_e, H_e]
        CUBLAS_GEMM(tokens[start:end], W_e, output[start:end])

# Step 3: 输出 [N, ΣH_e]  — 各 expert 输出 concatenated
```

在 HMoE 的异构设置中，Megablocks 的优势尤为突出：
- Arithmetic distribution (2304→5888): 8 个不同形状的 W_e，传统方法需要 8 次独立 GEMM launch 或 pad 到 max(H_e)
- Megablocks: 单次 kernel launch，内部各 CUDA block 处理不同形状的子 GEMM

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Megablocks 开源实现：https://github.com/stanford-futuredata/megablocks。用于 MoE 训练框架中替代传统的 expert parallelism + all-to-all + GEMM pipeline。在 HMoE 的训练流程中，Megablocks 作为 expert 计算的后端，处理异构 expert 的不规则 GEMM。配合 DeepSpeed Zero2（参数分片）和 gradient checkpointing（激活检查点）实现高效训练。ES-MoE (Kim et al. 2024) 是对 Megablocks 的补充——通过 expert-wise offloading 到 CPU memory 并按需加载回 GPU，进一步缓解异构 expert 导致的 GPU memory 不均衡。

涉及论文标题：
- HMoE: Heterogeneous Mixture of Experts for Language Modeling
