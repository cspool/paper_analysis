## N:M Semi-structured Sparsity (N:M 半结构化稀疏)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
N:M Semi-structured Sparsity 是 NVIDIA Ampere 架构引入的稀疏模式：每 M 个连续权值中保留 N 个非零。最常见 2:4——每 4 个权值保留 2 个（50% 稀疏），Tensor Core 直接跳过零值，接近 2× 加速。相比 unstructured sparsity（零值随机分布、难硬件加速），N:M 在硬件友好性和精度间取得平衡。

从算法pipeline角度拆解术语：
```
for row in d_row:
    for g in range(0, d_col, M=4):
        idx = argtop2(S[row, g:g+4])  # 保留最重要的 2 个
        mask[g:g+4] = 0; mask[g+idx] = 1
W_sparse = W * mask  # 2:4 pattern
```
MoE-Pruner 将 unstructured 度量 S 的 comparison group 从"每行"改为"每 M 个连续权值"扩展为 N:M。Mixtral-8x7B 2:4 下 perplexity 5.88（SparseGPT 7.09, Wanda 6.98）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- Ampere+ Tensor Core 通过 cuBLAS/cuSPARSE 支持。局限：精度损失高于同比例 unstructured；非所有 GEMM 受益。

涉及论文标题：
- MoE-Pruner: Pruning Mixture-of-Experts Large Language Model using the Hints from Its Router
