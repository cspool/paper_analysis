## Block-Sparse GEMM for MoE (MegaBlocks)

术语解释
由 Gale et al. (Stanford, 2023) 提出，将 MoE 计算重新表述为 block-sparse 操作，开发专用 block-sparse GPU kernel，在不丢弃 token 的前提下高效处理 MoE 的动态负载。

术语是什么？
传统 MoE 实现先 scatter token 到各 expert 组的连续 buffer，然后对每组执行 dense GEMM。这需要数据拷贝且可能因 expert capacity 限制丢弃 token。MegaBlocks 将 MoE 问题直接映射为 block-sparse matrix multiplication：按固定 block size（如 128×128）划分 token-expert 映射矩阵，只对非空 block 执行 batched dense GEMM。

从kernel调度角度拆解术语。
```
# MegaBlocks: token_expert_map S -> block-sparse -> batched dense GEMM
S = token_expert_map                          # [T, N]
blocks = split_into_blocks(S, block_size)     # List[Block]
nonzero_blocks = [b for b in blocks if b.nnz > 0]
for blk in nonzero_blocks:
    x_blk = x[blk.token_indices]                # [B, d_model]
    w_blk = experts_weights[blk.expert_id]      # [d_ffn, d_model]
    y_blk = x_blk @ w_blk.T                     # dense GEMM on block
```

术语一般如何实现？如何使用？
- 开源：https://github.com/stanford-futuredata/megablocks
- 基于 CUDA C++ + CUTLASS block-sparse GEMM
- GitHub Stars (2024.6): 1.1K
- 局限：scatter-to-group 数据拷贝增加内存，不易扩展到非 FFN 专家

涉及论文标题：
- A Survey on Mixture of Experts in Large Language Models
- Accelerating MoE Model Inference with Expert Sharding
- Dense Backpropagation Improves Training for Sparse Mixture-of-Experts（使用 gpt-neox + MegaBlocks + liger kernel (Triton) 进行 dropless MoE 训练，sequence length 2048, global batch size 1024）
- Duo-LLM: A Framework for Studying Adaptive Computation in Large Language Models
- Continual Pre-training of MoEs How robust is your router（CPT 实验使用 GPT-NeoX + Megablocks grouped GEMM kernel 在 64×A100 上进行 dropless MoE 训练。granular MoE (E=31, K=3) 每步约 1680ms，switch MoE (E=8, K=1) 约 1517ms，dense baseline 约 880ms。Megablocks grouped GEMM 将同一 batch 中路由到不同 expert 的 token 的 FFN 矩阵乘法打包为单次 batched GEMM，避免逐 expert 的小矩阵乘法 kernel launch overhead）

**MoEShard 中的 MegaBlocks 使用**：MoEShard 将 MegaBlocks 的 variable-sized block-sparse MM 用作 expert kernel fusion 的第二层优化。第一层（per-expert token concatenation）将 kernel launch 从 |E|×|G| 降至 |E|；第二层（MegaBlocks）进一步降至 1 次 kernel launch，使 kernel launch 数独立于 expert 数量。消融实验（Section 4.4）：expert < 64 时 MegaBlocks kernel 创建 overhead 使无 MegaBlocks 版略优；expert ≥ 64 时 MegaBlocks 优势递增；128 expert + 变 batch size 时 MegaBlocks 版全区间最优。

**Duo-LLM 中的 MegaBlocks 引用**：Duo-LLM 引用 MegaBlocks 证明 block-sparse matmul 可在单 GPU 上高效执行 MoE。论文假设 duo FFN 路由策略若减少 FLOPs 也将减少延迟，因为 auxiliary small FFN 足够小可与 big FFN 共存于单节点。但 Duo-LLM 未实现具体 efficient kernel 或 serving system，developing efficient implementation 被声明为"beyond the scope of this work"。

---
