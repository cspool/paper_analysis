## Full/Partial Block Optimization（Attention Block 分类优化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Full/Partial Block Optimization 是 FlexAttention 的 BlockMask 中的一种性能优化策略，将 score 矩阵的 block 分为三类以最小化运行时的 mask_mod 开销：
1. **Full Blocks**：block 内所有 score 均未被 mask（全部可见），运行时**跳过 mask_mod**，仅执行 score_mod。这是最常见的类型（如 causal mask 中严格上三角的 block）。
2. **Partial Blocks**：block 内部分 score 被 mask（部分设为 -inf），需运行时逐元素执行 mask_mod。这是对角线上的 block（如 causal mask 中同时包含可见和不可见 score 的对角 block）。
3. **Oblivious Blocks**：block 内所有 score 被 mask（全部 -inf），**完全跳过计算**。通过 kv_indices 自动排除。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
因果 mask（50% sparsity）的具体例子：
- Q_LEN=KV_LEN=16384, BS=128, 128×128 blocks
- Full blocks: 对角线上方约 50%（8128 个），跳过 mask_mod，仅执行 score_mod + softmax + PV GEMM
- Partial blocks: 对角线约 0.8%（128 个），需逐元素执行 mask_mod（对 q_idx < kv_idx 的位置设为 -inf）
- Oblivious blocks: 对角线下方约 50%，完全跳过计算

性能收益：对 causal mask 等常见模式，Full Block Optimization 带来约 15% 的额外性能提升。原因是 mask_mod 虽然是 element-wise 操作，但在每个 block 的 inner loop 中对所有 (q, kv) pair 逐元素执行仍带来可观的开销，而对角线上方的大多数 block 根本不需要任何 masking。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
分类在 `create_block_mask` 编译时完成：对每个 block，用 mask_mod 检查 block 的四个角是否全部 True（full）或全部 False（oblivious）或混合（partial）。结果编码在 BlockMask 中，运行时 GPU kernel 根据 block 类型自动选择执行路径。用户无需手动区分 block 类型。

涉及论文标题：
- Flex Attention: A Programming Model for Generating Optimized Attention Kernels
