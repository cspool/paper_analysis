## Dispatch Mask (in MoE Training / MoE训练中的调度掩码)

术语是什么？
Dispatch Mask 是 MoE 训练中使用 batched matrix multiplication 时的核心数据结构——一个巨大的映射表 (N_tokens_padded × N_tokens)，将任意顺序的 tokens 重新排序为 per-expert 连续排列，使每个 expert 的输入形成紧凑矩阵以支持 batched GEMM。

从kernel调度角度拆解术语：
Dispatch Mask 的构造：对每个 token，根据 gating result 计算其在 padding buffer 中的位置（expert_offset + counter），mask[pos][token_i] = 1。使用：input_per_expert = Dispatch_Mask @ input（稀疏矩阵乘法）。内存开销极大：MoE-L (d_model=1536, batch 32, seq 1024) → N_tokens=32768, Dispatch_Mask ~ (32×1024×32768)×4 bytes ≈ 4.3 GB per mask。

术语一般如何实现？如何使用？
在 Fairseq GShard/Tutel 中使用 sparse matmul 实现。megablocks 提出 block-sparse 变体减少存储。ES-MoE 的根本性方案：不使用 batched GEMM 和 dispatch mask，改用 sequential expert processing，tokens 按 gating 结果直接分配给各 expert。节省 >48 GiB GPU memory。

涉及论文标题：
- Scaling Beyond the GPU Memory Limit for Large Mixture-of-Experts Model Training
