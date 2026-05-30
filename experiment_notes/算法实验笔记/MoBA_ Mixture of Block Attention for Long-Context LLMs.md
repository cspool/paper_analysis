## MoBA: Mixture of Block Attention for Long-Context LLMs

- 属于算法pipeline的实现是什么？实验比较什么？
  MoBA（Mixture of Block Attention）是一种将 Mixture of Experts（MoE）原理应用于注意力机制的长上下文稀疏注意力架构。核心设计：(1) **Block Partitioning and Routing**：将完整上下文划分为 n 个等大小的 KV block（block size B=N/n），每个 query token 通过 gating 网络（inner product with mean-pooled K）计算与每个 block 的 affinity score s_i = ⟨q, mean_pool(K[I_i])⟩，使用 top-k gating 选择最相关的 k 个 block；(2) **Causality Preservation**：禁止 query 关注 future blocks（s_i=−∞），强制每个 token 必须关注当前 block 并施加 causal mask，当前 block 类似 MoE 中的 shared expert；(3) **Hybrid of MoBA and Full Attention**：MoBA 与 full attention 参数等价（无参数增减），支持在训练阶段无缝切换——两阶段训练先 MoBA 后 full attention，或 layer-wise hybrid（最后几层保留 full attention）；(4) **Fine-Grained Block Segmentation**：类似 MoE 中 fine-grained expert segmentation 可提升性能。

  实验比较 MoBA vs Full Attention 在：(a) scaling law 实验（5 个模型规模 568M-2.1B，Chinchilla scaling，seqlen 8K/32K，block size=512, top-k=3，sparsity 81.25%-95.31%）；(b) hybrid training 策略（MoBA 90% tokens + Full Attn 10% tokens vs full-only vs MoBA-only）；(c) layer-wise hybrid SFT（最后 N 层为 full attention）；(d) 大规模下游评估（Llama-8B-1M-MoBA vs Llama-8B-1M-Full，从 Llama 3.1 8B 出发做 continual pre-training 至 1M context，block size=4096, top-k=12, sparsity 95.31%，最后 3 层保留 full attention）；(e) 效率基准（forward pass 时间 vs FlashAttention，1M-10M context，speedup 6.5× at 1M, 16× at 10M）。

- 硬件平台是什么，配置是什么。
  Scaling law 实验：论文未明确说明 GPU 型号。大规模评估（Llama-8B-1M）：多个 GPU，使用 tensor parallelism（将 K/V broadcast 到 distributed query heads 解决 10M context 显存限制）。效率测试（Section 3.4）：对比 FlashAttention baseline 在单 GPU 上的 forward pass 时间（图 2）。基于 FlashAttention 和 DeepSpeed-MoE 的实现。

- 模型是什么。数据集和bench分别是什么。
  模型：5 个 scaling law 模型（568M/822M/1.1B/1.5B/2.1B，配置见 Table 1），Llama 3.1 8B Base 作为继续预训练起点→Llama-8B-1M-MoBA（32 layers, 最后 3 层 full attention, 29 层 MoBA）。
  
  Benchmark：AGIEval, BBH, CEval, GSM8K, HellaSWAG, Loogle, Competition Math, MBPP, MBPP Sanitized, MMLU, MMLU Pro, OpenAI HumanEval, SimpleQA, TriviaQA, LongBench@32K, RULER@128K, Needle in the Haystack (up to 1M)。

- 开源情况。基于开源文档和论文，使用例子解释，算法pipeline，至少具体到伪代码或张量计算。
  开源：https://github.com/MoonshotAI/MoBA。基于 PyTorch + FlashAttention + DeepSpeed-MoE 实现。

  **Algorithm 1 张量级 pipeline（MoBA 前向）**：
  ```
  输入: Q, K, V ∈ R^{N×h×d}, block size B, top-k
  n = N/B  # number of blocks

  # Step 1: Split KV into blocks
  K̃_i, Ṽ_i = split_blocks(K, V, B)  # i ∈ [n], K̃_i ∈ R^{B×h×d}

  # Step 2: Compute gating scores (Eq. 6)
  K̄ = mean_pool(K, B)  # R^{n×h×d}, mean pooling along seq dim per block
  S = Q @ K̄^T          # R^{N×h×n}, affinity score per query per head per block

  # Step 3: Causal mask + top-k gating (Eq. 5)
  M = create_causal_mask(N, n)     # mask future blocks to -inf
  G = topk(S + M, k)               # G: binary gating matrix R^{N×h×n}

  # Step 4: Organize query-to-block assignments
  # Self-attention block: current block (always attended, causal=True)
  Q^s, K̃^s, Ṽ^s = get_self_attn_block(Q, K̃, Ṽ)
  # MoBA attention blocks: top-k selected historical blocks (causal=False)
  Q^m, K̃^m, Ṽ^m = index_select_moba_attn_block(Q, K̃, Ṽ, G)

  # Step 5: Compute attention via FlashAttention varlen
  O^s = flash_attn_varlen(Q^s, K̃^s, Ṽ^s, causal=True)
  O^m = flash_attn_varlen(Q^m, K̃^m, Ṽ^m, causal=False)

  # Step 6: Combine with online softmax (tiling)
  O = combine_with_online_softmax(O^s, O^m)
  ```

  **计算复杂度**：每个 query 仅关注 k 个 block（每个 block B tokens）+ 当前 block = (k+1)B tokens，复杂度从 O(N²) 降至 O((k+1)B·N) = O(kN²/n)。例如 N=1M, B=4096, k=12 时 sparsity = 1−(4096×13)/1M = 94.7%。

  **Hybrid Training Recipe（两阶段）**：
  - Stage 1: 90% tokens 使用 MoBA 训练
  - Stage 2: 10% tokens 切换到 full attention 训练
  - 切换时无显著 loss spike

  **Layer-wise Hybrid (推理/SFT)**：
  - 最后 3 层保留 full attention，其余层使用 MoBA
  - SFT 阶段 prompt tokens 被 mask 掉 loss，稀疏梯度从 unmasked tokens backprop 受限
  - 使用推理时 switching：prefill 用 MoBA，generation 用 full attention

  **具体配置 (Llama-8B-1M-MoBA)**：
  - Context: 128K→256K→512K→1M continual pre-training
  - MoBA: block size=4096, top-k=12, sparsity=95.31%
  - Layer-wise: 最后 3 层 full attention, 29 层 MoBA
  - Position interpolation (Chen et al. 2023) 用于 256K transition
