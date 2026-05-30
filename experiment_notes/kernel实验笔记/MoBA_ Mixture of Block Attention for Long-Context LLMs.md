## MoBA: Mixture of Block Attention for Long-Context LLMs

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现了融合 FlashAttention 和 MoE 优化技术的 MoBA attention kernel（Algorithm 1）。核心实现包含五个步骤：(1) 根据 gating network 和 causal mask 确定 query-to-KV-block 分配；(2) 按分配的 KV block 重排 query tokens 顺序；(3) 对每个 KV block 和分配给它的 queries 使用 FlashAttention varlen 分别计算 block-wise attention；(4) 将 attention outputs 重排回原始顺序；(5) 使用 online softmax (tiling) 合并 output——因为一个 query 可能关注当前 block 和多个历史 block。Attention 计算被分为两部分：self-attention block（当前 block, causal=True）通过 `get_self_attn_block` 处理，MoBA blocks（top-k 选中的历史 blocks, causal=False）通过 `index_select_moba_attn_block` 处理，两部分用 `combine_with_online_softmax` 合并。实验比较 MoBA kernel vs FlashAttention 在 1M model 上的 forward pass 时间（seqlen 8K-1M, Figure 2a）和固定 sparsity 95.31% 下的 scaling（8K-10M, Figure 2b）。

- 后端平台是什么，配置是什么。
  具体 GPU 型号论文未明确说明（标注为 Moonshot AI 内部集群）。使用 tensor parallelism 扩展至 query head level：将 K、V tensors broadcast 到 distributed query heads 以解决 10M context 下的 GPU 显存限制。基于 FlashAttention (Dao et al. 2022) 和 DeepSpeed-MoE (Rajbhandari et al. 2022) 实现。

- 评估性能的软件/脚本是什么。修改了什么。
  基于 PyTorch + FlashAttention + DeepSpeed-MoE。核心实现/修改：
  
  1. **KV Block Splitting + Mean Pooling**（Algorithm 1 Lines 1-4）：将 K, V ∈ R^{N×h×d} 按 block size B 划分为 n=N/B 个 block K̃_i ∈ R^{B×h×d}，mean_pool 沿 sequence 维度计算 K̄ ∈ R^{n×h×d} 作为 block-level key representation。

  2. **Gating Score + Top-k Selection**（Lines 5-8）：Q @ K̄^T → S ∈ R^{N×h×n}，加 causal mask M（future blocks = -∞），topk(S+M, k) → G ∈ {0,1}^{N×h×n}，得到稀疏 query-to-block mapping。

  3. **Varlen FlashAttention Computation**（Lines 9-14）：`index_select_moba_attn_block` 根据 G 将 queries 分组到各 KV block，输出变长序列 Q^m/K̃^m/Ṽ^m；`flash_attention_varlen` 对每个 (query_group, kv_block) 对执行 FlashAttention；当前 block attention（`causal=True`）和历史 block attention（`causal=False`）分别计算。

  4. **Online Softmax Combining**（Line 16）：使用 online softmax tiling (Milakov et al. 2018; Liu et al. 2023) 将 self-attention output O^s 和 MoBA attention output O^m 合并——因为一个 query 可能同时关注当前 block 和多个历史 blocks，需从不同 attention 分片的 partial softmax 合并出最终结果。

  5. **Tensor Parallelism for 10M context**：将 K、V broadcast 到不同 query heads（tensor parallelism over heads），各 head 独立持有完整 K/V 但仅计算自己负责的 Q heads 的 attention，有效突破单 GPU 显存限制。

  6. **MoBA/Full Attention Switching**：attention 层可在 MoBA 和 full attention 间动态切换，gating 计算仅在 MoBA 模式下触发。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  开源：https://github.com/MoonshotAI/MoBA

  **评估原理与 Kernel 执行全流程（以单个 MoBA attention layer prefill 为例）**：

  ```
  输入: Q, K, V ∈ R^{N×h×d}（N=context length, h=num_heads, d=head_dim）
       B=block_size, k=top_k
  输出: O ∈ R^{N×h×d}

  Step 1: Block Partitioning [GPU: split]
    n = N / B
    for i in 1..n:
        K̃_i = K[(i-1)B : iB]  # [B, h, d], slice from HBM
        Ṽ_i = V[(i-1)B : iB]

  Step 2: Block Mean Pooling [GPU: reduce]
    K̄ = mean_pool(K, dim=0, group_size=B)  # reduce along seq dim
    # K̄ ∈ R^{n×h×d}, each entry is mean of B key vectors

  Step 3: Gating Score [GPU: bmm]
    S = Q @ K̄.transpose(-1, -2)  # [N, h, n]
    # S[q_idx, head, block_idx] = affinity of query q to KV block block_idx

  Step 4: Causal Mask + TopK [GPU: mask + topk]
    M[pos][i] = -inf if pos < i*B  # no future blocks
    G = topk(S + M, k, dim=-1)  # [N, h, n], binary
    # G[q][b] = 1 if block b is selected for query q

  Step 5: Query-to-Block Assignment [GPU: index_select]
    # Self-attention block
    for each block i:
        Q_i^s = queries whose position ∈ I_i  # queries in this block
        K_i^s = K̃_i, V_i^s = Ṽ_i
    # MoBA history blocks
    Q^m, K̃^m, Ṽ^m = index_select_moba_attn_block(Q, K̃, Ṽ, G)
    # Group queries by their assigned blocks, output varlen tensors

  Step 6: FlashAttention Varlen [GPU: FlashAttn kernel]
    # Self-attention (current block only, with causal mask)
    O^s = flash_attn_varlen(Q^s, K̃^s, Ṽ^s, causal=True)
    # MoBA attention (selected history + current blocks, no causal mask
    #   because causal is already enforced by block-level routing)
    O^m = flash_attn_varlen(Q^m, K̃^m, Ṽ^m, causal=False)

  Step 7: Online Softmax Combining [GPU: fused kernel]
    O = combine_with_online_softmax(O^s, O^m)
    # Principle: lse_i (log-sum-exp) from each partial attention,
    # re-weight and re-scale O^s and O^m with their respective lse
    # Equivalent to:
    #   lse_total = logsumexp([lse_s, lse_m])
    #   O = (exp(lse_s - lse_total) * O^s + exp(lse_m - lse_total) * O^m)
    # In practice: online tiling, no explicit lse materialization
  ```

  **评估指标与性能**（Figure 2）：
  - 1M model speedup (Figure 2a): 8K→1M 序列长度，MoBA vs FlashAttention forward pass time
    - 1M tokens: MoBA ~0.15s vs FlashAttn ~1.0s → **6.5× speedup**
    - 随序列增长 MoBA 呈 sub-quadratic scaling
  - Fixed sparsity ratio scaling (Figure 2b): 8K→10M, fixed blocks=64, top-k=3, sparsity=95.31%
    - 10M tokens: MoBA ~1.5s vs FlashAttn ~24s → **16× speedup**
    - 小序列（32K-512K）两者接近，长序列下优势显著

  **FlashAttention 集成要点**（Figure 1b）：
  - FlashAttention blocks 内嵌 MoBA 的 block routing 逻辑
  - Q^m, K̃^m, Ṽ^m 已经按 block 分组排列，FlashAttention varlen 直接处理
  - "Varlen" 指各 block 的 query count 不同（取决于 top-k routing 结果）
  - 最终 online softmax combine 保证数值等价于完整 softmax attention
