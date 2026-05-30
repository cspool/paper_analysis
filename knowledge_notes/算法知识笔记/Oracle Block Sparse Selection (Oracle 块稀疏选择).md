## Oracle Block Sparse Selection (Oracle 块稀疏选择)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Oracle Block Sparse Selection 是一种评估稀疏注意力方法准确率上界的实验技术：使用真实完整注意力分数（ground truth attention scores）来选择哪些 KV blocks 参与计算，而非使用任何近似或预测方法。由于需要先计算完整 attention 再做选择，oracle 方法本身无法加速推理，但可以回答"如果稀疏选择是完美的，模型准确率能保持到什么程度？"

在 SeerAttention-R 中，oracle sparsity 的实现：
1. 对每个 decode step，先计算完整 attention scores (Q @ K^T)
2. 对 attention scores 做 column-wise 1D maxpooling（每个 block 取最大值）
3. 对 GQA group 内做 maxpool 得到 KV-head 级别的分数
4. Top-K 选择分数最高的 blocks
5. 仅用选中的 blocks 重新计算 attention（实际上可以复用第一次的结果）

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
# Oracle Block Sparse Selection 评估流程
def oracle_sparse_eval(model, prompt, block_size=64, token_budget=4096):
    block_budget = token_budget // block_size
    
    for each decode step t:
        # Step 1: 计算完整 attention（仅在评估时做，实际部署不这么做）
        Q = current_query_token        # [1, num_heads, d_head]
        K = past_kv_cache              # [t, num_kv_heads, d_head]
        A_full = Q @ K.T / sqrt(d_head)  # [1, num_heads, t]
        
        # Step 2: Block-level maxpooling + GQA group pooling
        A_blocks = column_maxpool(A_full, block_size)  # [1, num_heads, num_blocks]
        A_kv = maxpool_over_gqa_group(A_blocks)         # [1, num_kv_heads, num_blocks]
        
        # Step 3: Oracle Top-K（完美选择）
        selected = topk(A_kv, k=block_budget)            # ground truth 最优选择
        
        # Step 4: 计算 sparse attention（可用完整 A_full 中对应 block 的结果）
        O = compute_attention_on_selected(Q, K, V, selected)
        
    return model_accuracy  # 这就是 sparse attention 的准确率上界
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Oracle sparsity 主要用于：(1) 验证 attention 是否本身具有稀疏性——即确定是否仅需一小部分 KV blocks 即可保持准确率；(2) 为稀疏预测方法（如 AttnGate）提供准确率上界参考。SeerAttention-R 的 oracle 实验显示：Qwen3-14B 在 AIME 上，block_size=64 时 2k token budget 达 lossless，验证了推理 attention 的内在稀疏性。稀疏预测方法（AttnGate）达到 4k budget 才能 lossless，与 oracle 的 2k budget 有 gap，反映了稀疏预测的近似误差。

涉及论文标题：
- SeerAttention-R: Sparse Attention Adaptation for Long Reasoning
