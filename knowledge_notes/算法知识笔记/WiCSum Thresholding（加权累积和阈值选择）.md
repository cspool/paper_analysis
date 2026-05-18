## WiCSum Thresholding（加权累积和阈值选择）

术语是什么？通过联网搜索让回答具体和精准。

WiCSum（Weighted Cumulative Sum）Thresholding是V-Rex提出的动态token选择算法，替代fixed top-k。对每layer/head独立计算Query×KeyCluster^T得ScoreCluster矩阵，按每行score×token_count加权求和得weighted sum，从高分bucket开始累积直到超过动态阈值（Th_wics = 0.3 × weighted sum）即停止。结果：每layer/head自适应选择不同数量token（论文Fig.20：layer间4.2%-44.0%，head间也有显著差异），平均比ReKV少检索3.0× token，accuracy仅下降0.8%。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// Input: ScoreCluster [num_queries, num_clusters]
//        HC_table: {token_count per cluster}
// Thr_wics = 0.3

// Step 1: Preprocess per row
for each row i:
    Sum_i = Σ_j (ScoreCluster[i][j] * HC_table[j].token_count)
    Th_wics_i = Sum_i * Thr_wics

// Step 2: Early-Exit Token Selection per row
sorted = sort_descending(ScoreCluster[i])
Acc_i = 0; selected = []
for t = 0 to num_clusters-1:
    Acc_i += sorted[t].score * HC_table[sorted[t].cluster].token_count
    selected.append(sorted[t].cluster)
    if Acc_i > Th_wics_i:
        break  // Early exit: avg only 16% of row processed

// Step 3: Aggregate
all_selected = unique(∪ selected_i for all rows)
selected_tokens = map_to_tokens(all_selected, HC_table)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

在V-Rex中，WTU（WiCSum Threshold Unit）硬件加速thresholding：每core含score memory/token count memory/upper-lower bucket sorters/multipliers/adder tree/bucket range updater。Preprocess step预计算weighted sum和threshold→Token selection step从高分bucket做bucket sort→cumulative sum→与threshold比较→early exit。通用PyTorch实现核心~50行：`cumsum = (sorted_scores * token_counts).cumsum(dim=-1); mask = cumsum <= threshold; selected = sorted_indices[mask]`。论文未开源。

涉及论文标题：
- V-Rex: Real-Time Streaming Video LLM Acceleration via Dynamic KV Cache Retrieval

