## Early-Exit Sorting（提前退出排序）

术语是什么？通过联网搜索让回答具体和精准。

Early-Exit Sorting是一种排序优化策略，在排序过程中一旦累积结果满足退出条件即终止排序，避免对全部元素完整排序。在V-Rex的WTU（WiCSum Threshold Unit）中使用：对ScoreCluster per row的score按降序排列，从高分开始累积weighted sum，当累积值超过阈值Th_wics时立即停止，剩余低分元素不再排序或处理。关键原理：少量大score通常占weighted sum的大多数（V-Rex paper报告平均仅需处理16% scores即可达到threshold），从高分bucket开始可快速触发early exit。与完整排序（O(n log n)）相比，early-exit sorting的实际复杂度接近O(k log n)（k为实际处理的元素数），在k << n时显著降低latency和energy。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

WTU中Early-Exit Sorting的硬件pipeline：

```
// 硬件数据流：Bucket-based early-exit pipeline
// Input: ScoreCluster[row] [num_clusters], token_count [num_clusters]
//        Th_wics (precomputed: weighted_sum * 0.3)

// == Preprocess Step (parallel across all WTU cores) ==
weighted_sum = 0; min_score = INF; max_score = -INF
for j in 0..num_clusters-1:
    weighted_sum += ScoreCluster[j] * token_count[j]
    min_score = min(min_score, ScoreCluster[j])
    max_score = max(max_score, ScoreCluster[j])
Th_wics = weighted_sum * 0.3
score_range = max_score - min_score

// == Token Selection Step (early-exit pipeline) ==
num_buckets = 16  // WTU bucket count
curr_range_start = max_score
selected = []
acc_sum = 0

while curr_range_start >= min_score:
    range_end = curr_range_start - score_range/num_buckets

    // Bucket Sort (upper/lower sorters, parallel)
    in_range_bitmask = (ScoreCluster >= range_end) &
                       (ScoreCluster <= curr_range_start)
    // Sorters generate bitmask of scores in current range
    in_range_scores = ScoreCluster[in_range_bitmask]
    in_range_counts = token_count[in_range_bitmask]

    // Cumulative Sum Check
    for s, c in zip(in_range_scores, in_range_counts):
        acc_sum += s * c
        selected.append(index of cluster)
        if acc_sum > Th_wics:
            goto EXIT_EARLY  // ← Early Exit trigger

    curr_range_start = range_end

EXIT_EARLY:
// Output: selected cluster indices → map to token indices via HC table
// avg only 16% of total scores processed per row
```

GPU vs Hardware对比：GPU上早期退出需要global synchronization和conditional branching，与SIMT模型冲突导致warp divergence和underutilization。WTU专用硬件通过bucket sorters（upper + lower）和adder tree流水线化处理，无需synchronization，bitmask-based selection消除branching overhead。消融实验：AGX+ReSV (GPU)上KV prediction占48% latency → +KVPU (hardware)降至0.5%，其中WTU early-exit sorting贡献了significant reduction。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

硬件实现（V-Rex WTU）：每WTU core含upper bucket sorter、lower bucket sorter、multipliers、adder tree、bucket range updater。数据处理流程：preprocess预计算weighted sum和threshold→从高分bucket开始bucket sort（bitmask-based parallel sort）→multipliers+adder tree累积→比较器check→early exit。软件GPU模拟：使用PyTorch的`torch.sort(descending=True)`配合`torch.cumsum`和mask可实现功能等价版本，但缺少真正的early exit（仍需完整sort）。通用CPU实现可使用partial sort（`std::partial_sort`）或priority queue（`std::priority_queue`）配合cumulative sum check实现近似early exit。

涉及论文标题：
- V-Rex: Real-Time Streaming Video LLM Acceleration via Dynamic KV Cache Retrieval

