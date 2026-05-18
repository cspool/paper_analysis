## Page-Aware Windowed Clustering（页感知窗口化聚类）

术语是什么？通过联网搜索让回答具体和精准。
Page-Aware Windowed Clustering是AQPIM提出的算法-硬件协同设计技术，通过将PQ clustering的centroid分配限制在DRAM page/row大小内来实现高效的indirect lookup。核心原理：HBM-PIM每个bank有1KB row buffer（可存512个FP16 inner product values）。AQPIM将input sequence分割为多个window，每个window内的所有tokens被映射到不超过K=512个centroids——这意味着这些centroids对应的512个inner product values完全fit在单个DRAM row中。当attention lookup通过indices访问inner product values时，因为所有值已在同一row buffer中，只需1次row activation即可完成所有512次lookup。Window advance时，前window的centroids被复制到新page并更新以适应新tokens。

从kernel调度角度拆解术语：
```
// Page-Aware Windowed Clustering for Decode Attention Lookup
// 硬件约束: 1KB row buffer per bank = 512 FP16 values

window_size = L  // 初始: single window covers full sequence
if N_tokens generates > 512 unique centroid assignments:
    num_windows = ceil(N_tokens / W)
    
for each window w:
    // Step 1: Cluster within window
    centroids_w = weighted_kmeans(KV[w_start:w_end], K=512)
    
    // Step 2: Attention Inner Product computation
    IP_values[0:511] = query_sub × centroids_w^T  // stored in 1 row buffer
    
    // Step 3: Intra-row lookups (all within row buffer)
    ACTIVATE row_buffer  // single activation
    for idx in indices_w:
        partial_sum += COLUMN_SELECT(row_buffer, idx)
    
    // Step 4: Sum across subvectors and windows
    qKT[w_start:w_end] = partial_sum

// 优势: Row activations = num_windows (而非 N_indices)
```

术语一般如何实现？如何使用？
Implementation依赖于tight co-design：(a) Algorithm side: clustering restricted to fit each window's centroid count ≤512 (FP16) = row buffer size；(b) Hardware side: intra-row indirection (GRF→MUX→column decoder) executes lookup without additional row activations；(c) Window management: centroids copied from previous window as the window slides forward, updated incrementally for new tokens (not full reclustering). 对于大多数long-context scenarios，1个window映射整个sequence到512个centroids即足够（accuracy saturates at K=512, Table III）。当需要更多centroids时，多window方案扩展：每window最多512 centroids，window间independence使得lookup parallelism不受影响。复杂度和overhead: 每window 1 row activation → total activations = num_subvectors × num_windows (远小于 naive O(N) per lookup)。

涉及论文标题：
- AQPIM: Breaking the PIM Capacity Wall for LLMs with In-Memory Activation Quantization

