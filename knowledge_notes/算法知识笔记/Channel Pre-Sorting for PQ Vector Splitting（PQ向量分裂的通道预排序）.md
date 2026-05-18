## Channel Pre-Sorting for PQ Vector Splitting（PQ向量分裂的通道预排序）

术语是什么？通过联网搜索让回答具体和精准。
Channel Pre-Sorting是AQPIM提出的PQ向量分裂预处理步骤，解决标准PQ split vectors不考虑inter-channel similarity导致高quantization error的问题。方法基于cosine similarity将高相关channel聚类到同一subvector中：随机选一reference channel→计算所有channel对其cosine similarity→greedily选择top-k most similar channels形成group→重复直至所有channel被分配。生成的sorting matrices P_k, P_v可以absorb到projection weights中：W_q'=W_q·P_k, W_k'=W_k·P_k, W_v'=W_v·P_v, W_o=W_o·P_v^T。矩阵离线生成（calibration dataset: Wikitext-2-v1），推理时零额外开销。与SKVQ的channel reorder方法类似但目标不同：SKVQ为minimize uniform quantization error within per-group scaling，AQPIM为maximize subvector内channel affinity for PQ clustering。

从算法pipeline角度拆解术语：
```
// 离线阶段: Channel Sorting Matrix Generation
// 输入: Calibration KV activation samples, d=4096 channels
// 输出: Permutation matrix P_k, P_v ∈ R^{d×d}

channels_remaining = {1, 2, ..., d}
groups = []  // m groups, each ~d/m channels
for g in 1..m:
    ref = random_choice(channels_remaining)
    similarities = []
    for ch in channels_remaining:
        similarities[ch] = cosine_similarity(K[:, ref], K[:, ch])
    top_k = argsort(similarities, descending)[:d/m]
    groups[g] = top_k
    channels_remaining -= top_k

// 构建permutation matrix: P_k[i,j]=1 if channel j belongs to group[i]
// 吸收到projection: W_k' = W_k @ P_k  // offline weight transformation
```

术语一般如何实现？如何使用？
Sorting matrices在calibration dataset (Wikitext-2-v1)上离线生成，作为static permutation absorb到model projection weights——在线inference完全透明。Ablation (Table IV): AQPIM w/o pre-sort avg 48.76 → full AQPIM avg 50.00 (+1.24 avg points)。Channel pre-sorting与importance-weighted clustering互补：pre-sorting提高subvector内cohesion (reduce intra-group quantization error)，weighting减少critical token的error。两者combined效果好于单独使用。

涉及论文标题：
- AQPIM: Breaking the PIM Capacity Wall for LLMs with In-Memory Activation Quantization

