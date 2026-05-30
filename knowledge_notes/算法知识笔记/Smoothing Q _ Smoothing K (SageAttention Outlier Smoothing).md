## Smoothing Q / Smoothing K (SageAttention Outlier Smoothing)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Smoothing Q 和 Smoothing K 是 SageAttention 系列（SageAttention → SageAttention2 → SageAttention3）中用于提升低比特量化 attention 精度的离群值平滑技术。问题来源于 Q 和 K 矩阵中存在统计离群值（outliers），这些大值在 per-block 或 per-tensor 量化时主导 scale factor 的计算（scale = max(|X|)/N），导致大量小值被量化到极少数 level 或完全归零。Smoothing K（SageAttention 提出）对 K 做 per-head 均值减法：K ← K - mean(K)，消除 K 的 DC 分量，因为 mean(K) 主导了 QK^T 结果但本身不含位置信息。Smoothing Q（SageAttention2 提出）对 Q 做 per-block 均值减法：Q_i ← Q_i - mean(Q_i)，补偿为 Q_i - q̄_i 参与 FP4 QK^T + 额外的 GEMV(q̄_i, K_j^T) 修正项。SageAttention3 继承了两者。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// Smoothing K (applied once per head, pre-loop)
K_m = mean(K)           // per-head mean, shape [1×d]
K ← K - K_m             // K centered

// Smoothing Q (per-block, inside loop)
for each Q_i block:
    q̄_i = mean(Q_i)     // per-block mean, shape [B_q]
    Q_i' = Q_i - q̄_i    // centered Q block
    
    // FP4 quantize centered Q
    s_Q, Q̂_i = φ(Q_i')
    
    // QK^T with correction
    S_ij = FP4MM(Q̂_i, s_Q, K̂_j, s_K) + GEMV(q̄_i, K_j^T)
    // GEMV term recovers the mean contribution exactly in FP16
```

Ablation 结果（CogVideoX-2B, CosSim）：No smoothing 0.916 → SmoothQuant 0.930 → Hadamard 0.941 → Smoothing_Q 0.983 → Smoothing_K 0.991。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现方式：Smoothing K 作为预处理 kernel，对每个 head 的 K 做 row-wise mean subtraction，与 K 的 FP4 量化 kernel 融合（加载 K → 计算 mean → 减去 mean → 用量化值填充 packed FP4）。Smoothing Q 在 attention inner loop 中，对每个 Q block 在线计算 mean 并减去，用 GEMV 补回（FP16 精度，计算量很小 O(B_q × d) vs O(B_q × B_kv × d)）。两项平滑技术在所有 SageAttention 系列的 INT8/INT4/FP4 attention 中均使用。SmoothQuant 和 Hadamard 变换虽然也是离群值抑制方法，但在 attention 量化中效果不如 Smoothing Q/K（因为前者针对 weight-activation 量化设计，后者针对 attention 特有分布）。

涉及论文标题：
- SageAttention3: Microscaling FP4 Attention for Inference and An Exploration of 8-Bit Training
