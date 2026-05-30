## Low-Dimensional Importance Query and Key-Cache Projection (低维重要性查询与键缓存投影)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
一种将高维 KV-cache key（维度 D=128/head）和预测器输出的 importance query 同时投影到低维交互空间（d'=16）进行高效 token 重要性评分的机制。核心由两部分组成：(1) Query Prediction：producer layer 的 hidden states H ∈ R^{B×L×E} 经 LayerNorm + 二层 MLP 生成 G 个 slot-specific 低维 importance queries Q_imp ∈ R^{(B·H)×G×L×d'}，每个 slot 对应一个 consumer layer；(2) Key-Cache Projection：对每个 consumer layer l，其真实 KV-cache keys K^(l) ∈ R^{B×H_kv×L×D} 通过学习投影矩阵 W_K^(l) ∈ R^{D×d'} 降维到同样的 d' 维空间，得到 K_imp^(l) = K^(l) · W_K^(l) ∈ R^{B×H_kv×L×d'}。Token 重要性分数 = Q_imp[slot] @ K_imp^T，计算复杂度为 O(L·d')，仅为完整 attention O(L·D) 的 d'/D = 16/128 = 1/8。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
在 Transformer pipeline 中的定位：

```
Transformer Layer (consumer layer l):
    # Stage 0: Importance scoring（仅在预测步运行）
    if is_prediction_step:
        # 在 producer layer 已计算好 Q_imp
        K_proj = K_cache[l] @ W_K[l]        # GEMM: [L, D] × [D, d'] → [L, d']
        scores = Q_imp[slot] @ K_proj.T      # MatMul: [H, 1, d'] × [H, d', L] → [H, L]
        # scores[h, t] = Σ_{k=1}^{d'} Q_imp[h,slot,k] · K_proj[h,t,k]
        topk_indices = argtopk(scores, B)     # 选 top-B token
    
    # Stage 1: 构建稀疏 KV 集合
    K_sparse = gather(K_cache[l], [sink | topk_indices | local_window])
    V_sparse = gather(V_cache[l], [sink | topk_indices | local_window])
    
    # Stage 2: 标准 Attention (FlashAttention)
    output = FlashAttention(Q_current, K_sparse, V_sparse)
```

重要性评分计算的具体张量操作：
- Q_imp shape: [B, H, G, d'] (per-producer step, per-token)
- K_proj shape: [B, H_kv, L_kv, d'] (预先计算并缓存)
- scores shape: [H, L_kv] per consumer layer
- 使用 GQA 时：H_kv 个 key head 的 score broadcast 到 H 个 query head

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：K-cache 投影在 token 离开 local window 时批量执行（每 N 步一批），利用 cuBLASS GEMM 批量处理 N 个 token 的投影，充分利用 GPU HBM 带宽。Q_imp 预测使用单次 MLP forward（约 512×d'×G×H 次乘加），远小于一层 transformer 的 attention + FFN。d'=16 的选择在精度和效率间平衡：更小的 d' 更快但 recall 下降（ablation 显示 d'=16 时 Recall@50% ≈ 67% for 3.48M predictor）。投影矩阵 W_K 与 token 位置无关（position-agnostic），仅依赖层号和 key head，因此可预先计算并缓存投影结果。

涉及论文标题：
- TokenButler: Token Importance is Predictable
