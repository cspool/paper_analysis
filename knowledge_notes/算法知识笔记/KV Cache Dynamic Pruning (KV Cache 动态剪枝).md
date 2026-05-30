## KV Cache Dynamic Pruning (KV Cache 动态剪枝)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
KV Cache Dynamic Pruning 是一种在 LLM/VLLM 推理的 decoding 阶段，动态评估并调整 KV cache 中保留的 token 集合的技术。与传统 one-shot token pruning（仅在 prefilling 阶段一次性评估 token 重要性并固定剪枝结果）不同，动态剪枝在 decoding 的每一步或每隔 N 步，重新计算当前预测 token 与 KV cache 中视觉 token 的 cross-attention 权重矩阵 A^(L) = Softmax(Q^(L) K^(L)^T / √D)，按 attention score 的 top-p% 阈值动态决定保留哪些 token 在 KV cache 中参与注意力计算。低于阈值的 token 并非永久丢弃，而是移入一个独立的 Dynamic Pruning Cache (DP Cache) 以备后续步骤重新召回。DyCoke 论文的消融实验证明：去除动态机制（改为 one-shot 剪枝）后 VideoDC benchmark 性能显著下降，验证了动态评估的必要性。相关方法包括 Lethe（AAAI 2026，layer-adaptive + recency-aware 动态剪枝）、KVzap（NVIDIA，MLP 预测重要性分数的动态剪枝）、SparK（channel-level 剪枝 + 动态重建）等。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
在 DyCoke 的 decoding pipeline 中，动态剪枝的执行流程如下：
```
# === Decoding Stage: KV Cache Dynamic Pruning ===
# 输入: KV_cache (prefilling后填充), 超参数 L (注意力层), P (保留比例p%)
# DP_cache = {}  (存储被剪枝的token，可召回)

for decoding_step t in range(max_new_tokens):
    # Step 1: 判断是否需要重新评估注意力分布
    if t == 1 or cosine_sim(attention_dist_prev, attention_dist_curr) < threshold_sim:
        # Step 2: 在第 L 层计算当前预测 token 对视觉 token 的 cross-attention
        A = Softmax(Q_pred K_visual^T / sqrt(D))  # shape: (1, N_visual)
        
        # Step 3: 按 top-P% 阈值分离保留与剪枝 token
        threshold_tau = percentile(A, 100 - P)
        keep_idx = where(A >= threshold_tau)
        prune_idx = where(A < threshold_tau)
        
        # Step 4: KV cache 与 DP cache 双向更新
        KV_cache_visual[L] = KV_cache_visual[L][keep_idx]
        DP_cache[L] = DP_cache[L] ∪ KV_cache_full[L][prune_idx]
        
        # Step 5: 召回 DP cache 中注意力回升的 token
        A_dp = Softmax(Q_pred K_DP^T / sqrt(D))
        recall_idx = where(A_dp >= threshold_tau_new)
        KV_cache_visual[L] ∪= DP_cache[L][recall_idx]
        DP_cache[L] -= DP_cache[L][recall_idx]
    
    # Step 6: 使用压缩后的 KV cache 执行 attention 并生成下一 token
    h_t = attention(Q_pred, KV_cache_visual, KV_cache_text)
    KV_cache = concat[KV_cache, (h_t W_K, h_t W_V)]
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
DyCoke 中的实现基于 LLaVA-NeXT 代码库，通过 lmms-eval 评估框架调用。关键超参数：L=3（在第 3 层评估 attention，实验证明 L>0 时对性能影响不敏感，说明动态剪枝的稳定性）、P=0.7（保留 top-70% attention 的 token，即剪枝 30%）、K=0.5~0.7（TTM 第一阶段剪枝率）。动态剪枝与 KV cache 的兼容性：DyCoke 兼容 Flash Attention，仅在特定层额外计算一次 cross-attention，复杂度远低于 prefilling 阶段。相关工具：kvpress（HuggingFace，支持 KVzap 等动态剪枝 pipeline）、TRL、PyTorch。

涉及论文标题：
- DyCoke__Dynamic_Compression_of_Tokens_for_Fast_Video_Large_Language_Models
