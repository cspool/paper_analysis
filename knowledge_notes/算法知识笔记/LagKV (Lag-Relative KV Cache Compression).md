## LagKV (Lag-Relative KV Cache Compression)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

LagKV 是一种完全不依赖注意力权重（attention-weight-free）的 KV Cache 压缩/逐出方法。其名称中的 "Lag" 指代滞后参考（lag-relative）——使用下一个相邻分区的 KV 统计量作为参考来归一化和评分当前分区的 token。核心洞察基于两点：(1) 自回归模型中 token-wise locality——临近位置的 token 具有更相似的 K/V 张量值；(2) K/V 张量的 per-channel 分布特性（K 的 channel-wise variance 一致，V 的 token-wise variance 显著），使得 channel-wise 标准差成为有效的 token 重要性指标。

评分流程：(a) 将 KV cache 按 lag size L 递归分区；(b) 对每个分区 p，使用分区 p+1 的 token-wise max/min 对分区 p 进行归一化（消除 token-wise locality 导致的 channel 偏移）；(c) 计算归一化后 K/V 的 channel-wise 标准差；(d) softmax 转化为概率分布；(e) 对 K 和 V 的分数求和；(f) top-K 选择保留 rL 个 token。同时保留 attention sink（前 S 个 token，默认 S=16）和最后一个分区作为滑动窗口。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
# === LagKV 递归压缩 (每层每 head 独立执行) ===
def lagkv_compress(K, V, S=16, L=128, r=0.5):
    # K, V: (num_heads, seq_len, head_dim)
    # Step 1: 保留 attention sink
    compressed_K = [K[:, :S, :]]
    compressed_V = [V[:, :S, :]]
    remaining = K[:, S:, :]  # length = n - S
    
    # Step 2: 若剩余长度 < 2L，不压缩
    if remaining.shape[1] < 2*L:
        return cat(compressed_K + [remaining]), cat(compressed_V + [remaining])
    
    # Step 3: 按 L 分区
    n_partitions = remaining.shape[1] // L
    K_parts = split(remaining, n_partitions, dim=1)
    V_parts = split(V[:, S:, :], n_partitions, dim=1)
    
    # Step 4: 递归压缩 (最后一个分区 = 滑动窗口，保留不压缩)
    for p in range(n_partitions - 1):
        K_cur, K_ref = K_parts[p], K_parts[p+1]  # K_ref 是 "lag chunk"
        V_cur, V_ref = V_parts[p], V_parts[p+1]
        
        # 4a. 参考 chunk 的 token-wise min/max (沿 seq 维度)
        min_K = K_ref.min(dim=1)  # (heads, head_dim)
        max_K = K_ref.max(dim=1)  # (heads, head_dim)
        min_V = V_ref.min(dim=1)
        max_V = V_ref.max(dim=1)
        
        # 4b. Max-min 归一化
        K_norm = (K_cur - min_K.unsqueeze(1)) / (max_K - min_K).unsqueeze(1)
        V_norm = (V_cur - min_V.unsqueeze(1)) / (max_V - min_V).unsqueeze(1)
        
        # 4c. Channel-wise 标准差 + Softmax
        score_K = softmax(K_norm.std(dim=-1), dim=1)  # (heads, L)
        score_V = softmax(V_norm.std(dim=-1), dim=1)  # (heads, L)
        
        # 4d. 求和得到最终 token score
        score = score_K + score_V  # (heads, L)
        
        # 4e. Top-K 选择
        k = int(r * L)
        keep_idx = topk(score, k, dim=1)
        compressed_K.append(gather(K_cur, keep_idx, dim=1))
        compressed_V.append(gather(V_cur, keep_idx, dim=1))
    
    # Step 5: 加上滑动窗口
    compressed_K.append(K_parts[-1])
    compressed_V.append(V_parts[-1])
    
    return cat(compressed_K, dim=1), cat(compressed_V, dim=1)

# === 压缩比计算 ===
# L_R = S + r*L*(floor((L_s-S)/L) - 1) + L + Mod(L_s-S, L)
# C = 1 - L_R/L_s
```

**关键数学公式**：
$$
\min_i^{p,Z} = \min_{\text{seq}}(Z_i^{p+1}), \quad \max_i^{p,Z} = \max_{\text{seq}}(Z_i^{p+1})
$$

$$
\bar{Z}_i^p = \frac{Z_i^p - \min_i^{p,Z}}{\max_i^{p,Z} - \min_i^{p,Z}}, \quad \text{score}(Z_i) = \operatorname{Softmax}(\operatorname{Std}(\bar{Z}_i))
$$

$$
\text{score}_i = \text{score}(K_i) + \text{score}(V_i)
$$

**Chunk-by-Chunk Prefill 变体**：将 prefill 也拆分为 chunk-by-chunk，每个 L-token chunk prefilled 后进行压缩。这使 hidden states 受压缩影响，但消除了 prefill 一次性全部 forward 的需求。实验显示 FGT 准确率从 100% 降至 ~80%（r=8×），但对序列长度和 needle depth 无强依赖。

术语一般如何实现？如何使用？

集成于 NVIDIA KVPress 框架 (https://github.com/NVIDIA/kvpress)，通过 `KVPressTextGenerationPipeline` 包装 HuggingFace model，在 `generate()` 过程中 hook `past_key_values` 应用压缩。与 FlashAttention 完全兼容——不依赖 attention weight 矩阵。在 Llama-3.1-8B-Instruct 和 Qwen2.5-7B-Instruct 上验证，RULER 16K 上超越 SnapKV 和 StreamingLLM 所有压缩比。64-digit passkey retrieval（L=1024, r=4×）exact match 89%（H2O 仅 35%）。代码开源：https://github.com/AI-Lab-China-Merchants-Bank/LagKV。

涉及论文标题：
- LagKV: Lag-Relative Information of the KV Cache Tells Which Tokens Are Important
