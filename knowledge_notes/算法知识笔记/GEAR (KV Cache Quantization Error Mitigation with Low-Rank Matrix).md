## GEAR (KV Cache Quantization Error Mitigation with Low-Rank Matrix)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

GEAR（Kang et al., 2024）是一种针对 KV cache 量化的误差修正算法。标准量化（如 INT4）将 FP16 KV cache 压缩为低精度表示，但会引入量化误差（即 $\hat{X} - X$，其中 $\hat{X}$ 是反量化后的值，$X$ 是原始值）。当量化误差集中在 outlier 位置时，会严重损害 LLM 输出质量。GEAR 通过两种机制共同近似和补偿量化误差：(1) **low-rank matrix** 用低秩分解近似整体量化误差的主体部分（低秩矩阵用 $UV^T$ 表示，其中 $U \in \mathbb{R}^{d \times r}$, $V \in \mathbb{R}^{r \times L}$，rank $r$ 控制近似精度）；(2) **sparse matrix** 保留少数全精度 outlier 值（sparsity ratio $s$ 控制保留比例），以处理量化误差中极端值集中的部分。两者叠加形成对量化误差的近似 $\tilde{E} \approx UV^T + S$，在推理时加回反量化结果以恢复精度。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**GEAR 在 KV cache 量化推理中的流程**：
```
# Prefill 阶段：正常计算并存储 FP16 KV cache
K_cache_fp16 = X @ W_K
V_cache_fp16 = X @ W_V

# Decoding 每步的 GEAR 量化+误差修正流程：
for each decode step:
    # 1. 量化 KV cache
    K_int4 = quantize(K_cache_fp16, group_size=G)
    V_int4 = quantize(V_cache_fp16, group_size=G)

    # 2. 反量化得到近似值
    K_approx = dequantize(K_int4)
    V_approx = dequantize(V_int4)

    # 3. 计算量化误差（对过去 window 内的 token）
    E_K = K_cache_fp16[recent_window] - K_approx[recent_window]
    # 不计算全量——仅对近期 window 做误差修正

    # 4. Low-rank 近似：对误差矩阵做 SVD，保留 top-r 奇异值
    U_K, Sigma_K, Vt_K = svd(E_K)
    U_K = U_K[:, :r] @ diag(sqrt(Sigma_K[:r]))
    V_K = Vt_K[:r, :]

    # 5. Sparse 保留：选 |E_K| 最大的 s% 位置保留全精度值
    outlier_indices = topk(abs(E_K), ratio=s)
    S_K = zeros_like(E_K)
    S_K[outlier_indices] = E_K[outlier_indices]

    # 6. Attention 计算时恢复：
    K_recovered = dequantize(K_int4) + (U_K @ V_K) + S_K

    # 7. 与 Q 做 Attention
    scores = Q @ K_recovered^T / sqrt(d_head)
    output = softmax(scores) @ V_recovered
```

**Annotations**: `G` = group_size（量化粒度），`r` = low-rank rank（典型 2%），`s` = sparse ratio（典型 2%）。Low-rank matrix 存储开销 = $r \times (d+L)$ 个 FP16 值，Sparse matrix 仅存储 outlier 位置和值。GEAR 的额外计算开销来自 SVD（一次性的误差分解）和 low-rank/sparse 矩阵的加法恢复，在 prefill 阶段会降低吞吐（论文 Table 3 显示 GEAR prefill 仅有 baseline 的 0.80-0.90×）。

术语一般如何实现？如何使用？

GEAR 开源实现：https://github.com/opengear-project/GEAR。关键参数：sparsity ratio $s$（默认 2%，控制保留全精度的 outlier 数量）和 rank $r$（默认 2%，控制 low-rank 近似矩阵的秩/精度）。GEAR 通过 $s$ 和 $r$ 控制误差修正的精度—内存tradeoff。论文 "Rethinking KV Cache Compression" 的评估显示 GEAR 在 prefill 阶段有显著吞吐下降（因额外 SVD + low-rank 计算开销），在 decode 阶段低 batch size/短 KV length 下可能与 FP16 baseline 持平，但在大 batch size/长 KV length 下吞吐收益有限。GEAR 在 LMDeploy v6.0.1 上的吞吐评估显示：LLaMA-7B TP=1 prefill 仅 0.86× FP16 baseline，decode 1.02×。

涉及论文标题：
- Rethinking Key-Value Cache Compression Techniques for Large Language Model Serving
