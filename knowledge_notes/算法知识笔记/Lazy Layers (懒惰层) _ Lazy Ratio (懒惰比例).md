## Lazy Layers (懒惰层) / Lazy Ratio (懒惰比例)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Lazy Layers 是 LightTransfer 论文提出的概念：在长上下文 LLM 推理中，某些 Transformer 层的注意力主要集中在两类"语义不重要"的 token 上——(1) 初始几个 token（$X_{\text{initial}}$，即 attention sink），(2) 最近的 token（$X_{\text{recent}}$，即 sliding window 内的 token）。这种注意力模式被类比为"读论文只读摘要和结论"，称为"懒惰行为"（lazy behavior）。表现出这种行为的层称为 Lazy Layers。

懒惰比例（Lazy Ratio）$r_i$ 是量化第 i 层懒惰程度的指标：
$$r_i = \frac{1}{w_{\text{last}}} \sum_{\hat{x} \in X_{\text{last}}} \sum_{x \in \{X_{\text{initial}}, X_{\text{recent}}\}} A_i(\hat{x}, x)$$

其中 $A_i(\hat{x}, x)$ 是第 i 层所有 head 平均后的注意力权重，从 query token $\hat{x}$ 到 key token $x$。$w_{\text{last}}$ 是用于评估的最后几个 query token 数量。$r_i$ 越高，说明该层越多注意力集中在 sink + recent token 上。

关键发现：(1) 对于给定的输入 prompt，懒惰层行为在生成过程中跨 token 相对一致；(2) 不同 prompt 下懒惰层的 index 位置可能不同，因此需要 test-time 动态识别。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**Lazy Ratio 计算（利用 FlashAttention LSE 避免重算完整 attention 矩阵）**：
```python
def lazy_ratio_calculation(q, k, v, w_last, w_sink, w_recent):
    # q, k, v: [bs, num_heads, seq_len, head_dim]
    attn_out, lse = flash_attn(q, k, v, causal=True, return_lse=True)
    # lse: [bs, num_heads, seq_len] -- log-sum-exp of attention scores

    q_last = q[:, :, -w_last:, :]   # 最后 w_last 个 query token
    k_comb = torch.cat([k[:, :, :w_sink, :],    # sink tokens
                         k[:, :, -w_recent:, :]], dim=2)  # recent tokens

    # O(w_last * (w_sink+w_recent)) 小矩阵乘法，常数复杂度
    log_lazy_ratio = torch.matmul(q_last, k_comb.transpose(-1, -2)).logsumexp(dim=-1) - lse
    return log_lazy_ratio  # 高值 → layer 懒惰
```

**LightTransfer-TEST 流程（Prefilling 阶段动态识别）**：
```
优先队列 Q (max-heap, 容量 P = 50% 总层数)

for layer i in 0..L-1:
    计算当前层的 full attention 并获取 KV cache
    计算 lazy ratio r_i
    将 (r_i, i) 加入 Q
    
    if Q 容量 > P:
        (r_max, lazy_layer) = Q.pop()  # 弹出 ratio 最高的层
        将 lazy_layer 的 KV cache 缩减为 {X[:w_sink], X[-w_recent:]}
```

术语一般如何实现？如何使用？

通过 FlashAttention 的 `return_lse=True` 参数获取 LSE 值作为注意力分布代理，避免 $O(n^2)$ 重计算。推荐超参数：$w_{\text{sink}}=4$, $w_{\text{recent}}=1020$, $w_{\text{last}}=32$。Lazy ratio 计算的额外开销极小（相对吞吐仅降低 0.0014-0.0058×），且序列越长开销占比越低（识别复杂度 O(1)）。LightTransfer-TEST 适用于输入足够长的任务（long-context understanding），LightTransfer-TRAIN 通过训练集预选懒惰层后 SFT 微调，适用于输入短但推理链长的任务（o1-like reasoning）。

涉及论文标题：
- LightTransfer: Your Long-Context LLM is Secretly a Hybrid Model with Effortless Adaptation
