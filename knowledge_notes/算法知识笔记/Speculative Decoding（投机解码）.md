## Speculative Decoding（投机解码）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Speculative Decoding 是一种加速 LLM 自回归推理的算法，不改变输出分布，保证与原始大模型完全一致的生成结果。其核心流程：(1) 使用一个计算成本低的小模型（draft model $M_q$）自回归生成 $\gamma$ 个候选 token；(2) 将 prefix $\sigma$ 与 $\gamma$ 个候选 token 拼接，送入大模型 $M_p$ 进行一次 forward pass；(3) 对比 $M_q$ 和 $M_p$ 在每个 token 位置的 logits，按某种准则（通常为 rejection sampling）接受或拒绝候选 token；(4) 若某个 token 被拒绝，从该位置起用 $M_p$ 重新采样。整个过程仅需一次大模型 forward pass 即可验证 $\gamma$ 个 token，而标准的自回归解码需要 $\gamma$ 次 forward pass。若接受率 $\alpha$ 高，则实际加速比接近 $\gamma$·$\alpha$。典型实现可达到 2-3× 加速（对 T5X/Chinchilla 70B）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Speculative Decoding Algorithm
def speculative_decode(prefix, M_q, M_p, gamma):
    # Stage 1: Draft phase
    draft_tokens = []
    current_prefix = prefix
    for i in range(gamma):
        token = M_q.autoregressive_step(current_prefix)
        draft_tokens.append(token)
        current_prefix = current_prefix + [token]
    
    # Stage 2: Verify phase (single forward pass)
    full_seq = prefix + draft_tokens
    logits_p = M_p.forward(full_seq)     # [len(full_seq), vocab_size]
    logits_q = M_q.forward(full_seq)     # [len(full_seq), vocab_size]
    
    # Stage 3: Accept/Reject
    accepted = []
    for i in range(gamma):
        pos = len(prefix) + i
        p_dist = softmax(logits_p[pos])
        q_dist = softmax(logits_q[pos])
        # Rejection sampling
        if random() < min(1, p_dist[draft_tokens[i]] / q_dist[draft_tokens[i]]):
            accepted.append(draft_tokens[i])
        else:
            # Rejected: resample from adjusted distribution
            adjusted = max(0, p_dist - q_dist)
            adjusted = adjusted / sum(adjusted)
            bonus_token = sample(adjusted)
            accepted.append(bonus_token)
            break
    return prefix + accepted
```

加速比分析：若大模型单步推理时间为 $T_p$，小模型为 $T_q$（$T_q \ll T_p$），接受率为 $\alpha$，则每轮期望生成 token 数为 $\frac{1-\alpha^{\gamma+1}}{1-\alpha}$，理论加速比为 $\frac{1-\alpha^{\gamma+1}}{(1-\alpha)(\gamma T_q + T_p)}$。

典型变体：Medusa（无需辅助小模型，通过预训练多个预测头同时预测多个 token）、Draft & Verify（跳过中间层替代独立小模型）、SpecTr（扩展候选 token 数量）、SpecInfer（云端多 draft model）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

开源实现广泛集成于主流框架：TensorRT-LLM 支持 speculative decoding、HuggingFace TGI 提供原生支持、vLLM 通过 draft model API 支持。使用时需准备一个与 target model 同 vocab 的小 draft model（如 LLaMA-68M 搭配 LLaMA-7B）。SpecExec 进一步将 speculative decoding 应用于消费级设备，通过将大模型参数 offload 到 RAM/SSD，在 4-bit 量化下运行 50B+ 模型达 4-6 tok/s。Apple 的 Speculative Streaming 将 drafting 融入 target model 本身（修改微调目标从 next-token prediction 到 future n-gram prediction），消除对独立 draft model 的需求。

涉及论文标题：
- A Survey of Resource-efficient LLM and Multimodal Foundation Models
