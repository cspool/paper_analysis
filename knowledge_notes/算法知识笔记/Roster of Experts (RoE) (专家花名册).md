## Roster of Experts (RoE) (专家花名册)

术语解释
一种无需训练的 MoE 推理算法，将单个 MoE 模型视作一个动态专家集成（dynamic ensemble of MoEs），通过 Gumbel-Top-K 随机路由在每 token 生成多条内部计算路径，聚合多路径输出以提升预测质量。RoE 是 Hyper-Parallel Scaling 概念在 MoE 模型上的具体实例化。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
RoE 由三个核心组件构成：

1. **Gumbel-Top-K 随机路由**：向 router logits 添加温度控制 Gumbel 噪声 → 从 router 隐含分布中无放回采样 k 个 expert → 产生 expert 组合多样性。
2. **多路径聚合**：对同一 token 执行 n 次独立 Gumbel-Top-K forward → 得到 n 组 logits → softmax 后概率平均（probability averaging）→ argmax 得最终 token。
3. **Clean Cache**：batch 中 sample 0 使用 deterministic routing (τ=0) → 其 KV-cache 作为共享"clean"缓存 → 其余 sample 复用此 cache → KV-cache 内存 = 单样本内存。

RoE 的关键性质：
- **Training-free**：不修改模型参数，直接应用于任何预训练 MoE 模型
- **Post-hoc**：可在部署后动态启用/禁用，按需 trade compute for quality
- **Orthogonal**：与 sequential/parallel scaling 可同时使用（论文仅用 greedy decoding 评估以隔离 RoE 收益）

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
RoE 完整算法（基于论文 Algorithm 和描述）：

```
# RoE 单步 Token 生成
输入: 当前 token embedding e, 逐层温度数组 τ[l], 样本数 n, 共享 KV-cache K
输出: 下一 token

def roe_generate_step(e, τ, n, K, model):
    # 准备 batched input
    batch_e = e.expand(n, -1)            # (n, d_model)

    for l, layer in enumerate(model.layers):
        # Attention: sample 0 计算，其余共享 KV-cache
        if is_clean_cache_enabled:
            attn_out_0, K_new = layer.attn(batch_e[0:1], K)
            K = K_new                       # 更新共享 KV-cache
            attn_out_i = layer.attn_shared_kv(batch_e[1:], K)  # 复用 K
            attn_out = cat([attn_out_0, attn_out_i], dim=0)
        else:
            attn_out = layer.attn(batch_e)

        # MoE Layer: per-sample Gumbel-Top-K
        if layer.is_moe:
            R = layer.router(attn_out)     # (n, E) router logits
            for i in range(n):
                τ_eff = 0.0 if (i == 0 and is_clean_cache) else τ[l]
                G = sample_gumbel(E)
                noisy_R = R[i] + τ_eff * G
                topk_idx, topk_w = topk(softmax(noisy_R), k)
                ff_out = sum(w * layer.experts[idx](attn_out[i])
                           for idx, w in zip(topk_idx, topk_w))
                batch_e[i] = attn_out[i] + ff_out  # residual
        else:
            batch_e = layer.ffn(attn_out) + attn_out

    # Logit 聚合: probability averaging
    logits = model.lm_head(batch_e)        # (n, vocab_size)
    probs = softmax(logits, dim=-1)        # (n, vocab_size)
    avg_probs = probs.mean(dim=0)          # (vocab_size,)
    next_token = argmax(avg_probs)

    return next_token, K
```

超参数配置（来自论文 Table 1）：
- OLMoE-7B 数学任务：n=32, τ_max=0.5, skip=1 首尾层
- Mixtral-8x7B 数学任务：n=64, τ_max=0.25, skip=5 首尾层
- 温度通过 Optuna TPE 在验证集上逐层搜索

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
使用 RoE 的典型流程：
1. **温度调优**（离线）：在目标任务验证集上用 Optuna TPE 搜索每层最优 τ_l（~50 trials）
2. **推理部署**（在线）：加载调优后的 τ 配置 → 对每个生成 token 执行 batched Gumbel-Top-K forward → Clean Cache 控制内存
3. **计算-质量 trade-off**：通过调整 n（样本数）控制——n 越大质量越高但计算开销越大（论文显示 n=32 已有显著收益）

论文关键结果：OLMoE-7B + RoE (n=32) ≈ 10.5B 标准 MoE 的性能，内存减少 25%，延迟降低 30%。

涉及论文标题：
- MoEs Are Stronger than You Think: Hyper-Parallel Inference Scaling with RoE
