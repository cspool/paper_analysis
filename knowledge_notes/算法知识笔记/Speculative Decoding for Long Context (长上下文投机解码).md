## Speculative Decoding for Long Context (长上下文投机解码)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Speculative Decoding（SD，投机解码）是一种 lossless 的 LLM 解码加速算法：使用轻量 draft model 快速生成 γ 个候选 token，target model 通过一次并行 forward pass 验证所有候选，通过 greedy matching 或概率接受保证输出与 target model 原生 AR 解码完全一致。标准 SD 的加速比公式为：

$$\frac{T_{Avg}^{SD}}{T_T} = \frac{1}{\Omega(\gamma, \alpha)} \left( \frac{\gamma \cdot T_D}{T_T} + \frac{T_V(\gamma)}{T_T} \right)$$

其中 $\Omega(\gamma,\alpha) = \frac{1 - \alpha^{\gamma + 1}}{1 - \alpha}$ 为每步验证的期望生成 token 数，$\alpha \in [0,1]$ 为 draft token 接受率，$T_D, T_T, T_V$ 分别为 draft、target 解码、验证时间。

MagicDec 的核心贡献是打破了 "SD 仅对小 batch 有效" 的传统认知。通过在长上下文 + 大 batch 场景下识别 KV cache 瓶颈转移（Section 3.2），MagicDec 证明了当 $S \ge S_{\text{inflection}}$ 时 SD 对大 batch 仍然有效甚至 speedup 随 batch 增大而提升。关键机制：(1) 长序列下 KV cache loading 成为主导瓶颈（memory-bound），验证与解码共享 KV budget → $T_V/T_T \approx 1$；(2) 使用压缩 KV 的 self-speculation 使 $T_D/T_T \to 0$；(3) KV 压缩获得更高接受率（>90%）→ 降低 costly verification 次数。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
# MagicDec 长上下文 SD 的 decode 循环
while not all_done:
    # Phase 1: Draft（使用压缩 KV cache，生成 γ 个候选 token）
    for i in 1..γ:
        q_new = W_q @ embed(last_token)           # query
        s = q_new @ K_draft^T / sqrt(d_head)       # 仅对 K_draft (size K<<S) 计算
        a = Softmax(s)
        o = a @ V_draft
        next_token = LMHead(FFN(o))
        draft_tokens.append(next_token)
        # 更新 draft KV（追加新 token）
        update(K_draft, V_draft, next_token)
    
    # Phase 2: Verify（使用完整 KV cache，一次 forward 验证全部候选）
    # 拼接 last_token + draft_tokens 的 γ+1 个查询位置
    q_all = W_q @ embed([last_token] + draft_tokens)
    s_full = q_all @ K_full^T / sqrt(d_head)       # 对完整 KV cache
    logits = LMHead(FFN(Softmax(s_full) @ V_full))
    
    # Phase 3: Greedy matching 确定接受
    accepted = []
    for i, draft_tok in enumerate(draft_tokens):
        if draft_tok == argmax(logits[i]):
            accepted.append(draft_tok)
        else:
            accepted.append(argmax(logits[i]))       # 不匹配则取 target token
            break
    output.extend(accepted)
    # 更新完整 KV cache（追加所有新 token 的 KV）
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

长上下文 SD 的部署要点：draft model 使用 target model 自身 + 压缩 KV（self-speculation），或小模型 + 压缩 KV。KV 压缩可选 static（SnapKV 无搜索开销但接受率上限 ~85-90%）或 dynamic（PQCache/TopK 接受率 >95% 但 batch-size 相关的 search cost T_select）。MagicDec 开源 https://github.com/Infini-AI-Lab/MagicDec 提供完整实现，基于 GPT-Fast + FlashInfer + torch.compile + CUDA graphs。适用场景：长上下文 LLM serving（S > 4000 tokens），batch size 32-256，speedup 1.2x-2.51x。

涉及论文标题：
- MagicDec: Breaking the Latency-Throughput Tradeoff for Long Context Generation with Speculative Decoding
