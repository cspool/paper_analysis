## CDAC (Communicate-Descend-Ascend-Communicate)

术语解释
CDAC 是传统 fine-grained MoE（如 DeepSeekMoE）的默认执行顺序：先进行高维 All-to-All 通信，再在 expert 内部进行降维-升维投影。BigMac 论文通过分析 CDAC 的通信瓶颈，提出了 DCCA 作为替代。

术语是什么？
在 CDAC 方式下，各 expert 内部的 FFN 计算为 $E_i(x) = \sigma(xW_{i,\downarrow})W_{i,\uparrow}$（先降维后升维）。由于 All-to-All 在 expert 计算之前/之后进行，通信始终在 token 的 full hidden dimension h 上进行。

通信量：$C = 2 \times top\_k \times \frac{ep-1}{ep} \times b \times s \times h$，与 h 成正比。对于 large hidden dimension（如 DeepSeek-V2 的 5120），通信开销极大。

从算法pipeline角度拆解术语：

```
# CDAC MoE layer forward (Fine-Grained MoE baseline)
def cdac_moe_forward(x):              # x: [batch, seq, h]
    gate = SoftMax(x @ W_gate)
    topk_w, topk_idx = TopK(gate, k=top_k)

    # Step 1: All-to-All DISPATCH (HIGH dim: h, 通信瓶颈!)
    dispatched = alltoall_scatter(x, topk_idx)

    # Step 2: Expert computation (内部先降后升)
    for each expert i:
        h_down = dispatched @ W_i_down     # h → h_ff (DESCEND)
        h_act = σ(h_down)                  # activation
        h_out = h_act @ W_i_up             # h_ff → h (ASCEND)
        output += topk_w[i] * h_out

    # Step 3: All-to-All COMBINE (HIGH dim: h, 通信瓶颈!)
    y = alltoall_gather(output)
    return y
```

CDAC 的核心缺陷：All-to-All 始终在最高维度 h 上进行 → 通信量巨大 → 尤在 top_k 大时成为主导延迟（高达 91.8%）。

术语一般如何实现？如何使用？
- 是 DeepSeekMoE、Qwen2-MoE 等 fine-grained MoE 模型的默认结构
- 需要 expert parallelism 支持（All-to-All dispatch/combine）
- 在 small model 或无 EP 时通信不是瓶颈，CDAC 无劣势

涉及论文标题：
- BigMac A Communication-Efficient Mixture-of-Experts Model Structure for Fast Training and Inference
