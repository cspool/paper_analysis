## Sublayer-wise Skipping (子层粒度跳过)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Sublayer-wise Skipping 是 AdaSkip 提出的核心创新——在 Transformer 推理中按 sublayer 粒度（独立评估 Attention sublayer 和 FFN sublayer）而非整层粒度进行选择性跳过。其关键观察（Observation 2）：Attention 和 FFN sublayer 的 IO Similarity 分布独立且特征不同——在长上下文推理中，Attention sublayer 的平均 IO Similarity 更高且更集中（如 LLaMA3.1-8B-128k 最后 11 层 attention Similarity 稳定在 ~0.97，FFN 仅 ~0.95），意味着更多 attention 可被跳过，且跳过 attention 还能节省 KV cache。

整层跳过每次 skip 同时省略 2 个 sublayer（attention + FFN），而 sublayer-wise skipping 每次 skip 1 个 sublayer。由于 IO Similarity 分布不同，sublayer-wise 有更多加速机会——尤其在长上下文场景下 attention 的 O(n²) 开销远大于 FFN 的 O(n)，优先跳过更多 attention 能获得更大加速。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
// 整层跳过（baseline）：每次 skip 同时跳过 attention + FFN
for layer l in 1..L:
    if l in layer_skip_set:
        x = x  // identity
    else:
        x = Attention(LayerNorm(x)) + x
        x = FFN(LayerNorm(x)) + x

// Sublayer-wise 跳过（AdaSkip）：独立判断 attention 和 FFN
for layer l in 1..L:
    if attn_idx[l] in skipped_set:
        x = x  // skip attention
    else:
        x = Attention(LayerNorm(x)) + x
    if ffn_idx[l] in skipped_set:
        x = x  // skip FFN
    else:
        x = FFN(LayerNorm(x)) + x
```

**Sublayer 排序与选择**（AdaSkip 核心）：
```
all_sublayers = [(Simi_attn[1], 'attn', 1), ..., (Simi_ffn[M], 'ffn', M)]
sorted = sort(all_sublayers, by=Simi, descending=True)
skipped = sorted[0:2m]  // 2M 个子层统一按 Similarity 排序，取前 2m 个
```

术语一般如何实现？如何使用？

需要对 HuggingFace Transformer 做细粒度 hook：每个 attention 和 FFN 子模块分配唯一的 sublayer index（index ∈ [0, 2M-1]），分别捕获输入/输出 hidden states 计算各自的 IO Similarity，并在 forward 中为 attention 和 FFN 分别在入口处插入 skip 判断条件。Skip 时用 Scale_j 补偿。

AdaSkip 开源：https://github.com/ASISys/AdaSkip

涉及论文标题：
- AdaSkip: Adaptive Sublayer Skipping for Accelerating Long-Context LLM Inference
