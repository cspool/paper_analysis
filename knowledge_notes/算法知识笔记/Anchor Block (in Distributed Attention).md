## Anchor Block (in Distributed Attention)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Anchor Block 是分布式近似注意力（STARATTN/APB）中的一个关键设计：在每 host 的 local context block 前 prepend 一段包含输入序列开始部分 token 的"锚点块"，使每个 host 的注意力计算能感知文档起始位置的上下文。其设计动机源于 Attention Sink 现象——模型需要起始 token 作为注意力质量的"倾泻槽"，否则每个独立 block 会自行产生 attention sink 导致注意力分布失真。

STARATTN 使用 l_a = l_b 的大 anchor block（anchor 与 local block 等长），APB 将其减小到 l_a = l_b/4 或 l_b/8。APB 还在 anchor block 中嵌入 query token，使 retaining heads 感知查询相关信息。

从算法pipeline角度拆解术语。

**Anchor Block 在 APB 中的作用**：

```
// 构造
A = {q_1, ..., q_{l_q}, d_1, ..., d_{l_a}}  // query + 文档开头 l_a 个 token
position_ids(A) = 0, 1, ..., l_q+l_a-1      // 从 0 开始的位置编码

// 每 host 的 context layout (host h > 1)
context = [A, P_h, B_h]  // anchor → passing → local

// Attention 计算
Q = [Q_a, Q_h]
K = [K_a, K_p^C, K_h]    // anchor KV + compressed passing KV + local KV
V = [V_a, V_p^C, V_h]
// anchor block 的 KV 参与所有 token 的 attention 计算

// Anchor block hidden states 通过 FFN
H_a^out = FFN(A_a)       // anchor 的输出被保留
```

**STARATTN vs APB anchor block 差异**：
| 维度 | STARATTN | APB |
|------|----------|-----|
| Anchor 长度 l_a | l_b (与 local block 等大) | l_b/4 或 l_b/8 |
| Anchor 内容 | 文档开头 l_a token | query + 文档开头 l_a token |
| FFN 开销 | 大（l_a = 16K → 大量重复计算） | 小（l_a = 4K） |
| 是否嵌入 query | 否 | 是 |

消融实验（Table 3）证明 anchor block 是最关键组件——移除后 E.MC 从 72 降至 28。

术语一般如何实现？如何使用？

Anchor block 在 tokenization 和 embedding 阶段构造——将 query token 和文档开头 token 拼接后分配连续 position IDs，然后以 prepend 方式与 local context block 合并送入 Transformer。实现无额外复杂度，仅需在输入预处理阶段调整 token 排列。APB 开源：https://github.com/thunlp/APB。

**Star Attention 的 Anchor Block 设计差异**：与 APB 相比，Star Attention 使用 l_a = l_b 的大 anchor block（anchor 与 local block 等长），内容仅为文档开头的 l_a 个 token（不含 query），位置编码保持原始位置。STARATTN 的消融实验（Table 4）证明：(1) anchor block 的内容至关重要——使用常量 token（空格/标点）时准确率降为 0%，使用随机 token 时降 10%；(2) anchor block 的位置编码影响较小——即使随机化位置 ID，准确率仅降约 2%。此外，Star Attention 的 anchor block KV 在阶段一后被丢弃（仅保留 context block 部分的 KV），而 APB 保留了 anchor block 的 hidden states 通过 FFN。

涉及论文标题：
- APB: Accelerating Distributed Long-Context Inference by Passing Compressed Context Blocks across GPUs
- Star_Attention__Efficient_LLM_Inference_over_Long_Sequences

---
