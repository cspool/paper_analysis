## Semantic Retrieval Head (语义检索注意力头)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Semantic Retrieval Head 是 CompressKV（Lin et al., 2025）提出的注意力头分类概念，是对传统 Retrieval Head 的扩展。传统 Retrieval Head 识别标准要求 head 的 top-1 attention 精确落在正确答案 token 上（仅捕捉 copy-paste 行为，如 Wu et al., ICLR 2025 的定义）。Semantic Retrieval Head 的识别标准不要求精确 top-k 命中——而是聚合 head 在整个 answer span A 上的 attention scores 来评估语义检索能力，公式为：

$$\text{SemanticRetrievalScore}(h) = \sum_{t=1}^{N} \mathbb{I}[y_t \in A] \sum_{j \in A} a_{t,j}^h$$

其中 y_t 是第 t 步生成的 token，A 是 answer span，a_{t,j}^h 是 head h 在 token j 上的 attention weight。得分越高，说明该 head 越能捕捉语义信息（包括 copy-paste 行为和更深的语义依赖），而非仅 copy-paste。

核心 insight：在 long-context 场景下，attention distribution 极其稀疏，大量 attention 分配给 initial/final tokens（attention sink）。传统 top-1/top-k 标准的 hit rate 极低（大部分 head 得分为零），且仅捕捉 copy-paste 模式，忽略语义依赖。例如生成 "sandwich" 时，模型不仅 attend "sandwich"，还 attend 周围语义相关 token（如 "eat", "a thing"）——传统标准不认可这些 head，但 Semantic Retrieval Head 标准能捕获。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**SRH 识别与 Token 选择过程**：

```
// === 离线阶段：SRH 识别 ===
// 在验证集（如 LongBench）上运行完整模型
for each layer l in 0..L-1:
    for each head h in 0..H-1:
        SRScore[l][h] = 0
        for each decoding step t where y_t in answer_span A:
            a_t = attention_weights[l][h][t]  // [seq_len]
            SRScore[l][h] += sum(a_t[j] for j in A)
    // L1 normalize within layer
    SRScore[l] = SRScore[l] / sum(SRScore[l])

// 取每层 top-k SRH（默认 k=4）
topk_SRH[l] = argsort(SRScore[l])[-k:]

// === 在线 Prefill 阶段：SRH 驱动的 Token 选择 ===
// observation window W = 8, pooling kernel size = 5
for each layer l:
    selected_heads = topk_SRH[l]
    S = zeros(seq_len)
    for h in selected_heads:
        A_h = attention_scores[l][h][:, -W:]  // [seq_len, W]
        S_h = sum(A_h, dim=-1)                 // [seq_len]
        S_h = avg_pool1d(S_h, kernel_size=5)  // [seq_len]
        S += S_h
    S = S / len(selected_heads)  // average
    keep_indices = topk(S, N)    // select top-N tokens
    K_cache = K[keep_indices]
    V_cache = V[keep_indices]
```

术语一般如何实现？如何使用？

SRH 识别离线完成，在 LongBench 或类似验证集上运行完整模型一次即可。对于 Llama-3.1-8B-Instruct（32 层），每层仅需 4 个 SRH 即可达到最佳性能——增加至 6/12/24 个 head 无进一步收益。在 Mistral-7B 和 Llama-3.1-8B 上，SRH 识别结果与传统 Retrieval Head 显著不同：传统方法中 layer 0 和 1 的所有 head 得分为零，而 SRH 方法能识别出低但有效的语义重要性 head。Masking top-30 SRH 导致 NIAH 准确率下降 ~74%（vs 传统 Retrieval Head 仅下降 ~12%）。代码开源：https://github.com/TUDa-HWAI/CompressKV.git。

涉及论文标题：
- CompressKV: Semantic Retrieval Heads Know What Tokens are Not Important Before Generation

---
