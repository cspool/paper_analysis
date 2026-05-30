## Attention Sink (注意力汇)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Attention Sink 是 Xiao et al. (2023) 发现的 Transformer 现象：初始 token（尤其 BOS）在不重要的语义下获得不成比例的高 attention（>50% for Llama3.2-3B）。根源：softmax 求和为 1 的约束迫使模型将过剩概率质量倾注到初始 token 作为"注意力垃圾桶"，导致 cross-token attention 预算不足，损害 recall。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Hymba 三管齐下：(1) Meta tokens 作为"更好的垃圾桶"吸收 sink；(2) SWA 限制 attention 范围，BOS 在大多数层不可见；(3) SSM heads 绕过 softmax attention，不受 sink 影响。相比 StreamingLLM（利用 sink 而非消除）、register tokens（仅 ViT）、Quiet Attention（softmax 分母 +1），Hymba 的方案最全面。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
缓解方案对比：(1) StreamingLLM——保留 sink tokens + 最近 tokens；(2) Register tokens——append 可学习 token；(3) Quiet Attention——softmax 分母 +1，等效 all-zero token；(4) SWA——物理隔离。Hymba 同时使用方案 (2)+(3)+(4)。

涉及论文标题：
- Hymba: A Hybrid-head Architecture for Small Language Models
