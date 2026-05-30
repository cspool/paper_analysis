## Global-to-Local Language Modeling

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Global-to-Local Language Modeling 是 Block Transformer 提出的分层语言建模范式。核心思想：将自回归 Transformer 的全局自注意力分解为两个阶段：(1) **Global stage** (Block Decoder)：以粗粒度 block 为单位做全局注意力，将全局上下文压缩为 context embedding；(2) **Local stage** (Token Decoder)：以细粒度 token 为单位仅对当前 block 做局部注意力，依赖 context embedding 获取全局信息。与滑动窗口注意力（SWA）的关键区别：SWA 仅在部分层使用局部注意力，上层依赖堆叠隐式获得全局感受野，prefill 无法跳过。Block Transformer 在**所有 upper layers** 强制局部性，彻底消除 token decoder 的 prefill 和 KV cache。与 MEGABYTE 的关键区别：Block Transformer 首次识别 token decoder（local module）的计算能力对性能同等重要，提出 1:1 参数比和 prefix token 机制，而 MEGABYTE 认为 local module 应尽可能小（6:1 参数比）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
KV cache reduction model (L=2048, LB=4):
- Block decoder: context length = 512, KV cache ↓4×, KV cache IO ↓16× (vs vanilla L=2048)
- Token decoder: context length = 4, KV cache ↓512×, KV cache IO ↓256× (vs vanilla)
- Overall: $O(L^2) \rightarrow O(L \cdot L_B)$, 即线性复杂度

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
通过修改标准 Transformer 架构实现。关键超参数：$L_B$ (block length)、prefix length (token decoder 计算宽度)、parameter allocation ratio ($N_b$:$N_t$ 层数比)。实验显示 $L_B=4$, prefix=2, ratio=1:1 在 perplexity-throughput 上 Pareto-optimal。更大 context length 下收益更大。

涉及论文标题：
- Block Transformer Global-to-Local Language Modeling for Fast Inference (NeurIPS 2024)

---
