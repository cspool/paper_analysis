## KV Cache Budget Dynamism

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
KV Cache Budget Dynamism 是指 sparse attention 中 KV cache budget（被选中参与计算的 token 数量 B）在运行时动态变化的现象。Twilight 识别了四个维度的 dynamism：(1) Prompt-wise：不同 task 的 attention distribution 不同；(2) Query-wise：同一 prompt 内不同 query 的 attention 分布不同；(3) Layer-wise：浅层和深层的最优 budget 不同；(4) Head-wise：retrieval heads（关注全局信息，diffuse → 需大 B）vs streaming heads（关注局部信息，focused → 需小 B），budget 需求完全不同。

从算法pipeline角度拆解术语：
根本原因：attention weight 分布在运行时呈现两种极端——focused attention（权重集中在少数 token）和 diffuse attention（权重接近均匀分布）。Top-k 用固定 B 无法同时覆盖两者：对 focused 造成 over-selection（浪费带宽），对 diffuse 造成 under-selection（丢失 context）。Top-p 用累积概率阈值 p 自适应 B 的大小——focused 分布自动选少量 token，diffuse 分布自动选更多 token——天然支持四维 dynamism。

术语一般如何实现？如何使用？
Twilight 通过 head-wise varlen attention（不同 head 不同 B）+ GQA group union（同一 query group 取各 head 选 token 的并集）+ flatten head dim + load balancing 处理不平衡。Longchat-7B 平均 budget 126-146 tokens（32k context, 99.6% pruning），LLaMA-3.1 平均 427-478 tokens（128k context, 99.6% pruning）。

涉及论文标题：
- Twilight: Adaptive Attention Sparsity with Hierarchical Top-p Pruning

---
