## Attention Bias Problem in KV Cache Eviction (KV缓存淘汰中的注意力偏差)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Attention Bias Problem 是 NACL 揭示的 KV Cache 淘汰策略中的系统性偏差。表现为 attention scores 高度集中在 initial tokens 和 recent tokens，对中间 token attention 显著偏低（Fig. 2）。随序列增长，attention 分布趋于扁平化（Fig. 2d），基于 attention 的评分更难区分关键 token。导致 H2O（全量累加）和 MSRNN（当前 token）均在 long-context 中误淘汰中间的 task-critical token（如 passkey）。

从算法pipeline角度拆解术语：

NACL 的两种对抗机制：(1) proxy tokens 的 attention 天然更均衡——proxy tokens（用户问题）在语义上与 prefix 关键信息相关，非仅位置相关；(2) RANDOM EVICTION 的 head-wise 采样为中间 token 提供额外保留机会。

术语一般如何实现？如何使用？

检测：可视化不同 query position 的 attention score 分布 heatmap。对抗方案：proxy tokens 替代全量累加（NACL）、随机采样补充（NACL）、衰减因子降低位置优势（A2SF forgetting factor）。

涉及论文标题：
- NACL: A General and Effective KV Cache Eviction Framework for LLMs at Inference Time
- A2SF: Accumulative Attention Scoring with Forgetting Factor for Token Pruning in Transformer Decoder
