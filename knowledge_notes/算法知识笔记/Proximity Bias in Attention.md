## Proximity Bias in Attention

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Proximity Bias 是 LLM 自注意力中的偏差现象：在生成新 token 时，prompt 中位置更靠近末尾（更接近当前解码 token）的 token 倾向于获得更高注意力分数，即使这些 token 在语义上不如某些远端 token 重要。与 Attention Sink（首 token 偏爱）共同构成使用原始注意力分数估计 token 重要性的两个主要偏差源。

从算法pipeline角度拆解术语：

在 token dropping 场景中，仅依赖最后 token 的注意力会导致偏向选择末尾 token 而忽略前段重要信息。SPECPREFILL 通过两种策略缓解：
(1) Look-ahead decoding — 解码 N 步后聚合多个位置的注意力，削弱单一位置偏差
(2) Max-mean aggregation — max over layers/heads 让被任意层关注的 token 浮现，mean over look-ahead steps 公平对待各步

术语一般如何实现？如何使用？

除 look-ahead 和聚合策略外，邻近 token 重要性相关的观察（Concurrent work CritiPrefill/Lv et al. 也发现此现象）促使 chunk-based selection 作为额外去噪手段。SPECPREFILL 的消融实验（Figure 2, 8）验证这些策略对短上下文任务的提升比长上下文更显著。

涉及论文标题：
- Speculative Prefill: Turbocharging TTFT with Lightweight and Training-Free Token Importance Estimation

---
