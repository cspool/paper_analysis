## Prefill Token Dropping for QPS Maximization

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Prefill Token Dropping for QPS Maximization 是 SPECPREFILL 的核心 serving 优化策略：通过丢弃 prompt 中不重要 token 降低 prefill FLOPs → 降低 TTFT → 提升最大 QPS。核心关系：Max QPS ∝ 1/TTFT，TTFT ∝ FLOPs，FLOPs ∝ keep_rate。对 405B 模型 10% 保持率，实测 7× QPS 提升。

从系统架构角度拆解术语：

在 vLLM pipeline 中的流程：
```
1. Request arrival → vLLM scheduler
2. Speculator forward (8B, look-ahead N steps)
3. Attention aggregation → chunk Top-K selection
4. Token extraction + position ID mapping (non-contiguous)
5. Base model prefill (selected tokens only) → KV cache write
6. Decode (autoregressive on full KV)
7. Response return
```
开销：speculator FLOPs/base FLOPs = 14.24%（70B）或 2.96%（405B）。理论 speedup = 1/(keep_rate + FLOPs_ratio)，实测接近上限（7.66× vs 7.72×）。

术语一般如何实现？如何使用？

vLLM monkey patch 实现。适用长上下文、含冗余的 prompts；不适用短 prompts 或信息密集型任务。speculator 可与 speculative decoding 共享 draft model 摊销 overhead（形成 unified small-model-assisted inference pipeline）。

涉及论文标题：
- Speculative Prefill: Turbocharging TTFT with Lightweight and Training-Free Token Importance Estimation
