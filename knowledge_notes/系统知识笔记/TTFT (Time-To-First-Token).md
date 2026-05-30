## TTFT (Time-To-First-Token)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

LLM 推理服务的核心延迟 SLO 指标，从请求发送到收到第一个 token 的时间，主要由 Prefill 阶段决定。重要性来自两方面：(1) 用户体验——直接影响响应感知；(2) 系统吞吐——最大支持 QPS ∝ 1/TTFT（在有限 timeout 下），TTFT 决定推理引擎能支持的最大并发请求数。

从系统架构角度拆解术语：

在 SPECPREFILL 的端到端 QPS 实验中（vLLM server + OpenAI API client），系统呈现三阶段模式：
```
Stage 1 (constant): QPS 低，请求在收到新请求前完成 → 延迟恒定
Stage 2 (linear):   TTFT 够小但 decode 来不及 → 延迟线性增长
Stage 3 (timeout):  连 prefill 都来不及 → 所有后续请求阻塞超时
```
最大支持 QPS 位于 Stage 2→3 转折点，约 1/TTFT。SPECPREFILL 通过 token dropping 将转折点推后。

术语一般如何实现？如何使用？

在 vLLM 等框架中通过 client-side 计时测量。SLO 目标以 P50/P95/P99 TTFT 表述（如 < 500ms）。优化方向：(1) 减少 prefill 计算量（token dropping、sparse attention）；(2) 提高硬件利用率（更大 batch、TP）；(3) 调度优化。SPECPREFILL 属方向 (1)，将 prefill FLOPs 降至 token 保持率比例。

涉及论文标题：
- Speculative Prefill: Turbocharging TTFT with Lightweight and Training-Free Token Importance Estimation

---
