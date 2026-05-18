## SLO Attainment（SLO达成率）

术语是什么？通过联网搜索让回答具体和精准。
SLO Attainment 是 multi-SLO LLM serving 的核心评估指标，定义为满足其 TPOT SLO 的请求占全体请求的比例。与 Goodput 互补：SLO attainment 衡量"多少比例请求被满足"，Goodput 衡量"满足 SLO 条件下的有效吞吐量"。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
计算流程（以 AdaServe 为例）：
1. Per-token tracking：每请求记录每个 token 生成时间戳
2. TPOT 计算：计算所有 decode token 的 TPOT（相邻 token 时间间隔）
3. SLO 判断：若 r_i 的 P99 TPOT ≤ SLO_i 阈值，则标记为 SLO-compliant
4. 聚合：SLO attainment = |SLO-compliant requests| / |all requests|
5. 可按 SLO 类别单独计算（coding copilot vs chatbot attainment）
AdaServe 在不同 RPS 下 SLO attainment 提升可达 2.1×/1.6× vs best baseline。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在 AdaServe 中通过 request manager 的 per-request latency tracking 和 post-hoc 聚合计算。面向 strict SLO 请求时优先优化 attainment，面向 relaxed SLO 请求时优先优化 goodput。

涉及论文标题：
- AdaServe: Accelerating Multi-SLO LLM Serving with SLO-Customized Speculative Decoding
