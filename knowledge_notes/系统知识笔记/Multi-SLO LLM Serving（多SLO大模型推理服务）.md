## Multi-SLO LLM Serving（多SLO大模型推理服务）

术语是什么？通过联网搜索让回答具体和精准。
Multi-SLO LLM Serving 指在同一个 LLM 推理服务集群上同时承载多种具有不同 SLO 需求的推理应用（如 coding copilot 需 TPOT < 50ms，chatbot 需 TPOT < 100ms，summarization 需 TPOT < 150ms），并在满足各应用差异化 SLO 的前提下最大化系统吞吐。与传统的 uniform SLO serving（所有请求共享同一延迟目标）不同，multi-SLO serving 要求调度器感知每个请求的 SLO 类别和当前 slack，进行细粒度的差异化资源分配。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
AdaServe 的 multi-SLO serving 流程：
1. **SLO 注册**：每种应用类型注册其 TPOT SLO（如 coding copilot = 50ms/token, chatbot = 100ms/token, summarization = 150ms/token）
2. **请求标注**：到达请求携带其 SLO 类别标签
3. **Per-request slack 计算**：每轮 iteration，scheduler 计算每请求的当前 slack = TPOT_SLO × tokens_generated - latency_elapsed（正 slack 表示超前，负 slack 表示滞后）
4. **差异化 token 分配**：lagging 请求（负 slack）在 SLO-customized selection 中获得优先 token allocation；ahead 请求（正 slack）让出 budget
5. **动态调节**：当 strict 请求比例升高时，更多 budget 流向 strict SLO 请求；当 relaxed 请求为主时，budget 转向 throughput 优化

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现要点：(1) SLO 类别标注机制（请求携带 SLO class ID）；(2) per-request latency tracking（记录首 token 时间和各 token 生成时间）；(3) 调度器以 per-request SLO slack 为决策输入。AdaServe 在 FlexFlow Serve 的 request manager 中维护 per-request 状态。

涉及论文标题：
- AdaServe: Accelerating Multi-SLO LLM Serving with SLO-Customized Speculative Decoding
