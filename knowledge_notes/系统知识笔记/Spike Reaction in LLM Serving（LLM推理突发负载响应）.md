## Spike Reaction in LLM Serving（LLM推理突发负载响应）

术语是什么？通过联网搜索让回答具体和精准。
Spike Reaction 是 PiLLM 处理突发请求负载的兜底机制。当 request dispatcher 无法将新请求分发到任何 idle 或 active instance（所有 instance 预计完成时间超出请求 deadline）时触发。Spike reaction 尝试将 spike queue 中的请求按 deadline 排序后贪心构造最大可行 batch（逐个添加请求并验证 batch 的 predicted latency ≤ 各请求 deadline），并在预测利用率超过阈值时快速激活新 GPU instance。作为 safety net 与正常 workload prediction 解耦——常规 elastic dispatch 依赖滑动窗口统计平滑预测，spike reaction 覆盖分布漂移或长尾 outlier 导致的预测失效。

从系统架构角度拆解术语：
1. **进入条件**：Dispatcher 遍历所有 instance 后返回空，请求入 spike queue。
2. **Batch packing**：按 deadline 排序，贪心构造最大可行 batch。
3. **利用率预测**：计算所有 running instance 加入 spike batch 后的预测利用率。
4. **扩容决策**：若预测利用率超阈值且 spike batch 仍无法排入现有 instance，激活新 instance。
5. **执行**：spike batch 在新激活或最快完成卸载的 instance 上执行。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现要点：(1) spike queue 独立于正常 dispatch queue；(2) 新 instance 冷启动延迟需计入 SLO——论文假设 instance 池预热（keep-warm）；(3) 分布快速漂移和长尾 outlier 过多时 spike reaction 触发频率升高，可通过调大 error bound ε 减少误触发但会增加资源 overcommit 风险；(4) spike reaction 的 batch packing 以请求 deadline 为主要约束而非最大化 batch size。

涉及论文标题：
- PiLLM: Resource-Efficient LLM Inference Using Workload Prediction
