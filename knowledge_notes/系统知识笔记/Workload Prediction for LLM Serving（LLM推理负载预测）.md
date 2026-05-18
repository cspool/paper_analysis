## Workload Prediction for LLM Serving（LLM推理负载预测）

术语是什么？通过联网搜索让回答具体和精准。
Workload Prediction for LLM Serving 是 PiLLM 提出的基于统计的批级别工作负载预测方法，用于驱动跨GPU资源管理和单GPU内KV cache调度。核心思想：用滑动窗口记录历史请求的输入/输出长度统计量（均值 μ、方差 σ²），通过中心极限定理为批级平均长度构造误差可控的预测上界。输出长度预测公式为 μ_d + σ_d/√|B| · Φ⁻¹(1-ε)，其中 |B| 为batch大小、Φ⁻¹ 为标准正态分布的逆CDF、ε 为错误容忍度参数。再用离线校准系数将长度预测转换为 prefill FLOPs、decode FLOPs、prefill KV memory 和 decode KV memory 四维资源需求。区别于 per-request prediction（单个请求输出长度难以预测、计算开销大），批级统计预测利用大数定律——随batch size增大，实际平均行为的方差以 1/√|B| 速率收敛——以可控统计误差界替代精确单请求预测。

从系统架构角度拆解术语，预测驱动调度的运转流程：
1. **统计采集**：API layer 在接收请求时记录输入长度；decode instance 周期性回传已完成请求的实际输出长度。
2. **滑动窗口维护**：Global scheduler 维护固定长度滑动窗口（窗口覆盖秒级到十秒级的最近历史），在线更新 μ 和 σ。
3. **批级预测**：管理时间窗口内一批请求到达时，Predictor 计算平均输入/输出长度上界。
4. **资源转换**：用离线 profiled 的校准系数将长度转为 prefill/decode FLOPs 和 KV cache 内存需求。
5. **调度决策**：Inter-GPU manager 根据预测 FLOPs 与目标阶段延迟计算所需 prefill/decode 实例数；Intra-GPU scheduler 用预测 KV cache 需求设置 batch 内存预算。
6. **反馈闭环**：完成请求的输出长度回传更新窗口统计。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现要点：(1) 滑动窗口长度需覆盖短期burst但不over-smooth——论文按管理窗口设置；(2) 离线校准通过 profiling 不同输入/输出长度下的实际 FLOPs 和 KV cache 内存拟合线性系数；(3) ε 是系统trade-off参数，越小预测越保守（留更多 buffer）、越大越激进（更省GPU但SLO风险升高）；(4) 分布漂移时预测变差，PiLLM 以 spike reaction 兜底。该预测方法与 disaggregated prefill/decode 范式天然适配——prefill compute-bound 和 decode memory-bound 的 FLOPs/memory 分别独立估算。

涉及论文标题：
- PiLLM: Resource-Efficient LLM Inference Using Workload Prediction
