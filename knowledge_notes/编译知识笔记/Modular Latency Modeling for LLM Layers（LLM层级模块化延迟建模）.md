## Modular Latency Modeling for LLM Layers（LLM层级模块化延迟建模）

术语是什么？通过联网搜索让回答具体和精准。
Modular Latency Modeling是Laser提出的LLM transformer layer延迟预测方法，将每层拆分为stateless module（QKV projection、attention output projection、FFN）和stateful module（self-attention）两类分别建模，然后将两者聚合得到per-layer latency。该模型用于支撑layer-level scheduling中的实时延迟评估——prefill Scheduler用其估计TTFT slack以决定抢占/合并，decode Planner用其估计per-iteration latency以构造execution plan。profiling开销低（<2秒），预测准确率94.6%-98.6%。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。

Latency model的数学形式和profiling流程：
```
// 1. Stateless module latency（分段线性模型）
//    QKV projection, attention output projection, FFN
//    受GPU tile quantization影响: token数跨特定阈值时延迟突增
ω(n) = {
    a_0 * n + b_0    t ∈ [1, n_0)
    ...
    a_m * n + b_m    t ∈ [n_{m-1}, n_m)
}
// n: number of input tokens
// Segment width固定为32 tokens（GPU tile sizes的公约数）
// Communication overhead (model parallelism) 线性增加并入上式

// 2. Stateful self-attention module latency（线性模型）
τ(n, Σ_{r=1}^{n} c_r) = α * n + β * Σ_{r=1}^{n} c_r + γ
// n: token count in current batch
// Σ c_r: total context length (KV cache size)
// α: per-token compute coefficient
// β: per-context-token memory access coefficient
// γ: constant overhead
// Pearson |correlation| > 0.78 确认线性关系的有效性

// 3. Per-layer latency aggregation
T(n, Σc_r) = ω(n) + τ(n, Σc_r)
// = stateless_module_latency + self_attention_latency

// 4. Per-iteration latency (aggregate across N layers)
Iter_latency = Σ_{j=1}^{N} T_j(n_j, Σc_j)
// 对decode: Planner simulate future iterations, take max as estimate
```

Profiling流程：系统初始化时对每个serving instance测量不同token count和context length下的module latency，拟合模型参数。Laser实测整个过程在<2秒完成，不足以影响instance启动延迟。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Laser在离线profiling阶段构建latency model，在线serving期间实时使用。与MuxWise/Bullet的SM-scaling roofline model（SRM）不同，Laser的modular model关注**时间域延迟预测**而非SM资源分配下的性能预测。该模型的两类module区分利用了transformer layer计算的结构特性：stateless modules对每个token独立计算（延迟与token count线性或分段线性），stateful attention涉及KV cache的跨token依赖（延迟与token count和context length同时相关）。将该模型推广到其他model architecture时，需重新profiling以获取model-specific系数。

涉及论文标题：
- Laser: Unlocking Layer-Level Scheduling for Efficient Multi-SLO LLM Serving
