## TTFT / TPOT SLO Constraints in Multimodal Serving

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

TTFT (Time-To-First-Token) 和 TPOT (Time-Per-Output-Token) 是多模态推理 Serving 系统的两个核心延迟 SLO（Service Level Objective）指标。TTFT 衡量从请求到达系统到生成第一个 token 的时间（包含 Encode + Prefill 延迟），TPOT 衡量后续每个 token 的平均生成时间（Decode 阶段的延迟）。在多模态场景下，TTFT 受视觉编码器和跨阶段数据传输的显著影响，而 TPOT 主要受 Decode 阶段资源竞争影响。EPD-Serve 根据解耦策略定义差异性 SLO：Encode-disaggregated 时 TPOT ≤ 80ms，Decode-disaggregated 时 TPOT ≤ 50ms，TTFT ≤ 2000ms 为通用上限。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

TTFT/TPOT 在 EPD-Serve 不同部署下的表现（openPangu-7B-VL, ShareGPT-4o, 10 req/s）：

```
Deployment    NPU  TTFT(ms)   TPOT(ms)  SLO Attain  Eff Throughput
TP1×2         2     658.27     95.56      2.15%        13.38
(E-PD)×2      2     548.32     62.22      3.13%        19.70
EP-D          2    5523.82     27.31      8.20%        21.54
(E-P)-D       2    2386.85     28.40     26.17%        77.36
(E-D)-P       2     651.86     50.71     22.66%        69.18
E-P-D         3     557.89     28.92     94.34%       192.70

约束: TTFT ≤ 2000ms, TPOT ≤ 50ms
```

关键洞察：(1) Decode 解耦是稳定低 TPOT 的核心——所有 D-disaggregated 部署 TPOT 均 < 51ms；(2) E-P-D 全解耦实现最高 SLO 达成率（94.34%），但需要 3 NPU；(3) TTFT 受 P-D 传输机制影响显著——EP-D 的 naive 同步传输导致 TTFT 高达 5524ms，远超 SLO 上限。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

TTFT 优化方向：(1) E-P 异步预取隐藏 Encode→Prefill 传输延迟；(2) P-D 分层分组传输隐藏 Prefill→Decode KV 传输延迟；(3) 模态感知路由避免多模态请求阻塞纯文本请求。TPOT 优化方向：(1) Decode 阶段独立部署避免与 Encode/Prefill 竞争 NPU 资源；(2) Continuous batching 最大化 Decode NPU 利用率；(3) Operator-level co-location 在共置场景下复用空闲计算周期。EPD-Serve 的 AISBench 工具控制请求注入速率 1-12 req/s 以模拟不同并发级别。SLO 达标率的计算基于所有请求中满足 SLO 约束的比例。

涉及论文标题：
- EPD-Serve A Flexible Multimodal EPD Disaggregation Inference Serving System On Ascend
