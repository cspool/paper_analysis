## Reasoning Cliff（推理悬崖：KV 增长越过 HBM 容量迫使调度器降级）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
本论文提出的现象：decode 阶段 KV cache 随输出序列长度线性增长（8B 模型 20M 聚合 token 输出需 >2TB），当累计 KV 超过可用 HBM 时，调度器被迫 preempt/recompute/拒收后续工作——此"悬崖"出现时机取决于输出长度与 batch 的交互：短输出负载下悬崖出现较晚（decode 后期），长输出 reasoning 负载下 KV 增长把悬崖拉前，batch 足够大时（Llama-405B batch 4K/5K）甚至在 prefill 阶段就耗尽容量、请求无法初始化。这是 Capacity Trap 的时间维度——容量饱和不是稳态事件而是随 decode 推进逼近的临界点。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
```
# Llama-405B（KV 1.05MB/token）batch 从 500→5K（本论文 Fig.14 实测）
batch 500：  悬崖出现在 decode 后期（KV 100% 在解码中途）
batch 4K/5K：悬崖前移到 prefill 期——KV 在准入/预填充阶段即耗尽，请求无法初始化
# 治理流程（调度器视角）：
admission 时估计未来 KV 增长（OSL×每 token KV）→ 预留 decode 容量 → 超过则拒收/降并发
# 8B 模型 20M token 聚合输出 → >2TB KV → 远超 8×141GB=1.1TB 集群容量（DP 下每卡 141GB 更小）
```
Annotations：悬崖点的数学本质是"KV(OSL) 与可用 HBM 的交叉时刻"；batch 与 OSL 共同决定交叉时点（batch 大→prefill 即撞墙）；调度器可做的只有降级（chunked prefill、抢占、拒收）——降级本身引入 Start-Up Latency/重算惩罚。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
作为观测概念，其落地手段：(1) admission control——按 OSL 分布与每 token KV 速率预留 decode 容量（论文明确建议"估计未来 KV 增长而非只看当前内存使用"）；(2) 架构级 KV 压缩（MLA、GQA、量化）推迟悬崖（R1-671B 靠 MLA 保持适度 KV 消耗速率）；(3) 容量池化（TP/PP/KV offload）抬高悬崖的绝对高度；(4) 分阶段部署——prefill/decode 解耦让悬崖只在 decode 域管理。论文把它与 Capacity Trap（并发维度）并列，作为"内存容量是一等设计参数"的核心证据。

涉及论文标题：
- Understanding Inference Scaling for LLMs Bottlenecks, Trade-offs, and Performance Principles
