## PIM Channel Underutilization in Long-Context LLM Inference

术语是什么？通过联网搜索让回答具体和精准。

PIM Channel Underutilization是长上下文LLM推理中PIM系统面临的关键效率问题：由于PIM channel的KV cache容量有限（每个channel通常只能容纳1-2个request的KV cache），当context length增加时单个request的KV cache膨胀、batch size被压缩，导致许多channel因无足够head-batch pair填充而空闲。PIMphony论文识别三个具体表现形式：(1) Batch-induced underutilization——长上下文下batch size小（可能batch=1），HFP（Head-First Partitioning）按head-batch pair分配channel，导致仅少数channel激活；(2) Imbalance-induced underutilization（Tensor Parallelism）——不同request token length不同，短request的channel早完成、长request的channel仍在执行，整体utilization受最慢channel限制；(3) Stage-sparse underutilization（Pipeline Parallelism）——每stage只服务当前layer相关的request，其他request的KV cache所在channel不参与该stage计算。论文在32K context CENT分析中观察到MAC utilization下降48%。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。

```
// PIM Channel Underutilization示例（CENT PIM-only, TP=2, LLM-7B-32K）:

// 配置: 2 PIM modules × 16 channels = 32 total channels
// Workload: batch=2, requests R1(32K tokens), R2(16K tokens)

// HFP分配 (Tensor Parallelism, TP=2):
// Module 0: heads 0-15, Module 1: heads 16-31
// Module 0内部: head-batch pairs分配到16 channels
//   CH0-3: R1 heads 0-3 → 处理32K tokens → 耗时T
//   CH4-7: R2 heads 0-3 → 处理16K tokens → 耗时T/2 → 之后idle
//   CH8-15: R1 heads 4-15 或 R2 heads 4-15 → 类似不平衡

// Channel utilization (时间维度):
// |=== CH0: R1-h0 active (T) ===============|
// |=== CH4: R2-h0 active (T/2) ===| idle (T/2) |
// 平均utilization: 约75% (若均匀分布)

// Pipeline Parallelism (PP=2):
// Module 0: layers 0-15, Module 1: layers 16-31
// 每个decode step:
//   Module 0处理时: layers 0-15的KV cache所在channels active
//   layers 16-31的KV cache所在channels idle
// 仅50% channels同时active
```

PIMphony的TCP通过将并行维度转为token维度解决此问题：token数量随context length增长，长上下文越长并行度越高——在16-channel/16-bank配置下QK^T token length>256即可full channel activation。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Channel underutilization的检测：(1) 通过PIM performance counter统计per-channel MAC active cycle ratio；(2) 通过simulator profiling分析channel-level utilization breakdown。缓解方案：(1) PIMphony的TCP——token-centric partitioning decouple channel utilization from batch size；(2) Dynamic batching——runtime根据可用channel容量动态调整batch composition；(3) Channel-aware request scheduling——将token length相似的request配对分配到同一PIM module减少imbalance。

涉及论文标题：
- PIMphony: Overcoming Bandwidth and Capacity Inefficiency in PIM-based Long-Context LLM Inference System
