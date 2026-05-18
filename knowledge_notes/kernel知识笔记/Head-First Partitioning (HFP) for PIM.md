## Head-First Partitioning (HFP) for PIM

术语是什么？通过联网搜索让回答具体和精准。

Head-First Partitioning（HFP）是现有PIM-based LLM加速器（CENT[16]、NeuPIMs[21]等）普遍采用的workload映射策略：将attention head与batch pair分配到PIM channel执行，每个channel负责特定(head, batch)组合的QK^T和SV计算。HFP隐含假设batch size或head数量足够多以填充所有channel。但在长上下文decoding中：(1) 单个request的KV cache足以占满一个channel的容量，压制batch size；(2) Tensor Parallelism下不同request token length差异导致channel执行时间不均衡；(3) Pipeline Parallelism下每stage只激活与当前layer相关的少数channel。PIMphony论文在32K context CENT分析中观察到HFP导致MAC utilization下降48%。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。

```
// HFP执行（PIM module with 4 channels, 2 heads, batch=2）:
// 分配: CH0→R(1,head=1), CH1→R(1,head=2), CH2→R(2,head=1), CH3→R(2,head=2)
//
// Decode step (batch=2, 每request需处理所有heads):
for each channel c:
    (r, h) = channel_assignment[c]  // 固定映射
    q = get_query(r, h)            // 从对应request和head取query
    K_cache = get_K_cache(r, h)    // 从该channel的KV cache读Key
    // QK^T: dot(q, K_cache[t]) for t in 0..T_r-1
    // SV: weighted sum of V_cache[t] by score[t]

// HFP问题示例 (TP=2, batch=2, 但request 1比request 2长得多):
// CH0: R(1,h1) — 处理128K tokens → 耗时很长
// CH1: R(2,h1) — 处理16K tokens  → 早早完成，之后idle
// CH2: R(1,h2) — 处理128K tokens → 耗时很长
// CH3: R(2,h2) — 处理16K tokens  → 早早完成，之后idle
// 总体MAC utilization = (128+16+128+16) / (128×4) ≈ 56%
// 极端情况batch=1时: 仅2/4 channel激活 → MAC util 50%
```

HFP的核心缺陷是并行维度（head/batch）在长上下文场景下稀缺且波动——head数固定（32-64），batch size被KV cache容量压缩，导致channel无法被充分填充。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

HFP在现有PIM系统中通过compiler或runtime在模型加载时静态分配实现——根据模型head数和PIM module/channel拓扑预先确定每个channel的(head, layer) assignment，推理过程中保持不变。其简单性和确定性使其易于实现，但在长上下文下的效率退化促使PIMphony提出TCP作为替代方案。

涉及论文标题：
- PIMphony: Overcoming Bandwidth and Capacity Inefficiency in PIM-based Long-Context LLM Inference System

