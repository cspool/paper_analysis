## Expert Selection Temporal Correlation (专家选择时间相关性)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Expert Selection Temporal Correlation 是指 MoE 模型在推理过程中，expert 选择在不同时间尺度上表现出的可预测模式。论文通过 >24,000 requests 的 profiling 识别出三个时间尺度的相关性：(1) **Layer-level correlation (Ob1)**：相邻两层之间 expert 选择的条件概率分布——给定 layer N 选择了 expert i，layer N+1 选择 expert j 的概率 $P(e_j^{N+1} | e_i^N)$，top 20% 候选 expert 覆盖了 50-77% 的条件概率质量；(2) **Token-level correlation (Ob2)**：同一层相邻两个 token 之间 expert 选择的条件概率——高层（17, 43）出现明显的对角线模式（同一 expert 在相邻 token 被反复选中，即 temporal locality），而低层（1, 3）不明显；(3) **Prefill-decode-level correlation (Ob3)**：prefill 和 decode 阶段的 expert 选择模式高度相似——cross-layer 和 cross-token heatmap 形状相似（Spearman's ρ ≥ 0.7 for most layers），top-5 prefill experts 覆盖 ~60% 的 top-5 decode experts，top-20 覆盖 ~90%。

从算法pipeline角度拆解术语：
三种时间相关性对应不同的 reuse distance 和优化机会：

```
时间尺度层次:
Pattern        | Reuse Distance         | 优化目标          | Memory Tier
Layer-level    | 短（相邻层连续执行）     | LLC/prefetch      | 快速小容量
Token-level    | 长（遍历所有层后）       | DRAM cache        | 大容量
Prefill-decode | 跨阶段（不同机器可能分离）| 初始 placement    | 静态/半静态
```

论文用 Conditional CDF 量化相关性：
- 对 layer-level：$F(x) = P(\text{top } x\% \text{ candidates cover } \ge y\% \text{ of conditional probability})$
- 结果：top 20% next-layer candidates cover 50% (DeepSeek), 65% (Qwen3), 77% (Llama4), 56% (Kimi K2) 的条件概率
- 对 token-level：top 20% next-token candidates cover 47% (DeepSeek), 62% (Qwen3), 80% (Llama4), 53% (Kimi K2)

Llama 4 的相关性最强，DeepSeek V3 最弱——这与模型架构差异相关（Llama 4 在 MoE layers 之间插入 dense FFN layers）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- Temporal correlation 通过离线 profiling 建立：使用 SGLang 部署模型 → 收集每层每 token 的 expert selection → 构建 heatmap（条件概率矩阵）→ 存储为 lookup table。
- 在 serving 系统中应用：(1) **Cross-hierarchy memory management** (Insight 2): layer-level correlation 指导 LLC/快速 memory tier 的 prefetch，token-level correlation 指导 DRAM/大容量 tier 的 cache；(2) **Data-driven predictor**：用 cross-token heatmap 预测下一 token 的 expert 选择；(3) **Prefill-guided placement**：用 prefill traces 预测 decode 阶段的 expert 需求，指导初始 placement。
- 论文开源了所有 heatmap 和 traces：https://huggingface.co/datasets/core12345/MoE_expert_selection_trace

涉及论文标题：
- Orders in Chaos: Enhancing Large-Scale MoE LLM Serving with Data Movement Forecasting
