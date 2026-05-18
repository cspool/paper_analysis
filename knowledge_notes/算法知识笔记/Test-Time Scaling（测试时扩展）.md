## Test-Time Scaling（测试时扩展）

术语是什么？通过联网搜索让回答具体和精准。

Test-Time Scaling是指在推理时通过增加计算量来提升预训练LLM的推理性能，而不修改模型参数的方法论。代表性技术包括Chain-of-Thought（引导模型生成中间推理步骤）、Tree-of-Thought（探索多条推理路径）、Self-Consistency（采样多个response选最优）等。AI Agent将test-time scaling进一步推进为dynamic reasoning：不仅通过prompt设计增强推理，还通过多步决策集成tool use和维护中间推理状态。这使得agent能根据中间结果动态调整行为，实现对外部环境的自适应。Test-time scaling的核心trade-off：增加计算可提升accuracy，但存在diminishing returns和不可持续的基础设施成本。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Test-time scaling在AI Agent中有两种基本形式：

**Sequential Scaling（顺序扩展）**：逐步增加推理步数
```
for step in range(iteration_budget):
    result = agent_step(context)
    if result.is_final: break
// 延迟随步数线性增长，峰值资源需求低
```

**Parallel Scaling（并行扩展）**：同时探索多条推理路径
```
// LATS tree expansion
children = []
for i in range(num_child_nodes):
    children.append(async_llm_call(context))  // 并行LLM calls
results = await gather(children)
best = select_best(results)
// 可同时降低延迟和提升准确率，但增加瞬时GPU memory和serving contention
```

论文核心发现：test-time scaling存在急剧递减的边际收益。Reflexion从16.9s→25.6s仅获4% accuracy gain；从56.0s→325.5s需31× cost获得同等marginal gain。Parallel scaling (LATS)增加child nodes从1→16可提升14.4pp准确率同时降低196.3s平均延迟（因为更快找到高质量path），但代价是更多并发LLM requests。8B模型配合LATS parallel scaling可接近70B模型性能但energy更低。单次agent query的GPU energy比ShareGPT高62.1×–136.5×。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Test-time scaling通过agent framework的参数配置实现：(1) iteration budget：控制最大推理步数和tool调用次数，限制sequential scaling的范围；(2) few-shot examples：适量增加可同时提升准确率并减少推理步数，但过多prompt tokens可能超过模型最优处理区间导致accuracy下降；(3) reflection depth：Reflexion中self-evaluation的轮数；(4) child nodes per expansion：LATS中Monte Carlo Tree Search每步并行探索的分支数；(5) model size scaling：更大模型用更少步骤达到高准确率但energy/query大幅增加（8B→70B：单GPU→8 GPU，单query energy增加约8×–12×）。实现上需在agent worker的state machine中设置stop condition和branching logic。

涉及论文标题：
- The Cost of Dynamic Reasoning: Demystifying AI Agents and Test-Time Scaling from an AI Infrastructure Perspective
- Scaling LLM Test-Time Compute with Mobile NPU on Smartphones

