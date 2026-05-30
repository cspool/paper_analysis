## Mixture-of-Agents (MoA, 混合代理)

术语是什么？
Mixture-of-Agents (MoA) 是一种多 LLM 协作推理范式，将多个 LLM 作为并行 proposer agents 各自生成答案，再由 aggregator agent 融合这些输出产生最终响应。MoA 可组织为多层结构——相邻层间的 agent 通过"输出融合"建立依赖关系：设前驱 agents A1, A2, A3 输出 o1, o2, o3，后继 aggregator Aagg 的输入 prompt 为 S(Aagg) = ∪(S_prefix, o1, o2, o3, S_suffix)。MoA 在推理、QA 和代码生成等任务上展示了显著的实证收益，但在系统效率上面临两个核心挑战：(1) 全连接拓扑导致冗余 agent 间通信；(2) agent 间异构延迟和复杂数据依赖使得现有 LLM serving 框架（PD disaggregation）无法有效支持——前驱 agent 解码与后继 agent prefilling 被视作严格串行。

从算法pipeline角度拆解：
```
# All-to-All MoA 推理 (两层)
Input: user query Q
Layer 1 (N proposer agents, 并行):
  for each agent a_{1,i}:
    o_{1,i} = LLM_i(prompt_template(Q))
  # 各 agent 用不同 LLM 骨干，推理延迟异构

Layer 2 (aggregator):
  S_agg = concat(S_prefix, o_{1,1}, ..., o_{1,N}, S_suffix)
  o_final = LLM_agg(S_agg)
```
延迟分析：T_total = max_i(t_{1,i}) + t_prefill_agg + t_decode_agg，瓶颈为最慢 proposer 的完成时间和 aggregator 的 prefill（不与其他计算重叠）。

术语一般如何实现？如何使用？
- 基于开源 LLM serving 框架（SGLang、vLLM）部署，每 agent 可选不同模型骨干
- 现有系统多用 all-to-all 连接（Multi-Agent Debate、Reconcile），但存在冗余连接和低 GPU 利用率问题
- 适用于多视角推理任务：复杂数学、科学 QA、代码生成、指令遵循

涉及论文标题：
- Efficient Mixture-of-Agents Serving via Tree-Structured Routing, Adaptive Pruning, and Dependency-Aware Prefill-Decode Overlap
