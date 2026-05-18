## LATS (Language Agent Tree Search)

术语是什么？通过联网搜索让回答具体和精准。

LATS（Language Agent Tree Search）是一种基于Monte Carlo Tree Search (MCTS)的AI agent workflow。与ReAct的顺序单路径推理不同，LATS在每步扩展时生成多个候选reasoning/action child nodes（通过并行LLM calls），模拟多条可能的推理路径，并使用MCTS的selection-expansion-simulation-backpropagation循环选择最优路径。LATS的tree search机制使得agent能在多个candidate paths中比较和选择，提升了决策质量但显著增加了LLM call数量（论文测量平均71.0次/request）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// LATS单请求执行pipeline (Monte Carlo Tree Search)
root = Node(state=initial_context)
for iteration in range(max_iterations):
    // Selection: 从root沿tree用UCT选择最有希望的leaf node
    node = select(root, policy=UCT)
    
    // Expansion: 为当前node生成多个child nodes
    children = []
    for i in range(num_child_nodes):  // Parallel scaling参数
        prompt = format_prompt(node.trajectory)
        child_output = async_llm_call(prompt)  // 并行LLM calls
        children.append(parse_node(child_output))
    node.children = children
    
    // Simulation/Evaluation: 评估各child node的value
    for child in children:
        child.value = evaluate(child)  // LLM self-evaluation
    
    // Backpropagation: 更新path上所有node的统计信息
    backpropagate(node, children)
    
// 返回best trajectory
return get_best_trajectory(root)
```

论文核心发现：LATS平均LLM calls/request最高（71.0次），parallel scaling增加child nodes从1→16可提升14.4pp准确率同时降低196.3s平均延迟（更快找到高质量path）。但代价是更多并发LLM requests→增加GPU memory pressure和KV cache占用。prefix caching可平均降低64.8% LATS memory requirement（共享prefix复用）。LATS只保留root→current node path（不concat全部历史），限制了context length增长。8B LATS HotpotQA accuracy达80%，energy/query 22.76 Wh（ShareGPT 0.32 Wh的71.7×）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

LATS实现基于MCTS算法框架：(1) tree state维护node的visit count和value estimate；(2) UCT (Upper Confidence Bound for Trees) formula用于selection阶段平衡探索与利用；(3) 论文优化了LATS原实现以支持concurrent LLM inference和parallel tool invocation（原版顺序执行加重延迟）；(4) 每个node的trajectory仅包含root→当前node的path（不concat全部历史），控制context length。开源：论文基于LATS官方实现优化（https://github.com/VIA-Research/AgentBench）。LATS在accuracy上优于ReAct和Reflexion，但computational overhead最高。

涉及论文标题：
- The Cost of Dynamic Reasoning: Demystifying AI Agents and Test-Time Scaling from an AI Infrastructure Perspective

