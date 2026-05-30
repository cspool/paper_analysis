## Topology-aware Gate (in MoE Training)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Topology-aware Gate 是 FasterMoE 提出的网络拓扑感知的 expert 选择策略。标准 MoE gate 仅基于 fit score 选择 top-k experts，导致大量跨节点通信在树形拓扑的上层链路上产生拥塞。Topology-aware Gate 限制跨节点 token 数量上限 L = (W_net / (M·W_local))·B，超出 L 的 token 在本地节点内重新选择 expert，从而将跨节点通信量降低至与节点内通信等时。同时保留 fit score 最高的 token-expert 对，减少对模型质量的影响。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# 树形拓扑下的 Topology-aware Gate
# N 个节点, 每节点 M 个 worker
# W_net: 跨节点带宽, W_local: 节点内带宽

def TopologyAwareGate(tokens, scores, B, M, W_net, W_local):
    # 计算跨节点 token 上限
    L = (W_net / (M * W_local)) * B
  
    # 收集所有希望跨节点的 tokens
    remote_candidates = []
    for t in tokens:
        best_expert = argmax(scores[t])
        if expert_node[best_expert] != local_node:
            remote_candidates.append((t, scores[t][best_expert]))
  
    # 按 fit score 降序排序
    remote_candidates.sort(key=lambda x: -x[1])
  
    # 仅允许 fit score 最高的 L 个跨节点
    allowed = set(t for t, _ in remote_candidates[:L])
  
    # 其余 token 在本地节点内重新选择
    for t in tokens:
        if t not in allowed:
            # 限制 expert 选择范围为本地节点
            local_experts = [e for e in experts if expert_node[e] == local_node]
            t.selected_expert = argmax(scores[t][local_experts])
  
    # 结果: 跨节点流量从 M(N-1)/N · BH 降至 W_net/W_local · BH
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

在 FasterMoE 中作为 FastMoE 的 custom gate 实现，与环境变量控制开关配合。实验显示（MoE-GPT, johnny 集群）：启用拓扑感知门控后 per-iteration 延迟减少 9.4%，但需额外 18% 的 iteration 才能收敛（因部分 token 被重新分配到次优 expert）。整体收敛时间比 GShard 快 1.37×，比 BASE Layer 快 2.19×。FasterMoE 强调这是一种 co-design 方法论——对于不同网络拓扑，应设计对应的专用 gate。

涉及论文标题：
- FasterMoE modeling and optimizing training of large-scale dynamic pre-trained models
