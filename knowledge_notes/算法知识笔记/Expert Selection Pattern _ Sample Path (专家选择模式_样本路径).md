## Expert Selection Pattern / Sample Path (专家选择模式/样本路径)

术语解释
Expert Selection Pattern 是 Lina 论文实证发现的 MoE 跨层关联规律：在 token 流经各 MoE layer 时，相邻层之间 expert 选择呈现可预测模式——在 layer i 中选择同一 expert 的 tokens，在 layer i+1 中倾向于再次选择同一 expert。Sample Path 是 token 连续穿过 l 层时选择的 expert 序列，用于估计下一层的 expert popularity distribution。

术语是什么？
Lina 的实验发现：tokens 在相邻层中选择同一 top-1 expert 的概率为 41.94%（k=1），选择同一 top-2 中任一 expert 的概率为 54.59%（k=2）。更深层该模式更明显。原因：Gate network 架构简单，路由决策主要基于 token 局部特征（POS、词义等）；Expert 专注于局部句法信息（非跨序列依赖）；特征固定于 token，导致相似 token 在各层被相同 expert 处理。

样本路径（Sample Path）定义：token 从 layer i-l 到 layer i 所经过的 expert 序列 `[e_{i-l}, ..., e_i]`。路径长度 l 控制 accuracy-overhead tradeoff。

从算法pipeline角度拆解术语。
```
# Expert Selection Pattern Profiling (Training阶段)
def profile_expert_selection_patterns(model, dataset, path_length=3):
    patterns = defaultdict(dict)  # {layer: {sample_path: distribution}}
    for batch in dataset:
        for layer_i in range(path_length, model.num_layers):
            sample_paths = collections.defaultdict(list)
            for token in batch.tokens:
                path = tuple(token.expert_history[layer_i-path_length:layer_i+1])
                sample_paths[path].append(token)
            # 计算每个 sample path 对应的 layer i+1 expert 分布
            for path, tokens in sample_paths.items():
                next_experts = [token.expert_history[layer_i+1] for token in tokens]
                dist = compute_distribution(next_experts)  # P(e) for each expert
                patterns[layer_i + 1][path] = dist
    return patterns

# Online Estimation (Inference阶段)
def estimate_expert_popularity(batch, current_layer, patterns, path_length=3):
    estimated = defaultdict(float)
    for token in batch.tokens:
        path = tuple(token.expert_history[-path_length:])
        dist = patterns[current_layer + 1].get(path, uniform_dist)
        for expert_id, prob in dist.items():
            estimated[expert_id] += prob / batch.num_tokens
    return {e: N * p for e, p in estimated.items()}  # n_e = N * P(e)
```

关键发现：
- k=1: 41.94% tokens 在相邻层中选择同一 expert
- k=2: 54.59%
- 更深层 pattern 更强（later layers → higher ratio）
- 不是 token 级精准预测，但提供 batch 级 expert popularity estimation

术语一般如何实现？如何使用？
- Profiling: 在 training 阶段 load balancing loss 稳定后采集 expert selection results
- 存储: unordered_map per layer（key: sample path tuple, value: distribution vector）
- 路径长度 l=3 为默认（l=1 accuracy 31.6%→l=3 60.4%→l=6 71.4%）
- 配合 Two-Phase Scheduling: Phase 1 用估算做预分配，Phase 2 偏差大时微调
- 局限性: 需要 training-stage profiling，每个 task 需独立 profile

涉及论文标题：
- Accelerating Distributed MoE Training and Inference with Lina

---
