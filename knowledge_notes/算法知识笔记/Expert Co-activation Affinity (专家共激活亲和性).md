## Expert Co-activation Affinity (专家共激活亲和性)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Expert Co-activation Affinity 是指在 MoE 模型的同一层中，某些 expert pairs 被同时激活的概率远超随机选择理论值的现象。在 top-k routing（k>1）的 MoE 模型中，每层有 n 个 expert，每个 token 选择 k 个 expert。如果 expert 选择完全随机，任意 expert pair (i,j) 被同时选中的理论概率为 $p = \frac{k(k-1)}{n(n-1)}$。但实际 profiling 发现，某些 expert pairs 的共激活频率是理论值的 20-40 倍，形成显著的"共激活亲和性"。这种亲和性在不同模型间表现不同：DeepSeek V3 的 heatmap 中频繁共激活的 pairs 形成 bright squares（受其 routing restriction 影响——token 仅路由到相邻 node），而 Qwen3 表现出更分散的 bright dots 模式。量化分析显示 top 10% 的 expert pairs 占据了 60-80% 的总激活量。

从算法pipeline角度拆解术语：
Expert Co-activation Affinity 直接影响 MoE 层计算的并行度和负载分布。在 EP 场景下，如果两个高频共激活的 expert 被分配到同一 GPU，则该 GPU 在该层将承受不成比例的计算负载（两个 expert 同时被大量 token 选中），而其他 GPU 可能空闲。反之，如果将共激活 expert pairs 分离到不同 GPU，可以最大化并行度，但引入跨 GPU 通信开销。

论文提出的 Expert-pair separation insight (Insight 5)：separate frequently co-activated expert pairs to maximize parallelism，但需要 trade-off communication costs。

专家共激活分析流程（基于论文 profiling methodology）：
```
输入: expert selection traces D (per-layer per-token expert IDs)
输出: co-activation heatmap H, top co-activated pairs

for each layer l:
    # 初始化 n×n 矩阵
    co_act_count = zeros(n, n)
    
    for each token t in traces[l]:
        selected_experts = traces[l][t]  # top-k expert IDs
        for each pair (i, j) in combinations(selected_experts, 2):
            co_act_count[i][j] += 1
            co_act_count[j][i] += 1  # symmetric
    
    # 归一化到理论随机概率
    total_tokens = len(traces[l])
    random_prob = k*(k-1) / (n*(n-1))
    H[l] = co_act_count / (total_tokens * random_prob)
    
    # H[l][i][j] > 1 表示高于随机期望的共激活
    # H[l][i][j] = 20-40 表示比随机高 20-40 倍
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- Co-activation heatmap 通过离线 profiling 构建：使用 SGLang 等 serving framework 收集 expert selection traces（论文在 8×H100 上收集了 >24,000 requests，>150 GB JSON traces）。
- 在系统优化中，共激活 affinity 信息用于：(1) Expert placement——将高频共激活的 expert pairs 分配到不同 compute unit 以均衡负载（但需与通信开销 trade-off）；(2) Expert 复制策略——若共激活无法通过 separation 解决（通信开销过高），可在多个 unit 复制高频共激活 expert pair 中的专家。
- 注意：Llama 4 每层只选 1 个 expert（k=1），因此不存在 co-activation 关系。论文仅分析 DeepSeek V3 (top-8) 和 Qwen3 (top-8) 的共激活模式。

涉及论文标题：
- Orders in Chaos: Enhancing Large-Scale MoE LLM Serving with Data Movement Forecasting

---
