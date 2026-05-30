## Expert Prefetch Pruning (专家预取剪枝)

术语是什么？
Expert Prefetch Pruning 是 PopFetcher 提出的优化 expert prefetching 决策空间的方法。由于大集群中 global expert 数量巨大（如 256 GPU × 128 experts/GPU = 32768 experts），每 worker 决策是否预取每个 remote expert（δ_{n,w}^i ∈ {0,1}）的搜索空间为指数级。PopFetcher 通过两重约束剪枝：(1) GPU memory limitation：预取 expert 总参数量 ≤ 可用 GPU memory；(2) Transfer time constraint：预取传输时间 ≤ 非 MoE 计算时间；(3) Popularity filtering：仅 top-k×N 个热门 expert 进入候选集；(4) Efficiency threshold：仅当 ε = P_w/W_{n,w} > 3αH 且 B_{n,w}^i > εαH/2(ε-3αH) 时该 expert 才值得预取。

从kernel调度角度拆解术语：
剪枝与预取决策的数学过程：
```
Input: expert_popularity[p_w^i], gpu_memory[Mem_w^free], bandwidth[W_{n,w}], compute[P_w]
Output: prefetch_plan[δ_{n,w}^i]

// Step 1: Popularity filtering
candidates = top_popularity(experts, k × N)  // 至多 top-k × N 个 expert

// Step 2: Efficiency threshold (Eq. 13)
for each expert E_n^i in candidates:
    ε = P_w / W_{n,w}
    if ε <= 3αH:
        skip                                    // 带宽充足时不值得 prefetch
    threshold = εαH / (2(ε - 3αH))
    if B_{n,w}^i <= threshold:
        skip                                    // 接收 token 太少不划算

// Step 3: Memory + time constraint (Eq. 8-10)
valid = []
for each remaining expert (sorted by popularity desc):
    prefetch_size = 2αH² / W_{n,w}
    if (total_prefetch + prefetch_size <= Mem_w^free) AND
       (total_transfer_time + prefetch_size <= Time^{non-MoE}):
        valid.append(expert)
        total_prefetch += prefetch_size

// Step 4: Solve min-max latency (Eq. 7)
δ^* = argmin_δ max_w Lat_w^{prefetch}(δ)
// 在中后期训练中可固定 δ^* 或降低 replanning 频率
```

术语一般如何实现？如何使用？
实现为 CPU 异步执行的 decision-maker 模块（Python），在 GPU 训练期间后台运行。popularity prediction 通过 All-Gather 聚合各 worker 的 per-expert token 计数（小向量，sync 开销 negligible < 100ms）。剪枝后搜索空间从指数级降至可穷举/贪心求解规模。中后期训练可利用 expert 分布的稳定性降低 replanning 频率。

涉及论文标题：
- PopFetcher Towards Accelerated Mixture-of-Experts Training Via Popularity Based Expert-Wise Prefetch
