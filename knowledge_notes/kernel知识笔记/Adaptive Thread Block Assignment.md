## Adaptive Thread Block Assignment

术语解释
Adaptive Thread Block Assignment 是 Comet 提出的运行时资源分配策略，针对 fused kernel 中通信 TB（n^c）和计算 TB（n^p）的比例进行自适应选择。由于通信和计算负载随输入 token 长度 M、并行策略（TP×EP）动态变化，最优的 n^c/n^p 分割点也随之改变。Comet 通过 offline profiling + runtime lookup 实现自适应。

术语是什么？
Comet 预编译多个不同 n^c/n^p 比例的 kernel 变体，在部署前对每种 (M, EP, TP) 配置进行 profiling，记录最优 n^c 值为 metadata。运行时根据实际配置查表选择最优 kernel。该方法基于观察：最优 n^c 随输入 token 长度 M 增大而增大（计算负载增长快于通信负载），随 TP 减小而增大（TP 越小每个 GPU 的 expert 越多，通信占比越高）。

从kernel调度角度拆解术语：

```
# Comet Adaptive Assignment 决策流程

# Offline Profiling Phase（部署前执行一次）:
metadata = {}
for M in candidate_token_lengths:         # e.g. [256, 512, ..., 16384]
    for (EP, TP) in candidate_parallelisms: # e.g. [(8,1), (4,2), (2,4)]
        best_nc = None
        best_latency = INF
        for nc in range(0, total_SMs):     # total_SMs = 132 (H800)
            np = total_SMs - nc
            kernel = precompiled_kernels[(nc, np)]
            latency = profile_kernel(kernel, M, EP, TP)
            if latency < best_latency:
                best_latency = latency
                best_nc = nc
        metadata[(M, EP, TP)] = best_nc

# Runtime Phase（每次 MoE layer forward）:
def comet_moe_forward(M, EP, TP):
    key = (M, EP, TP)
    nc = metadata[key]      # O(1) lookup
    kernel = precompiled_kernels[nc]
    kernel.launch(shared_tensor, routing_map, expert_weights)
```

观察到的规律（Figure 8）：
- TP=8, M=4096 → optimal n^c=18；M=16384 → optimal n^c=26
- M=16384, TP=8 → optimal n^c=26；TP=4 → optimal n^c=46
- 解释: 通信和计算的数据量均随 M 线性增长，但各自的 SM 资源需求 scalability 不同——计算需要更多 SM 处理更大的 M，而通信 I/O 饱和带宽后额外 SM 收益递减

术语一般如何实现？如何使用？
- 预编译内核库包含多个 n^c 变体（如 n^c ∈ {18, 26, 46, ...}），每种变体为独立 CUDA kernel
- Profiling 需在目标硬件上进行（不同 GPU 架构的 SM 数、NVLink 带宽、计算能力不同）
- 运行时查表 O(1)，无调度开销
- 局限：无法处理 M 的连续变化（只能匹配 profiled 离散点）；极端 imbalanced token distribution 时最优值可能偏移
- 未来可扩展为 runtime 动态调度的 feedback-based 自适应

涉及论文标题：
- Comet Fine-grained Computation-communication Overlapping for Mixture-of-Experts
