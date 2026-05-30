## Predictive Search for Wave Group Partition

术语是什么？

Predictive search for wave group partition 是 FlashOverlap 中用于在运行前自动搜索最优 wave group partition 的预测搜索算法。它用一个延迟预测器（latency predictor）替代在线 profiling，消除 profiling 开销——online profiling 需要 >1 分钟（>100× 模型推理延迟），不可接受。Predictor 基于 GEMM 配置和 bandwidth curve 以 <5% 的平均预测误差估计每种 partition 的 overlap 后延迟。

从kernel调度角度拆解术语：

Predictive search 算法（FlashOverlap Alg.1）：

```
Input: M, N, K, comm_op, gpu

# Offline
gemm_config = get_config(M, N, K, gpu)  # CUTLASS profiler
bdw_curve = sample_bandwidth(comm_op, gpu)  
T = gemm_config.tile_num / (gpu.sm_num - comm_op.sm_num)

# Online
candidates = get_candidates(T)  # pruned: |G1|≤2, |GP|≤4
t_min = +inf

for G in candidates:
    t_p_acc = 0; t_m_acc = 0
    for i, G_i in enumerate(G):
        data_size = get_data_size(G_{i-1})
        t_m = interp_latency(bdw_curve, data_size)
        t_p = gemm_config.duration / T * |G_i|
        t_m_acc = max(t_p_acc, t_m_acc) + t_m
        t_p_acc = t_p_acc + t_p
    # Last group communication
    t_m_acc = max(t_p_acc, t_m_acc) + interp_latency(bdw_curve, last_data_size)
    if t_m_acc < t_min: t_min = t_m_acc; G_optimal = G

return G_optimal
```
**Annotations**: 预测误差平均 3.41% (RTX 4090), 3.44% (A800)。搜索 partition 达到穷举 >99% 性能。MoE GEMM+A2A 场景中 predictor 取所有 GPU 延迟的 max（因 workload imbalance）。LLM inference 场景 pre-search 代表性 size 后 nearest-neighbor matching。

术语一般如何实现？如何使用？

Offline 阶段用 CUTLASS profiler + 通信采样。Online 阶段对每个新 GEMM size 运行 predictor。LLM training/T2V generation 等 GEMM size 固定场景，tuning 在 runtime 前一次性完成。实现包含在 github.com/infinigence/FlashOverlap 的 evaluation/preparation.py 和 tuning 模块中。

涉及论文标题：
- Efficient and Adaptable Overlapping for Computation and Communication via Signaling and Reordering
