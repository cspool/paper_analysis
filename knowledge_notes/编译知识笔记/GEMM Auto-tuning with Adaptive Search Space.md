## GEMM Auto-tuning with Adaptive Search Space

GEMM Auto-tuning with Adaptive Search Space（自适应搜索空间的 GEMM 自动调优）是 HyTiS 的运行时优化机制：对每个 GEMM problem shape，在 offline profiling 构建的 TO×LO 候选组合空间中进行 auto-tuning，选择最优 (K1, K2, layout) 组合。搜索空间是 architecture-aware（profiling 决定）且 workload-adaptive（l1/l2 阈值动态调整）。H100 平均搜索空间 14（max 66），A100 平均 16（max 77） vs Inductor-Triton 固定 19。

从编译框架角度拆解：
```
function autotune(P(M,N,K)):
    if cache_hit(P): return cached
    for K1 in S_TO:
        total_tiles = ceil(M/K1.bM) * ceil(N/K1.bN)
        partial_tiles = total_tiles % N_SM
        for K2 in S_LO:
            if partial_tiles == 0: valid (TO-only)
            elif partial_tiles <= N_SM: valid (two-level)
            else: skip
            for layout in [GM_opt, GN_opt]:
                bench(HyTiS_GEMM, K1, K2, layout, P)
    best = argmin(latency)
    cache.store(P, best)
    return best, n1_wave, n2_tiles
```

l1 阈值随 vtiles 增大递减（1.2→1.1→1.05），l2 固定 1.3。Caching 消除重复 Triton JIT compilation（相同 K1/K2/layout 跨不同 problem shape 间共享 kernel binary）。HyTiS Scheduler 不修改 Triton compiler，仅在 Triton 上层做 scheduling 决策。

涉及论文标题：
- HyTiS: Hybrid Tile Scheduling for GPU GEMM with Enhanced Wave Utilization and Cache Locality
