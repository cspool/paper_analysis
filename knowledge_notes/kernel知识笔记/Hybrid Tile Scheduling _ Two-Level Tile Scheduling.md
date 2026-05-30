## Hybrid Tile Scheduling / Two-Level Tile Scheduling

Hybrid Tile Scheduling（混合 tile 调度，亦称 Two-Level Tile Scheduling）是 HyTiS 的核心调度策略。核心思想：GPU GEMM 中不同 wave 有不同优化目标——full waves 硬件资源充足应最大化吞吐量，partial wave 硬件资源富余但 workload 不足应最小化延迟。因此采用两级不同 micro-kernel：Level-1 用 Throughput-Oriented (TO) large-tile micro-kernel 处理 full waves，Level-2 用 Latency-Oriented (LO) fine-grained micro-kernel 处理 partial wave。

从kernel调度角度拆解，HyTiS Algorithm 1 伪代码：
```
function HyTiS_GEMM(a, b, c, K1, K2, n1_wave, n2_tiles):
    pid = blockIdx.x
    k_tiles = ceil(K / K1.bK)
    // Level-1: full waves (TO)
    for i = 0 to k_tiles * n1_wave:
        ki = i % k_tiles; tid = pid
        if ki == 0:
            offs_m, offs_n = l1_offset_fn(tid)
            ta, tb = Load(a, offs_m), Load(b, offs_n)
        tc += K1.compute(ta, tb, tc)
        if ki == k_tiles - 1:
            store(tc, offs_m, offs_n); tid += N_SM
    // Level-2: partial wave (LO)
    if pid >= n2_tiles: return
    for i = 0 to ceil(K / K2.bK):
        offs_m, offs_n = l2_offset_fn(tid)
        ta, tb = Load(a, offs_m), Load(b, offs_n)
        tc = K2.compute(ta, tb)
        store(tc, offs_m, offs_n)
```
特殊 case：(1) TO-only——无可行 LO 候选，partial wave 也用 TO kernel；(2) LO-only——问题太小无 full wave，全用 LO kernel。搜索空间从 O(10^4) 降为 O(10^1)（offline profiling 构建 S_TO 和 S_LO 候选集，每个~10 个）。对比 greedy（直接选最优 TO+最优 LO）：hierarchical scheduling 考虑全局最优配对。

术语一般实现：在 Triton 3.2.0 上实现，Hopper 用 persistent kernel + TMA 消 CTA launch 开销，Ampere 用 data-parallel launch（TMA 不支持 + persistent kernel register 压力大）。与 Split-K/Stream-K 的区别：HyTiS 不沿 K 维拆分，零 reduction sync + 零额外 workspace。

涉及论文标题：
- HyTiS: Hybrid Tile Scheduling for GPU GEMM with Enhanced Wave Utilization and Cache Locality
