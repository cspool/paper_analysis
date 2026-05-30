## Wave Quantization

Wave Quantization（波量化问题）是 GPU 上执行 GEMM 等分块计算时出现的负载均衡问题。GPU 将 output tensor 划分为多个均匀 tile，每个 tile 分配给一个 SM。所有 tile 分组为 wave，每个 wave 包含 N_SM 个 tile 在 N_SM 个 SM 上并发执行。当总 tile 数不能被 N_SM 整除时，最后一个 wave（partial wave）中仅部分 SM 有工作，其余 SM 空闲——造成硬件利用率下降和性能陡降（performance cliff）。HyTiS 在 H100 上实测：M 维度微增（640→704 或 1664→1728）时 cuBLAS 性能骤降 36% 和 21%，即 wave quantization 导致的 partial wave SM 利用率不足。

从kernel调度角度拆解：给定 GEMM M×N×K 和 tile size bM×bN，total_tiles = ceil(M/bM) × ceil(N/bN)；full_waves = total_tiles / N_SM（整数除）；partial_tiles = total_tiles % N_SM。Homogeneous tile scheduling（cuBLAS）中所有 SM 用相同 micro-kernel，partial wave 仅 partial_tiles 个 SM 忙碌，其余 N_SM - partial_tiles 闲置。缓解策略：(1) 减小 tile size——增加 total_tiles 降低 partial ratio，但 full wave 中 compute-to-memory ratio 下降；(2) Split-K——沿 K 维拆分 tile 到更多 SM，引入 reduction sync；(3) Stream-K——skewed workload partition + fixup ops，额外 workspace 开销（比 cuBLAS 多 70%+ device memory）；(4) HyTiS——两级 tile scheduling，TO kernel 处理 full waves，LO kernel 处理 partial wave，零同步开销。

伪代码（HyTiS 解决 wave quantization 的搜索）：
```
for K1 in S_TO:
    total_tiles = ceil(M/K1.bM) * ceil(N/K1.bN)
    full_waves = total_tiles / N_SM
    partial_tiles = total_tiles % N_SM
    for K2 in S_LO:
        if partial_tiles == 0: plan = (K1, null)
        elif partial_tiles <= N_SM: plan = (K1, K2)
        else: invalid
return argmin(execution_latency(plan))
```

术语一般实现方式：混合 tile size 调度（HyTiS）、K 维拆分（Split-K/Stream-K）、kernel fusion 并发执行小 kernel（POD-Attention, HFuse）、wave-aware auto-tuning search。HyTiS 在 H100 上量化显著区 speedup 1.10-1.19× over cuBLAS/Inductor-Triton。

涉及论文标题：
- HyTiS: Hybrid Tile Scheduling for GPU GEMM with Enhanced Wave Utilization and Cache Locality
