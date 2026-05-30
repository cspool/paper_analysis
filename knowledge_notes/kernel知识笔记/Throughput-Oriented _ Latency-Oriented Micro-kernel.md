## Throughput-Oriented / Latency-Oriented Micro-kernel

Throughput-Oriented (TO) 和 Latency-Oriented (LO) Micro-kernel 是 HyTiS 中两类优化目标不同的 GEMM tile 计算单元。TO micro-kernel 用 large tile size（高 compute-to-memory ratio），目标在 full wave 中最大化 SM 吞吐量；LO micro-kernel 用 small tile size（低 per-wave latency），目标在 partial wave 中最小化执行时间。两者通过 offline profiling + constraint filtering 构建候选集。

从kernel调度角度拆解，offline profiling + selection 流程：
```
// Profiling: representative GEMM P(Ki) = (bM*4) x (N_SM*n0/4) x 1024
T(Ki) = (M_i * N_i) / (n0 * t(Ki))  // throughput metric

// TO selection (3 constraints):
// 1) SMEM(Ki) <= SMEM_0, REG_spill(Ki) == 0
// 2) not exists K' with larger dims and valid SMEM (SMEM utilization)
// 3) ISA constraints (H100: bM%64==0 for wgmma)
K_opt_TO = argmax(T(Ki))
S_TO = {Ki | diff(T(Ki), T(K_opt_TO)) < l1}

// LO selection:
t_wave(Ki) = t(Ki) / n0  // per-wave latency
K_opt_LO = argmin(t_wave(Ki))
S_LO = {Ki | diff(t_wave(Ki), t_wave(K_opt_LO)) < l2}
```

H100 上 wgmma 要求 bM%64==0 显著缩小搜索空间（平均 ~14 vs Inductor-Triton 固定 19）。l1 与 problem size 相关：vtiles（output tiles / 64×64）增大→l1→1。分段函数：vtiles<2500→l1=1.2，2500-5000→l1=1.1，>5000→l1=1.05。l2 固定 1.3。

术语一般实现：Triton 上定义 bM/bN/bK + thread block layout，复用 Triton intra-tile 优化（memory coalescing, swizzling, SMEM alloc, MMA）。Offline profiling 每 GPU + data layout 一次（H100 ~19 min, A100 ~36 min），结果跨 problem shapes 可复用（SM 架构独立性保证 micro-kernel 性能特征稳定性）。

涉及论文标题：
- HyTiS: Hybrid Tile Scheduling for GPU GEMM with Enhanced Wave Utilization and Cache Locality
