## Offline Profiling for GEMM Micro-kernels

Offline Profiling for GEMM Micro-kernels 是 HyTiS 提出的预先在目标 GPU 上对所有候选 micro-kernel tile 配置进行性能表征的方法。通过一次 profiling 构建 Two-Level Tile Scheduling 所需的 Throughput-Oriented (S^TO) 和 Latency-Oriented (S^LO) 候选集，将运行时 auto-tuning 搜索空间从 O(10^4) 降至 O(10^1)。关键前提：由于 SM 架构独立性，micro-kernel 在单个 SM 上的性能特征在不同 problem shape 下相对稳定，因此 offline profiling 结果可跨 workload 复用。

从编译框架角度拆解：
```
// Profiling workload: P(Ki) = GEMM(bM_i*4, N_SM*n0/4, K_0=1024)
// Collect: t(Ki), SMEM(Ki), REG_spill(Ki)

// TO selection with 3 constraint categories:
// 1) Resource: SMEM(Ki) <= SMEM_0, REG_spill == 0
// 2) SMEM utilization: no larger-dim K' with valid SMEM
// 3) ISA: H100 wgmma requires bM%64==0
T(Ki) = (M_i * N_i) / (n0 * t(Ki))  // throughput
S_TO = {Ki | diff(T(Ki), max(T)) < l1}

// LO selection:
t_wave(Ki) = t(Ki) / n0  // per-wave latency
S_LO = {Ki | diff(t_wave(Ki), min(t_wave)) < l2}
```

Profiling cost: H100 ~19 min, A100 ~36 min（per device + data layout）。H100 更快因 wgmma 指令限制 bM%64==0，显著缩小合法配置空间。Profiling 结果 cached to disk，运行时直接加载。

术语一般实现：在 Triton 3.2.0 上实现。Triton 提供 `@triton.autotune` 进行 per-kernel auto-tuning，但 HyTiS 的 profiling 是 coarse-grained——仅评估 tile size 层面性能，不深入 thread/warp 调度。与 PyTorch Inductor 固定 ~20 配置 vs CUTLASS 手工 precompiled specializations 的区别：HyTiS 用 profiling 自动发现 architecture-aware 最优配置。

涉及论文标题：
- HyTiS: Hybrid Tile Scheduling for GPU GEMM with Enhanced Wave Utilization and Cache Locality
