## Computation-Communication Overlap via SM Control (SM控制的MoE计算-通信重叠)

术语解释
在 MoE 分布式推理中，通过限制 GEMM kernel 占用的 SM（Streaming Multiprocessor）数量，将部分 SM 资源留给 all2all 通信 kernel，使二者在不同 SM 上并行执行，实现计算与通信的 pipeline 重叠。

术语是什么？
NVIDIA GPU 上的 all2all 通信需要 SM 资源来驱动（如 NCCL kernel 在 SM 上执行数据打包/解包和网络操作）。默认情况下，GEMM kernel 倾向于占用所有可用 SM（如 H800 上的 132 SM），导致通信 kernel 只能在 GEMM 完成后才能调度。EPS-MoE 的关键发现：对 GroupGemm，在输入 size 达到一定阈值后，减少其占用的 SM 数不会显著影响计算效率（图5c），因此可以限制 GEMM 的 SM 数，留出 SM 给通信 kernel。

从kernel调度角度拆解术语：

```
=== SM 分区策略 (H800, 132 SMs) ===

# 无重叠（baseline）：
CUDA Stream 0: [GroupGemm 132 SMs] → [all2all 132 SMs]
总时间 = T_geMM + T_comm

# 有 SM 控制（EPS-MoE）：
CUDA Stream 0: [GroupGemm 116 SMs]  
CUDA Stream 1: [all2all 16 SMs]
              └── 并行执行 ──┘
总时间 ≈ max(T_geMM' + Δ, T_comm' + Δ), Δ≈0

# 最佳 SM 配置搜索（表6, H800）：
for gemm_sm in range(92, 133, 8):
    for comm_sm in range(1, 133-gemm_sm):
        # 配置 GEMM kernel SM 数
        # 使用 CUDA stream 实现并行
        GEMM_kernel<<<grid, block, shared_mem, stream_A>>>(..., sm_limit=gemm_sm)
        all2all_kernel<<<grid, block, shared_mem, stream_B>>>(..., sm_limit=comm_sm)
        latency = measure_concurrent_execution()
# 最佳配置：GEMM 116 SM + 通信 16 SM（H800 132 SM total）
# GEMM 计算吞吐损失 < 3%，但通信完全隐藏
```

术语一般如何实现？如何使用？
- 基于 CUDA Stream 实现 GEMM 和通信 kernel 的并发提交
- 通过 NVIDIA Nsight Systems 分析 SM 利用率和 kernel 重叠效率
- EPS-MoE 实验表明：通信 kernel 仅需 10-20 SM 即可跑满 NVLink 带宽，GEMM 在 116 SM 时效率仅比 132 SM 降低 <3%
- 与 FP8 通信正交：FP8 减少通信量，SM 控制减少等待时间，两者结合效果最佳
- H800 上最佳配置：GEMM 116 SM + 通信 16 SM

涉及论文标题：
- EPS-MoE: Expert Pipeline Scheduler for Cost-Efficient MoE Inference
