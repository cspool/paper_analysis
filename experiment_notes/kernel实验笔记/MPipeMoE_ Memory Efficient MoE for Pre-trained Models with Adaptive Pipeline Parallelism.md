## MPipeMoE: Memory Efficient MoE for Pre-trained Models with Adaptive Pipeline Parallelism

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - MPipeMoE 在 MoE 训练中对三种 CUDA stream（computation / communication / memory copy）进行运行时调度，核心调度机制包括：
    - **Micro-batch pipeline 调度**：将 MoE 层的 All-to-All dispatch (S)、Expert FFN (C)、All-to-All collect (R) 三种 kernel 以 pipeline 方式跨 CUDA stream 调度。S 和 R 交替执行以增强 memory access locality，C 独立在 computation stream 上运行。Pipeline granularity n 决定 micro-batch 的大小，影响 kernel launch 频率和 GPU 利用率。
    - **Interference profiling（Figure 3）**：通过 micro-benchmark 量化三种操作并行执行时的相互 slowdown 因子：μ（通信受其他 stream 干扰的 slowdown）、σ（计算的 slowdown，实测 ≈1，即计算几乎不受影响）、η（memory copy 的 slowdown）。实验发现通信与 memory copy 并行时因带宽竞争导致显著 slowdown，而通信与计算重叠可行（μ_comm > 0.5, σ_comp > 0.5）。
    - **Pipeline paradigm 性能建模**：将 pipeline 划分为 5 个阶段（P0-P4），每个阶段由瓶颈 stream 的执行时间决定。执行时间 C = (1/W_comp) * max(q1, q2*α/μ, q3*β/η)，其中 α=W_comp/W_comm, β=W_comp/W_mem，Q=[q1,q2,q3] 为各类型的操作量。
    - **4 种 memory reuse scheduling 策略的 kernel 调度差异（Table II, Figure 7）**：
      - S1: forward 3 条 stream（comp + comm + mem D2H），backward 3 条 stream（comp + comm + mem H2D）
      - S2: forward 3 条 stream，backward 3 条 stream（额外通信恢复 T_DI）
      - S3: forward 3 条 stream，backward 3 条 stream（额外重计算恢复 T_M）
      - S4: forward 2 条 stream（comp + comm，无 memory copy），backward 3 条 stream（额外通信 + 重计算）
    - **Adaptive strategy selection**：基于 Eq 10 性能模型，在运行时根据 N（GPU 数量）和 B（batch size）选择开销最小的 (n, S) 组合。
  - 实验比较：
    - 通信效率 micro-benchmark：对比 FasterMoE（按 node 切分→P2P）vs MPipeMoE（按 batch 切分→保留 All-to-All）在不同 pipeline granularity n 下的 dispatch/recovery 时延。
    - Pipeline granularity 敏感度分析：不同 n（1/2/4/8/16）在不同 B（2k-32k）下的训练时间（Figure 12）。
    - 内存复用策略 S1-S4 在不同 (N, B) 组合下的 overhead 对比（Figure 13）。
    - 性能分解（Figure 11）：在 memory-time 坐标系下，PipeMoE(n=4) / PipeMoE(adaptive n) / MPipeMoE 的 trade-off。

- 后端平台是什么，配置是什么。
  - 8 台 NVIDIA DGX A100 服务器，每节点 8×A100 SXM 40GB GPU（共 64 GPU），200 Gbps HDR InfiniBand，节点内 NVLink 3.0 + NVSwitch。
  - CUDA 11.1、NCCL（All-to-All collective operator）、PyTorch 1.9.0。

- 评估性能的软件/脚本是什么。修改了什么。
  - 基于 PyTorch 1.9.0 + CUDA 11.1 实现自定义 MoE 训练库。修改了 MoE layer 的 forward/backward，引入多 CUDA stream 管理（computation stream、communication stream、memory copy stream）。
  - 自定义 micro-benchmark 用于测量 W_comp、W_comm、W_mem 的 piecewise 速度（区分小/大 volume 的不同硬件利用率）以及 μ、σ、η 干扰因子（Figure 3, Figure 9）。
  - Gating network 默认 top-1 routing（k=1），使用 NCCL All-to-All collective operator 进行 token dispatching/collecting。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源：https://github.com/whuzhangzheng/MPipeMoE
  - 评估原理（kernel 输入到性能输出全过程）：

```
// ===== Interference Profiling Micro-benchmark (Section II-C) =====
// 目的：测量 W_comp, W_comm, W_mem 及 slowdown 因子 μ, σ, η
// 原理：在 3 条 CUDA stream 上分别运行 GeMM kernel / NCCL All-to-All /
//       cudaMemcpy D2H，测量单独执行和并行执行的 wall-clock time

输入: 模型配置 (M, H, B), GPU 拓扑
输出: W_comp(B), W_comm(B), W_mem(B) piecewise速度,
      μ(干扰源类型), σ(干扰源类型), η(干扰源类型)

stream_comp:  launch GeMM kernel (M×H × B tokens)
stream_comm:  launch NCCL All-to-All (B×M bytes)
stream_mem:   launch cudaMemcpy D2H (B×M bytes)

// 单独执行
W_comp = FLOPs / t_comp_alone
W_comm = Bytes / t_comm_alone
W_mem  = Bytes / t_mem_alone

// 并行执行（测量 slowdown）
// e.g., comp + comm 并行:
t_comp_parallel = 在 comp+comm 并行中测得 comp kernel 实际耗时
t_comm_parallel = 在 comp+comm 并行中测得 comm kernel 实际耗时
μ_comm = W_comm * t_comm_alone / (Bytes / t_comm_parallel)
        // μ_comm < 1 表示通信因并行而减速
σ_comp = 类似定义
η_all  = 三流并行时 memory copy 的 slowdown

// Piecewise 速度函数（Figure 9 profile 结果）：
// 小 volume: GPU 利用率低，throughput 随 volume 线性增长
// 大 volume: GPU 饱和，throughput 稳定在峰值
// V_threshold_comp 为 GeMM 饱和阈值，V_threshold_comm 为 All-to-All 饱和阈值

// ===== Pipeline 执行时间模型 (Eq 10) =====
// 以 strategy S4 为例（Q_fw=[2,2,0], Q_bw=[5,3,0]）
// Forward pass: 2 个 GeMM + 2 个 All-to-All，无 memory copy
// Backward pass: 5 个 GeMM + 3 个 All-to-All，无 memory copy
// μ_all = μ_comp（仅 comp+comm 并行场景，无 mem stream）

b = B/n  // micro-batch size
v0_comp = b * H * M     // Eq 7: GeMM FLOPs per micro-batch
v0_comm = b * M         // Eq 8: All-to-All bytes per micro-batch
v0_mem  = b * M         // Eq 9: D2H/H2D bytes per micro-batch

// Forward 阶段执行时间
T_comp_fw = q1_fw * v0_comp / (σ * W_comp(b))
T_comm_fw = q2_fw * v0_comm / (μ * W_comm(b))
T_mem_fw  = q3_fw * v0_mem  / (η * W_mem(b))
T_fw = max(T_comp_fw, T_comm_fw, T_mem_fw)

// Backward 同理
T_bw = max(T_comp_bw, T_comm_bw, T_mem_bw)

// 端到端时间（n 个 micro-batch pipeline）
T_total ≈ max(T_fw, T_bw) * n  // 瓶颈 stream 决定

// 选择 T_total 最小的 (n, S) 组合作为运行时最优配置

// ===== 实际调度 =====
// 在 Python API 层面通过设置 pipeline=True, memory_reuse=True 启用
import pmoe
moe_layer = pmoe.MoELayer(d_model=1024, pipeline=True, memory_reuse=True)
```

  - 关键 kernel 调度结果：
    - 计算几乎不受其他 stream 影响（σ≈1），可与通信安全重叠。
    - 通信受并行计算影响但 slowdown 可接受（μ_comm > 0.5），重叠可行。
    - 通信与 memory copy 并行时因 PCIe/NVLink 带宽竞争导致显著 slowdown（η_all 较小），S2 在 N 大时性能差即因此（Figure 13）。
    - N 小时（8 GPU）S1/S2 更优（I/O bound 场景可容忍额外 memory copy）；N 大时（64 GPU）S4 更优（避免 memory bandwidth 竞争，重计算开销可被通信瓶颈掩盖）。
    - B 变化对策略选择不敏感（Figure 13），但 n 需要随 B 自适应调整（Figure 12）。
