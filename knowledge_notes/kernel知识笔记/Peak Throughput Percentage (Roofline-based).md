## Peak Throughput Percentage (Roofline-based)

术语是什么？
Peak Throughput Percentage（峰值吞吐百分比）是 AccelOpt/NKIBench 使用的 kernel 性能评估指标，定义为 `T / t × 100%`，其中 `t` 是实测 kernel 执行时间（不含编译时间），`T` 是基于 Roofline 模型（Williams et al., 2009）计算的理论最短执行时间。对于 Trainium 加速器，`T = max(Traffic_Min / Bandwidth, FLOPs_MM / Peak_MM, FLOPs_Vec / Peak_Vec)`，分别对应 memory bandwidth bound、tensor engine compute bound、vector engine compute bound 三种硬件瓶颈场景下的理论下限。该指标解决了传统 kernel benchmark 仅衡量相对 speedup 的局限——相对 speedup 的绝对数值取决于 baseline 选择（不同 baseline 得到不同 speedup），而 peak throughput percentage 提供与硬件理论极限比较的绝对性能坐标系。

从kernel调度角度拆解术语：
Peak Throughput Percentage 的计算流程：

```
输入: Kernel 实测 latency t, Input/Output tensor shapes, 硬件峰值参数

Step 1: 计算 Traffic_Min
  Traffic_Min = Σ(size of each input tensor) + Σ(size of each output tensor)
  // 单位为 bytes，仅计算必须的 I/O 数据量
  // 不含 spill、同地址读写等非必须数据搬移

Step 2: 计算 FLOPs_MM (Tensor Engine matmul FLOPs)
  遍历 kernel 中所有 nc_matmul 调用:
    For each nc_matmul(stationary[M×K], moving[K×N]):
        FLOPs_MM += 2 × M × N × K  // 每次乘加算 2 FLOPs

Step 3: 计算 FLOPs_Vec (Vector + Scalar Engine FLOPs)
  遍历 kernel 中所有非 matmul 操作:
    element-wise, activation, transpose, copy 等
    FLOPs_Vec = Σ(每个操作的 FLOPs)
    // Trainium 上 vector engine 和 scalar engine 可并发执行
    // Peak_Vec = Peak_Vector + Peak_Scalar (best case)

Step 4: 计算 Roofline 理论最短时延 T
  T_mem  = Traffic_Min / PeakBW      // memory bound 下限
  T_mm   = FLOPs_MM / Peak_MM        // tensor engine bound 下限
  T_vec  = FLOPs_Vec / Peak_Vec      // vector engine bound 下限
  T = max(T_mem, T_mm, T_vec)        // 三者的 bottleneck

Step 5: 计算 Peak Throughput Percentage
  percentage = T / t × 100%
  // 100% 意味着 kernel 达到硬件理论上限
```

关键设计：峰值假设最优情况——(1) 所有 engine 完全并发、(2) 无 spilling、(3) 无 pipeline bubbles、(4) matmul tile 恰好匹配 optimal configuration（128×128 + 128×512）。因此 >80% 已非常接近实际硬件极限。

术语一般如何实现？如何使用？
在 AccelOpt 中，该指标仅用于评估和 benchmark，未直接注入 agent prompt（论文建议作为 future work）。Roofline 模型的参数从硬件文档获取（Trainium 1: PeakBW=440.2 GB/s, PeakMM=23.75 TFLOPS, PeakVec=286.8 GFLOPS; Trainium 2: PeakBW=640.0 GB/s, PeakMM=19.75 TFLOPS, PeakVec=550.0 GFLOPS）。Traffic_Min 的计算假设理想 cache 命中（所有复用数据均在片上），实际 traffic 可能因 spilling 和 redundant load 而远大于 Traffic_Min。该指标在 NKIBench 的每个 task 中作为 reference value 提供。

涉及论文标题：
- AccelOpt: A Self-Improving LLM Agentic System for AI Accelerator Kernel Optimization
