## Integer Linear Programming (ILP) for Kernel Load Balancing

术语是什么？
Kitsune 使用 Integer Linear Programming（ILP，整数线性规划）为 spatial pipeline 中每个 stage（算子）分配最优的 CTA 数量。ILP 的目标是最大化 sf-node（spatially-fused subgraph）的吞吐量，约束包括：每 stage 的 CTA 数量、DRAM 带宽上限、L2 带宽上限、SIMT 和 Tensor 类型 CTA 的独立 SM 分配（利用异构重叠）。通过 zero-latency performance model 估计每个 stage 在不同 CTA 分配下的吞吐量。

从编译框架角度拆解术语：
Kitsune ILP formulation（Algorithm 2）：

```
maximize Throughput
subject to:
  // Stage性能约束: 每stage吞吐量不超过该stage的估计能力
  Throughput < t_i × ResourceScale(a_i) × Speedup(a_i)  (for i = 1..n)
  
  // 带宽约束: 总DRAM和L2访问不超过硬件上限
  Throughput × DRAM_Bytes < DRAM_Bandwidth
  Throughput × L2_Bytes < L2_Bandwidth
  
  // CTA分配范围
  1 ≤ a_i ≤ #SMs  (for i = 1..n)
  
  // 异构重叠: SIMT和Tensor CTA可独立分配到SM
  Σ IsSimt_i × a_i = #SMs
  Σ IsTensor_i × a_i = #SMs

其中:
  t_i = 算子的实测BSP吞吐量 (baseline measurement)
  ResourceScale(a_i) = 基于CTA数量的性能scaling估计
  Speedup(a_i) = 1/u_i, 其中u_i = 算子SIMT或Tensor pipeline的最大资源利用率
                   (因operands从DRAM搬到on-chip queue带来的加速)
  a_i = 分配给stage i的CTA数量
  IsSimt_i / IsTensor_i = 布尔标志: stage i主要使用SIMT还是Tensor Core
```

关键设计决策：
- **Zero-latency model**: 使用简单的解析性能模型（非cycle-level simulation），通过吞吐量估计避免复杂时序模拟。
- **Speedup(a_i) 参数**: 通过在真实GPU上测量 kernel 的 BSP 吞吐量 + 资源利用率（NSIGHT Compute）获得。在实际部署中需要 two-pass compiler、runtime optimization pass 或 kernel characteristics dictionary。
- **异构重叠**: IsSimt 和 IsTensor 的独立求和约束利用了论文的关键洞察——SIMT-heavy 和 TensorCore-heavy CTA 可以在同一 SM 上并发执行而无性能退化。

以 MeshGraphNets MLP 为例：
```
n=3 (Linear1, ReLU, Linear2), #SMs=108
t_Linear1=100 TFLOPS (BSP实测), t_ReLU=50 GB/s (memory bound), t_Linear2=80 TFLOPS
IsSimt = {0, 1, 0}, IsTensor = {1, 0, 1}

ILP求解: a_Linear1=64, a_ReLU=44, a_Linear2=44
  验证: Σ IsSimt_i × a_i = 0×64 + 1×44 + 0×44 = 44 ≤ 108
        Σ IsTensor_i × a_i = 1×64 + 0×44 + 1×44 = 108 ≤ 108
  分析: stage 0和2的Tensor CTA使用不同SM, stage 1的SIMT CTA与stage 0的Tensor CTA重叠colocate
```

术语一般如何实现？如何使用？
ILP 求解使用标准 ILP solver（如 Gurobi、CPLEX 或开源替代如 GLPK）。Kitsune compiler 在 Load Balance 阶段（§5.3）调用 ILP solver，输入为 sf-node 的 pipeline graph + hardware parameters（#SMs, DRAM bandwidth, L2 bandwidth）+ 预测量的 kernel characteristics（t_i, ResourceScale, Speedup）。输出为每个 stage 的 CTA 分配数 a_i，直接用于 cudaPipeline API 的 kernel launch 配置。

涉及论文标题：
- Kitsune: Enabling Dataflow Execution on GPUs

---
