## Pipeline Paradigm and Performance Model for Multi-Stream GPU Execution

术语是什么？
Pipeline Paradigm 是 MPMoE 为不同类型的内存复用策略抽象出的 3 种流水线执行模式，每种模式由不同的 CUDA stream 依赖关系定义。Performance Model 基于这些 paradigm 和 piecewise 速度函数，在运行时估算不同配置 (n, S) 的端到端执行时间，实现无需 profiling 的自适应配置选择。

从kernel调度角度拆解术语：
MPMoE 的 Performance Model 构建分三步：

```
// Step 1: Piecewise 速度函数（Figure 9）
// 小数据量时硬件利用不足，速度低于峰值；大数据量时速度饱和
W_comm(volume) = {
    k1_comm * volume,  if volume < V_threshold_comm  // 线性增长段
    k2_comm * volume,  otherwise                      // 饱和段 (k2<k1)
}
// 通过一次微基准 profiling 获得分段参数

// Step 2: Pipeline Paradigm 时间估算（Figure 8）
// Paradigm 1（仅 S+C+R，适用 S4 全阶段）:
//   P0: 仅 1 个 stream 工作，初始阶段
//   P1: 逐步启动所有 stream，饱和过渡阶段
//   P2: 所有 stream 饱和运行（可能有多个 P2）
//   P3: 逐步关闭部分 stream，熔化阶段
//   P4: 仅 1 个 stream 收尾
// 每个阶段执行时间 = 该阶段瓶颈 stream 的执行时间 / α(干扰因子)

// 以 Paradigm 1 的 P2 阶段为例:
t_S = W_comm(B/n)  // 单个 micro-batch 的 dispatch 时间
t_C = W_comp(B/n)  // 单个 micro-batch 的 expert 计算时间
t_R = W_comm(B/n)  // 单个 micro-batch 的 collect 时间
T_P2 = max(
    (t_S + t_R) / α(comm, comp),  // 通信流的瓶颈
    t_C / α(comp, comm)            // 计算流的瓶颈
)

// Step 3: 总时间汇总
T_total(n, S) = T_P0 + T_P1 + (n-3) * T_P2 + T_P3 + T_P4  // n 够大时
```

MPMoE-pm 通过此模型在无 profiling 开销下（<1% overhead）估算所有 (n, S) 组合的耗时，选择 T_total 最小的配置。

术语一般如何实现？如何使用？
- 适用场景：(a) 网络环境稳定时（如 Valor 集群），performance model 可替代 profile-based search；(b) 生产环境中避免每次配置变更都 profiling；(c) 作为 profile-based 方法的 warm-start（先用 model 估算，再用 profiling 微调）。
- 局限性：(a) 网络波动大时（如 Adira 集群）模型精度下降，MPMoE-pb 更优；(b) 依赖 α 因子的准确性，不同 GPU 架构和 NCCL 版本可能需要重新 calibrate；(c) 对极细粒度 pipeline（n>8）的 kernel launch overhead 建模不够精确。
- MPMoE-pm 的效果：比 MPMoE-pb 平均速度损失约 6-7%（1.66× vs 1.55× vs FasterMoE），但 profiling overhead 从 ~3% 降至 <1%。

涉及论文标题：
- MPMoE: Memory Efficient MoE for Pre-Trained Models With Adaptive Pipeline Parallelism
- MPipeMoE: Memory Efficient MoE for Pre-trained Models with Adaptive Pipeline Parallelism

---
