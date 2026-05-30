## Gradient-AllReduce in MoE Training

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Gradient-AllReduce 是数据并行（DP）中梯度同步的集合通信操作。在混合并行训练中，Gradient-AllReduce 为节点间通信，与 AlltoAll 共享 InfiniBand 带宽。若不加优化，会与 MoE 层的 AlltoAll 争用网络导致额外延迟。FSMoE 的自适应梯度分区将 Gradient-AllReduce 与 MoE 层协同设计，最大化隐藏梯度同步开销。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

FSMoE 两阶段梯度分区算法：

```
# Phase 1: 贪心分配
for each layer i (from last to first):
    t_olp_i = overlappable_time(layer_i)  # MoE层空闲时间
    n_grad_i = g_grad_inv(min(t_grad(remaining), t_olp_i))
    remaining -= n_grad_i

# Phase 2: 差分进化优化剩余梯度
if remaining > 0:
    minimize Σ f_moe^i(t_grad(x_g^i))  # f_moe^i: Algorithm 1
    subject to 0 ≤ x_g^i < n_rem^i + Σ(n_rem^j - x_g^j)
```

性能模型：t_{ar}(n) = α_{ar} + n·β_{ar}，在 Testbed-A 上 α_ar=5.11e-1, β_ar=4.95e-6。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

标准实现为 PyTorch DDP 的 `all_reduce(grad, SUM)`。PipeMoE+Lina 用固定 chunk size (30MB) 切分梯度，但无法适应不同配置。FSMoE 的自适应分区根据各层 overlappable parts 的实际时间动态分配，对比 Tutel-Improved 额外加速 5-7%。

涉及论文标题：
- FSMoE: A Flexible and Scalable Training System for Sparse Mixture-of-Experts Models
- FlowMoE: A Scalable Pipeline Scheduling Framework for Distributed Mixture-of-Experts Training

FlowMoE 将 Gradient-AllReduce 与 MoE 层的 A2A 通信协同调度：将每层 all-reduce 梯度切成 S_p 大小的 chunk，赋予低于 A2A 的优先级，在 A2A 通信间隙填充执行。BO 自动调优 S_p 以平衡重叠增益和系统开销。 (α-β Linear Model)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

α-β 线性模型将通信/计算操作耗时分解为固定启动开销 α 和每字节/每计算单元的可变开销 β：t(n) = α + n·β。按 pipeline degree r 切分后：t_r = α + n/r·β。FSMoE 使用此模型预测不同 r 下的执行时间以确定最优流水线度。拟合精度 R² > 0.998。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Profiling (训练前一次, <100s)
for msg_size in range(2^18, 24·2^18, 2^18):
    for op in [AlltoAll, AllGather, ReduceScatter, AllReduce]:
        t = nccl_test(op, msg_size)
α, β = least_squares(msg_sizes, times)  # <10ms

# 最优r求解 (4 cases, 平均193ms per config)
for c in {1,2,3,4}:
    r_c, t_c = SLSQP(minimize f_c(r), constraints=c)
r_opt = argmin(t_1, t_2, t_3, t_4)
```

Testbed-A 参数：GEMM α=4.26e-2, β=2.29e-11; AlltoAll α=2.87e-1, β=2.21e-7; AllGather α=3.37e-1, β=2.32e-6; ReduceScatter α=3.95e-1, β=2.34e-7; AllReduce α=5.11e-1, β=4.95e-6。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

FSMoE profiler 基于 nccl-tests + torch.matmul 微基准，训练前执行一次。换集群时重新 profiling 一次即可。α-β 模型假设线性关系——在 FSMoE 测量范围内（2^18~12·2^19 float elements）被实验验证。SLSQP 求解器使用 scipy.optimize.minimize(method='SLSQP')，二次收敛速度。

涉及论文标题：
- FSMoE: A Flexible and Scalable Training System for Sparse Mixture-of-Experts Models
