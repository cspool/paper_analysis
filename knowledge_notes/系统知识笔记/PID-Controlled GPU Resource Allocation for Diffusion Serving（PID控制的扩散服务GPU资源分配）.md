## PID-Controlled GPU Resource Allocation for Diffusion Serving（PID控制的扩散服务GPU资源分配）

术语是什么？

PID-Controlled GPU Resource Allocation是MoDM提出的面向扩散模型混合serving的GPU资源动态管理方法。系统使用PID (Proportional-Integral-Derivative) 控制器根据实时request rate、cache hit rate和去噪步数分布，动态调整分配给大模型和小模型的GPU Worker数量比例，在满足throughput约束的前提下最大化大模型数量（Quality-Optimized Mode）或最大化throughput（Throughput-Optimized Mode）。

从系统架构角度拆解术语：

PID-Controlled GPU Allocation在MoDM中的运转流程（Algorithm 1）：

```
Global Monitor每周期执行:
1. 统计metrics: R(请求率), H_cache(命中率), k_rates(去噪步数分布)
2. 计算miss_workload = (1-H_cache) x R
3. 计算hit_workload = H_cache x R x Sum P(K=k)x(1-k/T)

Quality-Optimized Mode:
4. num_large = ceil(miss_workload / P_large)  // 最小大模型数
5. while num_large <= N:  // 迭代增加大模型数
       available = num_largexP_large - miss_workload + (N-num_large)xP_small
       if available >= hit_workload: num_large++ else break

Throughput-Optimized Mode:
4. hit_workload_weighted = hit_workload x P_large / P_small
5. num_large = [miss_workload/(hit_workload_weighted+miss_workload)] x N

6. Delta_large = PID(num_large, current_num_large)  // PID平滑调整
7. N_large = max(1, min(round(current_num_large + Delta_large), N))
```

PID在此并非用于连续过程控制，而是作为"平滑器"——heuristic提供快速初始分配估计，PID在此基础上添加incremental adjustment以阻尼rapid changes、防止模型worker频繁切换导致的资源震荡。

术语一般如何实现？如何使用？

在MoDM中Global Monitor以Python实现，PID使用标准discrete PID公式。参数(Kp=0.6, Ki=0.05, Kd=0.05)通过实验tuned。heuristic+PID的组合策略：heuristic快速响应load变化，PID防止oscillation。在16xMI210实验中，对22 req/min以上动态负载切换稳定有效。

涉及论文标题：
- MoDM: Efficient Serving for Image Generation via Mixture-of-Diffusion Models

---
