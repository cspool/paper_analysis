## Hardware-Aware Balance Planning（硬件感知均衡规划）

术语是什么？
Hardware-Aware Balance Planning 是 PROBE 中基于运行时约束的 expert 复制与 token 分配联合优化求解器。与忽视传输成本的通用求解器不同，该 planner 严格将 expert 复制决策限制在设备特定的"hiding window"内——即每个 rank 上非通信 kernel（Attention、Grouped GEMM）的执行时间窗口。优化目标：min_P,A max_r (T_comp^r + T_comm^r)，约束条件：T_trans^r ≤ T_window^r（prefetch latency 不超过 overlap window）。

从系统架构角度拆解术语：
Greedy Planning Solver 的执行流程（Algorithm 1 的核心逻辑）：
```
输入: 预测的 per-expert workload n̂, baseline placement P'
输出: 最终 placement P, token assignment A

1. 初始化 A = locality-first routing（remote token 尽可能发往本地 replica）
2. L = 计算各 rank 延迟向量
3. loop (kmax=16):
   a. r_src = argmax_r L_r      // 识别 bottleneck rank
   b. r_dst = argmin_r L_r      // 识别 helper rank
   c. e* = r_src 上最重的 expert
   d. 双向 budget check: T_trans(r_src, r_dst, e*) ≤ min(T_window^src, T_window^dst)
   e. Water-filling rebalance: 将 remote token 从 r_src:e* 重定向到 r_dst:e*_replica
   f. Δ_src^out ∪= {e*}, Δ_dst^in ∪= {e*}
   g. 更新 A 和 L
   h. 若 gain ≤ ε 或 k ≥ kmax → break
```
硬件感知体现在：T_window^r 按设备计算-带宽比动态确定——高算力低带宽设备在小窗口中只能传输少量 expert，从而限制复制数量。

术语一般如何实现？如何使用？
实现为单 SM CUDA kernel，串行迭代更新，hard cap kmax=16。在 PROBE 中通过 NVSHMEM symmetric memory 读取/写入 placement 表。内存开销限制：每 rank 最多 3 个冗余 expert，double buffering 6 slots。Planning 延迟被 MoE Compute 完全隐藏。

涉及论文标题：
- PROBE: Co-Balancing Computation and Communication in MoE Inference via Real-Time Predictive Prefetching
