## Two-Stage Auto-Parallelization for MoE Training (Offline Pool + Online Adaptive / MoE训练的两阶段自动并行化)

术语是什么？
Two-Stage Auto-Parallelization 是 SmartMoE 提出的核心系统架构：将 MoE 模型的自动并行化分解为离线（offline）和在线（online）两个阶段。离线阶段在训练开始前执行，搜索最优的混合并行策略组合（DP/TP/PP/EP）并构建一个"池"（pool）——池内所有候选执行计划共享相同的混合并行配置，仅在 expert placement 上可变。在线阶段在训练运行时执行，轻量级地在池内搜索并切换到适应当前 token 负载分布的 expert placement 方案。这一设计解决了 MoE 训练的根本矛盾：最优并行策略随动态负载变化（需要在线调整），但精确搜索又非常耗时（无法在线执行）。

从系统架构角度拆解术语：
完整两阶段流程：
```
[Offline - 训练前, 数分钟级, 一次性]
1. 读取模型配置 (L MoE layers, E experts, N GPUs) + 硬件拓扑
2. 枚举候选混合并行策略空间：
   遍历 DP × TP × PP × EP 组合 + expert slot 配置
3. Workload-Aware 性能评估：
   对每个候选池，用 gating semantics 估算负载 → 评估性能
4. 选择最优池：
   出口 → Pool: {固定 DP/TP/PP/EP, 可变 expert placement}
   
[Online - 每 iteration, 毫秒级, 周期性]
5. Gate Forward: tokens → gate logits → expert indices
6. Collect History: all-gather per-expert token counts
7. Light-weight Search (每 10 iters):
   ExpertPlacementHybrid(E, N, C[E], M=GPUs_per_node)
   输出: 新 expert→GPU 映射表
8. Switch Decision:
   Δ = (当前方案延迟 - 新方案延迟) / 当前方案
   若 Δ > threshold → 执行切换
9a. 若切换: NCCL All-to-All 交换被移动的 expert 参数 (~20ms)
9b. Compute: 各 GPU 按新 placement 执行 expert FFN
```

离线 vs 在线的设计权衡：
| 维度 | Offline | Online |
|------|---------|--------|
| 搜索空间 | 全部混合并行组合 | 固定 pool 内的 expert placement |
| 算法 | 穷举 + 性能模型 | Greedy/DP/Hybrid (<1ms) |
| 时间限制 | 训练前，分钟级可行 | 每 10 iterations，毫秒级 |
| 切换成本 | 无（训练开始前） | expert 参数 All-to-All (~20ms) |

术语一般如何实现？如何使用？
SmartMoE 基于 FastMoE (PyTorch) 实现。离线阶段：pool search 使用 workload-aware 性能模型（基于 gating semantics）穷举评估候选池 → 选最优池。在线阶段：周期性（可调搜索频率，默认 10 iters）触发 expert placement search → CPU 运行 Greedy/Hybrid 算法（不与 GPU 计算竞争）→ 结果通过 NCCL 通信交换参数。三个关键配置参数需手动调优：搜索频率、切换阈值（threshold of switching overhead）、历史收集频率。安装使用：https://github.com/zms1999/SmartMoE。Artifact: https://github.com/MachineLearningSystem/23ATC-SmartMoE-AE。

涉及论文标题：
- SmartMoE Efficiently Training Sparsely-Activated Models through Combining Offline and Online Parallelization
