## Input-Dependent Computational Graph

术语是什么？
Input-Dependent Computational Graph（输入依赖的计算图）是指程序执行过程中 GPU kernel 之间的依赖关系（哪些 kernel 依赖哪些 kernel 的输出）由运行时输入决定、不可提前静态获知的计算图。这对于动态神经网络（如 InstaNAS，根据输入图像决定网络架构路径）和深度强化学习物理仿真（如 Brax，每次仿真的接触/碰撞计算路径取决于 agent 与环境的交互结果）是固有特性。Input-dependent 计算图对 GPU 并发 kernel 调度构成根本挑战：传统方法需要预先知道完整计算图（如 CUDA Graph 需提前构建 DAG，静态调度需提前分析图结构），而 input-dependent 场景中每次输入都需要重新分析和调度，开销不可接受。

从系统架构角度拆解术语：
Input-dependent 计算图的运行时行为：
```
固定输入A:                      不同输入B:
K0 ──► K1 ──► K3               K0 ──► K2 ──► K4
  │                    vs        │             │
  └──► K2 ──► K4                └──► K1 ──► K3

时间 T1 (处理输入A):            时间 T2 (处理输入B):
- CPU获知输入A                   - CPU获知输入B
- 确定K0→K1→K3和K0→K2→K4       - 确定K0→K2→K4和K0→K1→K3
- 需要调度K1||K2并发            - 需要调度K2||K1并发
- DAG结构不同, 需要重新分析      - DAG结构不同, 需要重新分析
```

ACS 通过避免提前构建完整 DAG 来解决此问题：运行时仅检查调度窗口内 N 个 kernel 的依赖关系（而非整个计算图），每次 kernel 完成时增量更新窗口内 kernel 的依赖状态。这相当于将"全局 DAG 分析"问题降维为"局部依赖检测"问题，使得每次输入变化时的额外开销从完整 DAG 构建+调度（占 47% 执行时间）降为微秒级的依赖检查（410-1640ns）。

术语一般如何实现？如何使用？
Input-dependent 计算图的主要存在场景：(1) 动态神经网络（InstaNAS、Dynamic Routing、Conditional Convolution 等），网络执行路径由输入决定，每次推理可能有不同的 kernel 序列；(2) 深度 RL 物理仿真（Brax、Isaac Gym），碰撞检测和物理求解的计算路径取决于 agent 行为；(3) 自适应精度推理（提前退出网络），根据中间结果决定是否继续或提前返回。解决 input-dependent 图的通用方法包括：运行时 JIT 编译（开销大）、DAG 缓存（命中率不可控）、ACS 的窗口调度（通用性好且开销低）。

涉及论文标题：
- ACS Concurrent Kernel Execution on Irregular, Input-Dependent Computational Graphs
