## RoWild（端到端机器人基准套件）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- RoWild（Robotics in the Wild）是 CMU RoboArch 组的开源、跨平台端到端机器人基准套件，源自 "Agents of Autonomy"（SIGMETRICS/PERFORMANCE 2024，DOI 10.1145/3652963.3655043）。它建模 29 个工业机器人实际软件管线，用有限集通用任务（场景理解、路径规划、状态估计、定位建图等）组合出六个端到端应用：DeliBot（粒子滤波定位 + Raycast）、PatrolBot（Kalman 滤波 + YOLOv10 目标检测）、MoveBot-PRM（Gunrock/Ligra 图处理 + nanoflann 最近邻 + PRM 运动规划 + CCCD 碰撞检测）、MoveBot-RRT（nanoflann + RRT + CCCD）、HomeBot（点云 SLAM）、FlyBot（Octree 碰撞检测 + OSQP MPC + A* 路径规划）。与 kernel 孤立的传统基准不同，它保留真实机器人管线的跨域结构（感知→定位→规划→控制循环），CPU/GPU 基线为平台优化的 state-of-the-art 实现。开源：C++、MIT 许可（https://github.com/cmu-roboarch/rowild ，项目页 https://cmu-roboarch.github.io/rowild/ ）。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 以 FlyBot 的管线为例（论文 Table I）：碰撞检测（Octree 树遍历，data-dependent irregular 指针追逐）→ MPC（OSQP 求解：稠密矩阵乘 + 变长向量更新，fine/structured 并行混合）→ 路径规划（A*：frontier 队列，data-dependent semi-regular）。论文用 RoWild 统计四类访存模式与四类并行度的运行时占比（Fig. 1）：regular 线性代数在 6 个机器人中 4 个不占主导，semi-regular 与 data-dependent 在 3 个中占主导；无单一并行形态主导——这是"加速器多态"的动机证据。指标：端到端 latency（论文用其做加速比与 PPW 的归一化基准）。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 使用：跨平台编译（低端嵌入式 CPU 到高端服务器 GPU），面向系统/硬件研究（评估缓存、预取、向量化）；可模块化配置任务/算法/参数；在 RTRBench 基础上扩展。本文用法：作为 Morphatron 与 ARM/Xeon/Orin Nano/RTX 3090 对比的统一 workload，并把六个应用按算法域映射到五种 morphas。论文指出其局限：只覆盖机器人算法子集、规模中等、未充分代表学习式方法的增长。

涉及论文标题：
- Accelerator Polymorphism: Transcending Domain-Specific Architectures with Robotics
