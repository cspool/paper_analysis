## List Scheduling（列表调度：资源受限调度的经典贪心启发式）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
List Scheduling 是资源受限调度（RCS）问题的经典贪心启发式，广泛用于 HLS（高层次综合）调度与并行编译器指令调度：对数据流图/依赖 DAG（节点=操作、边=依赖），每步维护 ready list（全部前驱已调度的节点集合），按优先级函数排序，贪心地把尽可能多的就绪节点分配进当前控制步（受该步可用功能单元数 K 约束），资源不足的高优先级外的节点推迟到下一控制步，直至所有节点调度完成。配套概念：ASAP（As Soon As Possible，不约束资源的最早调度，给出下界/关键路径长）与 ALAP（As Late As Possible，给定时钟约束下的最晚调度）；mobility = ALAP−ASAP 为 0 的节点在关键路径上。常见优先级：mobility 最小优先（关键路径优先）、ALAP/ASAP 值、Force-Directed（FDLS）、累计后继权重（2023 年 DSS 期刊证明 list scheduling 存在最优排序并提出 accumulated-weight 优先级）。RCS 与 LCS（time-constrained）均 NP-hard，list scheduling 是多项式时间的近优启发式（不保证最优，精确解需 ILP）。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
DIAMoND 的用法（把 HLS 调度搬进存储计算部署）：输入是"修改后的 Transformer 数据流图"——去掉不在 in-NAND 执行的算子（如 W_Q），顶点 = 各权重矩阵 V_i、边 = 矩阵间数据依赖；调度目标是把矩阵按 cycle 分组部署到 NAND 平面的 OU 资源上（每 cycle 可用 OU 数受限）。流程：ready list 收集依赖已满足的矩阵 → 按拓扑序优先级贪心逐 cycle 调度（只要 OU 资源充足就放入当前 cycle）→ 同 cycle 矩阵的子矩阵集拼接后经 Round-Robin + mask 引导联合映射到多个平面 → 同一平面上的专家集合形成 Expert Group（供 AES 的 mask 状态寄存器管理）。效果：平衡多 plane 的负载、确定每层 FFN 需要的 read cycle 数（AES 生效后 Up/Gate/Down 各 1 cycle = 3 cycles）。
```
LIST_SCHEDULE(V, E, K):        # V=权重矩阵, E=依赖, K=每cycle OU资源上限
    cycle <- 0; scheduled <- {}
    while |scheduled| < |V|:
        ready <- {v in V\scheduled : all predecessors of v in scheduled}
        ready.sort(by topological priority)      # 拓扑序/关键路径优先
        picked <- ready[0..min(K,|ready|)-1]     # 贪心占用资源
        place picked into cycle                  # 子矩阵拼接后映射 OU
        scheduled <- scheduled + picked; cycle <- cycle + 1
```
调度结果示例（Fig.10）：Mixtral 单层内 QKV 投影（除 W_Q）、attention 相关矩阵与 8 个专家的 FFN 矩阵被分到若干 cycle，同 cycle 矩阵共用平面资源。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：HLS 工具链（Bambu/PandA、Vivado HLS 等）内建 ASAP/ALAP/list/FDLS 调度；工业编译器指令调度（Tomasulo 后静态调度、表调度）同源。使用方式：任何"带依赖 DAG + 每步资源上限"的离线部署/映射问题——DIAMoND 用于权重矩阵到 NAND OU 的部署；本库相关条目：资源受限调度问题与 GA 调度策略搜索（AutoFHE 用遗传算法求同构 RCPSP 的近优解，list scheduling 是其经典贪心对照）。变体：多 cycle 操作、链式（chaining）、优先级动态重算（关键路径距离随调度推进更新）。

涉及论文标题：
- DIAMoND Dynamic Inference for Adaptive Edge MoE with Heterogeneous In-NAND and Near-DRAM Compute Architecture
