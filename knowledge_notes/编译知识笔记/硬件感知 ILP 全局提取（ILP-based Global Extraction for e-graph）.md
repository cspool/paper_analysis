## 硬件感知 ILP 全局提取（ILP-based Global Extraction for e-graph）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
ILP 全局提取是把 e-graph 的提取阶段（在等价程序空间中选一个具体程序）建模为整数线性规划（ILP）求解的技术。传统贪心提取（每 e-class 选局部最便宜的 e-node）无法处理依赖全局指令选择的优化目标——æSIP 要最小化"distinct 指令类型数"（因为它决定可裁剪的硬件单元），这天然是全局属性。æSIP 的 ILP 建模（论文 §IV-B3）：决策变量 $x_{e,n}$（e-class e 选中 e-node n）、$a_e$（e-class 激活）、$y_o$（指令类型 o 是否被任何选中节点使用）；目标 $\min \lambda\sum_o w_o y_o + (1-\lambda)\sum_{e,n} c_{e,n} x_{e,n}$——第一项最小化 distinct 指令类型集（硬件感知权重 $w_o$：乘法器/除法器权重远大于简单 ALU 操作），第二项最小化指令总数（$c_{e,n}$ 为局部代价如 latency）；$\lambda$ 权衡面积与延迟。约束：(2) 每激活 e-class 恰选一个 e-node（$\sum_n x_{e,n}=a_e$）；(3) pseudo-root/orphan 强制激活（$a_r=1$）；(4) 选父节点则子 e-class 激活（$x_{e,n}\le a_{e'}$）；(5) 选节点则类型启用（$x_{e,n}\le y_{op(n)}$）；(6) Big-M 线性化的 level 无环约束（$x_{e,n}=1 \Rightarrow \ell_{e'}\ge\ell_e+1$）。求解器：Gurobi（商用）。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
在 æSIP 编译/重写框架中的运转流程：
```
① 输入：局部饱和后的全局 e-graph（各 block pseudo-root + orphan 链接到 global root）
② 对 27 个 λ 值各构建一份 ILP（λ 细粒度靠近 0 捕捉面积主导的陡峭区）
③ Gurobi 求解：各 block 共享全局 y_o 变量 → ILP 隐式块级分解 → 并行加速
④ 每 λ 输出一个重写变体 = （选中的 e-node 集合 + 使用的指令类型集合）
⑤ 后处理重建汇编；latency 用 spike 预估、RTL 仿真后精确化
```
效果：extraction 0.3-78.4s（中位 21.9s），全局提取从小时级（monolithic）降到秒级（分治块级并行）；λ 扫描产生 area-latency Pareto 前沿（bitcnts -21.9% 面积 + 0.79× 时延、dijkstra -14.4% 面积 + 1.02× 时延、rijndael -14.6% 面积 + 0.94× 时延，Fig.9）；ecosystem 级共享复用同一 ILP 公式（输入 num-chip=k，联合分配程序到 ASIP 并选变体，num-chip=5 时 17.3% 面积降 + 11.9% 时延）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：Python/gurobipy 建模 + Gurobi 12.0.x（商用许可，Docker 镜像 polasip/esip:rv32im 内含；开源替代 HiGHS/SCIP/CBC 可替换）。使用：作为 e-graph 提取器（egg 生态的 exact extraction 路线，对比 e-boost [82] 的自适应启发式+精确求解）；也可用于其他 ILP 架构场景（本库已有相关条目：`混合整数线性规划（MILP）`（RHODES SoC 配置）、`MILP 通信模式构建`（PipeComm）、`ILP 公式化与 Flat ILP`（SATIC SAT-to-QUBO））。注意点：λ 需多值扫描覆盖权衡曲线；权重 w_o 需按硬件面积/功耗成本标定；无环约束用 Big-M 线性化（需选足够大 M）。

涉及论文标题：
- æSIP μArch-aware ASIP-ISA Co-Design via Program Synthesis, Equality Saturation, and External Don't Cares
