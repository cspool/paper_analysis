## 混合整数线性规划（MILP，Mixed-Integer Linear Programming）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
MILP 是带整数（通常二进制）决策变量 + 线性目标与线性约束的数学优化问题，NP-hard 但现代求解器（Gurobi、CPLEX、SCIP、开源 HiGHS）可在实践中高效求解。它在架构设计中常用于把"离散配置选择 + 连续资源分配"建模为可求解的优化。RHODES 把 RO 模型 reformulate 成标准 MILP：二进制决策向量 c∈{0,1}^|C|（恰好选 1 个 CPU 核配置）、g∈{0,1}^|G|（至多选 1 个 GPU SM×频率配置）、x∈{0,1}（compute 阶段选 GPU 还是 CPU）、辅助连续变量 w_i∈[0,1]（w_i=c_i·x 的线性化），配合线性约束（执行时间 Eq.6、功耗 Eq.7、面积 Eq.8、成本 Eq.9、tC Eq.10）与三目标（min tC / min T / min tCDP 经 Pareto 追踪线性化）。本库另有应用特化的 MILP 条目：`MILP 通信模式构建`（PipeComm 合成 collective 通信 pattern）与 `混合整数规划图划分（MIP Graph Partitioning，Gurobi）`，可对照查看 MILP 在不同架构场景的编码差异。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
RHODES 的 MILP 编码（一个碳约束下的最小化执行时间实例）：
```
决策变量: c_i∈{0,1} (选第 i 个 CPU 配置, Σc=1)；g_j∈{0,1} (选第 j 个 GPU 配置, Σg=x)；
          x∈{0,1} (compute 在 GPU=1)；w_i=c_i·x (线性化, 0≤w≤c, x−(1−c)≤w≤x)
工作量三阶段: setup/teardown 串行于 CPU (T_c,s + T_c,td)^T·c；
              compute 可 CPU (T_c,k)^T·(c−w) 或 GPU (T_g,k)^T·g  (Eq.6 ≤ T_max)
功耗: P_c^T·c + P_g,idle^T·g ≤ P_max  与  P_c,idle^T·c + P_g^T·g ≤ P_max  (Eq.7)
面积: (A_c+A_cIO)^T·c + A_g^T·g ≤ A_max  (Eq.8)
总碳: [(FPW+GPW+MPW)⊙A]^T·c + C_op 各项 ≤ tC_max，∀不确定参数∈各自 U  (Eq.10)
目标: min T (或 min tC / min tCDP)
```
tCDP= tC×T 本身非线性，RHODES 用"递增 tC 阈值序列逐个 min T"扫 Pareto 前沿再找 tCDP 最优，避免引入非线性。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：Julia 中建模（模型 solver-agnostic），调用 Gurobi 求解；实验单 RO run（45,088 配置、全不确定性参数）<5 分钟、0 最优间隙（ARM 8 核 8 线程）；WLP 扩展在 Intel Xeon E5-2680 28 线程上运行。使用：输入 SoC 配置空间（C={1..32} 核、G={m∈1..128, f∈210–765MHz 11 档}）、HILP profiling 数据（Rodinia 负载三阶段执行时间）、TDP 功耗、面积与碳参数，输出最优配置与各指标。开源：RHODES GitHub（https://github.com/mariamelgamal/RHODES，当前仅 README，代码未公开）。

涉及论文标题：
- RHODES: Robust Optimization for Uncertainty-Aware Design of CO2-Efficient Computing Systems
