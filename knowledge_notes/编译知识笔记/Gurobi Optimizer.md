## Gurobi Optimizer

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Gurobi Optimizer 是 Gurobi Optimization, LLC 开发的商业数学优化求解器，支持 Linear Programming (LP)、Integer Linear Programming (ILP)、Mixed-Integer Programming (MIP)、Quadratic Programming (QP) 等。在 MoETuner 中，Gurobi 12.0.0 用于求解 expert placement 的两阶段 ILP 问题（ILP 1: expert clustering for load balance, ILP 2: cluster-to-GPU assignment for communication minimization）。求解设置 tolerance = 0.025，表示求解器在目标函数值相对 gap 小于 2.5% 时停止迭代。

从编译框架角度拆解术语：
Gurobi 在 MoETuner 中的作用等价于编译框架中的启发式/最优求解 pass：
1. **输入**：Python API 构建的 ILP 模型（变量、约束、目标函数），以 token routing 统计数据参数化
2. **求解过程**：Gurobi 内部使用 Branch-and-Bound + Cutting Planes + Heuristics 组合算法 → 搜索整数解空间 → 每步计算 LP relaxation 的 lower/upper bound → 剪枝无效分支
3. **输出**：各决策变量的最优赋值（x_{c,e,l} 和 y_{c,g,l} 的具体 0/1 值）
4. **后处理**：将求解结果转换为 expert-to-GPU mapping tensor 保存

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 安装：`pip install gurobipy`（Python API），需要 license file（学术免费）
- MoETuner 中的使用模式：Python 脚本构建 ILP → `model.optimize()` → 读取 `x.X` 属性获取变量值
- 性能：对于 Mixtral-8x7B 规模（~1024 binary variables），Gurobi 在秒到分钟级完成求解
- 替代方案：OR-Tools CP-SAT (Google), CPLEX (IBM), CBC (COIN-OR 开源)
- 官方文档：https://www.gurobi.com

涉及论文标题：
- MoETuner: Optimized Mixture of Expert Serving with Balanced Expert Placement and Token Routing
- Optimizing Distributed Deployment of Mixture-of-Experts Model Inference in Serverless Computing

### Serverless MoE Deployment 补充
该论文在 serverless MoE 部署中使用 Gurobi 求解三个 MIQCP 子问题（分别固定通信方法 a=1/2/3），而非直接求解完整 MIQCP。每个子问题求解时限 60s，求解结果送入 ODS 算法做通信方法选择。相比直接 MIQCP（180s 时限，高吞吐量目标下无法收敛），分解求解+ODS 在有限时间内给出更优部署方案。
