## SAT（布尔可满足性）与 CNF（合取范式）/ kSAT

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SAT（Boolean Satisfiability，布尔可满足性）是判断是否存在一组布尔变量赋值使给定命题公式为真的问题，是第一个被证明为 NP-complete 的问题（Cook-Levin 定理），也是组合优化问题（COP）的经典代表。标准表示是合取范式 CNF（Conjunctive Normal Form）：一组子句（clause）的合取，每个子句是若干文字（literal）的析取，文字是变量 x 或其否定 ¬x。一个 CNF 实例可满足当且仅当每个子句都能取真。kSAT 表示每个子句恰好含 k 个文字的 SAT 子类；任何 k>3 的 kSAT 子句可在多项式时间内转化为等价的 3SAT 子句集合（kSAT→3SAT 转换）。SAT 的实际应用覆盖软件/硬件验证、计算生物学、密码分析、金融建模与神经网络验证（SAT 已成为形式验证的标准模型）。在 SATIC 论文中，SAT 是待映射到 Ising 机器的输入问题：变量对应 Ising spin（或 QUBO 二值变量），子句对应 spin 间交互，满足赋值对应能量低态。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
从算法 pipeline 角度，一个 SAT 求解流程（以 3SAT 为例，DIMACS CNF 格式输入）：
```
# 输入：CNF，如 F = (x1∨x2∨x3) ∧ (¬x3∨x4∨x5)（DIMACS：p cnf 5 2）
# 输出：可满足赋值 或 UNSAT
assign = {}                              # 部分赋值
repeat:
    (unit, val) = find_unit_clause(CNF, assign)   # 单元传播：单文字子句强制赋值
    while unit: assign[x]=val; simplify(CNF)
    if empty_clause(CNF): backtrack      # 冲突 → 回溯（DPLL/CDCL）
    if all_vars_assigned and all_clauses_true: return SAT, assign
    x = pick_variable(CNF)               # 变量选择启发式（如 VSIDS）
    branch on x=0/1
```
在 SATIC 的 Ising 编译 pipeline 中，SAT 问题作为 CNF 输入后：构建 VIG（变量为节点、共现为边）→ 在 CNF 层做子问题形成（freeze 未选变量 + 单元传播化简）→ 公式化为 QUBO（引入 ancillary 变量）→ Ising 硬件退火 → 回收解更新全局解向量 → CheckSolution 验证全部分子句。示例：F=(x1∨x2∨x3)∧(¬x3∨x4∨x5)，冻结 x4=0、x5=0 后单元传播得 (x1∨x2∨x3)∧(¬x3)，公式化仅需 4 spin。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现方式：经典求解器（DPLL/CDCL，如 MiniSat/CaDiCaL/WalkSAT）用回溯 + 冲突学习 + 启发式搜索；Ising/量子路线（本论文）把 SAT 编码为能量最小化问题交给 Ising 机/退火器（Lucas 证明 Karp 的 21 个 NP-complete 问题都可写成 Ising 哈密顿量）。使用方式：SAT 实例用 DIMACS CNF 格式存储（行 `p cnf <n> <m>` 声明变量与子句数，每行以 0 结尾列出一个子句的正负整数文字），SATIC 读取该格式后执行编译流程；benchmark 来源包括 SATLIB 与 QuICC-SAT-Datasets。SAT 在验证领域通常与 bounded model checking、equivalence checking 结合使用。

涉及论文标题：
- SATIC: An Optimizing Ising Compiler for SAT(isfiability)
