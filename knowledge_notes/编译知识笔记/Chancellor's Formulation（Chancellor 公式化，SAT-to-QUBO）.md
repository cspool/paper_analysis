## Chancellor's Formulation（Chancellor 公式化，SAT-to-QUBO）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Chancellor's formulation 是 Chancellor et al.（"A Direct Mapping of Max k-SAT and High Order Parity Checks to a Chimera Graph"，Scientific Reports 2016）提出的 SAT-to-QUBO 公式化方法，在公式化准确度与 ancillary 开销间取得良好平衡。对 3SAT，它把 n 变量 m 子句的实例映射为 n+m 个 spin（每子句引入 1 个 ancillary），是常用低开销 3SAT 编码。对 kSAT（k>2），先做 kSAT→3SAT 转换再套 Chancellor's，得到 n+m(2k−5) spin。其每个子句的能量构造（Eq.(3) 示例）保证：子句满足时存在 ancillary 取值使能量最低，子句不满足时能量为局部惩罚高值，即满足赋值对应 QUBO/Ising 能量低态。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
在 SATIC 编译框架中，Chancellor's 是 Formulate 阶段的公式化选项之一，用于 k≤3 的子句（尤其 3SAT 与单元传播后出现的 2SAT/1SAT 子句——2SAT 子句经 Chancellor's 无需 ancillary）。公式化流程（3SAT 子句 C=(x1∨x2∨x3)，ancillary a）：
```
# 每子句能量：H_C = min_a [ -Σx + Σ_{i<j} x_i x_j - a(Σx - 2) ]
# 系数 → QUBO Q 矩阵：
#   Q[x1][x1]=-1, Q[x2][x2]=-1, Q[x3][x3]=-1   (线性项)
#   Q[x1][x2]=+1, Q[x1][x3]=+1, Q[x2][x3]=+1   (子句内交互)
#   Q[a][a]=0,   Q[a][x1]=-1, Q[a][x2]=-1, Q[a][x3]=-1, Q[a][a]+=2
```
Chancellor's 的局限性：ancillary 数较多（n+m spin，50 变量/200 子句 3SAT → 250 变量 QUBO）；系数可能超出硬件范围被截断导致精度损失（SATIC 测试床 [−14,+14]）；其能量奖励为 −1/惩罚 0（与 ILP 的 0/+1 不同但谱距相同，可数学安全混合）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：在编译框架内以公式化例程实现（对每个子句按文字极性展开，负文字用 1−x 变换）；SATIC 中与 ILP/Flat ILP 通过 Clause Based Formulation Mix 按子句宽度混合使用（k≤3 用 Chancellor's，k>3 用 ILP/Flat ILP），并在公式化前用 Negative Literal Inversion 反转负文字为主变量的极性。使用效果（论文评估）：Clause Based Formulation Mix 在 Batch-4-100-1000 上把 solved instances 从 94 提升到 97（配合 Chancellor's 处理 2SAT）；与 NLI 组合把 solved 从 97 提到 100。

涉及论文标题：
- SATIC: An Optimizing Ising Compiler for SAT(isfiability)
