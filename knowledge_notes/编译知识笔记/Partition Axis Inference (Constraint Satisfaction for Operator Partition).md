## Partition Axis Inference (Constraint Satisfaction for Operator Partition)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Partition Axis Inference（分区轴推断）是 Lancet 编译器 Operator Partition Pass 中的子模块，使用约束满足问题（CSP）为指令序列中每个张量的输入/输出确定合法的分区维度。当对一段指令进行分区以组成 pipeline 时，必须确保原始张量可以从分区后张量正确重建——这要求每个算子的输入输出分轴满足数学约束。使用 OR-Tools CP-SAT solver 求解。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。

CSP 形式化定义：

```
变量: a = {a_x_i^n, a_y_j^n}
  a_x_i^n: 第n条指令第i个输入的分区轴 ∈ {0,1,..., -1(不分区), A_irr(MoE不规则)}
  a_y_j^n: 第n条指令第j个输出的分轴

约束1 — 算子分区约束 F_Z^f:
  每个算子 f 定义合法的输入输出分轴组合
  例: 矩阵乘法 Y = X·W
    (a_x1=0 ∧ a_x2=-1 ∧ a_y1=0)     // 沿X的行分区
    ∨ (a_x1=-1 ∧ a_x2=1 ∧ a_y1=1)   // 沿W的列分区

约束2 — 张量依赖一致性:
  若 output_j^i → input_l^k（指令i的第j个输出喂给指令k的第l个输入）:
    a_y_j^i = a_x_l^k  // 同一张量的分区轴不能改变
  （切换分轴需要跨partition数据, 中断pipeline）

求解: find a s.t. ∧_i F_Z^{f^i} ∧ ∧_D (a_y_j^i = a_x_l^k)
```

MoE 特殊处理 — A_irr 分区轴：
- all-to-all 和 expert: 可接受 capacity 维度的常规分区（当 range 仅覆盖 all-to-all+expert）或 A_irr（当 range 扩展到 non-MoE 计算，需沿 batch 维度分区）
- MoE gather: 只接受 A_irr 输入（不能接受 capacity 分区，因为 capacity 分区导致 output tokens 位置不规则，interrupt pipeline）
- 可分区的 gating function: 接受 batch-partitioned 输入，产生 A_irr-partitioned 输出

Lancet 为所有常见 Transformer 算子实现了 F_Z 约束。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Partition Axis Inference 作为 DP 搜索的内部循环被调用——每次 DP 尝试候选 (i,n,k) 时，CSP solver 判定该 range 是否可分区并返回具体分轴方案。求解失败时 P(i,n,k)=∞（该候选不可行）。大部分算子有多条合法分轴约束（如 MatMul 的两种分区方式），CSP solver 自动选择满足全局依赖一致性的方案。该模块对用户透明，无需手动标注分轴。

涉及论文标题：
- Lancet: Accelerating Mixture-of-Experts Training via Whole Graph Computation-Communication Overlapping
