## Unit Propagation（单元传播）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Unit propagation（单元传播/单位传播）是 SAT 求解的基础推理规则：若 CNF 中存在只含一个文字的子句（unit clause，单元子句），则该文字对应的变量取值被强制确定（若子句为 (l)，则 l 必须为真），赋值后删除满足的子句并从其余子句中删除已满足文字/缩短含冲突文字的子句，可能递归产生新的单元子句，直到没有可传播的单元子句。时间复杂度 O(L)（L 为文字数），是 DPLL/CDCL 的核心组件。在 SATIC 中，单元传播被用作编译期的化简工具而非求解工具：freeze（冻结）一个未选变量后运行单元传播，把子 CNF 中受影响的子句宽度降低（3SAT 子句常降为 2SAT/1SAT），从而公式化后 ancillary 更少、子问题更易适配 Ising 机器容量。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
在 SATIC 编译框架中，单元传播位于 VarFreeze 之后、Formulate 之前（Algorithm 1 的 UnitProp 例程）。运转流程（Eq.(5) 示例，4-spin 机器）：
```
# 原 CNF: F = (x1∨x2∨x3) ∧ (¬x3∨x4∨x5)
# Step 1: freeze 未选变量 x4=0, x5=0（取值自全局解向量 S_global）
# Step 2: 代入并传播 → 子句 (¬x3∨x4∨x5) 中 x4=x5=0 使 ¬x3 成为唯一可满足文字
#         单元子句 (¬x3) → 强制 x3=0，传播后该子句被满足删除
# 子 CNF: (x1∨x2∨x3) ∧ (¬x3)   [单文字子句无 ancillary]
# Step 3: Formulate → 4 变量 QUBO（C1 引入 1 ancillary，单元子句 0 ancillary），恰好装满 4 spin
```
Bulk Freeze 优化批量冻结 B 个变量后一次传播，减少重复传播开销；传播后 2SAT 子句频繁出现，触发 Clause Based Formulation Mix 用 Chancellor's 处理。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：经典求解器用 watched literal 数据结构使传播摊销为近 O(1)/子句更新；SATIC 作为编译化简，冻结变量后线性扫描子句（O(L)）更新宽度。使用：压缩子句宽度 → 减少公式化 ancillary → 提高 Ising 机器容量利用率（73× 容量扩展的关键机制之一）；在经典 DPLL/CDCL 中则是回溯搜索的推进器。论文评估中 Bulk Freeze 显著降低单元传播时间（即使计入自身开销）。

涉及论文标题：
- SATIC: An Optimizing Ising Compiler for SAT(isfiability)
