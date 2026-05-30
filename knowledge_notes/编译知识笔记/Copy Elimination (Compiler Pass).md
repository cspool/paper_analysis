## Copy Elimination (Compiler Pass)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Copy Elimination是Cypress编译器的第三个pass（位于dependence analysis和vectorization之后），用于消除dependence analysis中copy-in/copy-out discipline引入的冗余数据拷贝。由于dependence analysis采用简单的copy-in/copy-out策略（每个sub-task launch都为tensor创建fresh allocation并执行copy），会引入大量逻辑上不必要的数据搬运——copy elimination通过四类rewrite pattern消除这些冗余。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
Cypress的四类Copy Elimination Pattern（Figure 10）及运转示例：

Pattern 1 — Spill Elimination（Figure 10a）:
  场景：parent tensor被copy到其partition的子tensor，子tensor又被copy回parent
  Before:
    e2 = copy(t, P[i]), {e1}
    e3 = copy(P[i], t), {e2[:]}
  After:   (两者均删除)
    同步e2[:]被消除——因为partition相同，point-wise依赖已保证正确性
  
Pattern 2 — Spill Hoisting（Figure 10b）:
  场景：循环内parent→child copy和child→parent copy
  Before:
    for i:
      e3 = copy(P[j], t)
      b
      e5 = copy(t, P[j]), {e4}
  After (hoist to pre/postamble):
    e3 = copy(P[j], t), {e1}    // hoisted before loop
    for i:
      b                          // loop body without copies
    e5 = copy(t, P[j]), {e2}    // hoisted after loop

Pattern 3 — Duplicate Elimination（Figure 10c）:
  场景：对同一tensor的重复copy-in
  Before: e1 = copy(P[i], t); b1; e2 = copy(P[i], t); b2
  After:  e1 = copy(P[i], t); b1; b2[替换e2→e1]

Pattern 4 — Self Copy Elimination（Figure 10d）:
  场景：同一allocation的self-copy
  Before: e2 = copy(t, t), {e1}
  After:  b2[替换e2→e1]
  注意：这类pattern保留event依赖（不能消除sync），因为t仍然需要等所有parallel copies完成

应用顺序关键：
  spill类patterns（1,2）优先——可以同时消除copy和同步
  保留依赖的patterns（3,4）后应用——必须保留synchronization

具体实例（论文Section 4.2.3）：
  warpgroup将accumulator从寄存器copy到shared memory → 各线程的copy被消除
  (self copy elimination)，但warpgroup-level的synchronization保留
  → TMA需要知道所有线程完成before store

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现方式：
- MLIR-based pattern rewriting系统
- 每个pattern匹配IR中的特定copy模式并重写
- Pattern ordering由compiler控制——spill-related patterns优先以最大化消除同步
- 论文提到未来可用egraph (equality saturation)避免ordering heuristic
- Copy elimination与resource allocation交互——被消除的intermediate tensors不需要物理allocation

涉及论文标题：
- Task-Based Tensor Computations on Modern GPUs
