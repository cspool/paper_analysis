## Temporary Store Elimination (in Distributed Setting)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Temporary Store Elimination 是 Diffuse 在 task fusion 后执行的优化，用于将被 fused task 完全内部产生和消费、且外部不可见的分布式 store 降级为 task-local allocation，从而使其可在后续 kernel fusion 中被完全优化掉。在分布式设置中，判断一个 store 是否为 temporary 需要满足三个约束（Definition 4）：(1) 所有读取该 store 的 task 前都有一个写入该 store 的 task，且写入覆盖整个 store；(2) fused prefix 之后没有 task 读取或 reduce 该 store；(3) 应用层无对 store 的活跃引用。Temporary store 从分布式分配（如 Legion region，跨 GPU HBM）降级为 task-local allocation（如 GPU register/SRAM），为 kernel fusion 消除中间数据创造了条件。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。

```
Task Stream (cuPyNumeric: z = 2.0*x; w = y+z; v = w**2; norm(w[len(w)//2:])):
  T1: MULT([(x,R), (z,W)])     // z = 2.0*x
  T2: ADD([(y,R), (z,R), (w,W)]) // w = y+z
  T3: POW([(w,R), (v,W)])      // v = w**2
  T4: NORM([(w[len(w)//2:],R), (norm,Rd)])  // norm = ||w_half||

Fusion: [T1, T2, T3] → Fused(MULT+ADD+POW), T4 cannot fuse (reduction)

Temporary Analysis on [T1, T2, T3]:
  - z: written by T1, read by T2, no external ref → z is temporary
  - w: written by T2, read by T3, BUT also read by T4 (outside fused prefix!) → w is NOT temporary
  - v: written by T3, no reads → 但应用层持有 v 的引用（via Python variable v）→ v is NOT temporary
  - x, y: only read → NOT temporary (inputs)

Result: 仅 z 被降级为 task-local allocation
  → MLIR: memref.alloca %z → 后续循环融合将其 inlined 消除
```

第三个约束（无应用层引用）通过 split reference counting 方案实现：区分应用层持有的引用和 Diffuse runtime 持有的引用。当应用调用 `del` 操作或变量离开作用域后，应用引用计数减少。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现通过 task stream 上的前向 dataflow 分析（追踪每个 store 的生产者-消费者关系）加上 split reference counting（区分应用引用和 runtime 引用）。Temporary store 在被降级后的处理在 MLIR compilation pipeline 中完成：task-local volatile allocation → polyhedral loop fusion → allocation elimination。该优化是关键加速来源——论文明确指出没有 temporary elimination 则 kernel fusion 无法产生加速。

涉及论文标题：
- Composing Distributed Computations Through Task and Kernel Fusion
