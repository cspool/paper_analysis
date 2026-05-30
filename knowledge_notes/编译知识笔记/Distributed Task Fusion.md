## Distributed Task Fusion

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Distributed Task Fusion 是 Diffuse 系统的核心机制，指将 task-based distributed runtime 中多个顺序发射的 index task（分布式并行 task group）合并为单个 fused task 的过程。与单机 task fusion（仅减少 launch overhead）不同，分布式 task fusion 需要处理跨 processor 的通信依赖：如果两个 point task 之间存在跨 processor 依赖（即 T1 在 processor p 的 point task 写入了 processor p' 的 point task 需要读取的数据），则融合不安全。Diffuse 通过 scale-free IR 上的 fusion constraints 分析（而非 materialize 完整的 dependence map）来识别可安全融合的 task 序列，使分析复杂度与机器规模无关。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。

Distributed Task Fusion 的编译框架流程：

```
Task Stream (program order):
  T1: ADD(center, north, t1)     // 4 parallel point tasks on 4 GPUs
  T2: ADD(t1, east, t2)          // same launch domain (4,)
  T3: ADD(t2, west, t3)
  T4: ADD(t3, south, avg)
  T5: MULT(0.2, avg, work)
  T6: COPY(work, center)         // writes to aliasing data!

     ▼ 进入 Task Window (size=5)
     
Fusion Analysis:
  T1-T5: launch-domain-equivalence ✓ (same (4,) domain)
         true-dependence ✓ (same partitions, no aliasing writes)
         anti-dependence ✓ (no write-after-read on aliasing partitions)
         reduction ✓ (no reduction conflicts)
  → T1-T5 可融合为 FUSED_ADD_MULT
  
  T6加入: true-dependence ✗!
    T5 writes to (work, P_work)
    T6 reads... not the problem
    BUT: T6 writes to (center, P_center) via COPY
    T1-T5 READ from (center, P_center_aliased), (north, P_north), etc.
    center/north/east/west/south 是 grid 的 aliasing views
    → partition of center_read ≠ partition of center_write
    → true-dependence constraint 阻止 T6 融入 fused task

  最终 fused prefix: [T1, T2, T3, T4, T5] → FUSED_ADD_MULT
     T6 单独保留
```

关键 insight：由于 aliasing views (center, north, east, west, south 共享 grid array entries)，COPY(work, center) 写回 center 时，必须保证所有读取 aliasing views 的 task 已完成并在必要时进行数据传播。Diffuse 通过 fusion constraints 的 partition equality check 正确识别了这一危险情况，阻止了不安全融合。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现由四部分组成：(1) **Task Window**：缓冲 task stream，大小由 Diffuse 自动选择（逐步增大直到所有 task 被融合）；(2) **Fusion Constraints**（见单独条目）：四个 constraint 的 dataflow 分析，贪心寻找最长可融合前缀；(3) **Fused Task Construction**：构建复合 task，参数为所有子 task 的 store 参数的并集，privilege 冲突时 promotion（如同时 R 和 W → RW）；(4) **Memoization**：通过 canonical De-Bruijn index 表示复用 isomorphic task stream 的分析结果。Distributed task fusion 本身仅减少 runtime overhead——论文明确指出若不配合 kernel fusion 则无实际加速（因为 task granularity 已超过 Legion 的最小有效粒度 ~1ms/task）。真正的 speedup 来自 kernel fusion（下一阶段）。

涉及论文标题：
- Composing Distributed Computations Through Task and Kernel Fusion
