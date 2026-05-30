## Grid Size Selection Heuristic (Stream-K)

术语是什么？
Stream-K的grid size选择启发式是一个解析模型，用于在启动kernel前确定最优的CTA数g，以最小化总执行时间。模型将单个CTA的运行时间建模为四项之和：CTA_time(g) = a + b·(FixupPeers(g)>1) + c·ItersPerCta(g) + d·(FixupPeers(g)-1)，其中：(a)固定开销——grid launch latency、compulsory cache misses、final output tile write；(b)partial sum输出条件开销——当tile数不能被完美量化时产生；(c)每MAC-loop iteration的指令/stall workload；(d)每协作CTA的partial sum读取和累积开销。ItersPerCta(g) = ⌈total_iters/g⌉，FixupPeers(g) = ⌈⌈k/BLK_K⌉/ItersPerCta(g)⌉为覆盖同一tile的CTA数。

从kernel调度角度拆解术语：
该模型的行为因问题shape而异：
- **大k、少量output tile**：reduction in MAC-loop时间单调优于fixup cost增长 → g_opt = p（全处理器宽度并行）
- **中等k、中等output tile**：fixup cost在g超过某个点后超过iteration减少收益 → g_opt < p（出现全局最小点）
- **极小m×n、极大k**（1个output tile）：虽然强伸缩潜力大，但per-peer serial reduction cost全部由单个CTA承担 → g_opt << p

参数{a,b,c,d}对每个(blocking factors, 数据类型, GPU架构)组合唯一，通过微基准经验测定，只需每个target architecture执行一次，然后将参数静态编译入库。

术语一般如何实现？如何使用？
与cuBLAS/CUTLASS的ensemble方法（静态生成20+ kernel variant + 运行时复杂heuristics/ML选择）不同，Stream-K的启发式模型：(1)仅需一次per-architecture微基准标定参数；(2)参数静态编译入库；(3)运行时快速评估模型选择g。这消除了对复杂kernel selection heuristics的需求，同时保持了动态适应性。模型也可判断何时退化为纯data-parallel（g = output tile数）是最优的。

涉及论文标题：
- Stream-K: Work-centric Parallel Decomposition for Dense Matrix-Matrix Multiplication on the GPU

---
