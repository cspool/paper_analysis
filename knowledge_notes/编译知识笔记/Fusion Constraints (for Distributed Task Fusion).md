## Fusion Constraints (for Distributed Task Fusion)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Fusion Constraints 是 Diffuse 用于判断分布式 index task 序列是否可安全融合的四条规则。四个 constraint 均为 sound（保证融合正确）但不 complete（存在可融合但 constraint 未识别的机会）。每个 constraint 对应一种可能导致跨 processor 通信的依赖模式，在 task window 上通过前向 dataflow 分析验证：

1. **Launch-Domain-Equivalence**：所有候选 task 的 launch domain 必须相同。
2. **True-Dependence**：若 T_i 写入 (S, P)，则后续 T_j 不能通过不同的 partition P' 读取或写入 S（此场景需要跨 processor 通信传播更新值）。允许通过相同 partition P 的读写（保证 point-wise dependency）。
3. **Anti-Dependence**：若 T_i 读取 (S, P)，则后续 T_j 不能通过不同的 partition P' 写入或 reduce 到 S。允许通过相同 partition 的后续写入。
4. **Reduction**：若 T_i 对 (S, P) 执行 reduction，则后续 T_j 不能读取、写入或 reduce 到 S（避免观察到部分约简值）。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。

约束检查的前向 dataflow 分析流程：

```
Task Window: [T1, T2, T3, T4, T5]
Analysis State: tracking (Store, Partition, Access Pattern) per task

遍历 T1:
  write(T1) = {(t1, P1, W)}
  read(T1)  = {(center, Pc, R), (north, Pn, R)}
  state ← all accesses

遍历 T2:
  check T2's accesses against state:
    T2 reads (t1, P1, R) → P1 == P1 ✓ (same partition as T1's write)
    T2 reads (east, Pe, R) → no prior write to east ✓
    T2 writes (t2, P2, W) → no prior access to t2 ✓
  state ← merge T2's accesses

遍历 T5: all checks pass → fusible prefix = [T1..T5]

遍历 T6: COPY(work, center)
  T6 writes (center, Pc_copy, W)
  check against state: T1 read (center, Pc_read, R), Pc_read ≠ Pc_copy
  → true-dependence violation! (anti-dep by symmetric logic)
  → T6 不可融入 fused prefix
```

约束检查的核心是 constant-time partition equality check：Diffuse IR 将 partition 按 syntactic kind 分组（None, Tiling），同种类 partition 可通过比较结构参数（shape, offset, proj）在 O(1) 时间内判断是否相等/aliasing。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现为 task window 上的前向 dataflow 分析。对每个 candidate task，检查其参数 (S, P, pr) 与当前累积的 effects。四个 constraint 的验证条件精确形式见图 5（论文原文）。Soundness 由 Theorem 1 证明：若所有 constraint 通过，则所有依赖均为 point-wise（D(T_i, T_j)[p] ⊆ {p}），因此融合安全。Greedy 策略选择最长满足所有 constraint 的前缀进行融合。窗口大小由 Diffuse 自动选择（从较小的值开始，逐步增大直到所有 task 均被融合或达到最大 window 限制）。

涉及论文标题：
- Composing Distributed Computations Through Task and Kernel Fusion
