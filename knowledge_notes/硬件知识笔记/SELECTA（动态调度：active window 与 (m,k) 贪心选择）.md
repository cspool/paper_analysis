## SELECTA（动态调度：active window 与 (m,k) 贪心选择）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SELECTA 是 SegFold（ISCA'26，UCLA）提出的 SpGEMM 动态调度机制，是 Segment 动态数据流的两大组件之一。它把"固定循环序的 (m,k) 迭代"换成"逐周期按运行期稀疏模式贪心选择 (m,k) 对"：内存控制器维护一个覆盖 K 维的 active window（默认 32 个 k 值），每个周期扫描 A 的列主序 bitmask 元数据，贪心选出至多 Rmax（PE 行容量）个 (m,k) 对，准则为：① 优先选择共享同一 k 的多个 m（最大化对应 B 行的 row-wise 复用）；② 避免选同 m 不同 k 的对（它们更新同一输出行，可能产生 C 行约简冲突）；③ 窗口内某 k 的所有 A-B 交集处理完（ALLDONE）即退休该 k 并补入新 k（inter-tile 滑窗），实现 k 级流水。合法性依据是 K 维约简的结合律：SpGEMM 的约简可按任意顺序计算而不改变累加结果。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
SELECTA 的运转流程（Algorithm 1）：
```
输入: A bitmask（列主序）, B 行元数据, 硬件状态; Wmax window 大小; Rmax PE 行容量
1  while |Wk| < Wmax 且 HASMOREK(): Wk ← Wk ∪ {next k}      // 滑窗填满
2  selected ← ∅; usedM ← ∅
3  foreach k ∈ Wk:                                          // intra-tile 贪心
4      if |selected| ≥ Rmax then break
5      foreach parallel m 使 A[m,k]=1:
6          if m ∉ usedM 且 |selected| < Rmax:
7              selected ← selected ∪ {(m,k)}; usedM ← usedM ∪ {m}
8  foreach k ∈ Wk 使 ALLDONE(k): Wk ← Wk \ {k}              // 退休完成 k
9  return selected 及对应 partial B 行
```
例子（论文 Fig.2d）：4×4 tile 中 Gustavson（m-prior）选 {A0,2, A1,1, A2,0, A3,0} 只复用一次 B 行 0；outer product（k-prior）选 {A2,0, A3,0, A1,1, A3,1} 中 A3,0/A3,1 同 m 冲突；SELECTA 选 {A0,2, A1,2, A2,0, A3,0}——B 行 2 被两个 A 元素复用、无同 m 冲突。A 的 bitmask 扫描是 O(W) 组合电路，W=32 时单周期完成。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：SELECTA 逻辑位于内存控制器内（SegFold 的 memory controller 连接片上 metadata buffer 与 Index-to-PE Mapper），A 以列主序只存非零（bitmask 记录消费状态），B 以 DCSR + 每 active 行 start pointer（跟踪 partial B 行进度）；window 大小是硬件参数（W=32 为默认，敏感性实验从 1 扫到 64，≥32 后收益饱和）。使用：作为 csegfold 模拟器建模的核心调度逻辑，评估中"固定 k 迭代序"消融使性能降至 baseline 的 0.670±0.065，证明动态 k 重排对暴露 segment 级并行、保持 PE 忙碌至关重要。

涉及论文标题：
- SegFold: Accelerating Sparse GEMM with a Fine-Grained Dynamic Dataflow
