## Weight Gradient Computation Scheduling Pass (dW Scheduling)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Weight Gradient Computation Scheduling Pass 是 Lancet 编译器中优化反向传播的 IR 级 Pass。核心洞察：反向传播中 layer N 的 weight gradient computation (dW = dY · X^T) 计算不依赖于 layer N-1 的 all-to-all 通信——dW 只需 activation gradient dY 和本地保存的 activation X，两者都已在本地可用。因此 dW 与 all-to-all 之间在依赖图中无有向路径，可被重新调度到 all-to-all 之后与其重叠执行。将 dW 调度问题建模为广义分配问题 (GAP) 的变体：目标最大化被重叠的 all-to-all 时间，约束每个 dW 最多分配给一个 all-to-all，且只有无依赖路径的 dW 可分配。因 GAP 已知 NP-hard，使用 Best-Fit 贪心启发式求解。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。

Pass 工作流程（论文 Algorithm 1）：

```
输入: IR 指令序列 I = [I_1, ..., I_N]
输出: 重排后的 I'

Step 1: CreateDependencyGraph(I) → G = (I, E)
  有向边 E_{i,j}: I_j 依赖 I_i 的输出

Step 2: Weight Gradient Computation Labelling
  I^a ← [I_i ∈ I | f^i 是 all-to-all 算子]
  对每个 all-to-all I_a:
    W^{I_a} ← {I_j | I_a 和 I_j 在 G 中无有向路径}

Step 3: Best-Fit Greedy Scheduling
  t^a, t^W ← GetInstrExecTime(I)  // Caching Op Profiler
  W^{used} ← {}
  for each I_i^a in backward order:
    t_u ← t_i^a  // 当前 all-to-all 的未重叠时间
    while t_u > 0 and W^{I_i^a} 中仍有未使用指令:
      j_min ← argmin_j |t_u - t_j^W|  // 选执行时间最接近 t_u 的 dW
      t_u ← t_u - t_j^W
      W^{used}.insert(I_j^W)
      Asg.insert({I_j^W: I_i^a})

Step 4: I' ← ReorderInstructions(Asg)
  // 将分配的 dW 指令移到对应 all-to-all 之后
```

数学形式：

$$\max_{\mathbf{x}} \sum_{j=1}^{|\mathcal{I}^a|} \min\{t_j^a, \sum_{i=1}^{|\mathcal{I}^W|} t_i^W \cdot x_{i,j}\}$$

s.t. Σ_j x_{i,j} ≤ 1（每个 dW 最多分配一次），x_{i,j}=0 若 I_i^W ∉ W^{I_j^a}（依赖约束）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

在 Lancet 中，dW Scheduling Pass 作为第一个编译 Pass 执行（先于 Operator Partition Pass），因为重排序后的 IR 会被后续 Pass 进一步优化。该 Pass 不需要硬件 profile（执行时间从 Caching Op Profiler 查询），只需依赖图分析。对 GPT2-L-MoE（24 layers, hidden 1024）的效果优于 GPT2-S-MoE（12 layers），因为更多层=更多 dW 可调度=更多重叠机会。该技术与 ByteScheduler/P3 等非 MoE communication scheduling 互补——它们处理 all-reduce 重叠，Lancet 处理 all-to-all 重叠。

涉及论文标题：
- Lancet: Accelerating Mixture-of-Experts Training via Whole Graph Computation-Communication Overlapping
