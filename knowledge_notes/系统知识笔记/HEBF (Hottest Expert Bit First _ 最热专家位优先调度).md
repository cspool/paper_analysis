## HEBF (Hottest Expert Bit First / 最热专家位优先调度)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
HEBF (Hottest Expert Bit First) 是 D2MoE 提出的启发式在线调度算法，用于最小化 Bit-Width-Aware I/O-Compute Pipeline 中的并行气泡。HEBF 的核心思想：激活频率越高的 expert + bit-width 组合应优先加载到 I/O 队列中，因为其计算时间更长，可以在其计算期间重叠更多后续 expert 的 I/O 加载。

HEBF 的关键约束：
- Constraint (6a): 计算必须在加载完成后开始：L(s+1, j, k) ≤ C(s, j, k)
- Constraint (6b): 每个 expert 按 bit-width 升序加载以最大化低 bit-width 复用：L(s, j, k) ≤ L(s, j, k+1)
- Constraint (6c): 最小化 wait time = C(s, j, k) - C(s-1, j, k) - B_{j,k}·T_comp(k)

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

```
=== HEBF Algorithm ===
输入: B[j][k] (expert j 的 bit-width k 被选择次数), 
      K (候选 bit-width 数量), N (expert 数量)

Step 1: 为每个 expert 构建按 bit-width 升序排列的队列
    for each expert j:
        Q_j = []  # sorted queue of (bitwidth_k, count=B[j][k])
        for k = 0 to K-1:
            if B[j][k] > 0:
                Q_j.append((k, B[j][k]))  # 默认 bit-width 升序

Step 2: Pop + Enqueue 到 I/O Queue
    I/O_Queue = []
    while any Q_j is not empty:
        # 从所有 expert queue 的 head 中选 count 最大的
        best_expert, best_bitwidth = argmax(head of each Q_j)
        I/O_Queue.append((best_expert, best_bitwidth))
        pop head of Q_{best_expert}

Step 3: 按 I/O_Queue 顺序执行
    for each (expert_j, bitwidth_k) in I/O_Queue:
        Load expert_j bitwidth_k from disk → GPU
        # loading 期间 HEBF 自动重叠：
        # 当前 expert 加载时 → 前一 expert 正在计算
        # 当前 expert 计算时 → 下一 expert 正在加载

输出: 最小化 bubble 的 I/O-Compute 执行 schedule
```

**关键直觉**：如果 Expert A 的 INT2 被 10 个 request 选中而 Expert B 的 INT2 被 2 个选中，HEBF 优先加载 Expert A。原因是 A 的计算时间（10× 次 GEMM）远长于 B，在 A 计算期间可以完成更多 I/O。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
D2MoE 在每次 MoE gating + bit-width routing 完成后实时运行 HEBF。论文分析显示，虽然 HEBF overhead 随 request 数量线性增长，但其在总推理时间中的占比实际下降（因为需要调度加载的 expert 数量随 request 增加更快）。HEBF 在 ablation study 中贡献了 1.11×-1.21× 的吞吐提升（on top of "+MWQ"）。

HEBF 也可以使用整数线性规划 (ILP) 或动态规划 (DP) 求精确解，但这些方法在线执行 overhead 过大（每 token、每 layer 都要重新求解），不适合实时推理场景。HEBF 是轻量化的启发式近似解。

涉及论文标题：
- D2MoE: Dual Routing and Dynamic Scheduling for Efficient On-Device MoE-based LLM Serving
