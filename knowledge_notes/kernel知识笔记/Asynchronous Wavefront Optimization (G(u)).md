## Asynchronous Wavefront Optimization (G(u))

术语是什么？通过联网搜索让回答具体和精准。
Asynchronous Wavefront Optimization是Infera TEU SelectKernels中第一阶段data block selection的优化目标：在data dependency DAG中选择data blocks时，不是简单选择所有zero in-degree节点，而是通过递归定义的G(u) metric估计每个候选节点对未来asynchronous wavefront（互不依赖、可并行执行的data block数量）的贡献，优先选择能最大化未来并行度的节点。G(u)越大，说明选择该节点后能释放更多下游独立可执行节点。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
G(u)的递归定义和计算过程：
```
// G(u): 节点u的asynchronous wavefront增益
// Γ⁺(u): u的出邻居集合
// d⁻(v): v的入度

G(u) = (∑_{v ∈ Γ⁺(u)} G(v) + 2) / |Γ⁺(u)| - 1

// 递归终止: 在特定深度时 G(v) = -1

// 选择策略: 对所有zero in-degree节点计算G(u), 选最大值
```

计算例子（简化3层DAG）：
```
    A (completed)     B (completed)
   / \               / \
  C   D             E   F
   \ /               \ /
    G (in-degree=2)   H (in-degree=2)
     \               /
      I (in-degree=2)
```
- 节点C: out-neighbor G (d⁻(G)=2) → 假设depth limit, G(G)=-1 → G(C) = ( (-1) + 2 ) / 1 - 1 = 0
- 节点D: out-neighbor G (d⁻(G)=2) → G(D) = ( (-1) + 2 ) / 1 - 1 = 0
- 如果 C 和 D 都完成: G的in-degree变为0 → 释放G为可执行

递归特性：transitive dependency propagation使得子节点的asynchrony gain均匀传播到所有父节点。终止深度防止长链过度影响G(u)值。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Infera在SelectKernels阶段实现G(u)计算：data dependency DAG在编译期通过computation graph静态分析构建，推理时动态维护节点状态（completed/pending/running）。G(u)的计算overhead低，因为大部分在编译期预计算（DAG拓扑、d⁻值），推理时仅对受影响的节点增量更新。选择G(u)最高的zero in-degree data blocks后，再进入第二阶段kernel selection（min #inst/IPC s.t. TLP≥4）。

涉及论文标题：
- Automated End-to-End Model Serving with Cooperative Compilation and Scheduling

