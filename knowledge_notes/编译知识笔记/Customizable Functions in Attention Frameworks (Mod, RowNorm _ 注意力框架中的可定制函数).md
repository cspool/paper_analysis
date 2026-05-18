## Customizable Functions in Attention Frameworks (Mod, RowNorm / 注意力框架中的可定制函数)

术语是什么？通过联网搜索让回答具体和精准。

Customizable Functions 是 MetaAttention programming interface 中允许用户定义 attention 变体数值变换的接口。分为两种类型：(1) Modification (Mod)——支持 elementwise 变换，包括 scaling 和 masking，如标准 softmax attention 中的 Q/√d_k 缩放、causal mask 应用、sparse mask 乘法；(2) Row-wise Normalization (RowNorm)——支持跨 tensor rows 的全局调整，包含 elementwise 和 row-reduce 组合计算，如 softmax、sigmoid、RetNet 的 reduceAbsSum-based normalization。RowNorm 还提供 RowNorm online 接口（见 kernel调度 知识库），支持将 row-wise normalization 实现为 online 算法以进行 memory-efficient tiling。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。

在 MetaAttention 编译流程中，customizable functions 的处理分为三个阶段：

**1. Tracing 阶段**：frontend 将用户定义的 Python 函数 trace 为 tensor DAG。每个节点归类为 elementwise operation（add、tanh、mul 等）或 row-reduce operation（reduceSum、reduceMax 等），节点携带 tensor shape、dependency 和 grad（用于 backward）元数据。

**2. Scheduling 阶段**：customizable functions 产生的临时张量（如 mask 应用后的 scores、RowNorm 的中间累积状态）作为 IntermediateTensor 纳入 scheduling space。scheduler 为其分配 tile、memory location 和 pipeline stage，保证它们与核心 relevance scoring/aggregation 的 tile 一致。

**3. Lowering 阶段**：elementwise 节点以 SIMT 方式在 register 或 on-chip memory 中融合执行（无单独 kernel launch）；row-reduce 节点使用 intra-warp parallel reduction（利用 warp shuffle 指令如 `__shfl_xor_sync` 高效做跨线程 reduce），降低同步和数据移动开销。最终这些操作被 inline 到 high-performance attention loop 中，实现 zero additional kernel launch overhead。

具体例子：RetNet attention 的 customizable functions 定义（Python-like pseudo）：
```
def scores_Mod(scores):
    return scores * mask                  // elementwise: SIMT fused

def scores_RowNorm(scores):
    t = scores.reduceAbsSum()             // row-reduce: intra-warp reduction
    t = max(t, 1)
    return scores / t                     // elementwise: SIMT fused
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

在 MetaAttention 中，用户定义 customizable functions 仅需写 Python 表达式级别的代码。以 MLA 为例（90 LoC），用户定义 Q/K/V 的 Mod 函数处理 scaling，定义 scores_RowNorm 为 softmax（通过 RowNorm online 接口表达）。Sigmoid Attention 仅 54 LoC，Mamba2 SSM 仅 27 LoC。相比之下，手写 CUDA/Triton kernel 需要数百至数千行代码处理相同的自定义逻辑（Table 5）。这种设计的关键约束是 customizable functions 仅支持 elementwise 和 row-reduce 两类操作——这既保证了优化空间（编译器可以理解并融合这些有限类型的操作），又覆盖了绝大多数 attention 变体的数值变换需求（Table 3 展示的 10+ 种 mechanism）。类似的设计在 FlexAttention（仅支持 parallel attention 的 score_mod callback）和 FlashInfer（预定义大部分计算）中也有体现，但 MetaAttention 的接口更宽（同时支持 parallel/recurrent pattern 和非标准 shape）。

涉及论文标题：
- MetaAttention: A Unified and Performant Attention Framework Across Hardware Backends
