## Relevance Scoring and Aggregation（相关性评分与聚合 / 统一注意力抽象）

术语是什么？通过联网搜索让回答具体和精准。

Relevance Scoring and Aggregation 是 MetaAttention 提出的统一注意力机制抽象。它将所有 attention 变体分解为两个不可变的核心操作：(1) Relevance Scoring——计算输入 token 之间的成对相似度或相关性权重，通常通过 Q 与 K 的内积、或其他相似度度量实现；(2) Aggregation——利用 relevance scores 将上下文信息整合为每个 token 的输出表示，通常通过 scores 与 V 的矩阵乘法实现。这两个操作捕捉了所有 attention 机制的共同本质：计算 token 间关系（relevance scoring）并用该关系加权聚合信息（aggregation）。围绕这两个固定操作，MetaAttention 暴露可定制函数（Mod/RowNorm）和输入 shape 配置，以表达 softmax、sigmoid、linear、sparse、recurrent 等多种变体。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

在 MetaAttention 框架中，Relevance Scoring 和 Aggregation 根据 attention 属于 Parallel Pattern 还是 Recurrent Pattern 有不同的实例化：

**Parallel Pattern**（全局 K/V context）：
```
// relevance_scoring: Q 与整个 K 序列做矩阵乘法
relevance = matmul(Q[i], K)    // Q[i]: [head_dim_qk], K: [seq_len_kv, head_dim_qk]
                                 // 输出 relevance: [seq_len_kv]，表示 token i 对所有 KV token 的相关性

// aggregate: 用 relevance 加权聚合 V
state = matmul(relevance, V)   // relevance: [seq_len_kv], V: [seq_len_kv, head_dim_v]
                                 // 输出 state: [head_dim_v]
```
实例：标准 Softmax Attention 中，relevance = QK^T/√d_k，aggregation = softmax(relevance) × V。RetNet 中，relevance = QK^T，但 normalization 从 softmax 替换为基于 reduceAbsSum 的 RowNorm。

**Recurrent Pattern**（迭代维护压缩 hidden state）：
```
// relevance_scoring: Q 与当前压缩 state 做矩阵乘法
output = matmul(Q[i], H)       // Q[i]: [head_dim_qk], H: [head_dim_qk, head_dim_v]
                                 // 输出 output: [head_dim_v]

// aggregate: 用当前 K[i]、V[i] 更新压缩 state
H = H + matmul(K[i], V[i])    // K[i]: [head_dim_qk, 1], V[i]: [1, head_dim_v]
                                 // H 累积历史 KV 信息为固定大小矩阵
```
实例：Mamba2 SSM 中，H 对应 SSM 的 hidden state，relevance scoring 通过 Q 与 H 的乘积输出当前 token 表示，aggregation 通过 K 和 V 的外积更新 H。RetNet Recurrent 中类似，H 为压缩的 retention state。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

在 MetaAttention 实现（7.3k 行 C++/Python）中，Relevance Scoring 和 Aggregation 作为固定计算被硬编码在 attention template 中。用户无需实现这两个操作，而是：(1) 选择 Parallel 或 Recurrent Pattern 确定其计算方式；(2) 通过 customizable functions（Mod、RowNorm、RowNorm online）修改 relevance scores 的数值变换（如 mask、scaling、normalization）；(3) 声明 Q/K/V 的 shape 以支持非标准维度（如 dimqk ≠ dimv）。MetaAttention runtime 自动将这两个核心操作映射到硬件高效实现：Parallel Pattern 中 relevance scoring 使用 Tensor Cores MMA 计算 QK^T，Aggregation 同样使用 MMA 计算 scores × V；Recurrent Pattern 中 state 维护在 on-chip memory，通过 chunk parallelism 并行处理序列块。这种抽象确保框架在不牺牲性能的前提下覆盖 Softmax Attention、Sigmoid Attention、ReLU Attention、RetNet、Mamba2、MLA、Sparse GQA 等 10+ 种 attention 变体。

涉及论文标题：
- MetaAttention: A Unified and Performant Attention Framework Across Hardware Backends

