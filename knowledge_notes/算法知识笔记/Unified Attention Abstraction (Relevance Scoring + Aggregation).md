## Unified Attention Abstraction (Relevance Scoring + Aggregation)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Unified Attention Abstraction（统一注意力抽象）是 MetaAttention 提出的核心抽象，将各类 attention 机制的共同本质归结为两个基本操作：(1) **Relevance Scoring（相关性评分）**——计算输入 tokens 之间的成对相似度或交互，通常通过内积或其他相似度度量实现，形成 token-to-token 关系的数学描述；(2) **Aggregation（聚合）**——利用 relevance scores 将上下文信息整合为每个 token 的表示，即加权求和 Value vectors。这两个操作捕获了所有 attention 变体的共同骨架：先计算 token 间相关性，再用相关性加权聚合信息。

该抽象的关键在于其**完备性**——能够表达 Softmax Attention、Sigmoid Attention、ReLU Attention、Linear Attention（Mamba2）、RetNet、Multi-head Latent Attention (MLA)、Sliding Window Attention、Sparse Attention (SeerAttention)、Gated Retention 等十余种 attention 变体。每个变体的差异被归约为：(1) relevance scoring 的具体计算方式（如 matmul vs chunk-wise matmul vs state-based matmul）；(2) 中间 tensor 的自定义变换（masking、scaling、normalization）；(3) aggregation 的具体方式（全局 vs 增量压缩 state）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

抽象在 attention pipeline 中的体现——所有 attention 变体共享同一高层结构：
```
# 统一的 attention pipeline（MetaAttention unified template）
def attention(Q, K, V, customizable_functions):
    state = init_state()         # 初始状态（并行模式: None; 循环模式: zeros）
    for segment in sequence:
        # Step 1: Relevance Scoring（固定，不可自定义）
        scores = relevance_scoring(Q[segment], K, state)
        
        # Step 2: Customizable Score Transformation
        scores = scores_Mod(scores)         # 元素级变换（mask/scale）
        weights = scores_RowNorm(scores)    # 行归一化（softmax/sigmoid/L2）
        
        # Step 3: Aggregation（固定，不可自定义）
        output = aggregate(weights, V, state)
        
        # Step 4: Customizable Output Transformation
        output = output_Mod(output)         # 最终输出变换
    return output
```

两种实例化模式：
- **Parallel Pattern**: `relevance_scoring = matmul(Q, K^T)`，`aggregate = matmul(weights, V)`。适用需要全局上下文的 attention（Softmax/Sigmoid/MLA/RetNet Parallel）
- **Recurrent Pattern**: `relevance_scoring = matmul(Q, h)`（h 为压缩 hidden state），`aggregate: h = h + matmul(K[i]^T, V[i])`。适用 stateful attention（Mamba2/RetNet Recurrent/Gated Retention）

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

MetaAttention 在 7.3k 行 C++/Python 中实现该抽象。用户仅需声明 pattern 类型（Parallel/Recurrent）、定义 input tensor shapes、编写 customizable functions（Mod 和 RowNorm），框架自动完成 scheduling、code generation 和 multi-backend execution。实现的关键技术：RowNorm Online 接口泛化 online softmax 到任意 row-wise normalization；IntermediateTensor scheduling 自动传播 tile shape 和 memory placement；TMA+Tensor Core (NVIDIA) 或 Matrix Core+async copy (AMD) 双 backend 支持。详见论文 Section 3 (Programming with MetaAttention)。

涉及论文标题：
- MetaAttention: A Unified and Performant Attention Framework across Hardware Backends
