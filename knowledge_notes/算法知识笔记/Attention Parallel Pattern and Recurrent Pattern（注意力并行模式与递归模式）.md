## Attention Parallel Pattern and Recurrent Pattern（注意力并行模式与递归模式）

术语是什么？通过联网搜索让回答具体和精准。

Attention Parallel Pattern 和 Recurrent Pattern 是 MetaAttention 将统一 attention 模板实例化得到的两种计算模式，覆盖所有主流 attention 变体。Parallel Pattern 对应需要全局 K/V context 的 attention，relevance scoring 和 aggregation 以并行矩阵乘法方式在整个 K/V 序列上执行。Recurrent Pattern 对应可以将 context 压缩到固定大小 hidden state 的 attention，relevance scoring 和 aggregation 以迭代方式逐 token 更新和查询 hidden state。两种 pattern 共享相同的编程接口（input shapes + customizable functions），但底层数据流和优化策略不同。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

**Parallel Pattern** 数据流：
```
输入: Q[batch, head, seq_len, dimqk], K[batch, head, seq_len_kv, dimqk], V[batch, head, seq_len_kv, dimv]
阶段1 (Relevance Scoring): scores = matmul(Q, K^T)          // [batch, head, seq_len, seq_len_kv]
阶段2 (Customizable Mod):    scores = scores_mod(scores)      // 如 causal mask, scaling
阶段3 (Customizable RowNorm): scores = scores_rownorm(scores) // 如 softmax, RetNet norm
阶段4 (Aggregation):         output = matmul(scores, V)       // [batch, head, seq_len, dimv]
阶段5 (Customizable Output Mod): output = output_mod(output)
```
覆盖：Softmax Attention、Sigmoid Attention、ReLU Attention、RetNet Parallel、Sparse GQA、Multi-head Latent Attention (MLA，query seqlen=1 解码场景)。

**Recurrent Pattern** 数据流：
```
初始化: H = zeros[head_dim_qk, head_dim_v]  // 压缩 hidden state
对于 i = 1 到 seq_len:
    // relevance scoring: 用 Q[i] 查询压缩 state
    output[i] = matmul(Q_mod(Q[i]), H)       // [head_dim_v]
    // aggregation: 用 K[i], V[i] 更新压缩 state
    H = H + matmul(K_mod(K[i]), V_mod(V[i]))
    H = h_mod(H)                              // 可选 state 变换
```
覆盖：Mamba2 SSM、RetNet Recurrent、YOCO Gated Retention、RFA-Big。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

在 MetaAttention 中，两种 pattern 在 attention runtime 中以不同 kernel template 实现。Parallel Pattern kernel 使用类似 FlashAttention 的 online tiling 策略：沿 KV sequence 维度分 tile，在每个 tile 内计算局部 relevance scores → 应用 customizable functions → 使用 RowNorm online 更新全局归一化状态 → 聚合 V → 移动到下一个 KV tile。Recurrent Pattern kernel 使用 chunk parallelism：将长序列沿 sequence 维度切为多个 chunk，每个 chunk 内维护本地 recurrent state，chunk 间传递 state，chunk 内 elementwise/reduction 逻辑融合到 single fused kernel。用户通过 programming interface 声明 attention pattern（`pattern: Parallel` 或 `pattern: Recurrent`），MetaAttention 自动选择对应 kernel template。

涉及论文标题：
- MetaAttention: A Unified and Performant Attention Framework Across Hardware Backends

