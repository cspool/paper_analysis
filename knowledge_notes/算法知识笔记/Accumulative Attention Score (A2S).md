## Accumulative Attention Score (A2S)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Accumulative Attention Score (A2S) 是一种通过累加 softmax 后注意力分数来评估 token 重要性的方法。其核心直觉：在 Attention 操作中，少数 token（如主语、核心动词）持续获得高注意力分数，而多数 token（如介词、冠词）分数很低。A2S 将每个 token 在所有 query 下的 Attention Score 累加，作为区分重要与不重要 token 的指标。

在 Encoder 模型中（SpAtten）：$A_k^l = \sum_{i=1}^l \sum_{h=1}^H \sum_{q=1}^N S_{q,k}^{i,h}$，跨层累积。

在 Decoder 模型中（H2O）：$A_{n,k}^{l,h} = \sum_{q=k}^{n} S_{q,k}^{l,h}$，沿 Generation Step 累积。由于 Causal Mask 的存在（$S_{q,k}^{l,h} = 0, \forall q < k$），第 k 个 token 只累积 n-k 次分数，导致早期 token 天然拥有更多累积次数，形成不公平比较。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**H2O 中 A2S 的计算流程**：

```
// 初始化 A2S 数组
A = zeros(N)  // N: 当前序列长度

// 每个 Generation Step 更新 A2S
for n in 1..max_gen:
    S_n = softmax(Q_n @ K^T / sqrt(d_k))  // [1, N], 下三角非零
    for k in 1..n:  // k <= n (causal mask)
        A[k] += S_n[k]

    // 保留 A 值最高的 K 个 token
    keep_indices = top_k(A, K)
    K_cache = K_cache[keep_indices]
    V_cache = V_cache[keep_indices]
    A = A[keep_indices]

    // 注：H2O 还保留一半预算用于 local cache (最近 token)
```

**A2S 的核心问题**：
第 k 个 token 被累积 n-k+1 次（第 k 步到第 n 步），第 n 个 token 仅被累积 1 次。Softmax 值恒非负，累积次数越多 A2S 越大→早期 token"虚胖"，近期重要 token 被误杀。

术语一般如何实现？如何使用？

A2S 以即插即用方式集成到推理流程。对每层每个 head 维护一个长度为当前序列的 A2S 向量。每步 Attention 计算后更新 A2S 并做 top-k 选择。H2O 的实现开源，A2SF 作为其改进版本也在 https://github.com/Dirac-Notation/A2SF 开源。

涉及论文标题：
- A2SF: Accumulative Attention Scoring with Forgetting Factor for Token Pruning in Transformer Decoder
- SpindleKV: A Novel KV Cache Reduction Method Balancing Both Shallow and Deep Layers

---
