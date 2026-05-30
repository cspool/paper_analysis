## LogSumExp Reduction

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。
LogSumExp (LSE) Reduction 是一种基于 logsumexp 操作结合律的并行归约策略。logsumexp 操作定义为：
$$\operatorname{logsumexp}(\{x_1, ..., x_N\}) = \log \sum_{i=1}^{N} \exp(x_i)$$
LSE 的关键性质是 **associativity**：
$$\operatorname{logsumexp}(\{A, \operatorname{logsumexp}(\{B, C\})\}) = \operatorname{logsumexp}(\{\operatorname{logsumexp}(\{A, B\}), C\})$$
这意味着 LSE 可以像加法一样进行树形归约：先将数据分块，每块独立计算局部 LSE，再通过树形结构层层合并局部结果得到全局 LSE。同理，max 操作也具有结合律。在 attention 计算中，LSE 出现在 softmax 的分母归一化项中：
$$z = \frac{\sum \exp(q \cdot k_i^T) v_i}{\sum \exp(q \cdot k_i^T)} = \frac{\sum \exp(q \cdot k_i^T) v_i}{\exp(\operatorname{logsumexp}(\{q \cdot k_i^T\}))}$$

从算法pipeline角度拆解术语。
LSE Reduction 在 Tree Attention 中的具体计算过程：
```
# 假设 p=4 GPU，序列分 4 个 chunk
# 每 GPU: lse_i = log Σ_{j in chunk_i} exp(q·k_j^T)

# == Tree Reduction of max (2 层) ==
# Level 1 (intra-node, NVLink):
m_12 = max(lse_1, lse_2)   # GPU 1,2 归约
m_34 = max(lse_3, lse_4)   # GPU 3,4 归约
# Level 2 (inter-node, InfiniBand):
m_global = max(m_12, m_34) # 跨节点归约

# == 数值稳定化 (用 m_global 稳定所有局部值) ==
# r_i = q·k_i^T - m_global
# n_i = Σ exp(r_i) * v_i  (局部分子)
# d_i = Σ exp(r_i)        (局部分母)

# == Tree Reduction of sum (2 层) ==
# Level 1:
n_12 = n_1 + n_2; d_12 = d_1 + d_2
n_34 = n_3 + n_4; d_34 = d_3 + d_4
# Level 2:
n_global = n_12 + n_34; d_global = d_12 + d_34

# 输出: z = n_global / d_global
```

时间复杂度：Theorem 1 证明 associative reduction 在 p 个处理器上的时间为 O(N/p + log p)，其中 O(N/p) 是每处理器本地计算，O(log p) 是树形归约的通信步数。

术语一般如何实现？如何使用？
实现：通过 NCCL 的 AllReduce 操作，在 reduce 阶段使用树形归约算法。在 JAX 中通过 `lax.pmax`（对应 AllReduce(max)）和 `lax.psum`（对应 AllReduce(sum)）调用。NCCL 自动检测网络拓扑——intra-node 使用 ring reduce（NVLink 高带宽 900 GBps），inter-node 使用 tree reduce（InfiniBand 较低带宽 ~50 GBps per link）。

关键区别：传统的 attention 计算中，LSE 归约是隐式的（Flash Attention 的 online softmax 在单 GPU 内的 SM 间归约）。Tree Attention 将 LSE 归约显式化并扩展到跨 GPU 场景，揭示了 attention 的 logsumexp 归约在分布式环境中可高效并行化——与 Flash Decoding 在 GPU 内 SM 级别做 split-KV+归约的思想类似，但在跨 GPU 层级。

涉及论文标题：
- Tree Attention: Topology-aware Decoding for Long-Context Attention on GPU clusters
