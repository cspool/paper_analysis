## Page-Level KV Cache Criticality Estimation (基于 Min/Max Key 元数据的 Page 级关键性估计)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Page-Level KV Cache Criticality Estimation 是 Quest 的核心算法组件：利用 KV cache 每个 page 内 Key 向量的 per-channel 最小值 m_i 和最大值 M_i 作为紧凑元数据，结合当前 Query 向量 Q 计算每 page attention score 的数学上界，以此估计该 page 对当前 query 的关键性。关键数学保证：对 page 内任意 token t，$U_i = \max(Q_i \cdot m_i, Q_i \cdot M_i) \geq Q_i \cdot K_i^{(t)}$（因为 $m_i \leq K_i^{(t)} \leq M_i$），因此 $\sum_i U_i$ 是该 page 内最高可能的 pre-softmax attention score 的上界。选择上界最高的 K 个 page 等价于"不会遗漏任何可能得到高 attention score 的 page"。

这一设计的精妙之处：(a) 元数据大小仅 2/PageSize of KV cache（page_size=16 时 ~12.5%），criticality estimation 的内存加载远小于完整 KV cache；(b) 上界保证了选择的"安全性"——top-K pages by upper bound 一定包含真正高 attention 的 token；(c) 计算极简——仅需 per-channel max + reduce-sum，无矩阵乘法，因而 criticality estimation 开销极小。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**Quest Criticality Estimation 的详细张量计算**：

```
输入：Q ∈ R^{d_head}, M ∈ R^{num_pages × d_head}, m ∈ R^{num_pages × d_head}
输出：top_k_indices ∈ Z^K

// Vectorized computation (CUDA kernel):
// Step 1: per-channel upper bound (element-wise, parallel over pages)
// U[p][i] = max(Q[i] * m[p][i], Q[i] * M[p][i])
// 等价于: sign-preserving max selection per channel

// Step 2: reduce-sum over channels (warp-level reduce)
// score[p] = sum_{i=1}^{d_head} U[p][i]
// This is the upper bound of max_{t∈p} (Q · K_t)

// Step 3: Top-K selection (RAFT batched Top-K)
// top_k_indices = argsort(-score)[:K]

// 数学正确性证明:
// For any token t in page p:
//   For each channel i: M_i^p ≥ K_i^{(t)} ≥ m_i^p
//   → max(Q_i · m_i^p, Q_i · M_i^p) ≥ Q_i · K_i^{(t)}
//   → sum_i max(Q_i · m_i^p, Q_i · M_i^p) ≥ sum_i Q_i · K_i^{(t)} = Q · K_t
//   → score_p ≥ max_{t∈p} (Q · K_t)
// Therefore score_p 是 page p 内最高 token attention score (pre-softmax) 的上界

// 复杂度: O(num_pages × d_head) ≈ O((seq_len/page_size) × d_head)
// vs full attention: O(seq_len × d_head)
// 节省因子: page_size × (因仅加载 metadata，非完整 K cache)
```

**论文 Fig. 3 验证**：Quest 的 query-aware sparsity（基于上界估计选择 page）与 oracle sparsity（基于真实 attention score 的 top-K）高度对齐，证明了上界估计的有效性。除前两层外（稀疏度 <10%），其余层的 Quest sparsity 与 oracle 几乎重合。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Quest 开源实现（https://github.com/mit-han-lab/Quest）：在 FlashInfer 中实现为 custom CUDA kernel。(a) Per-page metadata 存储在 dedicated GPU buffer，每 page 的 (M_i, m_i) 为 2 × d_head × FP16 元素，page_size=16 时 metadata overhead = 2 × d_head / 16 = d_head/8 per token；(b) Criticality estimation kernel 使用 grid-stride loop over pages，每个 thread block 处理多个 pages，利用 warp-level reduce 做 channel sum；(c) NVBench micro-benchmark 显示 criticality estimation latency 随 seq_len 增长而趋近 1/PageSize of FlashInfer full attention。Token budget (K × page_size) 是可调超参数：PG19 perplexity 中用 4096 (~1/8 of 32K)，LongBench 中 1K budget 即达 full cache 可比性能，Passkey retrieval 中 64-1024 budget 即 100% 准确率。

涉及论文标题：
- Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference

---
