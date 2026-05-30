## Attention State / Attention Composition (⊕ operator)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Attention State 是 FlashInfer 定义的 attention 计算的标准输出格式，基于 Block-Parallel Transformer (BPT, Liu & Abbeel, 2023) 的 observation：attention outputs for the same query and different keys/values can be composed。Attention State 是一个 tuple $\begin{bmatrix} \mathbf{O}(\mathcal{I}) \\ \mathbf{LSE}(\mathcal{I}) \end{bmatrix}$，其中 $\mathbf{O}(\mathcal{I})$ 是 query 对 index set $\mathcal{I}$ 的 attention output，$\mathbf{LSE}(\mathcal{I}) = \log \sum_{i \in \mathcal{I}} \exp(\mathbf{q} \cdot \mathbf{k}_i)$ 是 attention scale（log-sum-exp of attention scores）。Attention State 的关键性质：$\oplus$ operator 是 **associative and commutative** 的，即 $\text{State}(\mathcal{I} \cup \mathcal{J}) = \text{State}(\mathcal{I}) \oplus \text{State}(\mathcal{J})$，且合并顺序任意。这意味着多个 partial attention computation 的结果可在任意顺序下合并为正确 final output。⊕ operator 定义为：

$$\begin{bmatrix} \mathbf{O}(\mathcal{I} \cup \mathcal{J}) \\ \mathbf{LSE}(\mathcal{I} \cup \mathcal{J}) \end{bmatrix} = \begin{bmatrix} \frac{\exp(\mathbf{LSE}(\mathcal{I}))\mathbf{O}(\mathcal{I}) + \exp(\mathbf{LSE}(\mathcal{J}))\mathbf{O}(\mathcal{J})}{\exp(\mathbf{LSE}(\mathcal{I})) + \exp(\mathbf{LSE}(\mathcal{J}))} \\ \log(\exp(\mathbf{LSE}(\mathcal{I})) + \exp(\mathbf{LSE}(\mathcal{J}))) \end{bmatrix}$$

在 FlashInfer 中，Attention State 用作 partial attention computation 的 canonical output，⊕ 用作 standard reduction operator（类比 GEMM 中的 summation）。这使得：(1) Load-balanced kernel 可将长 KV-cache 拆分为多个 chunks，由不同 CTAs 并行处理，各 CTA 输出 partial AttentionState；(2) Contraction kernel 用 ⊕ compose 所有 partial states 为 final output；(3) 合并顺序无关，允许 deterministic fixed-order aggregation（vs Stream-K 的 atomic non-deterministic aggregation）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Attention State 在 FlashInfer 中的完整 pipeline：

```
// ===== Attention Computation with Attention State =====
// Input: Q ∈ R^{T_q × d}, K ∈ R^{L × d}, V ∈ R^{L × d}
// Output: O ∈ R^{T_q × d}

// Step 1: Split K/V into chunks (for load-balanced scheduling)
chunks = [(K_1, V_1, L_1), (K_2, V_2, L_2), ..., (K_n, V_n, L_n)]
// 其中 Σ L_i = L

// Step 2: Parallel per-chunk attention computation
for each chunk_i in parallel:  // different CTAs
    // Standard FlashAttention for this chunk
    O_i = 0; m_i = -inf; l_i = 0
    for each KV tile in chunk_i:
        S = Q × K_tile^T
        m_new = max(m_i, rowmax(S))
        P = exp(S - m_new)
        l_new = l_i * exp(m_i - m_new) + rowsum(P)
        O_i = O_i * exp(m_i - m_new) + P × V_tile
        m_i = m_new
        l_i = l_new
    
    // Convert to AttentionState: canonical output format
    // O_i is already accumulated attention output
    // LSE_i = log(l_i) + m_i  (recover log-sum-exp from running stats)
    partial_state[i] = AttentionState(
        O = O_i / l_i,           // normalize
        LSE = log(l_i) + m_i     // attention scale
    )

// Step 3: Merge all partial states via ⊕ (contraction)
O_final = zeros(T_q, d)
LSE_final = -inf  // log(0) equivalent
for each partial_state[i]:
    O_final, LSE_final = (O_final, LSE_final) ⊕ partial_state[i]
    // ⊕ expansion:
    // weight_final = exp(LSE_final)
    // weight_i = exp(partial_state[i].LSE)
    // O_final = (weight_final * O_final + weight_i * partial_state[i].O) 
    //         / (weight_final + weight_i)
    // LSE_final = log(weight_final + weight_i)

// ===== Associativity Proof Sketch =====
// ⊕ is associative because:
//   (A ⊕ B) ⊕ C = A ⊕ (B ⊕ C)
// This follows from the associativity of addition and the monotonicity of exp/log
// ⊕ is commutative because:
//   A ⊕ B = B ⊕ A
// This follows from the commutativity of addition

// ===== Key Insight =====
// The same ⊕ operation works for:
// 1. Merging parallel chunks from load-balanced scheduling
// 2. Merging prefix and suffix in composable formats
// 3. Merging speculative decoding tree branches
// All three use cases share the same attention composition operator
```

与相关概念的比较：
- **FlashDecoding (Dao et al., 2023)**：使用 Split-K 将 KV split 为 chunks，各 chunk 输出 partial softmax → final reduction。本质上是 Attention State + ⊕ 的特例
- **Ring-Attention (Liu et al., 2023)**：利用 ⊕ 的 associative 性质将 attention 分布到多设备，ring communication 传递 partial states
- **FlashInfer**：将 ⊕ 标准化为 attention kernel 的 canonical reduction operator，使 load-balanced scheduling、composable formats、tree attention 统一使用同一 merge 原语

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

FlashInfer 中 Attention State 的实现：
- 在 CUDA 层面，Attention State 用两个 per-query-row floats 表示：(O_i ∈ R^d, LSE_i ∈ R)
- Contraction kernel 接收多个 partial Attention States → GPU 上用 fast math (exp/log on CUDA MUFU) 执行 ⊕ compose
- Implementation 注意数值稳定性：weight = exp(LSE - LSE_max) 避免 overflow（与 online softmax 中的 rescaling 类似）
- 多种使用场景复用同一 contraction kernel：load-balanced scheduling（同 batch 内 chunks）、composable formats（prefix + suffix）、speculative tree attention（tree branches）

涉及论文标题：
- FlashInfer Efficient and Customizable Attention Engine for LLM Inference Serving
