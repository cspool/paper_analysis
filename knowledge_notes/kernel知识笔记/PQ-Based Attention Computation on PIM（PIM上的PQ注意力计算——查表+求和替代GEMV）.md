## PQ-Based Attention Computation on PIM（PIM上的PQ注意力计算——查表+求和替代GEMV）

术语是什么？通过联网搜索让回答具体和精准。
PQ-Based Attention Computation是AQPIM提出的将传统attention GEMV (qK^T) 转换为codebook lookup + partial summation的kernel设计。核心transform：不存储和计算full key matrix K (N×d)，而是存储key codebook (d×K, K个FP16 centroid per subvector) + key indices (m×N, m个subvector各1个INT index per token)。Attention计算流程：query split成m subvectors → 各subvector与对应codebook做小规模matmul得inner product matrix (1×K) → 用key indices从IPM中lookup对应值 → m个值求和得qK^T approximation。该设计使attention compute complexity从O(N×d)降至O(K×d + m×N)，当K≪N时（512 vs 32K）大幅减少计算量。关键优势：(1) 无需explicit dequantization——直接操作compressed representation；(2) 复用现有FP16 MAC units（no INT ALU needed）；(3) 与GQA/MQA正交兼容。

从kernel调度角度拆解术语：
```
// PQ-Based Attention Kernel (Decode step, 1 new query token)
// Input: q ∈ R^{1×d}, key_codebook ∈ R^{m×d/m×K}, key_indices ∈ Z^{m×N}
//        value_codebook ∈ R^{m×K×d/m}, value_indices ∈ Z^{m×N}

// === BankPE: ATNK (Attention-Kernel Key) ===
q_sub[1..m] = split_into_m_subvectors(q)  // m=32, each [1, d/m]
for sub in 1..m:  // Parallel across banks
    // Inner Product Matrix: query_sub × key_codebook
    IPM[sub][0:K-1] = q_sub[sub] @ key_codebook[sub]^T  // [1,d/m] × [d/m,K] → [1,K]
    // IPM stored in row buffer (512 FP16 values = 1KB)

// === BufferPE: SFM (Softmax with lookup) ===
for sub in 1..m:  // Sequential or parallel
    IPM[sub] received from BankPE via PIM_MV_BA
// Lookup + Sum + Softmax:
for n in 1..N:  // For each token in sequence
    qKT[n] = 0
    for sub in 1..m:
        idx = key_indices[sub][n]  // ∈ [0, K-1]
        qKT[n] += IPM[sub][idx]    // intra-row indirection lookup

attn[0:N-1] = softmax(qKT / sqrt(d_head))  // BufferPE SFM unit

// === BankPE: ATNV (Attention-Kernel Value) ===
// attn weights distributed back to BankPE via PIM_MV_BF
for sub in 1..m:  // Parallel across banks
    // Reconstruct value vectors: Σ attn[n] × value_codebook[sub][index]
    output[sub][0:d/m-1] = 0
    for n in 1..N:
        idx = value_indices[sub][n]
        output[sub] += attn[n] × value_codebook[sub][idx][0:d/m-1]

output = concat(output[1..m])  // → GPU via PIM_RD
```

术语一般如何实现？如何使用？
PIM Command Sequence for one PQ-attention decode step:
1. PIM_SET_CONFIG: configure m=32, K=512, d_head dimensions
2. PIM_WR: write query vector to BankPE
3. PIM_MAC_AB: BankPE computes query×codebook → IPM in row buffer
4. PIM_MV_BA: transfer IPM to BufferPE
5. PIM_RET: intra-row indirection lookup (PIM_RET command) → read IPM values by indices
6. PIM_SFM: softmax on looked-up values
7. PIM_MV_BF: transfer attention weights back to BankPE
8. PIM_MAC_AB: value reconstruction (attn × value codebook)
9. PIM_RD: read final attention output → GPU

Performance: GEMV complexity O(N×d) → PQ attention O(K×d + m×N), K=512 ≪ N=32768 → matmul cost constant w.r.t. sequence length (maintained as K grows negligibly). 实测: decoding per-step latency AQPIM=0.12× vs GPU baseline at S_len=32768 (8.33× speedup, Fig.12).

涉及论文标题：
- AQPIM: Breaking the PIM Capacity Wall for LLMs with In-Memory Activation Quantization

