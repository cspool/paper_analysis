## Product Quantization (PQ) for KV Cache Compression（乘积量化KV缓存压缩）

术语是什么？通过联网搜索让回答具体和精准。
Product Quantization (PQ) 是一种基于向量量化的压缩技术，由Jegou等人于2011年提出[H. Jegou et al., "Product Quantization for Nearest Neighbor Search," TPAMI 2011]，起源于近似最近邻搜索领域。PQ的核心操作是：(1) Vector Splitting：将d维高维向量分解为m个子向量（每组d/m维）；(2) 对每个subvector group独立运行K-means clustering，生成K个centroid的codebook；(3) 原始向量由m个index（每个∈[0,K-1]）表示，指向各subvector group的centroid。AQPIM首次将PQ引入PIM-based KV cache在线量化，利用PIM高内部带宽(>TB/s)支持在prefill阶段on-the-fly执行K-means clustering（4迭代收敛），将PQ从传统离线weight-only压缩扩展到在线activation压缩。与per-weight scalar quantization（如uniform INT4/KVQuant）不同，PQ通过subvector-level clustering捕获KV cache activation的context-dependent locality和similarity（UMAP visualization显示KV vector呈tight cluster分布），可达到80%+ memory reduction ratio。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。
AQPIM中PQ-based KV cache量化pipeline（以Mistral-7B, m=32, K=512为例）：

```
// ===== Prefill阶段: Codebook Generation (GPU-PIM并行) =====
// 输入: K, V ∈ R^{N×d} (N tokens, d=4096 hidden dim)
// Parameter: m=32 subvectors, K=512 centroids per subvector

for each attention layer:
    // Step 1: Channel Pre-Sorting (offline, absorb to projection)
    //    W_q'=W_q·P_k, W_k'=W_k·P_k, W_v'=W_v·P_v, W_o=W_o·P_v^T
    //    P_k, P_v generated offline via cosine similarity grouping
    
    // Step 2: Vector Splitting
    for i in 1..m:  // m=32 subvectors
        K_sub[i] = K[:, (i-1)*d/m : i*d/m]  // [N, d/m]
        V_sub[i] = V[:, (i-1)*d/m : i*d/m]
    
    // Step 3: Importance-Weighted K-means (per subvector, 4 iterations)
    w = sum(S[-32:, :], axis=0)  // attention score weights, t=32
    for iter in 1..4:
        // E-step: Assign tokens to nearest centroid
        for n in 1..N:
            c[n] = argmin_k ||K_sub[i][n,:] - centroid[k]||²
        
        // M-step: Weighted centroid update
        for k in 1..K:
            centroid[k] = Σ_{n: c[n]=k} w[n] × K_sub[i][n,:] / Σ_{n: c[n]=k} w[n]
    
    // Output per subvector: centroids[i] ∈ R^{K×d/m}, indices[i] ∈ Z_K^N

// ===== Decode阶段: PQ-Based Attention =====
// 输入: q ∈ R^{1×d} (new token query)
// Key codebooks + indices, Value codebooks + indices

// Step 1: Query subvector splitting
q_sub[1..m] = split(q, m)  // m=32, each [1, d/m]

// Step 2: Inner Product Matrix (query × codebook)
for i in 1..m:
    IPM[i] = q_sub[i] × centroids_k[i]^T  // [1, d/m] × [d/m, K] = [1, K]

// Step 3: Lookup indices → sum → qK^T approximation
qKT_approx = zeros(1, N)
for i in 1..m:
    for n in 1..N:
        qKT_approx[n] += IPM[i][ indices_k[i][n] ]

// Step 4: Softmax + Value Reconstruction
attn = softmax(qKT_approx / sqrt(d_head))  // [1, N]
output = Σ_{i=1..m} Σ_{n=1..N} attn[n] × centroids_v[i][indices_v[i][n]]
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
PQ的经典开源实现包括Faiss [https://github.com/facebookresearch/faiss, Meta AI]的ProductQuantizer类（支持PQ/OPQ/IVFPQ等变体，用于ANN search）。LLM领域的PQ应用：PQCache [Zhang et al., 2024] 使用PQ identify important tokens for KV cache offloading，但保留full KV copy在CPU memory；Squeezed Attention [Hooper et al., 2024] 用PQ做sparse attention token selection。AQPIM首次directly使用PQ as KV source（无full KV copy），并in PIM执行online clustering和compressed attention computation。核心实现要点：(a) subvector count m和centroid count K的trade-off——m=32, K=512在LongBench上accuracy饱和；(b) 保留前8 sink tokens和最近32 sliding window tokens为full precision以维持accuracy；(c) clustering overhead通过PIM-GPU parallel prefill完全隐藏（codebook_gen latency < prefill_total，图4）；(d) OnlinePQ (progressive centroid update)未带来accuracy增益因此被省略。

涉及论文标题：
- AQPIM: Breaking the PIM Capacity Wall for LLMs with In-Memory Activation Quantization

