## Importance-Weighted K-Means Clustering（重要性加权K-Means聚类）

术语是什么？通过联网搜索让回答具体和精准。
Importance-Weighted K-Means Clustering是AQPIM提出的K-means变体，将token的attention-based importance score作为权重融入clustering objective function。标准K-means最小化 sum||x_n - µ_{c(n)}||² 对所有token平等对待，但LLM attention中某些token（如sink tokens, high-attention tokens）始终接收高attention scores，quantization error对这些critical token的accuracy影响更大。AQPIM修改objective为 Σ w_n ||x_n - µ_{c(n)}||²，其中weight w_n = sum(S[-t:, n], axis=0) 是该token收到的最近t个token attention scores之和。M-step中centroid更新为weighted average: µ_k = Σ_{n∈C_k} w_n x_n / Σ_{n∈C_k} w_n，使高importance token对centroid位置影响更大，从而获得更小的quantization error。该技术与FlashAttention协同：attention scores S既用于attention计算又用于weight计算，额外overhead minimal。

从算法pipeline角度拆解术语：
```
// 输入: K_sub ∈ R^{N×d/m}, S ∈ R^{N×N} (attention score matrix)
// 参数: K=512 centroids, t=32 (weight window)

// Step 1: Compute importance weights from attention scores
w = zeros(N)
for j in max(0, N-t) .. N-1:  // last t tokens
    for i in 0 .. N-1:
        w[i] += S[j, i]

// Step 2: Weighted K-means iteration (4 rounds)
centroids = random_init(K, d/m)
for iter in 1..4:
    // E-step: Standard nearest-centroid assignment (no weighting)
    for n in 1..N:
        c[n] = argmin_k ||K_sub[n,:] - centroids[k,:]||²
    
    // M-step: Weighted centroid update
    for k in 1..K:
        weight_sum = Σ_{n: c[n]=k} w[n]
        centroids[k,:] = Σ_{n: c[n]=k} w[n] × K_sub[n,:] / weight_sum
```

术语一般如何实现？如何使用？
Weights w计算在GPU prefilling阶段完成——利用FlashAttention已计算的attention scores S，仅需额外sum操作（see Eq.1: w = sum(S[-t:, :], axis=0)）。Weighted M-step中的weighted sum在BankPE (FP16 MUL+SUM)和BufferPE (reciprocal计算→送回BankPE做final multiplication)上分布式执行。Ablation study显示(Table IV)：Standard PQ avg 44.29 vs w/o weighting 43.25 vs full AQPIM 50.00，importance weighting在aggressive compression (K=128)场景贡献约+0.81 avg points。t=32（sliding window大小）的选择平衡了weight stability和计算开销。

涉及论文标题：
- AQPIM: Breaking the PIM Capacity Wall for LLMs with In-Memory Activation Quantization

