# <span id="page-1-0"></span>2 Method: Focus

In standard attention, for a sequence of T tokens, Q, K, V ∈ R <sup>T</sup> <sup>×</sup><sup>d</sup> are projected from hidden states, and each token attends to all others via softmax(QK⊤/ √ d)V, computing all T 2 token pairs. We propose to replace the full T × T score matrix QK<sup>⊤</sup> with two levels: (1) distant tokens attend only if they belong to the same learned group, and (2) nearby tokens always attend to each other within a local window.

**Learned grouping.** Let  $\mathbf{C} \in \mathbb{R}^{K \times d_g}$  be the learnable centroid vectors that define K token groups. A learned projection  $W_g \in \mathbb{R}^{d \times d_g}$  maps tokens into the centroid space. The soft group assignment for token i is:

$$\mathbf{g}_i = \text{normalize}\left(\frac{W_g \mathbf{h}_i \cdot \mathbf{C}^{\top}}{\tau}\right) \in \mathbb{R}^K$$
 (1)

where  $\tau$  is temperature.

We found that softmax normalization leads to group collapse (Section 4), and use **Sinkhorn** normalization to enforce balanced groups as a structural constraint. Given scores  $\mathbf{S} \in \mathbb{R}^{T \times K}$ :

- 1.  $\mathbf{Q} \leftarrow \exp(\mathbf{S}/\tau)$
- 2. For i = 1 to  $N: \mathbf{Q} \leftarrow \mathbf{Q}/\text{sum}(\mathbf{Q}, \text{dim=tokens})$ , then  $\mathbf{Q} \leftarrow \mathbf{Q}/\text{sum}(\mathbf{Q}, \text{dim=groups})$

After N=10 iterations, assignments are approximately doubly-stochastic: both row sums (each token's total assignment) and column sums (each group's total mass) are equalized. This prevents any single group from dominating, while still allowing the LM gradient to learn which tokens belong to which group.

**Gated attention.** The group affinity between tokens i and j is  $\mathbf{g}_i^{\top} \mathbf{g}_j$ : tokens in the same group have high affinity, tokens in different groups have low affinity. We use this to combine local windowed attention with group-gated distant attention:

$$s_{ij} = \mathbf{q}_i^{\top} \mathbf{k}_j \cdot \left( \mathbf{1}_{\text{local}}(i, j) + (1 - \mathbf{1}_{\text{local}}(i, j)) \cdot \sigma(\lambda \cdot \mathbf{g}_i^{\top} \mathbf{g}_j) \right)$$
(2)

Local tokens (within window w) always attend with full attention. For distant tokens in different groups,  $\mathbf{g}_i^{\mathsf{T}} \mathbf{g}_j \approx 0$ , so the gate drives  $s_{ij} \to 0$ —these pairs are pruned. Only same-group distant pairs survive. The gate determines whether information flows; the standard score  $\mathbf{q}_i^{\mathsf{T}} \mathbf{k}_j$  determines how much.

**Separation of routing and attention.** A key design principle is that centroids determine who can attend to whom—routing only. Content flows via the pretrained QKV attention, which determines what information transfers. This separation is why composability works: the pretrained attention computation proceeds unchanged within each group.

Efficiency at inference. Note that during training, soft gating computes all  $O(n^2)$  pairs, and there is no training-time speedup. At inference, each token is assigned to its top-k groups from  $\mathbf{g}_i$ , and two tokens attend only if they share at least one group. Different-group distant pairs are never computed—eliminated entirely, not merely scaled to zero.

The sparsity pattern decomposes into two standard FlashAttention [Dao et al., 2022, Dao, 2024] calls with no custom kernels:

- 1. Local: flash\_attn\_func with sliding window (O(nw)).
- 2. **Group:** Sort tokens by group (stable sort preserves causal order), reshape into K sequences, call flash\_attn\_func with causal=True  $(O(n^2/K))$ .

The key insight is that these two sets are **disjoint by construction**: set  $\mathcal{A}$  (same-group) requires g(i) = g(j), while set  $\mathcal{B}$  (cross-group local) requires  $g(i) \neq g(j)$ . Because  $\mathcal{A} \cap \mathcal{B} = \emptyset$  and  $\mathcal{A} \cup \mathcal{B}$  covers all attended pairs, the logsumexp merge is mathematically exact—no double-counting, no subtraction, no numerical instability. Sorting adds  $O(n \log n)$  overhead, negligible at long sequences (12ms at 1M tokens vs 1.5s for attention). This achieves  $8.6 \times$  speedup at 1M tokens (Table 6; full decomposition details and correctness proof in Appendix D).

<span id="page-3-2"></span>Table 1: Retrofit comparison on GPT-2 124M / PG-19. Focus is the only method that improves PPL and preserves all benchmarks.

| Method                               | Params | PPL ↓ | HellaSwag | ARC-E | PIQA | LAMBADA |
|--------------------------------------|--------|-------|-----------|-------|------|---------|
| Pretrained (full attn)               | 0      | 42.8  | 31.1      | 39.5  | 62.5 | 32.6    |
| Longformer [Beltagy et al., 2020]    | 0      | 38.9  | 30.0      | 37.5  | 58.9 | 6.6     |
| Performer [Choromanski et al., 2021] | 0      | 112.0 | 26.9      | 30.8  | 55.0 | 0.3     |
| Routing Trans. [Roy et al., 2021]    | 0      | 37.4  | 29.6      | 38.3  | 58.4 | 6.4     |
| Full attention FT                    | 124M   | 36.4  | 30.0      | 37.8  | 59.9 | 7.8     |
| Focus (ours)                         | 100K   | 36.2  | 31.1      | 39.5  | 62.5 | 32.6    |

How many dimensions does grouping need? Recall that the projection W<sup>g</sup> ∈ R <sup>d</sup>×d<sup>g</sup> maps tokens into the centroid space. This can be low-rank: rather than using the full d-dimensional space, we project into a small dg-dimensional subspace. On GPT-2 124M, we find that dg=16 suffices:

| dg         | Centroid params | % of model | PPL  |
|------------|-----------------|------------|------|
| 768 (full) | 7.1M            | 5.39%      | 34.8 |
| 128        | 1.2M            | 0.90%      | 34.5 |
| 32         | 296K            | 0.22%      | 34.5 |
| 16         | 148K            | 0.11%      | 34.5 |

A 16-dimensional subspace gives 50× fewer parameters than the full projection with no quality loss. This shows that token grouping is inherently low-dimensional: deciding which group a token belongs to is much simpler than computing attention itself.

