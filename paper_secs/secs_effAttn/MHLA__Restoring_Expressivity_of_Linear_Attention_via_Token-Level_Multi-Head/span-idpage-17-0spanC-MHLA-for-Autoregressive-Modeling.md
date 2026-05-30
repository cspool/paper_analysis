# <span id="page-17-0"></span>C MHLA for Autoregressive Modeling

In autoregressive modeling, the causal mask prevents each token from attending to future tokens. While linear attention normally achieves  $O(Nd^2)$  complexity by reusing a global key-value summary, under causal masking, the summary must be recomputed or updated for each prefix, which naively results in  $O(N^2d)$  cost over the full sequence. To avoid this quadratic overhead, a widely adopted solution for linear attention is chunkwise parallel training [46], which splits the sequence into blocks of size C and processes them in parallel to avoid the quadratic cost of recomputing attention over all past tokens. For block b, a local key-value summary is computed as  $S_b = \sum_{j \in b} K_j V_j^{\top} \in \mathbb{R}^{d \times d}$ , and the global summary is updated recursively:

$$S_i^{\text{global}} = S_{i-1}^{\text{global}} + S_i, \qquad H_i = Q_i S_{i-1}^{\text{global}} + (Q_i \tilde{K}_i^\top) V_i.$$

Here, the first term propagates context from preceding blocks via the prefix summary  $S_{i-1}^{\text{global}}$ , while the second term captures intra-block attention. This chunkwise scheme preserves causality and allows block-parallel training with per-block complexity  $O(Cd^2 + C^2d)$ , leading to an overall cost  $O(\frac{L}{C}(Cd^2 + C^2d))$  for a sequence of length L.

MHLA with chunkwise parallel training. MHLA extends this scheme by replacing the single global summary with query-conditioned mixtures of local summaries. Specifically, for block i we form a mixed summary

$$\widetilde{S}_i = \sum_{b \le i} m_{i,b} S_b, \qquad H_i = Q_i \widetilde{S}_{i-1} + m_{i,b} (Q_i \widetilde{K}_i^{\mathsf{T}}) V_i.$$

where  $m_{i,b}$  are the learnable mixing coefficients from the causal coefficient matrix  $\mathcal{M}_c^{\text{causal}}$  (upper-triangular entries masked to enforce causality). Queries in block i then interact only with  $\widetilde{S}_i$ , yielding block-specific, query-adaptive context representations rather than a shared global one. Because the mixing is performed once per block and reused for all tokens in that block, the asymptotic complexity matches that of chunkwise linear attention.

Causal inference. At inference time, we maintain the set of past local summaries  $\{S_1, \ldots, S_{i-1}\}$  and incrementally update the current block summary  $S_i$  as new tokens arrive. When a block is complete, its contribution to future mixtures is fixed and cached. For a new token in block i, we simply update  $S_i \leftarrow S_i + \widetilde{K}_t V_t^{\top}$  and recompute the block's mixed summary  $\widetilde{S}_i$  by applying  $m_{i,i}$  to the incremental update. This avoids recomputation over previous blocks and keeps per-token complexity  $O(d^2)$ .

### <span id="page-17-2"></span>**D** Dataset

<span id="page-17-1"></span>To assess the effectiveness of our approach, we conduct extensive experiments on four tasks: image classification, class-to-image (C2I) generation, text-to-image (T2I) generation, and natural language processing. Following prior works [21, 22, 24], we train classification and C2I models on ImageNet-1K [14] and evaluate them on the standard validation set. For T2I generation, we finetune a pretrained model using a relative small collection of 31,292k images gotten from the internet. For natural language processing, we train models with a subset of SlimPajama [43] with 5B tokens.

