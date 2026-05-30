# <span id="page-16-0"></span>**B** Query-Conditioned Selectivity in Softmax Attention

A key advantage of softmax self-attention is its query-conditioned selectivity. Recall the standard attention formulation:

$$\operatorname{Attn}(Q, K, V)_i = \sum_{j=1}^N \alpha_{ij} v_j, \qquad \alpha_{ij} = \frac{\exp(q_i^\top k_j)}{\sum_{t=1}^N \exp(q_i^\top k_t)}.$$

Two properties are crucial: (i) **Query-conditioned weighting:** each query  $q_i$  produces its own distribution  $\{\alpha_{ij}\}_{j=1}^N$ , so the relative importance of token  $k_j$  is fully dependent on  $q_i$ ; (ii) **Per-token weighting:** the weights act directly on each  $v_j$ , without collapsing V into a global summary. Together, these properties give softmax attention the ability to produce highly adaptive, sharply concentrated context vectors.

By contrast, global linear attention aggregates all tokens into a single summary matrix  $S^{\text{global}} = \sum_{j=1}^{N} \widetilde{K}_{j} V_{j}^{\top}$  shared by all queries, yielding

$$\operatorname{Attn}_{\operatorname{lin}}(Q, K, V)_{i} = \frac{\widetilde{q}_{i}^{\top} S^{\operatorname{global}}}{\widetilde{q}_{i}^{\top} \left(\sum_{j=1}^{N} \widetilde{K}_{j}\right)},$$

where the per-token contributions are no longer explicitly separable by i. As a result, different queries obtain nearly identical context vectors, losing query-conditioned selectivity.

MHLA restores query-conditioned selectivity. MHLA bridges this gap by introducing a learnable coefficient matrix  $\mathcal{M}_c$  that forms query-block-specific mixtures of local summaries:

$$\widetilde{S}_i = \sum_{b=1}^{M} m_{i,b} S_b \qquad \Rightarrow \qquad \operatorname{Attn}_{\mathrm{MHLA}}(Q, K, V)_i = \widetilde{q}_i^{\top} \widetilde{S}_i.$$

Because  $m_{i,b}$  varies with the query block i, MHLA assigns different effective weights to the same token depending on the querying block. Expanding  $S_b$  into its token-level definition gives

$$\widetilde{q}_i^{\top} \widetilde{S}_i = \sum_{t=1}^N m_{i,b(t)} (\widetilde{q}_i^{\top} \widetilde{K}_t) V_t^{\top},$$

revealing a two-stage weighting mechanism: (i) block-level selection  $m_{i,b(t)}$  that is query-conditioned, followed by (ii) within-block token reweighting via the kernel inner product  $\widetilde{q}_i^{\top} \widetilde{K}_t$ . This design reintroduces query-conditioned selectivity and per-token weighting while preserving the linear-time complexity of kernelized attention.

