# <span id="page-15-0"></span>D FlashAttention Decomposition

The Focus attention mask under hard group assignment is:

$$\mathcal{M}(i,j) = \mathbf{1}[j \le i] \land (\mathbf{1}[g(i) = g(j)] \lor \mathbf{1}[i - j \le w])$$
(3)

where g(i) is the group assignment of token i and w is the local window size.

The overlap problem. The natural decomposition into same-group pairs S and local pairs L fails because S ∩ L ̸= ∅—same-group local pairs are double-counted. Subtraction in logsumexp space (log(exp(a) + exp(b) − exp(c))) is numerically catastrophic (cosine similarity 0.79 against reference).

Disjoint decomposition. We split M into two sets that are disjoint by construction:

$$\mathcal{A} = \{(i,j) : j \le i \land g(i) = g(j)\}$$
 (same-group causal) (4)

$$\mathcal{B} = \{(i,j) : j \le i \land i - j \le w \land g(i) \ne g(j)\}$$
 (cross-group local) (5)

A ∩ B = ∅ (one requires same group, the other different group) and A ∪ B = M (every attended pair is either same-group or cross-group-local). The logsumexp merge is mathematically exact.

Set A is computed by sorting tokens by group (stable sort preserves causal order), reshaping into K sequences, and calling flash\_attn\_func with causal=True. Complexity: O(n <sup>2</sup>/K).

Set B extracts local keys for each query and masks same-group pairs to −∞. Complexity: O(nw), never the bottleneck.

Merge: o[i] = (e ℓA[i] · oA[i] + e ℓB[i] · oB[i])/(e <sup>ℓ</sup>A[i] + e ℓB[i] ), where ℓA, ℓ<sup>B</sup> are per-query logsumexp values.

Empirical verification. All configurations achieve cosine similarity 1.0000 against the O(n 2 ) reference, confirming mathematical exactness. The complete implementation is 320 lines of Python using only flash\_attn\_func and standard PyTorch—no custom CUDA kernels, no Triton, no compilation.