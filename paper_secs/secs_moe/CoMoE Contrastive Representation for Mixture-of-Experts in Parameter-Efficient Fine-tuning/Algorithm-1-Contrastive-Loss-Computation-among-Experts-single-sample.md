# Algorithm 1: Contrastive Loss Computation among Experts (single-sample)

```
Input: Top-k expert indices T \in \mathbb{N}^k
    Expert representations \{E_j(x)\}_{j=1}^n, where E_j(x) \in \mathbb{R}^D
    Temperature \tau
    Output: Contrastive loss \mathcal{L}_{contrast}
 1 r \sim \mathcal{U}\{1, ..., k\}
                                                                                                                                    // Random anchor position
 a \leftarrow T[r]
                                                                                                                                          // Anchor expert index
g \in \text{Normalize}(E_a(x))
4 P \leftarrow \{\text{Normalize}(E_{T[i]}(x)) \mid j \neq r\}
                                                                                                                                 // Positive set, size k-1
 s \ N \leftarrow \{ \text{Normalize}(E_j(x)) \mid j \notin T \}
                                                                                                                                 // Negative set, size n-k
6 s_{\text{pos}} \leftarrow (q \cdot P^{\top})/\tau
7 s_{\text{neg}} \leftarrow (q \cdot N^{\top})/\tau
8 \log \text{its} \leftarrow [s_{\text{pos}}, s_{\text{neg}}]
9 \mathcal{L}_{\text{contrast}} = -\log \left(\frac{\sum \exp(s_{\text{pos}})}{\sum \exp(\log \text{its}) + \varepsilon}\right)
10 return \mathcal{L}_{\mathrm{contrast}}
```

- 2. **Anchor selection.** Uniformly sample an index r from  $\{1, \ldots, k\}$ . Define the anchor (query) vector as  $q = \text{Normalize}(E_{T[r]}(x))$ .
- Positive set. Aggregate the remaining (k 1) expert representations, excluding the one indexed by T[r] from the index set T, into a set P, applying normalization:

$$P = \{ \text{Normalize}(E_{T[j]}(x)) \mid j \neq r \}.$$

4. **Negative set.** Collect and normalize representations from experts not included in the top-*k* indices:

$$N = \{ \text{Normalize}(E_i(x)) \mid i \notin T \}.$$

5. Similarity computation. Compute cosine similarities between anchor vector q and each representation in P and N, scaled by temperature  $\tau$ , yielding similarity scores  $s_{\rm pos}$  and  $s_{\rm neg}$ :

$$s_{\text{pos}} = \frac{q \cdot P^{\top}}{\tau}, \quad s_{\text{neg}} = \frac{q \cdot N^{\top}}{\tau}.$$

6. **InfoNCE loss.** Concatenate the logits and compute the InfoNCE loss as

$$\mathcal{L}_{\text{contrast}} = -\log\left(\frac{\sum \exp(s_{\text{pos}})}{\sum \exp([s_{\text{pos}}, s_{\text{neg}}]) + \varepsilon}\right).$$

where  $\varepsilon$  is a small positive value (e.g.,  $10^{-3}$ ) used to ensure numerical stability, avoiding computational issues caused by the denominator being zero.

## **C** Datasets

Detailed information about the datasets used in the experiments is presented in Table 5. All datasets are downloaded from HuggingFace.

