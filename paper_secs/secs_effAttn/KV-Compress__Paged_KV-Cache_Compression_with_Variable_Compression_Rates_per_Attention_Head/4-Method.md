# 4 Method

The method presented in this paper is a modification of Ada-SnapKV [1]. We first discuss query-group compression, then detail our adaptation of PagedAttention that makes variable-head-rate eviction practical. Finally we go over the remaining algorithm choices made in our final method.

#### 4.1 Query Group Compression

While GQA has seen widespread adoption over the last year in models such as Llama and Mistral, existing methods for KV cache compression were not designed with a GQA KV cache in mind. Implementations of both SnapKV and PyramidKV, for example, cache and compress KV tensors *after* repetition is carried out for alignment with the query tensor. This means that: 1. The compression rate must exceed the ratio of query heads to KV heads in order to improve upon the compression that would be achieved by using a more efficient framework where KVs are not physically repeated in memory. 2. There is a great deal of redundancy in cache before compression occurs (in the case of Mistral and Llama-3,  $\frac{3}{4}$  of KVs are duplicates) and this redundancy is not being taken advantage of in the compression.

We seek a compression method where KVs are evicted from a non-repeated cache that is applicable to current GQA models run in state-of-the-art inference frameworks. This can be done with a straightforward modification to existing eviction-based methods, where the metrics used to determine KV eviction are aggregated for each key over queries in that key's respective query group. We can then continue with compression of the non-repeated cache, using the aggregate metric to inform eviction decisions.

Following this modification, equation [6](#page-6-0) becomes

$$M_{h_k,j}^{(full)} = \sum_{i} \sum_{h \in H_k} A_{hij} , \quad \text{for } H_k = \{ \forall h : rh_k \le h < r(h_k + 1) \}$$
 (9)

where we add an additional summation over the metrics computed for all queries in the current key's query group, Hk. Similarly, equation [7](#page-6-1) becomes

$$M_{h_k,j}^{(w)} = \sum_{i=s}^{L} \sum_{h \in H_k} A_{hij} , \quad \text{for } s = L - w$$
 (10)

<span id="page-8-0"></span>
$$H_k = \{ \ \forall \ h : rh_k \le h < r(h_k + 1) \ \} \ . \tag{11}$$

## 4.2 Supporting Variable-Head-Rate Eviction

Ada-SnapKV explores evicting a variable number of KVs from each attention head of the KV cache. Selecting KVs to evict in this case can be done by sorting metrics over a flattened tensor where head and sequence length dimensions are combined, then selecting KVs corresponding to the first eH metrics for eviction. Unlike Ada-SnapKV, we seek to additionally support variable rates of compression across *layers*, following this same methodology. In this section we discuss the steps taken to make such compression feasible.

