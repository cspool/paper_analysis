# <span id="page-6-1"></span>Algorithm 1 Top-p via Binary Search.

```
Input: normalized attention weights W ∈ R
                                          BS×H×N ,
top-p threshold p, hyper-parameter ϵ.
Output: indices I, mask M ∈ {0, 1}
                                  BS×H×N .
l = 0, r = max(W), m = (l + r)/2;
repeat
  W0 = where(W < m, 0.0, W);
  W1 = where(W ≤ l, INF, W);
  W2 = where(W > r, −INF, W);
  if sum(W0) ≥ p then
    l = m;
  else
    r = m;
  end if
until max(W2) − min(W1) ≥ ϵ
Select indices I and set mask M where W ≥ l;
return I, M;
```

Dynamism. The top-p Pruner enables head-wise dynamic budgets, but also raises load imbalance issues in the attention kernel. Traditional implementations allocate uniform computation resources to all heads. FlashInfer [\[47\]](#page-16-12) deeply investigates this load imbalance problem, but only for requests with dynamic lengths. Twilight further reuses the load balancing algorithm in FlashInfer to address head-wise dynamism, by flattening the head dimension.

#### <span id="page-6-0"></span>4.3 Overhead Analysis and Discussion

Execution Time. The execution time of Twilight consists of three parts according to the pipeline in [Figure 5:](#page-5-1) TTokenSel + TPruner + TSparseAttn. Compared to the baseline sparse attention without Twilight, our method introduces an extra latency term TPruner but reduces TSparseAttn. Our hierarchical architecture naturally matches the hierarchical sparsity, where the number of tokens gradually decreases as the precision increases. Suppose the base algorithm in the Token Selector estimates token importance with a 1/16 sparsity and/or precision reduction. Then the theoretical speedup can be formulated as N/16+B<sup>0</sup> N/16+B0/4+B<sup>1</sup> , where B<sup>0</sup> = |I0| is the budget of the base Token Selector, and B<sup>1</sup> = |I1| is the budget after pruned by Twilight with INT4. Assuming B<sup>0</sup> = N/4 and B<sup>1</sup> = N/64, the speedup would be approximately 2×. Here we omit the overheads of the top-p kernel since SpGEMV dominates the latency when B<sup>0</sup> is around N/8 to N/4.

Memory Overheads. Twilight introduces an extra INT4 quantized K cache, which brings a 1/2 × 1/4 = 1/8 extra KV cache memory overhead. However, this additional cost does not appear in all cases. First, some base algorithms, like DS [\[12\]](#page-14-5), already maintain an INT4 K cache. Second, some recent efforts have explored INT4 full attention [\[41\]](#page-16-6). This allows us to directly reuse the estimated attention weights calculated by the INT4 K cache in the attention computation, without maintaining the original FP16 K cache. Moreover, offloading and selective quantization (e.g., keeping the extra INT4 K cache only for hot tokens) can be leveraged if the GPU memory becomes a bottleneck, which we leave as future work.

Integration with LLM Serving Systems. Our system design naturally aligns with PagedAttention [\[44\]](#page-16-9), so Twilight can be seamlessly integrated into popular serving systems like vLLM [\[44\]](#page-16-9) and SGLang [\[45\]](#page-16-10). Other common techniques, such as prefix sharing and multi-phase attention [\[48,](#page-16-13) [45,](#page-16-10) [49,](#page-16-14) [50,](#page-16-15) [51\]](#page-16-16), are also compatible with Twilight since we use page-level or token-level sparse operations, and can achieve a flexible computation flow.

