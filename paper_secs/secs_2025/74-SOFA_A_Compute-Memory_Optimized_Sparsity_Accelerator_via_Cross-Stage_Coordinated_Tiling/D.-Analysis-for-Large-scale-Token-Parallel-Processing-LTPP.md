# D. Analysis for Large-scale Token Parallel Processing (LTPP)

Despite its promising adaptability, dynamic sparsity incurs additional overhead during inference, due to the *Pre-compute* and *Top-k stages*. As a result, previous works [23], [28], [29], [33], [34] were constrained to processing queries with low parallelism, to minimize the memory and computation overhead. However, as modern LLMs demand significantly longer context than before (GPT4 32k [42], LongLLaMa 256k [43]), the rapid processing of these extended context becomes increasingly crucial [44]. This highlights the necessity for accelerators with LTPP capabilities. However, the current dynamic sparsity attention workflow poses three challenges for LTPP. Illustrated in Fig. 2:

- 1) Supposing processing T tokens in parallel, the precompute and sorting complexity rises to  $\mathcal{O}(TSH)$  and  $\mathcal{O}(TSSk)$ , respectively. Taking Llama-13B  $(T=512,\ k=0.25)$  as an example, the required numbers of comparisons and multiplication would be over  $10^{11}$  and  $10^8$ , respectively. In this case, prediction requires performing over  $2^{11}$  MACs and  $2^{10}$  comparisons, accounting for more than 57% of the total execution latency. Such prohibitive overhead will negate the improvements brought by sparsity.
- 2) As *top-k* sorting and *softmax* is applied row-wise, matrices **Pre-Atten** and **A** must be stored to DRAM first and then loaded by row blocks, thus leading to massive DRAM access. Such extensive memory access would lead to inefficient inference. In 45 nm CMOS technology, the energy cost of a DRAM access (5 to 20 pJ/bit) is two orders of magnitude higher than that of internal cache access (0.1 pJ/bit) [45], while its bandwidth (DDR4 25.6GB/s) is also orders-of-magnitude lower than the SRAM (19TB/s) [39]. A coarse scheme is to enlarge the on-chip SRAM capacity but this would lead to area inefficiency. Taking (T=512, S=2048) for instance, it directly necessitates 5MB SRAM, leading to 5.47 mm<sup>2</sup> footprint under TSMC 28nm technology, which is  $7.4 \times$ ,  $8.9 \times$  of the overall area of SpAtten [33] and ELSA [29], respectively.

 $\label{table I} \textbf{TABLE I} \\ \textbf{SUMMARY FOR SOTA TRANSFORMER ACCELERATORS}.$ 

|               | Optimization |           |     |           |       |  |  |  |
|---------------|--------------|-----------|-----|-----------|-------|--|--|--|
| Accelerator   | C            | ompute    | N   | Cross     |       |  |  |  |
|               | QKV          | Attention | QKV | Attention | Stage |  |  |  |
| $A^{3}[28]$   | ×            | ✓         | ×   | ×         | ×     |  |  |  |
| ELSA [29]     | ×            | ✓         | ×   | ×         | ×     |  |  |  |
| Sanger [30]   | ×            | ✓         | ×   | ×         | X     |  |  |  |
| DOTA [31]     | ×            | ✓         | ×   | ×         | ×     |  |  |  |
| Energon [34]  | ×            | ✓         | ×   | Low       | ×     |  |  |  |
| DTATrans [32] | ×            | ✓         | ×   | ×         | ×     |  |  |  |
| SpAtten [33]  | ✓            | ✓         | ×   | Low       | X     |  |  |  |
| FACT [23]     | <b>√</b>     | ✓         | ×   | ×         | ×     |  |  |  |
| SOFA          | /            | /         | ~   | <b>V</b>  | /     |  |  |  |

3) FlashAttention2 (FA-2) employs a tiling scheme for the *softmax* operation to keep the working set of data in the faster on-chip memory, thus successfully reducing off-chip memory access overhead. However, the benefits come with soaring computation costs, making it unsuitable for dynamic sparsity scenarios in LTPP. As an example, when the tile size is  $B_c=4$  for a sequence length S=1024, FA-2 must frequently compute and compare values across these tiles to ensure correct global results. This leads to a computational load approximately  $1.5\times$  higher than that of a regular implementation without tiling.

We argue that the main bottleneck in extending existing dynamic sparsity methodology towards LTPP lies in information decoupling among stages, thus missing the cross-stage-tiling opportunity. Table I offers an overview of the effectiveness of existing approaches in optimizing Transformer components. Works [28]-[31] focus on reducing pre-computation overhead, such as ELAS [29] using Binary Hash, A<sup>3</sup> [28] employing Greedy search and DOTA [31] using low-rank transformation. However, these methods still cannot address the row dependency of key operators, like topk and softmax, thus still resulting in significant memory access overhead under LTPP. Further, SpAtten [33] and DTATrans [32] involve sorting the cumulative distribution probabilities of tokens, introducing substantial sorting complexity and latency in LTPP scenarios. While SpAtten [33] and Energon [34] realize challenges with extensive memory access, their sparsity strategies fail to handle the severe memory access overhead with the LTPP scenario. In summary, existing works are all limited on individual-stage optimization, thereby overlooking opportunities for cross-stage joint optimizations, making them inadequate for supporting LTPP. This motivates us to propose a cross-stage compute-memory efficient accelerator design, targeting the LTPP scenario.

