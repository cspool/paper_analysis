# A.1 Algorithm

The pseudo-code of the method is shown in Algorithm [1.](#page-15-0)

## A.2 Implementation Details

Max Pooling of Attention Weights Latest open-source LLMs [\[30,](#page-12-6) [31\]](#page-14-1) have widely adopted Grouped-Query Attention (GQA) [\[32\]](#page-14-2), where multiple query heads share a common pair of keyvalue heads to substantially reduce memory access overhead during inference. In key-value (KV) cache eviction strategies, it's thus often necessary to downscale attention scores from (Q\_head, seq\_len, seq\_len) to (KV\_head, seq\_len, seq\_len). While previous works such as SnapKV [\[3\]](#page-10-2) have predominantly employed mean pooling to aggregate attention scores across query head groups, we hypothesize that max pooling could better preserve the most critical tokens for each query head. Our empirical results demonstrate that max pooling leads to improved performance, and we adopt it for all main experiments.

