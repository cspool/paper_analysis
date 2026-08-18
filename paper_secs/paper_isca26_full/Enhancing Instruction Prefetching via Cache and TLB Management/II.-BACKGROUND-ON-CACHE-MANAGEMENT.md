# II. BACKGROUND ON CACHE MANAGEMENT

Previously proposed cache replacement policies can be broadly classified in general-purpose policies, prefetch-aware policies, and code-aware policies.

*General-purpose cache replacement policies* drive replacement decisions using i) the block recency without using histories of prior misses [\[19\]](#page-13-17), [\[23\]](#page-13-21)–[\[28\]](#page-13-22) or (ii) features that correlate with the behavior of past accesses and anticipate the reuse distance of cache lines [\[8\]](#page-13-7), [\[29\]](#page-13-23)–[\[40\]](#page-14-0). The state-of-theart general-purpose cache replacement policy for lower-level caches is *Mockingjay* [\[8\]](#page-13-7), a scheme that uses patterns in long PC histories to accurately predict reuse distances.

*Prefetch-aware cache replacement policies* distinguish between demand and prefetch requests to make replacement decisions. PACMan [\[21\]](#page-13-19) dynamically adjusts insertion and promotion policies to mitigate the negative impact of inaccurate prefetches. PACIPlV [\[22\]](#page-13-20) proposes a replacement policy based on an offline exploration considering different insertion and promotion Re-Reference Prediction Values (RRPVs) [\[41\]](#page-14-1) for demand and prefetch lines.

*Code-aware cache replacement policies* target server applications with large code footprints and prioritize code lines over data lines in lower-level caches. The state-of-the-art policies in this category are CLIP [\[20\]](#page-13-18) and Emissary [\[6\]](#page-13-5). CLIP [\[20\]](#page-13-18) is built over the RRIP policy [\[24\]](#page-13-24) and increases the priority of code lines in the L2C at the expense of having more data misses. Emissary [\[6\]](#page-13-5) prevents the eviction of the most critical for performance code blocks from L2C.

