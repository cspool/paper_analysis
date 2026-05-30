# *A. Query Transformation*

Modern LLMs adopt diverse attention variants [10], [17], [34] with different key–value (KV) sharing patterns. BitDecoding aims to support all these variants.

For instance, in GQA and MQA, multiple query heads share a KV head, reducing the number of KV projections and memory accesses. The degree of sharing is measured by g<sup>q</sup> = hq/hkv, where h<sup>q</sup> and hkv are the numbers of query and KV heads, respectively: g<sup>q</sup> = 1 corresponds to MHA, g<sup>q</sup> > 1 denotes GQA, and hkv = 1 (i.e., g<sup>q</sup> = hq) characterizes MQA.

A challenge arises in decoding: since Q len = 1 (one token at a time), the query tensor has a very small batch dimension, and a naive Q · K<sup>⊤</sup> underfills Tensor Cores, yielding poor warp occupancy and low throughput.

To address this, we perform a *query transformation* that reorganizes the query layout to better match Tensor Core tiling. As illustrated in Fig. 7 (left), we reshape the query tensor from [1,(gq, hkv)] to [gq, hkv], effectively forming a larger Q tile without changing the semantics of attention or its KVsharing pattern. Grouped query heads are then processed in parallel as a larger GEMM block, fully populating Tensor Core fragments, improving warp occupancy, and increasing throughput.

