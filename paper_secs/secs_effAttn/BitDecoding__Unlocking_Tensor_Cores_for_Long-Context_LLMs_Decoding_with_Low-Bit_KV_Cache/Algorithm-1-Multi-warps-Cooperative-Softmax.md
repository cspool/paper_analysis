# Algorithm 1 Multi-warps Cooperative Softmax

```
Require: sTMP \in \mathbb{R}^{W_n} and sAcc \in \mathbb{R}^{T_m \times T_n} in SMEM.

Require: Load Q_i \in \mathbb{R}^{T_m \times d} and K_i, V_i \in \mathbb{R}^{T_n \times d} to REG.

1: S_i = Q_i K_j^T where S_i \in \mathbb{R}^{T_m \times T_n}.

2: m_i^{new} = \max(m_i, \operatorname{rowmax}(S_i, sTMP)).

3: P_i = \exp(S_i - m_i^{new}) where P_i \in \mathbb{R}^{T_m \times T_n}.

4: sAcc = \operatorname{tiled\_copy\_r2s}(P_i).

5: P_i' = \operatorname{tiled\_copy\_s2r}(sAcc)

6: O_i^{new} = P_i'V_j + \operatorname{diag}(e^{m_i - m_i^{new}})O_i.
```

## V. System Implementation

In this section, we describe how we implement BitDecoding, as illustrated in Fig. 7. Our implementation consists of three major components: (i) a *query transformation* component that supports diverse attention variants in LLMs; (ii) a *Residual Kernel* that performs low-cost quantization and packing while remaining general to both tensor-wise and channel-wise scaling across quantization algorithms; and (iii) a *Packing Kernel* with a fine-grained pipeline that fully utilizes both Tensor Cores and CUDA cores. Finally, we discuss architecture-specific optimizations that leverage the advanced features of the latest GPU generations (e.g., Hopper and Blackwell) to further enhance decoding throughput.

![](_page_6_Figure_0.jpeg)

Fig. 7: System overview of BitDecoding. (1) Query Transformation restructures the query tensor layout to enable efficient warp-level execution for attention variants on Tensor Cores. (2) Residual Kernel performs quantization and packing with minimal overhead, supporting both tensor-wise and channel-wise scaling. (3) Packing Kernel executes dequantization and matrix multiplication using a fine-grained, asynchronous pipeline, maximizing Tensor Cores and CUDA Cores utilization with low-bit parameters.

## *A. Query Transformation*

Modern LLMs adopt diverse attention variants [10], [17], [34] with different key–value (KV) sharing patterns. BitDecoding aims to support all these variants.

For instance, in GQA and MQA, multiple query heads share a KV head, reducing the number of KV projections and memory accesses. The degree of sharing is measured by g<sup>q</sup> = hq/hkv, where h<sup>q</sup> and hkv are the numbers of query and KV heads, respectively: g<sup>q</sup> = 1 corresponds to MHA, g<sup>q</sup> > 1 denotes GQA, and hkv = 1 (i.e., g<sup>q</sup> = hq) characterizes MQA.

A challenge arises in decoding: since Q len = 1 (one token at a time), the query tensor has a very small batch dimension, and a naive Q · K<sup>⊤</sup> underfills Tensor Cores, yielding poor warp occupancy and low throughput.

To address this, we perform a *query transformation* that reorganizes the query layout to better match Tensor Core tiling. As illustrated in Fig. 7 (left), we reshape the query tensor from [1,(gq, hkv)] to [gq, hkv], effectively forming a larger Q tile without changing the semantics of attention or its KVsharing pattern. Grouped query heads are then processed in parallel as a larger GEMM block, fully populating Tensor Core fragments, improving warp occupancy, and increasing throughput.

