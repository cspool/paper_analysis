# 4 AttenIO Architecture

#### 4.1 Overview

Figure 6 presents an overview of AttenIO, an I/O-driven accelerator designed for the exact self-attention mechanism on long input sequences. The key components of AttenIO include a controller, a PE array, an EXP unit, a KV buffer, and an on-chip cache. The controller implements the I/O-optimal dataflow to minimize I/O operations between off-chip memory and the on-chip cache. It also coordinates the execution of computations and data movement to enable three levels of fine-grained communication-computation overlapping during self-attention with online softmax. The KV buffer alternately stores one block of K and V, increasing opportunities for overlap between computation and data movement. The PE array supports both matrix processing and the general arithmetic operations required by softmax. The EXP unit consists of multiple exponential modules [44, 72, 75] that compute the exponentials involved in the softmax function. Moreover, the PE array and EXP unit work together to support parallel patterns in a pipelined fashion, enabling efficient softmax execution. Together, these components enable reduced I/O operations, fine-grained communicationcomputation overlapping, and parallel softmax execution.

### 4.2 Communication-Computation Overlapping

To hide the long-latency of I/O operations and achieve high performance in processing long-sequence attention, AttenIO employs three levels of fine-grained communication-computation overlapping: intra-inner-iteration, inter-inner-iteration, and inter-outer-iteration, as illustrated in Figure 7. To support preloading one block of K or V while the other

<span id="page-8-0"></span>![](_page_8_Figure_2.jpeg)

Figure 8. Organization of the PE array.

block of V or K is used for computations, aligned with the I/O-optimal dataflow, we introduce a KV buffer that alternately stores a single block (d elements) from K and V.

**Intra-Inner-Iteration Overlapping.** This level optimizes operations within a single iteration of the inner loop (iteration of j in Algorithm 1). While the PE array computes  $S_i^{(j)} = Q_i K_j^T$  using  $K_j$  from the KV buffer (line 9), the controller initiates preloading of the next required block of V ( $V_j$ ) into the cache (line 12). Once the computation of  $S_i^{(j)}$  is completed, the preloaded  $V_j$  is immediately moved from the cache to the KV buffer for the following computations. This overlapping ensures that data loading and computation proceed concurrently within the same inner-loop iteration.

**Inter-Inner-Iteration Overlapping.** This level optimizes operations across different iterations of the inner loop (iteration of j in Algorithm 1). During the computations of  $O_i$  in the current iteration j (line 13), the controller preloads the next required block of  $K(K_{j+1})$  into the on-chip cache (line 8). Once the computations involving  $V_j$  are completed, the space in the KV buffer is freed for  $K_{j+1}$ , and the preloaded  $K_{j+1}$  is immediately moved from the cache to the KV buffer. This overlapping reduces I/O stalls between inner-loop iterations.

**Inter-Outer-Iteration Overlapping.** This level focuses on optimizing the operations between different blocks of the outer loop (iteration of i in Algorithm 1). As the PE array performs the final computations of  $O_i = \text{diag}(\ell_i)^{-1}O_i$  for the current block i (line 15), the controller releases the  $Q_i$  and initiates loading of  $Q_{i+1}$  into cache (line 6). This overlapping reduces idle time between outer-loop iterations, thus increasing utilization of the PE array.

## 4.3 PE Array Organization

Figure 8 shows the organization of the PE array, consisting of pq MACs arranged in p MAC lines, each containing q MACs. The PE array primarily computes  $Q_iK_j^T$  and  $\tilde{P}_i^{(j)}V_j$  following our proposed I/O-optimal dataflow (detailed in Section 3.3).

For the computation of  $Q_iK_j^T$ , the PE array performs the operation over d iterative steps. In each step, each MAC computes one element of the ab partial results of  $S_i^{(j)}$ . According to the I/O-optimal dataflow, the optimal value of b is 1, meaning each row of  $K_j^T$  contains only one element. In each step, each MAC fetches a different element from a column of  $Q_i$  from the on-chip cache, while one element from a corresponding row of  $K_j^T$  is fetched from the KV buffer and broadcast across all MACs for multiplications. The products

<span id="page-8-1"></span>![](_page_8_Picture_11.jpeg)

**Figure 9.** Executing online softmax with parallel patterns across pipelined stages: (a) Updating  $m_i$  and computing  $\tilde{P}_i^{(j)}$ , and (b) Updating  $\ell_i$ .

are then accumulated into the existing partial sums of  $S_i^{(j)}$ . This process repeats until all partial results are fully accumulated, yielding the final  $S_i^{(j)}$ . The subsequent computation of  $\tilde{P}_i^{(j)}V_j$  involves multiplying  $\tilde{P}_i^{(j)} \in \mathbb{R}^{a \times 1}$  and  $V_j \in \mathbb{R}^{1 \times d}$ . Each MAC is assigned a different element of  $\tilde{P}_i^{(j)}$ , while in each step, one element of  $V_j$  is fetched from the KV buffer and broadcast across all MACs. This produces a partial results in each step and continues for d steps.

### 4.4 Softmax with Parallel Patterns

By employing the I/O-optimal dataflow, the need for rowwise reductions in softmax computation is eliminated, converting online softmax into a sequence of element-wise operations. This transformation enables parallel patterns, where independent element-wise operations are structured across pipeline stages, thereby improving the overall efficiency of softmax processing. Figure 9 illustrates the execution of online softmax computation with parallel patterns.

In Figure 9(a), three parallel patterns are applied for updating  $m_i$  and then computing  $\tilde{P}_i^{(j)}$ , corresponding to lines 10 and 11 of Algorithm 1. As discussed in Section 3.3, the I/O-optimal dataflow is achieved when b equals 1, resulting in  $S_i^{(j)} \in \mathbb{R}^{a \times 1}$ . This eliminates the need for rowmax( $S_i^{(j)}$ ), as each row contains only a single element, thereby avoiding row-wise traversal and reducing computational overhead. All calculations for updating  $m_i$  and computing  $\tilde{P}_i^{(j)}$  become independent and element-wise. Based on this insight, three parallel patterns are introduced to exploit data-level parallelism: calculating element-wise maximum values between  $m_i^{\text{old}}$  and  $S_i^{(j)}$ ; computing the element-wise difference between  $S_i^{(j)}$  and  $m_i$ ; and calculating the exponential of each element to obtain  $\tilde{P}_i^{(j)}$ .

Figure 9(b) illustrates four parallel patterns for updating  $\ell_i$ , corresponding to line 11 of Algorithm 1. Similar to the

Table 2. Configuration of AttenIO.

<span id="page-9-1"></span>

| Components      | Configuration                                       |
|-----------------|-----------------------------------------------------|
| PE Array        | 64×32 MACs, 1 GHz                                   |
| EXP Unit        | 128 EXP Modules, 1 GHz                              |
| KV Buffer       | 0.25 KB                                             |
| On-Chip Cache   | 512 KB                                              |
| Off-Chip Memory | 128 GB/s, 16 64-bit HBM channels, 8GB/s per channel |

analysis of  $S_i^{(j)}$ , we have  $\tilde{P}_i^{(j)} \in \mathbb{R}^{a \times 1}$ . Therefore, there is no need to traverse all elements of a row in  $\tilde{P}_i^{(j)}$  for processing rowsum( $\tilde{P}_i^{(j)}$ ); all calculations for updating  $\ell_i$  are performed using four element-wise parallel patterns, including: calculating the difference between  $m_i$  and  $m_i^{\text{old}}$ ; computing the exponential of the results from the previous stage; performing multiplication of the exponential results with  $\ell_i$ ; and aggregating the results with  $\tilde{P}_i^{(j)}$  to update  $\ell_i$ .

