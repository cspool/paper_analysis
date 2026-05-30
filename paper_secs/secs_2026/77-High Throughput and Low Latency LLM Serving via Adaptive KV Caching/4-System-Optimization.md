# 4 System Optimization

## 4.1 Optimization Framework

The eLLM framework co-optimizes throughput and latency for LLMs deployed on a cluster of N GPUs, each with memory capacity  $M_G$ . During each decoding iteration, when B requests reside in the waiting queue, eLLM dynamically maximizes output token throughput while ensuring TPOT SLOs. This dual-goal optimization is formalized as:

<span id="page-5-1"></span>
$$\max_{(b,r,\delta)} \frac{b}{T(b,r,\delta)}$$
s.t.  $T(b,r,\delta) + WT_{max} \leq \text{SLO},$ 

$$\sum_{i=1}^{b} 4Lhs_i(1-r) + M_W \leq M_G \cdot N - M_o,$$

$$0 \leq r \leq 1,$$

$$1 \leq b \leq B.$$
(1)

In this formulation, the optimization variables are b, r, and  $\delta$ . The batch size b determines the number of concurrent

requests processed per iteration. The uncached token ratio r specifies how many previous tokens are recomputed, with 1-r representing cached tokens stored via KV states.  $\delta$  governs threads number for recomputation kernel (K1 in Fig. 5(a)) of kernel fusion. Together, these variables influence the latency of processing batch b. We define  $T(b,r,\delta)$  as the total batch processing time, and the objective function maximizes throughput as  $\frac{b}{T(b,r,\delta)}$ .

The first constraint guarantees that the total latency for batch b, including processing and waiting time  $(WT_{max})$ , adheres to the SLO. The second enforces GPU memory limits: cached tokens for b requests  $\sum_{i=1}^{b} 4Lhs_i(1-r)(4$  means K and V caches in half-precision, hence  $2\times 2$  bytes per element), model weights  $(M_W)$ , and memory space for overlapping and kernel fusion  $(M_o)$  must fit within the cluster's total memory  $(M_G \cdot N)$ . Here, L, h, and s denote the number of layers, hidden dimension, and maximum sequence length in the batch, respectively. The remaining constraints ensure that the batch size b does not exceed the number of requests (B) in the waiting queue.

Directly solving this online mixed-integer nonlinear programming problem is computationally prohibitive due to the dynamic nature of the inference process  $(C_1)$  and the interdependencies among variables  $(C_2)$ . To address these challenges, eLLM adopts a dual-level optimization framework. For  $C_1$ , eLLM introduces a request-level optimization strategy (detailed in § 4.2) that dynamically adjusts the batch size (b) and uncached token ratio (r) based on explicit quantification of recomputation and decoding costs. This strategy focuses on optimizing b and r while disregarding the thread configuration variable  $\delta$  to improve optimization efficiency.

For the second challenge  $C_2$ , eLLM incorporates a layer-level optimization strategy (discussed in § 4.3) that optimizes kernel fusion and communication-computation overlap at the layer granularity. This approach enables precise estimation of  $M_0$ , which is then fed back to the request-level optimization. Initially,  $M_0$  is set to the maximum value 40hbsr, but after layer-level refinement, the reduced  $M_0$  frees up memory. This allows the system to iteratively increase the batch size b and decrease the uncached token ratio r, creating a feedback loop that not only enhances system throughput but also resolves the interdependence between token-wise caching decisions and kernel fusion efficiency.

