# <span id="page-6-0"></span>4.2 Request-level Optimization

<span id="page-6-2"></span>**4.2.1 Inference latency estimation.** It is essential to estimate inference latency as a function of batch size b and the ratio of uncached tokens r. LLM services, which predominantly utilize transformer architectures (as detailed in § 2.1), incur computational costs during both the prefilling and decoding stages. These costs depend on various factors, including model parameters (e.g., the number of layers L, hidden dimension h, and vocabulary size V), sequence length s, and the type of hardware accelerators (e.g., GPUs) employed

for processing. The total floating-point operations (FLOPs) required to process a batch can be determined based on the model's parameters, input prompt length, and batch size b. Since each GPU type has a unique floating-point operations per second (FLOPS) capacity, latency can be approximated by dividing the total FLOPs by the GPU's computational capacity. This relationship provides a practical method to estimate latency for a batch of requests under specific configurations.

When employing adaptive token-wise caching, uncached tokens must be recomputed and concatenated with cached tokens to generate new output tokens. For a request i with request length  $s_i$ , each inference iteration involves two primary stages: prefilling for  $s_i r$  tokens and decoding for a new token with previous cached tokens  $s_i$ . The computational costs can be estimated as follows.

During the recomputation phase (corresponding to the prefilling stage), the FLOPs required for the Attention module are calculated as  $8s_irh^2 + 4s_i^2r^2h$ . For the MLP module, which consists of two linear layers, the total FLOPs amount to  $16s_irh^2$ . Additionally, the final output layer, a linear layer that maps the hidden state to the vocabulary dimension to generate logits for each token, contributes  $2s_irhV$  FLOPs [12, 38]. Consequently, the **recomputation volume** for b requests across all L layers can be expressed as:

<span id="page-6-3"></span>
$$V_{b,r}^{\text{rec}} = \sum_{i=1}^{b} \left( 24h^2 L s_i r + 4h L s_i^2 r^2 + 2h V s_i r + \epsilon_0 \right). \tag{2}$$

For decoding, the FLOPs for the Attention module differ from those in the prefilling phase, as KV are cached, the FLOPs are  $8h^2 + 4h(s_i + 1)$ . For the MLP and output layers, the FLOPs are  $16h^2$  and  $2s_ihV$ , respectively. Thus, the **decoding volume** for b requests with all L layers can be expressed as:

<span id="page-6-4"></span>
$$V_{b,r}^{\text{dec}} = \sum_{i=1}^{b} \left( 24h^2L + 4hL(s_i + 1) + 2hVs_i + \epsilon_1 \right).$$
 (3)

In these formulas,  $\epsilon_0$  and  $\epsilon_1$  denote the computational overhead associated with minor operations, such as Layernorm, which are omitted from the theoretical cost models.

Putting everything together, the latency for a batch of requests *b* with uncached token ratio *r* is expressed as:

$$T(b,r) = \frac{1}{\text{FLOPS}} \left( V_{b,r}^{\text{rec}} + V_{b,r}^{\text{dec}} \right). \tag{4}$$

**4.2.2 Batching and token-wise caching.** To efficiently address the optimization problem while incorporating inference latency estimation, we reformulate the objective by transforming the maximization into an equivalent minimization problem. This transformation leverages standard optimization frameworks, facilitating faster convergence during iterative solving. The reformulated, simplified objective is:

<span id="page-6-1"></span>
$$\min_{(b,r)} \frac{\sum_{i=1}^{b} \left[ 24h^2L(s_ir+1) + 4hL(s_i^2r^2 + s_i + 1) + 2hVs_i(1+r) \right]}{\text{FLOPS} \cdot b}$$

subject to the constraints defined in Eq. (1). eLLM adopts First-Come, First-Served (FCFS) as its default scheduling

<span id="page-7-1"></span>![](_page_7_Figure_2.jpeg)

(a) Layer-wise overlapping (b) KV cache swap overhead

**Figure 8.** Illustration of the layer-wise recomputation-communication overlap.  $T_R$  and  $T_S$  mean time for recomputation and swapping, respectively in (a).

policy for incoming requests, though its architecture supports seamless integration of alternative scheduling strategies. Under FCFS, the sequence length  $s_i$  for each request is initialized during the prefill phase upon its arrival order and dynamically updated in subsequent decoding iterations. In each decoding iteration, eLLM substitutes the requestspecific information ( $s_i$ , B,  $WT_{max}$ ) from the waiting queue, the model parameters  $(M_W, h, L, V)$ , and the GPU details (FLOPS,  $M_G$ , and N) into Eq. (5). The resulting optimization problem is then efficiently solved using the Sequential Least Squares Programming (SLSQP) solver [14] provided by the Scipy library [44]. The optimized values of b and r are subsequently used in each decoding iteration to update the batched requests via the Request Batching component and manage uncached tokens through the Token-wise Caching component.

## <span id="page-7-0"></span>4.3 Layer-level Optimization

Building on request-level optimizations, eLLM incorporates layer-level optimizations designed to maximize computation-communication overlap and refine thread allocation strategies for fused kernels.

**4.3.1 Layer-wise overlapping.** To fully leverage host memory concurrently, eLLM integrates an offloading mechanism that selectively transfers KV states to host memory while overlapping transfer costs with computation. After the prefill phase, where a request's prompt tokens are processed, the KV caches for all layers of the cached tokens are stored in GPU memory. For uncached tokens (with a ratio of *r*), several layers of their KV states can be temporarily offloaded to host memory and reloaded as needed, while other layers are recomputed directly on the GPU during decoding. This layer-wise overlapping approach enables eLLM to minimize the total communication and computation costs.

Since the costs of recomputation and communication vary based on the number of tokens and layers involved, eLLM dynamically determines the optimal layers for recomputation and swapping between host and GPU memory. As illustrated in Fig. 8(a), eLLM can, for instance, recompute layers i and i + 1 while simultaneously swapping layer i + 2 from host to GPU, or alternatively recompute layer i alone while swapping layers i + 1 and i + 2. However, suboptimal layer

<span id="page-7-5"></span>![](_page_7_Figure_10.jpeg)

**Figure 9.** Layer-level optimization.

configurations can lead to *time bubbles* during overlapping operations (e.g., ① and ② in Fig. 8(a)). These bubbles occur when significant mismatches between recomputation time ( $T_{Rec}$ ) and swapping time ( $T_{Swap}$ ) result in idle periods. In contrast, well-matched configurations, such as ③ in Fig. 8(a), achieve near-equality between  $T_{Rec}$  and  $T_{Swap}$ , effectively minimizing idle time. Accurate estimation of these latencies is thus critical to optimizing overlap efficiency.

**Swapping Latency.** The profiling results for Llama2-13B on an NVIDIA A100 GPU reveal a linear relationship between swap time and the number of tokens transferred when moving KV caches from GPU to host memory, as shown in Fig. 8(b). Based on this observation, eLLM models the overhead for layer-wise KV cache swapping using the following linear equation parameterized by the number of layers:

<span id="page-7-2"></span>
$$T_{k,l_1} = (\alpha k + \beta) \cdot l_1/L, \tag{6}$$

where  $l_1$  denotes the number of layers being swapped, k is swapped token number, and parameters  $\alpha$ ,  $\beta$  are pre-profiled constants specific to the LLM and hardware configuration.

**Recomputing Latency.** The recomputation latency is influenced by three critical parameters: the current batch size b, the uncached token ratio r, and the number of layers  $l_2$  requiring recomputation. As analyzed in § 4.2.1, this latency can be quantified as:

<span id="page-7-3"></span>
$$T_{b,r,l_2} = \frac{1}{\text{FLOPS}} \sum_{i=1}^{b} \left[ \left( 24h^2 s_i r + 4h s_i^2 r^2 \right) \frac{l_2}{L} + 2s_i r h V \right]. \tag{7}$$

By leveraging the latency models presented in Equations (6) and (7), eLLM identifies the optimal layer configurations for recomputation and swapping to minimize overlapping time bubbles. To reduce search overhead and align with the kernel fusion design (detailed in § 4.3.2), recomputation is restricted to at most two layers, and swapping is limited to three layers simultaneously. This constraint yields six feasible configurations. The *Comm-Com Overlapping* algorithm evaluates all valid combinations of layers and selects the configuration that minimizes the discrepancy between swapping latency and recomputation overhead.

<span id="page-7-4"></span>**4.3.2 Layer-wise kernel fusion.** To optimize the recomputation of uncached tokens (r), eLLM leverages kernel fusion with the decoding kernel to maximize SM utilization. As illustrated in Fig. 9, this layer-level optimization integrates overlapping policies to accelerate inference. Specifically, eLLM mitigates kernel launch overhead and memory access time between adjacent layers (e.g., Layers i and i+1 in Fig. 9) through vertical fusion. Meanwhile, eLLM integrates horizontal fusion of recomputing and decoding

kernels (e.g., K1 and K2 in Fig. 9) with complementary resource demands. To implement this, additional GPU memory is required to store the KV states of swapping layers  $4hbsrl_1$ , recomputed layers for K1 ( $4hbsrl_2$ ), and decoding layers for K2 ( $4hbsr(l_1 + l_2)$ ), resulting in a total memory consumption of  $8hbsr(l_1 + l_2)$ . The fused kernel executes concurrently with memory swapping operations across two CUDA streams, enabling parallelized memory management and kernel execution.

However, the computational volumes of K1 and K2 exhibit a marked disparity, as they correspond to distinct computational logics. For the NVIDIA GPU architecture, each SM can execute multiple thread blocks in parallel. A grid comprises multiple thread blocks, and a kernel function can launch one or more grids simultaneously. The number of threads per block is pre-configured by the programmer, with a maximum of 1024 threads allocatable per block on the A100 GPU. To address this disparity, eLLM employs kernel fusion to dynamically allocate varying numbers of threads to balance the execution time of K1 and K2. This strategy is inspired by the analysis in § 4.2.1, where the latencies of K1 and K2 are estimated using Equations 2 and 3, respectively. The computational volumes of K1 and K2 can be expressed as:

$$V_{K1} = \sum_{i=1}^{b} \left( 24h^2 l_2 s_i r + 4h l_2 s_i^2 r^2 + 2h V s_i r \right), \tag{8}$$

$$V_{K2} = \sum_{i=1}^{b} \left( 24h^2(l_1 + l_2) + 4h(l_1 + l_2)(s_i + 1) + 2hVs_i \right). \tag{9}$$

The *Threads Allocation* strategy leverages the ratio of  $V_{K1}/V_{K2}$  to dynamically allocate threads to K1 and K2. The total number of threads per block, denoted as  $\mathcal{T}$ , is chosen under the hardware limit (e.g.,  $\mathcal{T} \leq 1024$  on the A100 GPU). Given an allocation of  $\delta$  threads to K1, the remaining  $\mathcal{T} - \delta$  threads are assigned to K2. Additionally, to align with warp granularity—where the NVIDIA warp scheduler dispatches 32 threads per warp at a time [34]—the allocation is adjusted to ensure thread counts are multiples of 32. This ensures efficient utilization of the GPU's parallel execution units while maintaining compatibility with hardware scheduling constraints.

During the request-level optimization, the value of  $M_o$  is initially set to 40hbsr, based on the assumption that the swapping layers are limited to 3 and the recomputing layers are limited to 2. However, once the optimal overlapping configuration and thread allocation are determined,  $M_o$  can be calculated using the formula  $8hbsr(l_1 + l_2)$ , as explained in § 4.3.2. This refined value of  $M_o$  is then utilized to further optimize the batch size and uncached token ratio at the request level, as detailed in § 4.2, thereby enhancing the overall throughput efficiency.

