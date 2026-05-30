# <span id="page-4-1"></span>3.2 DISTRIBUTED LATENCY AND MEMORY ANALYSIS

To gain deeper insight into the worst-case cost model of MoE layers under EP, we analyze both latency and peak memory usage in a holistic manner. We first consider the computation local to a single GPU device. Given a batch routed to G local experts, the MoE layer performs G GEMM operations, and the total latency can be approximated as

$$T_{\text{local}} = \sum_{i=0}^{G-1} (T_{\text{overhead}} + B_i \times T_{B_i,D,H})$$
(3)

where Toverhead denotes the kernel launch latency, and T<sup>B</sup>i,D,H is the per-token compute time, which depends on the token count B<sup>i</sup> and model dimensions D and H (defined in § [2.1\)](#page-2-0). The efficiency of T<sup>B</sup>i,D,H is directly impacted by how GEMM kernels are implemented, optimized, and tuned with

<span id="page-5-1"></span>**Algorithm 2** Least-Loaded Assignment (LLA): Calculate a plan of that assign an expert' portions of tokens to different devices, as well as a corresponding weight transfer plan.

```
Input: global expert loads l \in \mathbb{R}^N, # local experts M, factor \alpha, minimum tokens per GEMM m
//\hat{l} is sorted loads, I_{\hat{l}} is sorted indices
l, I_{\hat{l}} \leftarrow \text{sort}(l, \text{decreasing=true})
// native/pending/assigned load per GPU
g_n \in \mathbb{Z}^{P^-} \leftarrow \text{sum of loads of local experts}
g_p \leftarrow g_n
g_a \in \mathbb{Z}^P \leftarrow 0 \text{ (zeros)}
// max tokens per GPU allowed
m_{\alpha} \leftarrow \alpha \times \frac{1}{P} \times \sum_{i=0}^{N-1} \hat{\boldsymbol{l}}_{i}
\mathcal{A} \leftarrow \{\} // assignments map for each expert
for i, e in zip(I_{\hat{i}}, \hat{l}) do
   ng \leftarrow \operatorname{floor}(i/M)
   g_p[ng] \leftarrow g_p[ng] - e
   // available tokens on native GPU
   na \leftarrow m_{\alpha} - g_a[ng] - g_p[ng]
    A \leftarrow [] // assignments
   if na \ge e then
       // Case 1: Native GPU can handle all tokens
       A \leftarrow A + [(ng, 0, e)]
       g_a[ng] \leftarrow g_a[ng] + e
   else if na > 0 then
       // Case 2: Native GPU takes what it can, spill the rest to other GPUs
       nc \leftarrow \min(na, e)
       to \leftarrow nc // token offset
       A \leftarrow A + [(\mathsf{ng}, 0, \mathsf{nc})]
       g_a[ng] \leftarrow g_a[ng] + nc
       r \leftarrow e - nc // \text{ remaining}
       Call LLAS(ng, r, to, A, g_a, g_p, m_\alpha, m) in Alg. 3
       // Case 3: Native GPU overflowed, spill entire expert work to other GPUs
       Call LLAS(ng, e, 0, A, g_a, g_p, m_\alpha, m)
   end if
   \mathcal{A}[i] \leftarrow A
end for
W \leftarrow \text{construct weight transfer plan from } A
Output: A, W
```

respect to different input and output sizes and configurations. In general, GEMMs become more efficient as  $B_i$ , D and H increase. For example, with D and H fixed,  $T_{B_1,D,H} < T_{B_2,D,H}$  when  $B_1 > B_2$ . Therefore, given a fixed number of FLOPs, executing a small number of large GEMMs is significantly more efficient than executing many small GEMMs. EP exploits this property by aggregating tokens across devices, thereby reducing the number of local experts G and increasing the effective batch size  $B_i$  per expert.

The many  $T_{\rm overhead}$  can be reduced to only one by using a fused grouped-GEMM kernel, but that is not always faster because singular hardware-optimized GEMM kernels (cuBLAS) are more efficient at large D, and H. Fig. 8 shows that, even though we compute the exact same FLOPs, the elapsed time increases with the number of experts, and launching many small cuBLAS GEMMs is still faster than a single fused Triton grouped-GEMM kernel<sup>1</sup>. The peak memory usage of the MoE layer is

<span id="page-5-0"></span><sup>&</sup>lt;sup>1</sup>cuBLAS is proprietary software by NVIDIA that is highly optimized for the hardware, while the Triton grouped-GEMM is an agnostic implementation.

#### <span id="page-6-0"></span>Algorithm 3 Least-Loaded Assignment Spill (LLAS): Spilling the remaining tokens to other GPUs

```
Input: native GPU ng, remaining tokens r, token offset to, assignments A, assigned load g_a,
pending load g_p, m_\alpha, m
while r > 0 do o \in \mathbb{Z}^{P-1} \leftarrow other GPUs g \neq ng sorted by g_a[g] + g_p[g]
   for o in o do
      c \leftarrow \min(r, m_{\alpha} - g_a[o] - g_p[o])
      if c < m and r > c then
         skip // chunk too small
      end if
      A \leftarrow A + [(o, to, to + c)] // assign load
      g_a[o] \leftarrow g_a[o] + c
      to \leftarrow to + c // increment token offset
      break
   end for
   if none of o assigned then
      // force assign the least loaded GPU
      A \leftarrow A + [(o, to, to + r)]
      g_a[o] \leftarrow g_a[o] + r
   end if
end while
```

defined approximately as:

$$M_{\text{local}} = \sum_{i=0}^{G-1} (B_i \times D + D \times H + B_i \times H)$$
(4)

Under standard expert parallelism,  $B_i$  is the total number of tokens routed to expert i from across all EP devices. In the worst case,  $B_i$  may approach the global batch size, causing all tokens to be concentrated on a single device while others are idle. This causes spiking latency and memory usage, or even out-of-memory crashes for the overloaded device.

Figs. 1a and 1b show the slowdown and peak memory usage of a standard EP setup under different imbalance scenarios. As shown in Fig. 1a, EP could be 4.6x slower when 95% of tokens are routed to a single expert compared to the balanced baseline. As for peak memory usage, EP's peak memory usage per GPU may grow up to 4x, potentially causing OOM errors.

## 4 LEAST-LOADED EXPERT PARALLELISM (LLEP)

We explain in detail how our proposed LLEP works. Conceptually, our method will detect ahead of time the degree of imbalance of the global routing according to per-expert loads. If the imbalance is lower than a threshold  $\lambda$ , then we consider the routing as balanced and proceed to the standard EP procedure. Otherwise, we will execute the least-loaded assignment algorithm (Alg. 2) to determine for each GPU device that it needs to compute GEMMs for which experts and with how much portions of the global tokens routed to them. If the GPU does not contain an assigned expert as resident, it will import the expert from its host GPU. The assignment takes into account the overhead cost of weight and data transfers, in comparison to the latency and memory cost of processing the tokens only for local experts. Algs. 2 to 4 formally describe our method in detail.

**Constraints.** Our method works by making routing decisions that are subject to the some constraints. First, factor  $\alpha$  in Alg. 2 determines how much maximum token capacity a GPU can handle, which we defined as  $m_{\alpha} = \alpha \sum_{i=0}^{N-1} \hat{\boldsymbol{l}}_i/P$  tokens.  $m_{\alpha}$  is not necessarily a physical memory limit, but rather a threshold that the GPU is considered overloaded. If a local expert load exceeds  $m_{\alpha}$ , it will spill the

excess load to other GPUs. Second, m is the minimum tokens per GEMM for it to be efficient. If a local expert load exceed the local GPU's occupied capacity, but the excess is less than m, we consider it's not worth it to spill and instead force the local GPU to compute it despite over-capacity (see § [3.2\)](#page-4-1). Third, imbalance ratio threshold λ is used to determine whether the global loads are relatively balanced, in which case we switch back to standard EP. The reason is that our method employs a greedy least-loaded assignment (LLA) algorithm (Alg. [2\)](#page-5-1) that would produce the same routing plan as standard EP anyway, while causing a tiny time overhead. Without skipping this imbalance ratio check, our method is shown to be slightly slower than standard EP under perfectly balanced scenarios. The optimal values for α, m, and λ depend on N, P, Bp, K, D, H, the overall model size, and the physical system configuration. Thus, we recommend to tune these values for each use case.

Elaboration. The least-loaded assignment (LLA) algorithm (Alg. [2\)](#page-5-1) determines, for each expert, which GPUs handle which portions of the global expert's load. First, it sorts the expert loads in decreasing order. Then, it determines the GPU allocations for each expert from largest-load to smallest-load ones. For each expert, it first determines if the native GPU (the one that hosts the expert's weights) can handle all the tokens of the expert. If it can, it assigns all the tokens to the native GPU. If it cannot, it spills the excess tokens to the least-loaded available GPU up to the capacity threshold. If there are still remaining excess tokens, it will continue this spilling loop (LLAS, Alg. [3\)](#page-6-0) until all the tokens are assigned. Once the tokens routing plan is finalized, it will also construct the weight transfer plan accordingly. For example, if excess load of expert i native to GPU p is spilled to GPU q, then the weight transfer plan will include a weight transfer operation from p → q for W<sup>i</sup> . The LLA algorithm ensures that each GPU prioritize computing most, if not all, of the its local experts' load first before accepting foreign experts' load. This is to minimize the number of weight transfers required. The final LLEP algorithm (Alg. [4\)](#page-8-0) will then execute the dispatch-compute-combine operations according to the routing plans obtained from LLA. Specifically, for each device, in addition to GEMM computation for native experts, LLEP will also compute the GEMMs for foreign experts that are assigned to the device. Unlike others, LLEP supports proper gradient propagation. During the backward pass, the gradients for the spilled expert weights are returned to their native devices and accumulated with their native gradients respectively.

Implementation & Optimization. In the experiments, we implement our method with the standard Torch's NCCL for All-to-All and peer-to-peer (P2P) operatives. The LLA algorithm is implemented in pure Python. While our simple LLEP implementation is already showing significant speedup and memory saving, there are further opportunities to optimize and reduce overhead. For instance, the communication operatives can be written as low-level C++/Triton kernels, or using a modified version of DeepEP [\(Liu et al., 2024\)](#page-11-0). Such a fused operative may also perform direct All-to-All on unsorted tensors B<sup>p</sup> and Gp, avoiding the memory-intensive index select operation (Alg. [4\)](#page-8-0). The communication can be overlapped with computation or hidden behind the Grouped-GEMM operation. For multi-node setups, we can further modify LLEP to prefer spilling work to intra-node devices to limit the higher inter-node communication overhead.

