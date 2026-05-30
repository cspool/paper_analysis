# Efficient MoE Serving in the Memory-Bound Regime: Balance Activated Experts, Not Tokens

Yanpeng Yu<sup>1,\*</sup> Haiyue Ma<sup>2,\*</sup> Krish Agarwal<sup>3</sup> Nicolai Oswald<sup>4</sup> Qijing Huang<sup>4</sup> Hugo Linsenmaier<sup>4</sup> Chunhui Mei<sup>4</sup> Ritchie Zhao<sup>4</sup> Ritika Borkar<sup>4</sup> Bita Darvish Rouhani<sup>4</sup> David Nellans<sup>4</sup> Ronny Krashinsky<sup>4</sup> Anurag Khandelwal<sup>1</sup>

<sup>1</sup>Yale University <sup>2</sup>Princeton University

<sup>3</sup>Carnegie Mellon University <sup>4</sup>NVIDIA

Abstract—Expert Parallelism (EP) permits Mixture of Experts (MoE) models to scale beyond a single GPU. To address load imbalance across GPUs in EP, existing approaches aim to balance the number of tokens each GPU processes. Surprisingly, we find that this objective degrades performance rather than improving it when processing is memory-bound — a common occurrence in MoE serving, especially in the decode phase. Our analysis reveals that balancing the number of tokens processed per GPU increases the number of activated experts, exacerbating memory pressure in the memory-bound regime.

We propose METRO<sup>1</sup>, a novel token-routing algorithm for high-performance expert-parallel MoE serving in the memorybound regime that balances the number of activated experts per GPU rather than token counts. METRO achieves nearoptimal routing quality with minimal computational overhead by jointly optimizing algorithmic efficiency and leveraging the GPU's parallel processing power. To guarantee routing quality, METRO also employs a novel allGather scheme to gather global top-k knowledge, which has minimal overhead compared to conventional allToAll. Our evaluation of METRO against EPLB on both real systems (vLLM over 8 A100 GPUs) and a proprietary simulator (8-16 B200 GPUs) shows that METRO reduces decode latency by 11 - 22%, and total token throughput by 3 - 21%for Qwen3 and DeepSeek-V3 serving, where prefill and decode phases are co-deployed. In addition, by trading latency headroom for throughput, METRO improves decode throughput by up to  $4.11\times$  over EPLB at a fixed decode SLO.

## I. INTRODUCTION

Mixture-of-Expert (MoE) models [14], [42] are growing increasingly popular due to their lower computational cost per token generation compared to denser models. When MoE models' footprint scales beyond a single GPU's memory capacity, Expert Parallelism (EP) [17], [28], [41] is emerging as the de facto choice for scaling MoE models to multiple GPUs. In it, experts are placed on different GPUs and process tokens in parallel. EP is often augmented with load-balancing mechanisms [8], [9], [21], [24], [31], [34], [40], [44], [46], [47], [49] to address load imbalance across GPUs. This includes creating replicas of popular experts (expert replication), strategically placing these replicas across GPUs (expert placement), and routing tokens across the replicas of an expert (token routing<sup>2</sup>).

The underlying principle of the load-balancing mechanisms outlined above is minimizing the runtime of the slowest

![](_page_0_Figure_13.jpeg)

Fig. 1. METRO achieves universal (up to 22%) performance improvement on both decode latency and total token throughput (prefill-decode codeployed) over EPLB's token routing across models, datasets, and hardware setups. Results are from both real system evaluation and simulation (§VI-B). Replication ratio: 50%. Placement algorithm: EPLB.

GPU in the pool, since it determines the end-to-end runtime. Existing EP expert placement/replication and token-routing algorithms [8], [9], [21], [24], [31], [34], [40], [44], [46], [47], [49] attempt to achieve this by balancing the number of tokens processed by each GPU (i.e., EP rank). Such "token-balancing" implicitly assumes that the GPU runtime for inference requests scales linearly with the number of tokens processed. However, this is only true when the inference workload is *compute-bound*, i.e., the GPU runtime is bottlenecked by tensor core throughput rather than memory bandwidth (§III-A).

The reality, however, is more nuanced — real-world serving workloads comprise a mix of *both* compute-bound (i.e., prefill) and memory-bound phases (i.e., decode). Indeed, we find that in the memory-bound regime, a GPU's MoE layer runtime is not governed by the number of processed tokens, but instead determined by the number of 'activated' expert replicas — the replicas that actually process tokens in a given batch. This is because in the memory-bound regime, runtime increases with the amount of memory traffic, which is dominated by loading expert weights from GPU memory to tensor cores. Unfortunately, existing EP load-balancing schemes that balance tokens across GPUs inadvertently increase the number of activated expert replicas, *degrading* performance in the memory-bound regime (§III-B).

These observations motivate our central research question: in a MoE serving system that comprises both compute-bound prefill and memory-bound decode phases, *can we balance EP* 

<sup>\*</sup>Equal contribution; listed in random order.

<sup>&</sup>lt;sup>1</sup>Minimum Expert Token ROuting

 $<sup>^2</sup>$ We refer to token routing as the process of selecting specific expert replica(s) to 'activate' to serve a particular token, *not* the process of choosing the top-k experts.

load without hurting — and ideally improving — the performance of the memory-bound phase (i.e., decode) performance, so that both phases can benefit from replication strategies? Since tailoring expert replication and placement to the decode phase can interfere with the prefill phase's performance, we restrict our focus to the token-routing algorithm in this work.

We present METRO, a novel token-routing algorithm for EP load balancing in the memory-bound regime that minimizes GPU runtime by *minimizing the number of activated expert replicas across GPUs*.

Realizing METRO requires addressing several distinct challenges ( $\S$ IV). First, we find that an optimal solution to the routing problem — i.e., minimizing the number of activated expert replicas across GPUs — incurs prohibitive computational overheads in practice. METRO therefore adopts a GPU-native greedy approximate algorithm that preserves nearoptimal routing quality while limiting computational overheads. Second, unlike balancing tokens, minimizing activated experts in METRO requires each GPU to collect global top-k expert selections from all GPUs to make informed routing decisions. To enable this, METRO replaces the conventional all-to-all EP dispatch with a novel, low-overhead all-gather token dispatching scheme that disseminates the global top-k information across GPUs.

We have evaluated METRO on both real systems (vLLM [27] over 8 NVIDIA A100 GPUs [35]) and a proprietary industrial multi-GPU performance simulator (8–16 B200 GPUs [37]). Our results show that METRO consistently outperforms the baseline EPLB routing algorithm across various settings and replication ratios. Specifically, as shown in Fig. 1, METRO achieves up to 21.8% reduction in decode latency at  $1.5\times$ replication, translating into up to 21.0% improvement in total token throughput by effectively minimizing the number of activated experts (§VI-B). Moreover, METRO achieves up to  $4.11 \times$ higher decode throughput compared to EPLB with a fixed decode SLO, by trading its latency headroom for throughput (§VI-C). These gains are observed across a wide range of real-world MoE models (Qwen3-30B, Qwen3-235B [42], and DeepSeek-V3 [14]) and workloads (InstructCoder [32], NuminaMath-1.5 [30], GSM8K [10], and Humaneval [7]).

In summary, this paper makes the following contributions:

- We identify that in the memory-bound regime, a GPU's MoE layer runtime is determined by the number of 'activated' expert replicas.
- We show that, in the memory-bound regime, existing EP load-balancing schemes that balance tokens degrade performance by inflating the number of activated experts.
- We present METRO, a novel token-routing algorithm for EP load balancing in the memory-bound regime, whose key idea is to minimize GPU runtime by minimizing the number of activated expert replicas.
- We implement and evaluate METRO in both a realworld vLLM-based system and a proprietary industrial simulator. Our results show that METRO reduces MoE decode latency while improving total token throughput compared to existing EP load-balancing schemes.

![](_page_1_Picture_9.jpeg)

Fig. 2. Expert-parallel MoE inference workflow with expert placement and replication, as well as token routing – the algorithm to dynamically route tokens to expert replicas.

#### II. BACKGROUND

## A. Autoregressive LLM Inference

LLM inference consists of two phases: a prefill phase followed by a decode phase. In prefill, the model processes the entire input prompt in a single forward pass to produce the first output token. Since this phase processes many tokens at once, it is typically compute-bound and achieves high GPU utilization. In the subsequent decode phase, the model generates tokens autoregressively, feeding each newly generated token back into the model until an end-of-sequence token is produced. Since the decode phase processes only one token per request, even when batching across multiple requests, it cannot fully saturate the GPU cores and is typically memorybound. Due to such heterogeneity, recent works have proposed disaggregating prefill and decode for large-scale deployments with multiple server nodes and up to hundreds of GPUs [3], [38], [39], [53]. However, on smaller systems — e.g., a single server with a few GPUs — prefill and decode are still commonly co-deployed [4], [18], [19], [25], [33], [54], due to various reasons as we will discuss in §VII. In this paper, we first target the co-deployed setting for small multi-GPU systems and later discuss how our techniques also benefit disaggregated prefill-decode deployments in §VII.

## B. MoE Inference and Expert Parallelism

Mixture-of-Expert (MoE) models [14], [42] are growing increasingly popular due to their lower computational cost per token generation compared to dense models. When MoE models' footprint scales beyond a single GPU's memory capacity, Expert Parallelism (EP) [17], [28], [41] is arguably the most popular algorithm to scale MoE to multiple GPUs. In it, experts are placed on different GPUs and process tokens in parallel. EP is often augmented with systematic loadbalancing mechanisms [8], [9], [21], [24], [31], [34], [40], [44], [46], [47], [49]. This includes creating replicas of popular experts (expert replication), strategically placing these replicas across GPUs (expert placement), and routing tokens across the replicas of an expert after top-k selection (token routing).

Fig. II-A shows the expert-parallel inference workflow for MoE's expert layer, augmented with the mechanisms described above. Each GPU first computes a top-k value for each of its

input tokens, which determines the set of experts to activate for that token. Then, token routing determines which physical expert replica each token should be routed to. Next, an all-to-all communication is used to dispatch tokens to the GPUs that host the identified target expert replica. Then, tokens' output embeddings are computed using the corresponding expert replicas. Finally, each token's output embedding is sent back to its original GPU using another all-to-all communication, and combined there to produce the final output.

## C. Existing replication and routing algorithms balance tokens

To our knowledge, all existing EP expert placement/replication and token-routing algorithms aim to balance the number of tokens processed across GPUs (i.e., EP ranks). For example, DeepSeek's Expert Parallelism Load Balancer (EPLB) [8], the state-of-the-art expert placement and replication algorithm, balances tokens in two steps. First, it creates replicas for each expert, with replica count proportional to the number of tokens processed by the expert in the last time window. Second, it places these replicas on GPUs to balance the number of tokens they are expected to process in the next time window. EPLB's replica placement algorithm (i.e., the second step) assumes that the token-routing algorithm distributes each expert's token evenly across its replicas. Most EPLB implementations (e.g., by vLLM [27] and SGLang [52]) follow such a token-routing algorithm. While other studies [9], [21], [24], [31], [34], [40], [44], [46], [47], [49] have proposed various other expert placement, replication, and token-routing algorithms, they share the same optimization objective as EPLB — balancing tokens across GPUs.

## III. MOTIVATION

Contrary to conventional wisdom, we find that balancing tokens can degrade, rather than improve, MoE inference performance across a large class of real-world settings. Specifically, we first show that balancing tokens implicitly assumes the compute-bound regime, whereas MoE inference comprises of compute-bound (i.e., prefill) and memory-bound (i.e., decode) phases (§III-A). We then show that in the memory-bound regime, balancing tokens *degrades* performance rather than improving it, by increasing the number of activated experts and exacerbating the memory pressure (§III-B).

## A. Balancing tokens assumes the compute-bound regime while MoE's decode phase is increasingly memory-bound

Balancing tokens to minimize runtime assumes that the inference workload is entirely compute-bound, i.e., the GPU runtime is bottlenecked by tensor core throughput rather than memory bandwidth. This is because only in the compute-bound regime is the number of tokens processed by a GPU proportional to its runtime, making balancing tokens equivalent to balancing (and thus minimizing) GPU runtime.

While such compute-bound assumption holds in many settings — especially those where existing token-based load balancing algorithms were originally proposed (i.e., training and prefill) — recent studies [4], [13], [48] have observed that

![](_page_2_Figure_8.jpeg)

Fig. 3. DeepSeek-V3 and Qwen3-30B attainable operational intensities VS. FLOPs/byte ratio of H100 and B200 (§III-A). The former is two orders of magnitude lower than the latter with batch size smaller than 64 tokens, and 47% -  $3.0\times$  lower with a batch size of 1024 tokens.

MoE inference workloads are increasingly memory-bound especially in the decode phase — due to several emerging trends. These include limited inter-token weight reuse compared to dense models due to increasing MoE sparsity [6], the use of smaller batch sizes due to the memory capacity pressure of increasing context lengths [5], [20], [27], and the increasing FLOPs/byte ratio of recent GPU hardware [48], [51]. Indeed, Fig. 3 shows that the attainable operational intensities (i.e., the ratio of the amount of arithmetic operations performed to the amount of data moved to/from memory [45]) during decode (following the calculation in [13]) for two state-of-the-art MoE models, DeepSeek-V3 [14] and Qwen3-30B [42], are two orders of magnitude lower than the FLOPs/byte ratio of recent NVIDIA datacenter GPUs (i.e., H100 [36] and B200 [37]) with a batch size smaller than 64 tokens. Even batching as high as 1024 decode tokens, the GPUs' FLOPs/byte ratio is still 47% -  $3.0\times$  higher than the operational intensities of the MoE models. Moreover, the models' attainable operational intensities in practice are often lower than the ideal value due to buffer capacity constraints across the memory hierarchy [23]. Therefore, it is critical to re-examine EP load balancing in the memory-bound regime, where balancing tokens does not directly translate into balancing GPU runtimes.

## B. Balancing tokens inflates activated experts and thus degrades performance in the memory-bound regime

We find that in the memory-bound regime, a GPU's MoE layer runtime is not governed by the number of processed tokens, but instead determined by the number of 'activated' expert replicas — the replicas that actually process tokens in a given batch. This is because, in the memory-bound regime, runtime scales with total memory traffic, which is dominated by loading expert weights from GPU memory to tensor cores. Indeed, the token activations' footprint is significantly smaller: the activation's memory traffic is < 0.6% of the expert weights' memory traffic for a decode batch as large as 1K tokens based on the analytical model proposed in recent work [13]. We empirically confirm this observation in Fig. 5d and Fig. 5b, which shows a strong correlation between the number of activated experts and vLLM decode latency (evaluation setup detailed in §VI-A).

![](_page_3_Figure_0.jpeg)

Fig. 4. Example: balancing tokens for token routing doubles the activated experts per GPU and the theoretical runtime in the memory-bound regime, compared to the hypothetical ideal token routing (§III-B).

Unfortunately, balancing tokens across GPUs increases the number of activated expert replicas, thereby degrading performance in the memory-bound regime. Fig. 4 uses a toy example to compare two token routing schemes under the same expert replication and placement scheme for a batch of size 16. For the token-balanced routing scheme to achieve perfect token balance (i.e., two tokens per GPU), it must evenly distribute tokens across all available replicas. This causes two expert replicas to be activated on each GPU. However, as demonstrated by the hypothetical ideal routing scheme, all tokens routed to a particular expert should ideally be routed to a single replica to minimize the activated experts per GPU (i.e., one per GPU). Since runtime increases with the number of activated experts rather than the number of tokens per GPU in the memory-bound regime, balancing tokens would double the GPU runtime relative to the ideal in this example!

To understand the end-to-end performance degradation of balancing tokens in the memory-bound regime, we measure the performance impact of EPLB on prefill latency (i.e., Timeto-first-token, or TTFT), decode latency (i.e., Time-per-outputtoken, or TPOT), and the total token throughput for a stateof-the-art serving system, vLLM [27]. We use a state-of-theart MoE model, Qwen3-30B [42] with a representative coding dataset, InstructCoder [32]. We observe similar trends for other datasets (with results deferred to §VI-A). Fig. 5 shows that while EPLB can reduce prefill latency (Fig. 5a) by 17% with up to  $1.5 \times$  replication when the batch size is large enough for prefill to enter the compute-bound regime (i.e., 32 requests per GPU), it increases the number of activated experts per decode batch (Fig. 5d) as the replication factor increases nearly 30% more activated experts per decode batch with  $1.5 \times$  replication. Consequently, not only does the decode latency increase with more replication (i.e., by 14% with  $1.5\times$ replication, Fig. 5b), but the overall token throughput also decreases (i.e., by 10% with  $1.5\times$  replication, Fig. 5c).

These observations motivate our central research question: In a MoE serving system where the compute-bound prefill and memory-bound decode phases are co-deployed, can we balance EP load without hurting — and ideally improving

![](_page_3_Figure_5.jpeg)

Fig. 5. The performance impact of EPLB on prefill latency (a), decode latency (b), overall token throughput (c), and maximum number of activated experts across GPUs per decode batch (d) for Qwen3-30B [42] on vLLM [27] (§III-B). Context length: 8K. Dataset: InstructCoder [32]. EPLB reduces prefill latency by 17% with batch size 32, but inflates the number of activated experts by 30% with 1.5x replication. As a result, the decode latency increases by 14% and the overall token throughput decreases by 10% with 1.5x replication. We also find that the prefill phase can be in the memory-bound regime when the batch size is small (i.e., 8 and 16 requests per GPU), where EPLB cannot improve its performance.

— the memory-bound decode phase performance, allowing both phases to benefit from additional memory capacity? Since tailoring expert replication and placement to the decode phase may interfere with prefill phase performance, we restrict our focus to the token-routing algorithm. To this end, we present METRO, a novel token-routing algorithm for EP load balancing in the memory-bound regime. METRO minimizes GPU runtime by minimizing the number of activated experts across GPUs, as we detail in the next section.

## IV. METRO DESIGN

We first formulate token routing for minimizing the number of activated experts as an optimization problem (§IV-A). Unfortunately, we find that while an optimal algorithm exists for the above problem, it incurs prohibitively high computational overhead in practice (§IV-B). To overcome this limitation, we propose METRO, a routing algorithm that achieves *near-optimal* routing quality with *minimal overhead* (§IV-C).

## A. Problem Formulation

We formulate token routing (dubbed MIN-EXP-ROUTING) as an Integer Linear Program (ILP), as detailed next.

**MIN-EXP-ROUTING.** We consider N experts across G GPUs, with the binary expert–GPU mapping  $A \in \{0,1\}^{N \times G}$  identifying which GPU hosts each expert. We define the number of

tokens per expert as T[1..N] for a particular batch, where T[i] corresponds to the number of tokens for expert i in that batch. The problem of minimizing the number of activated experts reduces to finding assignments for:

- integer variables x<sub>i,g</sub> ≥ 0, which denote tokens of expert i routed to GPU g,
- binary variables  $y_{i,g} \in \{0,1\}$ , which identifies whether expert i is activated on GPU g, and,
- the integer variable  $\lambda \ge 0$ , which denotes the maximum number of activated experts across all GPUs,

to minimize  $\lambda$ :

 $\min \lambda$ 

subject to, for all  $g \in [G]$  and  $i \in [N]$ ,

$$\sum_{i=1}^{N} y_{i,g} \le \lambda,\tag{1}$$

$$\sum_{g=1}^{G} x_{i,g} = T[i], \tag{2}$$

$$x_{i,g} = y_{i,g} = 0$$
 if  $A_{i,g} = 0$ , (3)

$$x_{i,g} \le T[i] \cdot y_{i,g}. \tag{4}$$

Intuitively, constraint (1) ensures that the number of activated experts for each GPU does not exceed the upper bound  $\lambda$ . Constraint (2) ensures no tokens are dropped, i.e., all T[i] tokens of expert i must be routed. Constraint (3) ensures that routing abides by the placement matrix A, i.e., tokens should not be routed to GPUs that do not host the corresponding expert. Finally, constraint (4) ensures that tokens are routed to a GPU  $(x_{i,g} > 0)$  only if the corresponding expert is activated there  $(y_{i,g} > 0)$ .

**Simplifying MIN-EXP-ROUTING.** Interestingly, we find that all feasible solutions to MIN-EXP-ROUTING share a common property that allows us to focus on a simpler variant of MIN-EXP-ROUTING:

**Lemma 1.** Any feasible solution to MIN-EXP-ROUTING either routes tokens only to one replica of any expert, or can be mapped to a solution that does without increasing the objective value.

Proof sketch. Consider any feasible solution that achieves an objective value of  $\lambda = \lambda_0$ . In this solution, if the tokens to be processed by any expert i are distributed across multiple GPUs (i.e., its replicas), we construct another solution in which all such experts i's tokens are routed to a single GPU (i.e., a single replica). This solution preserves constraints (2), (3), and (4). At the same time, it does not increase the number of activated experts on any GPU, i.e., the solution still preserves constraint (1) and thus still achieves  $\lambda \leq \lambda_0$ , proving the second half of the lemma's claim. If all tokens in the original feasible solution are already routed to a single replica of each expert, then the first part of the lemma's claim holds, completing our proof.

![](_page_4_Figure_15.jpeg)

Fig. 6. The runtime of CPU-based and GPU-based optimal algorithms VS. the runtime of a single FFN layer of Qwen3-30B on vLLM over 8 A100 GPUs. The CPU-based and GPU-based algorithms incur prohibitively high overheads of 31.4%-41.3% and 86.4%-103.8% relative to the FFN, respectively.

Leveraging Lemma 1, we restrict our focus to solving the simpler problem of identifying for each expert (that has tokens to process) one of its replicas to activate, such that the maximum number of activated expert replicas per GPU is minimized. This is because any feasible solution to MIN-EXP-ROUTING can be reduced to a feasible solution to this simpler problem. We subsequently use MIN-EXP-ROUTING to refer to this simpler problem variant.

## B. An Optimal Algorithm

We find that MIN-EXP-ROUTING reduces to the classical optimization problem [15], [26], [29] of matching jobs of identical runtime (i.e., experts) to machines (i.e., GPUs) with restriction (i.e., expert-GPU placement matrix A) to minimize the maximum runtime across machines (i.e., maximum activated experts across GPUs). Similar to prior approaches for solving such problems [15], [26], [29], our optimal algorithm searches for the minimal feasible objective (i.e.,  $\lambda$ ) via binary search, where each candidate  $\lambda$  is tested for feasibility via bipartite matching [22]. Specifically, for a particular candidate  $\lambda_0$ , we can build a bipartite graph with experts (that process at least one token in this batch) on the left and GPUs on the right. Edge (i, g) is present iff expert i is placed on GPU g(i.e.,  $A_{i,g} = 1$ ). Unlike the standard bipartite matching, where each node is matched at most once, our setting requires a variant in which each GPU g node can be matched at most  $\lambda_0$  times. This variation accounts for the fact that each GPU can activate at most  $\lambda_0$  expert replicas. With this reduction, a feasible bipartite matching in this graph exists iff  $\lambda_0$  is feasible for MIN-EXP-ROUTING.

Computational overheads. From a computational complexity perspective, the binary search incurs a complexity factor of  $O(\log \lceil \frac{|A|}{G} \rceil)$  since  $\lambda$  is at most  $\lceil \frac{|A|}{G} \rceil$ . For each candidate  $\lambda$ , a bipartite-matching-based feasibility test using max-flow algorithms incurs a worst-case complexity of  $O((N+G)^2 \cdot (\lceil \frac{|A|}{G} \rceil + N + G))$ .

To study this overhead in practice, we measure the runtime of both the state-of-the-art CPU-based and GPU-based imple-

## Algorithm 1 METRO's greedy approximate algorithm

```
1: Input: N, G, A \in \{0, 1\}^{N \times G}, T[1..N]
 2: Output: y_{i,q}
 3: Initialization: L[g] \leftarrow 0 and init lock l_g for each g =
     1, \ldots, G; y_{i,g} \leftarrow 0 for each g = 1, \ldots, G, i = 1, \ldots, N.
 4: for i = 1 to N do in parallel
        if T[i] > 0
 5:
            \mathcal{G}_i \leftarrow \{g \mid A_{i,g} = 1\}
 6:
            acquire all locks \{\ell_g \mid g \in \mathcal{G}_i\} in a total order
 7:
            choose g^* \in \mathcal{G}_i with the smallest L[g]
 8:
            y_{i,g^*} \leftarrow 1; \quad L[g^*] \leftarrow L[g^*] + 1
 9:
            release all locks \{\ell_g \mid g \in \mathcal{G}_i\}
10:
11:
12: end for
```

![](_page_5_Figure_2.jpeg)

Fig. 7. METRO replaces the conventional all-to-all with all-gather dispatch before top-k for every GPU to obtain the global top-k knowledge as input to Algorithm 1, with minimal overhead (IV-C).

mentations of the optimal algorithm. In our implementations, we use Dinic's max-flow algorithm [16] for feasibility testing on the CPU, and a recent, state-of-the-art, GPU-optimized push-relabel max-flow algorithm [43] on the GPU. Unfortunately, we find that the optimal algorithm, whether executed on CPU or GPU, has prohibitively high computational overhead relative to the model's computation time. Fig. 6 shows that the CPU-based and GPU-based optimal algorithms incur runtime overheads of  $116.3\mu s - 128.8\mu s$  and  $290.0\mu s - 292.1\mu s$ , respectively, across different replication ratios. Relative to the runtime of a single FFN layer of Qwen3-30B on vLLM over 8 A100 GPUs, these correspond to prohibitively high overheads of 31.4% - 41.3% and 86.4% - 103.8%, respectively. The GPU-based algorithm is slower than the CPUbased one because the bipartite graph is insufficiently large to fully exploit the GPU's parallelism. Furthermore, we exclude any CPU-based algorithms because transferring the algorithm inputs (i.e., the top-k selection tensors, which reside on the GPU) alone costs  $26.5\mu s - 29.2\mu s$ , introducing up to 10.4%overhead relative to the FFN.

## C. METRO: A Greedy Approximate Algorithm

To simultaneously achieve low computational overhead while preserving near-optimal routing quality, we propose METRO, a GPU-native approximate algorithm that prioritizes computational efficiency. In addition, METRO introduces a

novel all-gather communication scheme that collects the global top-k required by the algorithm while incurring insignificant overhead in practice.

METRO's greedy approximate algorithm. Algorithm 1 details METRO's greedy approximate algorithm. It inputs a problem instance of MIN-EXP-ROUTING, and outputs the assignments for decision variables  $y_{i,g}$  that denote whether expert i is activated on GPU g in the routing decision. We omit other decision variables (i.e.,  $x_{i,g}$  and  $\lambda$ ) since Lemma 1 allows us to directly compute them using  $y_{i,g}$ :

$$x_{i,g} = \begin{cases} T[i] & \text{if } y_{i,g} = 1\\ 0 & \text{otherwise} \end{cases}$$
$$\lambda = \max_{g=1...G} \sum_{i=1}^{N} y_{i,g}$$

Algorithm 1 maintains a per-GPU activated expert counter L[1..N] that is guarded by corresponding locks  $l_{1..N}$  (line 3). The high-level idea is to greedily assign each expert to the GPU with the fewest activated experts. In more detail, for each expert that has at least one token to process for this batch, METRO first identifies its candidate GPUs by looking up the expert-GPU placement matrix A (line 6). Then, before matching experts to GPUs, to ensure the atomicity of each expert's assigning process, each thread acquires locks for its expert's all candidate GPUs (line 7), and only releases them after the expert assignment is done (line 10). To avoid deadlocks, lock acquisition is done in a total order (e.g., GPU ID order in our implementation). After all locks are acquired, each expert chooses the GPU g that has the least activated experts L[q] (line 8) and assigns itself to q (line 9). We defer architecture-specific implementation details to §V.

**All-gathering Global Top-**k **Knowledge.** METRO requires global top-k information (T[1..N]) across all GPUs, which conflicts with the conventional all-to-all dispatch where top-k results remain local to each GPU. To address this, as shown in Fig. 7, we replace all-to-all with an all-gather dispatch before top-k so that every GPU can obtain the global top-k knowledge with minimal overhead. Specifically, as shown in Fig. 7, tokens are first all-gathered to all GPUs; each GPU then runs top-k on the full token set to build the global top-k knowledge (i.e., T[1..N]). Then, each GPU executes Algorithm 1 to route tokens, compute FFNs, and finally performs the all-to-all combine.

Minimal performance overhead. METRO introduces three sources of performance overheads, all of which are outweighed by the performance improvement it enables by effectively minimizing experts. While we break down these overheads empirically in §VI-B, we provide an intuitive analysis of METRO's overheads here. First, Algorithm 1 runs in O(|A|) time, since the placement of each export on a GPU is considered exactly once. This is substantially lower than the optimal algorithm's  $O((N+G)^2 \cdot (\lceil |A|/G \rceil + N+G) \cdot \log \lceil |A|/G \rceil)$ . Second, while the all-gather dispatch requires redundant top-k

![](_page_6_Figure_0.jpeg)

Fig. 8. The maximum number of activated experts per GPU per decode batch (32 tokens) for EPLB routing, the optimal algorithms, and METRO across different models (i.e., DeepSeek-V3 and Qwen3-30B) and datasets (i.e., Humaneval and GSM8K) (§IV-B). METRO is within 10.9% higher than the optimal algorithm (i.e., optimal routing) and is lower than EPLB by up to 42.3%.

over the full token set on each GPU, top-k computation time is negligible compared to other components (i.e., attention, FFN, and communication). Consequently, our empirical results show that the overheads introduced by redundant top-k computations in METRO remain negligible. Lastly, while METRO's all-gather dispatch is theoretically more expensive than the conventional all-to-all, we observe no statistically significant increase in communication time. This is because in the memory-bound regime (i.e., small batches), NVLink latency dominates while the bandwidth cost is minor: with 32 decode tokens per GPU (8 GPUs) and the fp16 data type, all-to-all sends 256KB/GPU and all-gather sends 2MB/GPU, which on 600 GB/s NVLink translates to only  $\sim 400ns$  and  $\sim 3\mu s$ , respectively — far below the tens to  $\sim 100\mu s$  fixed cost of launching NCCL collectives.

Near-optimal routing quality. METRO's greedy algorithm achieves near-optimal routing quality. Fig. 8 shows that METRO's maximum number of activated experts per GPU per decode batch (32 tokens) is within 10.9% of the optimal algorithm across different models (i.e., DeepSeek-V3 and Qwen3-30B), different datasets (i.e., Humaneval and GSM8K), and replication ratios. This is up to 42.3% lower than EPLB's routing algorithm that evenly distributes tokens across replicas.

## V. IMPLEMENTATION

CUDA Graph integration. To integrate METRO into vLLM with no additional GPU kernel launch overheads, we take advantage of vLLM's compilation framework [2] to integrate METRO into its decode phase CUDA Graphs. We precompile graphs for power-of-two batch sizes — up to 32 tokens per GPU — which cover all batch sizes considered in our evaluations. For non-power-of-two batches, vLLM pads to the next power-of-two and reuses the corresponding graph.

**Efficient GPU kernel implementation.** We implement METRO (i.e., Algorithm 1) as a CUDA kernel that runs on a single streaming multiprocessor (SM). Since the algorithm's parallelism is fundamentally bounded by the number of experts (e.g., 128 for Qwen and 256 for DeepSeek-V3) and locking

TABLE I SIMULATION SETUP

| Parameter        | Specification                                   |  |
|------------------|-------------------------------------------------|--|
|                  | Real System                                     |  |
| Models           | Qwen3-30B-A3B                                   |  |
| Datasets         | NuminaMath (Decode-heavy),                      |  |
|                  | InstructCoder (Decode-heavy)                    |  |
| GPU Arch         | 8 NVIDIA A100 40GB GPUs                         |  |
| NVLink           | 600 GB/s, all GPUs within the same domain       |  |
|                  | Simulation                                      |  |
| Models           | Qwen3-235B-A22B, Deepseek-V3-671B               |  |
| Datasets         | GSM8K (Prefill-heavy), Humaneval (Decode-heavy) |  |
| Modeled GPU Arch | 8/16 NVIDIA B200 192GB GPUs                     |  |
| NVLink           | 900 GB/s, all GPUs within the same domain       |  |

further reduces concurrency to below 64, a single A100 SM provides sufficient parallel processing power. Confining the kernel to a single SM also lets us keep the load counters L and locks l in SM-local shared memory for faster access. We use a simple test-and-set lock [1] for synchronization.

#### VI. EVALUATION

We evaluate METRO's performance improvements relative to the state-of-the-art EP load balancer atop both a real system implementation and a proprietary industrial performance simulator. Our simulation studies allow us to effectively explore the vast design space of different models, datasets, and hardware configurations, without being limited by the cost- and timeinefficiencies of a real-system implementation.

## A. Setup

Our experimental setup is summarized in Table I.

**Models.** For our real-system evaluation, we use Qwen3-30B-A3B [42] as a representative fine-grained MoE (i.e., 128 experts) that fits the memory capacity of our GPU node.

For simulation studies, we used two models: Qwen3-235B-A22B [42] (128 experts), and Deepseek-V3-671B [14] (256 experts). We expect our simulation studies to validate the trends and insights from the real system implementation with larger models.

Workloads and traces. We evaluate our real-system implementation using two workloads: InstructCoder [32] and NuminaMath [30]. InstructCoder is a code-editing instruction dataset of over 114K instruction + input-code + output-code triplets used to assess code generation models. NuminaMath is a large-scale dataset (i.e., 900K) of competition-level math problems with chain-of-thought reasoning, spanning high-school contests, Olympiads, and international sources, for post-training reasoning models.

Our simulation studies use Humaneval [7] and GSM8K [10]. Humaneval is a benchmark of 164 Python programming problems used to assess code generation models. GSM8K is a collection of 8,500 grade-school-level math word problems for evaluating the mathematical reasoning of language models.

We chose NuminaMath, Humaneval, and InstructCoder as representative decode-heavy workloads, and GSM8K as

a prefill-heavy workload, to demonstrate METRO's benefits across different workload types.

Hardware and software setup. Our real-system implementation employed vLLM on a Google Cloud a2-highgpu-8g VM instance equipped with 8 NVIDIA A100 40GB GPUs interconnected by 600GB/s NVLink. We used data parallelism for the attention layers and EP for the expert FFN. We limited the maximum batch size for the decode phase to 32 tokens per GPU and the maximum prompts for the prefill phase to 32 per GPU. We set the context length to 8K.

Our simulation studies modeled NVIDIA B200 GPUs [37], the state-of-the-art GPU production-optimized for LLM workloads. To accommodate full model parameters while maintaining efficient GPU utilization, we used eight B200 GPUs (as configured in the NVIDIA DGX B200 Systems [12]) for the Qwen experiments and sixteen B200 GPUs for the DeepSeek experiments. In both setups, all GPUs are interconnected by 900GB/s NVLink. We set the sequence length to 1K (input) and 2K (output) for the decode phase. We set the global decode batch size to 1K, and simulate chunked prefill [4] to limit prefill batch size to 8K to meet a reasonable servicelevel objective (SLO). We used full EP across all GPUs, which yields the best performance for both prefill and decode at the chosen batch size compared to tensor parallelism (TP). Evaluation across a broader range of batch sizes and parallelism settings (§VI-C) shows that our insights from the default parameters generalize to a wide range of configurations.

Compared baselines. We compare METRO against EPLB's routing algorithm that evenly distributes tokens across expert replicas. We do not modify the expert placement and replication scheme to avoid any interference with the latency of the prefill phase — both METRO and EPLB routing algorithms use EPLB's expert placement and replication. 1.0× replication represents EPLB placement with no replication (i.e., no tokenrouting strategy is needed). Note that METRO is only applied to the decode (i.e.memory-bound) phase and the prefill phase still uses EPLB's token routing.

Simulator description. We used a proprietary industrial simulator, which is a fine-grained analytical roofline model designed to capture detailed performance behaviors of largescale GPU systems. The simulator supports multi-GPU configurations and accounts for workload imbalance by estimating the runtime based on the performance of the most bottlenecked GPU. It models hardware architecture across multiple levels, including register, shared memory, compute, L2, HBM, and network operations. Its network component computes data transfer volumes to estimate throughput, latency, and potential congestion effects. In addition, it models various mapping strategies across different parallelism levels, including tensor parallelism (TP) and expert parallelism (EP). While the overall silicon validation has been conducted internally to ensure modeling accuracy and reliability, we are unable to share its details due to NDA constraints.

Simulation traces. For each workload, we collected real traces

from running the model with token routing information for each layer. We used the replayed trace to identify two key metrics: the maximum number of tokens processed per GPU and the maximum number of activated experts per GPU. We fed these as the inputs to the simulator to describe the workload on the bottleneck GPU and derive the runtime.

## *B. End-to-end Performance Analysis*

Overall performance. Fig. 9 and Fig. 10 show our results for the real system and simulation, respectively. We consider two metrics: Total Token Throughput for prefill and decode when they are co-deployed, and Decode Latency (Time-per-Output-Token, TPOT). Compared to EPLB, METRO reduces decode latency by 1.9% - 21.8%, and improves throughput by 0.7% - 21.0%, and , across replication ratio 1.125× - 1.5× and datasets. Moreover, METRO's performance improvements increase with more replication due to EPLB's increased number of activated experts, which degrades performance (increasing decode latency by as much as 20% at 1.5× replication compared to no replication).

Takeaway: METRO improves upon EPLB routing in both decode latency (1.9%–21.8%) and total token throughput (0.7%–21.0%) across different workloads, models, and hardware configurations.

Throughput implications for prefill- vs. decode-heavy workloads. Since METRO primarily benefits the memorybound decode phase, its performance implications for total token throughput can vary across workloads with different prefill-to-decode ratios. METRO improves throughput by up to 21.0% for decode-heavy workloads (Fig. 9a, 9b, 10a, 10c), and up to 4.2% for prefill-heavy workloads (Fig. 10b, 10d).

Since prefill is compute-bound, increasing the replication ratio with available memory capacity improves token balancing and prefill performance. For decode-heavy workloads, however, EPLB routing during the memory-bound decode phase negatively affects latency, leading to a notable drop in overall throughput compared to no replication. METRO enables prefill to benefit from high replication while simultaneously improving decode performance, thereby significantly boosting overall throughput compared to EPLB routing.

For prefill-heavy workloads, high replication yields better overall throughput with EPLB routing despite the performance degradation in decode. While decode does not consume the majority of runtime, applying METRO to reduce decode latency still moderately improves throughput over EPLB routing.

![](_page_8_Figure_0.jpeg)

Fig. 9. Total token throughput and decode latency (i.e., Time-per-Output-Token, TPOT) for Qwen3-30B and DeepSeek-V3 across InstructCoder and NuminaMath datasets (\$VI-A), with full EP and decode batch size = 32 per GPU. METRO improves throughput by up to 15.9%, and reduces decode latency by up to 12.9% at  $1.5\times$  replication, compared to EPLB.

![](_page_8_Figure_2.jpeg)

Fig. 10. Simulated total token throughput and decode latency (Time-per-Output-Token, TPOT) for Qwen3-235B and DeepSeek-V3 models across GSM8K and Humaneval datasets (§VI-A). Y-axis is normalized to EPLB with 1.0x memory capacity due to confidentiality. Full EP and global decode batch size = 1024. With EPLB routing, more replication improves the prefill-heavy workload (GSM8K)'s throughput but hurts the decode-heavy workload (Humaneval)'s throughput. METRO's consistent benefit on decode latency (up to 21.8% at  $1.5\times$  replication) improves total token throughput to outperform EPLB routing for every replication ratio (up to 21.0% at  $1.5\times$  replication).

<u>Takeaway</u>: with EPLB routing, while prefill-heavy workloads benefit from higher replication, decodeheavy workloads fare poorly under the same high replication factors. METRO eliminates the need for workload-specific calibration, delivering consistently higher performance than the EPLB routing (up to 21.0% for decode-heavy and up to 4.2% for prefill-heavy workloads).

**Latency Breakdown.** Fig. 11 shows the breakdown for one layer of Qwen3-30B with various replication ratios on our real system implementation. All three sources of METRO's performance overheads (detailed in §IV-C) are insignificant

compared to its reduction on FFN latency. First, executing Algorithm IV-B adds up to  $26\mu s$  of latency for routing at a replication ratio of  $1.5\times$ , but this is more than offset by the  $81\mu s$  reduction in FFN time. Second, top-k originally accounts for only  $17\mu s$ – $19\mu s$  (<5% of the layer time), and extending it to all tokens adds at most  $3\mu s$  (<1%). Lastly, we observe no statistically significant increase in all-gather's communication time compared to all-to-all. As explained in §IV-B, this is because in the memory-bound regime (i.e., small batches), NVLink latency dominates and bandwidth cost is far below the fixed cost of launching NCCL collectives.

![](_page_9_Figure_0.jpeg)

Fig. 11. Runtime breakdown for one layer of Qwen3-30B with various replication ratios (§IV-C). Results are averaged across all layers. METRO's all-gather scheme introduces negligible overheads on top-k and inter-GPU communication. METRO's greedy approximate algorithm (i.e., Algorithm IV-C) introduces minimal computational overhead (i.e., up to 26µs), which is more than offset by its FFN time reduction (i.e., up to 81µs). Consequently, METRO reduces end-to-end decode latency by up to ∼ 10% compared to EPLB.

## *C. Decode Throughput-Latency Pareto-optimality Analysis*

While our earlier evaluations focus on a single, representative decode configuration (i.e., full EP, global batch size = 1K), real-world deployments may require varying servicelevel objectives (SLOs) which demand a broader set of deployment configurations. Specifically, there exists a fundamental tradeoff between decode throughput and latency (Time-per-Output-Token, TPOT). Different points along the tradeoff space may prefer different batch sizes or different model parallelisms. To efficiently navigate through these configurations, we conduct a decode-only analysis (since METRO does not affect prefill performance). We simulated all batch sizes and parallelism configurations listed in Table II across various replication ratios, and visualized the Pareto-optimal configuration curve for decode throughput and latency (i.e., TPOT) in Fig. 12. All simulations use the same hardware and dataset configurations described in §VI-A. We focus on understanding two research questions:

- Q1: How does METRO improve the decode throughputlatency Pareto frontier compared to EPLB?
- Q2: How does prioritizing higher decode throughput vs. lower decode latency impact the choice of deployment configuration and, in turn, the benefits of METRO?

Q1: METRO's impact on the Pareto frontier. Fig. 12 shows that for a fixed TPOT (representing a specific SLO), METRO delivers remarkably higher decode throughput of 1.98× – 4.11× compared to EPLB across models and datasets. This throughput improvement exceeds that in our earlier evaluation (§VI-B) since more flexible batch-size choices coupled with METRO 's latency reduction enable the use of larger batches to increase throughput while still meeting request SLOs. Notably, the highest throughput improvement is achieved with DeepSeek-V3 and Humaneval (i.e., Fig. 12c) at 1.5× replication and normalized 1/TPOT = 0.7, because METRO's reduced TPOT allows it to use 4× larger batch size compared to EPLB with the same request SLOs.

TABLE II DECODE PARETO CURVE SIMULATION SETUP

| Models      | Qwen3-235B-A22B       | DeepSeek-V3-671B            |
|-------------|-----------------------|-----------------------------|
| Hardware    | 8*B200                | 16*B200                     |
| Batch Sizes | 1024/512/256/128/64   | 1024/512/256/128            |
| Parallelism | TP1,2,4,8 * EP1,2,4,8 | TP1,2,4,8,16 * EP1,2,4,8,16 |

Moreover, we observe that higher expert replication ratios amplify METRO's performance gain over EPLB (a similar observation was discussed in §VI-B). This is because with more replication, EPLB increases the number of activated experts across all configurations. In contrast, the number of activated experts under METRO remains the same or even reduces (i.e., Humaneval, Fig. 12a and Fig. 12c) with more replication.

Note that while no-replication yields comparable performance to METRO in the decode phase, it leads to significantly higher latency for the prefill phase (i.e., by up to 38% compared to 1.5x replication), potentially violating SLOs and degrading overall token throughput.

Takeaway: METRO consistently advances the decode throughput-latency Pareto frontier compared to EPLB across datasets, models, and expert replication ratios, achieving 1.98× – 4.11× higher decode throughput under the same SLO with EPLB.

## Q2: SLO's impact on METRO's performance gain.

We observe that when an extremely low TPOT (i.e., strict SLO) is required, METRO's performance gains can be diminished. Specifically, as shown in Fig. 13, achieving lower TPOT requires smaller batch sizes to ensure that tokens are processed sooner. When the required TPOT is extremely low (i.e., 1/TPOT > 0.9), the batch size is small enough (i.e., ≤ 64) to cause the workload to *no longer be memorybound*, making the inter-GPU network latency the dominant bottleneck. In such scenarios, maximizing TP (rather than EP) is preferable, as smaller batch sizes require much lower communication bandwidth under TP — which is typically constrained by network bandwidth bottlenecks. Since a full TP solution eliminates any load imbalance across GPUs that may degrade performance, all curves converge to full TP at the bottom-right of Fig. 13, which makes any EP load balancing unnecessary. In addition, we do not observe any Pareto-optimal configurations using pipeline-parallelism. This is because of its degradation on decode latency (i.e., by a factor of pipeline stages) outweighs its potential improvement on throughput.

![](_page_10_Figure_0.jpeg)

(a) Qwen3-235B, Humaneval, 8\*B200 (b) Qwen3-235B, GSM8K, 8\*B200 (c) DeepSeekV3, Humaneval, 16\*B200 (d) DeepSeekV3, GSM8K, 16\*B200

Fig. 12. Pareto curves of the decode phase with simulation setup in §VI-A and Table II. Both the x-axis and the y-axis are normalized to the maximum attainable values due to confidentiality. For a fixed TPOT (representing a specific SLO), METRO delivers remarkably higher decode throughput of 1.98× – 4.11× across models and datasets. As the throughput and TPOT objectives change, METRO consistently outperforms EPLB routing across different replication ratios, batch sizes, and TP/EP mappings for both models and datasets. Higher replication ratio improves METRO's performance gain because it exacerbates EPLB's inflation on activated experts.

![](_page_10_Figure_3.jpeg)

Fig. 13. Annotated Pareto curve analysis using Fig. 12a as an example (Q2). Other curves shown in Fig. 12 follow similar trends. Batch size decreases when moving from high-throughput to low latency. With extremely low latency requirements, the batch size is small enough that network latency becomes the bottleneck, where TP is preferred because its communication overhead is minimal, making EP load balancing unnecessary.

Takeaway: METRO can benefit major SLO regimes of the decode phase; in extreme cases, low TPOT (strict SLO) requires a smaller batch size, making network latency the bottleneck instead of memory bandwidth, where full TP delivers the best performance, eliminating the need for any EP load balancing schemes.

## VII. DISCUSSION AND FUTURE WORK

We now discuss METRO's applicability to emerging disaggregated prefill-decode deployments and to future models, workloads, and accelerator hardware.

## *A. Prefill-decode disaggregation*

Recent works have argued for disaggregating prefill and decode phases [4], [18], [19], [33], [38], [50], a deployment strategy that executes prefill and decode on separate GPUs in a multi-GPU system instead of co-locating them on all GPUs, where EP load balancing for the compute-bound phase (i.e., prefill) and the memory-bound phase (i.e., decode) can be decoupled. Under disaggregation, while the decode phase would favor no-replication than EPLB with higher replication ratio (Fig. 12), METRO can still improve on no-replication in most setups — by up to 4.3% with 1.125× replication and up to 5.0% with 1.5× replication as shown in Fig. 12.

On the other hand, we note that METRO's major benefits still lies in prefill-decode co-deployed settings as we evaluated in §VI, a deployment strategy more commonly seen in smaller systems [4], [18], [19], [25], [33], [54] — e.g., a single server with a few GPUs. We anticipate that such prefill-decode codeployment will remain popular for small systems in the future. This is due to disaggregation's various inefficiencies for small systems that may outweigh its benefit, including the communication overheads to transfer KV cache between the prefill and decode instances [19], [50], the memory capacity pressure stemming from duplicating model weights across the two instances [50], and the need to dynamically reprovision resources across the two phases under workloads with unpredictable prefill-decode token ratios [4]. As such, we anticipate that METRO will remain beneficial to small multi-GPU systems' MoE deployment in the future.

## *B. Future Models, Workloads and Hardware*

Looking ahead, we anticipate a similar, if not greater, need for memory-efficient expert routing schemes. With increasingly larger MoE models, future workloads are likely to be more memory bandwidth limited due to ballooning model weights and sparser activation patterns [6], [14], [42]. From the hardware perspective, although new generations of GPUs offer higher compute capability and memory bandwidth, the increase in bandwidth might not keep pace with compute; memory remains expensive, and its linear growth seems to be lagging behind compute improvements [11], [13], [35]–[37]. As such, we believe that optimizing for memory bandwidth will remain a critical challenge. Since METRO is designed to minimize memory traffic during expert routing, it will retain its benefits in terms of lower decode latency and higher overall throughput for future workloads.

## VIII. CONCLUSION

Existing load-balancing approaches for expert-parallel MoE serving aim to balance the number of tokens each GPU processes. We show that this approach *degrades* performance rather than improving it when inference is memory-bound due to the inflated number of activated experts that exacerbates memory pressure. We propose METRO, a novel token-routing algorithm for high-performance expert-parallel MoE serving in the memory-bound regime that minimizes the number of activated experts per GPU rather than balancing tokens. Our evaluation of METRO against EPLB on both real systems (vLLM over 8 A100 GPUs) and a proprietary simulator (8-16 B200 GPUs) shows that METRO reduces decode latency by 11 - 22%, and total token throughput by 3 - 21% for Qwen3 and DeepSeek-V3 serving, where prefill and decode phases are co-deployed. In addition, by trading latency headroom for throughput, METRO improves decode throughput by up to 4.11× over EPLB at a fixed decode SLO.

## REFERENCES

- [1] "Test-and-set lock," https://www.cs.rochester.edu/research/ synchronization/pseudocode/ss.html#tas.
- [2] "vllm documentation: Cuda graphs," https://docs.vllm.ai/en/latest/ design/cuda graphs/.
- [3] "Nvidia dynamo: A low-latency distributed inference framework for scaling reasoning AI models," NVIDIA Developer Blog, 2025. [Online]. Available: https://developer.nvidia.com/dynamo
- [4] A. Agrawal, N. Kedia, A. Panwar, J. Mohan, N. Kwatra, B. S. Gulavani, A. Tumanov, and R. Ramjee, "Taming throughput-latency tradeoff in llm inference with sarathi-serve," in Proc. USENIX OSDI, 2024.
- [5] Anthropic, "Claude sonnet 4 now supports 1m tokens of context," https: //www.anthropic.com/news/1m-context, 2025.
- [6] W. Cai, J. Jiang, F. Wang, J. Tang, S. Kim, and J. Huang, " A Survey on Mixture of Experts in Large Language Models ," IEEE Transactions on Knowledge & Data Engineering, 2025.
- [7] M. Chen, J. Tworek, H. Jun, Q. Yuan, H. P. de Oliveira Pinto, J. Kaplan, H. Edwards, Y. Burda, N. Joseph, G. Brockman, A. Ray, R. Puri, G. Krueger, M. Petrov, H. Khlaaf, G. Sastry, P. Mishkin, B. Chan, S. Gray, N. Ryder, M. Pavlov, A. Power, L. Kaiser, M. Bavarian, C. Winter, P. Tillet, F. P. Such, D. Cummings, M. Plappert, F. Chantzis, E. Barnes, A. Herbert-Voss, W. H. Guss, A. Nichol, A. Paino, N. Tezak, J. Tang, I. Babuschkin, S. Balaji, S. Jain, W. Saunders, C. Hesse, A. N. Carr, J. Leike, J. Achiam, V. Misra, E. Morikawa, A. Radford, M. Knight, M. Brundage, M. Murati, K. Mayer, P. Welinder, B. McGrew, D. Amodei, S. McCandlish, I. Sutskever, and W. Zaremba, "Evaluating large language models trained on code," 2021.
- [8] S. Chen, A-transformer, haosdent, and C. Chen, "EPLB: Expert parallelism load balancer," https://github.com/deepseek-ai/EPLB, 2025.
- [9] A. Cheng, S. Liu, M. Pan, Z. Li, B. Wang, A. Krentsel, T. Xia, M. Cemri, J. Park, S. Yang, J. Chen, L. Agrawal, A. Desai, J. Xing, K. Sen, M. Zaharia, and I. Stoica, "Barbarians at the gate: How ai is upending systems research," 2025. [Online]. Available: https://arxiv.org/abs/2510.06189
- [10] K. Cobbe, V. Kosaraju, M. Bavarian, M. Chen, H. Jun, L. Kaiser, M. Plappert, J. Tworek, J. Hilton, R. Nakano, C. Hesse, and J. Schulman, "Training verifiers to solve math word problems," arXiv preprint arXiv:2110.14168, 2021.
- [11] D. Community, "Gpu memory bandwidth and its impact on performance," 2023, accessed: 2025-11-12. [Online]. Available: https: //www.digitalocean.com/community/tutorials/gpu-memory-bandwidth
- [12] N. Corporation. (2025) Nvidia dgx b200 the foundation for your ai factory. Product specifications and overview for the NVIDIA DGX B200 system. [Online]. Available: https://www.nvidia.com/en-us/datacenter/dgx-b200/
- [13] M. Davies, N. Crago, K. Sankaralingam, and C. Kozyrakis, "Efficient llm inference: Bandwidth, compute, synchronization, and capacity are all you need," 2025. [Online]. Available: https://arxiv.org/abs/2507.14397
- [14] DeepSeek-AI, A. Liu, B. Feng, B. Xue, B. Wang, B. Wu, C. Lu, C. Zhao, C. Deng, C. Zhang, C. Ruan, D. Dai, D. Guo, D. Yang, D. Chen, D. Ji, E. Li, F. Lin, F. Dai, F. Luo, G. Hao, G. Chen, G. Li, H. Zhang, H. Bao, H. Xu, H. Wang, H. Zhang, H. Ding, H. Xin, H. Gao, H. Li, H. Qu, J. L. Cai, J. Liang, J. Guo, J. Ni, J. Li, J. Wang, J. Chen, J. Chen, J. Yuan, J. Qiu, J. Li, J. Song, K. Dong, K. Hu, K. Gao, K. Guan, K. Huang, K. Yu, L. Wang, L. Zhang, L. Xu, L. Xia, L. Zhao, L. Wang, L. Zhang, M. Li, M. Wang, M. Zhang, M. Zhang, M. Tang, M. Li, N. Tian, P. Huang, P. Wang, P. Zhang, Q. Wang, Q. Zhu, Q. Chen, Q. Du, R. J. Chen, R. L. Jin, R. Ge, R. Zhang, R. Pan, R. Wang, R. Xu, R. Zhang, R. Chen, S. S. Li, S. Lu, S. Zhou, S. Chen, S. Wu, S. Ye, S. Ye, S. Ma, S. Wang, S. Zhou, S. Yu, S. Zhou, S. Pan, T. Wang, T. Yun, T. Pei, T. Sun, W. L. Xiao, W. Zeng, W. Zhao, W. An, W. Liu, W. Liang, W. Gao, W. Yu, W. Zhang, X. Q. Li, X. Jin, X. Wang, X. Bi, X. Liu, X. Wang, X. Shen, X. Chen, X. Zhang, X. Chen, X. Nie, X. Sun, X. Wang, X. Cheng, X. Liu, X. Xie, X. Liu, X. Yu, X. Song, X. Shan, X. Zhou, X. Yang, X. Li, X. Su, X. Lin, Y. K. Li, Y. Q. Wang, Y. X. Wei, Y. X. Zhu, Y. Zhang, Y. Xu, Y. Xu, Y. Huang, Y. Li, Y. Zhao, Y. Sun, Y. Li, Y. Wang, Y. Yu, Y. Zheng, Y. Zhang, Y. Shi, Y. Xiong, Y. He, Y. Tang, Y. Piao, Y. Wang, Y. Tan, Y. Ma, Y. Liu, Y. Guo, Y. Wu, Y. Ou, Y. Zhu, Y. Wang, Y. Gong, Y. Zou, Y. He, Y. Zha, Y. Xiong, Y. Ma, Y. Yan, Y. Luo, Y. You, Y. Liu, Y. Zhou, Z. F. Wu, Z. Z. Ren, Z. Ren, Z. Sha, Z. Fu, Z. Xu, Z. Huang, Z. Zhang, Z. Xie, Z. Zhang, Z. Hao, Z. Gou, Z. Ma, Z. Yan, Z. Shao, Z. Xu, Z. Wu, Z. Zhang, Z. Li, Z. Gu, Z. Zhu, Z. Liu, Z. Li,

- Z. Xie, Z. Song, Z. Gao, and Z. Pan, "Deepseek-v3 technical report," 2025. [Online]. Available: https://arxiv.org/abs/2412.19437
- [15] M. I. Dessouky, B. J. Lageweg, J. K. Lenstra, and S. L. van de Velde, "Scheduling identical jobs on uniform parallel machines," Statistica Neerlandica, 1990.
- [16] E. A. Dinic, "Algorithm for solution of a problem of maximum flow in networks with power estimation," in Soviet Math. Doklady, 1970.
- [17] W. Fedus, B. Zoph, and N. Shazeer, "Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity," 2022. [Online]. Available: https://arxiv.org/abs/2101.03961
- [18] Y. Fu, L. Xue, Y. Huang, A.-O. Brabete, D. Ustiugov, Y. Patel, and L. Mai, "ServerlessLLM: Low-Latency serverless inference for large language models," in Proc. USENIX OSDI, 2024.
- [19] B. Gao, Z. He, P. Sharma, Q. Kang, D. Jevdjic, J. Deng, X. Yang, Z. Yu, and P. Zuo, "Cost-Efficient large language model serving for multi-turn conversations with CachedAttention," in Proc. USENIX ATC, 2024.
- [20] Gemini Team, Google, "Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context," arXiv preprint arXiv:2403.05530, 2024. [Online]. Available: https: //arxiv.org/abs/2403.05530
- [21] J. He, J. Zhai, T. Antunes, H. Wang, F. Luo, S. Shi, and Q. Li, "Fastermoe: modeling and optimizing training of large-scale dynamic pre-trained models," in Proc. ACM PPoPP, 2022.
- [22] J. E. Hopcroft and R. M. Karp, "A n5/2 algorithm for maximum matchings in bipartite," in 12th Annual Symposium on Switching and Automata Theory (swat 1971), 1971.
- [23] Q. Huang, P.-A. Tsai, J. S. Emer, and A. Parashar, "Mind the gap: Attainable data movement and operational intensity bounds for tensor algorithms," in Proc. ACM/IEEE ISCA, 2025.
- [24] C. Hwang, W. Cui, Y. Xiong, Z. Yang, Z. Liu, H. Hu, Z. Wang, R. Salas, J. Jose, P. Ram et al., "Tutel: Adaptive mixture-of-experts at scale," in Proc. MLSys, 2023.
- [25] A. K. Kamath, R. Prabhu, J. Mohan, S. Peter, R. Ramjee, and A. Panwar, "Pod-attention: Unlocking full prefill-decode overlap for faster llm inference," in Proc. ACM ASPLOS, 2025.
- [26] S. G. Kolliopoulos and Y. Moysoglou, "The 2-valued case of makespan minimization with assignment constraints," 2012. [Online]. Available: https://arxiv.org/abs/1212.1609
- [27] W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. Yu, J. E. Gonzalez, H. Zhang, and I. Stoica, "Efficient memory management for large language model inference," arXiv preprint arXiv:2309.06180, 2023.
- [28] D. Lepikhin, H. Lee, Y. Xu, D. Chen, O. Firat, Y. Huang, M. Krikun, N. Shazeer, and Z. Chen, "Gshard: Scaling giant models with conditional computation and automatic sharding," 2020. [Online]. Available: https://arxiv.org/abs/2006.16668
- [29] C.-L. Li, "Scheduling unit-length jobs with machine eligibility restrictions," European Journal of Operational Research, 2006.
- [30] J. LI, E. Beeching, L. Tunstall, B. Lipkin, R. Soletskyi, S. C. Huang, K. Rasul, L. Yu, A. Jiang, Z. Shen, Z. Qin, B. Dong, L. Zhou, Y. Fleureau, G. Lample, and S. Polu, "Numinamath," [https: //huggingface.co/AI-MO/NuminaMath-1.5](https://github.com/projectnumina/aimo-progress-prize/blob/main/report/numina dataset.pdf), 2024.
- [31] J. Li, Y. Jiang, Y. Zhu, C. Wang, and H. Xu, "Accelerating distributed MoE training and inference with lina," in Proc. USENIX ATC, 2023.
- [32] K. Li, Q. Hu, X. Zhao, H. Chen, Y. Xie, T. Liu, Q. Xie, and J. He, "Instructcoder: Instruction tuning large language models for code editing," 2024. [Online]. Available: https://arxiv.org/abs/2310.20329
- [33] X. Miao, G. Oliaro, Z. Zhang, X. Cheng, Z. Wang, Z. Zhang, R. Y. Y. Wong, A. Zhu, L. Yang, X. Shi, C. Shi, Z. Chen, D. Arfeen, R. Abhyankar, and Z. Jia, "Specinfer: Accelerating large language model serving with tree-based speculative inference and verification," in Proc. ACM ASPLOS, 2024.
- [34] X. Nie, X. Miao, Z. Wang, Z. Yang, J. Xue, L. Ma, G. Cao, and B. Cui, "Flexmoe: Scaling large-scale sparse pre-trained model training via dynamic device placement," Proc. ACM Manag. Data, 2023.
- [35] NVIDIA, "NVIDIA A100 Tensor Core GPU Architecture," http://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidiaampere-architecture-whitepaper.pdf, 2020.
- [36] ——, "NVIDIA H100 Tensor Core GPU Architecture," https://www.advancedclustering.com/wp-content/uploads/2022/03/ gtc22-whitepaper-hopper.pdf, 2022, h100 (Hopper) architecture whitepaper; Accessed: 2025-11-02.

- [37] ——, "NVIDIA Blackwell Architecture Technical Overview," https: //resources.nvidia.com/en-us-blackwell-architecture, 2025, blackwell generation overview including B200; Accessed: 2025-11-02.
- [38] P. Patel, E. Choukse, C. Zhang, A. Shah, I. n. Goiri, S. Maleki, and R. Bianchini, "Splitwise: Efficient generative llm inference using phase splitting," in Proc. ACM/IEEE ISCA, 2025.
- [39] R. Qin, Z. Li, W. He, J. Cui, F. Ren, M. Zhang, Y. Wu, W. Zheng, and X. Xu, "Mooncake: Trading more storage for less computation — a KVCache-centric architecture for serving LLM chatbot," in Proc. USENIX FAST, 2025.
- [40] S. Rajbhandari, C. Li, Z. Yao, M. Zhang, R. Y. Aminabadi, A. A. Awan, J. Rasley, and Y. He, "Deepspeed-moe: Advancing mixture-of-experts inference and training to power next-generation ai scale," 2022. [Online]. Available: https://arxiv.org/abs/2201.05596
- [41] N. Shazeer, A. Mirhoseini, K. Maziarz, A. Davis, Q. Le, G. Hinton, and J. Dean, "Outrageously large neural networks: The sparselygated mixture-of-experts layer," 2017. [Online]. Available: https: //arxiv.org/abs/1701.06538
- [42] Q. Team, "Qwen3 technical report," 2025. [Online]. Available: https://arxiv.org/abs/2505.09388
- [43] A. VanAusdal and M. Burtscher, "An efficient push-relabel implementation for max-flow computations on gpus," in Proc. IEEE International Performance Computing and Communications Conference, 2025.
- [44] W. Wang, Z. Lai, S. Li, W. Liu, K. Ge, Y. Liu, A. Shen, and D. Li, "Prophet: Fine-grained load balancing for parallel training of large-scale moe models," in IEEE CLUSTER, 2023.
- [45] S. Williams, A. Waterman, and D. Patterson, "Roofline: an insightful visual performance model for multicore architectures," Commun. ACM, 2009.
- [46] J. Yao, Q. Anthony, A. Shafi, H. Subramoni, and D. K. DK Panda, "Exploiting inter-layer expert affinity for accelerating mixture-of-experts model inference," in Proc. IEEE IPDPS, 2024.
- [47] D. Yu, L. Shen, H. Hao, W. Gong, H. Wu, J. Bian, L. Dai, and H. Xiong, "Moesys: A distributed and efficient mixture-of-experts training and inference system for internet services," IEEE Transactions on Services Computing, 2024.
- [48] Z. Yuan, Y. Shang, Y. Zhou, Z. Dong, Z. Zhou, C. Xue, B. Wu, Z. Li, Q. Gu, Y. J. Lee, Y. Yan, B. Chen, G. Sun, and K. Keutzer, "Llm inference unveiled: Survey and roofline model insights," 2024. [Online]. Available: https://arxiv.org/abs/2402.16363
- [49] M. Zhai, J. He, Z. Ma, Z. Zong, R. Zhang, and J. Zhai, "SmartMoE: Efficiently training Sparsely-Activated models through combining offline and online parallelization," in Proc. USENIX ATC, 2023.
- [50] H. Zhang, P. Patel, A. Ning, and D. Wentzlaff, "Spad: Specialized prefill and decode hardware for disaggregated llm inference," arXiv preprint arXiv:2510.08544, 2025.
- [51] L. Zhang, J. Huang, S. Di, S. Matsuoka, and M. Wahib, "Can tensor cores benefit memory-bound kernels? (no!)," 2025. [Online]. Available: https://arxiv.org/abs/2502.16851
- [52] L. Zheng, L. Yin, Z. Xie, C. Sun, J. Huang, C. H. Yu, S. Cao, C. Kozyrakis, I. Stoica, J. E. Gonzalez, C. Barrett, and Y. Sheng, "Sglang: efficient execution of structured language model programs," in Proc. NeurIPS, 2025.
- [53] Y. Zhong, S. Liu, J. Chen, J. Hu, Y. Zhu, X. Liu, X. Jin, and H. Zhang, "Distserve: disaggregating prefill and decoding for goodput-optimized large language model serving," in Proc. USENIX OSDI, 2024.
- [54] K. Zhu, Y. Gao, Y. Zhao, L. Zhao, G. Zuo, Y. Gu, D. Xie, T. Tang, Q. Xu, Z. Ye, K. Kamahori, C.-Y. Lin, Z. Wang, S. Wang, A. Krishnamurthy, and B. Kasikci, "Nanoflow: towards optimal large language model serving throughput," in Proc. USENIX OSDI, 2025.