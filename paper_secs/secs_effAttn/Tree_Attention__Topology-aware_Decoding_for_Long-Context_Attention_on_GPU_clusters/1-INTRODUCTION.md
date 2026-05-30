# 1 INTRODUCTION

The self-attention operation is the core computational building block of the transformer architecture [Bahdanau et al.](#page-10-0) [\(2014\)](#page-10-0); [Vaswani et al.](#page-14-0) [\(2017\)](#page-14-0), which has become an ubiquitous and highly effective workhorse architecture currently applied at scale to language [Brown et al.](#page-10-1) [\(2020\)](#page-10-1); [Kaplan et al.](#page-13-0) [\(2020\)](#page-13-0); [Hoffmann et al.](#page-12-0) [\(2022\)](#page-12-0); [Team et al.](#page-14-1) [\(2023\)](#page-14-1); [Achiam et al.](#page-10-2) [\(2023\)](#page-10-2); [Pilault et al.](#page-14-2) [\(2023\)](#page-14-2), vision [Dosovitskiy et al.](#page-11-0) [\(2020\)](#page-11-0), audio [Betker](#page-10-3) [\(2023\)](#page-10-3), and decision-making [Chen et al.](#page-10-4) [\(2021\)](#page-10-4); [Reed et al.](#page-14-3) [\(2022\)](#page-14-3). Nonetheless, the quadratic time complexity of self-attention means that significant resources are required to train and generate from transformer-based Large Language Models (LLMs), especially for models with large context lengths.

During inference, the attention block largely determines the computational and memory requirements, which become more demanding as the input sequence length increases. Although LLMs generate one token at a time, the entire sequence of past tokens must still be stored in memory and used to compute attention scores during generation. Since attention performs a similarity matching of every token representation with every other, it incurs quadratic computational complexity in terms of flops.

There have been recent advances in training LLMs to handle extremely long contexts (up to 1M tokens) [Chen et al.](#page-10-5) [\(2023\)](#page-10-5); [kai](#page-10-6) [\(2023\)](#page-10-6); [Peng et al.](#page-13-1) [\(2023\)](#page-13-1). Such models attain qualitatively new capabilities such as extremely large-scale in-context learning of entire small datasets held in the prompt [Reid et al.](#page-14-4) [\(2024\)](#page-14-4); [Lee et al.](#page-13-2) [\(2024\)](#page-13-2); [Bertsch et al.](#page-10-7) [\(2024\)](#page-10-7). They can also avoid putting multi-modal continuous data through a lossy tokenization scheme [Reid et al.](#page-14-4) [\(2024\)](#page-14-4); [Team](#page-14-5) [\(2024\)](#page-14-5) by directly operating at the byte level [Xue et al.](#page-15-0) [\(2022\)](#page-15-0); [Wu et al.](#page-14-6) [\(2024\)](#page-14-6). The issue however is that performing inference on such long contexts is very expensive.

To speed up inference and alleviate memory requirements, recent works have attempted to alter the attention mechanism itself, either by linearizing it [Katharopoulos et al.](#page-13-3) [\(2020\)](#page-13-3), or approximating it by a kernel map [Choromanski et al.](#page-10-8) [\(2020b\)](#page-10-8); [Peng et al.](#page-13-4) [\(2021\)](#page-13-4); [Arora et al.](#page-10-9) [\(2024\)](#page-10-9), which reduces the complexity to linear at the cost of reduced expressiveness. Others have invented alternative sequence

<sup>∗</sup> Joint first-authors

mixing architectures such as state-space models which are designed to be efficiently computable in linear time and constant memory Gu & Dao (2023); Dao & Gu (2024); Katsch (2023); Sun et al. (2023); Glorioso et al. (2024).

<span id="page-1-3"></span>![](_page_1_Figure_1.jpeg)

<span id="page-1-0"></span>(a) Multi Node Tree Attention (Ours)

<span id="page-1-1"></span>(b) Multi Node Ring Attention

Figure 1: Ring and Tree Attention Topologies. Due to the associative properties of the logsumexp and max operations of Tree Attention (Fig. 1(a)), is possible to structure the reduction across the sequence as a tree, requiring asymptotically fewer communication steps than Ring Attention (Fig. 1(b)) as well as less memory and communications volume.

Other approaches utilize efficient algorithms to reduce the computational burden of attention while keeping the core computation the same. These include memory-efficient attention Rabe & Staats (2021), Flash Attention Dao et al. (2022) and Flash Decoding fla (2024), which provide a set of IO-aware kernels to map the attention operation to the GPU hardware resources in an extremely efficient way, significantly reducing the memory overhead required. Further works Character AI (2024); Kang et al. (2024); Liu et al. (2024); Nawrot et al. (2024) explore compressing or otherwise reducing the KV cache required in generation. Finally, Ring Attention Liu et al. (2023) proposes a way to parallelize the attention computation across the sequence axis between GPUs, thus enabling significantly longer contexts than can be served on a single GPU. Since our proposed method is an exact calculation of attention<sup>1</sup>, it is a plugin replacement for any multi-GPU sequence parallel mechanism such as the state of the art Ring Attention mechanisms. By leveraging the exact energy function for the self-attention block, we develop a method to speed up inference for long context use-cases when keys and values are sharded across multiple GPUs along the sequence axis.

Our proposed algorithm for computing attention via the gradient of the energy function is built on top of an efficient parallel computation and tree reduction communication strategy. In particular, this formulation lets us devise an asymptotically faster algorithm for performing decoding in which the number of communication steps scales logarithmically with the number of devices, instead of linearly in alternatives such as Ring Attention Liu et al. (2023). Our topology-aware approach illustrated in Fig. 1 significantly outperforms leading attention parallelization methods such as Ring Attention on multiple devices.

#### 2 RELATED WORKS

The computational complexity of self-attention, introduced by Vaswani et al. (2017), poses challenges for long sequences due to its quadratic dependency on sequence length,  $O(n^2 \cdot d)$ . To address this, attention **approximation** mechanisms like Linformer (Wang et al., 2020) and Performer (Choromanski et al., 2020a) reduce complexity to linear O(n) using low-rank projections and kernelized approximations on a *single device*. Sparse models such as Longformer (Beltagy et al., 2020) and BigBird (Zaheer et al., 2020) further optimize computations by restricting attention to local windows or sparsity patterns, significantly reducing resource demands while maintaining performance for specific tasks. Such methods however provide approximations to the attention mechanism while we seek to parallelize the **exact** attention computation across the sequence axis.

Theoretical work has also contributed to improving the efficiency of both exact and approximate methods. Kernel-based approaches, such as those by Tsai et al. (2019), suggest alternative formulations to self-attention that are computationally efficient. Surveys like Tay et al. (2020) highlight these advancements, emphasizing the synergy between parallelization strategies and sparsity or approximation techniques, ensuring self-attention remains scalable even in constrained computational

<span id="page-1-2"></span><sup>&</sup>lt;sup>1</sup>It can be shown empirically that Ring Attention and Tree Attention are exact computations of Attention since both methods have exactly the same activations as the forward pass of Vanilla Attention.

environments. It must be noted as well that Duman Keles et al. (2022) established lower bounds on the computational complexity of self-attention, demonstrating that achieving sub-quadratic time complexity is unlikely unless the Strong Exponential Time Hypothesis (SETH) is false.

In addition to approximation methods, several approaches focus on parallelizing the exact attention computations. FlashAttention (Dao et al., 2022), for instance, reorganizes the attention computation into smaller, memory-efficient blocks that leverage GPU memory hierarchies to enable faster and parallelized processing of exact attention, and by doing so reduces the memory complexity from quadratic to linear. Other techniques use optimized matrix operations and tiling strategies to distribute attention computations across cores or threads efficiently (Shen et al., 2021). While these methods aim to maximize throughput while maintaining the precision of exact attention, they focus on speeding up single-device attention computation. Since we parallelize exact attention across multiple devices, Ring Attention (Liu et al., 2023) is most comparable to our work. Finally, to the best of our knowledge, there are no other techniques that explore multi-device parallel decoding as we have done in this paper.

#### 3 Self-Attention

The self-attention operation can be represented as a set of dot product similarity searches between queries and keys. These similarity scores are then reduced along the sequence axis and softmaxed, so that for a given query, there is a probability distribution of the similarities of each given key. We then take the expectation of the value vectors against this distribution. We denote the queries assigned to a sequence of length N as  $\{q_a, a = 1, \dots N\}$ , where each query is a vector of size d that stands for hidden dimension,  $q_a \in \mathbb{R}^d$ , and similarly the keys and values  $\{(k_a, v_a), a = 1, \dots N\}$ . Attention can be written as

$$z^{a} = \sum_{i=1}^{N} \operatorname{softmax}(q_{a} \cdot k_{i}^{T}) v_{i}.$$

Naively computing attention in this way requires materializing the qk matrix with computational and memory cost quadratic in the sequence length. Memory-efficient attention Rabe & Staats (2021) is an iterative way to compute the softmax similarities without ever having to materialize the full attention matrix. It performs the following operations, one query (or a chunk of queries) at a time:

$$s_i^{(j)} = \exp(q_i \cdot k_i) \tag{1}$$

$$n_i^{(j)} = n_{i-1}^{(j)} + v_i s_i^{(j)} \tag{2}$$

$$d_i^{(j)} = d_{i-1}^{(j)} + s_i^{(j)} \tag{3}$$

Then, once the values v and softmax denominator d are computed, we divide to get the final softmaxed scores  $z^{(j)} = \frac{n^{(j)}}{d^{(j)}}$  for every query index j. Computing attention in this iterative manner significantly reduces the required memory.

Flash Attention Dao et al. (2022) utilizes a similar approach to reduce the memory and computational cost of attention, but the algorithm is not adapted for multi-GPU computation. Flash Attention performs the iterative algorithm of Rabe & Staats (2021) in a blockwise manner, utilizing the block-parallel computational primitives available inside single GPU tensor cores. Additionally, it precisely sizes the blocks such that they can fit into the SRAM of the GPU for the entire attention computation, effectively performing kernel fusion and preventing many unnecessary IO operations.

### 4 Self-Attention as the Gradient of an Energy Function

Following the ubiquitous success of the transformer architecture, there has been significant effort to mathematically understand the nature and meaning of the attention operation and link it to energy models (Krotov & Hopfield, 2016; Krotov, 2021; Millidge et al., 2022; Hoover et al., 2024), such as Hopfield Networks (Ramsauer et al., 2020; D'Amico & Negri, 2024). Ramsauer et al. (2020) pioneered this field by performing a similar but distinct analysis to relate self-attention with the modern Hopfield networks, providing a novel and insightful interpretation of self-attention as performing hetero-associative memory lookups using a high-powered nonlinear similarity function. This work

was later extended by Hoover et al. (2023), who derived a modified version of the transformer based off an energy function. However, while it has long been known that the softmax operation can be derived as the gradient of the following scalar function:

$$\partial_{z_j} \log \sum_{a=1}^n \exp(z_a) = \frac{e^{z_j}}{\sum_{a=1}^n e^{z_a}} = \operatorname{softmax}(z_j), \tag{4}$$

known as the log-sum-exp, an equivalent function for the self-attention block has not yet been derived. We develop in this paper a link between attention and energy functions by introducing an auxiliary *source* vector  $\zeta$ , which represents the "external contributions" to the system's energy (Hopfield, 1982). The *source*  $\zeta$  is the parameter with respect to which we compute the gradient of the scalar energy function to obtain the self-attention operation. As we will see, we need the source in order to write down the generating function of the moments of the distribution since taking the gradient with respect to  $\zeta$  yields the exact self-attention operation.

This insight allows us to make the following observation:

<span id="page-3-0"></span>**Observation 1.** Attention can be expressed at the gradient of an scalar energy function  $F(\zeta)$  with respect to the source  $\zeta$ , such that:

$$\sum_{a=1}^{N} \operatorname{softmax}(q \cdot k_a) v_a = \frac{\partial F}{\partial \zeta} \Big|_{\zeta=0},$$
(5)

where the moment generating function (i.e. the energy function)  $F(\zeta)$  is defined as:

$$F(\zeta) = \log \sum_{a} \exp\left(q \cdot k_a^T + \zeta \cdot v_a^T\right). \tag{6}$$

The proof of Observation 1 can be found in Appendix C.1. Please note that this formulation also allows to make a Bayesian interpretation of Attention in Appendix C.2 and motivates our Tree Attention algorithm in the next Section 5.

### <span id="page-3-1"></span>5 Tree Attention

In this section we show how the formulation of the attention operation as the gradient of an energy function suggests an efficient parallel strategy for computing it. The key insight is to leverage an efficient algorithm to compute the energy, and then differentiate it in order to obtain an efficient algorithm to compute attention.

#### 5.1 EFFICIENT ENERGY FUNCTION COMPUTATION

Let us focus on the case of decoding with a KV cache in a causal language model where we have one query and N keys and values. In this case, the energy function is:

$$F_{dec} = \log \sum_{a=1}^{N} \exp(q \cdot k_a^T + \zeta \cdot v_a^T) \equiv \operatorname{logsumexp}_a(\{q \cdot k_a^T + \zeta \cdot v_a^T, a = 1, \dots, N\}). \tag{7}$$

A crucial fact is that both  $logsumexp_a$  and  $max_a$  are associative operations:

 $logsumexp_a(\lbrace T_a, logsumexp_a(\lbrace R_a, S_a\rbrace)\rbrace) = logsumexp_a(\lbrace logsumexp_a(\lbrace T_a, R_a\rbrace), S_a\rbrace),$ 

$$\max_{a}(\{\max_{a}(\{T_{a},R_{a}\}),S_{a}\}) = \max_{a}(\{T_{a},\max_{a}(\{R_{a},S_{a}\})\}).$$

We can prove that this associative property allows these reductions to be performed efficiently in parallel with logarithmic time complexity, provided we have adequately many parallel workers:

<span id="page-3-2"></span>**Theorem 1.** The time complexity of a reduction operation involving an associative function, such as  $\log \operatorname{sumexp}_a$  or  $\max_a$ , over an array of size N using p parallel processors is  $O\left(\frac{N}{p} + \log p\right)$ . When the number of processors p is equal to N, the time complexity is reduced to  $O(\log N)$ .

The proof of Theorem 1 is in Appendix E.

Putting this result together, and for  $\hat{a}, \hat{b} \in \{1, \dots, t\}$  intra-chunk indices, we get the following highly parallel Algorithm 1:

#### <span id="page-4-0"></span>Algorithm 1 Single Query Energy Forward (calculating logsumexp)

- 1: Divide  $\mathbf{k}, \mathbf{v} \in \mathbb{R}^{N \times d_h}$  into p chunks  $\{\mathbf{k}_{\hat{a}}, \mathbf{v}_{\hat{a}}, \hat{a} \in \{1, \dots, N/p\}\}$  of size t = N/p
- 2: Scatter a copy of  $\mathbf{q}$ ,  $\zeta$ , and each  $\mathbf{k}_{\hat{a}}$ ,  $\mathbf{v}_{\hat{a}}$  to each of the p processors.
- 3: In parallel compute  $r_{\hat{a}} = \mathbf{q} \cdot \mathbf{k}_{\hat{a}}^T + \zeta \cdot \mathbf{v}_{\hat{a}}^T$
- 4: Compute  $m = \text{Reduce}(\max, r_{\hat{a}})$  by doing a tree reduction.
- 5: Scatter *m* to every device and update  $r_{\hat{a}} \rightarrow r_{\hat{a}} m$ .
- 6: Compute  $lse = \text{Reduce}(\text{logsumexp}, r_{\hat{a}})$  by doing a tree reduction.
- 7: Save lse, m for gradient w.r.t  $\zeta$ .
- 8: Return *lse*

#### 5.2 EFFICIENT PARALLEL DECODING

One of the core insights of automatic differentiation is that the gradient of a function  $\nabla_x f(x)$  can be computed with the same time complexity as computing f(x) Vieira (2016). The caveat however is that if the function has a deep computational graph, then the memory footprint of computing the gradient grows with that depth as backpropagation requires storing the values of the intermediate tensors. In our case, the computational graph involved in computing the energy is shallow and therefore the memory overhead is negligible. This means that if we can compute the energy efficiently, we obtain an efficient algorithm for computing its gradient (i.e. the self-attention operation) automatically.

In our case, we want to compute the gradient of the energy function with respect to  $\zeta_A$  and then set it to zero. This can be done with automatic differentiation engines having set  $\zeta$  to be a tensor of zeros from the very outset. We can however manually implement a gradient with respect to  $\zeta$  pass of the above Algorithm 1 that does not materialize  $\zeta$  in Algorithm 2 below. Note in particular that when we set  $\zeta_A = 0$ ,  $A \in \{1, \dots, d_h\}$  then *lse* involves only the logsum exp of the dot product between queries and keys.

#### <span id="page-4-1"></span>Algorithm 2 Tree Decoding (using atomic operation on each device)

- 1: Divide  $\mathbf{k}, \mathbf{v} \in \mathbb{R}^{N \times d_h}$  into p chunks  $\{\mathbf{k}_{\hat{a}}, \mathbf{v}_{\hat{a}}, \hat{a} \in \{1, \dots, N/p\}\}$  of size t = N/p
- 2: Calculate *m* and *lse* using Algorithm 1.
- 3: Scatter a copy of  $\mathbf{q}$ , m and lse, and each  $\mathbf{k}_{\hat{a}}$ ,  $\mathbf{v}_{\hat{a}}$  to each of the p processors.
- 4: In parallel compute  $r_{\hat{a}} = \mathbf{q} \cdot \mathbf{k}_{\hat{a}}^T m$ 5: Compute  $R_{\hat{a}} = \frac{\exp(r_{\hat{a}})}{\exp(lse)} \cdot v_{\hat{a}} = \exp(r_{\hat{a}} lse) \cdot v_{\hat{a}}$
- 6: Compute  $z = \text{Reduce}(\text{sum}, R_{\hat{a}})$
- 7: Return z

Notice here that by storing lse, m for the backward pass, the only remaining reduction operation that needs to be performed is the one in line 5 of the above algorithm. This single reduction takes O(N/p) time to compute the local sums on each device and log p time to communicate and combine partial results, and therefore we get the same asymptotic complexity as the logsumexp calculation.

In practice, we implement the forward and gradient w.r.t.  $\zeta$  in a single function which returns both the value and the gradient of the energy function. We can therefore put together Algorithms 1 and 2 into the following efficient parallel decoding Algorithm 3:

### <span id="page-5-0"></span>Algorithm 3 Tree Decoding (using Flash Attention 2 on each device)

- 1: Divide  $\mathbf{k}, \mathbf{v} \in \mathbb{R}^{N \times d_h}$  among p GPUs, each with a chunk  $\{\mathbf{k}_{\hat{a}}, \mathbf{v}_{\hat{a}}, \hat{a} \in \{1, \cdots, N/p\}\}$  of size t = N/p and scatter  $\mathbf{q}$  to each GPU.
- 2: Use Flash Attention 2 to compute  $o = \frac{\sum_{\hat{a}} \exp(\mathbf{q} \cdot \mathbf{k}_{\hat{a}}^T) \mathbf{v}_{\hat{a}}}{\sum_{\hat{b}} \exp(\mathbf{q} \cdot \mathbf{k}_{\hat{b}}^T)}$  and  $lse = log \sum_{\hat{b}} \exp(\mathbf{q} \cdot \mathbf{k}_{\hat{b}}^T)$ .
- 3: Recompute the global max m = Allreduce(max, lse).
- 4: Get local numerator and denominator by computing: n = o \* exp(lse m), d = exp(lse m).
- 5: Compute global numerator and denominator with:  $n_g = \text{Allreduce}(\text{sum}, n), d_g = \text{Allreduce}(\text{sum}, d)$ .
- 6: Return result  $z = \frac{n_g}{d_a}$ .

This algorithm requires three Allreduce operations in total, meaning that the required time complexity is  $O(3(N/p + \log p))$ .

### 5.3 EFFICIENT COLLECTIVE OPERATIONS USING TOPOLOGY-AWARENESS

Communication overheads While the theoretical analysis above indicates that we should see speedups when using tree-based reductions, this is not necessarily guaranteed in practice due to various potential overheads. In particular, our argument for the time complexity of our proposed Tree Decoding algorithm assumes that communication of partial results is instantaneous, which in practice is never the case. In fact, as we scale the sequence length, or the number of GPUs especially to the multi-node setting, the time taken for communication is the dominant contribution to the total execution time. However, importantly, beyond its asymptotic benefits, Tree Attention benefits from taking advantage of the two-level topology which is standard in modern GPU clusters.

We benchmark our algorithm against a previously proposed sequence parallel attention algorithm called Ring Attention. Like our algorithm, Ring Attention assumes that the sequence is sharded across GPUs and performs the attention computation without gathering all of the sequence on to a single device. Instead, it communicates shards of the keys and values in a point-to-point manner between neighboring GPUs that are logically arranged in a ring topology. This communication is overlapped with the computation of the local shard of the output. In contrast with this strategy, our algorithm scatters the query and communicates the partial result across all GPUs when performing the AllReduce operation, but does not move the key and value shards between GPUs. Consequently, in the decoding case, our method benefits from having lower communication volume and suffers less from the communication cost overhead than Ring Attention does.

<span id="page-5-1"></span>![](_page_5_Figure_11.jpeg)

Figure 2: NCCL Send/Recv between two H100 GPUs intranode and inter-node. GPU clusters offer a two-tier topology where intra-node bandwidth is significantly higher than internode. Algorithms such as Tree Attention exploit this topology by reducing inter-node communication requirements, enabling better overlap of communication with computation.

#### Implications of network bandwidth

heirarchy Ring Attention is inherently not topology-aware, and only scales within a network of homogeneous bandwidth. However, this is in conflict with the two-level network topology of modern GPU clusters, which use high-bandwidth interconnects within nodes (NVLINK or PCIe) and comparatively lower-bandwidth interconnects across nodes (InfiniBand or Ethernet). The interconnects greatly differ in bandwidth and latency (see Figure 2). Therefore, Ring Attention is bottlenecked by the slowest interconnect, and cannot always overlap the attention computation with communication. We discuss this point further in 6.3 Tree Attention improves on Ring

Attention by using network topology-aware communication patterns to increase overlap of computation and communication, and decrease this scalability bottleneck on communication from the distributed attention computation.

In practice, collective communication libraries like NCCL attempt to automatically detect what the right communication strategy is based on considerations such as data volume and network topology. In DGX clusters, for collective operations within a node, ring reduce is performed whereas a tree reduction is performed across nodes. We see that therefore using built-in collective operations such as Allreduce leads to a better performance when decoding from long contexts across multiple GPUs than enforcing the Ring Attention's point to point communication pattern. We show how the following strategy outperforms Ring Attention when decoding from very long contexts across multiple GPUs.

In our empirical experiments, we use Flash Attention 2 (Dao, 2023) within each device, both for our algorithm and for Ring Attention<sup>2</sup>. We provide a simple JAX implementation of our method in Appendix D. Note that our method mirrors Flash Decoding (fla, 2024) except in that case, the parallelization happens at the level of different streaming multiprocessors (SMs) within a GPU whereas we parallelize between different GPUs. All computations are performed in BF16.

### 6 RESULTS

Similar to Ring Attention, Tree Attention is an exact computation of attention. Since training and evaluation metrics are the same as for attention, our experimental results are focused primarily on latency in section 6.1, peak memory usage in section 6.2 and communication volumes in section 6.3. Since our algorithm computes numerically identical results as the forward pass of standard attention, our performance results transfer seamlessly to transformer architectures.

We performed experiments in Sections 6.1 to 6.3 on a DGX H100 cluster consisting of 16 nodes, each containing 8 H100 GPUs. All GPUs within the node are connected via an all-to-all NVLINK 4.0 (900GBps) topology. Nodes are connected to each other via 8 InfiniBand NDR interconnects per node (1 per GPU), each of which provides 400 Gbps (leading to an aggregate 3.2 Tbps node injection bandwidth).

We also show Ring Attention and Tree Attention comparisons when used in a Llama 3 model (Grattafiori et al., 2024) in Sections 6.4 and C.3 on viarous GPU and interconnect types: 8 H100 GPUs with NVLINK 4.0, 8 AMD MI300X GPUs with AMD infinity fabric for intranode communication and RoCE for inter-node communication, and 2 RTX 4090 GPUs with PCIe interconnect.

#### <span id="page-6-1"></span>6.1 LATENCY

In terms of practical usefulness, our study of the energy function brought to light a previously unnoted parallelizability inside the attention computation – that of the reduction of the logsumexp across the sequence dimension, which can be implemented as a parallel Allreduce. As stated in Theorem 1, it becomes theoretically possible to implement attention, per query as an  $N/p + \log(p)$  parallel operations rather than N, where the logarithmic term is proportional to the number of devices available for parallelization. When the attention is sharded across multiple devices, this asymptotic speedup creates a considerable speedup over alternative methods for decoding.

To empirically test the theoretical benefits of our Tree Attention method, we compute latency by measuring the time required to perform decoding for different sequence lengths and varying number of H100 nodes. We compare Tree Attention to our own Ring Attention execution times in Fig. 3. Both methods use Flash Attention 2 Dao (2023) for the individual-GPU attention computation. For our experiments, we benchmark on a standard attention block consisting of 16 heads of dimension 128 across different sequence lengths.

Our latency results shows how Tree Attention improves over Ring Attention as we increase the sequence length in Fig. 3(a) and increase the number of GPUs in Fig. 3(b). To better highlight execution time trends with an increasing sequence length, we have also added relative

<span id="page-6-0"></span> $<sup>^2</sup>$ A JAX-based Ring Attention implementation that uses Flash Attention 2 can be found here: https://github.com/nshepperd/flash\_attn\_jax.

<span id="page-7-2"></span><span id="page-7-1"></span>![](_page_7_Figure_0.jpeg)

(a) Relative Execution time at different sequence lengths

![](_page_7_Figure_2.jpeg)

(b) Absolute Execution time for varying cluster sizes

<span id="page-7-3"></span>Figure 3: Execution time of 16-head Tree Attention vs Ring Attention for different sizes of GPU cluster (from 1 to 16 H100 DGX nodes). Relative execution times are indexed to the Ring Attention times at a sequence length of 80k tokens.

execution time of both methods with respect to the execution time of ring attention at a sequence length of 80k. With relative execution time in Fig. 3(a), we notice that Tree attention's execution time flattens as the number of GPUs increases, while Ring Attention relative execution time continues to increase. As the plots demonstrate, as we scale the sequence length or the number of GPUs, the gap between Tree Attention and Ring Attention execution time widens asymptotically. Remarkably, Tree Attention achieves close ×8 speedups when we use 128 GPUs on a sequence length of 5.12M. We expect this trend to continue for larger sequence lengths. Please note that our DGX cluster is made up of 16 nodes each with 8 GPUs. Results for 8 GPUS use one node, for 64 GPUs uses 8 nodes and for 128 GPUs uses 16 nodes.

#### <span id="page-7-0"></span>6.2 MEMORY COST

<span id="page-7-4"></span>![](_page_7_Figure_7.jpeg)

Figure 4: Peak memory usage of a single attention block with Tree Attention vs Ring Attention when sharded between two RTX 4090s. Results were taken using the JAX memory profiler on one GPU. The difference in peak memory scales with hidden size and sequence length.

To perform Ring Attention with a distributed KV cache, it is necessary to broadcast the query corresponding to the final element of the sequence back to all devices, as outlined in step 2 of our Algorithm 1. Each device will then hold a tuple  $(\mathbf{q}, \mathbf{k}_{\hat{a}}, \mathbf{v}_{\hat{a}})$ , where  $\hat{a}$  is the chunk index, which includes the query vector and a local chunk of the keys and values specific to the sequence chunk on that device. The memory cost to store these objects is the same as for Tree Decoding.

Additionally, Ring Attention must store the  $\mathbf{k}_{\hat{a}'}$ ,  $\mathbf{v}_{\hat{a}'}$  coming from the neighbouring device and the chunk of the output o that has the same shape as the query held by that device. In contrast, our method requires storing instead only the communicated chunk of the numerator  $\mathbf{n}$ , denominator  $\mathbf{n}$  and  $\mathbf{n}$  we do not pre-allocate an output tensor but instead just return the result of doing the Allreduce to the numerator divided by the Allreduced denominator. In summary we have the following peak memory costs for Ring and Tree attention:

$$Mem_{ring} = 4btd + 2bd$$
 (8)

$$Mem_{tree} = 2btd + 2bd + 2bn_h, (9)$$

where  $d = d_h \times n_h$ , for head size  $d_h$  and  $n_h$  number of heads, b denotes the batch size and t = N/p. As such, so long as  $2bn_h \le 2btd$ , which will almost always be the case in realistic scenarios, our method always has a lower peak memory cost compared to Ring Attention.

We empirically measure peak memory utilization for our approach and Ring Attention to show that indeed memory usage is significantly less for Tree Attention in Figure 4. As predicted by theory, scaling hidden size or sequence length scales Ring Attention peak memory usage about  $2\times$  faster than Tree Attention. For example, doubling the hidden size from 2048 to 4096, doubles the gap in peak memory between two methods, going from 524MB to 1040MB.

#### <span id="page-8-0"></span>6.3 COMMUNICATION VOLUME

For Ring Attention's P2P communication strategy, the total volume of data being communicated between devices (in units of number of tensor elements) per iteration scales with p and is given by:

$$V_{ring} = 2btd \times p \tag{10}$$

where p is the number of devices. The first factor comes from counting the total number of communicated elements corresponding to  $\{(\mathbf{k}_{\hat{a}}, \mathbf{v}_{\hat{a}}), \hat{a} = 1, \dots, t\}$ , i.e.

numel 
$$(\{(\mathbf{k}_{\hat{a}}, \mathbf{v}_{\hat{a}}), \hat{a} = 1, \cdots, t\}) = 2btd.$$
 (11)

The Allreduce strategy we use in Tree Decoding requires the following volume Anthony et al. (2024):

$$V_{\text{Allreduce}} = 2 \times \frac{p-1}{p} \times \text{numel.}$$
 (12)

We communicate a shard of the numerator, denominator and max, requiring:

$$numel (n, d, m) = bd + 2bn_h.$$
 (13)

Note that we first perform on device the local reductions to obtain the local numerator and denominator on each device which consequently makes it so that t, i.e. the size of the local sequence chunk does not appear in the above expression. We then obtain:

$$V_{Tree} = 2\frac{p-1}{p} \times (bd + 2bn_h). \tag{14}$$

Our theoretical analysis shows that per iteration our algorithm maintains a lower communication volume than Ring Attention. Note however that Ring Attention when performed in the training setting with many queries overlaps communication and computation so as to hide its communication costs. However, overlapping communication and computation in the decoding case is infeasible because of how fast the attention computation on a single GPU is relative to how long it takes to communicate the chunk of keys and values between two devices.

<span id="page-8-1"></span>Concretely, let us take the example of decoding from a context of length 640000 split between 8 GPUs within one node. Let us take a hidden size of 2048 and fix our data type to be bfloat16. Each device for decoding takes  $O(10^{-5})$  seconds to perform the Flash Attention computation. The time it takes to move the keys and values of the corresponding size between adjacent GPUs as per Fig. 2 is roughly  $O(10^{-3})$  seconds. The latency incurred between nodes is even greater and therefore overlapping is not feasible due to this disparity in timescales.

### 6.4 PERFORMANCE WITH A LLAMA TRANSFORMER MODEL

To show that Tree attention can also be used in real world applications, we also measured end-to-end throughput with the Llama 3.1 8B model [Grattafiori et al.](#page-11-4) [\(2024\)](#page-11-4) on prompt sequences of length 32k, 64k, 128k and 256k using ring attention or tree attention for decoding (with prefill) 10 tokens in Table [1.](#page-9-0) We ran these experiments on 8 H100 GPUs in a DGX cluster (connected with NVLink) as well as 4 MI300X GPUs in an AMD cluster connected with AMD infinity fabric. In Table [2](#page-19-2) of Appendix [C.3,](#page-19-1) we also show similar throughput results on 2 RTX 4090 GPUs connected with PCIe. In all cases we see that Tree attention for decoding has significantly lower latency than Ring Attention for decoding with a prefill stage. Ring Attention is up to ×4 faster using 8x H100s and up to ×3 faster using 4x MI300x. We expect this gap to increase as we increase the number of nodes.

While we have previously discussed that Ring Attention works best when used with the Ring Topology of TPU clusters, Table [1](#page-9-0) and [2](#page-19-2) show that Tree Attention results generalize well to various types of systems, number of GPUs, communication protocols and network topologies.

<span id="page-9-0"></span>Table 1: Average Decoding Time (in seconds) with a prefill stage comparisons, using the 8B Llama 3.1 model with Tree Attention (ours) and Ring Attention (SOTA) across various sequence lengths and GPU types. Average results and standard error (±) are computed using 10 trial runs.

| Sequence<br>Length | 8x H100s       |                |         | 4x MI300x       |                 |         |
|--------------------|----------------|----------------|---------|-----------------|-----------------|---------|
|                    | Tree Attn      | Ring Attn      | Speedup | Tree Attn       | Ring Attn       | Speedup |
| 32k                | 0.60 ±<br>0.15 | 2.57 ±<br>0.35 | ×4      | 1.05 ±<br>0.01  | 3.57 ±<br>0.25  | ×3      |
| 64k                | 1.08 ±<br>0.10 | 4.42 ±<br>0.38 | ×4      | 2.36 ±<br>0.01  | 7.33 ±<br>0.25  | ×3      |
| 128k               | 2.68 ±<br>0.28 | 6.38 ±<br>0.58 | ×2      | 6.43 ±<br>0.25  | 16.40 ±<br>0.40 | ×3      |
| 256k               | 2.89 ±<br>0.62 | 8.19 ±<br>1.07 | ×3      | 15.30 ±<br>4.93 | 35.12 ±<br>5.02 | ×2      |

### 7 DISCUSSION AND CONCLUSION

In this paper, we have derived the energy function for self-attention and demonstrated how the computation of the derivative of this function provides a novel and efficient method for computing attention in parallel. This advantage is especially apparent when performing decoding across multiple devices, in which case our Tree Attention enables us to substantially outperform SOTA Ring Attention with an *asymptotically* superior algorithm, with ×8 speedups when we use 128 GPUs on a sequence length of 5.12M. We also see that the AllReduce operation that we use involves sending partially reduced objects, which greatly reduces the volume of communicated data as well as the peak memory requirement. In a real-world application, using the Llama 3.1 model with 1B and 8B parameters, we find that decoding with a prefill stage using Tree Attention gets us ×3-5 speedupds compared to Ring Attention. Further, by testing our method on various types of GPUs clusters including AMD MI300xs, we show that Tree Attention generalizes very well to various communication protocols and network topologies.

