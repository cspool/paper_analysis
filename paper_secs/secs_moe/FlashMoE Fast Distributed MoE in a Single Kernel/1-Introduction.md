# 1 Introduction

State-of-the-art large language models (LLMs) [\[1](#page-10-0)[–5\]](#page-10-1) have adopted the Mixture-of-Experts (MoE) architecture for its computational efficiency and strong performance across a range of tasks. The traditional Transformer block consists of a self-attention module followed by a dense feed-forward network (FFN) [\[6\]](#page-10-2). In contrast, MoE architectures replace this single FFN (Figure [2\(](#page-1-0)a)) with many identically sized FFNs, known as experts (Figure [2\(](#page-1-0)b)). A trainable neural network, known as a gate function, sparsely activates these experts by dynamically routing input tokens to the experts selected at runtime. This increase in model parameters due to more FFNs improves model quality without the corresponding increase in computational cost.

<span id="page-1-0"></span>![](_page_1_Figure_2.jpeg)

Figure 2: Transformer blocks (a) without MoE, (b) with MoE, and (c) with distributed MoE and expert parallelism. T, E, and O represent input tokens, experts, and output activations, respectively.

#### Communication overheads in MoE. As MoE

model sizes grow, GPU memory constraints prevent hosting all experts on a single device. The standard practice is to distribute experts across multiple GPUs using expert parallelism (EP), which requires token routing via many-to-many communication primitives like *AlltoAll* [\[1,](#page-10-0) [4,](#page-10-3) [3,](#page-10-4) [7\]](#page-10-5) (Figure [2\(](#page-1-0)c)). Another round of *AlltoAll* communication restores the permuted tokens processed by experts to their original order in the sequence. *AlltoAll* communication is challenging to optimize on GPU networks and is highly sensitive to straggler delays — a phenomenon where a single straggler GPU delays all others from making progress [\[8\]](#page-10-6). These communication operations can account for 68% of the total runtime [\[9,](#page-10-7) [10\]](#page-10-8), causing GPUs to remain idle (Figure [3,](#page-1-1) top left).

Kernel launch overheads in MoE. To mitigate these communication bottlenecks, recent work pipelines computation with communication kernels (Figure [3,](#page-1-1) left middle). However, the effectiveness of these solutions is limited by the overhead of launching many kernels from the CPU. Specifically, existing implementations [\[11–](#page-10-9)[14\]](#page-10-10) launch a large number of kernels per a single layer pass (see Table [1\)](#page-2-0). Frequent kernel launches negatively affect performance by: (1) creating non-deterministic kernel start times across GPUs, exacerbating straggler issues; (2) introducing unnecessary synchronization points, causing GPUs to wait on peers or the CPU before proceeding; and (3) incurring repeated global memory round trips at kernel boundaries. Although CUDA graphs [\[15\]](#page-11-0) can partially mitigate the first issue in static workloads, they are incompatible with MoE's dynamic expert routing patterns. Addressing the remaining issues requires novel solutions, which we provide in this work through complete kernel fusion and asynchronous device-initiated communication.

<span id="page-1-1"></span>![](_page_1_Figure_7.jpeg)

Figure 3: Comparing FlashMoE with state-of-the-art techniques that either do not overlap communication and computation (left, top) or do some overlap (left, middle). FlashMoE is a persistent kernel that fuses all computation and communication of the MoE operator (left, bottom). FlashMoE implements device-initiated computation (gate, expert FFN, scale) and communication tasks (right).

Our Contributions: distributed MoE in a single kernel. To overcome these fundamental inefficiencies in state-of-the-art MoE models, we develop FlashMoE a megakernel that integrates all MoE computation and communication tasks into a single persistent GPU kernel *i.e.,* a kernel that remains active for the entirety of the MoE operator (Figure [3](#page-1-1) bottom left). Instead of multiple kernel launches coordinated by the CPU, FlashMoE requires launching only one kernel, significantly reducing the involvement of the CPU. Within the fused kernel, FlashMoE implements a reactive programming model to achieve fine-grained parallelism and loosely coupled, non-blocking execution among tens of thousands of GPU threads.

In-kernel Block scheduling and Tile parallelism. FlashMoE implements *tile-level parallelism*, meaning it partitions input token matrices into smaller, independent units called *tiles*, which are processed by blocks but managed (scheduled and constructed) by warps. We specialize every thread block, except one, as *processors* to perform compute. In addition, we designate a dedicated Operating System (OS) block (4 warps) to perform administrative tasks of (1) scheduling computational work to processors (*scheduler*), and (2) decoding computational tasks from messages received from other GPUs (*subscriber*). This design allows Flash-MoE to dynamically assign tasks to GPU blocks based on *readiness*, ensuring that no GPU SM remains idle throughout the lifetime of the MoE

<span id="page-2-0"></span>

| MoE Implementation           | GPU Ops |  |  |
|------------------------------|---------|--|--|
| FlashMoE (this work)         | 1       |  |  |
| COMET [12]                   | 33      |  |  |
| Megatron-LM CUTLASS [13, 16] | 85      |  |  |
| Megatron-LM TE [13, 16]      | 261     |  |  |
| Megatron-LM + DeepEP [1]     | 432     |  |  |
| DeepSpeedMoE [11]            | 550     |  |  |

Table 1: We report number of GPU operations launched by MoE implementations by profiling with Nsight Systems [\[17\]](#page-11-2). We count operations in a single MoE layer (Gate → Dispatch → Expert → Combine) on 2 A100 GPUs, where each GPU has 32 experts. FlashMoE is the first to fully fuse the distributed MoE layer into a single GPU kernel.

operator. FlashMoE selects tile dimensions to maximize GPU arithmetic intensity while benefitting from a high-degree of parallelism.

Asynchronous and payload-efficient communication. By redesigning the MoE operator from the ground up, FlashMoE resolves fundamental inefficiencies inherent in the conventional MoE execution pipeline. One notable inefficiency is token padding during communication. To simplify programming complexity and due to symmetry constraints of collective communication APIs, existing implementations have to zero-pad token payloads to match predefined buffer sizes. This occurs when tokens are asymmetrically routed to experts, resulting in GPUs receiving much less than the expected capacity. However, these null payloads waste communication bandwidth, bloat data transfer latency and may lead to unnecessary computations on null matrices in some implementations. FlashMoE introduces *payload-efficient* communication by sending non-padded tokens only to GPUs with actively selected experts, conserving both communication and computational resources.

Technical challenges. Realizing the single-kernel design of FlashMoE required solving several technical challenges to achieve high performance: (1) lightweight computational dependency management; (2) navigating optimal SM occupancy configurations; (3) implementing in-device BLAS operations; (4) minimizing inter- and intra-device synchronization overheads; (5) implementing transfer-awareness by leveraging DMA over Unified Virtual Addressing (UVA) when available. In addressing these challenges, FlashMoE's design presents a radical departure from traditional synchronous *AlltoAll* collectives, where GPUs exhibit significant idle time during layer execution. For device-initiated communication, FlashMoE uses NVSHMEM [\[18\]](#page-11-3) to establish a global address space across all GPUs for Direct Memory Access (DMA) communication. For in-device BLAS, FlashMoE develops custom high-performance GEMM operations via CUTLASS [\[19\]](#page-11-4).

Results. Our evaluations show that FlashMoE achieves 6× latency speedup, 9× higher GPU utilization, 4× better weak scaling efficiency and 5.7× increased throughput compared to state-ofthe-art implementations. We project these performance gains becoming even better in multi-node scenarios, where inter-node communication occurs using lower bandwidth inter-node links (*e.g.,* RDMA, Infiniband).

## 2 Motivation

Synchronous Communication. *AlltoAll* or *AllGather* communication as currently used in MoE frameworks is a *synchronous* collective operation, whose completion requires the participation of all involved GPUs. Here, disparities in processing speeds or kernel scheduling among GPUs induce

<span id="page-3-0"></span>![](_page_3_Figure_0.jpeg)

![](_page_3_Figure_1.jpeg)

(a) GPU SM Utilization across baselines

(b) Kernel Launch overhead (CUDA API row)

Figure 4: 4a shows GPU utilization averaged across 100 MoE forward passes on 2 A100s with 300 GB/s unidirectional bandwidth, where we observe up to 90% idle time, due to kernel launch gaps and non-overlapping communication.

a straggler effect detrimental (Figure 13) to (1) the collective operation's performance and (2) E2E performance, as stalled GPUs cannot proceed to downstream dependent or independent tasks until the collective terminates. We expound on this problem in §A.

**Kernel Launch Overhead.** We compare the kernel launch overheads between FlashMoE and existing baselines. Table 1 shows the number of kernel launches during a single forward pass: FlashMoE launches exactly one persistent kernel, while the baselines launch up to 550 short-lived kernels to perform the same computation. Figure 4 provides a visual comparison using CUDA API traces captured by NSight Systems, illustrating the difference between FlashMoE and DeepEP. DeepEP exhibits many small CUDA API calls, with frequent stalls between individual operators, leading to increased GPU idle time (Figure 4a). In contrast, FlashMoE maintains high GPU utilization by avoiding launch overhead and synchronization gaps—achieving **93.17**% GPU utilization compared to 14% for DeepEP. See §4 for experimental results and §B for a discussion of related work.

### 3 Fused MoE Kernel Design

Modern distributed MoE systems suffer from two limitations: (1) frequent many-to-many (AlltoAll or AllGather) collectives on the critical path, and (2) significant overhead from repeated kernel launches. We address these in FlashMoE, a fully fused MoE operator implemented as a single persistent GPU kernel. Unlike previous approaches [12, 1, 11, 13, 20, 10, 21– 25], FlashMoE is the first solution to implement a completely fused Distributed MoE kernel, eliminating kernel launch overhead entirely by requiring only a single kernel launch (see Table 1).

**Actor-based model.** The design of FlashMoE is based on the actor model of concurrent computation [26–28]. We implement this model by specializ-

<span id="page-3-1"></span>![](_page_3_Figure_10.jpeg)

Figure 5: FlashMoE Fused Kernel

ing GPU thread blocks and warps into three distinct actor roles: (1) **Processor** (§F.1), (2) **Subscriber** (§F.3), and (3) **Scheduler**(§F.2). The Processor performs compute (GEMMs and element-wise operations) and tile communication. We use CUTLASS [19] as the underlying infrastructure for high-performance BLAS routines and NVSHMEM for kernel-initiated communication [18]. The Subscriber and Scheduler perform administrative functions. Specifically, the Scheduler assigns computational tasks to available thread blocks. Our key innovation is making the Scheduler both *multithreaded*, enabling high scheduling throughput, and *work-conserving*, ensuring consistently high GPU SM utilization. On the other hand, the Subscriber decodes *tile packets* from peer GPUs to task

#### Algorithm 1: FlashMoE Distributed MoE Fused Kernel

```
Input: A, O \in \mathbb{R}^{S \times H}, X \in \mathbb{R}^{E \times H \times D}, N
1 begin
         T_{\phi}, G_{\phi} \leftarrow \mathbf{FusedGate}(A)
2
3
         if blockId + 1 < N then
              \mathbf{Dispatch}(T_{\phi}, A)
              processor::start()
 5
 6
         else
               if warpID == 0 then
                   scheduler::start()
 8
                    subscriber::start(T_{\phi}, G_{\phi}, O, X)
10
               end if
11
         end if
13 end
```

<span id="page-4-1"></span>
$$D^{j} \xrightarrow{\substack{\text{Dispatch} \\ \text{packets}}} S_{b}^{i} \xrightarrow{\substack{\text{Tasks} \\ \text{Tasks}}} S_{h}^{i} \xrightarrow{\substack{\text{Schedule} \\ \text{Task} \\ \text{$GEMM_{0}$}}} P^{i} \xrightarrow{\substack{\text{Notify} \\ \text{Tasks} \\ \text{$GEMM_{1}$}}} P^{i} \xrightarrow{\substack{\text{Send} \\ \text{Task} \\ \text{$GEMM_{1}$}}} S_{b}^{j} \xrightarrow{\substack{\text{Notify} \\ \text{Tasks} \\ \text{$V$}}} S_{h}^{i} \xrightarrow{\substack{\text{Schedule} \\ \text{Tasks} \\ \text{Combine}}} P^{j}$$

Figure 6: DMoE Functional Dependencies Expressed as a Chain of Actor Interactions. We denote  $S_b$ ,  $S_h$ , and P as the Subscriber, Scheduler and Processor actors, respectively. For any actor  $a \in \{S_b, S_h, P\}$ ,  $a^i$  identifies an actor on GPU i. We define  $D_i^j$  as the operator, where GPU j dispatches packets of GPU i. This diagram expresses task dependencies at the granularity of a tile, namely  $GEMM_0$ ,  $GEMM_1$ , combine and communication produce an output tile. Notifications occur as signals propagated through shared memory (subscriber  $\leftrightarrow$  scheduler) or global memory (scheduler  $\leftrightarrow$  processor or inter-GPU communication). Note one-sided inter-GPU transfers (packet or single tile) are coupled with a signal to notify  $S_b^j$  on the receiving GPU j of the message's delivery.

descriptors (§3.1). Of the N thread blocks on a GPU, we specialize N-1 to adopt the **Processor** role. We specialize the last block as the Operating System (OS). Within this block, we specialize three warps for the **Subscriber** role and one warp for the **Scheduler** role. This split of thread blocks across actors is intentional: our goal is to use few resources for administrative tasks while reserving bulk of the resources for performing MoE computation tasks. Figure 5 summarizes the FlashMoE architecture and its constituent actors, while Algorithm 1 gives a very close translation of the system in code. Note that  $A \in \mathbb{R}^{S \times H}$  is the input token matrix;  $O \in \mathbb{R}^{S \times H}$  the output matrix; and  $X \in \mathbb{R}^{E \times H \times D}$  is a 3-D tensor of expert weights, where E denotes the number of local experts for the executing GPU, H is the embedding dimension, D is the FFN intermediate dimension and S is the sequence length.  $T_{\phi} \in (\mathbb{N} \times \mathbb{R})^{E \times C}$  is a routing table data structure, where  $T_{\phi}\left(e,c\right) = (i,w)$  indicates that token i at slot c dispatches to expert e. w is the combine weight (Equation 2) and C is expert capacity. The tuple structure of  $T_{\phi}$  is an implementation detail.  $G_{\phi} \in \mathbb{R}^{S \times E}$  captures the affinity scores produced by the gate (Equation 3). Inter-actor interactions in FlashMoE. FlashMoE decomposes MoE computation and communication at the granularity of a tile, a statically sized partition of a tensor, to achieve parallel execution and efficient overlap of tasks. Each tile maps to a discrete unit of work encapsulated by a task descriptor. The **Subscriber** decodes these task descriptors from the remote tile packets it receives. Concurrently, the Scheduler receives notifications about available tasks and dispatches them for execution to Processor actors that perform computations defined by these tasks, namely the feed-forward network (FFN) and expert-combine operations. Figure 6 show the chain of actor interactions, demonstrating how FlashMoE enforces DMoE functional dependencies.

**Determining tile dimensions in FlashMoE.** Selecting appropriate tile dimensions in FlashMoE is crucial to ensure efficient GPU utilization. An undersized tile underutilizes the GPU, while excessively large tiles create register pressure, causing performance-degrading register spills to local memory. After careful parameter sweeps, we choose tile dimensions of (128, 64). Our key insights are: increasing tile width significantly raises the register usage per thread, potentially triggering costly spills; increasing tile height without adjusting thread count increases workload per thread, harming performance. Raising the thread count per block beyond our fixed value of 128 threads reduces the

number of concurrent blocks, negatively affecting SM occupancy. Larger thread-block sizes also increase overhead from intra-block synchronization (*\_\_syncthreads()* barriers), further degrading performance. Thus, our chosen tile dimensions balance register usage, shared-memory constraints, and GPU occupancy to deliver optimal performance.

#### <span id="page-5-0"></span>3.1 Task Abstraction for Computation

Computational operators. The FFN operator is a standard position-wise feed-forward network widely used in Transformer architectures [\[6\]](#page-10-2), composed of two linear transformations separated by a nonlinear activation ϕ (e.g., GELU or ReLU):

$$FFN(x) = W_2 \cdot \phi(xW_1 + b_1) + b_2 \tag{1}$$

Here, W<sup>1</sup> and W<sup>2</sup> represent learnable weight matrices, and b<sup>1</sup> and b<sup>2</sup> are biases. The expert-combine operation, used in architectures like GShard [\[29\]](#page-12-2) and DeepSeek [\[1\]](#page-10-0), merges outputs from multiple experts by computing a weighted combination based on their affinity scores:

<span id="page-5-1"></span>
$$C_i = \sum_{j=1}^k g_{i,e} \tag{2}$$

<span id="page-5-2"></span>
$$\mathbf{h}_{i} = \sum_{j=1}^{k} \frac{g_{i,e}}{C_{i}} \cdot \mathbf{h}_{i}^{k} \tag{3}$$

In these equations, i ∈ 0, S − 1 represents an input token index, e = Ei,k identifies the k-th expert selected for token i, and gi,e is the affinity score indicating how relevant expert e is for token i.

Unified task abstraction. We unify the FFN and combine operations under a common abstraction called a *task*. Tasks provide a uniform interface for communicating tile-level work among Subscribers, Schedulers, and Processors. Formally, a task descriptor t ∈ T is defined as a tuple:

$$t = (\mathcal{M}, \star, \phi)$$

where M is a set of metadata (*e.g.,* device ID, tile index), ⋆ is a binary tensor operation (specifically, matrix multiplication · or Hadamard product ⊙), and ϕ is an element-wise activation function (e.g., ReLU or identity).

We define a task t operating on input tensors A, B, D, producing output tensor C, as follows:

<span id="page-5-3"></span>
$$\mathcal{F}_t(A, B, C, D) := C \leftarrow \phi \left( A \star_t B + D \right) \tag{4}$$

The operator ⋆<sup>t</sup> (instantiated from ⋆) may behave differently depending on the task metadata M, and the result of A ⋆<sup>t</sup> B is accumulated into D. We provide an example of task metadata in [§E.](#page-19-0)

In practice, we implement each task defined by Equation [4](#page-5-3) as a *single fused* \_\_device\_\_ decorated function which the Processor (Algorithm [2\)](#page-20-1) invokes at runtime. Fusion for t entails applying ϕ and the succeeding addition operation to registers storing the results of the binary operator ⋆t. To illustrate its flexibility, we show how the FFN and expert-combine operations can be expressed using this task framework. Note that we omit the matrix multiplication symbol (·) for simplicity. Also, ϕ<sup>1</sup> can be any activation function, while ϕ<sup>2</sup> is the identity function. The FFN is expressed as:

$$t_{1} = (\mathcal{M}, \cdot, \phi_{1}), \quad t_{2} = (\mathcal{M}, \cdot, \phi_{2}),$$

$$\mathcal{F}_{t_{1}}(A, B_{1}, C_{1}, D_{1}) \coloneqq C_{1} \leftarrow \phi_{1} (AB_{1} + D_{1}),$$

$$\mathcal{F}_{t_{2}}(C_{1}, B_{2}, C_{2}, D_{2}) \coloneqq C_{2} \leftarrow \phi_{2} (C_{1}B_{2} + D_{2}).$$

Whereas, the expert-combine operation is formalized as:

$$t_3 = (\mathcal{M}, \odot, \phi_2),$$
  
$$\mathcal{F}_{t_3}(A, S, C, C) \coloneqq C \leftarrow \phi_2 (A \odot S + C).$$

#### 3.2 Symmetric Tensor Layout for Inter-GPU Communication

Within a single GPU device, the actors in FlashMoE communicate through the GPU's memory subsystem. Specifically, the Scheduler and Subscriber actors exchange data via fast shared memory,

<span id="page-6-0"></span>![](_page_6_Figure_0.jpeg)

(a) Layout across 2 Expert-parallel Processes.

(b) State machine for DMA (top) and RDMA (bottom) communication.

Figure 7: Symmetric Tensor Layout

while other actor pairs communicate through global memory. For communication across multiple devices, FlashMoE uses *device-initiated communication*, leveraging the one-sided PGAS (Partitioned Global Address Space) programming model [30]. However, achieving scalable and correct one-sided memory accesses in PGAS without costly synchronization is a known challenge [1, 31]. We address this challenge with a provably correct and scalable solution: a symmetric tensor layout L, supporting fully non-blocking memory accesses. We define L as:

$$L \in \mathbb{R}^{P \times R \times B \times E \times C \times H}$$

where: P is the expert parallel world size, R identifies communication rounds (i.e., two rounds, one for token dispatch and one for combine), B is number of staging buffers, E is the number of local experts, E is the upscaled expert capacity (§3.2.1) and E is the token embedding dimension. Our core insight to enable non-blocking communication is temporal buffering. Specifically, we overprovision memory for the underlying token matrix by at least  $2 \cdot r$  times, where E is the number of communication rounds in the dependency graph, and the factor of 2 accounts for separate buffers for incoming and outgoing data within each communication round. For MoE models, we have E in this modest increase in memory usage eliminates the need for synchronization during one-sided data transfers. Figure 7b illustrates how cells within this symmetric tensor layout are indexed and used for Direct Memory Access (DMA) and Remote DMA (RDMA) operations. As Theorem 3.1 reinforces, this indexing scheme over E is the underlying mechanism that allows for fully non-blocking accesses eliding synchronization because all accesses are write conflict-free. See§ E for the proof.

<span id="page-6-1"></span>**Theorem 3.1.** The symmetric tensor layout L is write-write conflict-free.

To construct L, we start from the original token buffer  $T \in \mathbb{R}^{S \times H}$ , where S is the sequence length and H is the token embedding dimension. We first reorganize the sequence dimension S into three sub-dimensions representing the expert capacity (C), local expert slots (E), and the expert parallel world size (W), st:

$$C \cdot E \cdot W = C \cdot E' = S'$$
, where  $S' \ge S$  and  $E' \ge E_W$ 

In the typical case of uniform expert distribution (illustrated in Figure 7a), we have S' = S and  $E' = E_W$ , where  $E_W$  is the total number of experts in the model. Thus, the size of the token buffer is  $Size(T) = S' \cdot H$ . In Figure 7a, each cell labeled  $E_i$  (with  $i \in \{0, \dots, 3\}$ ) is a matrix of size (C, H). Extending prior work [29, 12], we introduce additional temporal dimensions R (communication rounds) and B (staging buffers). Each communication round has two fixed staging slots: one for outgoing tokens and another for incoming tokens. Each slot, indexed by dimension P, forms a tensor of shape (S', H). Therefore, the tensor size Size(L) is generally at least four times the original token buffer size, becoming exactly four times larger in the case of uniform expert distribution. Empirically, we find  $Size(L) \approx 4 \cdot Size(T)$ , contributing memory overhead  $\leq 2\%$  of memory capacity for inference of popular models. We present a thorough breakdown in  $\S D$ .

#### <span id="page-7-1"></span>3.2.1 In-place Padding for Payload Efficiency

Due to the dynamic and uneven distribution of tokens in MoE dispatch [32], GPUs commonly receive fewer tokens than their predefined expert capacity. Current MoE frameworks [11] typically pad these buffers with null tokens before computation, unnecessarily increasing communication payloads and degrading performance. In contrast, we propose *in-place padding*, performing padding directly within the local symmetric tensor buffers and thus eliminating excess network communication.

As we show in Figure 7a as a reference, each cell  $E_i$  is sized according to the expert capacity C. We further align this capacity to ensure divisibility by the tile block size bM=128, guaranteeing safe and aligned memory reads by Processor threads consuming remote tokens. This in-place padding strategy slightly increases the memory footprint of L, as described below:

$$Size(L) \approx \begin{cases} 4 \cdot Size(T), & \frac{S}{E} \geq bM \\ 4 \cdot \frac{bM \cdot E}{S} \cdot Size(T), & \text{otherwise} \end{cases}$$

#### <span id="page-7-0"></span>4 Evaluation

We implement (§G) and evaluate FlashMoE across five metrics: Forward Latency (§ 4.1), GPU Utilization (§ 4.2), Overlap Efficiency (§ 4.4), Throughput (§ 4.3), and Expert Scalability (§ 4.5). We run experiments on a server with 8 NVIDIA H100 80G GPUs interconnected via NVLink, 125 GB of RAM, and 20 vCPUs. We used PyTorch 2.6.0, CUDA 12.8, and Ubuntu 22.04. All experiments use MoE transformer models configured with 16 attention heads, an embedding dimension of 2048, and an FFN intermediate size of 2048. We apply Distributed Data Parallelism (DDP) and Expert Parallelism for all experiments. We execute only the forward pass over a single MoE layer and measure the average runtime of 32 passes after 32 warmup passes. We use top-2 routing with a capacity factor of 1.0. We compare FlashMoE against several state-of-the-art MoE systems: (1) Comet [12], (2) FasterMoE [14], (3) Megatron-CUTLASS [13], and (4) Megatron-TE: Megatron-LM with Transformer Engine [33]. Comet relies on cudaMemcpyPeerAsync [34], while FasterMoE and Megatron-LM use NCCL exclusively for communication.

**Desiderata.** We observe Comet exhibiting anomalously bad performance values at 8 GPUs, so we exclude their results from evaluations at 8 GPUs and only include for results at  $\leq$  4 GPUs. We evaluate FlashMoE using FP32 precision whereas all baselines use FP16. We do so because (1) of incomplete fp16 tuning (§H) and (2) no baseline supports FP32. Note, this precision discrepancy disadvantages FlashMoE by doubling its communication volume and computational workload.

#### <span id="page-7-2"></span>4.1 Forward Latency

<span id="page-7-3"></span>![](_page_7_Figure_8.jpeg)

Figure 8: Forward Latency as the Number of Tokens per GPU increases.

We first measure the forward latency of FlashMoE across different sequence lengths on both 4 and 8 GPU setups (Figure 8). FlashMoE consistently outperforms all baselines, with especially notable improvements at longer sequence lengths. On 4 GPUs, it achieves up to **4.6**x speedup over Megatron-TE at 16K tokens, and **2.6**x over FasterMoE. The gains are even more pronounced at 8 GPUs where FlashMoE maintains low latency, exhibiting up to **6.4**x speedup over baselines that degrade steeply due to increasing communication costs as token buffers increase proportionally.

#### <span id="page-8-0"></span>4.2 GPU Utilization

To quantify GPU efficiency, we measure Streaming Multiprocessor (SM) utilization during the forward pass (Figure 9). FlashMoE achieves 93.17% average SM utilization, over 9x higher than FasterMoE (9.67%), 6.8x higher than DeepEP+Megatron-LM (13.55%) 4x higher than Megatron-TE (59.11%), and 2.2x higher than Comet (42.31%). This improvement stems from our fully fused kernel architecture and finegrained pipelining of compute and communication tasks. By eliminating idle gaps due to kernel launches and enabling in-kernel task scheduling, FlashMoE ensures SMs remain busy with productive work throughout execution.

<span id="page-8-3"></span>![](_page_8_Figure_2.jpeg)

Figure 9: SM utilization, defined as the ratio of cycles in which SMs have at least one warp in flight to the total number of cycles [17]. Values represent the average SM utilization over 100 iterations.

#### <span id="page-8-2"></span>4.3 Throughput

As shown in Figure 10, FlashMoE scales linearly with GPU count, reaching 17.7 MTokens/s at 8 GPUs. This is over **5.7**x higher than FasterMoE and **4.9**x higher than Megatron-TE and Megatron-CUTLASS. Notably, these results are achieved despite *Flash-MoE operating entirely in FP32, while baselines use FP16*. This indicates that FlashMoE 's design eliminates throughput bottlenecks not by exploiting lower precision, but by maximizing hardware utilization and eliminating host-driven inefficiencies.

<span id="page-8-4"></span>![](_page_8_Figure_6.jpeg)

Figure 10: Throughput when scaling the number of GPUs, computed as  $\frac{T \times N_G}{\text{latency}}$ .

#### <span id="page-8-1"></span>4.4 Overlap Efficiency

<span id="page-8-5"></span>![](_page_8_Figure_9.jpeg)

![](_page_8_Figure_10.jpeg)

- (a) Latency as Number of GPUs increases.
- (b) Weak scaling efficiency

Figure 11: Weak scaling efficiency. We define Overlap Efficiency  $O_e$  to be  $O_e = T(2)/T(N_G)$ , where  $T(N_G)$  is the latency at  $N_G$  GPUs and T(2) is the latency at 2 GPUs.

We evaluate the extent to which FlashMoE overlaps communication and computation by measuring weak scaling efficiency as the number of GPUs increases (Figure 11b). We note that most baselines fail to execute at a single GPU, hence why we use 2 GPUs as the reference point. We observe that Megatron-CUTLASS and Megatron-TE degrade significantly, with overlap efficiency dropping below 50% at  $\geq$  4 GPUs. FlashMoE gives up to 3.88x and 4x higher efficiency at 4 and 8 GPUs, respectively. Figure 11a further illuminates this efficiency, as FlashMoE shows stable forward latency growth. These results corroborate that FlashMoE's actor-based design and asynchronous data movement achieve near-ideal overlap.

#### <span id="page-9-0"></span>4.5 Expert Scalability

<span id="page-9-1"></span>![](_page_9_Figure_1.jpeg)

Figure 12: Forward Latency as the *Number of experts* increases.

We analyze how FlashMoE scales with increasing number of experts at fixed sequence length (T = 16K). Note that for the discussed plots, the number of experts on the x-axis is the *total number across all GPUs*. Each GPU gets 1/8th of this value. As seen in Figure 12, FlashMoE maintains *low, uniform* latency, as desired, even as the number of experts grows from 8 to 128. In contrast, baselines exhibit superlinear latency increases due to increased kernel launch overheads. FlashMoE outperforms these baselines by up to **4**X at 4 H100s and **6.6**X at 8 H100s, both at 128 experts. FlashMoE 's payload-efficient communication and scheduler-driven in-kernel dispatching allow it to sustain expert parallelism without incurring the communication and orchestration penalties seen in other systems. These results reinforce FlashMoE 's scalability for ultra-sparse MoE configurations.

#### 5 Limitations and Future Work

**Engineering complexity.** Fully fused, persistent kernels demand deep GPU + distributed-systems expertise; future work may investigate compiler/DSL abstractions to lower this barrier.

**FP16 inefficiency.** Our FP16 path is suboptimal (§H) due to insufficient tuning. We anticipate addressing this gap with autotuned GEMM operators like cuBLASDx [35] or CUTLASS builders.

**Training support.** This work targets inference; enabling training will require fusing backward computation and gradient communication with new bookkeeping and task descriptors.

### 6 Conclusion

We introduce FlashMoE, the first work to fuse the entire Distributed MoE operator into a single persistent GPU kernel that unifies computation, communication, and scheduling via actor-style concurrency, warp specialization, and async (R)DMA. We address two dominant bottlenecks in prior systems—CPU-managed synchronous communication and fragmented multi-kernel execution. Empirically, FlashMoE achieves up to  $6\times$  speedup,  $9\times$  higher GPU utilization, and  $5.7\times$  throughput for distributed MoE. Looking ahead, we see a shift from CPU orchestration to fully autonomous, GPU-native pipelines—extending this fusion approach to training and beyond.

### 7 Acknowledgements

This research is supported by NSF Award #2444537 and ACE, one of the seven centers in JUMP 2.0, a Semiconductor Research Corporation (SRC) program sponsored by DARPA. This work also used resources of the National Energy Research Scientific Computing Center, a DOE Office of Science User Facility supported by the Office of Science of the U.S. Department of Energy under Contract No. DE-AC02-05CH11231 using NERSC award ASCR-ERCAP0030076. We acknowledge and thank Dr. Giulia Guidi for providing access to these NERSC supercomputing resources.

