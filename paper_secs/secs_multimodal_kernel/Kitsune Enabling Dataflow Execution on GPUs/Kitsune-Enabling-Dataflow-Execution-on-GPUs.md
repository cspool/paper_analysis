# Kitsune: Enabling Dataflow Execution on GPUs

MICHAEL DAVIES, NVIDIA, USA NEAL CRAGO, NVIDIA, USA KARTHIKEYAN SANKARALINGAM, NVIDIA, USA STEPHEN W. KECKLER, NVIDIA, USA

State of art DL models are growing in size and complexity, with many modern models also increasing in heterogeneity of behavior. GPUs are still the dominant platform for DL applications, relying on a bulk-synchronous execution model which has many drawbacks and is ill-suited for the graph structure of DL applications. Many industry and academic works attempt to overcome these by employing vertical fusion but this approach still fails to realize three untapped opportunities: (1) the fact that many resources on the GPU are idle while only one operator executes due to temporal multiplexing of the SM; (2) lower energy from more intelligent on-chip data-movement which lends to higher performance in a power-provisioned environment. (3) inability to exploit hidden or reduction dimensions as a source of parallelism to ease pressure on batch size. This paper explores relatively uncharted territory, answering the following key question: Can modest adjustments to the current GPU architecture enable efficient dataflow execution, thereby circumventing the constraints of vertical fusion without necessitating a clean-slate architecture design. We develop Kitsune – a set of primitives that enable dataflow execution on GPUs and an end-to-end compiler based on PyTorch Dynamo. Across 5 challenge applications, Kitsune can provide 1.3×-2.3× and 1.1×-2.4× performance improvement as well as 41%-98% and 16%-42% off-chip traffic reduction for inference and training, respectively.

### ACM Reference Format:

Michael Davies, Neal Crago, Karthikeyan Sankaralingam, and Stephen W. Keckler. 2025. Kitsune: Enabling Dataflow Execution on GPUs. 1, 1 (February 2025), [20](#page-19-0) pages.<https://doi.org/10.1145/nnnnnnn.nnnnnnn>

### 1 Introduction

Graphics Processing Units (GPUs) have become the dominant platform for executing deep learning (DL) algorithms due to their amenability to matrix-multiplication and other common DL operations. Historically designed for Single Instruction, Multiple Thread (SIMT) execution with extensive register files, GPUs have evolved significantly. They now boast intricate memory hierarchies, specialized Tensor Cores for general matrix-multiply (GEMM) computations, and support for atomic memory instructions [\[34\]](#page-18-0). Depicted in Figure [1,](#page-1-0) GPUs (a) employ a relatively simple bulk-synchronous programming (BSP) model (c), where a set of independent work items for a single operator (commonly implemented as a single kernel) are run to completion followed by a global barrier before the next set is dispatched. However, the BSP model is a misfit to the directed-acyclic graph structure of DL applications, and hence encounters inefficiencies centered around three key areas: the inability to exploit on-chip data locality of intermediate data

Authors' Contact Information: Michael Davies, NVIDIA, USA, karus@nvidia.com; Neal Crago, NVIDIA, USA, ncrago@nvidia.com; Karthikeyan Sankaralingam, NVIDIA, USA, karus@nvidia.com; Stephen W. Keckler, NVIDIA, USA, skeckler@nvidia.com.

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.

© 2025 Copyright held by the owner/author(s). Publication rights licensed to ACM. Manuscript submitted to ACM

Manuscript submitted to ACM 1

<span id="page-1-0"></span>![](_page_1_Figure_0.jpeg)

Fig. 1. (a) Overview of GPU organization, (b) example DL graph, and (c) stylized comparison of execution techniques. In (c), TensorCore and SIMT resources of the GPU are depicted separately.

passed between operators due to large memory footprints spilling to DRAM, and idle resources due to limited parallelism or low arithmetic intensity within operators.

Vertical fusion, depicted in Figure 1 (c), is an approach for GPUs to amortize kernel launch overheads and improve data locality between operators and thus reduce off-chip memory traffic through fusing multiple operators into a single CUDA "mega kernel", establishing the need for flexibility in GPU execution. Under this paradigm, GPU's execution resources are *temporally* multiplexed between several "fused" operators, interleaving partial executions of each operator, allowing tiles of intermediate data to stay resident on chip for reuse, and removing the need for kernel barriers between fused operators. This multiplexing is depicted in Fig 1 (c) by how at a given time, only the TensorCores or SIMT resources are active. This technique has been commercialized in tools such as TensorRT [49] and advanced through academic endeavors like Welder [45], Astitch [62] and others [9, 15, 57, 58, 61]. Despite its effectiveness, vertical fusion leaves three performance opportunities untapped. First, because of temporal multiplexing, the technique does not take advantage of the many idle resources available while one operator is executing. Second, because of how vertically fused operations are structured, spilling large intermediates to DRAM can become unavoidable, incurring a round-trip DRAM latency penalty. Third, it is unable to exploit reduction or hidden dimensions for parallelism to ease the need for large batch-level parallelism.

Many academic and industry approaches (Groq for e.g.) recognize that dataflow execution (i.e. concurrently executing operators across *space* rather than time) aligns more naturally to the graph structure of DL applications – mitigating the above inefficiencies of BSP and vertical fusion with *clean-slate* architectures [1, 2, 21, 39, 40, 44, 50]. The focus of these efforts is dataflow execution of DL (sub)graph nodes at the single-chip level, while recognizing other aspects of dataflow execution also exist at the system level [3] and within the matrix-engines themselves [5, 6, 43]. **This paper explores relatively uncharted territory, answering whether modest adjustments to the dominant GPU architecture and software stack can enable efficient dataflow execution at the chip-level.** 

Our key insight is two complementary software-hardware primitives are sufficient to enable dataflow execution on GPUs. They are: 1) a software-only ring queue which facilitates inter-CTA (Cooperative Thread Array) communication by using the L2 cache and global atomics; 2) a modest change to the GPU's grid scheduler to enable it to exploit the heterogeneity of concurrently executing operators. We find that an effective end-to-end compiler can be built that uses these primitives to allow automatic lowering of DL applications to dataflow execution on GPUs, avoiding the need for new IRs or a complex code generation backend. This system, named "Kitsune", addresses the problems of the BSP

model: executing more than one operator concurrently and passing tiles of intermediate data through on-chip queues increases available parallelism and reduces memory bandwidth pressure. This is depicted in Figure [1](#page-1-0) (c), where multiple different operators can executed simultaneously across both the SIMT and TensorCore resources on the GPU.

The contributions of this work are as follows.

- (1) A systematic characterization of DL applications that highlights the mismatch between graph behavior and GPU bulk-synchronous execution.
- (2) A design and analysis of Kitsune's SW/HW primitives needed to enable synchronous dataflow execution on GPUs.
- (3) A design and implementation for the Kitsune compiler which enables applications to transparently leverage dataflow execution on GPUs.
- (4) An evaluation of Kitsune across several diverse DL models, spanning inference and training, on a SOTA A100 class GPU. We show 1.3× to 2.4× speedups, with 16%-98% reduction in memory traffic (which indirectly serves as a form of power/energy savings). We also compare Kitsune to SOTA vertical fusion techniques and elucidate the reasons why Kitsune is able to achieve superior performance.
- (5) A sensitivity study of Kitsune's hardware synergy. When increasing inexpensive hardware resources by 2× (on-chip compute, on-chip L2 cache bandwidth), while keeping expensive resource (memory bandwidth) unmodified, Kitsune effectively achieves 47% and 27% speedup for inference and training, respectively, while baseline execution shows only 18%-26%.

#### <span id="page-2-0"></span>2 Background

This section presents an overview of Deep Learning, GPU hardware, it's connection to the BSP execution model, and pointers to recent hardware support.

Deep Learning. DL applications use learned parameters to make predictions on data across a variety of application domains, combining input and parameter tensors (multidimensional arrays) with mathematical operators such as linear projections (Linear) to produce outputs. A computational graph is constructed during execution which is then used in automatic-differentiation for computing gradients to "train" parameters. Common operators include linear projection, element-wise operators such as ReLU and addition, attention, layernorm, softmax, and convolution. Linear projection, attention, and convolution are all computationally similar; reducing to general matrix-multiplications (GEMMs) whose dimensions are dictated by the operator parameters. Often GEMMs are colloquially used to express the entirety of work done by these operators.

GPU Hardware. Figure [1](#page-1-0) (a) presents a modern GPU chip design [\[32\]](#page-18-5). A GPU comprises a set of multiple Streaming Multiprocessor (SM) processing cores, a globally shared L2 cache (among all the SMs), and main memory accessible through a high bandwidth interface. SM execution is managed by a GPU-global grid scheduler which is responsible for dispatching work sent from the driver over PCIe. The SM includes local data storage, including a large register file and a memory that can either be configured as an L1 cache or a software-managed scratchpad memory (also known as shared memory). Each SM also includes compute functional units for general computation (SIMT Cores), and dedicated hardware for accelerating tensor operations such as matrix-multiplication (Tensor Cores). The memory system additionally includes support for atomics which are facilitated by the L2. SM counts range from 80 for V100 [\[35\]](#page-18-6), 108 for A100 [\[33\]](#page-18-7), and 132 for H100 [\[34\]](#page-18-0). Roughly speaking, the L2's bandwidth is 3× of main memory bandwidth [\[11–](#page-18-8)[13\]](#page-18-9).

<span id="page-3-0"></span>![](_page_3_Figure_0.jpeg)

Fig. 2. Visualization of the difference between Kitsune and Vertical Fusion for (a) an MLP with a large hidden dimension, (b) a reduction operation, and (c) an operator that sends intermediate to multiple consumers.

**GPU Execution model**. A GPU **kernel** (typically mapped one-to-one with DL operators) is code that is compiled and run on the GPU's SIMT abstraction. Kernels are run with a BSP execution model where one kernel occupies the GPU at a time and finishes completely before the next kernel starts. Each kernel's threads are organized into collections of threads known as **cooperative thread arrays** (CTAs, a.k.a. "threadblocks"), and all the CTAs of a kernel make up a **grid**. A CTA is a non-divisible quanta of work that is mapped to and executes to completion on an SM, where each thread maintains private state in the register file, and communicates with other threads in the same CTA via shared memory.

In the microarchitecture, threads within a CTA are grouped into fixed-size warps (32 for most modern GPUs) which execute in lock step. In modern GPUs, multiple CTAs can run simultaneously on a single SM. Modern GPUs allow multiple grids to execute simultaneously in limited situations, and have included rich support for atomic memory operations, allowing threads within a CTA, grid and across grids to synchronize with global atomics [29]. CUDA Streams [22] and CUDA Graphs [30] are APIs that enable users to specify which kernels are independent and can run simultaneously. In practice, neither of these result in co-executing kernels – due to first-in-first-out ordering and queuing hardware in the global grid scheduler. Current GPUs restrict that a new kernel can only start dispatching once all the CTAs from the current one have dispatched resulting in minimal execution overlap of the two kernels [31, 48].

### <span id="page-3-1"></span>3 Motivation and Program Behavior Characterization

In this section, we motivate dataflow execution by examining the opportunities present across a range of DL applications' operator graphs. The applications we focus on are summarized qualitatively in Table 1. Our DL applications include DLRM [27], MeshGraphNets [38], NeRF [26], GraphCast [18], and Llama 3 8B [7]. Note for Llama 3, we discuss it in terms of three separate use-cases: (1) training which encompasses the forward and backward pass for a whole set of tokens, (2) context-phase ("ctx") which encompasses just the forward pass prefill step of inference, and (3) decode-phase ("tok") which encompasses the autoregressive token-generation step of inference. The context and decode phases are

Table 1. Description of selected applications.

| Application   | Year | Use Case                       |
|---------------|------|--------------------------------|
| DLRM          | 2019 | Predicting ad clicks           |
| MeshGraphNets | 2020 | Mesh based physical simulation |
| NeRF          | 2021 | View synthesis                 |
| GraphCast     | 2022 | Weather forecast prediction    |
| Llama 3 8B    | 2024 | Language modeling              |

<span id="page-4-0"></span>inference only and will not appear in training results. We first discuss several common patterns, summarized in Figure [2,](#page-3-0) which we observe are frequently exhibited in popular DL applications, focusing on the limitations of state-of-art vertical fusion compared to Kitsune dataflow.

Operator Patterns. Figure [2](#page-3-0) depicts three common graph patterns composed from Linear, Elementwise, and Reduce operators. These patterns are abstracted from detailed shapes for our applications encompassing examples found in both inference (forward-pass) and training (back-propagation). Elementwise and Reduce operations are not computationally intensive and cannot use the TensorCores on the GPU, unlike GEMM operations which do. Fig [2](#page-3-0) (a) depicts a common scenario where a linear layer (I.e. a GEMM) produces a large output dimension ("N") which is then fed to a downstream Elementwise and subsequent linear layer. This is seen in many MLPs, and is especially common in the feed-forward network in transformer models which perfrom an projection (linear layer) into a high-dimension followed by a non-linear operation and subsequent projection back to a smaller dimension. Fig [2](#page-3-0) (b) depicts a simple reduction operation. This can be found typically in split-K GEMM operations where partial sums need to be reduced. In addition, reductions over the batch dimension are very common in back-propagation. Finally, Fig [2](#page-3-0) (c) depicts a scenario where one Elementwise feeds two consumer GEMM operations. This is very common in back-propagation, notably for a Linear-Activation pair the backward pass involves computing the gradient for the activatino function which feeds two gradient GEMMs one for the input activation and one for the weights.

Vertical Fusion Mechanism. Vertical fusion seeks to improve DL performance by combining multiple DL nodes and temporally switching between partial executions of each node to avoid main-memory traffic of intermediate data. Different code-regions in a single vertically fused kernel encode the entire computation of the fused subgraph, with each CTA working on data-parallel shards of the problem. Keeping with the BSP execution model in which vertical fusion operates, CTAs do not interact with each other and tiles of intermediate data are only passed between the partial executions within a CTA. Therefore, implementations of vertical fusion prioritize staging data in shared memory or the register file [\[45,](#page-19-2) [62\]](#page-19-3).

Vertical Fusion's Utilization Limitations. Vertical fusion is unable to exploit idle GPU resources. Figure [3](#page-6-0) shows, for our application selection, a breakdown of application runtime with respect to SM and DRAM utilization measured from performance counters by NSIGHT Compute for vanilla PyTorch and inference compiled with TensorRT (representative of vertical fusion). We define "low" utilization as less than 33% of peak, generating four categories. "Both Low" implies that both DRAM and SM utilization are less than 33%, "Low SM" and "Low DRAM" categories have only one resource below 33% utilization, and "Neither Low" is time spent with DRAM and SM utilization above 33%. While "Low" categories indicate portions of time spent with GPU resource severely underutilized, there remains some opportunity even in the "Neither Low" case.

Note that TensorRT does not support training so we only show TensorRT result for inference. Across our applications for bulk-synchronous (unfused) execution, we see 20-25% and 37-67% of runtime is spent with both low SM and DRAM utilization for inference and training, respectively; with the exception of DLRM (which has 77% and 89%) and Llama

Context / Train (which has 0.1% and 0.3%). Indeed TensorRT fusion does improve this picture for inference with all applications showing a decrease in "low" utilization with the exception of MeshGraphNets. Despite this, there is still ample opportunity for dataflow to capitalize on idle resources shown by the large amount of runtime spent with low utilization of one or both resources. Even if neither resource are low, as exemplified by NERF inference with TensorRT, there's still opportunity for dataflow: operators executing by dataflow eliminates DRAM traffic which would lower the effective DRAM utilization, leading to additional headroom.

Vertical Fusion's Coverage Limitations. We discuss coverage limitations by considering graph patterns shown in Figure [2.](#page-3-0) In Figure [2](#page-3-0) (a), when an operator produces an intermediate with a large hidden dimension (E.g. MLP with ≥ 768 on an A100 with 192 KB of shared memory), the resultant GEMM tiles exceed the shared-memory capacity. Because of this, even modestly sized intermediates can cause spills to off-chip DRAM[1](#page-5-0) As a result of this, the latency from a round-trip to/from off-chip DRAM is incurred for spilled data. On an A100 GPU, this latency is ≈ 409ns or 572 cycles at 1.4GHz. In addition to spilling, because of how vertically fused kernels temporally multiplex the SM, either the SIMT cores or Tensor cores will be idle during computation of each operation, leading to under-utilization of the SM. Naively mitigating this by assigning multiple CTAs to an SM has a major drawback of cutting the effective shared-memory per CTA by the same factor, exacerbating the capacity problem.

Figure [2](#page-3-0) (b) depicts a reduction operation. One notable and unavoidable place where reductions are common is in back-propagation where gradients are often reduced over the batch dimension. Despite the batch dimension usually being a source of abundant parallelism, here neither BSP or Vertical Fusion are able to extract parallelism from the batch dimension for gradient reduction operation. This means that a small number of CTAs end up performing a reduction, leaving most SMs idle.

Finally, Figure [2](#page-3-0) (c) depicts a case where one operator's output is consumed by multiple downstream operations. In particular, this pattern of multicast is representative of back-propagation for a standard Linear+Activation graph. Similar to (a) we find this can lead to spilling tiles of data to off-chip memory since the state needed for one successor child may over-run the shared memory, evicting an intermediate that is needed for a different child. We also see that heterogenous operations cannot simultaneously execute on the SM, leading to underutilization. In general, we observe prior work on vertical fusion does not support back-propagation at all, though we depict in our figure how it would be implemented.

Kitsune. Our insight is that dataflow – i.e. having different operators co-execute spatially across SMs, rather than temporally switching between executing operators across time – solves all these problems, while preserving the benefits of vertical fusion. Kitsune implements dataflow execution by mapping single operators to CTAs, then passing tiles of intermediate data to downstream operator CTAs using inter-CTA queues residing in on-chip memory to avoid off-chip memory accesses. In doing so, operator CTAs are concurrently mapped and executed across the SMs of the GPU. Multi-cast and parallel reduction simply become one-to-many and many-to-one communication patterns using our data-queue. The capacity issue, is then trivially solved by splitting hidden dimensions spatially. Using our modified grid scheduler, hardware under-utilization can be solved by assigning different types of CTAs to an SM for co-execution.

Revisiting Figure [2,](#page-3-0) Kitsune can extract performance wins for all of these patterns. First, Kitsune is able to simultaneously execute heterogeneous operations on an SM, addressing under-utilization. Second, with significantly reduced data-movement (especially to/from off-chip DRAM) energy is saved, potentially allowing for higher clock frequencies to be sustained. Finally third, Kitsune is able to extract parallelism from hidden and reduction dimensions.

<span id="page-5-0"></span><sup>1</sup> Indeed the L2 could provide some additional buffering but since every SM runs a data-parallel replica of the same subgraph with the same intermediate storage requirements, that capacity is quickly overrun as well.

<span id="page-6-0"></span>![](_page_6_Figure_0.jpeg)

Fig. 3. Application runtime spent in different combinations of measured SM and DRAM utilization. Low utilization means less than 33% of peak.

#### 4 Kitsune Primitives for Dataflow on GPUs

Kitsune enables the GPU to logically operate with a synchronous dataflow execution model that relaxes the assumptions of bulk-synchronous execution, relying on and leveraging dependence and communication between CTAs from different pipeline stages. The execution model comprises of CTAs explicitly communicating with each other which triggers and throttles execution speeds. When data is available in a queue, a CTA starts its execution, writing results to its producer queue. When there is no data in its queue, it idles. The first node of a subgraph reads activations from main-memory (essentially outputs of preceding subgraphs or bulk-synchronous code), and the last node writes results to main-memory. In addition to reading from a queue, a CTA is free to read any other values from memory, and similarly can write to main-memory in addition to writes to its producer queue to trigger its successor. In the formal context of execution models models [\[19\]](#page-18-18), Kitsune falls under the category of synchronous dataflow. Future work can examine further extensions like dynamic dataflow.

The following subsections develop Kitsune's two key primitives that enable this synchronous dataflow execution model. The first is a synchronized queue structure which allows inter-CTA communication ([§4.1\)](#page-6-1). The second is a modified grid scheduler that exploits heterogeneity among executing CTAs to facilitate fine-grained overlapping execution on the SM ([§4.2\)](#page-7-0). We conclude this section by discussing the logical execution model that Kitsune's primitives now provide.

## <span id="page-6-1"></span>4.1 Producer consumer communication

We use GPU atomics to design a synchronized, ring buffer queue for passing data between CTAs. Queues are pinned in the L2 cache using CUDA API functions [\[33\]](#page-18-7) (Fig [4\(](#page-7-1)a)). Each entry contains metadata protected by atomic accesses. Figure [4](#page-7-1) shows (a) a diagram of our queue design (with two entries for double-buffering), (b) a timeline of producer-consumer operations, (c) stylized code implementing the queue, and (d) application-level usage. Two CTAs communicating

<span id="page-7-1"></span>![](_page_7_Figure_0.jpeg)

Fig. 4. Queue design. Note: release routines are not shown for space reasons. They involve simple atomicAdd calls to update synchronization metadata and a CTA barrier with \_\_syncthreads().

(Fig 4(b)) "acquire" and "release" entries, achieving ordering via sequence numbers. The producer and consumer acquire entries (wr\_acquire and rd\_acquire in Fig 4(c)) by spinning on metadata variables until an entry is freed for use. acquire and release are exposed as an API which handle sequencing automatically. Typically, only one CTA is spinning on a variable at a given time – meaning our queue design results in very low contention.

Our queue is implemented as a library with two API functions: acquire and release. This allows for easy software integration, introducing minimal overhead exploiting the modern GPU's sophisticated warp-scheduler. Queue code is wrapped with if threadid==0, ensuring only one thread in a CTA does any of the queue management. To avoid data-races, "release" operations require a CTA-level barrier. Figure 4(d) shows how it can be used intuitively by a CUDA programmer or inserted by a source-to-source compiler into existing CUDA kernels. Synchronization variables are all padded to the size of a cache line to avoid false-sharing.

Queue Performance. Using a microbenchmark, we measure the A100 can sustain 100 M atomics / sec / CTA when under no contention. Based on additional measurements, we find this lends to an upper bound of 385-1541 GB/s per queue, far exceeding L2 and HBM bandwidth ( $\approx$ 61 GB/s per SM). We evaluate our queue's performance by measuring SM-SM bandwidth with varying payload sizes for 54 queues (108 CTAs for the 108 SMs of the A100 GPU). Figure 5 shows queue management overhead by measuring the performance of data transfers with and without synchronizing atomics enabled. We find with 128-256 KB payloads, aggregate bandwidth reaches 2 TB/s (37 GB/s/queue). Beyond 256 KB, performance drops due to queue sizes reaching the L2 capacity, causing accesses to spill out to HBM (Limiting us to 1.5 TB/s for A100). Synchronization overhead is large for small queue sizes:  $12\times$  reduction in bandwidth for 1KB payloads. With larger payloads this reduces: synchronization overhead is less than 63% for  $\geq$ 64KB payloads. Overall, we find GPU global atomics performance is more than enough for our use case. We also find our atomics-based L2 resident queue provides substantial inter-CTA communication bandwidth even in the presence of contention for payloads ranging between 64-256KB.

## <span id="page-7-0"></span>4.2 Scheduling heterogeneous CTAs

In order to capitalize on idle resources of the SM – for example, make full use of both the Tensor Core and SIMT Core simultaneously – we propose a modest change to the CUDA API and GPU Grid Scheduler to specify spatial pipelines (shortly defined) of kernels and maximize GPU resource usage. This is important for enabling and managing true co-execution of kernels which is not supported on current GPUs (§2).

<span id="page-8-0"></span>![](_page_8_Figure_0.jpeg)

Fig. 5. Performance of GPU atomics.

<span id="page-8-1"></span>![](_page_8_Figure_2.jpeg)

Fig. 6. Code snippet of the proposed cudaPipeline API.

CUDA API Exposure. We introduce an abstraction we call a CUDA "spatial pipeline", with a similar API to CUDA Graphs but different semantics. Like a CUDA graph, a spatial pipeline specifies a collection of kernels to execute with the key difference being it implies all kernels in the collection require being co-resident on the GPU. The calling code is responsible for limiting the number of CTAs launched per kernel to ensure co-residency is possible ([§5.3\)](#page-10-0). Figure [6](#page-8-1) shows a snippet of host code which specifies and configures the launch of a spatial pipeline. Data dependence information is specified similar to CUDA graphs, and kernels are configured with new metadata that specifies the primary type of dynamic resource they require, either SIMT or TENSOR.

Hardware Implementation. To complement our CUDA spatial pipeline abstraction, we propose a modest change to the grid scheduler that allow it to leverage the type information now passed via the kernel call header. On current GPUs, the grid scheduler hardware stores occupancy info for how much of each SM's resources are consumed, which is used to greedily find the first available SM for CTA dispatch using a hardware arbiter (i.e., round-robin) [\[48\]](#page-19-10). However, this greedy policy doesn't work for Kitsune as it doesn't guarantee overlap; We need to ensure that CTAs of different types are effectively paired for execution on the SM. We augment the round-robin prioritization hardware with two arbiters, one for each type. The two arbiters enables the scheduler to effectively pair different types together, by separately considering dispatch to the same SM. When a new kernel arrives, the arbiter is selected based on the type. The CTA scheduling then proceeds as usual, checking the occupancy of the current SM under consideration for dispatch.

### 5 Kitsune Compiler Design

In this section we develop the Kitsune compiler, which enables DL applications to transparently leverage dataflow. We implement Kitsune as a PyTorch [\[24\]](#page-18-19) compiler backend. We use PyTorch 2.0's Dynamo interface for extracting application graphs including both the forward pass and back-propagation for training. Our compiler backend consumes these graphs and constructs spatial pipelines for execution.

<span id="page-9-1"></span>![](_page_9_Figure_0.jpeg)

Fig. 7. Depiction of the Kitsune compiler flow. Our enabling primitives from the previous section are highlighted in purple.

To realize this compiler, several challenges must be addressed. First, subgraphs must be selected from the original application graph for fusion (§5.1). Second, a pipeline must be designed for the subgraph with stages corresponding to operators (§5.2). Third, the stages must be assigned GPU resources to achieve optimal performance addressing load-balancing (§5.3). Figure 7 depicts how each of these pieces are applied to a PyTorch application. The compiler leverages our software queue structure and modified GPU grid scheduler to enable inter-stage communication and intelligent CTA placement to exploit fine-grain SM resource sharing. Figure 8 depicts how our compiler lowers MeshGraphNets, and will serve as a running example throughout this section.

#### <span id="page-9-0"></span>5.1 Subgraph Selection

We first need to select subgraphs for dataflow execution which involves marking groups of DL operators in the computation graph for co-execution in a pipeline. We denote these groups of operators / nodes that form a pipeline as an sf-node. The output of this phase is labeled graph with sf-nodes identified. At the graph level, a spatially-fused group (sf-node) of operations must be "contiguous" as defined in [47] – that is, there must be no edge which exits the subgraph with a down stream edge that reenters it. Subgraph selection influences pipeline design, allocation, assignment and hence performance, potentially requiring an iterative solution. As a practical solution, we implement a single-pass design that use two rules to exclude a node from a subgraph: nodes that are bulk-sync friendly and nodes that index / gather across all data (gather nodes for embedding for example). With such node exclusions defined, subgraph selection converts to pattern-matching.

Our design and implementation of subgraph selection is heuristic based and uses manual pattern matching. By examining applications properties we identified node patterns that are candidates for subgraph exposing the vulnerabilities of bulk-synchronous execution and vertical fusion. It is essentially a set of regular expressions that express patterns including those seen in Figure 2. In particular, our implementation operates at the topological order which linearizes the graph into a list in PyTorch Dynamo (which is deterministic). In practice, additional regular expressions to express different orderings for the topological order are easy. A more formal automata that captures all possible linearizations of a subgraph is beyond the scope of this work.

We leverage PyTorch's Dynamo to extract whole operator graphs of the forward and backward passes for an application. We then created a library of patterns that expresses patterns that are candidates for subgraphs. We implement a pattern-matching algorithm for then selecting subgraphs from the original application graph for dataflow execution. This approach searches for user-specified chains of operators in a topological order. Adding new patterns is a trivial task of adding to our pattern library. Figure 8 (a) shows how we selected a subgraph for MeshGraphNets.

#### Algorithm 1: Algorithm for pipeline design

```
1 for  in ℎ do
2 if IsReduction() then
3  ,   ← SplitReduction()
4 ℎ.replace( [], [ ,  ] )
5  ←  
6 end
7 if IsIntermediate() then
8  ← CreateQueue()
9 for  in Dependents() do
10 .producer ← 
11 end
12 .dependents ← []
13 end
14 end
```

## <span id="page-10-1"></span>5.2 Pipeline Design

The pipeline design problem comprises of inserting queues between nodes of an sf-node, and if the work done between two nodes is trivially fusable, fuse them using epilogue fusion (or vertical fusion). The output is a transformed graph which includes one or more queue nodes added, which can then be lowered to CUDA code during code-generation.

Conceptually what this means is taking the original set of operations in the graph and either combining or splitting them to map to pipeline stages that are realized by pipeline-enabled CUDA kernels. For simple patterns with 1-1 producer-consumer relationships, the decision is trivial - and involves insertion of queue nodes between nodes of an sf-node. For more complex patterns like attention and back-propagation, we implement a parallel reduce which uses our queues to form a reduction tree. Figure [8](#page-11-0) (b) and (e) show a pipelined graph starting from our MeshGraphNets subgraph and back-propagation of a single Linear layer, respectively.

In terms of implementation this involves three steps. The graph rewrite algorithm is shown in Algorithm [1.](#page-10-2) In terms of code-generation, the queue implementation is discussed in [§4.1.](#page-6-1) The third step is to take CUDA kernels and transform them to read/write from queues, instead of from global memory. This last step also includes the process of working on tiles, since a queue's payload needs to be limited. In all cases, the notion of tiling already exists or is trivially doable; for GEMMs the code is already written to work on tiles of inputs and outputs. Completely automating this step for arbitrary code is likely infeasible and involves all the challenges of aliasing analysis etc. For Kitsune, we performed this step manually - it took about 8 person-hours or less for each kernel, with the source-code lines changed ranging from 10 to 40. The limitation this adds to Kitsune is that it is not completely turn-key for new operators not previously seen by the compiler, requiring very modest library modifications of the underlying new DL operator. In practice, library developers like NVIDIA and AMD can incorporate such a flow trivially into their development process.

