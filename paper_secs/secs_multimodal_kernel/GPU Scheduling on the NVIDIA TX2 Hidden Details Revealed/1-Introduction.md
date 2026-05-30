# 1 Introduction

Graphics processing units (GPUs) are currently seen as a key enabler for accelerating computations fundamental to realizing autonomous-driving capabilities. The massive parallelism afforded by GPUs makes them especially well-suited for accelerating computations such as motion planning, or for handling multiple input streams from sensors such as cameras, laser range finders (LIDAR), and radar. For this reason, several companies (Tesla, Audi, Volvo, *etc.* [20]) have announced their intentions of using GPU-equipped computing platforms in realizing autonomous features.

As mass-market vehicles evolve towards providing full autonomy, it will become ever more critical that GPU-using workloads are amenable to strict certification. However, certification requires a comprehensive model of the GPU scheduler. Unfortunately, currently available GPUs create numerous challenges in this regard, because many aspects underlying their design either are not documented or are described at such a high level that crucial technical details are not revealed. Without such details, it is impossible to certify a safety-critical design.

Real-time constraints and GPU scheduling. In this paper, we consider one key aspect of certification—the validation of real-time constraints—in the context of embedded

multicore+GPU platforms suitable for autonomous-driving use cases. These platforms usually follow a system-on-chip (SoC) design, where a GPU and one or more CPUs are hosted on the same chip. GPUs in SoC designs tend to be less capable than those in desktop systems. As a result, it is crucial to fully utilize the GPU processing capacity that *is* available. However, avoiding wasted GPU processing cycles can be difficult without detailed knowledge of how a GPU actually schedules its work. Unfortunately, acquiring this knowledge can be quite difficult, given the opaque and enigmatic nature of publicly available information regarding GPUs.

Focus of this paper. In this paper, we present an in-depth study of GPU scheduling on an exemplar of current GPUs targeted towards autonomous systems. This study was conducted using only black-box experimentation and publicly available documentation. The exemplar chosen for this study is the recently released NVIDIA TX2. The TX2 is part of the Jetson family of embedded computers, which is explicitly marketed for "autonomous everything" [17]. Moreover, it shares a common GPU architecture with the higher-end Drive PX2, which is currently available only to automotive companies and suppliers. The TX2 has two important attributes for embedded use cases: it provides significant computing capacity, and meets reasonable limits on monetary cost as well as size, weight, and power (SWaP).

As seen in Fig. 1, the TX2 is a single-board computer containing multiple CPUs and an integrated GPU. As explained in more detail later, *integrated* GPUs share DRAM memory with the host CPU platform, in contrast to *discrete* GPUs, which have private DRAM memory. Integrated GPUs are the de facto choice in embedded applications where SWaP is a concern. The TX2's cost, approximately 600 USD per board, is likely affordable even if multiple copies are needed.

Contributions. This paper is part of an effort to develop a model for describing GPU workloads and how they are scheduled. Our eventual goal is to develop a model for which real-time schedulability-analysis results can be derived, as has been done for various CPU workload and scheduling models that have been studied for years. We specifically target the TX2's GPU scheduler. The TX2 is a very complicated device, so discerning exactly how it functions is not easy.

Prior work has documented that NVIDIA GPU scheduling differs based on whether work is submitted to a GPU from a CPU executing (i) operating-system (OS) threads that share an address space or (ii) OS processes that have different address spaces [28]. However, to the best of our knowledge, the exact manner in which GPU scheduling is done in either case has never been publicly disclosed.

<sup>∗</sup>Work supported by NSF grants CPS 1239135, CNS 1409175, CPS 1446631, and CNS 1563845, AFOSR grant FA9550-14-1-0161, ARO grant W911NF-14-1-0499, and funding from General Motors.

For Case (i), we present a set of rules, based on experimental evidence involving benchmarking programs, that fully define how the TX2's GPU schedules work submitted to it, assuming that certain optional GPU features (see below) are not employed that introduce additional complexities. These rules take into account the ordering within and among streams of requests for GPU operations, the means by which various internal queues of requests are handled, the ability of the GPU to co-schedule operations so they execute concurrently, the selection mechanism used to determine the order in which requests are handled, and the resource limits that constrain the GPU's ability to handle new requests. Our rules indicate that the TX2's GPU employs a variant of hierarchical FIFO scheduling that is amenable to real-time schedulability analysis, although scheduling is not fully work-conserving, as GPU computations are subject to various blocking delays.

The optional GPU features that we initially ignore include two features that cause certain request streams to be treated specially, which further complicates GPU scheduling. These features include the usage of a special stream called the *NULL stream* and the usage of *stream priorities*. The available documentation regarding both of these features often lacks details necessary for predicting specific runtime behavior. Through further experiments, we show how each of these features affects our derived scheduling rules.

GPU programs are most commonly developed assuming that different GPU computations are requested by OS processes with different address spaces, so Case (ii) above applies. We investigated GPU scheduling in this case as well and, unfortunately, found it to be less deterministic than in Case (i). In particular, concurrent GPU computations in this case are multiprogrammed on the GPU using time slicing in a way that can add significant overhead and execution-time variation. This result calls into question the common practice today of following a process-oriented approach in developing GPU applications, as reflected in open-source frameworks such as Torch¹ and Caffe.² We also investigated the usage of stream priorities under Case (ii) and found that they have no effect in this case.

The work presented herein is a first step in a broader project, the goal of which is to enable the analysis and certification of safety-critical real-time workloads that depend on both CPU and GPU resources. In this broader context, our experimental methods can actually be viewed as a second contribution, as they have been devised in a way that should enable relatively straightforward extension to other NVIDIA GPU architectures that may be of interest in the future.

**Organization.** In the rest of this paper, we provide relevant background on GPU fundamentals (Sec. 2), present the basic TX2 GPU scheduling rules we have derived for Case (i) above, along with experimental evidence to support

![](_page_1_Figure_7.jpeg)

Figure 1: Jetson TX2 Architecture

them (Sec. 3), discuss additional complexities impacting these rules when certain important GPU features are enabled (Sec. 4), and conclude (Sec. 5). For ease of exposition, some experiments and details are deferred to appendices.

## 2 Background

In this section, we provide needed background on GPUs and the NVIDIA Jetson TX2. We also discuss prior related work.

#### 2.1 The NVIDIA Jetson TX2

As shown in Fig. 1, the TX2 employs an SoC design that incorporates a quad-core 2.0-GHz 64-bit ARMv8 A57 processor, a dual-core 2.0-GHz superscalar ARMv8 Denver processor, and an integrated Pascal GPU. There are two 2-MB L2 caches, one shared by the four A57 cores and one shared by the two Denver cores. The GPU has two *streaming multiprocessors* (SMs), each providing 128 1.3-GHz cores that share a 512-KB L2 cache. The six CPU cores and integrated GPU share 8 GB of 1.866-GHz DRAM memory.

An integrated GPU, like that on the TX2, tightly shares DRAM memory with CPU cores, typically draws between 5 and 15 watts, and requires minimal cooling and additional space. As noted earlier, the alternative to an integrated GPU is a discrete GPU. Discrete GPUs are packaged on adapter cards that plug into a host computer bus, have their own local DRAM memory that is completely independent from that used by CPU cores, typically draw between 150 and 250 watts, need active cooling, and occupy substantial space. For these reasons, integrated GPUs are found in most GPU-enabled computing platforms for embedded systems.

## 2.2 CUDA Programming Fundamentals

The following is a high-level description of CUDA, the API for GPU programming provided by NVIDIA.

A GPU is a co-processor that performs operations requested by CPU code. CUDA programs use a set of C or C++ library routines to request GPU operations implemented by a combination of hardware and device-driver software. The typical structure of a CUDA program is as follows: (i) al-

<sup>1</sup>http://torch.ch

<sup>&</sup>lt;sup>2</sup>http://caffe.berkeleyvision.org

#### Listing 1 Vector Addition Pseudocode.

```
1: kernel VECADD(A ptr to int, B: ptr to int, C: ptr to int)
   . Calculate index based on built-in thread and block information
2: i := blockDim.x * blockIdx.x + threadIdx.x
3: C[i] := A[i] + B[i]
4: end kernel
5: procedure MAIN
   . (i) Allocate GPU memory for arrays A, B, and C
6: cudaMalloc(d A)
7: . . .
   . (ii) Copy data from CPU to GPU memory for arrays A and B
8: cudaMemcpy(d A, h A)
9: . . .
   . (iii) Launch the kernel
10: vecAdd<<<numBlocks, threadsPerBlock>>>(d A, d B, d C)
   . (iv) Copy results from GPU to CPU array C
11: cudaMemcpy(h C, d C)
   . (v) Free GPU memory for arrays A, B, and C
12: cudaFree(d A)
13: . . .
14: end procedure
```

locate GPU-local (device) memory for data; (ii) use the GPU to copy data from host memory to GPU device memory; (iii) launch a program, called a *kernel*, 3 to run on the GPU cores to compute some function on the data; (iv) use the GPU to copy output data from device memory back to host memory; (v) free the device memory. An example vector-addition CPU procedure and associated kernel are given in Listing 1.

In the usual programming model for writing CUDA code, input data is partitioned among hardware threads on the GPU. Parallelism is achieved by specifying the number of threads that form a *thread block* and the total number of such blocks to be allocated. For example, in a vector-addition kernel that adds two 4,096-element arrays, where each thread operates on a single element as in Listing 1, the programmer could specify eight blocks of 512 threads. These values are given in the kernel-launch command, shown in Line 10 of Listing 1. Thread-related system variables, such as the number of threads per block and the total number of blocks, are used by the kernel code to identify the specific data element(s) handled by each thread. In the vector-addition example, each thread calculates the index i of the element it operates on in Line 2 using these CUDA-provided built-in variables.

To avoid any confusion, we henceforth use the term *thread* to mean a hardware thread on the GPU. We use the term *process* when referring to OS processes executing on CPUs that have separate address spaces and *task* to refer to OS threads executing on CPUs that share an address space. (The term "thread" is ordinarily used in referring to the latter.)

As seen in Listing 1, dispersing code to execute across one or more blocks, each with many threads, enables the significant parallelism afforded by GPUs to be exploited. On the TX2, there are 256 GPU cores, 128 per SM. Each SM is capable of running four units of 32 threads each (called a *warp*) at any given instant. The SM's four *warp schedulers*

![](_page_2_Figure_7.jpeg)

Figure 2: Diagram illustrating the relation between CUDA programs, kernels, and thread blocks.

(contained in hardware) take advantage of stalls, such as waiting to access memory, to immediately begin running a different warp on the set of 32 cores assigned to it. In this way, the SM's warp schedulers can hide memory latency.

For a simpler abstraction, programmers can think of a GPU as consisting of one or more *copy engines* (*CEs*) (the TX2 has only one) that copy data between host memory and device memory, and an *execution engine* (*EE*) that has one or more SMs (the TX2 has two) consisting of many cores that execute GPU kernels. An EE can execute multiple kernels simultaneously, but a CE can perform only one copy operation at a time. EEs and CEs can operate concurrently.

#### 2.3 Ordering and Execution of GPU Operations

Fig. 2 depicts the execution of the entities discussed so far. In this figure, there are two CUDA programs executed by processes running on different CPUs. As discussed above, a CUDA program consists of CPU code that invokes GPU code contained in CUDA kernels, each of which is executed by running a programmer-specified number of thread blocks on the GPU. Multiple thread blocks from the same kernel can execute concurrently if sufficient GPU resources exist, although this is not depicted in Fig. 2. The figure distinguishes between *block time*, which is the time taken to execute a single thread block, *kernel time*, which is the time taken to execute a single kernel (from when its first block commences execution until its last block completes), and *total time*, which is the time taken to execute an entire CUDA program (including CPU portions). We refer to copy operations and kernels collectively as *GPU operations*. For simplicity, copy operations are not depicted in Fig. 2.

In CUDA, GPU operations can be ordered by associating them with a *stream*. Operations in a given stream are executed in FIFO order, but the order of execution across different streams is determined by the GPU scheduling policy. Kernels from different streams may execute concurrently or even out of launch-time order. Kernel executions and copy operations from different streams can also operate concurrently depending on the GPU hardware. By default, GPU operations from all CPU programs are inserted into the single *default stream*. Unless specified otherwise, this is the

<sup>3</sup>Unfortunate terminology, not to be confused with an OS kernel.

*NULL stream* [19]. Programmers can create and use additional streams to allow for concurrent operations.

Kernel launches are asynchronous with respect to the CPU program: when the CUDA kernel-launch call returns, the CPU can perform additional computation including issuing more GPU operations. By default, copy operations are synchronous with respect to a CPU program: they do not return until the copy is complete. Asynchronous copy operations are also available so the CPU program can continue to perform computations, kernel launches, or copies. Copy operations, whether synchronous or asynchronous, will not start until all prior kernel launches from the same stream have finished. All asynchronous GPU operations require the CPU program to explicitly wait for all previously issued GPU operations to complete if CPU computations need to be synchronized with the GPU. This functionality is provided by CUDA APIs for synchronization or event management.

To our knowledge, complete details of kernel attributes and policies used by NVIDIA to schedule GPU operations are not available. The official CUDA documentation only states that operations within a stream are processed in FIFO order, and that kernels from different streams *may* run concurrently.<sup>4</sup> An NVIDIA developer presentation from 2011 [24] gives slightly more information: operations from multiple streams are placed into a single internal queue based on their issue order. However, as this talk covered an older GPU architecture, some of the details have changed; in newer NVIDIA GPUs, there are multiple internal queues [16].

In summary, we provide these definitions of a few terms used throughout the remainder of the paper:

- *CUDA kernel*: A section of code that runs on the GPU. A kernel is made of multiple thread blocks and is scheduled at the block level, *i.e.*, *blocks are schedulable entities*. Blocks may be executed in arbitrary order on the GPU.
- *Thread block* (*block*): A collection of GPU threads that may execute concurrently and run the same instructions but operate on different portions of data. The number of threads in a block and the number of blocks associated with a CUDA kernel are specified at launch time.
- *Streaming Multiprocessor* (*SM*): The individual cores in a CUDA-capable GPU are partitioned onto SMs. Threads within a single block will never simultaneously execute on different SMs.
- *CUDA Stream* (*stream*): A FIFO queue of GPU operations (copy operations and kernels). A single CPU process or task can attempt to issue concurrent GPU operations by distributing them among multiple streams.

### 2.4 Related Work

Because of the black-box nature of GPUs, in much prior work on real-time GPU management, a GPU-using program is required to lock an entire GPU, or individual EEs or CEs, while performing GPU operations [8, 10, 11, 25, 26, 27, 30]. The viewpoint taken here is that *concurrently executing kernels might adversely interfere with each other and thus should be disallowed*. However, disallowing concurrency may lead to wasted GPU processing cycles, which may be untenable when using less-capable integrated GPUs.

Motivated by this observation, our group investigated the effects of concurrent kernel execution on the NVIDIA TK1 [21] and TX1 [22, 23], two less-capable predecessors of the TX2. In two of these efforts [21, 23], we only considered GPU usage by CPU processes with different address spaces and did not attempt to discern how GPU scheduling is done. In the other prior effort [22], which is actually a precursor to this paper, we did consider GPU scheduling. However, this effort was directed at the less-capable TX1 and did not consider many of the complications (*e.g.*, interactions between user-specified streams and the NULL stream, kernel delays when available GPU resources are not sufficient, the effects of copy operations, *etc.*) that must be considered to completely characterize how GPU scheduling is done. All of these complications are considered herein.

Work has also been directed at splitting GPU tasks into smaller sub-tasks to approximate preemptive execution or improve utilization [2, 10, 12, 32]. A framework in this category called Kernelet [31] is particularly relevant to us as concurrent kernel execution is considered in order to improve utilization. However, the developers of Kernelet do not provide an in-depth investigation of actual GPU scheduling behaviors. Other work has been directed at timing analysis for GPU workloads [3, 4, 5, 6, 7], techniques for remedying performance bottlenecks [9], and techniques for managing or evaluating GPU hardware resources, including the cache [14, 15, 29] and direct I/O communication [1].

