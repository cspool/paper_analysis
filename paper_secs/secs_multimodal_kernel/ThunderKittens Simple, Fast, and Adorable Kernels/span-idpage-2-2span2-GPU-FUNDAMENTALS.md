# <span id="page-2-2"></span>2 GPU FUNDAMENTALS

GPU tasks are divided into small programs called *kernels*. A kernel typically loads data from high bandwidth memory (HBM), performs work on it, and writes the outputs back to HBM before concluding. Before we explain THUNDERKIT-TENS's abstractions, we provide background on GPU parallelism at the *warp*, *block* and *grid* levels. We follow NVIDIA's terminology and focus on the H100 SXM GPU, though the principles apply across GPU vendors and generations.

![](_page_2_Figure_9.jpeg)

<span id="page-2-1"></span>Figure 3: The software (and physical) GPU hierarchy.

### 2.1 GPU HIERARCHY

The GPU software hierarchy closely follows its physical hardware hierarchy (Figure [3\)](#page-2-1). Here, we illustrate several of its most important components and aspects.

- 1. Warps consist of groups of 32 consecutive *threads* that operate on data in small but fast *register* memory. These instructions run on physical *execution units*, and individual threads can frequently occupy multiple specialized execution pipelines (below) at once through instructionlevel parallelism, and different warps can further occupy available execution hardware:
  - (a) Load and store units, to bring data into and out of registers. Advanced GPUs have also introduced dedicated hardware acceleration (e.g. H100 Tensor Memory Accelerator) for asynchronous bulk data movement between HBM and shared memory.
  - (b) General purpose compute pipelines, such as ALU for max, min, FMA for multiplies and adds, and XU for complex operations like exp. Throughput differs across the pipelines.
  - (c) Accelerated matrix multiply hardware (tensor cores), which have most of the GPU compute.

Threads can temporarily stall for a variety of reasons, including (but not limited to) fixed instruction latencies, memory latencies, barriers, pipeline throttles, or instruction cache misses.

2. Thread blocks are groups of warps which together execute a kernel on a physical core, called a *streaming multiprocessor* (SM). Although each SM has just four physical execution units, up to 64 software warps can simultaneously run on it (called "occupancy"). These collocated warps often contend on hardware resources: registers, shared memory, issue slots, and compute pipelines, but together they can help keep many work streams running at the same time within each execution unit. Warps *synchronize* at barriers, during which they cannot issue new work.

Importantly, warps within the same block can quickly communicate through special *shared memory* (SMEM, 227 KB, 33 TB/s). To improve bandwidth, SMEM is grouped into 32 physical "banks" of memory, which can serve memory simultaneously. However, if different threads try to access the same bank at the same time (called a *bank conflict*), their accesses must be serialized, which both increases access latencies and reduces available bandwidth. Hopper has limit of 255 registers per thread and attempts to request more, results in *spills* to the L1 cache. SMEM can be reallocated as an *L1 cache* for fast access to frequently used memory like spilled registers.

3. Grids of multiple thread blocks are launched to run the kernel. The H100 SXM GPU has 132 physical SM's which can run thread blocks at the same time. Although SM's are capable of collocating multiple thread blocks, most AI kernels can achieve high performance by simply collocating more warps within a single thread block (increasing the occupancy).

Thread blocks on the same GPU share common memory resources: large but slow highbandwidth memory (80 GB, 3 TB/s), which has both the greatest latency and least bandwidth of all GPU memory, and a smaller but faster hardware-managed L2 cache (50 MB, 12 TB/s).

There are overheads to scheduling blocks. First, the block launch incurs *setup* costs and although this cost must be paid at least once at the initial kernel launch, kernels that continuously launch many large blocks can incur further costs. Second, there are *tail effect* costs if the grid is sized poorly. If a kernel of 133 blocks is executed on an H100 with 132 physical SMs, the kernel would require two waves to execute, the first with full efficiency, and the second with < 1% efficiency. More advanced schedules using independent CUDA streams can ameliorate these tail effects, such as recent work on asynchronous tensor-parallel schedules [Chang et al.](#page-11-4) [\(2024\)](#page-11-4).

### 2.2 COST MODEL

Summarizing the above components, we show a simplified cost model for GPU parallelism below. This cost model is inspired by the roofline model [Williams et al.](#page-13-0) [\(2008\)](#page-13-0). The overall kernel execution time COverall is the sum of the following costs where memory costs are a combination of the latency and bandwidth, and compute costs are a combination of latency and throughput.

$$\mathbf{C}_{Overall} = \max\left(\underbrace{\mathbf{C}_{HBM}, \mathbf{C}_{L2}, \mathbf{C}_{L1}, \mathbf{C}_{Shared}}_{\text{Memory}}, \underbrace{\mathbf{C}_{Tensor}, \mathbf{C}_{ALU}, \mathbf{C}_{FMA}, \mathbf{C}_{XU}}_{\text{Compute}}\right) + \underbrace{\mathbf{C}_{Setup} + \mathbf{C}_{Sync}}_{\text{Overhead}}$$

This model represents the *ideal case* of perfect overlapping between memory, compute, and tensor core costs. A kernel's actual cost will lie between the max and the sum of the terms, depending on the workload properties (*i.e.*, some operations are inherently sequential), and the implementation efficiency. We aim to (1) reduce these individual costs, and (2) improve their collective overlapping. In Section [3,](#page-4-0) where we detail TK, we connect our primitives and optimizations back to these costs.

### 2.3 GPU PROGRAMMING FRAMEWORKS

We are inspired by a number of related efforts to simplify the development of AI kernels, such as NVIDIA CUTLASS/CuTe [\(NVIDIA, 2017\)](#page-12-1) and Triton [\(Tillet et al., 2019\)](#page-12-2).

CUTLASS's myriad of nested CUDA templates helps power highly optimized AI kernels [\(Shah](#page-12-0) [et al., 2024;](#page-12-0) [Bikshandi & Shah, 2023a;](#page-10-0)[b\)](#page-10-1) and fundamentally, the same kernels are expressible in TK and CUTLASS, since both are *embedded* libraries, giving users the full power of C++. We take a complementary approach by being rather opinionated about the abstractions. We ask: *(1) How far can we get with a small set of templates? and (2) Does concision sacrifice performance?* An appealing outcome is improved accessibility to AI researchers, since it can be challenging to fully leverage the capabilities of CUTLASS [\(Bikshandi & Shah, 2023b\)](#page-10-1). We find that even industrially popular kernels written in CUTLASS, like FlashAttention-3, struggle from preventable issues like bank conflicts. We seek abstractions that manage such issues for users. Most recent AI architectures use high level compilers instead [\(Dao & Gu, 2024;](#page-11-5) [Yang & Zhang, 2024;](#page-13-1) [Fu et al., 2023c\)](#page-11-6).

Triton, PyTorch [\(Paszke et al., 2019\)](#page-12-3), TVM [\(Chen et al., 2018\)](#page-11-7), TensorFlow XLA [\(Abadi et al.,](#page-10-2) [2016\)](#page-10-2), and others approach the problem from a compiler perspective. The frameworks are not C++ embedded, so it can be challenging to use unsupported specialized hardware instructions. It can also be difficult to manage asynchronous execution and register usage in high level frameworks. We explore avenues that retain the simple, PyTorch-like feel *while* enabling maximum performance in the next section. An extended discussion of related work is in Appendix [A.](#page-14-1)

