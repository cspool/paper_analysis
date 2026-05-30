# 1 Introduction

Deep Neural Networks (DNNs) are widely used and often deployed on GPUs for inference [\[40,](#page-14-2) [43,](#page-14-3) [57\]](#page-15-1). On the one hand, the high price and power consumption of GPUs necessitate maximizing hardware utilization and reducing cost. On the other hand, end users expect inference service with fast response, low latency, and reliability. This calls for a wellbuilt model serving system for managing GPU resources and handling inference requests.

Existing machine learning frameworks [\[27,](#page-14-4) [45\]](#page-15-2) and model serving systems [\[44,](#page-14-5) [51\]](#page-15-3) are designed to improve inference performance and system usability. However, they are suboptimal in two main aspects:

- (1) They exhibit poor GPU hardware utilization during DNN inference. The fundamental issue is that they treat operators as first-class citizens. Operators are perceived as basic computation units and organized into compute kernels. Therefore, the computations of a kernel are performed either all at once or not, disabling the parallelism of partially dependent operators while forcing the parallelism of a single operator's computations. Additionally, the monotonous instruction pattern of kernels prevents the simultaneous utilization of all GPU units, e.g., floating-point units and global memory bandwidth. As shown in Fig. [1,](#page-0-0) this type of frameworks executes operators sequentially with low GPU utilization. These inefficiencies are fully discussed in § [2.3.](#page-2-0)
- (2) They employ rudimentary scheduling mechanisms for inference jobs, which prove inadequate and impractical for

handling concurrent inference of heterogeneous jobs. For example, they are unable to effectively schedule inference jobs with fairness and real-time requirements.

To address the above limitations, we adopt a tile-based cooperative compilation and scheduling approach. At compile time, we partition large operators into small tiles and compile the tiles into micro kernels of multiple versions. At inference time, we schedule the inference tasks and the generated kernels with the consideration of hardware utilization and task priority. This strategy favors (1) large scheduling space from fine-grained computation partition and multi-version kernel generation, and (2) smart scheduling strategy from holistic task/kernel scheduling. Thereby, we build Infera, a model serving system that provides high-performance endto-end DNN inference service for users. This goal requires addressing the two key challenges described below.

How to compile models for large scheduling space? The model compilation involves two key steps: partitioning operators into tiles and generating multi-version kernels for those tiles. The first step, operator partitioning, is relatively straightforward, as it determines scheduling granularity and can be effectively handled through computation graph analysis. The second step, however, remains a long-standing challenge, akin to the problem of finding high-performance kernels, for two main reasons.

First, the kernel configuration space is vast, discrete, and irregular, making it difficult to explore. Moreover, different hardware backends impose distinct optimization objectives. Existing approaches often rely on extensive searches [32, 59] or heavy computations [28, 58] on specialized GPUs, which are time- and resource-intensive and yield hardware-specific code with limited portability [55].

Second, the optimization direction for kernels is inherently ambiguous. Achieving high GPU utilization requires tight collaboration between compilers and schedulers: the inference-time scheduler relies on compile-time kernel metadata (e.g., resource usage, launch configuration) to make optimal scheduling decisions [49], while the kernel compiler must account for runtime conditions (e.g., kernel concurrency, GPU load) to generate scheduler-friendly kernels [42]. Yet these runtime conditions are highly dynamic and unpredictable, significantly complicating kernel optimization.

How to schedule tasks/kernels for high GPU utilization while satisfying user requirements? Users submit inference jobs with diverse priorities at arbitrary times. Merely classifying jobs by priority [37, 49] is insufficient for a practical system. Achieving effective task scheduling demands both comprehensive handling of user requirements and substantial algorithmic and engineering effort.

For kernel scheduling, we must first define what constitutes high GPU utilization and the corresponding kernel-level requirements, then implement them efficiently for real-

<span id="page-1-0"></span>![](_page_1_Figure_9.jpeg)

Figure 2: Kernel scheduling illustration of various GPU colocation schemes. Here, 3 kernels each with 3 thread blocks are issued to a 2-core GPU at the start time.  $k_i.b_j$  denotes the j-th thread block of the i-th kernel,  $i \oplus j$  stands for kernel fusion, and  $t_{i,j}$  represents the j-th tile of the i-th kernel.

time operation. However, arbitrary kernel scheduling is not natively supported by GPUs, which enforces its own internal scheduling logic [29]. For instance, the CUDA runtime executes kernels sequentially [9], limiting true parallelism to brief overlaps at kernel tails (i.e., the last wave), while most of the GPU remains monopolized by a single kernel.

Existing approaches to kernel colocation such as kernel scheduling [18, 44, 51], GPU partitioning [17, 29], and kernel fusion [34, 39]. They are ineffective or inefficient, as shown in Fig. 2. "operator-based kernel direct" only enables spatial sharing at kernel boundaries; "operator-based kernel affinity" cannot share cores spatially; "operator-based kernel fusion" supports spatial sharing but incurs high fusion overhead.

Infera addresses these challenges by a carefully co-designed model compiler (§ 4) and task scheduler (§ 5). The compiler uses a tile-based, zero-tuning approach that automatically generates high-performance kernel variants through static analysis alone, avoiding costly search or profiling. The scheduler includes (1) an assembly-level kernel fuser for fast, finegrained warp-level fusion, enabling flexible kernel scheduling and efficient GPU spatial sharing, and (2) a holistic task scheduler with diverse scheduling policies and rapid preemption to manage heterogeneous user jobs efficiently. Together, they enable Infera to automate compilation and deliver high-performance end-to-end inference.

This paper makes the following contributions.

- We examine the NVIDIA GPU compute pipeline to uncover the key to peak performance. (§ 2)
- We develop an automated end-to-end high-performance

model serving system through co-designing the compiler and scheduler. (§ [3\)](#page-4-1)

- We design a DL compiler capable of rapidly generating highly efficient kernels of various versions. (§ [4\)](#page-4-0)
- We design a DNN inference server with multi-policy scheduling and high GPU utilization. (§ [5\)](#page-6-0)

We evaluate Infera with real-world workloads on NVIDIA GPUs (§ [6\)](#page-10-0), and the results indicate that Infera offers speedups of 1.6× to 3.5× compared to existing frameworks.

