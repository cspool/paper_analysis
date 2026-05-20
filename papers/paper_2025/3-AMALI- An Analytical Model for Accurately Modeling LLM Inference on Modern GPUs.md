# **AMALI: An Analytical Model for Accurately Modeling LLM Inference on Modern GPUs** 

Shiheng Cao 

## Junshi Chen 

Junmin Wu 

University of Science and Technology of China Hefei, China caosh2022@mail.ustc.edu.cn 

University of Science and Technology of China Hefei, China Suzhou Institute for Advanced Research, University of Science and Technology of China Suzhou, China jmwu@ustc.edu.cn 

University of Science and Technology of China Hefei, China cjuns@ustc.edu.cn 

Zhibin Yu 

Hong An 

University of Science and Technology of China Hefei, China han@ustc.edu.cn 

Shenzhen Institutes of Advanced Technology(SIAT), Chinese Academy of Science(CAS) Shenzhen, China zb.yu@siat.ac.cn 

## **Abstract** 

compared to the state-of-the-art GCoM model. We further showcase that AMALI can be used to explore architecture design space by designing the tensor core capability of H100. The results show that AMALI accurately predicts the end-to-end performance improvements with the enhanced tensor core capability. 

Large language model (LLM) inference applications are surging in recent years, which largely relies on modern GPUs. On the other hand, GPU analytical model is a commonly used tool for architects to precisely identify bottlenecks quickly with deep insights. However, existing GPU analytical models fall short of accurately modeling LLM inference applications on modern GPUs, because of unsuitable tensor core modeling, ignoring constant cache as well as instruction cache modeling and abstracting away important details for LLM inference applications. 

## **CCS Concepts** 

- **Computing methodologies** → **Graphics processors** . 

## **Keywords** 

To address this problem, we propose a novel analytical model dubbed AMALI to accurately model LLM inference on modern GPUs with three innovations. First, we develop an instruction modifier and throughput based tensor core model by accurately capturing the math pipe throttle stalls to enhance the architecture modeling for modern GPUs. Second, we propose analytical models for constant cache and instruction cache by developing micro-benchmarks to measure CUDA kernel launching latencies. This significantly improves AMALI’s accuracy compared to real GPU hardware. Finally, we design a multi-warp model by leveraging warp instruction number distribution to reflect LLM inference application characteristics. 

graphics processing units, performance modeling, interval analysis 

## **ACM Reference Format:** 

Shiheng Cao, Junmin Wu, Junshi Chen, Hong An, and Zhibin Yu. 2025. AMALI: An Analytical Model for Accurately Modeling LLM Inference on Modern GPUs. In _Proceedings of the 52nd Annual International Symposium on Computer Architecture (ISCA ’25), June 21–25, 2025, Tokyo, Japan._ ACM, New York, NY, USA, 14 pages. https://doi.org/10.1145/3695053.3731064 

## **1 Introduction** 

Due to the unprecedented performance of large language models (LLMs), LLM inference has rapidly swarmed into a large number of applications such as OpenAI ChatGPT [40] and Github Copilot [16]. Moreover, it is projected that LLM inference applications would keep surging in the near future [7]. These applications largely rely on modern GPUs with special support such as tensor cores for LLMs to achieve high performance (e.g., higher tokens/s, shorter TTFT - time to first token, and TBT - time between tokens). 

We validate AMALI on an A100 GPU by using typical LLM inference applications. The results show that AMALI reduces the MAPE (mean absolute percentage error) from 127.56% to 23.59% 

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org. _ISCA ’25, Tokyo, Japan_ 

With the rapid evolving of LLM inference applications, it is of utter importance to have tools quickly identifying the performance bottleneck of LLM inference on modern GPUs with deep insights. Such tools shed light on GPU micro-architecture enhancement and performance optimization of LLM inference applications. In fact, in AI (Artificial Intelligence) era, the speed for performance evaluation is more important than accuracy in the early design stage of GPU 

© 2025 Copyright held by the owner/author(s). Publication rights licensed to ACM. ACM ISBN 979-8-4007-1261-6/25/06 https://doi.org/10.1145/3695053.3731064 

1495 

ISCA ’25, June 21–25, 2025, Tokyo, Japan 

Cao et al. 

architecture, because it evolves rapidly in the last decade, driven by the fast evolving machine learning (ML) workloads [52]. 

Performance evaluation tools for GPU architecture can be roughly classified into two categories: cycle-accurate simulators [9, 13, 17, 18, 28, 30, 43, 50, 52, 57] and analytical models [8, 21–25, 29, 47, 55, 56, 60], both are indispensable. Cycle-accurate GPU simulators are accurate but extremely slow. Moreover, these simulators can not provide easily understood insights because of their complexity. In contrast, GPU analytical models are orders of magnitude faster than cycle-accurate simulators but with lower accuracy. Furthermore, analytical models can provide easily understood as well as deep insights. For instance, besides predicting the total cycles taken by a GPU kernel, an analytical model can build a cycles-per-instruction (CPI) stack to help computer architects find the bottleneck of the kernel on various GPU architectures easily by showing the percentages of various stall events in its execution [24, 56]. As aforementioned, GPU architects need fast performance evaluation tools more in the AI era. We therefore focus on studying GPU analytical models and hope existing ones can successfully work for LLM inference. However, we find existing GPU analytical models [21–25, 29, 55, 56] fall short of modeling LLM inference performance on modern GPUs with enough accuracy, because of two reasons. First, these models inappropriately or even do not model the micro-architecture enhancements including tensor cores, immediate constant cache, and instruction caches of modern GPUs. Second, these models do not consider the characteristics of LLM inference which is significantly different from traditional ML workloads and other GPU applications. As a result, as applying the state-of-the-art (SOTA) analytical model, GCoM, on LLM inference, the error is significantly high (e.g., 127.6%) compared to real GPU hardware (, see Section 6), which is unacceptable. 

To address these issues, we propose AMALI, an analytical model, to model LLM inference on modern GPUs with enough accuracy. We carefully analyze modern GPU architectures, as well as LLM inference characteristics and come up with three innovations. First, by analyzing how tensor cores work with specific instructions such as HMMA, we propose an instruction modifier and throughput based tensor core model by precisely capturing the math pipe throttle stalls, facilitating accurately modeling the performance of the heavily used GEMM (general matrix multiplication) operations in LLM inference. 

Second, we model the immediate constant cache and instruction cache by designing micro-benchmarks to measure kernel launching latency. This launching latency is exactly the stalls caused by the immediate constant cache misses and instruction cache misses. As such, this innovation significantly improves the accuracy of our analytical model compared to real GPU hardware. 

Finally, we find that the warp distribution used by the SOTA GPU analytical model, GCoM [29], does not reflect the characteristics of LLM inference applications. We therefore propose to leverage warp instruction distribution to build a multi-warp model to model the LLM inference application on GPUs. 

In particular, the main contribution of this paper is as follows. 

- We develop a tensor core model to accurately capture the math pipe throttle stalls caused by tensor cores across various data types and tensor sizes. 

**==> picture [202 x 92] intentionally omitted <==**

**----- Start of picture text -----**<br>
SM SM ... SM warp<br>Warp Scheduler<br>Interconnect Network Register File warp<br>Constant cache<br>...<br>CUDA Tensor<br>L2 Data cache Cores Cores warp<br>Global memory L1 cache &<br>Constant memory Shared memory<br>sub-cores<br>**----- End of picture text -----**<br>


**Figure 1: An overview of Ampere GPU micro-architecture.** 

- We model the stalls caused by immediate constant cache misses and instruction cache misses, by developing microbenchmarks to measure kernel launch latency. 

- We model the instruction distribution of warps of LLM inference to enhance the kernel cycle prediction. 

- By putting it all together, we build a model named AMALI to predict the kernel cycles of a LLM inference. 

- We validate AMALI against NVIDIA A100 GPU by using several typical LLM inference applications. The experimental results show that AMALI achieves a MAPE of 23.59%, indicating a significant improvement over GCoM’s MAPE of 127.56% in total cycle prediction of a kernel. 

- We showcase that AMALI can be used to explore GPU architecture design space by designing the tensor core capability of H100. The results show that AMALI accurately predicts the end-to-end performance improvements (e.g., kernel cycles) with the enhanced tensor core capability. 

The rest of the paper is organized as follows. Section 2 describes the background of this paper. Section 3 depicts the baseline GPU analytical model and our motivation. Section 4 elaborates our AMALI analytical model. Section 5 presents the experimental setup. Section 6 provides the experimental results and analysis. Section 7 introduces the related work and Section 8 concludes the paper. 

## **2 Background 2.1 GPU Architectures** 

Without loosing generality, we employ NVIDIA GPUs (Graphics Processing Unit) to introduce GPU architecture. Figure 1 shows the Ampere GPU architecture [1, 11, 39]. As can be seen, a GPU consists of a number of streaming multi-processors (SM) connected by an on-chip interconnection network. To buffer data between SMs and memory, a L2 data cache is designed between the global as well as constant memory, and the interconnection network. 

Each SM consists of a L1 cache/shared memory and several subcores. The L1 cache/shared memory is shared among the sub-cores. Note that the L1 cache and shared memory in a SM share the same hardware which a part of it can be configured as L1 cache and the other part as shared memory. Each sub-core contains a warp scheduler, register files, a constant cache, a set of CUDA cores, and a set of tensor cores. The warp scheduler selects a warp, which contains a number (e.g., 32) of threads executing in a lock-step manner, to execute when the warp is ready. In each cycle, the warp scheduler issues an instruction from the selected warp. If the operand of the instruction is not ready, the warp scheduler suspends the warp and selects another warp to execute by employing a certain 

1496 

ISCA ’25, June 21–25, 2025, Tokyo, Japan 

AMALI: An Analytical Model for Accurately Modeling LLM Inference on Modern GPUs 

scheduling policy such as loosely round robbin(LRR) [31, 38] or greedy then oldest [45]. 

Tensor cores are customized compute units that can perform one matrix-multiply-and-accumulate on 4 × 4 matrices per clock cycle. This significantly accelerates GEMM computation like _𝐶_ = _𝐴_ × _𝐵_ + _𝐶_ . _𝐴_ and _𝐵_ are _𝑚_ × _𝑘_ and _𝑘_ × _𝑛_ matrices, respectively; _𝐶_ is the accumulator matrix. Tensor cores are therefore crucial components for LLM inference and other AI workloads. 

To program tensor cores, typical instructions are: 

_𝐻𝑀𝑀𝐴._ 16816 _.𝐹_ 32 _, 𝑅_ 0 _, 𝑅_ 108 _, 𝑅_ 140 _, 𝑅_ 0 (1) 

_𝐻𝑀𝑀𝐴._ 1688 _.𝐹_ 32 _, 𝑅_ 0 _, 𝑅_ 180 _, 𝑅_ 196 _, 𝑅_ 0 (2) 

The instruction name ’HMMA’ represents half-matrix multiply add, which indicates the input is half-precision. These instructions contain modifiers which locate after the dot symbols and influence instruction behavior. The first modifier, such as 16816 or 1688, denotes the tensor size of these instructions. For example, 16816 represents the input tensors _𝐴_ and _𝐵_ are 16 × 8 and 8 × 16 matrices, respectively. 1688 represents 16 × 8 ( _𝐴_ ) and 8 × 8 ( _𝐵_ ) input matrices. The second modifier such as _𝐹_ 32 shown in expressions (1) and (2) denotes the data type of the accumulator tensor. 

Each tensor core instruction shown in expressions (1) and (2) contains four registers. _𝑅_ 0 is the register used to store the accumulator/result matrix (C); the register _𝑅_ 108 or _𝑅_ 180 stores the input matrix _𝐴_ and _𝑅_ 140 or _𝑅_ 196 stores _𝐵_ . Note that these registers are shared by all the threads in a warp. In contrast, for CUDA cores, each thread in a warp can only access its own register, rather than the ones of other threads. In other words, the threads in a warp running on tensor cores access registers in a _per-warp_ scheme while those on CUDA cores access registers in a _per-thread_ scheme. The behavior of each thread with the per-warp scheme is non-deterministic whereas that of each thread with the per-thread scheme is deterministic. 

Moreover, NVIDIA GPUs have a small constant memory (e.g., 64KB) to hold constant variables like _warp id_ , _block id_ , and other data structures such as arrays. Constant memory is a part of global memory, and has a constant cache as shown in Fig.1. In a constant cache miss, it takes the memory read time (e.g., hundreds of cycles) to get data from the constant memory. In a constant cache hit, the data can be attained as fast as a register file access [38]. Note that not only the _ld_ instructions can explicitly access constant memory but also other instructions can _implicitly_ access it. For example, the instruction _IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28]_ accesses the constant memory since the memory address is with a special symbol ’c’, indicating the address is in constant memory [37]. 

## **2.2 CUDA Programming model** 

Compute Unified Device Architecture (CUDA) [15] is a programming model designed for NVIDIA GPUs and it allows programmers to write GPU functions using C style functions, called kernels. CUDA designs an execution model named SIMT (single instruction multiple threads) to execute kernels with a three-level hierarchy. The lowest level is thread which executes GPU instructions (e.g., SASS instructions). 32 threads are organized as a warp which executes in a lock-step manner. The upper level is thread block which contains a number of threads or warps. The highest level is called grid which consists of a number of thread blocks. This three-level 

hierarchy is convenient to manage a large number of threads or to program cubic graphics with three dimensions (e.g., _𝑥_ , _𝑦_ and _𝑧_ ). 

## **2.3 Large Language Model** 

Transformer [14, 34, 51] based large language models (LLM) have been used in a wide range of applications such as OpenAI ChatGPT [40] and Copilot [16]. The Transformer block consists of two critical components: multi-head attention and the feed-forward neural network. The core computations of both parts are the General Matrix Multiplication (GEMM) operations, which are executed on GPUs using Tensor Cores for optimized computational efficiency. 

In the multi-head attention mechanism, the input data is projected into multiple scaled dot-product attentions in parallel. This involves computing query, key (K), and value (V) matrices for each attention head through linear transformations (which are essentially matrix multiplications). The feed-forward block follows the attention mechanism and consists of two linear layers with a nonlinearity (such as ReLU) between them. These operations are highly suited for execution on Tensor Cores. 

LLM inference typically consists of two stages: prefill and decode. The prefill stage receives requests (also called prompts) consisting of tokens and processes them in parallel. The decode stage outputs responses in an auto-regressive manner. To accelerate the token generation, the computation of K and V matrices is cached and called KV cache. Longer prompts require larger KV cache. 

A popular software framework for LLM inference includes Pytorch [41, 46], CUDA [15] and other layers such as vLLM [54]. Typically, model implementation is written in Pytorch and the GEMM implementation is written in CUDA. Pytorch provides facilities to call CUDA APIs conveniently. A LLM inference may call thousands of CUDA kernels from Pytorch codes. 

## **2.4 Stall classification** 

Identifying stalls is extremely important to analyze the performance bottleneck of an application on a given GPU. GSI [2] classifies the stalls into seven categories and the SOTA GPU analytical model GCoM [29] employs this classification: 1) **idle stalls** caused by not enough threads/instructions to execute, 2) **control stalls (Ctrl)** caused by kernel code divergence (icache misses), 3) **synchronization stalls (Sync)** incurred by thread barriers, 4) **memory data stalls (MemData)** due to pending memory loads, 5) **memory structural stalls (MemStruct)** due to unavailable load/store ports, 6) **compute data stalls (ComData)** caused by the operands of an instruction not been produced by other instructions yet; 7) **compute structural stalls (ComStruct)** due to the unavailable required compute resources. This classification employs a view on general processors, which does not accurately reflect NVIDIA GPU-specific features such as constant memory. 

In contrast, NVIDIA’s profiling tool Nsight Compute (NCU) [36] provides a GPU-specific stall classification based on warp status, as shown in Table 1. **math pipe throttle** occurs when a warp is waiting for an available execution pipeline to execute; **no instructions** can happen due to instruction cache misses; **imc miss** indicates stalls caused by immediate constant cache misses; just to name a few. The "not modeled" stalls shown in Table 1 were not modeled by previous GPU analytical models [21–25, 29, 55, 56]. In 

1497 

ISCA ’25, June 21–25, 2025, Tokyo, Japan 

Cao et al. 

fact, building models based on these GPU specific stalls makes an analytical model more accurate than using the stall classification from a general processor view. 

## **3 Baseline Model and Motivation** 

## **3.1 Baseline Model** 

To model the GPU performance, prior studies [24, 29, 55, 56] build GPU analytical models with enhanced interval analysis. In fact, interval analysis is a powerful tool successfully used for CPU performance modeling [27]. It splits the execution of a thread into several intervals with time boundary when stalls occur. But this is not enough to accurately model GPU performance. The GPU analytical model MDM [56] therefore enhances the interval analysis by considering the memory stalls caused by the memory resource contention during the memory access from L1 cache to device memory. GCoM [29] further considers the computing resources contention, detailed architecture of modern GPUs (e.g., four sub-cores in a SM and sectored L1 D Cache), and the imbalance of workload based on MDM, improving the model accuracy and in turn becoming the SOTA GPU analytical model. 

Since our GPU analytical model is based on GCoM, we first briefly introduce GCoM and take it as our baseline. GCoM generally employs a hierarchical modeling approach (from **SM** to **sub-core** and then to **sub-core components** such as L1 D Cache) to model the cycles consumed by a CUDA kernel, so called _kernel cycles_ . Since a CUDA kernel is typically launched with specified thread block and grid dimensions, it therefore runs on a number of SMs in parallel. GCoM models the kernel cycles of such a CUDA kernel as the arithmetic mean of the cycles consumed by the CUDA kernel on all the active SMs at the highest level, as equation (3) shows. 

**==> picture [186 x 27] intentionally omitted <==**

with _𝐶[𝑘𝑒𝑟𝑛𝑒𝑙]_ the kernel cycles of a CUDA kernel, _𝑛𝑢𝑚𝑆𝑀𝑠_ the number of active SMs running the kernel, and _𝐶𝑖_ the cycles consumed by the CUDA kernel running on the _𝑖[𝑡ℎ]_ SM . 

To model _𝐶𝑖_ , GCoM firstly models the cycles consumed by the kernel on each sub-core ( _𝑠𝑢𝑏𝐶 𝑗_ ), at the next level of the hierarchy shown in Figure 1, as a sum of the active and idle cycles. The idle cycles are incurred by the the load imbalance among the sub-cores and therefore GCoM models it as the difference between the active cycles of the current sub-core and those of the longest running sub-core in the same SM. The active cycles, on the other hand, may be influenced by data dependencies, as well as long latency memory accesses. GCoM thus models the active cycles of a sub-core as equation (4) shows. 

**==> picture [232 x 14] intentionally omitted <==**

with _𝑠𝑢𝑏𝐶[𝑎𝑐𝑡𝑖𝑣𝑒] 𝑗_ the active cycles consumed by the kernel on the _𝑗[𝑡ℎ]_ sub-core, _𝑠𝑢𝑏𝐶[𝑏𝑎𝑠𝑒] 𝑗_ the cycles used to execute the warp instructions of the kernel on the _𝑗[𝑡ℎ]_ sub-core without any hazard, _𝑆𝑢𝑏𝐶𝑆[𝐶𝑜𝑚𝐷𝑎𝑡𝑎] 𝑗_ the stalled cycles caused by data (e.g., operand) hazards, and _𝑠𝑢𝑏𝐶𝑆[𝑀𝑒𝑚𝐷𝑎𝑡𝑎] 𝑗_ the stalled cycles incurred by long-latency memory accesses. 

As such, GCoM calculates _𝐶𝑖_ with equation (5), 

**==> picture [196 x 29] intentionally omitted <==**

with _𝑛𝑢𝑚𝑆𝑢𝑏𝑐𝑠_ the number of sub-cores in the _𝑖[𝑡ℎ]_ SM and _𝑆𝑖_ the stalled cycles of the _𝑖[𝑡ℎ]_ SM. 

The modeling of _𝑆𝑖_ in GCoM goes to the lowest level of the hierarchy shown in Figure 1, considering the L1 D Cache misses caused memory stalls; on the other hand, it also considers the compute resource contention incurred stalls, as well as memory resource contention caused stalls. Equation (6) shows the _𝑆𝑖_ model. 

_𝑆𝑖_ = _𝑆𝑖[𝑐𝑜𝑚𝑆𝑡𝑟𝑢𝑐𝑡]_ + _𝑆𝑖[𝑚𝑒𝑚𝑆𝑡𝑟𝑢𝑐𝑡]_ + _𝑆𝑖[𝑀𝑆𝐻𝑅]_ + _𝑆𝑖[𝑁𝑜𝐶]_ + _𝑆𝑖[𝐷𝑅𝐴𝑀]_ (6) 

with _𝑆𝑖[𝑐𝑜𝑚𝑆𝑡𝑟𝑢𝑐𝑡]_ the compute resource contention caused stalls, _𝑆𝑖[𝑚𝑒𝑚𝑆𝑡𝑟𝑢𝑐𝑡]_ the memory resource contention incurred stalls, _𝑆𝑖[𝑀𝑆𝐻𝑅]_ MSHR (miss status/handler registers) contention caused stalls, _𝑆𝑖[𝑁𝑜𝐶]_ the network on chip contention caused stalls, and _𝑆𝑖[𝐷𝑅𝐴𝑀]_ the LLC misses caused the memory access latencies. Note that the last three items in equation (6) are modeled by MDM [56] whereas GCoM models the left two items. 

We now introduce how GCoM models _𝑆[𝑐𝑜𝑚𝑆𝑡𝑟𝑢𝑐𝑡]_ and _𝑆[𝑚𝑒𝑚𝑆𝑡𝑟𝑢𝑐𝑡]_ . Since resource contention directly influences the cycles used to is- _𝑖 𝑖_ sue warp instructions, GCoM firstly models the issue cycles. To this end, it firstly determines the active sub-cores when there are a number of concurrently-executing warps, as equation (7) shows. 

**==> picture [187 x 9] intentionally omitted <==**

with _𝑥_ the number of concurrently-executing warps and _𝑛𝑢𝑚𝑆𝑢𝑏𝑐𝑠_ the number of sub-cores in a SM. Subsequently, GCoM models the maximum issue cycles in the _𝑘[𝑡ℎ]_ interval, the issue cycles as compute resources are sufficient in the _𝑘[𝑡ℎ]_ interval, the issue cycles to the _𝑚[𝑡ℎ]_ functional unit (FU) in the _𝑘[𝑡ℎ]_ interval, and the issue cycles to the L1 D Cache in the _𝑘[𝑡ℎ]_ interval as equations (8), (9), (10), and (11) show, respectively. 

**==> picture [229 x 44] intentionally omitted <==**

**==> picture [215 x 20] intentionally omitted <==**

**==> picture [162 x 28] intentionally omitted <==**

with _𝑥_ the number of concurrently-executing warps in the _𝑘[𝑡ℎ]_ interval, _𝑛𝑢𝑚𝐴𝑐𝑡𝑆𝐶𝑠_ the number of active sub-cores in a SM, _𝐼𝑠𝑠𝑢𝑒𝑅𝑎𝑡𝑒_ the warp instruction issue rate, _𝐼𝑘_ the number of warp instructions in the _𝑘[𝑡ℎ]_ interval of the representative warp, _𝐼𝑚_ the number of warp instructions dispatched to the _𝑚[𝑡ℎ]_ FU, _𝐼𝐼𝑚_ the initiation interval of the _𝑚[𝑡ℎ]_ FU, _𝑏𝑘_ the amount of L1 D Cache accesses incurred by the representative warp, and _𝐵𝑘[𝐿]_[1][the effective L1 D Cache] bandwidth in the _𝑘[𝑡ℎ]_ interval. 

Finally, GCoM models the stalls caused by compute resource contention by equation (12) 

**==> picture [229 x 23] intentionally omitted <==**

1498 

ISCA ’25, June 21–25, 2025, Tokyo, Japan 

AMALI: An Analytical Model for Accurately Modeling LLM Inference on Modern GPUs 

**Table 1: The stall event classification in Nsight compute; For simplicity, we omit several types of stalls:Synchronization and control-related stalls that prior work considers negligible, including warpgroup_arrive, barrier, membar, branch_resolving, sleeping and misc. Additional stalls with minimal impact: not_selected, drain and dispatch_stall** 

|**Stall type**|**Description**|**Classifcation in prior work**|
|---|---|---|
|selected|Warp was selected by the micro scheduler and issued an instruction.|Base in single warp model|
|wait|Warp was stalled waiting on a fxed latency execution dependency.|ComData|
|long_scoreboard|Warp was stalled waiting for a scoreboard dependency on a L1TEX (local|MemData for global memory|
||global surface texture) operation.|access|
|short_scoreboard|Warp was stalled waiting for a scoreboard dependency on a MIO (memory|MemData for share memory ac-|
||input/output) operation (not to L1TEX).|cess|
|math_pipe_throttle|Warp was stalled waiting for the execution pipe to be available.|ComStruct|
|tex_throttle|Warp was stalled waiting for the L1 instruction queue for texture operations|MemStruct|
||to be not full.||
|lg_throttle|Warp was stalled waiting for the L1 instruction queue for local and global|MemStruct|
||(LG) memory operations to be not full.||
|mio_throttle|Warp was stalled waiting for the MIO (memory input/output) instruction|MDM|
||queue to be not full.||
|no_instructions|Warp was stalled waiting to be selected to fetch an instruction or waiting|not modeled|
||on an instruction cache miss.||
|imc_miss|Warp was stalled waiting for an immediate constant cache (IMC) miss.|not modeled|



When _𝐶𝑘,_[Issue] L1[becomes] _[ 𝐶] 𝑘_[IssueMax] ( _𝑥_ ), GCoM employs equation (13) to model the memory contention caused stalls. 

**==> picture [229 x 23] intentionally omitted <==**

## **3.2 Prior Work Limitations** 

After briefly introducing the GPU analytical model GCoM, we now analyze its limitations. 

**Limitation #1: Initiation interval modeling inappropriately models tensor cores.** As shown in equation (10), GCoM needs to use initiation interval ( _𝐼𝐼𝑚_ ) of the _𝑚[𝑡ℎ]_ FU to calculate the issue cycles to the _𝑚[𝑡ℎ]_ FU in the _𝑘[𝑡ℎ]_ interval ( _𝐶𝑘,𝑚[𝐼𝑠𝑠𝑢𝑒]_[(] _[𝑥]_[)][). The] initiation interval denotes elapsed cycles between issuing two operations of the same type of FU [20]. The initiation intervals of different types of FU may be different. Prior GPU analytical models [8, 29, 48, 60] including GCoM [29] use it to model the computing resource contention, as equation (14) shows. 

**==> picture [201 x 21] intentionally omitted <==**

with _𝑤𝑎𝑟𝑝_  𝑠𝑖𝑧𝑒_ the number of threads in a warp which is typically 32 and _𝑓𝑢𝑛𝑐𝑡𝑖𝑜𝑛𝑎𝑙_  𝑢𝑛𝑖𝑡_  𝑙𝑎𝑛𝑒𝑠_ the number of FUs of a sub-core. 

As such, initiation interval actually models the throughput of a FU [20], because it can be calculated as the reciprocal of the elapsed cycles between two continuous computing results from the FU. When _𝑓𝑢𝑛𝑐𝑡𝑖𝑜𝑛𝑎𝑙_  𝑢𝑛𝑖𝑡_  𝑙𝑎𝑛𝑒𝑠_ is less than _𝑤𝑎𝑟𝑝_  𝑠𝑖𝑧𝑒_ , computing resource contention occurs and the warp scheduler takes the same number of cycles as the initiation interval to issue a warp instruction. This approach works well for modeling CUDA cores, where 

thread contention occurs as threads in a warp compete for access to computing resource, namely CUDA cores. 

However, this works poorly for tensor cores. When threads run on CUDA cores, each warp thread only accesses its own registers to execute instructions. In contrast, when threads run on tensor cores, all threads in warp share the same register file as described in Section 2. This allows the threads to work together to perform operations like matrix multiplications(e.g. HMMA instruction) in a unified way, rather than individually. As such, this makes the initiation interval of tensor cores can be significantly less than the number calculated as (14). 

We conduct experiment to confirm this analysis by comparing the math pipe throttle stalls defined in NCU against the computing resource contention modeled by GCoM using initiation interval when we run Llama2-7B inference on RTX 3090. Figure 2a shows the results. As can be seen, GCoM significantly overestimates the math pipe throttle stalls caused by resource contention of the GEMM kernel in Llama2-7B inference compared to the those occurred on the real hardware. This overestimation arises because the initial interval estimation for the tensor cores is too large relative to the real situation. 

**Limitation #2: Ignoring instruction modifiers does not lead to accurate modeling for tensor cores.** Existing GPU analytical models [8, 29, 48, 60] including GCoM [29] do not consider the instruction modifiers such as data type (e.g., F32). This might be acceptable for CUDA core instructions but unacceptable for tensor core instructions because tensor size modifiers influence the performance of tensor core instructions significantly. To confirm this, we firstly develop micro-benchmarks to measure the performance of 

1499 

ISCA ’25, June 21–25, 2025, Tokyo, Japan 

Cao et al. 

**==> picture [233 x 252] intentionally omitted <==**

**----- Start of picture text -----**<br>
selected+wait+scoreboard math pipe throttle memory side stall<br>imc miss and I$ miss ignored stalls<br>1<br>2<br>1 . 51 00 .. 86<br>0 . 4<br>0 . 5 0 . 2<br>0 0<br>(a) GEMM_fp16 (b) VELE<br>Figure 2: CPI stack constructed by GCoM and AMALI com-<br>pared to hardware(HW) with a NVIDIA RTX3090<br>·10 [5]<br>3 HMMA.1688.F32<br>HMMA.16816.F32<br>2<br>1<br>0<br>0 500 1 , 000 1 , 500 2 , 000 2 , 500<br>FMAs per instruction<br>Normalized CPI Normalized CPI<br>GCoM AMALI HW GCoM AMALI HW<br>Total cycles<br>**----- End of picture text -----**<br>


**Figure 2: CPI stack constructed by GCoM and AMALI compared to hardware(HW) with a NVIDIA RTX3090** 

**Figure 3: Total cycles of the** _𝐻𝑀𝑀𝐴_ **instruction with modifiers** 16816 **and** 1688 

tensor core instructions. Subsequently, we utilize cuAssembler [12] to modify the SASS trace of our micro-benchmark by altering only the instruction modifier of HMMA, tensor size, from 16816 to 1688 while keep other factors such as the modifier _𝐹_ 32 and instruction count unchanged. Figure 3 shows that the number of FMAs (floating multiply-add) per HMMA instruction with modifier 16816 doubles that of the instruction with modifier 1688. The same applies to the total cycles taken by HMMA with modifiers 16816 and 1688. 

This indicates that the cycles per instruction (CPI), which can be treated as throughput, of an HMMA with 16816 is double that of an HMMA with 1688. The reason is as follows. An HMMA with 16816 performs 16 × 8 × 16 = 2048 FMAs while that with 1688 performs 16 × 8 × 8 = 1024 FMAs. The FMAs per cycle is a design parameter of a GPU tensor core. For example, each tensor core of A100 GPU is designed to perform 8 × 4 × 8 = 256 FMAs [39] in a single cycle. Therefore, an A100 tensor core needs 8 and 4 cycles to execute an HMMA with 16816 and the one with 1688, respectively. That is, the CPI of HMMA with 16816 is 8, which is double that with 1688 (CPI=4). In summary, both our experiments and theoretical analysis show that modifiers significantly influence the throughput of tensor core instructions and in turn we must take modifiers into account as we model the throughput of tensor cores, see Section 4.7. 

**Limitation #3: Constant cache modeling does not consider implicit constant memory accesses and instruction cache modeling is ignored.** We find the _𝑖𝑚𝑐_  𝑚𝑖𝑠𝑠_ (immediate constant cache misses) defined by NCU may be caused by not only explicit but also implicit constant memory accesses. Existing GPU analytical models [8, 29, 48, 60] including GCoM [29] model constant memory access by leveraging explicit load and store instructions (e.g., _𝐿𝐷𝐶_ and _𝑆𝑇𝐶_ ). However, constant memory is also heavily accessed by 

**==> picture [152 x 73] intentionally omitted <==**

**----- Start of picture text -----**<br>
0 . 8<br>0 . 6 imc cache miss stall<br>no instruction stall<br>0 . 4 sum of two<br>0 . 2<br>0<br>10 [4] 10 [5] 10 [6]<br>Kernel Cycles<br>Stall Ratio<br>**----- End of picture text -----**<br>


**Figure 4: Imc cache miss stall and no instruction stall refer to kernel cycles in Llama2 inference** 

**==> picture [218 x 132] intentionally omitted <==**

**----- Start of picture text -----**<br>
3 , 000 40<br>2 , 000<br>1 , 000 20<br>0 0<br>0 2 , 000 4 , 000 6 , 000 200000 300000<br>Instr. count Instr. count<br>(a) ELE_4096_160 (b) GEMM_bf16_4096_172<br>800 3 , 000<br>600400 2 , 000<br>200 1 , 000<br>0 0<br>0 10 , 000 20 , 000 30 , 000 0 1 , 000 2 , 000<br>Instr. count Instr. count<br>(c) GEMM_bf16_6144_162 (d) UELE_6144_165<br>Warp Number<br>Warp Number<br>**----- End of picture text -----**<br>


**Figure 5: Distribution of instruction number. In this study, we use the notation {name}_{length}_{id} to denote a kernel, where name is the abbreviation of the kernel’s name, length represents the prompt’s length in terms of token count and id identifies the specific kernel index.** 

implicit instructions. Expression (15) shows an example. As can be seen, the constant memory address _𝑐_ [0 _𝑥_ 0][0 _𝑥_ 28] is encoded as an operand of the instruction by using the symbol " _𝑐_ ". Ignoring these instructions makes the modeling of immediate constant cache inaccurate. 

## _𝐼𝑀𝐴𝐷.𝑀𝑂𝑉.𝑈_ 32 _𝑅_ 1 _, 𝑅𝑍, 𝑅𝑍,𝑐_ [0 _𝑥_ 0][0 _𝑥_ 28] (15) 

On the other hand, prior models do not model instruction cache miss either, making them inaccurate for modeling the _𝑛𝑜_  𝑖𝑛𝑠𝑡𝑟𝑢𝑐𝑡𝑖𝑜𝑛_ stalls defined in NCU. As shown in Fig.2b, GCoM fails to account for stalls caused by constant cache misses and instruction cache misses. As a result, it significantly underestimates the cycles of the VELE kernel during Llama2-7B inference. 

Moreover, Figure 4 shows the sum of stalled cycles caused by immediate constant cache misses and instruction cache misses can be a high ratio (e.g. 70%) in the CPI stack, indicating they can not be ignored in GPU analytical models. 

**Limitation #4: Existing GPU analytical models do not consider the warp characteristics of LLM inference.** GPU analytical models [8, 29, 48, 60] including GCoM [29] employ K-means to select a representative warp to represent all the warps in a CUDA kernel. Prior studies claim that this approach is accurate enough [24, 29, 56]. This might be true for kernels from Rodinia [10] and Parboil [49] benchmark suites. However, we find that the warp execution flows in a CUDA kernel of LLM inference applications are significantly different, as Figure 5 shows. 

1500 

ISCA ’25, June 21–25, 2025, Tokyo, Japan 

AMALI: An Analytical Model for Accurately Modeling LLM Inference on Modern GPUs 

A couple of interesting observations can be made here. For one, different CUDA kernels in a LLM inference have significantly different number of warps, from tens to thousands. This is because LLMs consists of more operators than DNNs. Taking GEMM as an example, GEMMs in one LLM application are significantly more heterogeneous than those in a traditional DNN such as RNN [42]. For example, the GEMMs in the attention layer of LLMs are generally memory-bound while those in FFN layers are compute-bound. In contrast, the GEMMs in one DNN are generally compute-bound as shown in [42]. Moreover, the dimension of the GEMMs in LLM might be dramatically different. For instance, the _𝑏𝑠_ × _𝑠𝑒𝑞_  𝑙𝑡ℎ_ ( _𝑏𝑠_ - batch size, _𝑠𝑒𝑞𝑙𝑡ℎ_ - sequence length) corresponds to the _𝑀_ of a GEMM _𝑀_ × _𝑘_ × _𝑁_ in an LLM inference. In the prefill stage, suppose the _𝑠𝑒𝑞_  𝑙𝑡ℎ_ is 32,768 and _𝑏𝑠_ is 4, then _𝑀_ is 131 _,_ 072. In the decode stage, the _𝑠𝑒𝑞_  𝑙𝑡ℎ_ is always 1 because of the auto-regressive manner and the _𝑏𝑠_ can still be 4, then _𝑀_ is only 4. That is, the _𝑀_ of the GEMM _𝑀_ × _𝑘_ × _𝑁_ of the prefill stage is 32,768 times of the _𝑀_ of the decode stage!. 

Second, the number of instructions in some warps is dramatically different from that of other warps in the same CUDA kernel. Taking the kernel _ELE_4096_160_ as an example, each of 3,000 warps only contain several hundreds of instructions, as the left bar in Figure 5a shows. In contrast, the right highest bar in Figure 5a shows that each of 2,800 warps in the same kernel contains more than 6,000 instructions. Finally, the warp instruction number difference of some kernels such as _GEMM_bf16_4096_172_ is extremely large, from less than 100 _,_ 000 to more than 600 _,_ 000. 

In summary, such significant difference in warp instruction number in the same CUDA kernel in LLM inference makes using one representative warp to represent all the warps of a CUDA kernel infeasible. However, for interval analysis, using one representative warp is _required_ . We address this extreme challenge in Section 4.9. 

## **4 The AMALI Model** 

To address the above limitations in the case of running LLM inference on modern GPUs, we propose a novel GPU analytical model dubbed AMALI. It predicts the total cycles consumed by a CUDA kernel of a LLM inference application. 

**==> picture [227 x 99] intentionally omitted <==**

**----- Start of picture text -----**<br>
SASS Memory Cache<br>Predicted cycles<br>Tracer Access Simulator<br>SASSInfo AMAT 𝑆𝐼 , 𝐼𝐷 𝑆𝐼𝑃 𝐾𝐿𝐿<br>SASS Warp Interval Interval Interval<br>Parser Profile Analyzer Profiles Parser<br>Blocksize<br>Gridsize<br>KLL var<br>KLL comp Architectural Parameters<br>**----- End of picture text -----**<br>


**Figure 6: An overview of AMALI. SASS - CUDA assembly instruction. AMAT - Average Memory Access Time. KLL - Kernel Launch Latency.** _𝑆𝐼_ **- Results of interval analyzer.** _𝐼𝐷_ **Instruction Divergence.** 

## **4.2 SASS Tracer** 

It is developed based on the GPU instrumentation tool NVBit [53] to collect SASS instruction traces and the related information of CUDA kernels. In detail, it collects instruction names, instruction modifiers, the registers an instruction uses, memory accessing addresses, grid size, thread block size, consumed shared memory size, consumed register file size, warp IDs, and SM IDs. Previous works [3, 19, 29, 60] have demonstrated that modeling with the SASS offers greater accuracy than PTX, so our SASS Tracer focuses on collecting information of SASS instructions. 

## **4.3 SASS Parser** 

Our SASS Parser extracts required information such as grid size and thread block size from the traces produced by the SASS Tracer. Since a representative warp is required for interval modeling, we leverage the SASS Tracer to constructs a single-warp representation based on the FUs used by each warp, encoding each warp as a vector. To this end, the SASS Parser applies k-means clustering, following an approach similar to GCoM [29] and GPUMech [24] and capture the _selected stall_ events defined in NCU. Note that each warp is scheduled to a sub-core by using the scheme _𝑠𝑢𝑏_ − _𝑐𝑜𝑟𝑒_  𝑖𝑑_ = _𝑤𝑎𝑟𝑝_  𝑖𝑑_ %4, as described in [26]. 

## **4.4 Cache Simulator** 

## **4.1 Overview** 

Figure 6 shows an overview of AMALI. As can be seen, it consists of six components: SASS Tracer, SASS Parser, Cache Simulator, Interval Analyzer, Interval Parser, and KLL comp (kernel launching latency). The SASS Tracer collects instruction traces and related information of CUDA kernels. The Cache Simulator simulates the cache behavior based on the memory access traces obtained by the SASS Tracer. The SASS Parser extracts required information from SASS traces. The Interval Analyzer partitions the execution of a warp into intervals. The Interval Parser leverages the produced intervals to build models to predict cycles consumed by a kernel. KLL comp computes launching latency of a CUDA kernel. 

To use AMALI, we need to know the architecture parameters of a GPU. To this end, we develop micro-benchmarks and we follow the pointer chase method, as described in [4, 33], to measure the latency and throughput of FUs and the memory system. 

The Cache Simulator simulates the cache behavior by using the memory access addresses obtained by the SASS Tracer in conjunction with the specified GPU architecture configuration parameters. The goal of the cache simulation is to determine the Average Memory Access Times (AMATs). The AMATs and FUs latency are then used in interval analysis, as detailed in [27], to segment the execution of a warp into discrete intervals. 

## **4.5 Interval Analyzer** 

Our Interval Analyzer leverages the AMATs produced by the Cache Simulator and the warp profile obtained by the SASS Parser to partition the execution of a warp into discrete intervals. We now discuss how to match the stall events of data dependency in the interval analysis. First of all, AMALI estimates the total cycle of an kernel by using equation (16). 

**==> picture [166 x 10] intentionally omitted <==**

1501 

ISCA ’25, June 21–25, 2025, Tokyo, Japan 

Cao et al. 

**==> picture [230 x 126] intentionally omitted <==**

**----- Start of picture text -----**<br>
Quadratic Fit<br>1024 512 Slope<br>256 128<br>1 [·][10][4]<br>4<br>0 . 8<br>0 . 6 3<br>0 . 4 2<br>0 . 2 1<br>32 1 , 024 2 , 048 0 10 20 30<br>Gridsize Blocksize<br>(a) KLL values with different<br>gridsize, each color correspond- (b) Slopes with different block-<br>ing to a blocksize size<br>KLL Value Slope<br>**----- End of picture text -----**<br>


**Figure 7: KLL values with different gridsize and slopes with different blocksize** 

where _𝐶_ is the estimated cycles of a kernel. _𝑆𝐼_ and _𝑆𝐼𝑃_ represent the stalls captured by the Interval Analyzer and Interval Parser, respectively. _𝐾𝐿𝐿_ means the kernel launching latency and _𝐼𝐷_ denotes the idle cycles caused by instruction divergence. At the beginning, AMALI models the _selected stalled_ cycles as the number of instructions issued by the representative warp, as equation (17) shows: 

**==> picture [172 x 10] intentionally omitted <==**

where _𝐼𝑤𝑎𝑟𝑝_ is the number of instructions in the representative warp and _𝑆𝑆𝑒𝑙𝑒𝑐𝑡𝑒𝑑_ is the stalled cycles of the stall type ’selected’ as Table 1 shows. 

We adopt the traditional interval analysis method [24, 27] in our Interval Analyzer to model data dependencies caused by the global or shared memory access and computing instructions. For every instruction, we mark the end cycle of it. For a global memory instruction, the end cycle of it is the current cycle plus AMAT. For share memory instruction, the end cycle is the current cycle plus share memory access cycles depending on load/store [1]. 

For instructions executed on CUDA cores, AMALI models their performance based on fixed latency from architectural parameters. In contrast, the latency of tensor core instructions is characterized by throughput based on modifiers including data type and tensor size (, seeing the details in Section 4.7). When the data that an instruction uses are not immediately available, a stall interval is inserted until the instruction can be issued. The result of interval analysis is computed by using equation (18). 

**==> picture [237 x 11] intentionally omitted <==**

with _𝑆𝐼_ the predicted cycles of the interval, _𝑆𝑤𝑎𝑖𝑡_ the stalled cycles due to data dependency of computing instructions,the stalled cycles caused by shared memory, and _𝑆 𝑆𝑠ℎ𝑜𝑟𝑡𝑙𝑜𝑛𝑔_ _ 𝑠𝑐𝑜𝑟𝑒𝑏𝑜𝑎𝑟𝑑𝑠𝑐𝑜𝑟𝑒𝑏𝑜𝑎𝑟𝑑_ stalled cycles incurred by global memory. 

For a stalled interval, AMALI captures the stall reasons, which are categorized into computing, shared memory access, or other memory access stalls. The Interval Analyzer then analyzes these information of the representative warp to get the three stalled cycles: _𝑆𝑤𝑎𝑖𝑡_ , _𝑆𝑠ℎ𝑜𝑟𝑡_  𝑠𝑐𝑜𝑟𝑒𝑏𝑜𝑎𝑟𝑑_ and _𝑆𝑙𝑜𝑛𝑔_  𝑠𝑐𝑜𝑟𝑒𝑏𝑜𝑎𝑟𝑑_ . The sum of them can be calculated by equations (19) and (20). 

**==> picture [180 x 22] intentionally omitted <==**

**==> picture [199 x 68] intentionally omitted <==**

**----- Start of picture text -----**<br>
10 [8]<br>Cycle Per Token<br>10 [7] [.] [5]<br>0 5 10 244 249 254<br>Token Index<br>Cycle<br>**----- End of picture text -----**<br>


**Figure 8: Cycle per token in Llama2-7B inference** 

**==> picture [167 x 28] intentionally omitted <==**

where _𝑖_ and _𝑓𝑙𝑎𝑔_ [ _𝑘_ ] denote interval stall reasons, which can be include wait, long_scoreboard, or short_scoreboard. _𝑓𝑙𝑎𝑔_ [ _𝑘_ ] denotes the stall reason of the _𝑘[𝑡ℎ]_ interval and _𝑆𝑘_ means the length in cycles of the _𝑘[𝑡ℎ]_ stall interval. Note that _𝑥_ = _𝑦_ indicates the stall reasons are the same. 

## **4.6 Interval Parser** 

Our Interval Parser captures math_pipe_throttle, lg_throttle, tex_throttle and mio_throttle. We employ equation (21) to compute the sum of these stalled cycles: 

**==> picture [176 x 11] intentionally omitted <==**

where _𝑆𝐼𝑃_ is the resulted cycles by the Interval Parser. _𝑆𝑚𝑎𝑡ℎ_  𝑝𝑖𝑝𝑒_ , _𝑆𝑙𝑔_ , and _𝑆𝑚𝑖𝑜_ represent the math pipe throttle, lg throttle, and mio throttle caused stall cycles, respectively. 

To calculate the first two of the three "throttle" caused stall cycles, we use the core modeling approach of GCoM [29], as equations (22) and (23) show. 

**==> picture [167 x 29] intentionally omitted <==**

where the _𝑆[𝐶𝑜𝑚𝑆𝑡𝑢𝑐𝑡]_ and _𝑆[𝑀𝑒𝑚𝑠𝑡𝑟𝑢𝑐𝑡]_ are the stalled cycles caused by the computing and memory structure hazards, respectively. The _𝑆𝑚𝑎𝑡ℎ_  𝑝𝑖𝑝𝑒_ denotes the stalled cycles caused by the computing resource hazard while _𝑆𝑙𝑔_ represents the stalled cycles incurred by the L1 D Cache access contention. 

For the last "throttle" cased stalled cycles (mio_throttle), AMALI uses the MDM to model the memory side contentions which are _𝑆𝑀𝑆𝐻𝑅_ , _𝑆𝑁𝑜𝐶_ and _𝑆𝐷𝑅𝐴𝑀_ , as equation (24) shows. 

**==> picture [181 x 11] intentionally omitted <==**

where _𝑆𝑚𝑖𝑜_ denotes the stalled cycles caused by memory contention. _𝑆[𝑀𝑆𝐻𝑅] ,𝑆[𝑁𝑜𝐶]_ and _𝑆[𝐷𝑅𝐴𝑀]_ are the MSHRs, NoC and DRAM contentions incurred stalled cycles, respectively. 

## **4.7 Tensor Core Modeling** 

Unlike previous studies [24, 29, 56] model CUDA core performance without considering instruction modifiers, AMALI takes the instruction modifiers into account for modeling the tensor core performance. In fact, AMALI uses the throughput obtained from microbenchmarks along with modifiers to model tensor core, as equation (25) shows. 

**==> picture [158 x 21] intentionally omitted <==**

where _𝐼𝐼𝑇𝐶_ means initiation interval for tensor cores based on the current HMMA instruction. _𝐹𝑀𝐴_  𝑐𝑜𝑢𝑛𝑡_ means the FMA flops of 

1502 

ISCA ’25, June 21–25, 2025, Tokyo, Japan 

AMALI: An Analytical Model for Accurately Modeling LLM Inference on Modern GPUs 

the instruction, which is influenced by the modifier. _𝑇𝑃𝑑𝑡_ means the peak throughput in FMA operations per cycle (FMAs/cycle) based on the datatype _𝑑𝑡_ . In fact, FMAs/cycle is a design parameter for tensor cores. _𝐼𝐼𝑇𝐶_ is used to replace the _𝐼𝐼𝑚_ in equation (10) to predict the issue cycles on the tensor cores in an interval. The _𝐿𝑇𝐶_ is actually the latency of a tensor core instruction, which can be denoted by equation (26). 

**==> picture [142 x 9] intentionally omitted <==**

where _𝐿𝑇𝐶_ is the latency of a single tensor core instruction which is used in interval analysis. 

## **4.8 Modeling Constant/Instruction Cache** 

We use kernel launching latency (KLL) to model the stalled cycles caused by constant cache misses and instruction cache misses. The KLL is expressed by equation (27): 

**==> picture [152 x 10] intentionally omitted <==**

where _𝐾𝐿𝐿_ is the kernel launching latency, _𝑠_ is a parameter that depends on thread block size, _𝑘_ is a fixed number specific to the architecture and GS is the grid size of the kernel, which can be obtained from the SASS Tracer. The parameters _𝑠_ and _𝑘_ are determined by micro-benchmarks. 

Figure 7 shows the results of micro-benchmarks on the A100 GPU platform. For the Ampere architecture, with a fixed thread block size, the kernel launching latency exhibits a linear relationship with the grid size. However, the slope of this line is non-linear with respect to the thread block size. To model this slope, we employ a second-order function of thread block size as equation (28) shows: 

_𝑠_ = _𝛼_ · ( _𝐵𝑆_ )[2] + _𝛽_ · ( _𝐵𝑆_ ) + _𝛾_ (28) 

Here, _𝑠_ represents the slope of the line and _𝐵𝑆_ is the thread block size of a kernel, obtained from the SASS Tracer. The coefficients _𝛼_ , _𝛽_ , _𝛾_ are constants and can be determined by benchmarks. 

## **4.9 Modeling Warp Instruction Distribution** 

GCoM improves the model accuracy by considering the workload imbalance caused by the warp number distribution in the entire GPU. But we find that in LLM inference, the workload imbalance does not appear in warp number distribution. Instead, it appears in warp instruction number distribution. For accurately modeling kernel cycles in LLM inference, modeling instruction divergence ( _𝐼𝐷_ ) is important. We employ equation (29) to model _𝐼𝐷_ . 

**Table 2: GPU configuration for NVIDIA A100** 

|**Parameter**|**Value**|
|---|---|
|ClockFrequency|1410 MHz|
|SM|#108, 4 sub-coresper SM|
|WarpScheduler|single-issue, not modelpolicy|
|Functional units/SM|INT: #64,4 cycles/warp inst.<br>FP32: #64,4 cycles/warp inst.<br>FP64: #32,4 cycles/warp inst.<br>SFU: #16,23 cycles/warp inst.<br>Tensor Core: #4<br>256 bf16 FMAs/clk for fp32 accumulate<br>256 fp16 FMAs/clk for fp16 or fp32 acc|
|L1 cache|37 cycles, sectored, streaming, write-through,<br>128 B/line, 32 B/sector, unlimited MSHRs,<br>64 ways, 4 banks|
|Share memory|23 cycles for ld, 19 cycles for st|
|L2 cache|224 cycles, Sectored, streaming, write-back,<br>80 channels, 40 MB, 16 ways, 32 B/line|
|DRAM|290 cycles, 40 channels, 1940 GB/s, 1512 MHz|
|NoC|1200 MHz|



two AMD EPYC 7543 CPUs, a NVIDIA A100 GPU with 80GB VRAM and 2TB of DDR4 DRAM. The software environment is an Ubuntu 22.04.5 LTS system with CUDA version 11.7. The Architectural Parameters of A100 is shown in Table 2. To demonstrate AMALI can be used to explore the GPU design space, we use H100 as a validation GPU to validate the tensor core capability design proposed by AMALI. The tensor core capability of H100 is 8 × 8 × 8 (512) FMAs per cycle, which is double that of A100. 

## **5.2 Representative Applications** 

We employ Llama3-8B [14] with prompt length of 2048, 4096 and 6144 tokens. While Llama3-8B supports context windows up to 8192 tokens, we are limited to 6144 tokens in our experiments because the memory capability required by more than 6144 tokens exceeds the device memory capability of the A100 GPU. Moreover, we use Llama3-8B inference with a 256-token prompt but with different batch sizes. In addition, we employ Llama3-15B with different prompt lengths to evaluate AMALI. Finally, we choose CONV and GEMM from DeepBench [35], and BP, B+, DWT, PF from Rodina [10] to evaluate how AMALI performs on traditional GPU benchmarks. 

_𝐼𝐷_ = (maxsub-coreInstr − _𝐼𝑆𝐶_  𝑅𝑒𝑝𝑟.𝑤𝑎𝑟𝑝_ )/ _𝐼𝑠𝑠𝑢𝑒𝑅𝑎𝑡𝑒_ (29) 

where the _𝑚𝑎𝑥𝑠𝑢𝑏_ − _𝑐𝑜𝑟𝑒𝐼𝑛𝑠𝑡𝑟_ is the maximum number of warp instructions on a sub-core in the kernel. _𝐼𝑆𝐶_  𝑅𝑒𝑝𝑟.𝑤𝑎𝑟𝑝_ is the number of warp instructions of the sub-core which executes the representative warp. _𝐼𝑠𝑠𝑢𝑒𝑅𝑎𝑡𝑒_ is the instruction issuing rate in instructions per cycle. As such, the unit of _𝐼𝐷_ is cycles. By adding it to other stalled cycles shown in equation (16), we can improve the accuracy of modeling kernel cycles in LLM inference. 

## **5 Experiment Setup** 

## **5.1 Hardware and System Software** 

To evaluate AMALI, we compare the total kernel cycle prediction against GCoM. We ran the experiments on a server equipped with 

## **6 Results and Analysis** 

## **6.1 Determining the Token Count for Testing** 

Due to the large size of the per-token tracing file, we can not evaluate AMALI on a single GPU machine with too many tokens. But how many tokens we need to test in our experiment? We determine it by observing the time between tokens with different number of input and output tokens. Figure 8 shows that subsequent tokens executed with the same count of cycles after the first generated token. This result allows us to test only the first two tokens. The first token inference is the prefill phase and the second is the decode phase. Following previous work [29], we utilized the NCU to measure the _𝑔𝑝𝑐_ _ 𝑐𝑦𝑐𝑙𝑒𝑠_  𝑒𝑙𝑎𝑝𝑠𝑒𝑑.𝑎𝑣𝑔_ metric over 10 times, averaging 

1503 

ISCA ’25, June 21–25, 2025, Tokyo, Japan 

Cao et al. 

**==> picture [476 x 143] intentionally omitted <==**

**----- Start of picture text -----**<br>
GCoM GCoM+KLL GCoM+ID GCoM+TCM AMALI HW<br>4<br>3<br>2<br>1<br>0<br>UEL E 204 M8 bf1159 6 204 8 EL162 E SoftMa204 8 173 x 204 M8 bf1176 6 204 M8 bf1178 6 204UEL 8E 190204 M8 bf11582 6 204 M8 bf11590 6 409 6 EL161 E 409UEL 6 174 E 409UEL 6 175 E 409 M6 bf1177 6M 409bf1 6 194 6 409UEL 6 1537 E 409 M6 bf11588 6 409 M6 bf11617 6 409 M6 bf11623 6 614 M4 bf1161 6 614 4 EL172 E 614UEL 4 173 E 614 M4 bf1177 6 614 M4 bf1186GEMVx 6 614 4 202 6 614 M4 bf11863 6 614 4 1880<br>GEM GEM GEM GEM GEM GEM GEM GEM GEM GEM GEM GEM GEM GEM<br>Cycles<br>Normalized<br>**----- End of picture text -----**<br>


**Figure 9: The kernel cycle predicted by GCoM and AMALI and the ground truth captured on hardware(HW). GCoM+KLL - GCoM is extended with Kernel Launch Latency modeling. GCoM+ID - GCoM with warp Instruction Distribution modeling. GCoM+TCM - GCoM with Tensor Core Modeling.** 

the results to obtain a reliable ground truth of elapsed cycles for a kernel execution. 

## **6.2 Determining the coefficients** _𝛼_ **,** _𝛽_ **, and** _𝛾_ 

We develop micro-benchmarks to determine the coefficients _𝛼_ , _𝛽_ , and _𝛾_ used in equation (28). On A100, _𝛼_ , _𝛽_ , and _𝛾_ are 0.0036, 0.0366 and 1.1891, respectively. On different GPUs, they might be different. 

## **6.3 AMALI Accuracy** 

We first use the mean absolute percentage error (MAPE) to evaluate the accuracy of AMALI on the highly frequently executed, as well as relatively long kernels in Llama-8B inference on A100. We also evaluate how our individual extension beyond GCoM improves the accuracy: tensor core modeling (TCM), kernel launch latency modeling (KLL), and warp instruction distribution (ID). Figure 9 shows the results. A couple of interesting findings can be made here. For one, AMALI achieves significantly higher accuracy compared to GCoM thanks to the correct modeling on tensor cores, kernel launch latency, and warp instruction distribution. 

Second, for GEMM kernels, the throughput based tensor core modeling (TCM) of AMALI is the main contributor for its high accuracy. In contrast, GCoM uses the modeling approach for CUDA cores to model the tensor cores, which is unreasonable and results in high overestimate of these kernel cycles. Last but not least, for element-wise operations (ELE_...) which use CUDA cores, GCoM significantly underestimates the cycles taken by them. In contrast, AMALI accurately predict their cycles thanks to the modeling of kernel launch latency (KLL) and warp instruction distribution (ID). Figure 10 shows that, for GEMM kernels, TCM demonstrates the most significant improvement, whereas for ELE kernels, the KLL optimization is most effective. Overall, AMALI achieves a MAPE of 17.84% in predicting GEMM kernels and 27.29% for ELE kernels, whereas GCoM records MAPE of 183.95% and 77.81%, respectively. In summary, the average MAPE of GCoM for these kernels is 127.6% while that for AMALI is only 23.5%. 

Next, we evaluate the end to end performance prediction accuracy of AMALI using Llama3-8B with different prompt lengths from 128 tokens to 6144 tokens. Figure 11 shows the results. As can 

**==> picture [224 x 200] intentionally omitted <==**

**----- Start of picture text -----**<br>
200 100<br>150 80<br>60<br>100 40<br>50 20<br>0 0<br>(a) GEMM (b) ELE<br>Figure 10: MAPE comparison for GEMM and ELE<br>200<br>150 Prefill phase Decode phase<br>100<br>50<br>0<br>128 tks 256 tks 512 tks 1024 tks 2048 tks 4096 tks 6144 tks<br>GCoMGCoM+KLLGCoM+IDGCoM+TCMAMALI GCoMGCoM+KLLGCoM+IDGCoM+TCMAMALI<br>MAPE (%)<br>(%)<br>MAPE<br>GCoM GCoM+KLLGCoM+IDGCoM+TCM AMALI GCoM GCoM+KLLGCoM+IDGCoM+TCM AMALI GCoM GCoM+KLLGCoM+IDGCoM+TCM AMALI GCoM GCoM+KLLGCoM+IDGCoM+TCM AMALI GCoM GCoM+KLLGCoM+IDGCoM+TCM AMALI GCoM GCoM+KLLGCoM+IDGCoM+TCM AMALI GCoM GCoM+KLLGCoM+IDGCoM+TCM AMALI<br>**----- End of picture text -----**<br>


**Figure 11: MAPE comparison for Llama3-8B inference with different prompt lengths (tokens)** 

be seen, AMALI achieves significantly lower MAPEs than GCoM for all the experimented prompt lengths, and KLL, ID, as well as TCM all reduce the MAPEs of GCoM but with different levels. In the prefill phase, AMALI achieves a MAPE of 15.56%, while GCoM shows significantly higher error of 163.68%. Similarly, in the decode phase, AMALI maintains better performance with a MAPE of 34.90%, while error of GCoM remains as high as 82.29%. Moreover, GCoM shows significantly higher MAPEs for the prefill phase than those for the decode phase with the same long prompt (e.g., _>_ 1024 tokens). This is because the prefill with long prompts involves significantly larger GEMMs than those for decode, GEMMs execute on tensor cores, but GCoM does not model tensor cores accurately. In contrast, for both prefill and decode, AMALI achieves significantly low MAPEs for all the prompt lengths thanks to the accurate tensor core modeling. 

Moreover, we evaluate how AMALI performs with different batch sizes using Llama3-8B inference with a 256-token prompt. 

1504 

ISCA ’25, June 21–25, 2025, Tokyo, Japan 

AMALI: An Analytical Model for Accurately Modeling LLM Inference on Modern GPUs 

**==> picture [238 x 78] intentionally omitted <==**

**----- Start of picture text -----**<br>
Prefill phase Decode phase Prefill phase Decode phase<br>1008060 200150<br>40 100<br>20 50<br>0 0<br>bs=1 bs=2 bs=4 bs=8 2048 tokens 4096 tokens<br>(a) Llama3-8B inference across (b) Llama3-15B inference with<br>different batch sizes different prompt lengths<br>(%) (%)<br>MAPE MAPE<br>GCoM GCoM+KLL GCoM+ID GCoM+TCM AMALI GCoM GCoM+KLL GCoM+ID GCoM+TCM AMALI GCoM GCoM+KLL GCoM+ID GCoM+TCM AMALI GCoM GCoM+KLL GCoM+ID GCoM+TCM AMALI GCoM GCoM+KLL GCoM+ID GCoM+TCM AMALI GCoM GCoM+KLL GCoM+ID GCoM+TCM AMALI<br>**----- End of picture text -----**<br>


**Figure 12: MAPE comparison for Llama3-15B inference with different prompt lengths (tokens) and Llama3-8B inference across different batch sizes with a 256-token prompt** 

As Figure 12a shows, AMALI is still the best among the experimented analytical models. This indicates that AMALI works well for different batch sizes of LLM inferences. 

In addition, we evaluate AMALI with larger LLMs which can perform a sort of extreme stress testing for a Hardware. To this end, we build a model named Llama-15B based on Llama2-13B with data type of BF16. The memory capacity requirement for the weights of this model and the KV cache for 4096 tokens approaches 80GB of our A100 GPU. Figure 12b shows the results. As can be seen, AMALI still achieves the lowest MAPEs among all the experimented analytical models. This indicates that AMALI can predict the performance of different LLM inference models accurately. 

Next, we compare the performance prediction accuracy predicted by MDM, GCoM and AMALI using Llama3-8B inference with different prompt lengths. Figure 13 shows the results. As can be seen, AMALI, MDM, and GCoM achieve the lowest, second lowest, and highest MAPEs, respectively. AMALI achieves the lowest MAPEs as expected because it accurately models the tensor core throughput, warp instruction distribution, and kernel launch latency. However, it is counter-intuitive that GCoM shows higher MAPEs than MDM because GCoM extends MDM with compute-core modeling. Nevertheless, this is true and the reason is as follows. GCoM accurately models CUDA cores and uses the same approach to model tensor cores while the performance of tensor cores is dramatically different from that of CUDA cores. On the other hand, MDM does not model CUDA cores, neither tensor cores, which does not have the computing errors. This results in even higher errors of GCoM than MDM’s errors when GCoM predicts the performance of LLM inference which heavily uses tensor cores. 

Finally, we compare the accuracy of MDM, GCoM, and AMALI by using non-LLM applications: CONV and GEMM from DeepBench [35], and BP , B+ tree, DWT, and PF from Rodinia [10]. Figure 14 shows the results. As can be seen, for convolution and other non-GEMM kernels, GCoM and MDM both show high accuracy but AMALI achieves higher accuracy thanks to the constant and icache cache modeling and warp instruction distribution modeling. For GEMM and the like kernels, AMALI shows significantly higher accuracy than MDM and GCoM because AMALI accurately models the tensor cores. 

## **6.4 Design Space Exploration** 

AMALI’s FMAs/cycle (throughput) based tensor core modeling enables accurate performance predictions across different tensor core throughput settings. In fact, FMAs/cycle can be a design parameters for NVIDIA GPU tensor cores, and newer generation of 

**==> picture [225 x 187] intentionally omitted <==**

**----- Start of picture text -----**<br>
200<br>150 MDM GCoM AMALI<br>100<br>50<br>0<br>128 tks 256 tks 512 tks 1024 tks 2048 tks 4096 tks 6144 tks<br>Figure 13: MAPE comparison for Llama3-8B inference with<br>different prompt lengths (tokens) for MDM, GCoM, AMALI<br>MDM GCoM AMALI<br>80<br>60<br>40<br>20<br>0<br>CONV GEMM BP B+ DWT PF<br>DeepBench Rodinia<br>(%)<br>MAPE<br>MDM GCoM AMALI MDM GCoM AMALI MDM GCoM AMALI MDM GCoM AMALI MDM GCoM AMALI MDM GCoM AMALI MDM GCoM AMALI<br>(%)<br>MAPE<br>MDM GCoM AMALI MDM GCoM AMALI MDM GCoM AMALI MDM GCoM AMALI MDM GCoM AMALI MDM GCoM AMALI<br>**----- End of picture text -----**<br>


**Figure 13: MAPE comparison for Llama3-8B inference with different prompt lengths (tokens) for MDM, GCoM, AMALI** 

**Figure 14: MAPE comparison for CONV and GEMM in DeepBench and backprop, B+tree, Discrete wavelet transform and Path finder in Rodinia** 

tensor cores may have higher FMAs/cycle. Figure 15 compares the kernel cycle predictions between AMALI and GCoM with 128, 256, and 512 FMAs/cycle. As can be seen, AMALI predicts less cycles consumed by all the GEMM kernels when the tensor cores have higher FMAs/cycle (e.g., 512). In contrast, GCoM predicts the same cycles with variant tensor core capabilities, which is wrong. This is because GCoM uses the modeling approach for CUDA cores to model tensor cores, which can not reflect the impact of tensor cores with different FMAs/cycle on the overall kernel performance. 

We further employ AMALI to explore the tensor core throughput design by using it to predict the performance of five GEMM kernels from DeepBench [35] and compare the predictions against the measured performance on A100 and H100. The FMAs/cycle of tensor cores for A100 and H100 are 256 and 512, respectively. We take them as the tensor core throughput design values in AMALI to predict the cycles consumed by the five kernels running on A100 and H100. Higher tensor core throughput is expected to produce lower math_pipe_throttle and in turn less kernel cycles. 

Figure 16 shows that the math_pipe_throttle of a kernel on H100 is only half of that of the same kernel on A100, and in turn higher performance on H100. (The left and right adjacent bars in one block partitioned by the red dash lines denotes the consumed cycles of the same kernel running on A100 and H100, respectively). This is because the tensor core throughput of H100 is double that of A100. This indicates that AMALI can accurately evaluate impact of a design parameter on a special stall events. Moreover, compared to the measured kernels cycles, the prediction error of AMALI can be as low as 1.03% and the maximum error does not exceed 23%. The average errors are only 8.2% and 13.2% on A100 and H100, respectively. These results indicate AMALI is accurate enough for GPU design space exploration in early stages. 

## **6.5 Discussions** 

We propose instruction divergence to reduce the influence of warp instruction imbalance in a kernel. It is unreasonable to model cycles 

1505 

ISCA ’25, June 21–25, 2025, Tokyo, Japan 

Cao et al. 

**==> picture [223 x 102] intentionally omitted <==**

**----- Start of picture text -----**<br>
AMALI GCoM<br>4<br>3<br>2<br>1<br>0<br>GEMM 2 048 1 61 GEMM 2 048 1 62 GEMM 2 048 1 72<br>cycles<br>Normalized<br>F/c F/c F/c F/c F/c F/c F/c F/c F/c<br>128 256 512 128 256 512 128 256 512<br>**----- End of picture text -----**<br>


**Figure 15: Kernel cylces predicted by AMALI and GCoM with varying FMAs per clk per tensor core:128, 256, 512** 

by selecting the largest warp of a kernel as the representative warp. Interval analysis assumes all warps as the same warp, so using the largest warp as representative will occur enormously over estimation. So we only model the issued instruction number difference between the warps, which shows high accuracy. But there is still a room to further improve the accuracy by studying better strategy to model the warp instruction number divergence. 

## **7 Related work** 

## **7.1 Analytical model** 

The GPU analytical model mirrors many analytical technical aspects in the CPU analytical model, and interval analysis [27] is one of the power tools from CPU analysis studies. Huang et al. proposed GPUMech [24], which is the first analytical model for GPU based on interval analysis. MDM [55, 56] considered memory divergence in kernels. Lee et al. [29] added modern gpu architecture details including sector cache, sub-cores in SM, computing resource contention to the analytical model and considered the workload imbalance. But they modeled latency based on the number of FUs, rather than throughput, and ignored instruction divergence of warps. 

Besides the interval analysis-based GPU analytical models, there are other ways to model GPUs. Hong et al. [21] built a model based on the degree of memory warp parallelism and computation warp parallelism, and they further extended the MWP-CWP model by proposing an integrated power and performance (IPP) prediction model [23] for GPUs. Jain et al. [25] analyzed GPU simulation accuracy with GPGPU-Sim, showing high accuracy for computeintensive tasks but significant errors for memory-bound workloads. Lym et al. [32] found memory access patterns in deep learning algorithms like convolution and employed a specific analytical model, but they focused on memory traffic, and cannot capture the stalls caused by contention in the computing source. SeyyedAghaei et al. [47] used the behavior of the application running on the small device to estimate the performance on the large-scale platform. AIO [44] is a performance model for various accelerators. However, none of them modeled the tensor cores of GPUs. 

## **7.2 GPU Simulator** 

GPU simulator is the de-facto standard for exploring the bottleneck of kernels. Bakhoda et al. [9] built a detailed GPU simulator named GPGPU-sim and they further extended its capabilities by developing Accel-Sim [28], which introduces support for SASS. Leng et al. [30] 

**==> picture [236 x 113] intentionally omitted <==**

**Figure 16: Kernel cycles with break-downs predicted by AMALI of five kernels on A100 and H100, and the measured cycles. X axis represents the five GEMM (** _𝑀_ × _𝐾_ × _𝑁_ **) kernels. Each block partitioned by the red dash lines contains two bars, and the left and right ones denote the cycles consumed by the same kernel running on A100 and H100, respectively.** 

integrated GPUWattch with GPGPU-sim to model power performance. Ubal et al. [17, 50] proposed Multi2Sim, an open-source, modular and fully configurable toolset for ISA-level simulation of x86 CPUs and an AMD Evergreen GPUs. gem5-gpu [43] is an open-source simulator built upon gem5 and GPGPUSim, focused on modeling tightly integrated CPU-GPU systems, capable of enabling concurrent execution of CPUs and GPUs. Emerald [18] is a simulator that integrates with GPGPU-Sim, gem5, and Android to model graphics and GPGPU applications in mobile SoCs. ATTILA [13] is a cycle-level execution-driven GPU simulator that uses a boxand-signal-based model. Wang et al. [57] proposed a source code analysis approach to generate execution trace by pruning, loop bound analysis and branch extraction. PPT-GPU [3, 5] employed a memory model to obtain AMAT but, different from the analytical model, implemented a cycle-approximate simulator to estimate performance. Zhang et al. [59] introduced LLMCompass, a fast and accurate hardware evaluation framework for LLM inference, but it still faces the challenge of long simulation time. 

## **7.3 Machine learning based model** 

In terms of a machine learning-based model, there are some works [19, 58] that tried to use the machine learning-based method to predict the total cycle of GPU kernels. Ardalani et al. [6] proposed CrossArchitecture Performance Prediction (XAPP), a machine learningbased technique that uses single-threaded CPU implementations to predict GPU performance. But they face the problem of limited training data and cannot provide deep GPU architectural insights. 

## **8 Conclusion** 

We have successfully constructed a GPU analytical model named AMALI to accurately predict the performance of a CUDA kernel in the context of LLM inference applications on modern GPUs. AMALI meticulously models the tensor cores and constant/instruction cache of modern GPUs when they execute LLM inference applications. Moreover, AMALI builds a multi-warp model to reflect LLM inference’s unique characteristics. These techniques make AMALI a convincible as well as convenient tool to fast explore the GPU architecture design space for LLM inferences with deep insights. 

1506 

ISCA ’25, June 21–25, 2025, Tokyo, Japan 

AMALI: An Analytical Model for Accurately Modeling LLM Inference on Modern GPUs 

## **References** 

- [1] Hamdy Abdelkhalik, Yehia Arafa, Nandakishore Santhi, and Abdel-Hameed A. Badawy. 2022. Demystifying the Nvidia Ampere Architecture through Microbenchmarking and Instruction-level Analysis. In _2022 IEEE High Performance Extreme Computing Conference (HPEC)_ . IEEE, New York, NY, USA, 1–8. doi:10.1109/HPEC55821.2022.9926299 

- [2] Johnathan Alsop, Matthew D. Sinclair, Rakesh Komuravelli, and Sarita V. Adve. 2016. GSI: A GPU Stall Inspector to characterize the sources of memory stalls for tightly coupled GPUs. In _2016 IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS)_ . IEEE, Uppsala, Sweden, 172–182. doi:10. 1109/ISPASS.2016.7482092 

- [3] Yehia Arafa, Abdel-Hameed Badawy, Ammar ElWazir, Atanu Barai, Ali Eker, Gopinath Chennupati, Nandakishore Santhi, and Stephan Eidenbenz. 2021. Hybrid, Scalable, Trace-Driven Performance Modeling of GPGPUs. In _SC21: International Conference for High Performance Computing, Networking, Storage and Analysis_ . IEEE, New York, NY, USA, 1–15. doi:10.1145/3458817.3476221 

- [4] Yehia Arafa, Abdel-Hameed A. Badawy, Gopinath Chennupati, Nandakishore Santhi, and Stephan Eidenbenz. 2019. Low Overhead Instruction Latency Characterization for NVIDIA GPGPUs. In _2019 IEEE High Performance Extreme Computing Conference (HPEC)_ . IEEE, New York, NY, USA, 1–8. doi:10.1109/HPEC.2019. 8916466 

- [5] Yehia Arafa, Abdel-Hameed A. Badawy, Gopinath Chennupati, Nandakishore Santhi, and Stephan Eidenbenz. 2019. PPT-GPU: Scalable GPU Performance Modeling. _IEEE Computer Architecture Letters_ 18, 1 (2019), 55–58. doi:10.1109/ LCA.2019.2904497 

- [6] Newsha Ardalani, Clint Lestourgeon, Karthikeyan Sankaralingam, and Xiaojin Zhu. 2015. Cross-architecture performance prediction (XAPP) using CPU code to predict GPU performance. In _2015 48th Annual IEEE/ACM International Symposium on Microarchitecture (MICRO)_ . IEEE, New York, NY, USA, 725–737. doi:10.1145/2830772.2830780 

- [7] Arun Chandrasekaran. 2024. Spotlight on 2024 Gartner Hype Cycle™for Emerging Technologies. https://www.gartner.com/en/articles/hype-cycle-foremerging-technologies. [Accessed: 2025-02-09]. 

- [8] Sara S. Baghsorkhi, Matthieu Delahaye, Sanjay J. Patel, William D. Gropp, and Wen-mei W. Hwu. 2010. An adaptive performance modeling tool for GPU architectures. In _Proceedings of the 15th ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming_ (Bangalore, India) _(PPoPP ’10)_ . Association for Computing Machinery, New York, NY, USA, 105–114. doi:10.1145/1693453. 1693470 

- [9] Ali Bakhoda, George L. Yuan, Wilson W. L. Fung, Henry Wong, and Tor M. Aamodt. 2009. Analyzing CUDA workloads using a detailed GPU simulator. In _2009 IEEE International Symposium on Performance Analysis of Systems and Software_ . IEEE, New York, NY, USA, 163–174. doi:10.1109/ISPASS.2009.4919648 

- [10] Shuai Che, Michael Boyer, Jiayuan Meng, David Tarjan, Jeremy W. Sheaffer, SangHa Lee, and Kevin Skadron. 2009. Rodinia: A benchmark suite for heterogeneous computing. In _2009 IEEE International Symposium on Workload Characterization (IISWC)_ . IEEE, New York, NY, USA, 44–54. doi:10.1109/IISWC.2009.5306797 

- [11] Jack Choquette, Wishwesh Gandhi, Olivier Giroux, Nick Stam, and Ronny Krashinsky. 2021. NVIDIA A100 Tensor Core GPU: Performance and Innovation. _IEEE Micro_ 41, 2 (2021), 29–35. doi:10.1109/MM.2021.3061394 

- [12] CloudCores. 2024. CuAssembler: A CUDA PTX Assembly Tool. https://github. com/cloudcores/CuAssembler. [Accessed: 2024-11-10]. 

- [13] V.M. del Barrio, C. Gonzalez, J. Roca, A. Fernandez, and Espasa E. 2006. ATTILA: a cycle-level execution-driven simulator for modern GPU architectures. In _2006 IEEE International Symposium on Performance Analysis of Systems and Software_ . IEEE, New York, NY, USA, 231–241. doi:10.1109/ISPASS.2006.1620807 

- [14] A. Dubey, A. Jauhri, A. Pandey, et al. 2024. _The Llama 3 Herd of Models_ . Technical Report. Meta Platforms, Inc. arXiv:2407.21783 https://arxiv.org/abs/2407.21783 [Accessed: 2024-11-10]. 

- [15] Michael Garland, Scott Le Grand, John Nickolls, Joshua Anderson, Jim Hardwick, Scott Morton, Everett Phillips, Yao Zhang, and Vasily Volkov. 2008. Parallel Computing Experiences with CUDA. _IEEE Micro_ 28, 4 (2008), 13–27. doi:10.1109/ MM.2008.57 

- [16] GitHub. 2021. GitHub Copilot. https://github.com/features/copilot/. Accessed: 2024-11-16. 

- [17] Xun Gong, Rafael Ubal, and David Kaeli. 2017. Multi2Sim Kepler: A detailed architectural GPU simulator. In _2017 IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS)_ . IEEE, New York, NY, USA, 269–278. doi:10.1109/ISPASS.2017.7975298 

- [18] Ayub A. Gubran and Tor M. Aamodt. 2019. Emerald: Graphics Modeling for SoC Systems. In _2019 ACM/IEEE 46th Annual International Symposium on Computer Architecture (ISCA)_ . IEEE, New York, NY, USA, 169–182. 

- [19] João Guerreiro, Aleksandar Ilic, Nuno Roma, and Pedro Tomás. 2019. GPU Static Modeling Using PTX and Deep Structured Learning. _IEEE Access_ 7 (2019), 159150–159161. doi:10.1109/ACCESS.2019.2951218 

- [20] John L. Hennessy and David A. Patterson. 2011. _Computer Architecture: A Quantitative Approach_ (5th ed.). Morgan Kaufmann Publishers Inc., San Mateo, CA, USA. 

- [21] Sunpyo Hong and Hyesoon Kim. 2009. An analytical model for a GPU architecture with memory-level and thread-level parallelism awareness. In _Proceedings of the 36th Annual International Symposium on Computer Architecture_ (Austin, TX, USA) _(ISCA ’09)_ . Association for Computing Machinery, New York, NY, USA, 152–163. doi:10.1145/1555754.1555775 

- [22] Sunpyo Hong and Hyesoon Kim. 2009. An analytical model for a GPU architecture with memory-level and thread-level parallelism awareness. _SIGARCH Comput. Archit. News_ 37, 3 (June 2009), 152–163. doi:10.1145/1555815.1555775 

- [23] Sunpyo Hong and Hyesoon Kim. 2010. An integrated GPU power and performance model. In _Proceedings of the 37th Annual International Symposium on Computer Architecture_ (Saint-Malo, France) _(ISCA ’10)_ . Association for Computing Machinery, New York, NY, USA, 280–289. doi:10.1145/1815961.1815998 

- [24] Jen-Cheng Huang, Joo Hwan Lee, Hyesoon Kim, and Hsien-Hsin S. Lee. 2014. GPUMech: GPU Performance Modeling Technique Based on Interval Analysis. In _2014 47th Annual IEEE/ACM International Symposium on Microarchitecture_ . IEEE, Cambridge, UK, 268–279. doi:10.1109/MICRO.2014.59 

- [25] Akshay Jain, Mahmoud Khairy, and Timothy G. Rogers. 2018. A Quantitative Evaluation of Contemporary GPU Simulation Methodology. _Proc. ACM Meas. Anal. Comput. Syst._ 2, 2, Article 35 (June 2018), 28 pages. doi:10.1145/3224430 

- [26] Zhe Jia, Marco Maggioni, Benjamin Staiger, and Daniele Paolo Scarpazza. 2018. Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking. _ArXiv_ abs/1804.06826 (2018). https://api.semanticscholar.org/CorpusID:4930164 

- [27] T.S. Karkhanis and J.E. Smith. 2004. A first-order superscalar processor model. In _Proceedings. 31st Annual International Symposium on Computer Architecture, 2004._ IEEE, New York, NY, USA, 338–349. doi:10.1109/ISCA.2004.1310786 

- [28] Mahmoud Khairy, Zhesheng Shen, Tor M. Aamodt, and Timothy G. Rogers. 2020. Accel-Sim: An Extensible Simulation Framework for Validated GPU Modeling. In _2020 ACM/IEEE 47th Annual International Symposium on Computer Architecture (ISCA)_ . IEEE, New York, NY, USA, 473–486. doi:10.1109/ISCA45697.2020.00047 

- [29] Jounghoo Lee, Yeonan Ha, Suhyun Lee, Jinyoung Woo, Jinho Lee, Hanhwi Jang, and Youngsok Kim. 2022. GCoM: a detailed GPU core model for accurate analytical modeling of modern GPUs. In _Proceedings of the 49th Annual International Symposium on Computer Architecture_ (New York, New York) _(ISCA ’22)_ . Association for Computing Machinery, New York, NY, USA, 424–436. doi:10.1145/3470496.3527384 

- [30] Jingwen Leng, Tayler Hetherington, Ahmed ElTantawy, Syed Gilani, Nam Sung Kim, Tor M. Aamodt, and Vijay Janapa Reddi. 2013. GPUWattch: enabling energy optimizations in GPGPUs. In _Proceedings of the 40th Annual International Symposium on Computer Architecture_ (Tel-Aviv, Israel) _(ISCA ’13)_ . Association for Computing Machinery, New York, NY, USA, 487–498. doi:10.1145/2485922. 2485964 

- [31] Yuxi Liu, Zhibin Yu, Lieven Eeckhout, Vijay Janapa Reddi, Yingwei Luo, Xiaolin Wang, Zhenlin Wang, and Chengzhong Xu. 2016. Barrier-Aware Warp Scheduling for Throughput Processors. In _Proceedings of the 2016 International Conference on Supercomputing_ (Istanbul, Turkey) _(ICS ’16)_ . Association for Computing Machinery, New York, NY, USA, Article 42, 12 pages. doi:10.1145/2925426.2926267 

- [32] Sangkug Lym, Donghyuk Lee, Mike O’Connor, Niladrish Chatterjee, and Mattan Erez. 2019. DeLTA: GPU Performance Model for Deep Learning Applications with In-Depth Memory System Traffic Analysis. In _2019 IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS)_ . IEEE, Madison, WI, USA, 293–303. doi:10.1109/ISPASS.2019.00041 

- [33] Xinxin Mei and Xiaowen Chu. 2017. Dissecting GPU Memory Hierarchy Through Microbenchmarking. _IEEE Transactions on Parallel and Distributed Systems_ 28, 1 (2017), 72–86. doi:10.1109/TPDS.2016.2549523 

- [34] Meta-Llama. 2023. Llama. GitHub Repository. https://github.com/Meta-Llama/ Llama [Accessed: 2023-11-10]. 

- [35] Sharan Narang and Greg Diamos. 2016. Baidu DeepBench. https://svail.github. io/DeepBench. [Accessed: 2024-11-10]. 

- [36] NVIDIA Corporation 2020. _Nsight Compute CLI_ . NVIDIA Corporation. https: //developer.nvidia.com/nsight-compute-cli Updated in 2021. 

- [37] NVIDIA Corporation. 2024. CUDA Binary Utilities Documentation. https://docs. nvidia.com/cuda/cuda-binary-utilities/index.html. Accessed: 2024-11-19. 

- [38] NVIDIA Corporation. 2024. CUDA C++ Best Practices Guide. https://docs.nvidia. com/cuda/cuda-c-best-practices-guide/index.html. [Accessed: 2024-11-10]. 

- [39] NVIDIA Corporation. 2025. NVIDIA Ampere Architecture Whitepaper. https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidiaampere-architecture-whitepaper.pdf. Accessed: February 15, 2025. 

- [40] OpenAI. 2022. ChatGPT. https://openai.com/index/chatgpt/. Accessed: 2024-1116. 

- [41] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Köpf, Edward Z. Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. 2019. PyTorch: An Imperative Style, High-Performance Deep Learning Library. In _Advances in Neural Information Processing Systems 32: Annual Conference on Neural Information Processing Systems 2019, NeurIPS 2019, December 8-14, 2019, Vancouver, BC, Canada_ . Curran Associates, Inc., Vancouver, BC, Canada, 8024–8035. 

1507 

ISCA ’25, June 21–25, 2025, Tokyo, Japan 

Cao et al. 

- [42] S. Pati, S. Aga, N. Jayasena, and M. D. Sinclair. 2022. Demystifying BERT: System Design Implications. In _2022 IEEE International Symposium on Workload Characterization (IISWC)_ . IEEE, Austin, TX, USA, 296–309. doi:10.1109/IISWC55918. 2022.00033 

- [43] Jason Power, Joel Hestness, Marc S. Orr, Mark D. Hill, and David A. Wood. 2015. gem5-gpu: A Heterogeneous CPU-GPU Simulator. _IEEE Computer Architecture Letters_ 14, 1 (2015), 34–36. doi:10.1109/LCA.2014.2299539 

- [44] Joseph Rogers, Taha Soliman, and Magnus Jahre. 2024. AIO: An Abstraction for Performance Analysis Across Diverse Accelerator Architectures. In _2024 ACM/IEEE 51st Annual International Symposium on Computer Architecture (ISCA)_ . IEEE, Buenos Aires, Argentina, 487–500. doi:10.1109/ISCA59077.2024.00043 

- [45] Timothy G. Rogers, Mike O’Connor, and Tor M. Aamodt. 2012. Cache-Conscious Wavefront Scheduling. In _2012 45th Annual IEEE/ACM International Symposium on Microarchitecture_ . IEEE, New York, NY, USA, 72–83. doi:10.1109/MICRO.2012.16 

- [46] Run:AI. 2024. PyTorch GPU: A Guide to Multi GPU Training. https://www.run. ai/guides/gpu-deep-learning/pytorch-gpu. Accessed: 2024-11-19. 

- [47] Hossein SeyyedAghaei, Mahmood Naderan-Tahan, and Lieven Eeckhout. 2024. GPU Scale-Model Simulation. In _2024 IEEE International Symposium on HighPerformance Computer Architecture (HPCA)_ . IEEE, Edinburgh, United Kingdom, 1125–1140. doi:10.1109/HPCA57654.2024.00088 

- [48] Jaewoong Sim, Aniruddha Dasgupta, Hyesoon Kim, and Richard Vuduc. 2012. A performance analysis framework for identifying potential benefits in GPGPU applications. In _Proceedings of the 17th ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming_ (New Orleans, Louisiana, USA) _(PPoPP ’12)_ . Association for Computing Machinery, New York, NY, USA, 11–22. doi:10.1145/ 2145816.2145819 

- [49] John A. Stratton, Christopher Rodrigues, I-Jui Sung, Nady Obeid, Li-Wen Chang, Nasser Anssari, Geng Daniel Liu, and Wen mei W. Hwu. 2012. _IMPACT Technical Report, IMPACT-12-01_ . Technical Report IMPACT-12-01. University of Illinois at Urbana-Champaign, Urbana, IL, USA. https://api.semanticscholar.org/CorpusID: 497928 

- [50] Rafael Ubal, Byunghyun Jang, Perhaad Mistry, Dana Schaa, and David Kaeli. 2012. Multi2Sim: a simulation framework for CPU-GPU computing. In _Proceedings of the 21st International Conference on Parallel Architectures and Compilation Techniques_ (Minneapolis, Minnesota, USA) _(PACT ’12)_ . Association for Computing Machinery, New York, NY, USA, 335–344. doi:10.1145/2370816.2370865 

- [51] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In _Proceedings of the 31st International Conference on Neural Information_ 

   - _Processing Systems_ (Long Beach, California, USA) _(NIPS’17)_ . Curran Associates Inc., Red Hook, NY, USA, 6000–6010. 

- [52] Oreste Villa, Daniel Lustig, Zi Yan, Evgeny Bolotin, Yaosheng Fu, Niladrish Chatterjee, Nan Jiang, and David Nellans. 2021. Need for Speed: Experiences Building a Trustworthy System-Level GPU Simulator. In _2021 IEEE International Symposium on High-Performance Computer Architecture (HPCA)_ . IEEE, New York, NY, USA, 868–880. doi:10.1109/HPCA51647.2021.00077 

- [53] Oreste Villa, Mark Stephenson, David Nellans, and Stephen W. Keckler. 2019. NVBit: A Dynamic Binary Instrumentation Framework for NVIDIA GPUs. In _Proceedings of the 52nd Annual IEEE/ACM International Symposium on Microarchitecture_ (Columbus, OH, USA) _(MICRO ’52)_ . Association for Computing Machinery, New York, NY, USA, 372–383. doi:10.1145/3352460.3358307 

- [54] vLLM Team. 2024. Easy, fast, and cheap LLM serving for everyone. https: //docs.vllm.ai/en/latest/. [Accessed: 2025-02-09]. 

- [55] Lu Wang, Magnus Jahre, Almutaz Adileh, Zhiying Wang, and Lieven Eeckhout. 2019. Modeling Emerging Memory-Divergent GPU Applications. _IEEE Computer Architecture Letters_ 18, 2 (2019), 95–98. doi:10.1109/LCA.2019.2923618 

- [56] Lu Wang, Magnus Jahre, Almutaz Adileho, and Lieven Eeckhout. 2020. MDM: The GPU Memory Divergence Model. In _2020 53rd Annual IEEE/ACM International Symposium on Microarchitecture (MICRO)_ . IEEE, New York, NY, USA, 1009–1021. doi:10.1109/MICRO50266.2020.00085 

- [57] Xiebing Wang, Kai Huang, Alois Knoll, and Xuehai Qian. 2019. A Hybrid Framework for Fast and Accurate GPU Performance Estimation through Source-Level Analysis and Trace-Based Simulation. In _2019 IEEE International Symposium on High Performance Computer Architecture (HPCA)_ . IEEE, New York, NY, USA, 506–518. doi:10.1109/HPCA.2019.00062 

- [58] Gene Wu, Joseph L. Greathouse, Alexander Lyashevsky, Nuwan Jayasena, and Derek Chiou. 2015. GPGPU performance and power estimation using machine learning. In _2015 IEEE 21st International Symposium on High Performance Computer Architecture (HPCA)_ . IEEE, New York, NY, USA, 564–576. doi:10.1109/HPCA.2015.7056063 

- [59] Hengrui Zhang, August Ning, Rohan Baskar Prabhakar, and David Wentzlaff. 2024. LLMCompass: Enabling Efficient Hardware Design for Large Language Model Inference. In _2024 ACM/IEEE 51st Annual International Symposium on Computer Architecture (ISCA)_ . IEEE, New York, NY, USA, 1080–1096. doi:10.1109/ ISCA59077.2024.00082 

- [60] Yao Zhang and John D. Owens. 2011. A quantitative performance analysis model for GPU architectures. In _2011 IEEE 17th International Symposium on High Performance Computer Architecture_ . IEEE, San Antonio, TX, USA, 382–393. doi:10.1109/HPCA.2011.5749745 

1508 

