2026 IEEE International Symposium on High-Performance Computer Architecture (HPCA) 

# μShare: Non-Intrusive Kernel Co-Locating on NVIDIA GPUs 

Wenhao Huang[1] , Zhaolin Duan[1] , Laiping Zhao[1] _[∗]_ , Yuhao Zhang[1] , Yanjie Wang[1] , Yiming Li[1] , Yihan Wang[1] , Yichi Chen[1] , Zhihang Tang[1] , Kang Chen[2] , Deze Zeng[3] , Wenxin Li[1] , and Keqiu Li[1] 

> 1Tianjin University 2Tsinghua University 3China University of Geosciences _{_ hwh, duanzhaol, laiping, yuhaozhang, wangyanjie, l ym, ehein, chenyichi, tangzhihang, toliwenxin, keqiu _}_ @tju.edu.cn, chenkang@tsinghua.edu.cn, deze@cug.edu.cn 

_**Abstract**_ **—The hardware scheduler on NVIDIA GPUs is highly inefficient in utilizing micro-architectural hardware resources. It places blocks from the same kernel within the same GPU Streaming Multiprocessor (SM) core, resulting in a stacking colocating problem, where identical blocks are placed within the same SM core, saturating only a subset of intra-SM hardware resources while leaving others underutilized.** 

**The primary challenge in addressing this issue is that the NVIDIA hardware is closed-source, preventing us from directly modifying the hardware scheduler. To bridge the semantic gap between the resource demands of kernels and the scheduler, we introduce** _**μShare**_ **, which enables intra-SM scattered colocating of kernels through a non-intrusive** _**half-plus blocksize shaping**_ **method. It shapes the blocksize of kernels to a halfplus blocksize (i.e., slightly more than half of the SM’s thread capacity), scattering identical blocks of the same kernel across different SMs. It further adopts a** _**time-shifted launching**_ **method to reduce intra-SM resource contention. Compared to state-ofthe-art systems,** _**μShare**_ **does not require intrusive modifications to hardware or kernel code, yet it can still improve inference throughput by 26.90%-54.09% and increases low-level hardware utilization by 38.53%–61.15%.** 

## I. INTRODUCTION 

Software-defined resource control can leverage the open interfaces provided by hardware to improve resource efficiency [18], [20], [31], [45]. For example, Intel’s Resource Director Technology (RDT) [25] supports fine-grained partitioning of shared CPU-socket resources, allowing multiple applications to coexist with minimal interference. Similarly, GPUs offer inter-SM (Streaming Multiprocessor) isolation mechanisms (e.g., MIG [30], MPS [32], and CU masking [1]), and some systems leverage these techniques to co-locate multiple tasks on a single GPU with lower interference [5], [6], [10], [48], [53], [55], [57], [59]. 

Despite these advancements at the inter-SM level, intraSM resource utilization remains inefficient. Modern GPUs, such as NVIDIA’s Ampere-based A100, include 108 SMs, each comprising 32 FP64 cores, 64 FP32 cores, 64 INT32 cores, 4 Tensor cores, 16 SFU units, and 32 load/store (LDST) units. However, NVIDIA GPUs do not expose interfaces for fine-grained resource allocation within individual SMs, resulting in inefficient management of these on-chip resources. Our experiments demonstrate that kernel execution on GPUs frequently exhibits a resource-utilization pattern we term “1 

> *Corresponding Author 

**==> picture [245 x 109] intentionally omitted <==**

**----- Start of picture text -----**<br>
Low�utilization High�utilization<br>Shaping<br>Kernel�1 Block�1 ��� Kernel�1 Block�1 ���<br>Block�2 FP32 FP64 Block�2 FP32 FP64<br>Kernel�2 Block�1 ��� Kernel�2 Block�1 ���<br>Block�2 Block�2<br>FP32 FP64 FP32 FP64<br>(a) Stacked Co-locating (b) Scattered Co-locating<br>**----- End of picture text -----**<br>


Fig. 1: (a) Stacked co-locating: blocks of a kernel are stacked within the same SM. (b) Scattered co-locating: shaped blocks from different kernels are scattered and co-located within a single SM. 

more, 5 less” wherein one hardware resource is heavily utilized while the other five resources remain significantly underutilized. For example, during the execution of the common matrix multiplication kernel in inference workloads, the utilization of Tensor cores is 88.52%, while the average utilization of the other five hardware resources is only 5.45%. 

The major reason for low hardware utilization is the _semantic gap between the resource demands of kernels and the resource allocation by the GPU hardware scheduler_ , which refers that the actual hardware resource requirements of the kernels are not conveyed to the scheduler, and the scheduler can not perform scheduling based on the hardware demands of multiple kernels. As shown in Figure 1(a), when a kernel is launched, its multiple blocks are queued in the CUDA stream and scheduled one by one in order. This _blockoriented sequential scheduling_ often results in the “ _stacked co-location_ ” problem, where identical blocks are placed within the same SM core. As identical blocks share the same resource requirements, this results in low intra-SM hardware utilization. 

To improve intra-SM hardware resource utilization, existing works have proposed _intrusive modification_ methods. These modifications fall into two main categories: (1) _intrusive modifications to CUDA kernels_ [4], [14], [29], [36], [61]–[63], which fuse multiple kernels with complementary resource requirements into a single, more complex kernel, and (2) _intrusive modifications to hardware_ [11], [38], [43], where the GPU is redesigned to open up kernel scheduling interfaces, typically validated through simulators. However, in public cloud where NVIDIA GPUs are widely used, and with the 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 04,2026 at 04:44:23 UTC from IEEE Xplore.  Restrictions apply. 

hardware scheduler being **closed-source** , intrusive hardware and cross-user kernel fusion are not feasible. 

Thus, the first challenge in bridging the semantic gap is determining how to **non-intrusively** convey kernel resource demands and enable the GPU hardware scheduler to perform resource-coupled scheduling effectively. Under the closedsource nature of NVIDIA GPUs, the only thing we can do is to perform control-based optimizations before the kernels are launched on the GPU, such as configuring the launching timing and kernel parameters (e.g., blocksize, which refers to the thread count for each block, and shared memory). Thus, the challenge is divided into three subproblems: (1) The closed-source nature of NVIDIA GPUs prevents us from fully understanding the hardware scheduler’s strategy; how can modifying kernel parameters enable the hardware scheduler to achieve intra-SM co-location? (2) CUDA kernels have multiple configurable parameters; which kernel parameters can serve as a general-purpose solution to be modified? (3) Certain CUDA kernels (e.g., cuDNN) are closed-source and unmodifiable; how should kernels with unmodifiable parameters be co-located? 

To address this challenge, we propose a hardware–software co-design approach that enables intra-SM co-location of kernels without modifying kernel code or GPU hardware. This method consists of the following key components: For the first subproblem, we perform extensive profiling under diverse parameter configurations and execution scenarios (i.e., exclusive and concurrent), collecting data on block placements within SMs and their execution timings, which enables us to further understand the block scheduling process. Specifically, under the GPU’s left-over scheduling strategy [36], [51], [52] (i.e., blocks are scheduled to SM cores that meet the required thread count), we propose the “ _half-plus blocksize shaping_ ” technique to separate blocks of kernels with similar dominant resource demands. As shown in Figure 1(b), by shaping the kernel’s _blocksize_ to _half-plus_ (i.e., slightly more than half of the SM’s thread capacity), multiple blocks from the same kernel are scattered across different SMs, the remaining threads in each SM can be allocated to blocks of other kernels, thereby improving intra-SM hardware utilization. For the second subproblem, our profiling reveals that the blocksize parameter is the only general-purpose solution that effectively scatters identical blocks across different SMs. In contrast, other parameters, such as shared memory, do not achieve this effect. For the third subproblem, we propose the “ _time-shifted launching_ ” technique to co-launch kernels with different resource requirements. For kernels with unmodifiable parameters, we control their launch timing to enable scattered co-location with other kernels that have modifiable parameters. 

Based on the “ _scattered co-locating_ ” method, we develop _μShare_ , an inference system that bridges the semantic gap between kernel resource demands and hardware scheduler resource allocation. _μShare_ is the first to simultaneously achieve three capabilities: kernel-level control, intra-SM co-location, and non-intrusiveness to hardware and kernels’ code. _μShare_ operates in Linux user space, without generating any additional 

CPU-GPU communication, and without requiring the reading or modification of any user code, it can increase the system throughput of an NVIDIA GPU from 1722 QPS (queries per second) to 3046 QPS, and improve the aggregate utilization of the six intra-SM hardware units from 56.22% to 90.60%, with only several nanoseconds of control overhead per kernel. **Contributions** . Our contributions are as follows: 

- **Analysis and discovery** : The primary cause of low resource utilization of GPU SMs is the “ _stacked colocating_ ” problem in the hardware scheduler. 

- **Half-plus blocksize shaping method** : Achieves kernellevel scattered co-location by simply shaping the blocksize of some kernels to half-plus of SM thread capacity. 

- _**μShare**_ **prototype** : A non-invasive system including kernel interception, parameter modification, scattered colocating and inference latency guarantee. 

- **Experimental evaluations** : Compared to state-ofthe-art, _μShare_ improves throughput by 26.90%54.09% and increases low-level hardware utilization by 38.53%–61.15%, while guaranteeing the latency SLO (Service Level Objective). 

## II. BACKGROUND & MOTIVATION 

## _A. The GPU Scheduling Process_ 

**==> picture [247 x 64] intentionally omitted <==**

**----- Start of picture text -----**<br>
Three-Stage Independent Decision-Making<br>Stage 1:Launch Stage 2:Schedule Stage 3:Execute<br>Pytorch FP32 FP64 INT32<br>Kernel Block Warp SFU LDST Tensor<br>SM<br>Reduced Control Granularity<br>**----- End of picture text -----**<br>


Fig. 2: A kernel’s lifecycle includes three distinct stages: launching, scheduling, and execution. 

The execution process of a CUDA kernel can be divided into three stages: (1) _launching_ , (2) _scheduling_ , and (3) _execution_ . As shown in Figure 2, the three stages proceed sequentially, each implementing a specific task: the launching of _kernels_ , the scheduling of _blocks_ , and the execution of _warps_ . 

In the _launching_ stage, a _kernel_ is launched by the inference framework (e.g., TF-Serving [49], Pytorch [40], TVM [7] and TensorRT [50] ) on the CPU side, and it reaches the CUDA stream via the GPU’s Command Processor. A _kernel_ is a specialized function executed on the GPU, serving as a basic unit of the inference model. For example, _matrix multiplication_ is one of the most commonly used kernels in inference models. 

In the _scheduling_ stage, each _kernel_ is divided into several equally sized _blocks_ , with each _block_ ’s size referred to as the _blocksize_ . The GPU’s dispatch unit, implemented in hardware [3], will assign blocks to SM cores based on the availability of resources. 

In the _execution_ stage, each block is further divided into several equally sized _warps_ , i.e., a set of 32 threads executed together through single instruction multiple threads. A _warp_ is the basic unit of execution. The GPU’s _warp_ scheduler, also implemented in hardware [3], will select warps from blocks and start the execution. 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 04,2026 at 04:44:23 UTC from IEEE Xplore.  Restrictions apply. 

As the three decision-making stages throughout the execution process are distinct and the granularity of the scheduled objects gradually decreases, it is highly challenging to achieve the optimal resource-efficient scheduling [44]. 

## _B. Low Utilization of Low-Level Hardware_ 

To understand kernel’s use of low-level hardware, we select 10 commonly used models from MLPerf [42] and Pytorch benchmark [39], as shown in Table III. The maximum batch size for each model is determined under the condition that inference latency does not exceed 200 milliseconds (i.e., the common SLO for inference services in production environments [55]). All data are collected on NVIDIA A40 GPU. 

_**Observation #1**_ : _During the scheduling of stage #2, multiple blocks from the same kernel are stacked and co-located in SM cores. The next kernel’s blocks cannot execute until the current kernel nears completion._ 

To evaluate the scheduling of kernels, we select and launch two commonly used kernels in inference tasks: the _vectorized_ kernel (i.e., performs vectorized layer normalization, mainly using LDST hardware) and the _roll_ kernel (i.e., loops through the elements in the displacement array, mainly using INT32 hardware). These two kernels account for 28.90% of the total invocations among all open-source kernels, and we obtain the SM location and start time for each block by using CUDA inline assembly instructions [24] to read the GPU SM ID register and the GPU clock counter register, respectively. Figure 3(a) shows the scheduling results. We see that, although both _vectorized_ kernel and _roll_ kernel are launched concurrently on two CUDA streams, blocks of _vectorized_ kernels are scheduled for execution first, leaving _roll_ kernel’s blocks to wait until _vectorized_ kernel’s blocks are completed. 

**==> picture [244 x 93] intentionally omitted <==**

**----- Start of picture text -----**<br>
84 100<br>72 Roll Blocks (129024, 89.81%)<br>Vectorized Blocks<br>60 75<br>48<br>36 50<br>(129024 ,  38.15% )<br>24<br>25 max_batch<br>12<br>no batch<br>0<br>0 27 54 81<br>Start Time (μs) 0 4.5*1e7 9*1e7<br>SM ID<br>Kernel Threads CDF(%)<br>**----- End of picture text -----**<br>


Fig. 3: (a) Launch timing and SM locations for roll kernel’s blocks and vectorized kernel’s blocks, with roll kernel’s blocks executing after nearly all vectorized kernel’s blocks have completed. (b) Under max batch, only 38.15% of kernels have thread requirements less than one GPU, while the remaining kernels fully occupy the GPU. 

The reason for the above phenomenon is that after a kernel launch, the GPU’s dispatch unit schedules all blocks of a kernel to SM cores [22]. When the number of threads in a kernel exceeds the total thread capacity of the GPU (i.e., the NVIDIA A40 GPU has 129,024 threads), the blocks of a kernel occupy all SM cores. Figure 3(b) shows the statistical analysis of thread counts from 6,802 executions of kernels across the 10 models. Under max batch, 61.85% have more threads than the GPU can accommodate, and these kernels 

account for 70.83% of the total execution time. As a result, all blocks of a kernel are stacked and co-located within the SM cores, and the blocks of another kernel cannot begin execution until the current blocks are close to completion. 

_**Observation #2**_ : _During the execution of stage #3, the utilization of six hardware resources by the kernels follows a ”1 more, 5 less” pattern. Therefore, under the stacked scheduling of blocks, the overall hardware resource utilization within the SM cores becomes inefficient._ 

GPU resource utilization can be measured using two main methods (Figure 4(a)): (1) measure the actual usage time of GPU resources relative to the total time using the NVIDIASMI tool [34]; (2) measure the low-level hardware utilization of kernels during the execution using the Nsight Compute tool [33]. NVIDIA-SMI uses the active time ratio as the criterion for GPU utilization, even if only one thread in one SM is active, it will display 100% GPU utilization, significantly exaggerating the actual utilization. We analyze kernel-level resource utilization based on 6,802 executions using the above two tools: NVIDIA-SMI reports 81.16% utilization, while Nsight Compute reports only 9.28% for low-level hardware. 

**==> picture [246 x 81] intentionally omitted <==**

**----- Start of picture text -----**<br>
�� ����������������� 1 more<br>Warp scheduler 75 5 less<br>50<br>t1 t2 t3 NVIDIA-SMI<br>25<br>���� ���� ������ Nsight Compute<br>��� ����� ���� 0<br>Low-level Hardware 0 5 10 15 20<br>Kernel ID<br>Utilization (%)<br>**----- End of picture text -----**<br>


Fig. 4: (a) The schematic diagram of NVIDIA-SMI and Nsight Compute tools. (b) The utilization of the six hardware resources follows a “1 more, 5 less” pattern, where the “1 more” refers to the highest utilization among the six hardware types, and the “5 less” represents the average utilization of the remaining five. 

The utilization of six hardware resources by kernels follows a “1 more, 5 less” pattern. Figure 4(b) presents the hardware utilization of the top 20 most frequently executed kernels, totaling 6,063 executions. In the figure, red dots indicate the utilization of the primary hardware resource for each kernel, with an average utilization of 30.19%, while blue dots represent the average utilization of the remaining five hardware types, which is only 5.07%. As all blocks of a kernel have identical hardware resource demands, when a kernel runs on a dedicated SM under stacked co-location, five types of hardware resources typically exhibit low utilization. 

## _C. Blocksize Shaping Improves Utilization_ 

_**Observation #3**_ : _Blocksize shaping helps: Given that the NVIDIA GPU hardware scheduler is closed-source, shaping the blocksize is the only thing we can do to influence the colocating decisions. By setting the blocksize to half-plus of SM threads, it becomes indirectly possible for blocks to break the “stacked co-location” pattern._ 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 04,2026 at 04:44:23 UTC from IEEE Xplore.  Restrictions apply. 

As shown in Figure 2, the launching stage of kernels is controlled by the inference framework, where the blocksize parameter is exposed. However, the scheduling and execution stages are managed by closed-source GPU hardware scheduler. Therefore, we can only manipulate the launching stage to indirectly optimize scheduling and execution. 

When the remaining available thread capacity inside an SM is greater than the blocksize, the scheduler can schedule the block to the SM. To prevent multiple blocks of the same kernel from being co-located on the same SM core, we can set the blocksize to slightly more than half of the SM’s thread capacity (i.e., _half-plus_ ). This ensures that the combined thread count of any two blocks exceeds the maximum thread limit per SM core, thereby preventing multiple blocks of the same kernel from being scheduled to the same SM. 

**==> picture [247 x 84] intentionally omitted <==**

**----- Start of picture text -----**<br>
Vectorized kernel block Roll kernel block<br>Blocksize Blocksize Blocksize<br>1024 512 512<br>1024 512 1024<br>Blocksize<br>    Limit<br>0 100 200 300 0 100 200 300 0 100 200 300<br>Time ( μs ) Time ( μs ) Time ( μs )<br>SM<br>**----- End of picture text -----**<br>


Fig. 5: (a) When both kernels use large blocks: serial execution. (b) When both kernels use small blocks: serial execution. (c) When one kernel uses a large block and the other a small block: parallel execution. 

We analyze the execution process within an SM using the _roll_ and _vectorized_ kernels with different blocksize. As shown in Figure 5(a), when the blocksize of the two kernels is both set to exceed half of the thread capacity (i.e., _blocksize_ = 1024 _>_ 1536 _/_ 2, where the thread limit of an SM in NVIDIA A40 is 1536), after the threads of SM are allocated to the first block, the remaining threads cannot accommodate the second block, the blocks of the two kernels can only execute serially. In Figure 5(b), when the blocksize of the two kernels is set to less than half of the thread capacity (i.e., _blocksize_ = 512 _<_ 1536 _/_ 2), blocks of _vectorized_ are stacked and executed first; upon completion, the execution of _roll_ begins. However, in Figure 5(c), if one kernel is configured with a large blocksize (i.e., 1024) and the other with a small blocksize (i.e., 512), after placing the large block into an SM, only the small block can fit with the remaining threads, allowing blocks from different kernels to be co-located within the SM. This enables the SM to utilize multiple types of hardware resources simultaneously. 

In addition, CUDA allows configuring the amount of additional shared memory during kernel launch; however, our experiments show that large shared memory does not enable co-location of different kernels within the same SM. Hence, we focus solely on modifying the blocksize. 

_**Observation #4**_ : _Before a kernel is launched in stage #1, its blocksize is typically preset to a non-half-plus value. Since the available resources within SM cores fluctuate dynamically, this static setting can lead to inefficient use of resources._ 

**==> picture [244 x 93] intentionally omitted <==**

Fig. 6: (a) The blocksize of the roll kernel is statically preset to 512, which results in the highest resource efficiency for this kernel. (b)When the roll kernel is executed concurrently with a preceding vectorized kernel, the blocksize with the highest resource efficiency for the roll kernel changes to 1024. 

Existing inference frameworks use a statically preset method to set the blocksize of kernels. By default, typical frameworks like PyTorch [40], TVM [7] and TensorRT [50] use an “enumerate and choosing the best” policy to set the blocksize, i.e., evaluating the resource efficiency at various blocksize and choose the one with the highest efficiency. As a result, the blocksize is never set to a _half-plus_ value, as it is not divisible by the SM thread count and results in wasted threads during single-kernel execution. For example, as shown in Figure 6(a), PyTorch presets the blocksize of the roll kernel to 512, while the SM thread count is 1,536, achieving the highest throughput of 22,863. This is at least 1.39 times more efficient compared to other blocksize. 

However, the available resources within SM cores dynamically change as kernels are continuously scheduled and executed. The statically preset blocksize, initially designed for dedicated GPU usage, no longer achieves optimal throughput. For example, as shown in Figure 6(b), when the _roll_ kernel is co-located with a _vectorized_ kernel configured with blocksize 256, the blocksize of the _roll_ kernel that achieves the highest throughput shifts from 512 to 1,024, yielding at least a 1.98× improvement compared to other blocksize settings. Moreover, in production systems, the static setting of blocksize is susceptible to resource availability, which can prevent the system from achieving higher throughput. 

**==> picture [253 x 129] intentionally omitted <==**

**----- Start of picture text -----**<br>
We select three kernels<br>Rn E G<br>with different dominant<br>40 -40 hardware to co-locate with<br>other kernels. When the<br>20 -20 dominant resources differ,<br>the half-plus configuration<br>0 E H ReRo I GRn V L I GRn V E L H ReRo I V L H Re improves throughput by<br>Co-locate Kernel Name 19.94% (first 19 bars);<br>otherwise, throughput<br>Fig. 7: Throughput improve- decreases by 10.37%(last 4<br>ment via half-plus.<br>bars).<br>ent<br>m  via Half-plus(%)<br>Throughput Improve<br>**----- End of picture text -----**<br>


## _D. Implications_ 

In summary, we find that the low utilization of low-level hardware within SM cores is mainly due to two reasons: (1) the _stacked co-location_ of blocks from the same kernel ( **Observation 1** ) and (2) the _uneven utilization_ of resources 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 04,2026 at 04:44:23 UTC from IEEE Xplore.  Restrictions apply. 

by blocks ( **Observation 2** ). Existing works achieve intra-SM co-location through intrusive modifications to GPU hardware or kernel code, which are not supported in public cloud environments. 

To enable non-intrusive co-location of different kernels, we find that setting the blocksize to _half-plus_ of SM threads is a possible way ( **Observation 3** ). However, existing inference systems typically _statically preset_ the blocksize to a non- _halfplus_ value, which fails to improve resource efficiency under co-location scenarios ( **Observation 4** ). Therefore, we design a method to dynamically adjust the blocksize of kernels to enable co-location of different kernels at runtime. 

queued. In this way, the blocks with half-plus blocksize can be co-located with those with smaller blocksize for deployment (Observation #3). 

## _B. Kernel Profiler_ 

The _profiler_ mainly characterizes the resource demands of each kernel to facilitate the co-locating of kernels with complementary resource demands. To meet the latency SLO, it also profiles the launch time of each kernel when executed with the maximum batchsize allowed within the SLO. In particular, it runs the inference with the maximum batchsize in model exclusive GPU environment and records a 9-tuple for each kernel _k_ : 

III. DESIGN 

In this section, we present the design of _μShare_ . 

## _A. Overview_ 

**==> picture [243 x 129] intentionally omitted <==**

**----- Start of picture text -----**<br>
k1 <rfp32,�rfp64,�rint32,�rldst,�rsfu,�rtensor,�rmem,�rreg,�tlaunch><br>� �<br>Profiler k2 <rfp32,�rfp64,�rint32,�rldst,�rsfu,�rtensor,�rmem,�rreg,�tlaunch><br>Developer Deploy k3 <rfp32,�rfp64,�rint32,�rldst,�rsfu,�rtensor,�rmem,�rreg,�tlaunch><br>� SLO� �<br>GPU�SMs<br>Manager<br>Block�Shaper<br>Requsets Blocksize �<br>Pytorch Launch late�kernel �<br>Time<br>� Aware Launch SM�Hardware<br>Kernel� � Time � FP32 LDST Tensor<br>Intercepter normal�kernel INT32 FP64 SFU<br>**----- End of picture text -----**<br>


Fig. 8: The system architecture of _μShare_ . 

**The insight of** _**μShare**_ **is to adjust the blocksize of kernels to “half-plus” of SM thread capacity to achieve scattered co-location of kernels.** Figure 8 shows the overall architecture of _μShare_ . After an AI model is developed, the _profiler_ will analyze the model to find the maximum batchsize that meets the SLO (Service Level Objective). Then, under the maximum batchsize, it profiles each kernel’s low-level hardware usage as well as its shared memory and register consumption to support the co-location of kernels with complementary resource demands. 

After profiling, the model is deployed to the inference framework (e.g., Pytorch). When user requests arrive, the _batch manager_ batches multiple requests and sends the batch to PyTorch for execution. PyTorch then sequentially launches the kernels to the GPU for computation. To avoid intrusive modifications to Pytorch and CUDA, we design a _kernel interceptor_ which can intercept the launched kernel and send it to _shaper_ . For kernels with a late launch time that could potentially cause SLO violations, the _shaper_ modifies their blocksize parameters to _half-plus_ to accelerate computation. For other normal kernels, the _shaper_ does not modify their blocksize but adjusts their launch time for a _time-shifted launch_ . That is, according to kernel profiles, only kernels that are complementary in resource utilization to those currently executing on the GPU will be launched, while others will be 

**==> picture [266 x 23] intentionally omitted <==**

where _{rfp[k]_ 32 _[, r] fp[k]_ 64 _[, r] int[k]_ 32 _[, r] ldst[k][, r] sfu[k][, r] tensor[k][}]_[denotes][the] low-level hardware utilization during the execution of kernel _k_ , including FP32 core, FP64 core, INT32 core, LD/ST unit, SFU unit, and Tensor core, _{rmem[k]_[,] _[r] reg[k][}]_[denotes][the][shared] memory and register usage of kernel _k_ , respectively; and _t[k] launch_[denotes][its][launch][time.] 

We utilize NVIDIA’s Night Compute tool [33] to record the utilization rate of six low-level hardware during the execution of each kernel. Additionally, we utilize NVIDIA’s Night Systems tool [35] to record the kernel launch time. 

## _C. Kernel Interceptor_ 

When the inference requests begin execution, their kernels are launched by the inference framework. Then, _μShare_ intercepts the launched kernels through its _kernel interceptor_ without modifying the kernel code or GPU hardware scheduler. Hence, such a _non-intrusive_ design can significantly reduce development complexity. 

Since CUDA’s kernel launch functions (e.g., cudaLaunchKernel or cublasSgemm in cuBLAS) explicitly expose input parameters, and the addresses of their compiled dynamic link libraries (e.g., libcudart.so, libcublas.so, libcudnn.so) can be obtained easily, it is possible to interception kernels in a non-intrusive way. Specifically, the _kernel intercepter_ captures dynamic link libraries using the LD PRELOAD [27] and the dlopen and dlsym [13] functionalities provided by Unix systems. It creates functions with the same names as CUDA’s kernel launch functions. By using LD PRELOAD, these homonymous functions are loaded first. Then, dlopen and dlsym get the original addresses and input parameters (e.g., blocksize) of the CUDA’s kernel launch functions. This allows us to modify the input parameters before passing them back to the original functions, restoring their execution. 

Furthermore, CUDA’s kernel launch functions can be divided into two categories based on whether the kernel blocksize parameters can be modified. The first category is modifiable, such as the cudaLaunchKernel shown in Listing 1. All CUDA kernels launched using CUDA syntactic sugar <<<gridsize, blocksize, sharedMem, stream>>> are directed to this function. Among these, the blocksize and gridsize 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 04,2026 at 04:44:23 UTC from IEEE Xplore.  Restrictions apply. 

parameters are used for subsequent parameter modification operations, representing the number of threads within a modified block and the number of blocks, respectively. The parameters of this syntactic sugar correspond to the gridDim and blockDim in List 1. 

**extern** __host__ cudaError_t CUDARTAPI cudaLaunchKernel( **const void** *func, dim3 gridDim, dim3 blockDim, **void** **args, size_t sharedMem, cudaStream_t stream 

TABLE II: Statistics of blocksize kernels 

||**Name**<br>CUTLASS Gemm <br>CUDNN NCHW|**FP32**<br> 5.93 <br> 21.36|**FP64**<br> 0.00 <br> 0.00|**INT32**<br> 11.44 <br> 54.34|**LDST**<br> 17.51 <br> 48.78|**SFU**<br> 0.00 <br> 0.00|**Tensor**<br> 80.49<br>0.00|**Count**<br>1293<br>560|**Time(μs)**<br>223538<br>55438|
|---|---|---|---|---|---|---|---|---|---|
||BatchNorm|29.79|0.00|5.81|7.64|5.39|0.00|496|44829|
||CUBLAS Gemv|10.37|0.00|19.29|59.31|0.00|0.00|495|49980|
||MatMul|2.54|0.26|6.51|16.59|0.00|92.39|141|33665|
||CUDNN NHWC|7.59|0.00|14.53|55.10|0.00|0.00|116|9560|
||CUTLASS Comb|1.39|0.00|10.96|18.63|0.00|77.11|99|104|
||Conv|21.39|0.00|57.75|42.98|8.15|0.00|86|93|
||Other|–|–|–|–|–|–|4|107|
||**Total**|–|–|–|–|–|–|**3290**|**417314**|



); 

Listing 1: Blocksize modifiable launch function. 

The second category is unmodifiable, which utilize CUDA wrapper libraries (such as cuDNN or cuBLAS) for launching, like the cublasSgemm shown in Listing 2. These functions hide the blocksize parameters within closed-source code. Additionally, kernels that produce incorrect results after blocksize modification are also _unmodifiable_ (e.g., tiling-based kernels like Conv2d, which trigger a CUDA internal error when modified). 

**void** CUBLASWINAPI cublasSgemm( **char** transa, **char** transb, **void** **args ); 

Listing 2: Blocksize unmodifiable launch function. 

We analyze 67 kernels of 10 models (Table III) during inference, which are executed a total of 6,802 times. Among these, modifiable kernels are executed 3,512 times, accounting for 51.63% of the total executions, while unmodifiable kernels are executed 3,290 times, accounting for 48.37%. Table I and II provide a detailed breakdown of each kernel’s name, execution count, and percentage. Since modifiable kernels account for more than half of the total executions, there is significant potential for optimizing kernel co-location. 

TABLE I: Statistics of blocksize kernels 

|**Name**|**FP32 **|**FP64 **|**INT32 **|**LDST **|**SFU **|**Tensor **|**Count **|**Time(μs)**|
|---|---|---|---|---|---|---|---|---|
|RNN Cell<br>Vec Element<br>Elemwise|9.39<br>7.91<br>11.12|0.00<br>0.00<br> 0.00|12.64<br>9.29<br>38.12|16.05<br>13.06<br>17.18|6.96<br>4.08<br>0.00|0.00<br>0.00<br>0.00|1002<br>971<br>947|40378<br>159150<br>116440|
|Layer Norm|13.43|0.00|33.08|58.02|11.03|0.00|128|27497|
|Histo<br>Reduction<br>Roll|1.60<br>23.50 <br>20.82|0.00<br> 0.00<br> 0.00|4.36<br>57.43<br>33.25|2.39<br>32.29<br>12.47|0.00<br>0.00<br> 24.94|0.00<br>0.00<br>0.00|80<br>66<br>44|138<br>9974<br>14713|
|Indexed Elem|1.21|0.00|0.92|0.29|0.00|0.00|26|141|
|Other|–|–|–|–|–|–|248|2278|
|**Total**|–|–|–|–|–|–|**3512**|**370709**|



## _D. Shaper_ 

While it is not possible to develop a new co-locating policy under closed-source GPU hardware scheduler ( **Observation 3** ), the _shaper_ can indirectly influence the scheduling results 

by adjusting their _blocksize_ and the _relaunch time_ (i.e., the time when the _shaper_ relaunches the kernel to GPU after it has been intercepted.). 

Denote by _O_ the set of kernels intercepted by the _kernel interceptor_ . We divide the set _O_ into two subsets _X_ and _Y_ , such that _O_ = _X ∪Y_ and _X ∩Y_ = _∅_ , where _X_ refers to the set of kernels for which we are going to modify their _blocksize_ , and _Y_ refers to the remaining set of kernels for which we are going to modify their _relaunch time_ . To obtain the set _X_ , we calculate the kernel launch slack: 

**==> picture [200 x 14] intentionally omitted <==**

where _t[k] intercept_[is][the][real-time][launch][time][of][kernel] _[k]_[,][and] _t[k] launch_[is][the][profiled][launch][time][of][kernel] _[k]_[by][the] _[profiler]_ (Formula 1). We then sort these kernels in an ascending order of _s[k]_ . The first _x_ kernels in this sorted list are added to the set _X_ (i.e., _|X|_ = _x_ ), while the remaining kernels are added to the set _Y_ . Hence, kernels in set _X_ have tighter latency constraints and we need to reshape their blocksize and launch them immediately. 

**Half-plus Blocksize Shaping:** For a kernel _ki ∈ X_ , _∀i ∈_ [1 _, .., x_ ], the _shaper_ sets _ki_ ’s _blocksize_ to _half-plus_ : _thalf_ + _α_ , where _thalf_ denotes the half number of the SM thread capacity (e.g., _thalf_ = 768 in NVIDIA A40) and _α_ is a small positive integer. In this way, it is not possible to place more than one _ki_ ’s block in a single SM simultaneously. This enforces the distribution of _ki_ ’s blocks across different SMs, avoiding the “ _stacked co-location_ ” problem. Meanwhile, larger _blocksize_ improves kernel execution efficiency, reducing SLO violations. Note that, the value of _x_ is determined by the smallest _x_ such that the number of blocks of the first _x_ kernels exceeds the number of SMs. 

We consider two principles that guide our definition of _α_ : (1) The parameter _α_ should be able to reduce resource fragmentation. Since the default _blocksize_ is typically powers of 32, such as 32, 64, 128, 256, 512, or 1024, the value of _α_ should also follow this pattern. 

(2) The parameter _α_ should be able to reduce SLO violations. When the kernel launch time slack (i.e., _s[k]_ in Formula 2) is positive, we set _thalf_ + _α_ as the smallest number greater than half of SM thread capacity. For example, since the SM thread capacity is 1,536 in NVIDIA A40, we have _thalf_ + _α_ = 768 + 32 = 800 (note that 32 is the number of 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 04,2026 at 04:44:23 UTC from IEEE Xplore.  Restrictions apply. 

threads in a warp, which is the smallest unit of execution). When the kernel launch slack _s[k]_ is negative, we gradually increase _α_ by 32 to speed up the execution of the kernel. **Time-shifted Launch:** For a kernel _kj ∈ Y_ , _j ∈_ [ _x_ +1 _, .., x_ + _y_ ], the _block shaper_ sets _kj_ ’s _relaunch_ time and uses their default _blocksize_ . The reason is as follows: 

(1) Default _blocksize_ : The _blocksize_ of all kernels of models (Table III) ranges from 32 to 512, all of which are less than half of the SM thread capacity (Figure 9), making them easily co-locatable with the _half-plus_ kernels in set _X_ within the same SM core. Therefore, the _blocksize_ of _kj ∈ Y_ does not need to be 

(2) _Relaunch time_ : Kernel _kj_ is relaunched directly if both of the following conditions hold: For each of the six hardware resource types, the combined utilization of _ki_ and _kj_ does not exceed 100%, and available shared memory, registers (i.e., the limiting factors for GPU scheduling other than blocksize, profiled in Formula 1) are sufficient. 

Otherwise, _kj_ waits for _β_ microseconds for a _time-shifted launch_ before rechecking the above conditions, and its kernel launch slack is updated according to Formula 2. After updating the slack values and reordering all kernels in the set _O_ based on the updated slack, if _kj_ moves into the top-x positions of the list, its _blocksize_ will be set to _half-plus_ , like the other kernels in _X_ , and it will be immediately launched for execution. This process is repeated until all kernels have been launched. 

**==> picture [247 x 93] intentionally omitted <==**

**----- Start of picture text -----**<br>
512 8<br>SLO violation<br>slow<br>256 6 i n cr e a s e rapid<br>decline<br>128<br>6432 4 b [max] / n<br>0 1 2 3 4 5 6 7 8 9 10<br>Time Window Index<br>QPS<br>Blocksize<br>spm v gemmcutlk genvve_elsoftm in_elel_keve_lanchw nhwc elem co_ofcat roll<br>**----- End of picture text -----**<br>


Fig. 9: The default blocksize Fig. 10: The control process of kernels is less than half of of batch size after each time SM threads. window. 

## _E. Batch Manager_ 

The _batch manager_ is responsible for managing the _batch size_ , which determines how many user requests the system can process in a single inference [15]. Aggregating user requests into a large batch can improve system throughput. However, when the request arrival rate is low, the waiting time required to accumulate a large batch may lead to SLO (Service Level Objective) violations. In addition, when multiple models share a GPU, resource contention between models may increase latency, further risking SLO violations [8], [9]. Therefore, it is necessary to adjust the batch size in real time based on both interference and request frequency. 

We adopt a _feedback-based_ strategy for adjusting the _batchsize_ , i.e., adjusting the _batchsize_ based on the monitored realtime latency. If the monitored latency is less than the SLO, increase the _batchsize_ . Otherwise, decrease the _batchsize_ . To avoid oscillations caused by short-term workload bursts, we 

monitor the response latency over a time window and employ an _exponential decay_ algorithm [12] to derive the SLO slack (denoted by _[←→] sj_ ) within the window _j_ : 

**==> picture [109 x 29] intentionally omitted <==**

where _nj_ refers to the number of requests in window _j_ , _tSLO_ is the latency target defined in SLO, and _ti_ is the actual response latency for request _i_ . Note that _i_ is the index of the requests sorted in reverse chronological order. 

Since the inference time is positively correlated with the _batchsize_ , we set the initial _batchsize_ of a model to _b[max] /n_ , where _n_ is the number of co-locating models on the current GPU, and _b[max]_ is the maximum _batchsize_ that satisfies the SLO for that model on the current GPU. Whenever a time window _j_ ends, _μShare_ updates _[←→] sj_ and adjusts the _batchsize_ accordingly. To ensure that the SLO is not violated as much as possible, the principle of adjusting the _batchsize_ is to be conservative when increasing it, but aggressive when decreasing it (Figure 10). 

When the SLO slack _[←→] sj_ is positive, increase the _batchsize_ used in the next window _j_ +1 (denoted by _bj_ +1) linearly. That is, 

**==> picture [83 x 14] intentionally omitted <==**

where _k_ is a positive coefficient. 

When the SLO slack _[←→] sj_ is negative, decrease the _batchsize_ used in the next window _j_ + 1 exponentially. That is, _bj_ +1 = _max{bj − e[λ][×←] s→j ,_ 1 _}_ 

where _λ_ is a negative coefficient. 

## IV. IMPLEMENTATION 

## _A. Overall Implementation_ 

_μShare_ uses PyTorch [40] as the GPU inference framework. The components of _μShare_ are compiled as shared libraries (i.e., .so files) and loaded into the PyTorch process using the LD_PRELOAD environment variable to interact with it. To support custom kernel blocksize, we set PyTorch’s blocksize limit (i.e., defined in the C10_LAUNCH_BOUNDS(blocksize) macro) to be consistent with the CUDA limit, i.e., 1024. 

In particular, the _kernel interceptor_ uses the dlopen() function from the libdl [13] library to open CUDA’s dynamic link libraries (e.g., CUDA’s libcublas.so) at runtime and return handles to the CUDA’s kernel launch functions. It then uses the dlsym() function from the libdl library to obtain the addresses of the library functions and their input parameters based on the handles. The _block shaper_ uses the shm_open() function from the libc [16] library to create a shared memory area. The _block shaper_ exposes an interface to the _kernel interceptor_ , with kernel_process() used for uploading parameters such as kernel blocksize. This interface uses the mmap() function from the libc library to map and modify values in shared memory. After modifying the uploaded kernel parameters, the _block shaper_ returns the 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 04,2026 at 04:44:23 UTC from IEEE Xplore.  Restrictions apply. 

parameters to the address of the shared library function obtained by the dlsym() function, restoring kernel execution. 

## _B. Support for Different NVIDIA GPUs_ 

_μShare_ provides support for different NVIDIA GPUs. Since the number of threads per SM affects the blocksize configuration strategy, _μShare_ categorizes recent GPUs into two types: **(1) GPUs with 1,536 threads per SM:** Such as NVIDIA A40, RTX 4090, and RTX 3080 Ti. Under the CUDA constraint, where the maximum blocksize is 1024, a half-plus block can be deployed on these GPUs with 1,536 threads per SM. Moreover, setting the blocksize as a multiple of 32 (i.e., the number of threads in a warp) can avoid thread resource fragmentation. Therefore, the range of the half-plus blocksize _b_ is given by: _{b |_ 1536 _/_ 2 _< b ≤_ 1024 _, b ≡_ 0 (mod 32) _}_ . For example, the minimum blocksize is 800. 

To determine a specific blocksize _b_ , the _block shaper_ reads the kernel’s launch slack value from shared memory using the mmap() function. If the slack value is negative, _b_ is set to the previous kernel’s blocksize incremented by 32; otherwise, it defaults to 800. The _block shaper_ then injects _b_ into the kernel’s launch address intercepted using the dlsym() function, thereby resuming kernel execution. 

**(2) GPUs with 2,048 threads per SM:** Such as NVIDIA A100, A800, and H200. The half-plus strategy is not suitable for these GPUs. Even with a maximum blocksize of 1024, the CUDA scheduler still performs stacked co-location of two large blocks within the SM cores. 

In this case, the optimal configuration is to set the blocksize to 1/3-plus of 2,048. This allows the CUDA scheduler to deploy two large 1/3-plus blocks from the same kernel within one SM. Subsequently, since the remaining threads in the SM are less than 1/3 of 2,048, it is not possible to deploy another large block. These remaining threads can be allocated to small blocks from another kernel with complementary hardware resource demands, thereby achieving scattered co-location. 

Thus, the range of the 1/3-plus blocksize _b_ is given by: _{b |_ 2048 _/_ 3 _< b ≤_ 1024 _, b ≡_ 0 (mod 32) _}_ . For example, the minimum blocksize is 704. After determining the range of blocksize, the subsequent operations are the same as in the previous case. 

## V. EVALUATION 

## _A. Methodology_ 

**Testbed:** We deploy _μShare_ across eight servers, each equipped with an Intel Xeon Gold 6338 CPU and either an NVIDIA A40 or A800 GPU, due to the different SM thread limits of these GPUs. Each CPU has 128 logical cores, with a base frequency of 2.00 GHz and a maximum frequency of 3.20 GHz, along with 251GB of server memory. We utilize PyTorch 2.2.0 as the inference framework. 

(1) NVIDIA A40 GPU: The NVIDIA A40 GPU has 84 SMs and 44.784GB of memory, each SM has 1536 threads, 102,400 bytes of shared memory, and 65,536 registers. The CUDA version is 11.8. 

TABLE III: The benchmark models. 

||**Model Name**|**max**<br>**batch **|**Architecture **|**Weight(MB) **|**App Field**|
|---|---|---|---|---|---|
||Llama2-7b|14|LLM|14,336|Txt Proc|
||GPT-2|50|LLM|548|Txt Proc|
||Bert|46|Transformer|507.44|Txt Proc|
||ResNet50-v1.5|295|CNN|97.71|Img Class|
||MobileNet<br>v2<br>Swin. Transf.<br>Vis. Transf.<br>Yolostiny|427<br>8<br>77<br>295|CNN<br>Transformer<br>Transformer<br>CNN|13.54<br>331.49<br>327.37<br>24.82|Obj Det<br>Img Class<br>Img Class<br>Obj Det|
||Resnet101<br>EffcientNet<br>B7|199<br>93|CNN<br>CNN|170.58<br>254.68|Obj Det<br>Img Class|



(2) NVIDIA A800 GPU: The NVIDIA A800 GPU has 108 SMs and 80GB of memory, each SM has 2048 threads, 167,936 bytes of shared memory, and 65,536 registers. The CUDA version is 12.1. 

**Workloads:** We select 10 commonly used models from MLPerf [42] and Pytorch benchmark [39] as shown in Table III. The selected models cover CNN, Transformer, RNN, as well as large language models (LLMs). Following the common setting in production environments [55], the SLO for inference models is set to 200ms. For Llama2-7b, which has a relatively long execution time, the request SLO is set to 400ms, and both the input and output lengths of the LLM are fixed to ten tokens. 

We use Azure’s inference trace (Fig. 11) from INFless [55] and scale it according to the number of GPUs used at runtime. During inference, each model runs 4 replicas, for a total of 40 replicas distributed across eight GPUs. 

**==> picture [127 x 93] intentionally omitted <==**

Fig. 11: The ten production **Comparison Systems:** We trace examples. compare _μShare_ against the state-of-the-art systems including _INFless_ and _Orion_ : _Orion_ [46]: _Orion_ co-locates kernels with different computational and memory resource demands on the GPU by controlling their launch time. 

_INFless_ [55]: _INFless_ profiles the resource capacity requirements of models, and co-locates models within SMs and memory that can be accommodated by the GPU capacity. **Parameter Configuration:** _μShare_ sets three parameters _k_ , _λ_ , and _β_ , where _k_ is the linear trend increase parameter of the batch size, _λ_ is the exponential trend decrease parameter of the batch size, and _β_ is the interval parameter for delaying kernel launches. These parameters can be customized before the system runs. Through multiple experiments, we determine that the optimal values for ensuring SLO guarantees are _k_ = 0 _._ 05, _λ_ = _−_ 0 _._ 1, and _β_ = 10. 

## _B. Throughput Evaluation_ 

**High Throughput:** _μShare increases the system throughput by 26.90%-54.09%._ We compare the throughput of _μShare_ , _INFless_ , and _Orion_ in co-located scenarios, including the system throughput comparison (Figure 12(a)), and the normalized system throughput comparison (Figure 12(b)). The normalized 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 04,2026 at 04:44:23 UTC from IEEE Xplore.  Restrictions apply. 

throughput is the throughput of each model divided by the unit batch size. The unit batch size is the batch size under the model’s use of MPS (Multi-Process Service) [32] and the memory control interface of PyTorch to evenly distribute the unit’s SM and memory resources of the GPU. _μShare_ has the highest throughput in all scenarios. 

**==> picture [244 x 92] intentionally omitted <==**

Fig. 12: (a) Actual throughput comparison. (b) Normalized throughput comparison. 

Compared to _INFless_ , _μShare_ improves the throughput of each model by 15.02%-66.59%. The throughput of the two systems is 716.53 and 564.28, with peak throughput values of 3046 and 1722, respectively, and the normalized throughput of the two systems is 58.91 and 46.42, respectively. _μShare_ shows a 26.90% improvement over _INFless_ . The reason for the higher throughput of _μShare_ is that it can achieve scattered colocation of kernels with different low-level hardware requirements within an SM, reducing idle hardware within the SM and thus improving resource efficiency. In contrast, _INFless_ adopts stacked co-location and cannot effectively utilize the various hardware resources within the SM. 

Compared to _Orion_ , _μShare_ improves the throughput of each model by 21.65%-92.66%. The system throughput of _Orion_ is 464, with peak throughput value of 1192, and the normalized throughput is 38.23. _μShare_ shows a 54.09% improvement over _Orion_ . Although _Orion_ is a kernel-level inference system, it adopts stacked co-location and cannot fully utilize the lowlevel hardware resources within the SM. Moreover, _Orion_ uses a more conservative co-location strategy, allowing at most one compute-intensive kernel and one memory-intensive kernel to be co-located on the GPU to strictly control interference. 

**==> picture [244 x 92] intentionally omitted <==**

Fig. 13: The normalized throughput at different proportions of unmodifiable kernels on (a) A40 GPU and (b) A800 GPU. 

**Proportion of Unmodifiable Kernels:** _The throughput of_ μShare _increases as the proportion of unmodifiable kernels decrease._ By default, the proportion of unmodifiable kernels across 10 models is 48.37%, while modifiable kernels account for 51.63% (Section 3.3). To analyze the impact of the proportion of unmodifiable kernels on throughput, we 

control the number of modifiable kernels by modifying only 0, 1, 2, 3, 4, 5 out of every 5 modifiable kernels, with the remaining kernels considered unmodifiable. This gradually reduces the proportion of unmodifiable kernels from 100% to 89.67%, 79.35%, 69.02%, 58.70%, and 48.37%, and _μShare_ ’s throughput increases from 47.59 to 48.23, 51.42, 52.79, 54.42 and 58.81 (Figure 13(a)). Therefore, the throughput of _μShare_ increases as the proportion of unmodifiable kernels decreases, and even in the worst-case scenario where all kernels are unmodifiable, _μShare_ ’s performance only falls back to kernellevel co-location based on resource coupling, which is equivalent to _INFless_ throughput and higher than that of _Orion_ . **Compare Intra-SM Co-location Technique:** _Tacker_ [62] is an intra-SM co-location system based on kernel fusion. It provides fused ResNet50 and BERT models. Since _Tacker_ does not support SLO management across multiple inference models during co-location, for fairness we also disable _μShare_ ’s latency management mechanism. 

Under inference trace 11, _μShare_ reduces the end-to-end latency of Bert and Resnet50 by 24.71% and 16.00%, respectively (Figure 14), achieving an overall throughput improvement of 20.38%. Because kernel fusion approaches merge only adjacent intra-model kernels, whereas _μShare_ co-locates kernels across tasks, providing more co-location opportunities. 

**==> picture [246 x 92] intentionally omitted <==**

Fig. 14: Latency comparison Fig. 15: Throughput compare with intra-SM co-location. with cuda graph technique. 

**Compare CUDA Graph:** CUDA Graph fuses multiple kernels into a single graph for unified launch, which is beneficial when launch overhead is significant. However, in large-batch or colocation scenarios, the kernel execution time overshadows the launch time, so CUDA Graphs have little impact in our case. 

We compare _μShare_ with _INFless_ after adding CUDA Graph optimization, as is shown in Figure 15, where _INFless_ itself improves throughput by only 2.97%, while _μShare_ still achieves a throughput improvement of 26.44%. 

## _C. SLO Guarantee_ 

**Low SLO Violation:** _μShare can guarantee the SLO at most time. The SLO violation rate is as low as 3.35%._ We compare the ability of three systems to guarantee the SLO of all models by repeating the experiment 20 times on NVIDIA A40 GPU for each system. The violation ratios of each model are displayed in a box plot in Figure 16. Among them, the average violation rates for the 10 models of _μShare_ , _INFless_ , and _Orion_ are 3.35%, 2.05%, and 1.12%, respectively. Compared to existing systems, _μShare_ ’s SLO violation only increases by 1.30%-2.23%, but it achieves a throughput improvement of 26.90%-54.09% (Evaluation 5.2). 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 04,2026 at 04:44:23 UTC from IEEE Xplore.  Restrictions apply. 

**==> picture [247 x 82] intentionally omitted <==**

**----- Start of picture text -----**<br>
20<br>μShare Orion INFless<br>10<br>0<br>MobilenetLlama2-7bVis. Transf.Resnet50YolostinyBertSwin. Transf.Resnet101EfficientNetGPT-2<br>SLO violation (%)<br>**----- End of picture text -----**<br>


Fig. 16: SLO violation comparison of _μShare_ with baselines. 

_μShare_ designs a dual-SLO guarantee mechanism for inference requests to kernels. On the inference request side, it uses SLO-based feedback control of batch size to avoid input inference requests exceeding the system load. On the kernel side, it uses feedback control based on kernel launch time to adjust the number of SM threads for each kernel, accelerating kernel execution to ensure SLO. In contrast, although _INFless_ and _Orion_ can adjust resource allocation based on SLO violation feedback after inference requests are completed, they cannot perform SLO detection during kernel execution, and therefore sacrifice some throughput to ensure a higher SLO satisfaction rate. 

**Trade-off between Throughput and SLO:** _μShare_ can trade off throughput and SLO violation by tuning the Batch Manager’s hyperparameters _k_ and _λ_ . We evaluate nine configurations combining _k_ = _{_ 0 _._ 05 _,_ 0 _._ 03 _,_ 0 _._ 01 _}_ and _λ_ = _{−_ 0 _._ 1 _, −_ 0 _._ 15 _, −_ 0 _._ 2 _}_ . Specifically, _μShare_ v1–v3 correspond to _λ_ = _−_ 0 _._ 1 with _k_ = _{_ 0 _._ 05 _,_ 0 _._ 03 _,_ 0 _._ 01 _}_ , _μShare_ v4–v6 to _λ_ = _−_ 0 _._ 15, and _μShare_ v7–v9 to _λ_ = _−_ 0 _._ 2. 

The throughput of _μShare_ v1–v9 gradually decreases from 58.91 to 53.64 (Figure 17), while the SLO violation rate drops from 3.35% to 0.63% (Figure 18). At _μShare_ v7 ( _k_ = 0.05, _λ_ = –0.2), _μShare_ achieves an SLO violation rate of 0.84%—lower than those of the baseline systems (2.05% and 1.12%)—while maintaining a throughput improvement of 19.28%–44.83%. 

**==> picture [247 x 65] intentionally omitted <==**

Fig. 17: Throughput under different hyperparameter settings. 

**==> picture [243 x 79] intentionally omitted <==**

**----- Start of picture text -----**<br>
20Orion INFless20 μShare_v1 20 v2 v3 v420 v5 v6 20v7 v8 v9<br>10 10 10 10 10<br>Mobilenet Llama2-7b Vis. Transf. Resnet50 Yolostiny<br>20 20 20 20 20<br>10 10 10 10 10<br>       SLO violation (%) Bert Swin. Transf. Resnet101 EfficientNet GPT-2<br>**----- End of picture text -----**<br>


Fig. 18: SLO violation under different hyperparameter settings. 

## _D. Latency_ 

The end-to-end inference latency comparison between _μShare_ v7 ( _k_ = 0.05, _λ_ = –0.2) and the two baselines is shown in Figure 19. _μShare_ reduces average latency by 25.72%–29.53% and 99th-percentile latency by 25.33%–31.31%. 

The lower latency of _μShare_ results from intra-SM kernel parallelism, which effectively improves execution efficiency and kernel-level concurrency. In contrast, _INFless_ exhibits higher execution latency because it lacks kernel-level scheduling for efficiency optimization, while _Orion_ ’s conservative co-location strategy limits the number of concurrent kernels, leading to increased request queuing latency. 

**==> picture [243 x 79] intentionally omitted <==**

**----- Start of picture text -----**<br>
μShare Orion INFless<br>1 1 1 1 1<br>0 100 200 0 200 4 00 0 5 0 100 0 100 200 0 100 200<br>1 Mobilenet 1 Llama2-7b 1 Vis. Transf. 1 Resnet50 1 Yolostiny<br>0 100 1 5 0 0 100 200 0 100 200 0 100 200 0 1 5 0 200<br>Bert Swin. Transf. Resnet101 EfficientNet GPT-2<br>CDF<br>CDF<br>**----- End of picture text -----**<br>


Fig. 19: Latency comparison of _μShare_ and baselines. 

## _E. Low-level Hardware Utilization_ 

**High Hardware Utilization:** _μShare increases the average low-level hardward utilization by 38.53%-61.15%._ We compare the low-level hardware utilization of _μShare_ , _INFless_ , and _Orion_ in co-located scenarios and visualize the intraSM hardware utilization within a 200ms window of the same execution stage, as shown in Figure 20. To measure hardware utilization under co-location, we first use the NVIDIA Nsight Systems tool to record the execution time and parameters of all concurrently running kernels. We then use the NVIDIA Nsight Compute tool to individually measure the utilization of six SM hardware resources for each kernel under co-located execution. Finally, for each time interval, we aggregate the utilization of all kernels active during that interval. 

The average utilization of the six low-level hardware components under _μShare_ , _INFless_ , and _Orion_ is 15.10%, 10.90%, and 9.37%, respectively. _μShare_ achieves a 38.53%–61.15% improvement over two baselines. This improvement arises because scattered co-locating enables different blocks to execute concurrently within the same SM. 

## _F. Performance on A800 GPU_ 

**High Throughput:** _μShare improves system throughput by 16.45%-52.29% on NVIDIA A800 GPUs while guaranteeing the SLO._ We compare the throughput of _μShare_ , _INFless_ , and _Orion_ on A800 GPU. The normalized throughputs of _μShare_ and _INFless_ and _Orion_ are 99.39 and 85.35 and 65.26 (Figure 21(a)), respectively. _μShare_ improves system normalized throughput by 16.45%-52.29%. Additionally, as the proportion of unmodifiable kernels decreases from 100% to the default 48.37%, the throughput of _μShare_ improves from 86.13 to 98.50 (Figure 13(b)). And the average violation rates across _μShare_ , _INFless_ , and _Orion_ are 3.40%, 2.29%, and 1.42% (Figure 21(b)), respectively. _μShare_ shows only a 1.11%-1.98% increase in SLO violations but achieves a throughput improvement of 16.45%-52.29%. 

In addition, the improvement in _μShare_ on the A800 GPU is slightly smaller than on the A40 GPU. This is because the A800 uses 1/3-plus shaping compared to the half-plus 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 04,2026 at 04:44:23 UTC from IEEE Xplore.  Restrictions apply. 

**==> picture [504 x 217] intentionally omitted <==**

**----- Start of picture text -----**<br>
LDST SFU ALU FP32 FP64 Tensor<br>10 [2]<br>μShare avg _ util: 15.1% I NFless avg_util: 1 0.9% Ori o n avg_uti l: 9.37%<br>10 [1]<br>10 [0]<br>0 40 80 120 160 200 0 40 80 120 160 200 0 40 80 120 160 200<br>μShare Timeline (ms) INFless Timeline (ms) Orion Timeline (ms)<br>Fig. 20: Comparison of six low-level hardware utilization timelines between μShare and baselines.<br>10.0 blocksize of any kernels, the system throughput decreases<br>7.5 30.95% in Figure 22(a) and the SLO violation rate increases<br>by 6.33% in Figure 22(b). This is because the system shifting<br>5.0 from kernel-level scattered co-location to kernel-level stacked<br>2.5 co-location, leading to inefficient utilization of intra-SM hard-<br>ware resources. Furthermore, when μShare no longer adjusts<br>0.0 the batch size of inference requests, the system throughput<br>INFless Orion μShare<br>Hardware Utilization (%)<br>SLO Violation Rate<br>**----- End of picture text -----**<br>


blocksize of any kernels, the system throughput decreases by 30.95% in Figure 22(a) and the SLO violation rate increases by 6.33% in Figure 22(b). This is because the system shifting from kernel-level scattered co-location to kernel-level stacked co-location, leading to inefficient utilization of intra-SM hardware resources. Furthermore, when _μShare_ no longer adjusts the batch size of inference requests, the system throughput increases by 10.67% while the SLO violation rate increases by 21.90%. This is because the system loses the ability to adapt batch sizes based on inference latency feedback, resulting in a sharp throughput increase when the input load exceeds system capacity, but at the cost of significantly higher SLO violations. 

Fig. 21: (a) Normalized throughput on NVIDIA A800. (b)SLO violation on NVIDIA A800. 

shaping of the A40, which increases the upper limit of blocks utilizing the same resources within a single SM core from 1/2 to 2/3 (i.e., two 1/3-plus blocks per SM), which may lead to slightly unbalanced SM thread allocation, resulting in lower improvement on the A800 compared to the A40. 

## _H. Co-locating Scientific Computing Workloads_ 

**Broad Applicability:** _μShare supports all applications that execute workloads as CUDA kernels._ With the rise of AI for science (co-locating scientific computing, inference, and training workloads) [26], [28], LLM+RAG (co-locating LLM inference and RAG) [23], [41], and other technologies [17], [58], the variety of applications running in datacenters has become increasingly diverse. As a result, we evaluate _μShare_ by co-locating five scientific computing applications from the Parboil benchmark [47] with five inference models listed in Table III. Unlike inference models deployed as services, scientific computing applications are compiled into binary executables. We randomly select and execute these executables, allowing the same application to be invoked multiple times. As shown in Figure 23, _μShare_ improves the overall system throughput by 18.18%–28.62%. 

## _G. Performance Analysis Breakdown_ 

**Shaping Improvement:** _Half-plus blocksize Shaping can effectively improve the throughput by 19.40% while reducing SLO violation rate._ We analyze the impact of the _μShare_ system’s multiple components on throughput and SLO. We set up four breakdown scenarios: (1) _μShare_ , where blocksize dynamically adjusts according to kernel launch time. (2) _μShare shape 1024_ , where blocksize is set to a fixed value, e.g., 1024. (3) _μShare w/o shape_ , where blocksize is no longer adjusted. (4) _μShare w/o batch_ , where inference task’s batch size is no longer adjusted based on latency feedback. 

**==> picture [119 x 93] intentionally omitted <==**

**----- Start of picture text -----**<br>
30<br>20<br>10<br>0<br> w/o qps w/o shapeshape_1024μShare<br>SLO Violation Rate<br>**----- End of picture text -----**<br>


**==> picture [119 x 93] intentionally omitted <==**

**==> picture [253 x 103] intentionally omitted <==**

**----- Start of picture text -----**<br>
3<br>10<br>μShare Orion INFless<br>2<br>10<br>1<br>10<br>Fig. 23: Throughput evaluation under co-location scientific<br>computing and inference applications.<br>MobilenetVis. Transf.Resnet50YolostinyBert Bfs Cutcp Histo Lbm Mri-g<br>Throughput<br>**----- End of picture text -----**<br>


Fig. 22: Breakdown analysis of the _μShare_ : impact on (a) throughput and (b) SLO violation rate. 

These improvements are achieved because scientific computing applications typically utilize FP64 cores, whereas inference applications mainly rely on FP32 cores, LDST units, and Tensor Cores. By co-locating kernels with complementary hardware demands within the same SM, _μShare_ improves hardware utilization and enhances overall system throughput. 

When _μShare_ fixes the blocksize of all modifiable kernels at 1024, the system throughput decreases by 3.36% in Figure 22(a) and the SLO violation rate increases by 1.32% in Figure 22(b). This is because fixing the blocksize prevents further optimization of SLO through adjusting the number of threads for the kernels. In contrast, when _μShare_ does not modify the 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 04,2026 at 04:44:23 UTC from IEEE Xplore.  Restrictions apply. 

## _I. Kernels and Utilization in SM at Runtime_ 

The co-located kernels within each SM and the dominant utilization of two kernels in _μShare_ (two solid lines) and one kernel in INFless (one dashed line) are shown in Fig. 24. _μShare_ significantly improves hardware utilization. 

**==> picture [247 x 81] intentionally omitted <==**

Fig. 24: Kernels and utilization in SM at runtime. 

## _J. Overhead_ 

**Low Overhead:** We measure the online execution overhead of _μShare_ . The resource consumption of the system is shown in Figure 26 revealing that the CPU overhead during runtime is a mere 6.85% of a single core, suggesting an exceedingly low resource overhead. Additionally, we employ shared memory to enhance communication between the inference and control processes, the average system control overhead for a single kernel is merely 60.35 nanoseconds, as illustrated in Figure 25. Moreover, the offline profiling cost for each model ranges from 105 to 393 seconds. For Llama2-7b, which contains significantly more kernels, the profiling time is 7,160 seconds. 

**==> picture [246 x 92] intentionally omitted <==**

Fig. 25: The average processFig. 26: The overhead during ing time for each kernel. system runtime. 

## VI. RELATED WORK 

**GPU Spatial Sharing:** Spatial sharing enhances throughput by enabling multitasking on shared GPUs: NVIDIA’s MultiInstance GPU (MIG) [30], Multi-Process Service (MPS) [32] and AMD CU masking [1] divide a single GPU into multiple partitions to serve several tasks. Orion [46] reduces interference by coupling the execution of SM-intensive and memory-intensive kernels. INFless [55] minimizes resource fragmentation through uneven allocation of SM and memory. INFless and Batchmaker [15] dynamically adjust batch sizes to guarantee latency. REEF [21] and LAX [56] ensure performance through dynamic resource allocation at runtime. Baymax [6], Prophet [5], and Gpulet [10] perform scheduling optimization by predicting QoS or interference. GPU spatial sharing techniques can improve the utilization of SMs and memory. However, they still face the problem of stacked colocation, which cannot effectively increase the utilization of hardware resources within the SM. 

**GPU Temporal Sharing:** Temporal sharing enhance throughput by switching tasks across GPU time slices: AntMan [54] improves utilization through dynamic management of memory and compute units. IADeep [9] co-optimizes task assignments and interference mitigation. Clockwork [19] avoids long-tail delays by calculating task execution time. PipeSwitch [2] achieves efficient pipeline execution through concurrent loading and inference of model layers. However, GPU temporal sharing can only execute one kernel per unit of time, and a single kernel typically cannot fully utilize both SM and memory resources, resulting in underutilization of the GPU. **Intra-SM Sharing:** Intra-SM sharing techniques enable multiple kernels to share the same SM core, fall into two primary categories: intrusive kernel modifications and intrusive hardware modifications. Intrusive kernel modifications include kernel fusion and persistent kernel. Kernel fusion, such as Tacker [62], T3 [37], Rammer [29], COMBO [4], and SpDNNs [14], merges the code of multiple kernels with different hardware resource demands into a single new kernel. Persistent kernel, such as ISPA [61], Plasticine [63], and Elastic kernel [36], launches an empty non-terminating kernel, and user kernels are then submitted to this non-terminating kernel for execution. Intrusive hardware modifications, such as CCWS [43], Prema [11], and PriorityRR [38], redesign the GPU to expose kernel scheduling interfaces and are validated through simulators. However, due to the closed-source nature of Nvidia GPU and the prohibition of accessing and modifying user code on public cloud platforms, the application scenarios of intrusive modifications are severely limited. 

**Non-SM Resource Sharing:** Resources shared across SMs include memory bandwidth and the L2 cache. SGDRC [60] reverse-engineers GPU drivers to identify VRAM channel mappings and places data from different kernels into specific channels, achieving memory bandwidth isolation. This work is orthogonal to our intra-SM co-location approach. 

## VII. CONCLUSIONS 

In this paper, we analyze the semantic gap between the resource demands of kernels and the resource allocation performed by the NVIDIA GPU hardware scheduler. This gap leads to stacked co-location of kernels and results in low utilization of low-level hardware resources. Without intrusive modifications to GPU, we propose a hardware–software codesign approach, _half-plus blocksize shaping_ , which achieves scattered co-location of kernels. Building on this concept, we construct the _μShare_ system, which effectively improves GPU resource efficiency across various NVIDIA GPUs and diverse co-location scenarios. 

## VIII. ACKNOWLEDGMENTS 

This work is supported by the National Key Research and Development Program of China under Grant No.2024YFB4505204, the National Natural Science Foundation of China under Grant 62372322, 62432015, and Tianjin Science and Technology Plan Project 24ZXKJGX00060. 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 04,2026 at 04:44:23 UTC from IEEE Xplore.  Restrictions apply. 

## REFERENCES 

- [1] “Setting the number of compute units,” https://rocm.docs.amd.com/en/ latest/how-to/setting-cus.html/, 2020. 

- [2] Z. Bai, Z. Zhang, Y. Zhu, and X. Jin, “PipeSwitch: Fast pipelined context switching for deep learning applications,” in _14th USENIX Symposium on Operating Systems Design and Implementation (OSDI 20)_ . USENIX Association, Nov. 2020, pp. 499–514. [Online]. Available: https://www.usenix.org/conference/osdi20/presentation/bai 

- [3] J. Bakita and J. H. Anderson, “Demystifying nvidia gpu internals to enable reliable gpu management,” in _2024 IEEE 30th Real-Time and Embedded Technology and Applications Symposium (RTAS)_ , 2024, pp. 294–305. 

- [4] B. Chen, H. Zhao, W. Cui, Y. He, S. Zhang, Q. Chen, Z. Li, and M. Guo, “Maximizing the utilization of gpus used by cloud gaming through adaptive co-location with combo,” in _Proceedings of the 2023 ACM Symposium on Cloud Computing_ , ser. SoCC ’23. New York, NY, USA: Association for Computing Machinery, 2023, p. 265–280. [Online]. Available: https://doi.org/10.1145/3620678.3624660 

- [5] Q. Chen, H. Yang, M. Guo, R. S. Kannan, J. Mars, and L. Tang, “Prophet: Precise qos prediction on non-preemptive accelerators to improve utilization in warehouse-scale computers,” _SIGARCH Comput. Archit. News_ , vol. 45, no. 1, p. 17–32, Apr. 2017. [Online]. Available: https://doi.org/10.1145/3093337.3037700 

- [6] Q. Chen, H. Yang, J. Mars, and L. Tang, “Baymax: Qos awareness and increased utilization for non-preemptive accelerators in warehouse scale computers,” in _Proceedings of the Twenty-First International Conference on Architectural Support for Programming Languages and Operating Systems_ , ser. ASPLOS ’16. New York, NY, USA: Association for Computing Machinery, 2016, p. 681–696. [Online]. Available: https://doi.org/10.1145/2872362.2872368 

- [7] T. Chen, T. Moreau, Z. Jiang, L. Zheng, E. Yan, M. Cowan, H. Shen, L. Wang, Y. Hu, L. Ceze, C. Guestrin, and A. Krishnamurthy, “Tvm: an automated end-to-end optimizing compiler for deep learning,” in _Proceedings of the 13th USENIX Conference on Operating Systems Design and Implementation_ , ser. OSDI’18. USA: USENIX Association, 2018, p. 579–594. 

- [8] W. Chen, C. Lu, H. Xu, K. Ye, and C. Xu, “Multiplexing dynamic deep learning workloads with slo-awareness in gpu clusters,” in _Proceedings of the Twentieth European Conference on Computer Systems_ , ser. EuroSys ’25. New York, NY, USA: Association for Computing Machinery, 2025, p. 589–604. [Online]. Available: https://doi.org/10.1145/3689031.3696074 

- [9] W. Chen, Z. Mo, H. Xu, K. Ye, and C. Xu, “Interference-aware multiplexing for deep learning in gpu clusters: A middleware approach,” in _Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis_ , ser. SC ’23. New York, NY, USA: Association for Computing Machinery, 2023. [Online]. Available: https://doi.org/10.1145/3581784.3607060 

- [10] S. Choi, S. Lee, Y. Kim, J. Park, Y. Kwon, and J. Huh, “Serving heterogeneous machine learning models on Multi-GPU servers with Spatio-Temporal sharing,” in _2022 USENIX Annual Technical Conference (USENIX ATC 22)_ . Carlsbad, CA: USENIX Association, Jul. 2022, pp. 199–216. [Online]. Available: https: //www.usenix.org/conference/atc22/presentation/choi-seungbeom 

- [11] Y. Choi and M. Rhu, “Prema: A predictive multi-task scheduling algorithm for preemptible neural processing units,” in _2020 IEEE International Symposium on High Performance Computer Architecture (HPCA)_ , 2020, pp. 220–233. 

- [12] G. Cormode, V. Shkapenyuk, D. Srivastava, and B. Xu, “Forward decay: A practical time decay model for streaming systems,” in _2009 IEEE 25th International Conference on Data Engineering_ , 2009, pp. 138–149. 

- [13] “Linux and UNIX Man Pages,” https://www.unix.com/man-page/bsd/3/ dlopen/, 1993. 

- [14] M. Dun, X. Zhang, H. Cao, Y. Zhang, J. Huang, and X. Ye, “Adaptive sparse deep neural network inference on resource-constrained costefficient gpus,” in _2023 IEEE High Performance Extreme Computing Conference (HPEC)_ , 2023, pp. 1–7. 

- [15] P. Gao, L. Yu, Y. Wu, and J. Li, “Low latency rnn inference with cellular batching,” in _Proceedings of the Thirteenth EuroSys Conference_ , ser. EuroSys ’18. New York, NY, USA: Association for Computing Machinery, 2018. [Online]. Available: https://doi.org/10. 1145/3190508.3190541 

- [16] “The GNU C Library (glibc),” https://sourceware.org/glibc//, 2024. 

- [17] J. Gu, Y. Zhu, P. Wang, M. Chadha, and M. Gerndt, “Fast-gshare: Enabling efficient spatio-temporal gpu sharing in serverless computing for deep learning inference,” in _Proceedings of the 52nd International Conference on Parallel Processing_ , ser. ICPP ’23. New York, NY, USA: Association for Computing Machinery, 2023, p. 635–644. [Online]. Available: https://doi.org/10.1145/3605573.3605638 

- [18] J. Gu, B. Zhu, M. Li, W. Li, Y. Xia, and H. Chen, “A HardwareSoftware co-design for efficient Intra-Enclave isolation,” in _31st USENIX Security Symposium (USENIX Security 22)_ . Boston, MA: USENIX Association, Aug. 2022, pp. 3129–3145. [Online]. Available: https: //www.usenix.org/conference/usenixsecurity22/presentation/gu-jinyu 

- [19] A. Gujarati, R. Karimi, S. Alzayat, W. Hao, A. Kaufmann, Y. Vigfusson, and J. Mace, “Serving dnns like clockwork: performance predictability from the bottom up,” in _Proceedings of the 14th USENIX Conference on Operating Systems Design and Implementation_ , ser. OSDI’20. USA: USENIX Association, 2020. 

- [20] T. J. Ham, Y. Lee, S. H. Seo, S. Kim, H. Choi, S. J. Jung, and J. W. Lee, “Elsa: Hardware-software co-design for efficient, lightweight selfattention mechanism in neural networks,” in _2021 ACM/IEEE 48th Annual International Symposium on Computer Architecture (ISCA)_ , 2021, pp. 692–705. 

- [21] M. Han, H. Zhang, R. Chen, and H. Chen, “Microsecond-scale preemption for concurrent GPU-accelerated DNN inferences,” in _16th USENIX Symposium on Operating Systems Design and Implementation (OSDI 22)_ . Carlsbad, CA: USENIX Association, Jul. 2022, pp. 539– 558. [Online]. Available: https://www.usenix.org/conference/osdi22/ presentation/han 

- [22] L. Hu, X. Che, and Z. Xie, “Gpgpu cloud: A paradigm for general purpose computing,” _Tsinghua Science and Technology_ , vol. 18, no. 1, pp. 22–33, 2013. [Online]. Available: https: //www.sciopen.com/article/10.1109/TST.2013.6449404 

- [23] M. A. Ibrahim, O. Kayiran, Y. Eckert, G. H. Loh, and A. Jog, “Analyzing and leveraging decoupled l1 caches in gpus,” in _2021 IEEE International Symposium on High-Performance Computer Architecture (HPCA)_ , 2021, pp. 467–478. 

- [24] “Inline PTX Assembly in CUDA,” https://docs.nvidia.com/cuda/inlineptx-assembly/index.html, 2012. 

- [25] “intel rdt: Intel Cache Allocation Technology,” https://lwn.net/Articles/ 634676/, 2024. 

- [26] J. Jiang, H. Zhang, D. Liu, J. Du, X. Yao, J. Wei, P. Chen, D. Huang, and Y. Lu, “Efficient coupling streaming ai and ensemble simulations on hpc clusters,” in _Euro-Par 2024: Parallel Processing_ , J. Carretero, S. Shende, J. Garcia-Blas, I. Brandic, K. Olcoz, and M. Schreiber, Eds. Cham: Springer Nature Switzerland, 2024, pp. 313–328. 

- [27] “Set Environment Variables,” https://tldp.org/HOWTO/Oracle-9iFedora-3-Install-HOWTO/sect 04.html, 2005. 

- [28] H. Lee, M. Turilli, S. Jha, D. Bhowmik, H. Ma, and A. Ramanathan, “Deepdrivemd: Deep-learning driven adaptive molecular simulations for protein folding,” in _2019 IEEE/ACM Third Workshop on Deep Learning on Supercomputers (DLS)_ , 2019, pp. 12–19. 

- [29] L. Ma, Z. Xie, Z. Yang, J. Xue, Y. Miao, W. Cui, W. Hu, F. Yang, L. Zhang, and L. Zhou, “Rammer: Enabling holistic deep learning compiler optimizations with rTasks,” in _14th USENIX Symposium on Operating Systems Design and Implementation (OSDI 20)_ . USENIX Association, Nov. 2020, pp. 881–897. [Online]. Available: https://www.usenix.org/conference/osdi20/presentation/ma 

- [30] “NVIDIA Multi-Instance GPU,” https://docs.nvidia.com/datacenter/ tesla/mig-user-guide//, 2020. 

- [31] J. Mo, J. Gopinath, and B. Reagen, “Haac: A hardware-software co-design to accelerate garbled circuits,” in _Proceedings of the 50th Annual International Symposium on Computer Architecture_ , ser. ISCA ’23. New York, NY, USA: Association for Computing Machinery, 2023. [Online]. Available: https://doi.org/10.1145/3579371.3589045 

- [32] “NVIDIA Multi-Process Service,” https://docs.nvidia.com/deploy/mps//, 2013. 

- [33] “NVIDIA Nsight Compute,” https://developer.nvidia.com/nsightcompute/, 2024. 

- [34] “System Management Interface SMI,” https://developer.nvidia.com/ system-management-interface/, 2024. 

- [35] “NVIDIA Nsight Systems,” https://developer.nvidia.com/nsightsystems/, 2024. 

- [36] S. Pai, M. J. Thazhuthaveetil, and R. Govindarajan, “Improving gpgpu concurrency with elastic kernels,” in _Proceedings of the Eighteenth International Conference on Architectural Support for Programming_ 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 04,2026 at 04:44:23 UTC from IEEE Xplore.  Restrictions apply. 

_Languages and Operating Systems_ , ser. ASPLOS ’13. New York, NY, USA: Association for Computing Machinery, 2013, p. 407–418. [Online]. Available: https://doi.org/10.1145/2451116.2451160 

- [37] S. Pati, S. Aga, M. Islam, N. Jayasena, and M. D. Sinclair, “T3: Transparent tracking & triggering for fine-grained overlap of compute & collectives,” in _Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2_ , ser. ASPLOS ’24. New York, NY, USA: Association for Computing Machinery, 2024, p. 1146–1164. [Online]. Available: https://doi.org/10.1145/3620665.3640410 

- [38] S. Puthoor, X. Tang, J. Gross, and B. M. Beckmann, “Oversubscribed command queues in gpus,” in _Proceedings of the 11th Workshop on General Purpose GPUs_ , ser. GPGPU-11. New York, NY, USA: Association for Computing Machinery, 2018, p. 50–60. [Online]. Available: https://doi.org/10.1145/3180270.3180271 

- [39] “Pytorch Models And Pre-trained Weights,” https://pytorch.org/vision/ stable/models.html#general-information-on-pre-trained-weights/, 2017. 

- [40] “Pytorch,” https://pytorch.org/, 2024. 

- [41] D. Quinn, M. Nouri, N. Patel, J. Salihu, A. Salemi, S. Lee, H. Zamani, and M. Alian, “Accelerating retrieval-augmented generation,” in _Proceedings of the 30th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 1_ , ser. ASPLOS ’25. New York, NY, USA: Association for Computing Machinery, 2025, p. 15–32. [Online]. Available: https: //doi.org/10.1145/3669940.3707264 

- [42] V. J. Reddi, C. Cheng, D. Kanter, P. Mattson, G. Schmuelling, C.-J. Wu, B. Anderson, M. Breughe, M. Charlebois, W. Chou, R. Chukka, C. Coleman, S. Davis, P. Deng, G. Diamos, J. Duke, D. Fick, J. S. Gardner, I. Hubara, S. Idgunji, T. B. Jablin, J. Jiao, T. S. John, P. Kanwar, D. Lee, J. Liao, A. Lokhmotov, F. Massa, P. Meng, P. Micikevicius, C. Osborne, G. Pekhimenko, A. T. R. Rajan, D. Sequeira, A. Sirasao, F. Sun, H. Tang, M. Thomson, F. Wei, E. Wu, L. Xu, K. Yamada, B. Yu, G. Yuan, A. Zhong, P. Zhang, and Y. Zhou, “Mlperf inference benchmark,” 2020. 

- [43] T. G. Rogers, M. O’Connor, and T. M. Aamodt, “Cache-conscious wavefront scheduling,” in _2012 45th Annual IEEE/ACM International Symposium on Microarchitecture_ , 2012, pp. 72–83. 

- [44] R. Sabbadin, H. Fargier, and J. Lang, “Towards qualitative approaches to multi-stage decision making,” _International Journal of Approximate Reasoning_ , vol. 19, no. 3, pp. 441–471, 1998. [Online]. Available: https://www.sciencedirect.com/science/article/pii/S0888613X98100191 

- [45] A. Sarker, A. C. Canto, M. Mozaffari Kermani, and R. Azarderakhsh, “Error detection architectures for hardware/software co-design approaches of number-theoretic transform,” _IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems_ , vol. 42, no. 7, pp. 2418–2422, 2023. 

- [46] F. Strati, X. Ma, and A. Klimovic, “Orion: Interference-aware, fine-grained gpu sharing for ml applications,” in _Proceedings of the Nineteenth European Conference on Computer Systems_ , ser. EuroSys ’24. New York, NY, USA: Association for Computing Machinery, 2024, p. 1075–1092. [Online]. Available: https://doi.org/10. 1145/3627703.3629578 

   - [53] Z. Wang, J. Yang, R. Melhem, B. Childers, Y. Zhang, and M. Guo, “Simultaneous multikernel gpu: Multi-tasking throughput processors via fine-grained sharing,” in _2016 IEEE International Symposium on High Performance Computer Architecture (HPCA)_ , 2016, pp. 358–369. 

   - [54] W. Xiao, S. Ren, Y. Li, Y. Zhang, P. Hou, Z. Li, Y. Feng, W. Lin, and Y. Jia, “AntMan: Dynamic scaling on GPU clusters for deep learning,” in _14th USENIX Symposium on Operating Systems Design and Implementation (OSDI 20)_ . USENIX Association, Nov. 2020, pp. 533–548. [Online]. Available: https://www.usenix.org/conference/ osdi20/presentation/xiao 

   - [55] Y. Yang, L. Zhao, Y. Li, H. Zhang, J. Li, M. Zhao, X. Chen, and K. Li, “Infless: a native serverless system for low-latency, high-throughput inference,” in _Proceedings of the 27th ACM International Conference on Architectural Support for Programming Languages and Operating Systems_ , ser. ASPLOS ’22. New York, NY, USA: Association for Computing Machinery, 2022, p. 768–781. [Online]. Available: https://doi.org/10.1145/3503222.3507709 

   - [56] T. T. Yeh, M. D. Sinclair, B. M. Beckmann, and T. G. Rogers, “Deadlineaware offloading for high-throughput accelerators,” in _2021 IEEE International Symposium on High-Performance Computer Architecture (HPCA)_ , 2021, pp. 479–492. 

   - [57] C. Yu, Y. Bai, H. Yang, K. Cheng, Y. Gu, Z. Luan, and D. Qian, “Smguard: A flexible and fine-grained resource management framework for gpus,” _IEEE Transactions on Parallel and Distributed Systems_ , vol. 29, no. 12, pp. 2849–2862, 2018. 

   - [58] W. Zhang, B. Chen, Z. Han, Q. Chen, P. Cheng, F. Yang, R. Shu, Y. Yang, and M. Guo, “PilotFish: Harvesting free cycles of cloud gaming with deep learning training,” in _2022 USENIX Annual Technical Conference (USENIX ATC 22)_ . Carlsbad, CA: USENIX Association, Jul. 2022, pp. 217–232. [Online]. Available: https://www.usenix.org/conference/atc22/presentation/zhang-wei 

   - [59] W. Zhang, Q. Chen, N. Zheng, W. Cui, K. Fu, and M. Guo, “Toward qos-awareness and improved utilization of spatial multitasking gpus,” _IEEE Transactions on Computers_ , vol. 71, no. 4, pp. 866–879, 2022. 

   - [60] Y. Zhang, H. Yu, C. Han, C. Wang, B. Lu, Y. Li, Z. Jiang, Y. Li, X. Chu, and H. Li, “Sgdrc: Software-defined dynamic resource control for concurrent dnn inference on nvidia gpus,” in _Proceedings of the 30th ACM SIGPLAN Annual Symposium on Principles and Practice of Parallel Programming_ , ser. PPoPP ’25. New York, NY, USA: Association for Computing Machinery, 2025, p. 267–281. [Online]. Available: https://doi.org/10.1145/3710848.3710863 

   - [61] H. Zhao, W. Cui, Q. Chen, and M. Guo, “Ispa: Exploiting intrasm parallelism in gpus via fine-grained resource management,” _IEEE Transactions on Computers_ , vol. 72, no. 5, pp. 1473–1487, 2023. 

   - [62] H. Zhao, W. Cui, Q. Chen, Y. Zhang, Y. Lu, C. Li, J. Leng, and M. Guo, “Tacker: Tensor-cuda core kernel fusion for improving the gpu utilization while ensuring qos,” in _2022 IEEE International Symposium on HighPerformance Computer Architecture (HPCA)_ , 2022, pp. 800–813. 

   - [63] H. Zhao, W. Cui, Q. Chen, J. Zhao, J. Leng, and M. Guo, “Exploiting intra-sm parallelism in gpus via persistent and elastic blocks,” in _2021 IEEE 39th International Conference on Computer Design (ICCD)_ , 2021, pp. 290–298. 

- [47] J. A. Stratton, C. Rodrigues, I.-J. Sung, N. Obeid, L.-W. Chang, N. Anssari, G. D. Liu, and W.-m. W. Hwu, “Parboil: A revised benchmark suite for scientific and commercial throughput computing,” _Center for Reliable and High-Performance Computing_ , vol. 127, no. 7.2, 2012. 

- [48] Q. Sun, Y. Liu, H. Yang, Z. Luan, and D. Qian, “Smqos: Improving utilization and energy efficiency with qos awareness on gpus,” in _2019 IEEE International Conference on Cluster Computing (CLUSTER)_ , 2019, pp. 1–5. 

- [49] “Tensorflow Serving Models,” https://www.tensorflow.org/tfx/guide/ serving, 2024. 

- [50] “NVIDIA TensorRT,” https://docs.nvidia.com/tensorrt/index.html, 2024. 

- [51] R. Veldema and M. Philippsen, “Parallel memory defragmentation on a gpu,” in _Proceedings of the 2012 ACM SIGPLAN Workshop on Memory Systems Performance and Correctness_ , ser. MSPC ’12. New York, NY, USA: Association for Computing Machinery, 2012, p. 38–47. [Online]. Available: https://doi.org/10.1145/2247684.2247693 

- [52] R. Veldema and M. Philippsen, “Parallel memory defragmentation on a gpu,” in _Proceedings of the 2012 ACM SIGPLAN Workshop on Memory Systems Performance and Correctness_ , ser. MSPC ’12. New York, NY, USA: Association for Computing Machinery, 2012, p. 38–47. [Online]. Available: https://doi.org/10.1145/2247684.2247693 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 04,2026 at 04:44:23 UTC from IEEE Xplore.  Restrictions apply. 

