# Hexa-MoE: Efficient and Heterogeneous-aware Training for Mixture-of-Experts

# Shuqing Luo<sup>1</sup> Jie Peng<sup>2</sup> Pingzhi Li<sup>3</sup> Hanrui Wang<sup>4</sup> Tianlong Chen<sup>3</sup>

### **Abstract**

Mixture-of-experts (MoE) has emerged as a practical approach to scale up parameters for Transformer model to achieve better generalization while keeping computational efficiency. Current MoE models are mainly deployed with expert parallelism on distributed devices, which typically depends on homogeneous devices, and suffers from heavy communication overhead as well as computation redundancy under scaled workload. To tackle these teasers, we propose a Heterogeneousaware Expert Allocation framework, **HEXA-MoE**, with significantly enhanced efficiency. It contains two components: (1) Expert-Specific Operators. We replace the typical GeMM or grouped GeMM interfaces with our proposed expert-specific operators, making expert computing to be performed in an in-place manner with almost no redundant FLOPs. (2) Adaptive Data- and Model-Centric configurations for different workload scales. We introduce a memory-efficient computationcommunication overlapping scheme to tackle the heavy memory consumption in current datacentric libraries, which can accelerate training with heavy workloads. Comprehensive experiments on the Swin-MoE benchmark consistently reveal the effectiveness of HEXA-MoE, i.e., reducing  $10\% \sim 48\%$  memory consumption and achieving  $0.5 \sim 4.3 \times$  speed up compared to current state-of-the-art MoE libraries. We further examine HEXA-MoE on heterogeneous devices, and promising results show that employing optimal parallel configuration can better utilize global computation resources, and substantially minimize overall latency. Codes are available at here.

### 1. Introduction

![](_page_0_Figure_7.jpeg)

<span id="page-0-0"></span>Figure 1. Convergence Analysis. HEXA-MOE can significantly surpass Tutel on MoE training due to the specialized designs.

Transformers have become the *de facto* architecture for a vast range of machine learning tasks including natural language process (Vaswani, 2017; Raffel et al., 2020), computer vision (Liu et al., 2021; He et al., 2022b) and multi-modal learning (Li et al., 2022; Liu et al., 2024). To scale up model parameters for stronger learning capacity and better generalization following the empirical scaling law (Kaplan et al., 2020), Mixture-of-experts (MoE) (Jiang et al., 2024) has been demonstrated as a practical and computation-efficient approach. For a single Transformer layer, MoE expands the feed-forward network (FFN) to a group of experts with the same architecture and different parameters, activated selectively during computing. This dynamic nature poses peculiar challenges to system design. Current MoE frameworks usually present some unique attributes that differentiate them from dense models:

- Distributed computing is required due to its sheer size, and expert parallelism (Lepikhin et al., 2020) is the most commonly used technique, which distributes the experts in one layer on different devices.
- The dynamic workload for each expert makes it necessary to employ additional dispatch and combine operations in expert parallel to utilize the general matrix multiplication (GeMM) or grouped GeMM interface.
- **3** Dispatch and combine rely on synchronous all-to-all communication to assign tokens to distributed experts.

These peculiar designs lead to apparent inefficiency for MoE training. On the one hand, current MoE libraries are either static like Tutel (Hwang et al., 2023), or dynamic like MegaBlocks (Gale et al., 2023). The former depends on an additional hyper-parameter named expert capacity to

<sup>&</sup>lt;sup>1</sup>Peking University, China <sup>2</sup>University of Science and Technology of China, China <sup>3</sup>University of North Carolina, Chapel Hill, USA <sup>4</sup>University of California, Los Angeles, USA. Correspondence to: Tianlong Chen <tianlong@cs.unc.edu>.

calibrate the workload for different experts, which suffers from redundant memory allocation or access, while the latter tends to raise out of memory (OOM) error during runtime due to its uncertainty. Meanwhile, the synchronous all-to-all communication can occupy over 40% runtime with scaled model and devices [\(Hwang et al.,](#page-8-7) [2023\)](#page-8-7). On the other hand, expert parallelism is mainly deployed on homogeneous devices. However, a homogeneous cluster with cutting-edge devices is much more expensive than heterogeneous ones due to the fast iteration of modern GPU and the decrease on prices for outdated ones. Adapting expert parallelism to *heterogeneous* devices would rely on rescheduling expert placement to utilize global computing resources sufficiently. However, the inherent dynamic property of MoE makes it difficult to implement, as the workload of each expert changes dynamically in each step [\(He et al.,](#page-8-9) [2022a;](#page-8-9) [Li et al.,](#page-8-10) [2023\)](#page-8-10). Since heterogeneous devices are cheaper and easier to access, if MoE training can be refactored to be heterogeneous-aware, it would advance wider deployment.

We propose HEXA-MoE, a completely-static MoE training framework with minimized memory consumption and heterogeneous-awareness. We first find that if implementing MoE layer with GeMM, the unnecessary components such as token padding or discarding are unavoidable, while grouped GeMM [\(Gale et al.,](#page-8-8) [2023\)](#page-8-8) would lead to dynamic runtime behavior and uncertainty. To tackle it, we investigate the forward and backward propagation for the MoE layer thoroughly, and propose to replace (grouped) GeMM with *expert-specific* operators. Specifically, the basic MoE operations are essentially build upon 3 operators, *i*.*e*., expertspecific matrix multiplication (*ESMM*), summation (*ESS*) and transposed matrix multiplication (*ESTMM*). The forward propagation only requires *ESMM*, while backward requires all. We implement them with specialized GPU kernels, and distribute the MoE layer with tensor parallelism [\(Narayanan et al.,](#page-9-2) [2021\)](#page-9-2) as an alternative to expert parallelism, which is easy to be heterogeneous-aware.

HEXA-MoE also considers different workload scales for MoE training, and is adapted to both data- and model-centric configurations. Their difference lies in the content of communication. For model-centric, local mini-batches are synchronized among devices and model parameters are kept still, while for data-centric, local parameters are gathered for each device, while local mini-batches are kept still [\(Liu et al.,](#page-8-11) [2023\)](#page-8-11). Data-centric setting outperforms model-centric with scaled workloads. However, previous library preserves all the gathered parameter shards on each device for backward pass, leading to huge memory consumption. To tackle this teaser, we design a novel memory-efficient communicationcommunication overlapping scheme utilizing a pipelineshared cache on each device, which is updated dynamically for both forward and backward pass. Both settings are completely static in runtime, and the workload of each device

can be easily determined from batch size or tensor parallelism configuration (sub-dimension of FFN intermediate size), making it easy to be adapted to heterogeneous devices via specialized expert allocation.

Our contributions can be summarized as follows:

- ⋆ A completely static MoE training framework with minimized memory footprint. To our best knowledge, HEXA-MoE is the first MoE library with completely static and minimal memory footprint, enabling in-place computing with specialized kernel, and employing tensor parallelism for distributed training, delivering exact result with faster speed, as shown in Figure [1.](#page-0-0)
- ⋆ Advanced efficiency. HEXA-MoE is built upon our proposed expert-specific operators, which tackles scaled workload with data-centric strategy via memoryefficient communication-computation overlapping. Experiments on Swin-MoE show that HEXA-MoE can reduce 10% ∼ 48% memory usage and achieve 0.5 ∼ 4.3× speed up compared to SOTA MoE libraries.
- ⋆ Heterogeneous-awareness. HEXA-MoE transforms conventional expert parallelism into data or tensor parallelism, making it heterogeneous-aware. Experiments show that HEXA-MoE can be effectively adapted to heterogeneous devices and substantially minimize overall latency via employing optimal parallel configuration.

# 2. Related Works

Scaling Transformers with MoE. Transformer models can be scaled up via the Mixture-of-Experts (MoE) for better learning capacity and generalization while enjoying a sublinear increase in computation overhead due to its sparsity. It has been proven in many research fields such as natural language processing [\(Fedus et al.,](#page-8-12) [2022;](#page-8-12) [Jiang et al.,](#page-8-5) [2024;](#page-8-5) [Du et al.,](#page-8-13) [2022\)](#page-8-13), machine vision [\(Riquelme et al.,](#page-9-3) [2021;](#page-9-3) [Fan](#page-8-14) [et al.,](#page-8-14) [2022\)](#page-8-14) and multi-modal learning [\(Bao et al.,](#page-8-15) [2022\)](#page-8-15).

Open-Source MoE Training Libraries. FastMoE [\(He](#page-8-16) [et al.,](#page-8-16) [2021\)](#page-8-16) pioneered the first PyTorch open-source implementation for distributed MoE. Based on it, Faster-MoE [\(He et al.,](#page-8-9) [2022a\)](#page-8-9) proposes dynamic shadowing and smart scheduling to tackle load imbalance and improve parallelism. Tutel [\(Hwang et al.,](#page-8-7) [2023\)](#page-8-7) implements switchable parallelism and dynamic pipelining, improving adaptability and scalability. MegaBlocks [\(Gale et al.,](#page-8-8) [2023\)](#page-8-8) proposes block-sparse operations and corresponding GPU kernels to mitigate dynamic routing issues in MoEs.

Improving efficiency for MoE computing. MoE computing faces significant challenges in communication and memory. Recent research has explored alleviating them from various perspectives: Lina [\(Li et al.,](#page-8-10) [2023\)](#page-8-10) proposes to dynamically schedule resources to reduce latency and tackle the all-to-all bottleneck. Janus [\(Liu et al.,](#page-8-11) [2023\)](#page-8-11) proposes a data-centric paradigm that replaces traditional all-to-all communication with asynchronous expert fetching, enabling overlapped communication and computation for enhanced

![](_page_2_Figure_1.jpeg)

Figure 2. Comparison between conventional and expert-specific formulation for MoE computing. We take top-1 routing for illustration and present the corresponding relation of each formula in the MoE forward and backward propagation.

parallelism. ScheMoE (Shi et al., 2024) proposes a generic scheduling framework for optimal communication and computation scheduling during MoE training. MPMoE (Zhang et al., 2024) accelerates MoE training through adaptive and memory-efficient pipeline parallelism. SmartMoE (Zhai et al., 2023) proposes an efficient searching algorithm to identify optimization opportunities within an expanded hybrid parallelism space, tailored for data-sensitive MoE models. PipeMoE (Shi et al., 2023) adaptively pipelines communications and computations in MoE to mask communication latency. It also provides an optimal strategy to determine pipeline degree to minimize overall iteration time.

**Principles of GPU Acceleration** Modern GPUs provide massive threads for parallel execution, grouped into *thread-blocks*, and executed on streaming multiprocessors (SMs). GPUs have a memory hierarchy, outlined as large but slow-accessed high bandwidth memory (HBM), and small but faster-accessed shared memory (SRAM). Matrix multiplication is optimized on GPU with *tiling*, *i.e.*, partitioning the output matrix into small 2D blocks, where each block is computed by a *thread-block* in parallel. The size of individual blocks can be adjusted to improve runtime performance.

#### 3. Method

We first formulate the forward and backward propagation for a single MoE layer with GeMM and expert-specific operators respectively (Sec. 3.1), which deliver exact results in different manner. After that, we expound the details for our specialized kernel (Sec. 3.2), where a novel expert-specific fused operator is introduced for parallel MoE backward pass. These empower HEXA-MoE with high computational efficiency. Next, we consider different workload scales, and adapt HEXA-MoE efficiently to both *data*- and *model*-centric settings (Sec. 3.3). Finally, we adapt HEXA-MoE to heterogeneous devices, and provide an expert allocation approach to minimize the average latency so as to utilize better heterogeneous computing capacities (Sec. 3.4).

#### <span id="page-2-1"></span><span id="page-2-0"></span>3.1. MoE Computing Formulation

**MoE Computing with GeMM.** To employ GeMM for expert computing, we need to calibrate the amount of dispatched tokens for each expert via token padding or discarding. Taking top-1 routing as an example, we formulate the forward and backward propagation in Figure 2, denoting  $\ell$  as loss value and  $\boldsymbol{x}^e$  as the  $N_e$  tokens dispatched to expert e. In backward propagation, the gradients of the output have been provided by the auto-differentiation program.

Based on dispatch and combine, all variables can be derived via basic matrix operations such as summation and multiplication. Although we can utilize the high-performance GeMM interface, these operations are memory inefficient, since the workload of each expert varies dynamically in each step, and token padding or discarding has to be employed to construct new mini-batches for local experts. It can be overcome by our expert-specific design.

MoE Computing with Expert-Specific Operators. Based on the above formulations, we propose to refactor the MoE workflow in an in-place manner to address the teaser of inefficiency and dynamic. Specifically, we find that the forward and backward propagation of a single MoE layer can be reformulated with 3 basic specialized operators from an expert-specific perspective, namely *expert-specific* operators. We take top-1 routing for illustration, while formulations for top-k routing are provided in Appendix B.

- **©** Expert-specific matrix multiplication (ESMM): Given input x with N tokens, routing choice  $\mathcal{R}(x)$ , weight W and bias b, the output  $y = ESMM(x, W, b, \mathcal{R}(x))$ , where  $y_i$  is derived from  $x_i$ ,  $W_{\mathcal{R}(x_i)}$  and  $b_{\mathcal{R}(x_i)}$ .
- **Expert-specific summation** (ESS): Given input x with N tokens and routing choice  $\mathcal{R}(x)$ , the output  $y = ESS(x, \mathcal{R}(x))$ , where tokens routed to expert e are added up and recorded in y[e].
- **8** Expert-specific transposed matrix multiplication (*ESTMM*): the inputs include  $x_1$  and  $x_2$ , both with N

<span id="page-3-1"></span>![](_page_3_Figure_1.jpeg)

![](_page_3_Figure_2.jpeg)

(a) Re-Index Vector Construction

<span id="page-3-3"></span>![](_page_3_Figure_4.jpeg)

(b) Expert-Specific Matrix Multiplication

<span id="page-3-2"></span>![](_page_3_Figure_6.jpeg)

(c) Expert-Specific Summation (d) Expert-Specific Transposed Matrix Multiplication Figure 3. Illustration of the proposed operators. We take 10 tokens, 4 global experts, and tiling size 4 as an example. For ESTMM, the 2 input batches are in a re-indexed format, while for others both the raw batch and re-index vector are provided.

tokens, and the routing choice  $\mathcal{R}(\boldsymbol{x})$ . Both  $\boldsymbol{x}_1$  and  $\boldsymbol{x}_2$  are the ESMM result of  $\boldsymbol{x}$ , thus sharing the same routing choice. The output  $\boldsymbol{y} = ESTMM(\boldsymbol{x}_1, \boldsymbol{x}_2, \mathcal{R}(\boldsymbol{x}))$ . For the i-th channel of  $\boldsymbol{x}_1$  and j-th channel of  $\boldsymbol{x}_2$ , we first prepare a zero vector  $\boldsymbol{c}$  with E elements, and accumulate  $\boldsymbol{x}_1[m,i] \cdot \boldsymbol{x}_2[m,j]$  to  $\boldsymbol{c}[\mathcal{R}(\boldsymbol{x}_m)]$  for all  $0 \leq m < N$ . After that we assign  $\boldsymbol{c}[k]$  to  $\boldsymbol{y}[k,i,j]$  for all  $0 \leq k < E$ .

Based on these *expert-specific* operators, we can now implement MoE computing in a novel in-place manner, as formulated in Figure 2. We compare each stage between our method and conventional formulation, and the output of each token is ensured to be consistent for both cases. In forward pass, *ESMM* serves as an alternative to GeMM, while in backward pass, the gradient of *y* is also provided by the auto-differentiation program. *ESS* performs as an alternative to tensor summation, while *ESMM* and *ESTMM* are designed as 2 specialized cases for matrix multiplication from the expert-specific perspective. The expert-specific design enables HEXA-MoE to be a completely static framework at runtime, introducing almost no redundant FLOPs.

### <span id="page-3-0"></span>3.2. Optimized CUDA Implementations

**Re-Index based Expert-Specific Operators.** To implement tiled matrix multiplication for the expert-specific operators, the HBM I/O should be re-directed, since naively implement them cannot fully utilize GPU locality as well as tensor cores, which restricts runtime performance. In

<span id="page-3-4"></span>HEXA-MoE, we introduce re-index vector as I/O guidance, illustrated in Figure 3(a). Specifically, we re-organize the order of token indices based on the routing choice, *i.e.*, gathering the token indices routed to the same expert into a sub-vector, and padding it with -1 to make it divisible by the tiling size. Algorithm details are provided in Appendix B.

For ESMM, we illustrate it in Figure 3(b), where a threadblock first loads a sub-vector, followed by input tokens based on the vector values, as well as the corresponding expert parameters. Since the loaded tokens are routed to the same expert, we only need to load weights for one expert. We then accumulate the dot product results along the dimension of the input feature, and write the result back to HBM guided by the sub-vector. For ESS, we illustrate it in Figure 3(c), where each *thread-block* is assigned with certain channels of one expert, with tokens specialized by the sub-vectors for that expert. After accumulating all the assigned tokens, it writes the result back to HBM. For ESTMM, we illustrate it in Figure 3(d), where input batches are presented in a re-indexed manner. The two inputs here are the ESMM results of the same tensor, sharing the same re-index vector. Each thread block loads certain channels of both inputs for tokens routed to the same expert. The cumulative dot production result is write back to the corresponding area on HBM after computing. More algorithm details of this section are provided in Appendix B.

<span id="page-4-2"></span>![](_page_4_Figure_1.jpeg)

<span id="page-4-3"></span>![](_page_4_Figure_2.jpeg)

(b) Pipeline-shared cache before and after all gather. Figure 4. Visualization of the training pipeline and shared

cache in data-centric setting. Each device copies the kept parameter shard to the cache before all gather communication, and after that it can access the whole parameters of an MoE layer.

Memory Optimization for Top-k Routing. We find that when extending the routing policy from top-1 to top-k, the memory footprint would increase significantly. To tackle it, we only enlarge the memory allocation for the intermediate tokens to k times. The input and output tensors (and their gradients) are the cumulation of k *ESMM* results, which can either be implemented in serial, or employ the atomic add interface[1](#page-4-1) , which lead to similar runtime.

Fused Kernel in Backward Propagation. For a single layer MLP y = x · W + b, the gradients for x, W and b can be computed in parallel. Similarly, we can integrate *ESS*, *ESTMM*, and *ESMM* together into one kernel, namely expert-specific fused kernel (*ESFK*) for enhanced parallelism. However, directly integrating them is difficult, since the shape of *thread-block*s and the implementation details for each operator vary a lot. To this end, we set the shape of *thread-block* for each operator to be the same, and expand one dimension for *ESMM* and *ESS* so that the thread grids are all 3-dimensional. Taking a single FFN layer under top-1 routing as an example, where we present the shape of allocated thread block and thread grid for both forward and backward propagation in Table [6](#page-14-0) in Appendix [C.](#page-13-0) By shape transposing or dim expanding for individual *thread-block*s, we can align the 1st and 2nd dim while aggregate the 3rd dim for an integrated *thread-grid*. In this way, the backward pass for an MoE layer can be implemented with only 2 fused kernels and an element-wise dot production.

<span id="page-4-1"></span>1 Some data types may not be supported for atomic add on NVIDIA GPUs. We initialize the output tensor with float32, and transform it to the target type after computing.

#### <span id="page-4-0"></span>3.3. Parallel Strategy for Different Workload Scale

Data Parallelism for Data-Centric Setting. Data-centric is a practical and efficient approach for training deep neural network models with heavy workloads. HEXA-MoE distributes MoE parameters to different devices based on the partition of FFN intermediate size, namely tensor parallelism. In data-centric setting, each device gathers the whole MoE parameters of one layer from other devices, and computes locally, similar to data parallelism. Although data-centric MoE training has been explored by Janus [\(Liu](#page-8-11) [et al.,](#page-8-11) [2023\)](#page-8-11), the teaser of memory efficiency has not been fully tackled. Janus pre-fetchs the required MoE parameters for each layer progressively in forward pass, and preserves all of them locally for backward pass so that no communication would be required in backward. However, the sheer size of MoE parameters can lead to a significant increase in memory usage, making it inefficient for deployment.

To tackle this issue, we propose to allocate an additional region on HBM of each GPU to dynamically cache the gathered MoE shards for each pipeline stage, named as pipeline-shared cache. Since each device keeps a subset of the FFN intermediate size for all experts in each layer, we employ all gather communication among devices before computing for an MoE layer, preparing for the required full parameters in both forward and backward pass, as illustrated in Figure [4\(a\)](#page-4-2) and [4\(b\).](#page-4-3) For better parallelism, all gather communication can be overlapped with other operations such as attention and router, as shown in Figure [4\(a\).](#page-4-2) In this way, each device would not have to preserve the full MoE parameters for backward pass, while communication overhead can also be overlapped, therefore both memory efficiency and computing efficiency can be achieved.

Tensor Parallelism for Model-Centric Setting. For small scale of workload less than model parameters, modelcentric configuration turns out to be more efficient than data-centric. To distribute MoE parameters on multiple devices, we modify the classical tensor parallelism with our proposed expert-specific operators. All gather communication is employed to synchronize local data batches before and after each MoE layer. We also distribute each MoE layer among different devices along the FFN intermediate size for each expert, and during MoE computing, each device computes the all gathered data batches with only the local MoE parameter chunk using *ESMM*, after that the output tokens are all reduced with sum operation in forward pass. During backward propagation, all gather and all reduce communications are interchanged, while *ESMM* are replaced by *ESMM*, *ESS*, and *ESTMM* to get the gradients for input tokens, bias, and weights, respectively, which can also be replaced by the fused operator *ESFK*.

![](_page_5_Figure_1.jpeg)

Figure 5. Average memory usage for training Swin-Transformer-MoE models. We take 8 global experts and examine all cases from top-1 to top-8 routings. Experiments are conducted on 2 homogeneous GPUs using automatic mixed precision in PyTorch. The batch size is set to 40 for all cases. We record the average GPU memory consumption (GB) on each device.

#### <span id="page-5-0"></span>3.4. Heterogeneous-aware Expert Allocation

Workload Division for different configurations. For data-centric, the workload mainly depends on local batch size, as it can be essentially viewed as data parallelism. For model-centric, the workload of each device mainly depends on the allocated sub-dimension of FFN intermediate size for each expert, as expert-specific operators enable the implementation of tensor parallelism.

#### Practical Expert Allocation on Heterogeneous Devices.

We propose a specialized expert allocation approach to utilize heterogeneous computing resources, by adjusting the workload of each device. Specifically, we have to first examine the computing capacity of each device by averaging its latency on a benchmark task with heavy computing, such as large matrix multiplication. Denoting the results as  $\{t_i\}_{i=0}^{N-1}$  for N GPUs. For data-centric, we assign different local batch size  $\{B_i\}_{i=0}^{N-1}$  to different devices based on the

<span id="page-5-1"></span>
$$B_{i} = \frac{\frac{1}{\iota_{i}}}{\sum_{j=0}^{N-1} 1/t_{j}} \cdot B_{\text{global}}, \tag{1}$$

examined latency, as illustrated in Equation 1:  $B_i = \frac{1/t_i}{\sum_{j=0}^{N-1} 1/t_j} \cdot B_{\text{global}}, \qquad (1)$  where we denote  $B_{\text{global}}$  as the global batch size, *i.e.*,  $B_{\text{global}} = \sum_{j=0}^{N-1} B_i$ . For model-centric, we denote the sub-dimension of hidden size for one Mark 1. sub-dimension of hidden size for one MoE layer on each device as  $\{h_i\}_{i=0}^{N-1}$  with total hidden size H. The assigned

<span id="page-5-2"></span>
$$h_i = \frac{1/t_i}{\sum_{j=0}^{N-1} 1/t_j} \cdot H. \tag{2}$$

sub-dimension  $h_i$  on device i can be derived as Equation 2.  $h_i = \frac{1/t_i}{\sum_{j=0}^{N-1} 1/t_j} \cdot H. \tag{2}$  To ensure that both  $\{B_i\}_{i=0}^{N-1}$  and  $\{h_i\}_{i=0}^{N-1}$  are integers, we need to further round the sum and  $\{h_i\}_{i=0}^{N-1}$  are integers. need to further round them up or down while making sure that the summation is exactly  $B_{global}$  or H.

#### 4. Experiments

#### 4.1. Experimental Setup

We implement all the experiments using 2 machines, namely  $M_{\text{homo}}$  and  $M_{\text{hete}}$ . Specifically,  $M_{\text{homo}}$  is composed of 4 ho-

<span id="page-5-4"></span><span id="page-5-3"></span>Table 1. Details of the machines and GPUs used for both homogeneous and heterogeneous experiments.

|     | Notation                   | Number & Hardware Specs                     | Memory  |
|-----|----------------------------|---------------------------------------------|---------|
| CPU | $M_{\rm homo}$             | $2\times$ Intel Xeon Platinum 8352V 2.10GHz | 1008 GB |
|     | $M_{\rm hete}$             | 2× Intel Xeon Gold 6130 2.10GHz             | 62.5 GB |
|     | $D$ (in $M_{\rm homo}$ )   | $4\times$ NVIDIA GeForce RTX 4090           | 24 GB   |
| GPU | $D_0$ (in $M_{\rm hete}$ ) | $1 \times$ NVIDIA TITAN RTX                 | 24 GB   |
| ĺ   | $D_1$ (in $M_{\rm hete}$ ) | 1× NVIDIA GeForce RTX 2080 Ti               | 11 GB   |

mogeneous GPUs, while  $M_{\text{hete}}$  contains 2 heterogeneous GPUs. Details for the machines and GPUs are provided in Table 1. We take the training process of Swin-Transformer-MoE as the benchmark for all the experiments, including memory analysis, latency analysis, heterogeneous computing analysis, and ablation studies. Apart from heterogeneous computing analysis, all the experiments are conducted on  $M_{\text{homo}}$ . We also adopt both the Small and Base scales for the Swin-MoE model, following Tutel (Hwang et al., 2023). Experiments are conducted with PyTorch, taking nccl as the communication backend. Batch size is recorded as the workload of single device, and automatic mixed precision is used for training. atomicAdd is employed for HEXA-MoE to aggregate the expert computing results for each token.

#### 4.2. Experimental Results

Memory Analysis on Homogeneous Devices. We analyze the memory footprint for different MoE libraries on homogeneous devices with 8 global experts, and examine from top-1 to top-8 routing. Results are visualized in Figure 5. HEXA-MoE can reduce 10%-48% memory footprint compared to Tutel (Hwang et al., 2023) and MegaBlocks (Gale et al., 2023), and the memory footprint for model-centric is slightly less than data-centric owing to the pipeline-shared cache. From top-1 to top-k routing, the memory footprint increase of HEXA-MoE is more gentle than others, owing to

![](_page_6_Figure_1.jpeg)

Figure 6. Average latency for training Swin-Transformer-MoE models. Experiments are conducted on 4 homogeneous GPUs with 4 global experts. We set different batch sizes for different models under different routing strategies to maximize the utilization of GPU memory. We record the average latency for one step (s) during training with 2k steps in total.

![](_page_6_Figure_3.jpeg)

<span id="page-6-1"></span>Figure 7. Latency comparison with data-centric and modelcentric configurations. We record the average latency per step with 2k steps in total. The data-centric setting presents a more gentle trend when scaling workload.

the introduction of memory optimization. Memory saving mainly benefits from the in-place design of *expert-specific* operators and memory optimization.

Latency Analysis on Homogeneous Devices. We analyze the average latency for one training step, and present the results in Figure [6.](#page-6-0) To fully utilize the GPU HBM, we set different batch sizes for different routing configurations and different model scales. HEXA-MoE achieves 0.5- 4.3× speed up compared to Tutel [\(Hwang et al.,](#page-8-7) [2023\)](#page-8-7) and MegaBlocks [\(Gale et al.,](#page-8-8) [2023\)](#page-8-8). Since the scale of workload is larger than MoE parameters in our experiments, the average latency under a data-centric setting is less than modelcentric in these cases, and the advantage of HEXA-MoE turns more obvious while enlarging batch size. To better visualize the latency comparison between data-centric and model-centric configurations, we present the latency for the Swin-MoE-Base model under different batch sizes in Figure [7.](#page-6-1) When the workload is relatively small, a modelcentric setting is more efficient, while a data-centric setting becomes more efficient with a relatively large workload. Experiments are all conducted on the homogeneous machine with automatic mixed precision in PyTorch.

Heterogeneous Experiments. We demonstrate the heterogeneous awareness of our HEXA-MoE via experiments on different devices under both data-centric and model-centric configurations, and visualize the average latency with different division strategies in Figure [8.](#page-7-0) Since the 2 GPUs in our heterogeneous machine have similar computing capacities, we also adjust the power limit for each device. We first examine the computing capacity for these experimental settings, and provide the results in Table [2.](#page-6-2) The details of the proxy task are provided in Appendix [C.](#page-13-0)

<span id="page-6-2"></span><span id="page-6-0"></span>Table 2. Examining computing capacity proportion for heterogeneous devices. P, T and R denote power constraint (W), average latency (s), and computing capacity proportion.

|    | Case 1<br>P<br>T<br>R |      |      |     | Case 2 |      | Case 3 |      |      |  |
|----|-----------------------|------|------|-----|--------|------|--------|------|------|--|
|    |                       |      |      | P   | T      | R    | P      | R    |      |  |
| D0 | 100                   | 4.58 | 0.40 | 300 | 3.20   | 0.50 | 300    | 3.28 | 0.74 |  |
| D1 | 300                   | 3.06 | 0.60 | 300 | 3.18   | 0.50 | 100    | 9.42 | 0.26 |  |

In Figure [8\(a\),](#page-7-1) we adjust the batch size on each device under data-centric configuration. Essentially the optimal proportion is very close to the examined computing capacity proportion, and it has a certain deviation from uniform division when the power constraints of the two devices are not equal, while the optimal division is also uniform division when the power constraints are equal. Specifically, when setting the power constraint for D<sup>0</sup> to be 100W and D<sup>1</sup> to be 300W, employing our heterogeneous-aware allocation can make the average latency to be 13.2% lower than naive division, and when setting D<sup>0</sup> to be 300W and D<sup>1</sup> to be 100W, our approach can reduce the average latency by 25.3%.

In Figure [8\(b\),](#page-7-2) we adjust the allocated sub-dimension proportion of the hidden size for each MoE layer on each device. Adapting the sub-dimension proportion to be close to the computing capacity proportion of heterogeneous devices can also substantially minimize average latency, similar to the data-centric setting. Specifically, when setting the power constraint of D<sup>0</sup> to be 100W and D<sup>1</sup> to be 300W, our approach can achieve a 6.3% reduction on average latency compared to uniform division, and when setting D<sup>0</sup> to be 300W and D<sup>1</sup> to be 100W, we can reduce it by 11.9%. Although the reduction is not as significant as a data-centric setting, we can still achieve a relative speed-up. These demonstrate that our HEXA-MoE can essentially maximize the utilization of heterogeneous computing resources.

# 4.3. Ablation Studies

We examine the effectiveness of each component for HEXA-MoE via measuring the impact of expert-specific operators, pipeline-shared cache, fused kernel, data- & modelcentric and memory optimization, respectively. 2 metrics are evaluated, including average latency and memory footprint. We take distributed training of Swin-MoE-Base model on homogeneous devices as benchmark for ablation studies.

<span id="page-7-2"></span><span id="page-7-1"></span>![](_page_7_Figure_1.jpeg)

Figure 8. Latency analysis on heterogeneous devices with both data-centric and model-centric configuration. Experiments are all conducted with Swin-MoE-Small model using 4 global experts and top-2 routing. The average latency falls minimal when the division proportion is close to the computing capacity proportion in each case.

Since dispatch & combine operations are inevitable if we dispense with our expert-specific operators, we directly take the performance of Tutel in this case.

**Memory Footprint Breakdown.** We take 8 experts, top-4 routing, and batch size 40 as an example to break down memory footprint, and visualize the results in Figure 9(a). We can draw some insights from it:

- \* The employment of pipeline-shared cache slightly increases memory footprint, while expert-specific fused kernel has no impact.
- \* Pipeline-shared cache can effectively reduce memory footprint under a data-centric setting, since the memory footprint would surpass Tutel without it.
- HEXA-MoE can reduce memory footprint compared to baseline even without memory optimization, and optimizing it can further reduce memory footprint.

**Latency Breakdown.** We set 4 experts, top-4 routing, and batch size 80 to break down training latency, and visualize results in Figure 9(b). We can draw some insights as:

- \* The employment of expert-specific fused kernel, data-centric setting, and communication-computation overlap can effectively reduce latency. Meanwhile, the combination of data-centric setting and communication-computation overlap can further empower our HEXA-MoE with notably reduced latency.
- \* Expert-specific operators play a significant part in speeding up MoE computing since the latency for Tutel is much longer than our model-centric setting.
- \* Although pipeline-shared cache and memory optimization can slightly increase latency, they can also lead to significant reduction on memory footprint, as demonstrated in memory footprint breakdown.

<span id="page-7-3"></span><span id="page-7-0"></span>![](_page_7_Figure_12.jpeg)

<span id="page-7-4"></span>Figure 9. Memory and latency breakdown. We analyze the effectiveness of our proposed pipeline-shared cache, fused kernel, and memory optimization. We take Tutel as a baseline and visualize the differences with it.

### **5. Conclusion**

We introduce HEXA-MoE, a completely static MoE training framework with minimal memory footprint and heterogeneous-awareness, materialized by the proposed *expert-specific* operators and tensor parallelism. We provide specialized GPU kernels, which can both save memory footprint and reduce overall latency. For different workload scales, we provide data- and model-centric configurations for enhanced efficiency, and propose a memory-efficient communication-computation overlap-

ping scheme to overcome the shortcomings of previous work. Homogeneous experiments show that HEXA-MoE can save 10%-48% memory footprint while achieving 0.5- 4.3× speed up compared to state-of-the-art MoE libraries. Heterogeneous experiments show that HEXA-MoE can substantially minimize runtime and better utilize global computing resources by employing optimal parallel configuration. HEXA-MoE endows MoE computing with an in-place manner, where routing choice would yield almost no impact on hardware workload, which may inspire further research on algorithm designs of MoE.

# Impact Statement

This paper aims to improve the computation efficiency for training MoE models. The efficiency advantage of such models might help democratize access of large-scale foundation models. On the other hand, whether such new algorithm would affect known issues such as biased and harmful outputs of language models remains an unexplored research question.

# References

- <span id="page-8-15"></span>Bao, H., Wang, W., Dong, L., Liu, Q., Mohammed, O. K., Aggarwal, K., Som, S., Piao, S., and Wei, F. Vlmo: Unified vision-language pre-training with mixture-ofmodality-experts. *Advances in Neural Information Processing Systems*, 35:32897–32912, 2022.
- <span id="page-8-17"></span>Choquette, J., Giroux, O., and Foley, D. Volta: Performance and programmability. *Ieee Micro*, 38(2):42–52, 2018.
- <span id="page-8-13"></span>Du, N., Huang, Y., Dai, A. M., Tong, S., Lepikhin, D., Xu, Y., Krikun, M., Zhou, Y., Yu, A. W., Firat, O., et al. Glam: Efficient scaling of language models with mixture-ofexperts. In *International Conference on Machine Learning*, pp. 5547–5569. PMLR, 2022.
- <span id="page-8-14"></span>Fan, Z., Sarkar, R., Jiang, Z., Chen, T., Zou, K., Cheng, Y., Hao, C., Wang, Z., et al. M<sup>3</sup>vit: Mixture-of-experts vision transformer for efficient multi-task learning with modelaccelerator co-design. *Advances in Neural Information Processing Systems*, 35:28441–28457, 2022.
- <span id="page-8-12"></span>Fedus, W., Zoph, B., and Shazeer, N. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. *Journal of Machine Learning Research*, 23(120):1–39, 2022.
- <span id="page-8-8"></span>Gale, T., Narayanan, D., Young, C., and Zaharia, M. Megablocks: Efficient sparse training with mixture-ofexperts. *Proceedings of Machine Learning and Systems*, 5:288–304, 2023.
- <span id="page-8-16"></span>He, J., Qiu, J., Zeng, A., Yang, Z., Zhai, J., and Tang, J.

- Fastmoe: A fast mixture-of-expert training system. *arXiv preprint arXiv:2103.13262*, 2021.
- <span id="page-8-9"></span>He, J., Zhai, J., Antunes, T., Wang, H., Luo, F., Shi, S., and Li, Q. Fastermoe: modeling and optimizing training of large-scale dynamic pre-trained models. In *Proceedings of the 27th ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming*, pp. 120–134, 2022a.
- <span id="page-8-1"></span>He, K., Chen, X., Xie, S., Li, Y., Dollar, P., and Girshick, ´ R. Masked autoencoders are scalable vision learners. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pp. 16000–16009, 2022b.
- <span id="page-8-7"></span>Hwang, C., Cui, W., Xiong, Y., Yang, Z., Liu, Z., Hu, H., Wang, Z., Salas, R., Jose, J., Ram, P., et al. Tutel: Adaptive mixture-of-experts at scale. *Proceedings of Machine Learning and Systems*, 5:269–287, 2023.
- <span id="page-8-5"></span>Jiang, A. Q., Sablayrolles, A., Roux, A., Mensch, A., Savary, B., Bamford, C., Chaplot, D. S., Casas, D. d. l., Hanna, E. B., Bressand, F., et al. Mixtral of experts. *arXiv preprint arXiv:2401.04088*, 2024.
- <span id="page-8-4"></span>Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R., Gray, S., Radford, A., Wu, J., and Amodei, D. Scaling laws for neural language models. *arXiv preprint arXiv:2001.08361*, 2020.
- <span id="page-8-6"></span>Lepikhin, D., Lee, H., Xu, Y., Chen, D., Firat, O., Huang, Y., Krikun, M., Shazeer, N., and Chen, Z. Gshard: Scaling giant models with conditional computation and automatic sharding. In *International Conference on Learning Representations*, 2020.
- <span id="page-8-2"></span>Li, J., Li, D., Xiong, C., and Hoi, S. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In *International conference on machine learning*, pp. 12888–12900. PMLR, 2022.
- <span id="page-8-10"></span>Li, J., Jiang, Y., Zhu, Y., Wang, C., and Xu, H. Accelerating distributed {MoE} training and inference with lina. In *2023 USENIX Annual Technical Conference (USENIX ATC 23)*, pp. 945–959, 2023.
- <span id="page-8-3"></span>Liu, H., Li, C., Wu, Q., and Lee, Y. J. Visual instruction tuning. *Advances in neural information processing systems*, 36, 2024.
- <span id="page-8-11"></span>Liu, J., Wang, J. H., and Jiang, Y. Janus: A unified distributed training framework for sparse mixture-of-experts models. In *Proceedings of the ACM SIGCOMM 2023 Conference*, pp. 486–498, 2023.
- <span id="page-8-0"></span>Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., and Guo, B. Swin transformer: Hierarchical vision transformer using shifted windows. In *Proceedings of the*

- *IEEE/CVF international conference on computer vision*, pp. 10012–10022, 2021.
- <span id="page-9-2"></span>Narayanan, D., Shoeybi, M., Casper, J., LeGresley, P., Patwary, M., Korthikanti, V., Vainbrand, D., Kashinkunti, P., Bernauer, J., Catanzaro, B., et al. Efficient large-scale language model training on gpu clusters using megatronlm. In *Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis*, pp. 1–15, 2021.
- <span id="page-9-1"></span>Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., and Liu, P. J. Exploring the limits of transfer learning with a unified text-to-text transformer. *Journal of machine learning research*, 21 (140):1–67, 2020.
- <span id="page-9-3"></span>Riquelme, C., Puigcerver, J., Mustafa, B., Neumann, M., Jenatton, R., Susano Pinto, A., Keysers, D., and Houlsby, N. Scaling vision with sparse mixture of experts. *Advances in Neural Information Processing Systems*, 34: 8583–8595, 2021.
- <span id="page-9-7"></span>Shi, S., Pan, X., Chu, X., and Li, B. Pipemoe: Accelerating mixture-of-experts through adaptive pipelining. In *IEEE INFOCOM 2023-IEEE Conference on Computer Communications*, pp. 1–10. IEEE, 2023.
- <span id="page-9-4"></span>Shi, S., Pan, X., Wang, Q., Liu, C., Ren, X., Hu, Z., Yang, Y., Li, B., and Chu, X. Schemoe: An extensible mixture-ofexperts distributed training system with tasks scheduling. In *Proceedings of the Nineteenth European Conference on Computer Systems*, pp. 236–249, 2024.
- <span id="page-9-0"></span>Vaswani, A. Attention is all you need. *Advances in Neural Information Processing Systems*, 2017.
- <span id="page-9-6"></span>Zhai, M., He, J., Ma, Z., Zong, Z., Zhang, R., and Zhai, J. {SmartMoE}: Efficiently training {Sparsely-Activated} models through combining offline and online parallelization. In *2023 USENIX Annual Technical Conference (USENIX ATC 23)*, pp. 961–975, 2023.
- <span id="page-9-5"></span>Zhang, Z., Xia, Y., Wang, H., Yang, D., Hu, C., Zhou, X., and Cheng, D. Mpmoe: Memory efficient moe for pretrained models with adaptive pipeline parallelism. *IEEE Transactions on Parallel and Distributed Systems*, 2024.

### A. Principles of CUDA Programming

**Introduction to CUDA Programming:** A GPU program can be viewed as a thread grid in Figure 10(a), which is 3-dimensional and contains massive thread blocks. A thread block is also 3-dimensional and contains up to 1024 threads. Computation is mapped to these threads and implemented in parallel. Physically a GPU is composed of massive streaming multiprocessors (SM), where we show the memory hierarchy in Figure 10(b). Each SM loads data from global memory to its registers for parallel computing, and caching with shared memory can be faster. An SM contains many processing units including CUDA core and Tensor core. The former is used for general parallel computing, while the latter is specialized for matrix multiplication with mixed precision, which is first introduced in NVIDIA Volta architecture (Choquette et al., 2018). During execution, an SM is given one or more thread blocks, which are partitioned into warps and each warp gets scheduled by a warp scheduler.

<span id="page-10-2"></span>Mathematical Symbols: We provide the description of mathematical symbols used in this paper in Table 3, including variables in MoE computing and constant values in CUDA programming. To utilize Tensor core for faster matrix processing, we have to call the nvcuda::wmma (warped matrix multiplication and add) interface to compute a fixed-sized matrix multiplication such as  $16\times16\times16$  in a single warp, therefore the number of threads in a block has to be divisible by the constant value WARP. In the CUDA implementation of our method, a thread block loads BLK tokens each time from global memory to shared memory to conduct the parallelized computation.

| Table 3. | Description | of symbols. |
|----------|-------------|-------------|
|----------|-------------|-------------|

| Symbol                                          | Description                                                         |  |  |  |  |  |
|-------------------------------------------------|---------------------------------------------------------------------|--|--|--|--|--|
| $\overline{E}$                                  | Number of global experts.                                           |  |  |  |  |  |
| $\overline{}$                                   | Number of the input tokens.                                         |  |  |  |  |  |
| $\mathcal{F}$                                   | Activation function between the two MLPs.                           |  |  |  |  |  |
| $\overline{\mathcal{F}'}$                       | Element-wise differential for $\mathcal{F}$ .                       |  |  |  |  |  |
| ·                                               | Element-wise product.                                               |  |  |  |  |  |
| $D_i, D_o$                                      | Input and output size of the FFN.                                   |  |  |  |  |  |
| x                                               | Data batch with $N$ tokens.                                         |  |  |  |  |  |
| $\boldsymbol{x}^e, N_e$                         | Tokens in $\boldsymbol{x}$ routed to expert $e$ with number $N_e$ . |  |  |  |  |  |
| $\{\mathcal{R}_i(\boldsymbol{x})\}_{i=0}^{k-1}$ | Routing choice for $\boldsymbol{x}$ under top- $k$ routing.         |  |  |  |  |  |
| $W_1, W_2$                                      | Weights for 2 MLPs, shaped as $(E, D_1, D_2)$ .                     |  |  |  |  |  |
| $b_1, b_2$                                      | Biases of the two MLPs, shaped as $(E, D)$ .                        |  |  |  |  |  |
| BLK                                             | Block size for expert-specific operators.                           |  |  |  |  |  |
| WARP                                            | Warp size in a thread block.                                        |  |  |  |  |  |
| TIMES                                           | A thread block has WARP×TIMES threads.                              |  |  |  |  |  |

<span id="page-10-0"></span>![](_page_10_Figure_6.jpeg)

(a) Thread hierarchy for CUDA programming.

<span id="page-10-1"></span>(b) Memory hierarchy for a GPU.

Figure 10. Thread hierarchy for CUDA programming and memory hierarchy for a typical NVIDIA GPU.

# <span id="page-11-0"></span>B. Algorithm Details

```
Algorithm 1 Constructing re-index vector
  Input: Routing choice R with shape (N,).
  Initialize: Tensor ctr with shape (E,) initialized with 0.
  Parallel for i = 0 to N − 1 do
     atomicAdd(ctr[R[i]], 1)
  end for
  Parallel for i = 0 to E − 1 do
     ctr[i] = BLK · ⌊ctr[i] / BLK⌋
  end for
       N′ =
             E
              P−1
              i=0
                  ctr[i]
  Initialize: Tensor v with shape (N′
                                      ,) initialized with -1, and tensor idx with shape (1 + E,) initialized with 0.
  Parallel for i = 0 to E − 1 do
     ctr[i] = BLK · ⌊ctr[i] / BLK⌋
  end for
  for i = 1 to E do
     idx[i] = idx[i − 1] + ctr[i − 1]
  end for
  Copy: idx = idx
  Parallel for i = 0 to N − 1 do
     pos = atomicAdd(idx[R[i]], 1)
     v[pos] = i
  end for
  Output: Tensor v and idx
```

### Algorithm 2 Expert-specific matrix multiplication

```
Input: Routing choice R with shape (N,), vector v with length N′
                                                                  , input tokens x with shape (N, D1), weights w with
shape (E, D1, D2) and bias b with shape (E, D2)
Initialize: Output tokens y with shape (N, D2) initialized with 0.
Parallel for i in range (0, N′
                             , BLK) do
  Parallel for j in range (0, D2, BLK) do
     exp = R[v[i]]
    Initialize zero tensor c with shape (BLK, BLK)
    load bsub = b[exp, j : j + BLK]
     c = bsub.repeat(BLK, 1)
    for k in range (0, D1, BLK) do
       Initialize zero tensor xsub with shape (BLK, BLK)
       Parallel for t = 0 to BLK do
         load xsub[t] = x[v[i + t], k : k + BLK]
       end for
       load wsub = w[exp, k : k + BLK, j : j + BLK]
        c = c + xsub · wsub
    end for
    Parallel for t = 0 to BLK do
       Write back: y[v[i + t], j : j + BLK] = c[t]
    end for
  end for
end for
Output: Tensor y
```

#### Algorithm 3 Expert-specific summation

```
Input: Routing choice \mathcal{R} with shape (N, ), vector v with length N', vector idx with length 1 + E and input tokens x
with shape (N, D)
Initialize: Output tokens y with shape (E, D) initialized with 0.
Parallel for i = 0 to E - 1 do
  Parallel for j in range (0, D, BLK) do
      exp = \mathcal{R}[v[idx[i]]]
     Initialize zero tensor c with shape (1, BLK)
     Initialize zero tensor x_{sub} with shape (BLK, BLK)
     for k in range (idx[i],idx[i+1],\operatorname{BLK}) do
       Parallel for t=0 to BLK do
          load x_{sub}[t] = x[v[k+t], j:j+BLK]
       end for c = c + \sum_{t=0}^{\mathtt{BLK}-1} x_{sub}[t]
  Write back: y[exp, j: j + BLK] = c
  end for
end for
Output: Tensor y
```

#### **Algorithm 4** Expert-specific transposed matrix multiplication

```
Input: Routing choice \mathcal{R} with shape (N,), vector v with length N', vector idx with length 1+E, the first token batch
x_1 with shape (N, D_1) and the second token batch x_2 with shape (N, D_2)
Initialize: Output y with shape (E, D_1, D_2) initialized with 0.
Parallel for i = 0 to E - 1 do
   Parallel for m in range (0, D_1, BLK) do
      Parallel for n in range (0, D_2, BLK) do
           exp = \mathcal{R}[v[idx[i]]]
          Initialize zero tensor c with shape (BLK, BLK)
          \begin{array}{l} \textbf{Initialize} \ \text{zero tensor} \ x_{sub}^1 \ \text{with shape} \ (\texttt{BLK}, \texttt{BLK}) \\ \textbf{Initialize} \ \text{zero tensor} \ x_{sub}^2 \ \text{with shape} \ (\texttt{BLK}, \texttt{BLK}) \end{array}
          for k in range (idx[i], idx[i+1], BLK) do
             Parallel for t=0 to BLK do
                  \begin{array}{l} \mathbf{load} \ x_{sub}^1[t] = x_1[v[k+t], m:m + \mathtt{BLK}] \\ \mathbf{load} \ x_{sub}^2[t] = x_2[v[k+t], n:n + \mathtt{BLK}] \end{array} 
              c = c + x_{sub}^1.transpose() \cdot x_{sub}^2
          end for
          Write back in parallel:
               y[exp, m: m + BLK, n: n + BLK] = c
      end for
   end for
end for
Output: Tensor y
```

We present the algorithm details for re-index vector construction as well as the expert-specific operators, and taking top-1 routing as an example for illustration, shown in Algorithm 1, 2, 3 and 4. In Algorithm 1, we re-arrange the routing choice vector  $\mathcal{R}$  into re-indexed token vector v, along with the token index starting vector idx, which satisfies idx[0] = 0 and idx[E] = N'. We provide the length and range for the vectors in Table 4.

In Algorithm 2, 3 and 4, we assume that the feature dimension for each tensor is all divisible by BLK. Notice that when

Table 4. Explanation for the auxiliary vectors.

<span id="page-13-1"></span>

| vector         | length   | range                                                   |
|----------------|----------|---------------------------------------------------------|
| $\mathcal{R}$  | $\mid N$ | $0 \le \mathcal{R}[i] < E \text{ for } 0 \le i < N$     |
| $\overline{v}$ | N'       | $\big  \qquad 0 \le v[i] < N \text{ for } 0 \le i < N'$ |
| idx            | 1+E      | $0 \le idx[i] < N' \text{ for } 0 \le i < 1 + E$        |

accessing the re-index vector v, we may get value -1, since the workload for each expert is dynamic. In this case, we would skip this index, and remain zero for the temporary loading variable.

We also provide the formulation for top-k routing with our expert-specific operators, and take a comparison with top-1 routing. We denote the routing choice for top-k as  $\{\mathcal{R}_i(\boldsymbol{x})\}_{i=0}^{k-1}$ , and other symbols keep consistent with Figure 2. We present the comparison on formulations for MoE forward and backward propagation in Table 5.

<span id="page-13-2"></span> $\it Table~5.$  Comparison between top-1 and top- $\it k$  routing for the formulation of MoE forward and backward with our expert-specific operators.

| Stage    | Notation Layer |            | Expert-Specific Formulation (top-1 routing)                                                                                                                             | Expert-Specific Formulation (top-k routing)                                                                                                                                                                                                                                             |  |  |  |  |
|----------|----------------|------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--|--|--|--|
|          | 1              | 1st MLP    | $\boldsymbol{y}_1 = \textit{ESMM}(\boldsymbol{x}, \boldsymbol{W}_1, \boldsymbol{b}_1, \mathcal{R}(\boldsymbol{x}))$                                                     | $\bigg  \{\boldsymbol{y}_1^i\}_{i=0}^{k-1}: \boldsymbol{y}_1^i = \textit{ESMM}(\boldsymbol{x}, \boldsymbol{W}_1, \boldsymbol{b}_1, \mathcal{R}_i(\boldsymbol{x}))$                                                                                                                      |  |  |  |  |
| Forward  | 2              | Activation | $\boldsymbol{y}_2 = \mathcal{F}(\boldsymbol{y}_1)$                                                                                                                      | $\big \{\boldsymbol{y}_2^i\}_{i=0}^{k-1}:\boldsymbol{y}_2^i=\mathcal{F}(\boldsymbol{y}_1^i)$                                                                                                                                                                                            |  |  |  |  |
|          | 3              | 2nd MLP    | $\boldsymbol{y} = \textit{ESMM}(\boldsymbol{y}_2, \boldsymbol{W}_2, \boldsymbol{b}_2, \mathcal{R}(\boldsymbol{x}))$                                                     | $\boxed{\boldsymbol{y} = \sum_{i=0}^{k-1} \textit{ESMM}(\boldsymbol{y}_2^i, \boldsymbol{W}_2, \boldsymbol{b}_2, \mathcal{R}_i(\boldsymbol{x}))}$                                                                                                                                        |  |  |  |  |
|          | 4              |            | $\frac{\partial \ell}{\partial \boldsymbol{b}_2} = \textit{ESS}(\frac{\partial \ell}{\partial \boldsymbol{y}}, \mathcal{R}(\boldsymbol{x}))$                            | $\frac{\partial \ell}{\partial \boldsymbol{b}_2} = \sum_{i=0}^{k-1} \textit{ESS}(\frac{\partial \ell}{\partial \boldsymbol{y}}, \mathcal{R}_i(\boldsymbol{x}))$                                                                                                                         |  |  |  |  |
|          | (5)            | 2nd MLP    | $\frac{\partial \ell}{\partial \boldsymbol{W}_1} = \textit{ESTMM}(\boldsymbol{x}, \frac{\partial \ell}{\partial \boldsymbol{y}_1}, \mathcal{R}(\boldsymbol{x}))$        | $\frac{\partial \ell}{\partial \boldsymbol{W}_1} = \sum_{i=0}^{k-1} \textit{ESTMM}(\boldsymbol{x}, \frac{\partial \ell}{\partial \boldsymbol{y}_1^i}, \mathcal{R}_i(\boldsymbol{x}))$                                                                                                   |  |  |  |  |
| Backward | 6              |            | $\frac{\partial \ell}{\partial \boldsymbol{y}_2} = \textit{ESMM}(\frac{\partial \ell}{\partial \boldsymbol{y}}, \boldsymbol{W}_2^T, null, \mathcal{R}(\boldsymbol{x}))$ | $   \left\{ \frac{\partial \ell}{\partial \boldsymbol{y}_{2}^{i}} \right\}_{i=0}^{k-1} : \frac{\partial \ell}{\partial \boldsymbol{y}_{2}^{i}} = \textit{ESMM}(\frac{\partial \ell}{\partial \boldsymbol{y}}, \boldsymbol{W}_{2}^{T}, \textit{null}, \mathcal{R}_{i}(\boldsymbol{x})) $ |  |  |  |  |
| Buckward | 7              | Activation | $\frac{\partial \ell}{\partial \boldsymbol{y}_1} = \frac{\partial \ell}{\partial \boldsymbol{y}_2} \odot \mathcal{F}'(\boldsymbol{y}_1)$                                | $\left\{\frac{\partial \ell}{\partial \boldsymbol{y}_1^i}\right\}_{i=0}^{k-1}:\frac{\partial \ell}{\partial \boldsymbol{y}_1^i}=\frac{\partial \ell}{\partial \boldsymbol{y}_2^i}\odot\mathcal{F}'(\boldsymbol{y}_1^i)$                                                                 |  |  |  |  |
|          | 8              |            | $\frac{\partial \ell}{\bm{b}_1} = \textit{ESS}(\frac{\partial \ell}{\partial \bm{y}_1}, \mathcal{R}(\bm{x}))$                                                           | $\frac{\partial \ell}{\boldsymbol{b}_1} = \sum_{i=0}^{k-1} \textit{ESS}(\frac{\partial \ell}{\partial \boldsymbol{y}_1^i}, \mathcal{R}_i(\boldsymbol{x}))$                                                                                                                              |  |  |  |  |
|          | 9              | 1st MLP    | $\frac{\partial \ell}{\partial \boldsymbol{W}_1} = \textit{ESTMM}(\boldsymbol{x}, \frac{\partial \ell}{\partial \boldsymbol{y}_1}, \mathcal{R}(\boldsymbol{x}))$        | $\frac{\partial \ell}{\partial \boldsymbol{W}_1} = \sum_{i=0}^{k-1} \textit{ESTMM}(\boldsymbol{x}, \frac{\partial \ell}{\partial \boldsymbol{y}_1^i}, \mathcal{R}_i(\boldsymbol{x}))$                                                                                                   |  |  |  |  |
|          | 10             |            | $\frac{\partial \ell}{\partial \boldsymbol{x}} = \textit{ESMM}(\frac{\partial \ell}{\partial \boldsymbol{y}_1}, \boldsymbol{W}_1^T, null, \mathcal{R}(\boldsymbol{x}))$ | $ \frac{\partial \ell}{\partial \boldsymbol{x}} = \sum_{i=0}^{k-1} \textit{ESMM}(\frac{\partial \ell}{\partial \boldsymbol{y}_1^i}, \boldsymbol{W}_1^T, \textit{null}, \mathcal{R}_i(\boldsymbol{x})) $                                                                                 |  |  |  |  |

### <span id="page-13-0"></span>C. Experimental Details

We provide the details of our CUDA program via enumerating the shape of the thread block and thread grid for expert-specific operators in a single MoE layer in Table 6.

We also provide the PyTorch-style pseudocode for the proxy task we used to examine the computing capacity of the heterogeneous devices, as shown in Algorithm 5. We adopt a for loop composed of large matrix multiplications with the same scale as the test program.

<span id="page-14-0"></span>Table 6. Shape of the thread block and thread grid for expert-specific operators in one MoE layer. We take top-1 routing as an example, where N' denotes the length of the re-index vector, which is slightly larger than N and is divisible by BLK. Thread blocks are all defined with the same shape to facilitate the fused kernel.

|          | Operator       | Input                                                                                                                                                                | Output                                                  | Thread Block     | Thread Grid                                                                                                                                                                                |  |  |
|----------|----------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------|------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--|--|
| forward  | <br>  ESMM<br> | $\begin{array}{c c c} & \boldsymbol{x} & (N,D_1) \\ \hline & \boldsymbol{w} & (E,D_1,D_2) \\ \hline & \boldsymbol{b} & (E,D_2) \\ \hline \end{array} \boldsymbol{y}$ | $  (N, D_2)  $                                          | (WARP,<br>TIMES) | $\big( \lceil N' / \texttt{BLK} \rceil, \ \lceil D_2 / (\texttt{TIMES} \cdot \texttt{BLK}) \rceil \big)$                                                                                   |  |  |
|          | ESMM           | $\begin{array}{ c c c c c c c c c c c c c c c c c c c$                                                                                                               | $(N, D_1)$                                              | (WARP,<br>TIMES) | $\left[ \begin{array}{c} \left( \lceil N' / \mathtt{BLK} \rceil, \\ \lceil D_1 / (\mathtt{TIMES} \cdot \mathtt{BLK}) \rceil \right) \end{array} \right]$                                   |  |  |
| vard     | ESS            | $  x   (N, D_2)   y$                                                                                                                                                 | $(E,D_2)$                                               | (WARP,<br>TIMES) | $(E, \lceil D_2/(\mathtt{TIMES} \cdot \mathtt{BLK}) \rceil)$                                                                                                                               |  |  |
| backward | ESTMM          | $\begin{array}{ c c c c c c c c c c c c c c c c c c c$                                                                                                               | $\left \begin{array}{c} (E,D_1,D_2) \end{array}\right $ | (WARP,<br>TIMES) | $D_1/(\text{TIMES} \cdot \text{BLK}),$                                                                                                                                                     |  |  |
|          | <br>  ESFK     | $ \begin{array}{c c c c c c c c c c c c c c c c c c c $                                                                                                              | $   (N, D_1) $ $   (E, D_2) $ $   (E, D_1, D_2) $       | (WARP,           | $(E, \lceil D_1/(\texttt{TIMES} \cdot \texttt{BLK}) \rceil, \\ \lceil N'/\texttt{BLK} \rceil + \lceil D_2/\texttt{BLK} \rceil + \\ \lceil D_2/(\texttt{TIMES} \cdot \texttt{BLK}) \rceil)$ |  |  |

#### Algorithm 5 PyTorch-style pseudocode of MoE pipeline.

```
import torch\nimport time

device = 'cuda'
size = 2048
times = 1024

start_time = time.time()
for j in range(times):
    mat1 = torch.randn(size, size, device=device)
    mat2 = torch.randn(size, size, device=device)
    y = torch.matmul(mat1, mat2)\nend_time = time.time()

print(end_time - start_time)
```

# **D.** Experimental Results

We provide the exact values for both memory footprint analysis and average latency analysis in Table 7 and 8, respectively. Specifically, for latency analysis, we provide the average value for each case with 0.5k, 1k, 1.5k and 2k total steps.

<span id="page-15-0"></span>Table 7. Memory analysis with Tutel, MegaBlocks and **HEXA-MoE** on Swin-Transformer-MoE benchmark (Base and Small). Experiments are conducted on 2 homogeneous GPUs with automatic mixed precision in PyTorch and batch size 40 for all the experiments. We set the number of global experts to 8, and record the average GPU memory footprint (GB) on each device.

|       | Method               | top-1 | top-2 | top-3 | top-4 | top-5 | top-6 | top-7 | top-8 |
|-------|----------------------|-------|-------|-------|-------|-------|-------|-------|-------|
|       | Tutel                | 12.7  | 13.9  | 15.3  | 16.0  | 17.4  | 19.0  | 20.3  | 21.8  |
| Base  | MegaBlocks (MoE)     | 13.1  | 13.8  | 14.8  | 15.4  | 16.0  | 16.9  | 17.8  | 18.7  |
|       | MegaBlocks (dMoE)    | 12.9  | 13.7  | 14.9  | 15.6  | 16.7  | 17.8  | 18.6  | 19.7  |
|       | Ours (data-centric)  | 10.9  | 11.2  | 11.3  | 11.7  | 12.0  | 12.0  | 12.3  | 12.4  |
|       | Ours (model-centric) | 10.0  | 10.2  | 10.3  | 10.6  | 10.5  | 10.7  | 10.9  | 11.4  |
|       | Tutel                | 9.0   | 10.0  | 11.0  | 11.6  | 12.7  | 13.8  | 15.0  | 16.0  |
| Small | MegaBlocks (MoE)     | 9.2   | 9.8   | 10.2  | 10.8  | 11.2  | 11.8  | 12.5  | 13.0  |
|       | MegaBlocks (dMoE)    | 9.0   | 9.7   | 10.4  | 11.4  | 12.0  | 12.4  | 13.3  | 13.9  |
|       | Ours (data-centric)  | 8.1   | 8.3   | 8.2   | 8.5   | 8.6   | 8.7   | 9.0   | 9.2   |
|       | Ours (model-centric) | 7.7   | 7.8   | 7.8   | 8.0   | 7.9   | 8.2   | 8.3   | 8.5   |

<span id="page-16-0"></span>Table 8. Latency analysis for Tutel, MegaBlocks and HEXA-MoE on Swin-Transformer-MoE benchmark with base and small scale. Experiments are conducted on 4 homogeneous GPUs with 4 experts. We set different batch size (bs) for different models under different routing strategy to maximize the utilization of GPU memory. We record the average latency of one step (s) during training.

|       |              | Method               | 0.5k | 1k   | 1.5k | 2k   |              | 0.5k | 1k   | 1.5k | 2k   |
|-------|--------------|----------------------|------|------|------|------|--------------|------|------|------|------|
|       | 0            | Tutel                | 2.96 | 2.90 | 2.47 | 2.40 | 0            | 2.14 | 2.59 | 2.73 | 2.66 |
|       | top-1,bs=110 | MegaBlocks (MoE)     | 2.10 | 2.66 | 2.57 | 2.43 | top-2,bs=100 | 2.40 | 2.58 | 2.51 | 2.49 |
|       | -dc          | MegaBlocks (dMoE)    | 2.06 | 2.02 | 2.13 | 2.19 | p-2,l        | 2.47 | 2.72 | 2.63 | 2.55 |
| Base  | =            | Ours (model-centric) | 1.52 | 1.51 | 1.51 | 1.51 | t [          | 1.61 | 1.60 | 1.60 | 1.60 |
| В     |              | Ours (data-centric)  | 1.01 | 0.99 | 0.99 | 0.99 |              | 1.17 | 1.16 | 1.16 | 1.16 |
|       |              | Tutel                | 2.23 | 2.43 | 2.48 | 2.54 |              | 2.27 | 2.34 | 2.20 | 2.18 |
|       | top-3,bs=90  | MegaBlocks (MoE)     | 2.09 | 2.31 | 2.35 | 2.17 | )8=sq        | 2.11 | 2.18 | 2.05 | 2.03 |
|       | <br>pp-3,l   | MegaBlocks (dMoE)    | 2.55 | 2.53 | 2.44 | 2.41 | top-4,bs=80  | 2.00 | 2.19 | 2.23 | 2.12 |
|       | =            | Ours (model-centric) | 1.70 | 1.69 | 1.69 | 1.69 |              | 1.83 | 1.82 | 1.82 | 1.82 |
|       |              | Ours (data-centric)  | 1.32 | 1.31 | 1.31 | 1.30 |              | 1.42 | 1.41 | 1.41 | 1.41 |
|       | 0            | Tutel                | 3.59 | 3.63 | 3.62 | 3.01 | 0            | 2.83 | 2.96 | 2.72 | 2.53 |
|       | <br> S=14    | MegaBlocks (MoE)     | 3.41 | 3.64 | 3.72 | 3.23 | S=13         | 3.04 | 3.26 | 3.11 | 3.02 |
|       | top-1,bs=140 | MegaBlocks (dMoE)    | 3.51 | 3.58 | 3.76 | 3.47 | top-2,bs=130 | 2.25 | 3.10 | 3.01 | 2.87 |
| Small | 2            | Ours (model-centric) | 1.34 | 1.34 | 1.33 | 1.33 | =            | 1.49 | 1.49 | 1.48 | 1.48 |
| Sī    |              | Ours (data-centric)  | 0.69 | 0.68 | 0.68 | 0.68 |              | 0.87 | 0.86 | 0.86 | 0.85 |
|       | 0            | Tutel                | 2.23 | 2.92 | 3.17 | 3.28 | 0            | 3.09 | 2.86 | 3.03 | 3.00 |
|       | top-3,bs=120 | MegaBlocks (MoE)     | 2.63 | 2.79 | 2.92 | 3.07 | top-4,bs=110 | 3.04 | 3.09 | 3.01 | 2.91 |
|       |              | MegaBlocks (dMoE)    | 3.08 | 3.24 | 3.38 | 3.09 | p-4,k        | 3.03 | 3.19 | 3.18 | 2.91 |
|       | tc           | Ours (model-centric) | 1.61 | 1.60 | 1.59 | 1.59 | t t          | 1.67 | 1.67 | 1.68 | 1.68 |
|       |              | Ours (data-centric)  | 1.07 | 1.05 | 1.05 | 1.05 |              | 1.09 | 1.08 | 1.07 | 1.07 |