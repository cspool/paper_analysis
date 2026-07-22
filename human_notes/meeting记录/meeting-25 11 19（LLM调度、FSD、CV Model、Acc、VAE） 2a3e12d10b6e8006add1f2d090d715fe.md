# meeting-25/11/19（LLM调度、FSD、CV Model、Acc、VAE）

# 想法

云的服务模型；云边端；llm拆；

云边端拆LLM：llm的中间数据在边缘cache，多用户共享中间结果和参数；

共享方式：batch；很早batch，底层多样，高层共性没有暴露和共享；

后端SLM生成的结果共享；

隐私放到小模型；

movqgan中有conv

单纯卷性能已经有范式，优化重点在于应用场景的复杂多变；

im2col模块能增大llm、conv调度的选择？llm和conv的同卡调度会有好处吗？不会，不同机器应调度不同类型模型；

冷热LLM在多卡的调度场景，结合调度方法，作单卡体系结构层面的优化（互联、kernel）；

MaaS的两种调度：静态绑定GPU+动态分配GPU，多任务共享GPU；

layer之间的抢占、token之间的抢占，动态DNN；

边缘场景的多任务，自动驾驶的chain of thought，能引入DNN，conv在边缘侧推理；

IoT：每个核的PCIe；

自动驾驶、VR的多任务，体系结构优化？

slam；

多算法内核；

资源受限，DFG不一定最优，引入抢占；

DiT的DFG的输入token不同，动态；

输入的长度动态，资源受限；

MoE的动态FFN；

不同layout下的concate+shortcut（residual），多分支的数据依赖；

# 实验

## LayoutLoop & TimeLoop

[https://github.com/maeri-project/FEATHER/tree/main](https://github.com/maeri-project/FEATHER/tree/main)

[https://github.com/NVlabs/timeloop/tree/master](https://github.com/NVlabs/timeloop/tree/master)

[https://timeloop.csail.mit.edu/](https://timeloop.csail.mit.edu/)

# 动机

## ASPDAC25：KAPLA

张量程序的“**硬件可见**”的Mapping，**stack**表示空间并行，**update**表示时间迭代；

> **[图片提取文字 (image.png)]:**
> ```
> Listing 1: Directive examples for CONV and depthwise CONV
>  layers, with row-stationary PE mapping [3], output + batch
>  hybrid node parallelization [6], and layer pipelining [7].
> 1 CONV:
>   REGF:
>    tensor(0)(N=1, C=2, Xi=5, Yi=1)
>    tensor(w1)(C=2, K=3, R=5, S=1)
> 4
>    tensor{1}(N=1, K=3, Xo=1, Yo=1)
> 5
>    stack(Yi+=1, Yo+=1, 8) % PE columns
> 6
>    stack(S+=1, Yi+=1, 5) % PE rows
>    update(Xi+=1, Xo+=1)                                    
> 8
>    update(Yi+=8, Yo+=8) % folding
> 9
>    update(N+=1)
> 10
> 11
>    update(C+=2)
> 12
>    update(K+=3)
> 13
>   GBUF:
> 14
>    tensor(0)(N=4, C=4, Xi=19, Yi=19, shr=4)
>    tensor(w1)(C=4, K=6, R=5, S=5)
> 15
>    tensor{1}(N=4, K=6, Xo=15, Yo=15)
> 16
>    stack(K+=6, 4) % output node parallel
> 17
> 18
>    stack(N+=4, 16) % batch node parallel
> 19
>    update(C+=4)
> 20
>    update(K+=24)
>    update(N+=64)
> 21
> 22 DWCONV:
> 23
>   REGF:
> 24
>    % . . .
> 25
>   GBUF:
> 26
>    % DWCONV input is the same as CONV output
> 27
>    tensor{1}(N=4, C=4, Xi=9, Yi=15)
> 28
>    tensor(w2)(C=4, R=3, S=3)
> 29
>    tensor{2}(N=4, C=4, Xo=4, Yo=7)
> 30
>    stack(C+=4, 6) % channel node parallel
>    stack(N+=4, 16) % batch node parallel
> 31
> ```
> 
> stack(Xo+=4, 2) % output width node parallel
> 
> 32
> 
> 33 34
> 
> 35
> 
> update(Yo+=7)
> 
> update(C+=24)
> 
> update (N+=64)
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image.png)

> **[图片提取文字 (image.png)]:**
> optional sharing factor shr is used when the tensor is stored across multiple buffers in a shared manner and each buffer only holds 1/shr of the data [7]. **tensor** declarations are organized in a two-level name scope, across NN layers and across memory hierarchy levels (Listing 1). Within each layer, tensors with the same name but at different memory levels belong to the same full tensor, and tensors in faster levels are subsets of tensors in slower levels. Such naming allows us to infer the detailed tensor *layout* (e.g., tiling
> 
> strides) in each buffer from the dimension sizes of the tensors in
> 
> neighboring level buffers. For example, a tensor with C=3 and C=6
> 
> Directive definitions. tensor(dim=size, ...[, shr]) de-
> 
> clares a multi-dimensional (sub)tensor allocated at a certain level
> 
> of the memory hierarchy, with the given size along each dim. The
> 
> in two neighboring levels means that it should be tiled by 2. stack(dim+=shift, ..., repl) denotes spatial parallelization where repl PEs or nodes exist at a memory hierarchy level, each with a local buffer storing a copy of all tensors declared at this level, potentially offset by shifts along dims. Tensors are essentially stacked across repl buffers, or sharded. shifts need not match tensor sizes: smaller shifts partially overlap tensors, while larger
> 
> shifts distribute non-contiguous ranges. This enables fine-grained
> 
> interleaving or coarse-grained partitioning. Without shifts, ten-
> 
> sors are replicated. **stacks** at the same level are applied recursively in order, allowing hybrid and complex spatial parallelism. For instance, two **stacks** in Lines 6 and 7 at the REGF level of Listing 1 indicate a 2D PE array mapping.
> 
> update(dim+=step, ...) denotes ordered, nested temporal iterations in buffers, with tensors in each buffer updated synchronously
> 
> by increments of steps along their dimensions. This process replaces old data with new, and the step can either match the tensor shape or support overlapped windows and non-contiguous strides.
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%201.png)

对张量程序搜索：垂直segment、layer pipeline、node并行、node Mapping/loop优化

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 2: The hierarchical NN dataflow taxonomy.
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%202.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 4: An example of intra-layer stacking and caching.
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%203.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 3: The overall workflow of KAPLA. The blue, green, and red arrow chains represent the pruning, estimating, and scheduling processes, respectively.
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%204.png)

SeaCache

基于Cache的稀疏加速器；

Buffer Prospector：Buffer？

*

## HPCA25：Lincoln

Lincoln: Real-Time 50***∼***100B LLM Inference on Consumer Devices with LPDDR-Interfaced, Compute-Enabled Flash Memory

在消费设备LLM推理；

Hydrogen：面向Contention

*

Gemini：chiplet加速器的Mapping

*

Fast State Restoration：多任务恢复？

*

medusa：大模型推理 serverless？

*

## ATC25：Weaver（多GPU调度、MLS、attention-offload）

**Weaver: Efficient Multi-LLM Serving with Attention Offloading**

### idea

**冷热模型**，即不同GPU的不同LLM的请求特征不同，比如prompt和token阶段负担均衡（575：340）的balence负载、prompt阶段负担更重（749：232）的input-heavy负载；

执行input-heavy负载的prompt阶段的GPU在负载热模型，将其任务卸载到执行balance负载的token阶段的GPU（负载冷模型的GPU）；

**attention offload将attention layer拆分成含参数proj和无参数attention（解除pipeline对GPU的持续绑定）**，**主要设备计算投影**得到QKV（prompt/token和权重的矩阵乘法），得到的新的QKV传输到辅助设备（因此**KV-Cache存储在辅助设备**），辅助设备使用query计算KV、KV-Cache之间的关联（attention），得到的输出O回传到主要设备，然后循环；

**调度方法**：热模型GPU的attention计算卸载到冷模型GPU上执行；

主要技术：**多GPU调度**、GPU的冷热模型的**负载切换**（attention offload）；

> **[图片提取文字 (image.png)]:**
> ## Multi-LLM Serving Characteristic
> 
> ![](_page_0_Figure_1.jpeg)
> 
> Request loads are *skewed* between models
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%205.png)

> **[图片提取文字 (image.png)]:**
> ## **Existing Methods for Multi-LLM Serving**
> 
> ## #1 Dedicated Serving
> 
> ![](_page_0_Picture_2.jpeg)
> 
> ## #2 Model Parallelism
> 
> ![](_page_0_Picture_4.jpeg)
> 
> ![](_page_0_Picture_5.jpeg)
> 
> Underutilize GPU memory
> 
> ![](_page_0_Picture_7.jpeg)
> 
> 😕 Bring Communication overhead
> 
> Design Goal: maximize multi-LLM GPU memory utilization, without sacrificing performance.
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%206.png)

> **[图片提取文字 (image.png)]:**
> ## **Attention Computing**
> 
> ![](_page_0_Figure_1.jpeg)
> 
> History KV Cache
> 
> Only QKVO vector of the new token is needed
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%207.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%208.png)

### 方法

receiver CPU阻塞式优先调度sender的kernel，严重损耗receiver计算吞吐，让CPU在自身GPU kernel发射后接上一个polling kernel，polling kernel让GPU查看是否存在sender卸载的attention（类似**让receiver GPU轮询**），若存在则执行；

CPU不知道GPU的实时负载情况，让GPU完成自己kernel后查询任务列表，能保证自身任务的吞吐；

> **[图片提取文字 (image.png)]:**
> ## Our Approach
> 
> ## **Challenges**
> 
> C1:Blocked by pre-issued kernels
> 
> C2: Blocked by long kernel
> 
> ## **Solutions**
> 
> **GPU-driven Control Flow** 
> 
> Insert polling after each kernel
> 
> ## **Operator Splitting**
> 
> Break a large kernel into pieces
> 
> ![](_page_0_Picture_9.jpeg)
> 
> ![](_page_0_Picture_10.jpeg)
> 
> Achieve non-blocking attention offloading
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%209.png)

> **[图片提取文字 (image.png)]:**
> ## **GPU Driven Control Flow**
> 
> ![](_page_0_Figure_1.jpeg)
> 
> Offloaded computation is now blocked by only one kernel
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%2010.png)

> **[图片提取文字 (image.png)]:**
> ## **4.1 GPU-driven Dynamic Control Flow**To avoid the offloaded attention operator being blocked by
> 
> pre-issued kernels, one strawman approach would be to issue kernels in a blocking manner: only when a kernel is complete, the receiver CPU will issue the next kernel (prioritize issu-
> 
> ing the offload attention operator, if it exists). However, this
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%2011.png)

> **[图片提取文字 (image.png)]:**
> the sender needs to offload an attention operator, it writes the latest token's QKV tensor to shared memory using one-sided writes (1), and atomically increments a task counter (2). The receiver GPU polls the task counter (3). Upon detecting an increment, indicating a new task, the receiver GPU executes the attention operator, then writes the result tensor to shared memory (4), and finally increments a completion counter (**6**). Once the sender realizes that the completion counter has changed (**6**), it obtains the result tensor from shared memory and then continues with the subsequent computation. All the above operations of the receiver GPU, including polling, counter updates, data transfers, and attention execution, are encapsulated in a GPU kernel, which we call polling kernel. The receiver CPU pre-issues many kernels as in today's LLM serving systems to saturate the GPU hardware
> 
> queue, but each kernel is followed by a polling kernel. In this
> 
> way, the offloaded attention will only be blocked by the kernel
> 
> that is executing, not other pre-issued kernels.
> 
> With GPU-driven dynamic control flow, the process of
> 
> offloading an attention operator is shown in Figure 3. When
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%2012.png)

分割**长延迟kernel**，先评估分割后的等待时间，再决定是否分割kernel；

> **[图片提取文字 (image.png)]:**
> ## 4.2 Operator Splitting
> 
> kernel, WEAVER introduces operator splitting. Its key idea is to split the kernel of a large operator into smaller ones, so that the offloaded attention can be executed on GPUs within a bounded time. However, it is non-trivial to generate a good splitting plan, considering that 1) the operators in LLM serving have diverse execution time, 2) excessive fragmentation
> 
> To avoid head-of-line blocking resulting from a long-running
> 
> ing have diverse execution time, 2) excessive fragmentation due to operator splitting will induce performance overhead. To this end, we first employ queuing theory to model the sender's wait time, then adopt a priority-based algorithm to generate the splitting plan.
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%2013.png)

> **[图片提取文字 (image.png)]:**
> ## **Operator Splitting**
> 
> Simple Idea: Breaking large kernels
> 
> ![](_page_0_Figure_2.jpeg)
> 
> ## **Mathematic Observation**
> 
> Each Kernel *quadratically* contributes to the waiting time
> 
> 1 Operator priority queue: sort by the operator's running time
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%2014.png)

> **[图片提取文字 (image.png)]:**
> ## **Operator Splitting**
> 
> Simple Idea: Breaking large kernels
> 
> ![](_page_0_Figure_2.jpeg)
> 
> ## **Mathematic Observation**
> 
> Each Kernel *quadratically* contributes to the waiting time
> 
> ![](_page_0_Picture_5.jpeg)
> 
> ② Split the biggest operator into two halves and insert back
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%2015.png)

> **[图片提取文字 (image.png)]:**
> ## **Operator Splitting**
> 
> Simple Idea: Breaking large kernels
> 
> ![](_page_0_Figure_2.jpeg)
> 
> ## **Mathematic Observation**
> 
> Each Kernel *quadratically* contributes to the waiting time
> 
> ![](_page_0_Picture_5.jpeg)
> 
> ③ Reinsert into the queue. Repeat until the waiting time < threshold</p>
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%2016.png)

### 实验

> **[图片提取文字 (image.png)]:**
> ## **Evaluation**
> 
> - Baselines:
>   - Dedicated Serving<sub>[vLLM, v0.6.0]</sub>
>   - MuxServe (With MPS)
>   - MuxServe-Temporal
> - Testbed:
>   - A100-40GB (connected with NVLink)
>   - L40S (connected with PCIe)
> - Workloads:
>   - BurstGPT and Azure-Conversation
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%2017.png)

> **[图片提取文字 (image.png)]:**
> **Baselines.** We compare WEAVER to the following baselines. **1** Dedicated Serving, which uses dedicated GPUs to serve cold and hot models respectively, each running unmodified vLLM. **2** MuxServe [17], which supports two modes: 1) spatial-temporal multiplexing with NVIDIA MPS enabled and 2) temporary multiplexing only (i.e., Mux-Temporal). For
> 
> and 2) temporary multiplexing only (i.e., Mux-Temporal). For a fair comparison with MuxServe, we do not enable CUDA Graph in WEAVER and dedicated serving. Moreover, we also replace the attention kernel of MuxServe with more efficient
> 
> FlashAttention, which is the same as WEAVER.
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%2018.png)

> **[图片提取文字 (image.png)]:**
> Workloads and test models. We evaluate our system using two widely-used LLM workload traces: BurstGPT [31] (ChatGPT-4 split) and Azure-Conv [25]. We filter out sequences exceeding 2048 tokens, since some baselines do not support running such lengths. BurstGPT represents a balanced workload, with an average input length of 575 tokens and an output length of 340 tokens. In contrast, Azure-Conv represents an input-heavy workload, with an average input length of 749 tokens and an output length of 232 tokens. All experiments use the Llama-3-8B model by default. Following prior work [21], we sample the arrival time of requests with the Poisson distribution from the above traces. Unless otherwise stated, we set the request rate of the cold model to 1 request
> 
> per second and the offload ratio of WEAVER to 45%.
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%2019.png)

> **[图片提取文字 (image.png)]:**
> connect bandwidths: (i) 4× A100-40GB with NVLink; (ii) a cloud server containing  $4 \times L40S$  with PCIe as the interconnect. Following popular disaggregation serving design [34], each model (hot and cold) uses the 1p1d setup. We only multiplex the workload between decoding GPUs. On the L40S
> 
> **Testbeds.** We used two test platforms with different inter-
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%2020.png)

## ISCA24：Splitwise（集群调度、CLS ）

Splitwise: Efficient Generative LLM Inference Using Phase Splitting

将Conv和Coding（不同特征）的**request（端到端推理）调度到不同集群**，端到端的request调度浪费内存或算力资源，layer-wise调度的传输开销大且调度复杂而不适合集群调度，将**request粗粒度划分**为prompt阶段和token阶段后进行phase-wise的集群调度；

分析LLM的推理结构，拆解request为prompt、token阶段并部署到不同机器，运行时**动态调度**到GPU**集群**，**集群内模型并行**：算子/pipeline（layer并行）、张量/数据（layer dimension并行）；

### idea

llm推理分为2个阶段：**prompt输入解析**（提供初始的context，计算bound）、**token生成**（gemv逐个生成新K-Q-V和token，KV-Cache的访问bound和容量Bound）；

推理性能的4个指标：**E2E**（从输入到完成的端到端延迟）、**TTFT**（从输入到第一个token，即解析输入和prompt的延迟）、**TBT**（生成每个token的延迟的均值）、**吞吐**（每秒完成的请求/输入 ）

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%2021.png)

> **[图片提取文字 (image.png)]:**
> | Metric                     | Importance to user                     |
> |----------------------------|----------------------------------------|
> | End-to-end (E2E) latency   | Total query time that the user sees    |
> | Time to first token (TTFT) | How quickly user sees initial response |
> | Time between tokens (TBT)  | Average token streaming latency        |
> | Throughput                 | Requests per second                    |
> |                            |                                        |
> 
> TABLE II: Performance metrics for LLMs.
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%2022.png)

批处理方法：request-level、batch-continuous、batch-mixed；

> **[图片提取文字 (image.png)]:**
> throughput. Several prior works have explored batching [23], [81]. Figure 2 shows the timelines for inference with three common batching mechanisms. The default mechanism only batches at the *request-level* (Figure 2(a)). In this case, ready requests are batched together, but all the forward passes for these requests are completed before any other requests are run. Since requests can have long token generation phases, this can lead to long wait times for requests arriving in between, causing high TTFT and high E2E latencies. An optimization is continuous batching [81] (Figure 2(b)). In this case, scheduling decisions are made before each forward pass of the model. However, any given batch comprises either only of requests in their prompt phase or only requests in token phase. Prompt phase is considered more important since it impacts TTFT. Hence, a waiting prompt can preempt a token phase. Although this leads to shorter TTFT, it can substantially increase the tail
> 
> Inference requests can be batched together for higher
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%2023.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> prompt and token phases.
> 
> for TBT, and therefore the E2E latency. Finally, there is *mixed* 
> 
> batching (Figure 2(c)) [23]. With this batching, the scheduling decisions are made at each forward pass, and the prompt and token phases can run together. This reduces the impact on TBT, but does not eliminate it, since token phases scheduled with prompt phases will experience a longer runtime. In the rest of the paper, we use mixed batching unless stated otherwise.
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%2024.png)

### LLM推理的运行特征

Azure LLM推理

> **[图片提取文字 (image.png)]:**
> **Models.** Table III shows the models that we evaluate. Both BLOOM [69] and Llama2 [71] are state-of-the-art open source LLMs. Both models are decoder-only, transformer-based models. We use the version of each model with the most parameters, since these versions are the most representative for production-class accuracy. Unless stated otherwise, we run BLOOM-176B and Llama-70B on vLLM [51] on a machine with 8 H100 [16] GPUs.
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%2025.png)

> **[图片提取文字 (image.png)]:**
> **Production traces.** We use production traces taken from two Azure LLM inference services on November 11th 2023. Our traces represent the most common scenarios in LLM inference today: coding and conversation. We have released a subset of our traces at https://github.com/Azure/AzurePublicDataset [4]. The traces we use for characterization are 20 minutes long and include the arrival time, input size (number of prompt tokens),
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%2026.png)

Coding的输入token数量一般很大，Conv的输入token的数量方差很大；

Coding的输出token的数量较少（中位数13），Conv的输出token数量双峰分布（129）；

> **[图片提取文字 (image.png)]:**
> of the number of *prompt input* and *generated output* tokens. Figure 3a shows the distribution of number of prompt tokens. Since the coding LLM inference service is generally used to generate completions as the user is writing code, its input
> 
> prompt can include large chunks of the code written so far.
> 
> To better understand our traces, we examine the distribution
> 
> Thus, it has a large median prompt size of 1500 tokens. On the other hand, the conversation service has a wider range of input prompt tokens since it depends on the user. The median number of prompt tokens for this trace is 1020 tokens.
> 
> Figure 3b shows the distribution of the number of generated tokens. Since the coding service typically only generates the
> 
> next few words in the program as the user types, the median
> 
> number of output token is 13 tokens. On the other hand, the
> 
> conversation service has an almost bimodal distribution, with a median of 129 tokens generated.
> 
> Insight I: Different inference services may have widely different prompt and token distributions.
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%2027.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Fig. 3: Distribution for prompt and generated tokens.
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%2028.png)

一个batch中，Prompt解析和token生成过程中耗费的时间对比，Conv大约**60-70%**的时间用于**20个token**的生成，Coding超过**20%**的时间用于**1个token**的生成；

> **[图片提取文字 (image.png)]:**
> ## B. Batch utilization
> 
> LLMs show very similar trends.
> 
> with very few active tokens batched.
> 
> machine running various number of active tokens in a batch. Note that if a prompt of 100 tokens is running in its prompt phase, we count the active tokens as 100. However, once the request is in the token phase, we count it as one active token, since the tokens are generated one at a time (assuming a beam search size of one [51]). We find that most of the time (60–70%) for conversation is spent running only 20 tokens or fewer. Since the coding service has very few output tokens, it
> 
> experiences even worse batching in the token phase and runs
> 
> with a single token for more than 20% of the time. Both the
> 
> Insight II: Mixed continuous batching spends most of the time
> 
> To understand how much can these requests be batched, we
> 
> measure how often machines run at a given batch size. We use
> 
> mixed continuous batching as shown in Figure 2. To fit into a
> 
> single machine, we run a scaled-down version of the coding
> 
> Figure 4 shows the distribution of the time spent by the
> 
> and conversation traces with 2 requests per second.
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%2029.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Fig. 4: Cumulative distribution of time spent with various active batched tokens.
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%2030.png)

增大prompt size，会线性增加TTFT延迟（prompt解析是计算bound）；

增大输出token的batch（将不同request的token合并到1个batch），几乎不影响TBT（**不同request的token生成过程近乎“镜像”**）；

无论prompt和输出token size的比例（50，90，99），**E2E延迟中大部分是token生成过程**；

> **[图片提取文字 (image.png)]:**
> ## C. Latency
> 
> tokens on TTFT. The range of sizes was chosen based on the coding and conversation traces. We find that TTFT for both models grows almost linearly as the prompt size increases.
> 
> **TTFT.** Figure 5a shows the impact of the number of prompt
> 
> This behavior is due to the prompt phase having high GPU utilization and being computationally bound.
> 
> **TBT.** Figure 5b shows the impact of forcefully batching the output tokens of different requests together on the TBT. We observe very little impact on TBT as the batch size grows
> 
> observe very little impact on TBT as the batch size grows. With a batch size of 64, there is only 2× impact on TBT.
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%2031.png)

> **[图片提取文字 (image.png)]:**
> both models, with no batching. The variability between the request input and output sizes is apparent. Furthermore, we see that most of the E2E time is spent running the token phase. This holds true even for the coding trace, where prompt sizes are large and generated tokens few. In fact, we find that for BLOOM-176B, a prompt phase with 1500 input tokens takes
> 
> **E2E.** Figure 5c shows various percentiles of E2E latency for
> 
> the same time as token phase with only 6 output tokens. *Insight III:* For most requests, the majority of the E2E time is spent in the token generation phase.
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%2032.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> (a) TTFT by prompt(b) TBT by batch size. (c) Latencies on prod size. traces (no batching).
> 
> Fig. 5: TTFT, TBT, and E2E for BLOOM-176B and Llama-70B on DGX-H100.
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%2033.png)

prompt解析过程，**适当**增加Batchsize能增加**吞吐**，之后则没有效果；

token生成过程， **增加**Batchsize能线性增加**吞吐**，直到达到内存容量（和传输带宽）；

> **[图片提取文字 (image.png)]:**
> ## Figure 6 shows the impact of batching on the throughput (measured as tokens per second). For the prompt phase, we define the throughput as the number of prompt input tokens that are processed per second. We see that the throughput
> 
> decreases after 2048 prompt tokens, which corresponds to a
> 
> D. Throughput
> 
> batch size of less than 2 for the median prompt sizes from the traces. On the other hand, Figure 6b shows that the throughput in the token phase keeps increasing with batching until 64 batch-size, at which point, the machine runs out of memory. *Insight IV:* The prompt phase batch size should be limited
> 
> to ensure good performance. In contrast, batching the token
> 
> generation phase yields high throughput without any downside.
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%2034.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Fig. 6: Impact of batching on the throughput for the 2 LLMs.
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%2035.png)

**内存使用**：prompt解析是计算bound而内存占用不大，token生成的内存占用随着token生成指数增加；

> **[图片提取文字 (image.png)]:**
> E. Memory utilization
> 
> During an LLM inference, the GPU memory is used to host the model weights and activations, as well as the KV caches
> 
> (Section II-B). As the number of tokens in a batch increase, the memory capacity required for the KV cache also increases. Figure 7 shows the memory capacity utilization during each phase as the number of tokens in the batch increases. During
> 
> the prompt phase, the input prompt tokens generate the KV
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%2036.png)

> **[图片提取文字 (image.png)]:**
> cache. During the output token phase, each active generated token that is being processed accesses the KV cache of its entire context so far. **Insight V:** Batching during the prompt phase is compute-bound, whereas the token phase is limited by memory capacity.
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%2037.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Fig. 7: Required memory with batching in prompt/token phases.
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%2038.png)

prompt解析随着prompt size增加而更耗能，token生成的能耗和token size几乎无关，功耗限制了prompt解析的性能（**power cap**），因此**功耗**应更多提供给prompt解析；

> **[图片提取文字 (image.png)]:**
> ## F. Power utilization
> 
> prompt phase is compute intensive, its power draw increases with batch size. On the other hand, the token phase is memory bound and its power draw does not vary when increasing the number of tokens to process.
> 
> Providers can cap the power usage of the machines to reduce the peak power. Figure 9 shows the impact to latency when increasing the power caps for both prompt and token phases. The prompt phase is highly sensitive to the power cap and the
> 
> latency increases substantially. On the other hand, the token
> 
> generation phase incurs almost no latency impact when power
> 
> *Insight VI:* While the prompt phase utilizes the power budget
> 
> capping by over 50% (i.e., 700 to 350W).
> 
> of the GPU efficiently, the token phase does not.
> 
> When hosting machines, cloud providers need to consider
> 
> the peak power draw, which has direct impact in the datacenter
> 
> cost [26]. This is especially important when building GPU
> 
> clusters, since GPUs consume much higher power than regular
> 
> compute machines [63], [64]. Figure 8 shows the GPU power
> 
> draw normalized to the thermal design power (TDP) when
> 
> running prompt and token generation phases. Since the the
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%2039.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Fig. 8: Maximum and mean power utilization varying the batching size.
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%2040.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Fig. 9: Impact of power cap on the prompt and token generation latency with the maximum batch size possible.
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%2041.png)

prompt解析的性能受到**硬件算力**的影响，token生成则可使用更经济的硬件；

> **[图片提取文字 (image.png)]:**
> shows our findings. We see a lower performance impact on the token generation phase (TBT) as compared to the Prompt phase (TTFT). Since coding requests are dominated by prompt phase, by having very few generated tokens, the E2E latency impact from A100 is worse on coding than conversation. Furthermore, we see that A100 has better or equal inference cost and energy overall compared to H100. **Insight VII:** Token generation can be run on less computecapable hardware for better Perf/W and Perf/\$ efficiencies.
> 
> the two from running on different hardware. Table I shows the
> 
> specifications for DGX-A100 [15] and DGX-H100 [16]. The
> 
> memory-to-compute ratio favors A100 over H100. Table IV
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%2042.png)

> **[图片提取文字 (image.png)]:**
> |                                                        | Couing   |          |               | Conversation |         |               |
> |--------------------------------------------------------|----------|----------|---------------|--------------|---------|---------------|
> |                                                        | A100     | H100     | Ratio         | A100         | H100    | Ratio         |
> | TTFT                                                   | 185 ms   | 95 ms    | 0.51×         | 155 ms       | 84 ms   | 0.54×         |
> | TBT                                                    | 52 ms    | 31 ms    | $0.70 \times$ | 40 ms        | 28 ms   | $0.70 \times$ |
> | E2E                                                    | 856 ms   | 493 ms   | $0.58 \times$ | 4957 ms      | 3387 ms | $0.68 \times$ |
> | <b>Cost</b> [5]                                        | \$0.42   | \$0.52   | $1.24 \times$ | \$2.4        | \$3.6   | 1.5×          |
> | Energy                                                 | 1.37 Whr | 1.37 Whr | $1\times$     | 7.9 Whr      | 9.4 Whr | 1.2×          |
> | TABLE IV: P50 request metrics on A100 vs. H100 without |          |          |               |              |         |               |
> | batching on Llama-70B.                                 |          |          |               |              |         |               |
> 
> Convergation
> 
> Coding
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%2043.png)

### Splitwise LLM调度

多机器集群调度**CLS**：动态分配不同任务集群的机器数量和负载；

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Fig. 10: High-level system diagram of Splitwise.
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%2044.png)

> **[图片提取文字 (image.png)]:**
> Request routing. CLS uses Join the Shortest Queue (JSQ) scheduling [39], [85] to assign a prompt and a token machine to each request. Queue lengths are defined by the number of pending tokens. Each machine regularly communicates to the CLS changes in its memory capacity or pending queue. Note that this does not happen at every iteration boundary. We simultaneously assign both the prompt and token machine when scheduling requests, since we can then overlap KV-cache transfers with prompt computation to reduce transfer overheads (Section IV-C).
> 
> When routing requests, if the pending queue is bigger than a certain threshold, the CLS looks for target machines in the mixed pool. If the mixed pool is also full, it proceeds to look in the opposite pool (*i.e.*, a token machine to run prompts and vice versa) and moves the machine into the mixed pool. Machines in the mixed pool operate exactly as a non-Splitwise machine would, with mixed batching. Once the queue of mixed requests is drained, the CLS transitions the machine back to its original pool. For example, when the queue is too long, we can move a prompt machine to the mixed pool to run tokens;
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%2045.png)

单机器调度**MLS**：每种任务机器的request调度，prompt解析阶段的request抢占token生成的request；

KV-Cache的**传输优化**：prompt解析——KV-Cache传输——token迭代1是严格串行，较小的prompt size生成的KV-Cache传输开销不大，较大的prompt size在每个decoder layer计算完成后就开始传输（layer-wise）；

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Fig. 11: Optimizing KV-cache transfer in Splitwise.
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%2046.png)

> **[图片提取文字 (image.png)]:**
> ## B. Machine-level scheduling
> 
> The MLS runs on each machine and is responsible for tracking the GPU memory utilization, maintaining the pending queue 4, deciding the batch for each iteration, and reporting the relevant status to the CLS.
> 
> **Prompt machines.** The MLS simply uses first-come-first-serve (FCFS) to schedule prompts. The results in Figure 6a show that after 2048 prompt tokens, the throughput degrades. For this reason, the MLS restricts the batching of multiple prompts together to 2048 tokens in total. This is a configurable value, and can change for a different model or hardware.
> 
> batches as much as possible. Figure 6b shows that the token generation throughput keeps scaling up with the batch size until the machine runs out of memory. For this reason, the MLS tracks the memory and starts queueing tokens once the machine is close to running out of memory.
> 
> **Mixed machines.** To meet the TTFT SLO, the MLS must
> 
> **Token machines.** The MLS uses FCFS to schedule tokens and
> 
> prioritize running prompts and schedule any new prompts in the pending queue immediately. If the machine is running token phases and has no additional capacity to run the prompt phase, the MLS will *preempt* tokens. To avoid *starvation* of the token phase due to preemption, we increase the priority of the token with age and limit the number of preemptions that each request can have.
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%2047.png)

> **[图片提取文字 (image.png)]:**
> KV-cache transfer, and the token generation phase for a single batch of requests when naively transferring the KV cache in a serialized way. The KV-cache transfer starts only after the prompt phase has finished and the first token is generated. Further, it needs to complete before the next output token
> 
> Figure 11a shows the Gantt chart for the prompt phase, the
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%2048.png)

> **[图片提取文字 (image.png)]:**
> impacts the maximum TBT and end-to-end latency of inference. The time required for the transfer depends on the size of the KV cache (which is directly proportional to the number of prompt tokens) and on the bandwidth of the interconnect
> 
> can be generated in the token generation phase. This directly
> 
> between the prompt and the token machines. Even when using fast InfiniBand links, the transfer overhead for large prompt sizes could become a significant fraction of the TBT.
> 
> In Splitwise, we optimize the KV-cache transfer by overlapping it with the computation in the prompt phase. As each layer
> 
> in the LLM gets calculated in the prompt machine, the KV
> 
> cache corresponding to that layer is also generated. At the end of each layer, we trigger an asynchronous transfer of the KV-cache for that layer while the prompt computation continues to the next layer. Figure 11b shows this asynchronous transfer which reduces the transfer overheads. Layer-wise transfer also enables other optimizations, such as earlier start of the token phase in the token machines, as well as earlier release of KV-cache memory on the prompt machines.
> 
> Layer-wise KV-cache transfer happens in parallel with the
> 
> grained synchronization per layer for correctness. Thus, it is possible to incur performance interference and increase the TTFT, especially for smaller prompts. However, for small prompts the total KV-cache size is small and does not need the layer-wise transfer to hide the latency. Since the number of takens in a batch is already known at the start of computation
> 
> prompt computation for the next layer. This requires fine-
> 
> tokens in a batch is already known at the start of computation, Splitwise picks the best technique for KV-cache transfer. It uses serialized KV-cache transfer for smaller prompts and layer-wise transfer and for larger prompts. We show that the overall transfer and interference overheads are relatively small in Section VI-A.
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%2049.png)

Splitwise调度系统

**事件驱动模拟器**：评估不同服务调度到机器集群的性能；

**调度主体**：不同服务{对话、Coding}的运行特征和在特定硬件上的性能模型，类似于张量程序；

**搜索空间**：{AA、HH、HA、HHcap}的集群类型，不同任务{prompt解析、token生成、混池}的机器数量，类似于张量加速器的架构特征；

**约束**：不同服务{对话、Coding}的SLO；

**优化目标**：吞吐、成本、功耗；

> **[图片提取文字 (image.png)]:**
> ## D. Provisioning with Splitwise
> 
> in each of our evaluated systems.
> 
> We leverage Splitwise to optimize LLM inference cluster deployments for power, cost, and throughput.
> 
> Type of machines. We propose four main variants of Splitwise-
> 
> based systems: Splitwise-AA, Splitwise-HH, Splitwise-HA, and
> 
> Splitwise-HHcap. The nomenclature is simply drawn from the first letter representing the Prompt machine type, and the second letter representing the Token machine type. "A" represents a DGX-A100 machine, "H" represents a DGX-H100 machine, and "Hcap" represents a power-capped DGX-H100 machine.
> 
> Table V shows a summary of the cost, power, and hardware
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%2050.png)

> **[图片提取文字 (image.png)]:**
> using our event-driven cluster simulator, which is described in detail in Section V. We need to provide as input: (1) the target cluster design (e.g., Splitwise-HA or Splitwise-HHcap), (2) an LLM-specific performance model that can estimate the TTFT and TBT at various input, output, and batch sizes, (3) a short trace derived from the target prompt and token size distributions for the service (e.g., Figure 3), (4) the SLOs (e.g., Table VI), (5) the constraints (e.g., throughput), and (6) the optimization goal (e.g., minimize cost). Using this information, our provisioning framework searches the space for the desired optimal point. For example, searching with a throughput constraint and a cost minimization goal gives us iso-throughput cost-optimized clusters across different designs.
> 
> **Number of machines.** The LLM inference cluster deployment
> 
> must be sized with the appropriate number of prompt and token
> 
> machines. Our methodology involves searching the design space
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%2051.png)

> **[图片提取文字 (image.png)]:**
> dimensional search space for the number of prompt and token machines under Splitwise-HH for the coding workload (using a 2-minute trace). The simulator outputs the various percentiles for TTFT, TBT, and E2E latencies. Then, we select the clusters that meet the SLOs for each of these metrics and optimize our target function. For example, Figure 12 shows a \* for the setup with 27 prompt and 3 token machines with the lowest cost that achieves 70 RPS. We call this setup iso-throughput
> 
> cost-optimized.
> 
> **Search space.** Figure 12 shows an example of the two-
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%2052.png)

> **[图片提取文字 (image.png)]:**
> ## **Optimization.** We can use three optimization goals: *throughput*, *cost*, and *power*. Throughput optimization is important for both,
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%2053.png)

> **[图片提取文字 (image.png)]:**
> the cloud service provider (CSP) and the user. Cost optimization has different importance levels to the CSP and the user. For the CSP, a higher cost for the same throughput might be acceptable if there are gains in power and space requirements for the cluster. However, for the end-user, a higher cost at the same throughput is generally unacceptable. Finally, power optimization is attractive for a CSP, since it enables more GPUs to be deployed in the same datacenter [62], [63], but it may not be as important to the user. We only consider the provisioned power, and not the dynamic power utilization, in our study.
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%2054.png)

SLO：P50定义50%的时间中系统指标满足的约束，P99定义99%的时间中系统指标满足的约束；

由于P99定义的时间包含P50定义的时间，P99的约束一般比P50更松弛（满足P99约束的系统，一定满足P50约束）；

> **[图片提取文字 (image.png)]:**
> |            | P50           | P90  | P99        |
> |------------|---------------|------|------------|
> | TTFT       | 2×            | 3×   | 6×         |
> | <b>TBT</b> | $1.25 \times$ | 1.5× | $5 \times$ |
> | E2E        | $1.25 \times$ | 1.5× | $5 \times$ |
> 
> TABLE VI: SLO expressed as slowdown compared to a request running on DGX-A100 under no contention.
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%2055.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Fig. 12: Design space for provisioning a Splitwise-HH cluster. Cluster configurations targets a peak throughput of 70 RPS.
> 
> The cost-optimal Splitwise-HH configuration is marked with \* (27 prompt and 3 token machines).
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%2056.png)

### 实验

**实现**layer-wise KV-Cache传输机制（vllm、MSCCL++的put原语），**实现**mixed batch方法；

> **[图片提取文字 (image.png)]:**
> ## A. Experimental setup
> 
> discussed earlier in Figure 2(c).
> 
> vLLM on two DGX-A100 and two DGX-H10 virtual machines (VMs) on Microsoft Azure with specifications from Table I. These are the VMs used to collect the characterization data in Section III. These machines are connected with InfiniBand and the DGX-H100s have double the bandwidth (*i.e.*, 400 Gbps). Since vanilla vLLM only supports continuous batching with
> 
> token preemption which can lead to much higher TBT, we
> 
> implement state-of-the-art mixed continuous batching [81] as
> 
> To evaluate our proposal on real hardware, we implement
> 
> Splitwise's KV-cache transfer mechanism on top of vLLM [51].
> 
> Our implementation is open source [1]. We run this modified
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%2057.png)

> **[图片提取文字 (image.png)]:**
> prompt machine generates the first token, it transfers the KVcache to the token machine using the technique described in Section IV-C. We use MSCCL++ [11], an optimized GPUdriven communication library, to implement the naive and layer-wise KV cache transfers. In our implementation, the prompt machine uses the zerocopy one-sided put primitive of MSCCL++ to send KV-cache data over InfiniBand as soon as it is ready, without requiring the token machine to issue any receive instructions. Once we have issued a put for all layers, the prompt machine signals a semaphore that the token machine waits on. The synchronization done with the help of semaphores uses the same InfiniBand connection used to send KV-cache data. When processing a batch of prompts, each request is assigned a different semaphore since it may be routed to different token machines. We ship the KV-caches block-by-block in vLLM. To minimize the number of transfers, we also consider the contiguity of KV blocks as long as they use the same semaphore.
> 
> Our implementation of the Splitwise technique assigns
> 
> machines either a prompt role, or a token role. As the
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%2058.png)

**模拟器**：分析LLM在硬件上的运行时，构建**性能模型**；模拟器输入request负载、性能模型、性能SLO、硬件配置，**预测性能指标**；实际数据**验证**性能指标；

> **[图片提取文字 (image.png)]:**
> ## B. Simulator setupWe build a simulator to explore cluster designs and evaluate
> 
> Splitwise at scale. The simulator code is open source [20]. Figure 13 shows the design of our simulator. The simulator is
> 
> event-driven and faithfully models the Splitwise machine pools,
> 
> schedulers, machine-level memory and queues, and KV-cache transfer. We first profile the LLM on the target hardware with various input/output sizes 1. Based on the characterization profiles, we build a performance model. The simulator takes as input the request traces, SLOs, the performance model,
> 
> and the configurations for cluster and scheduler (2). For our
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%2059.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Fig. 13: Overview of the design of the Splitwise simulator.
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%2060.png)

> **[图片提取文字 (image.png)]:**
> Performance model. We build a piece-wise linear performance model using performance profiles at various batch sizes, input sizes, output sizes, in the required parallelism configuration on A100 and H100 machines from Section III. We validate that our performance model has high accuracy; it incurs a mean absolute percentage error (MAPE) of less than 3% when evaluated with a 80:20 train:test dataset split.
> 
> Communication model. In our evaluation, KV-cache transfers cause inter-machine communication, whereas tensor parallelism only causes intra-machine communication. We model intermachine communication overheads by benchmarking our KV-cache transfer implementation over Infiniband in Section VI-A.
> 
> SLOs. To determine the maximum throughput that can be supported by a given cluster design, we use P50, P90, and P99 SLOs for TTFT, TBT, and E2E latency metrics. Table VI
> 
> shows our SLO definition using DGX-A100 as a reference. We
> 
> require all nine SLOs to be met. SLOs on TTFT are slightly
> 
> looser, since it has a much smaller impact on the E2E latency.
> 
> Baselines. We compare our Splitwise designs against Baseline-
> 
> A100 and Baseline-H100. The clusters in these baselines
> 
> consist of just DGX-A100s and DGX-H100s, respectively. Both
> 
> baselines use the same mixed continuous batching that Splitwise
> 
> uses for mixed pool machines (described in Section IV-A).
> 
> evaluation, we use the prompt and token size distributions from
> 
> the production traces in Section III. We tune the Poisson arrival
> 
> rate to increase and decrease the load (requests per second)
> 
> for cluster sizing. The simulator provides the achieved metrics
> 
> per request (TTFT, TBT, E2E), and the machine utilization
> 
> levels (3). We cross-validated the performance model with
> 
> hardware experiments to ensure accuracy; we also validated
> 
> the simulator end-to-end using production load with over 50K
> 
> iterations to ensure fidelity (4
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%2061.png)

## KDD25：BurstGPT

**BurstGPT: A Real-World Workload Dataset to Optimize LLM Serving Systems**

LLM服务负载Trace

> **[图片提取文字 (image.png)]:**
> ## Abstract
> 
> mized to improve quality of service (QoS) and throughput. However, due to the lack of open-source LLM serving workloads, these systems are frequently evaluated under unrealistic workload as-
> 
> Serving systems for Large Language Models (LLMs) are often opti-
> 
> are deployed in real-world scenarios.
> 
> This work presents BurstGPT, an LLM serving workload with
> 
> sumptions. Consequently, performance may degrade when systems
> 
> 10.31 million traces from regional Azure OpenAI GPT services over 213 days. BurstGPT captures LLM serving characteristics from user, model and system perspectives: (1) User request concurrency:
> 
> burstiness variations of requests in Azure OpenAI GPT services, revealing diversified concurrency patterns in different services and model types. (2) <u>User conversation patterns:</u> counts and intervals within conversations for service optimizations. (3) <u>Model response</u>
> 
> within conversations for service optimizations. (3) Model response lengths: auto-regressive serving processes of GPT models, showing statistical relations between requests and their responses. (4) System response failures: failures of conversation and API ser-
> 
> vices, showing intensive resource needs and limited availability of LLM services in Azure. The details of the characteristics can
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%2062.png)

> **[图片提取文字 (image.png)]:**
> serve multiple purposes in LLM serving optimizations, such as system evaluation and trace provisioning. In our demo evaluation with BurstGPT, frequent variations in BurstGPT reveal declines in efficiency, stability, or reliability in realistic LLM serving. We identify that the generalization of KV cache management, scheduling and disaggregation optimizations can be improved under realistic workload evaluations. BurstGPT is publicly available now at https://github.com/HPMLL/BurstGPT and is widely used to develop prototypes of LLM serving frameworks in the industry.
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%2063.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> BurstGPT is a real-world workload trace from the Azure OpenAI GPT service. A scaled sample from a period of BurstGPT can be used to optimize serving systems using specific methods, considering realistic concurrency and response patterns. Note that we open-sourced two versions of BurstGPT: a cleaned trace and a raw trace, with failure logs excluded from the cleaned version.
> 
> Figure 1: Data collection and use method of BurstGPT.
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%2064.png)

## MuxServe：基于MPS的多llm

## NeurIPS25：FutureSightDrive（MLLM）

MLLM应用于自动驾驶：理解场景，预测未来场景，规划路径（逆向）；

> **[图片提取文字 (image.png)]:**
> VLAs to "think visually" using a novel visual spatio-temporal CoT. FSDrive first operates as a world model, generating a unified future frame that combines a predicted background with explicit, physically-plausible priors like future lane dividers and 3D object boxes. This imagined scene serves as the visual spatiotemporal CoT, capturing both spatial structure and temporal evolution in a single representation. The same VLA then functions as an inverse-dynamics model to plan trajectories conditioned on current observations and this visual CoT. We enable this with a **unified pre-training paradigm** that expands the model's vocabulary with visual tokens and jointly optimizes for semantic understanding (VQA) and future-frame prediction. A progressive curriculum first generates structural priors to enforce physical laws before rendering the full scene. Evaluations on nuScenes
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%2065.png)

> **[图片提取文字 (image.png)]:**
> ## 1 Introduction
> 
> and reasoning about safe decisions.
> 
> The advent of Multimodal Large Language Models (MLLMs) is reshaping autonomous driving, with Vision-Language-Action (VLA) models emerging as a promising end-to-end paradigm [20] 43 87 31. Harnessing the superior capabilities of MLLMs in world knowledge, reasoning, and interpretability, these models directly map visual observations and language instructions to vehicle control commands (e.g., speed and trajectory). This approach not only simplifies the conventional modular architecture, thereby minimizing potential information loss across components, but also enables the system to leverage vast pre-trained knowledge for analyzing complex driving environments
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%2066.png)

> **[图片提取文字 (image.png)]:**
> To enhance their reasoning abilities, many such models have incorporated the Chain-of-Thought (CoT) strategy, which encourages step-by-step thinking [67, 50, 15, 52]. However, in existing autonomous driving applications [27, 44, 14], this often involves generating discrete textual CoTs (e.g., language descriptions of the current scene or bounding box coordinates) as intermediate steps. This process forces a conversion of rich, continuous visual data into abstract, symbolic representations — a form of lossy compression that can obscure critical spatio-temporal relationships, discard fine-grained visual details, and introduce a "modality gap" between perception and planning [46, 55, 72], as illustrated in Figure T For autonomous vehicles requiring deep physical-world interaction, should their thinking process more closely resemble simulation and imagination of world, rather than merely relying on
> 
> logical deduction of language?
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%2067.png)

discrete text CoT用文本表达图像，模态转换过程存在信息丢失；

image-text CoT是预测的文本和生成的图像，不一致的表达导致传达信息低效；

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 1: Comparison of different CoT. Textual CoT expression provides insufficient information. The modalities between the image-text CoT are inconsistent. The proposed spatio-temporal CoT captures the temporal and spatial relationships in the future.
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%2068.png)

模型预测未来并生成图像（优先道路边界、行人），证明模型理解了所看见的画面，图像作为**CoT的媒介**（类似token）；

空间体现在理解**现实场景**，时间体现在预测**未来画面**；

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 2: Overview of FSDrive. Taking the currently surround images and task instructions as input, MLLM is trained in the form of next token prediction. MLLM predicts the future spatio-temporal CoT, and then generates trajectory based on the current observation and predicted future.
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%2069.png)

视觉**理解**：输入图像和指令，输出问题的答案，Model学会了解当前状况并回答关联问题；

视觉**生成**：输入图像和指令（lane、detection），**预测**未来的图像并**生成**，人为设计“道路边界——行人——完整生成”的生成顺序（**思考步骤**）；

思维链**CoT**：道路边界和行人检测提供**物理先验**信息，**指引**Model关注**未来场景**的可行区域和关键物体（**空间**），视觉内容的动态变化表征**时间**流动和场景自身规律；

预训练：通过VQA学习视觉理解和视觉生成（CoT）；

微调：针对自动驾驶场景特化**视觉理解**能力，为CoT增加**路径规划**能力，基于预测的未来场景和当前场景的**逆向**推理（先预测未来）；

> **[图片提取文字 (image.png)]:**
> Our method is directly built upon any existing MLLM that employs ViT-based encoders to convert images into continuous features. We preserve the original MLLM architecture without altering any structural components to maintain compatibility with pretrained weights. The sole modification involves expanding the MLLM's vocabulary by incorporating image tokens of the VQ-VAE into the text codebook, thereby extending the vocabulary's scope from language space to a multimodal space encompassing both visual and textual modalities. This enhancement enables the MLLM to predict image tokens, which can then be converted to image pixels through an VQ-VAE's detokenizer.
> 
> Pre-training for visual understanding. To effectively preserve the semantic understanding capa-
> 
> bilities of the native MLLM during the pre-training stage, as shown in the left part of Figure 2, we follow previous methods 64 27 by using a VQA task, which is crucial for autonomous vehicles to analyze complex driving scenarios. Given an image-text question-answer pair (I, L), where I represents the surround-view images of the current scene and L denotes the instructional question, model M generates a corresponding answer A:
> 
>  $A = \mathcal{M}(I, L).$ 
> 
> (3)
> 
> that generate future frames to learn physical laws, after activating the visual generation capability, we also enable the VLA to predict future frames, thereby capturing the dynamic evolution of the world. Specifically, given an image-instruction pair (I,L), the model predicts the next visual token of the future front-view frame through autoregressive generation:
> 
> $$P(q_1, q_2, \dots, q_{h \cdot w}) = \prod_{t=1}^{h \cdot w} P_{\theta}(q_i \mid q_{< i}).$$
> (4)
> 
> The predicted visual tokens are then converted back into image pixels by VQ-VAE's detokenizer. Since future frames naturally exist in video datasets without requiring any labeled data, this approach unlocks the potential to harness abundant video data for improving generation quality.
> 
> **Progressive image generation.** However, directly generating complete detailed future scenes may
> 
> fail to adhere to physical laws [78]. Therefore, during pre-training stage, we propose a progressive, easy-to-hard generation method, incorporating annotated data containing lane divider and 3D detection. Before generating visual tokens of future frames  $Q_f$ , we leverage the world knowledge of VLA to first reason about visual tokens of lane dividers  $Q_l$ , which serve as the skeleton of the road scene and define drivable areas to enforce static physical constraints. Subsequently, we reason about visual tokens of 3D bounding boxes  $Q_d$ , representing motion patterns of key objects to impose dynamic
> 
> physical constraints. This progressive method sequence explicitly guides the model to infer structural
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%2070.png)

> **[图片提取文字 (image.png)]:**
> layouts and geometric details of future scenes while enforcing physical laws. By leveraging these intermediate visual reasoning steps as context, the model learns to think visually about the dynamic evolution of scenes, ultimately enabling accurate future prediction:
> 
> $$P(Q_f \mid Q_l, Q_d) = \prod_{t=1}^{h \cdot w} P_{\theta}(q_i \mid q_{< i}, Q_l, Q_d). \tag{5}$$
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%2071.png)

> **[图片提取文字 (image.png)]:**
> ## 3.3 Think visually with spatio-temporal CoT
> 
> potential future developments to achieve forward-looking comprehension. This thinking process should resemble physical world simulation and imagination rather than purely text symbolic logical deduction. Since our model has already learned physical constraints through the progressive generation during pre-training, and considering efficiency, we no longer separately generate lane dividers, 3D detection, and future frames, but instead integrate all these results into a single unified frame. As shown in the right part of Figure 2 here, VLA serves as a world model to generate a unified image frame predicting the future world state: Inspired by visual prompting engineering 53 that draws red circles on images to guide model attention and by VLIPP 78 first predicts future bounding boxes to introduce physical priors when generating future frames, we represent future world spatial relationships through future red lane dividers and 3D detection boxes on the predicted unified frames. These coarse-grained visual cues direct the model's attention toward drivable areas and critical objects in future scenes while enforcing physically plausible constraints. Meanwhile, the temporal
> 
> relationships are represented by the ordinary future frame, where the dynamic evolution of visual
> 
> content intuitively characterizes temporal progression and the inherent laws of scene development.
> 
> Subsequently, spatio-temporal CoT  $Q_{CoT}$  serves as an intermediate reasoning step, allowing the
> 
> VLA to function as an inverse dynamics model that plans trajectory based on current observations
> 
> Autonomous driving planning requires not only understanding the current scene but also envisioning
> 
>  $P(W_t \mid I_t, Q_{CoT}, opt(T_{com}, T_{ego})) = \prod_{i=1}^n P_{\theta}(w_i \mid w_{< i}, I_t, Q_{CoT}, opt(T_{com}, T_{ego})).$  (6)
> 
> ## 3.4 Training strategy
> 
> and future predictions:
> 
> Our FSDrive can be initialized from any existing MLLM (e.g., Qwen2-VL, LLaVA), avoiding training from scratch and saving significant costs. During training, we fully fine-tune the LLM parameters while freezing all encoders. The training process is divided into two stages:
> 
> Stage 1: Unified pre-training, Our objective is to preserve understanding capabilities of MLLMs
> 
> through VQA tasks and activate their visual generation capabilities to predict future frames. VQA task data originates from OmniDrive-nuScenes [64]. We incorporate a large volume of unlabeled image data from nuScenes [II] for future frame prediction. To implement progressive easy-to-hard CoT, we integrate nuScenes annotated data to teach the model predicting image-formatted future lane dividers and 3D detection. Finally, we add future frame prediction with CoT datas containing intermediate reasoning steps. All the above understanding and generation tasks are trained together.
> 
> trajectory planning. Following OmniDrive [64], scene understanding utilizes DriveLM's GVQA [54] dataset. For trajectory planning, we follow VAD [29] [21] using nuScenes, where our spatio-temporal CoT integrates the holistic future scene, explicit lane dividers, and 3D detection results into a single future frame as intermediate reasoning steps. We train these tasks simultaneously using a single model, enabling task-specific predictions during inference through different task prompts.
> 
> Stage 2: Supervised fine-tuning. We focus on autonomous driving scene understanding and
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%2072.png)

**实现**：**Qwen2-VL-2B**作为MLLM，**MoVQGAN（VQVAE的改进Model）**提供MLLM**表示图像**的能力（decode_code），并将预测的visual token转为图像（生成CoT）；

LLM的CoT中3D检测和lane检测是LLM的CoT，能否换成小模型在边缘侧？2B意味着边缘部署的可能性？

MoVQGAN的normal infer和ema inference**（ema权重）**中用到了Conv、Res blk等等；

> **[图片提取文字 (image.png)]:**
> **Implementation details.** We initialize our model with Owen2-VL-2B [63] and pre-train it for 32 epochs to enable visual generation while preserving semantic understanding. During fine-tuning (12 epochs on 8 NVIDIA RTX A6000), we use  $1 \times 10^{-4}$  learning rate and batch size of 16. We expand the visual codebook of MoVQGAN [92] to the vocabulary of the large language model and use its detokenizer to convert the visual tokens predicted by the large language model to the pixel space.
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%2073.png)

> **[图片提取文字 (image.png)]:**
> ## What the paper shows versus what the code does
> 
> - The paper "MoVQ: Modulating Quantized Vectors for High-Fidelity Image Generation" emphasizes the high-level novel contributions: the multichannel vector quantization, spatially conditional normalization, and the discrete latent prior.
>   The architecture diagrams in the paper often show blocks like "Encoder", "Quantizer", "Decoder",
> - "Spatially conditional modulation", etc., rather than every low-level conv layer.
>   In contrast, the code (for example in the SBER-MoVQGAN version from the GitHub repo) is a full implementation: it includes convolutional layers, residual blocks, upsampling/downsamping
> - Therefore: The presence of conv layers in code is expected they form the backbone of the encoder/decoder network. The diagram in the paper simply abstracts away those details to focus on the novel components.
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%2074.png)

**相关工作**

多模态理解和生成：**VQ-VAE**将image表示成**token**来处理和预测，但**模态转换**丢失语义信息，降低下游任务的性能；**ViT encoder**直接编码**image**（**理解**）后使用difussion或自回归模型预测和**生成**；

LLM for AD：**DriveGPT4**对当前场景的多次问答解释车辆行为和预测车辆控制（token编码图像）；Drivegpt4: Interpretable end-to-end autonomous driving via large language model

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%2075.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Picture_0.jpeg)
> 
> Action description: The car is stopped Action justification: for the red light
> 
> ![](_page_0_Picture_2.jpeg)
> 
> Action description: The car pulls into the right lane
> Action justification: because traffic is moving faster in that lane.
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%2076.png)

**DriveVLM**中LLM低频率/粗粒度的预测路径，部署时交给**端到端**模型修正后得到最终规划；

Drivevlm: The convergence of autonomous driving and large vision-language models.

> **[图片提取文字 (image.png)]:**
> ## **DriveVLM**
> 
> ## Large Vision Language Model
> 
> #### Scene Description
> 
> ![](_page_0_Picture_3.jpeg)
> 
> ![](_page_0_Picture_4.jpeg)
> 
> Weather: cloudy.
> Road type: suburban.
> Time: Daytime.
> 
> Lane condition: right lane impassable, left lane passable.
> 
> **Critical objects**: police car at [(x1, y1), (x2, y2)]....
> 
> ### **Scene Analysis**
> 
> SYSTEM> Matched objects: police car, history trajectory:xxx.
> Unmatched objects: ...
> 
> Describe the critical objects and their influence on the ego-vehicle.
> 
> ![](_page_0_Picture_11.jpeg)
> 
> **Characteristics**: Parking on the right side of the road.
> 
> **Influence**: Blocking the right lane and indicating a potential for accidents or other incidents.
> 
> Summarized Analysis: ...
> 
> # Hierarchical Planning
> 
> SYSTEM> Ego state and historical trajectory are [...], determine meta-actions, decisions, and plan future waypoints.
> 
> #### 
> 
> **Meta-actions**: [slow down, shift slightly to the right, go straight at a constant speed].
> 
> **Decision**: Slow down and shift slightly to the right to overtake the barrier and then go straight at a constant speed.
> 
> **Waypoints**: [(x1, y1), ..., (xn, yn)].
> 
> Trajectory
> 
> Refinement
> 
> Low Frequency
> 
> Matching
> 
> **Prompting** 
> 
> **DriveVLM-Dual** 
> 
> 3D Perception
> 
> **Motion Prediction** 
> 
> Trajectory Planning
> 
> High Frequency
> 
> Traditional Pipeline
> 
> ![](_page_0_Picture_31.jpeg)
> 
> Sequence of Images
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%2077.png)

边缘侧：3D detection——motion prediction——trajectory planning（LLM提供初始解）；

云中心：VLM理解车辆和环境数据，规划未来路径；

> **[图片提取文字 (image.png)]:**
> ## 3.5 DriveVLM-Dual
> 
> To mitigate the challenges of high latency and imprecise spatial and motion understanding in VLMs, we propose DriveVLM-Dual, a collaboration between DriveVLM and the traditional autonomous driving system. This novel approach involves two key strategies: incorporating 3D perception for critical object analysis, and high-frequency trajectory refinement.
> 
> Integrating 3D Perception. We represent objects detected by a  ${}^{3}$ D detector as  $O_{3D} = \{c^i_{3D}, b^i_{3D}\}$ , where  $b^i_{3D}$  denotes the i-th bounding box and  $c^i_{3D}$  denotes its category. These 3D bounding boxes are then back-projected onto 2D images to derive corresponding 2D bounding boxes  $b^i_{2D}$ . We conduct IoU matching between these 2D bounding boxes  $b^i_{2D}$  and  $b^j_c$ .  $b^j_c$  are the bounding boxes of previously identified critical objects  $O_{\rm critical} = \{c^j_c, b^j_c\}$ . We classify critical objects that meet a certain approximate IoU threshold and belong to the same category as matched critical objects  $O^{\rm matched}_c$ , defined as
> 
> $$O_c^{\text{matched}} = \{c_c^j, b_c^j\}, \quad \text{if } c_c^j = c_{\text{2D}}^i \text{ and a} \text{IoU}(b_c^j, b_{\text{2D}}^i) > \tau, \text{ where a} \text{IoU}(b_c^j, b_{\text{2D}}^i) = \frac{S_{b_c^j \cap b_{\text{2D}}^i}}{S_{b_{\text{2D}}^i}},$$
>  Those critical objects without a corresponding match in the 3D data are noted as  $O_c^{\text{unmatched}}$ .
> 
> In the scene analysis module, for  $O_c^{\rm matched}$ , the center coordinates, orientations, and historical trajectories of the corresponding 3D objects are used as language prompts for the model, assisting in object analysis. Conversely, for  $O_c^{\rm unmatched}$ , analysis relies solely on the language tokens derived from the image. This design enables DriveVLM-Dual to understand the locations and motions of critical objects more accurately, enhancing the overall performance.
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%2078.png)

> **[图片提取文字 (image.png)]:**
> obtaining a trajectory from DriveVLM at low frequency, denoted as  $W_{\rm slow}$ , we take it as a reference trajectory for a classical planner for high-frequency trajectory refinement. In the case of an optimization-based planner,  $W_{\rm slow}$  serves as the initial solution for the optimization solver. For a neural network-based planner,  $W_{\rm slow}$  is used as an input query, combined with additional input features f, and then decoded into a new planning trajectory denoted as  $W_{\rm fast}$ . The formulation of this process can be described as:  $W_{\rm fast} = {\rm Planner}([W_{\rm slow}, f]). \tag{1}$ 
> 
> This refinement step ensures that the trajectory produced by DriveVLM-Dual (1) achieves higher
> 
> trajectory quality, and (2) meets real-time requirements. In practice, the two branches operate asyn-
> 
> chronously in a slow-fast manner, where the planner module in the traditional autonomous driving
> 
> branch can selectively receive trajectory from the VLM branch as additional input.
> 
> **High-frequency Trajectory Refinement.** To achieve real-time, high-frequency inference capa-
> 
> bilities, we integrate it with a conventional planner to form a slow-fast dual system, combining
> 
> the advanced capabilities of DriveVLM with the efficiency of traditional planning methods. After
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%2079.png)

**EMMA**利用多模态模型将non-sensor（非图像）输入和输出都当作文本，和sensor图像一起理解后预测路径；Emma: End-to-end multimodal model for autonomous driving

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 1: EMMA overview diagram. It takes 3 inputs (left): 1) a high-level command from the router, 2) historical status of the ego vehicle, and 3) surround-view camera videos. The model then predicts ego future trajectories (right) for motion planning that will be transformed into vehicle driving control signals. Further, we can ask the model to explain its rationale (top) before predicting trajectories, which enhances both the performance and explainability of the model through chain-of-thought reasoning. Notably, we incorporate visual grounding into the rationale so that the model also predicts the accurate 3D/BEV location for critical objects. In addition to end-to-end planning, we highlight three additional perception capabilities of our model (bottom).
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%2080.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 2: Illustration of EMMA Generalist. Starting with a task prompt (left), EMMA generates a corresponding textual prediction (middle right), which can then be decoded into a target output format, visualized and overlaid with the input image (right). EMMA Generalist is highly versatile, capable of performing a wide range of driving-related tasks, such as end-to-end motion planning, object detection, road graph estimation, and scene understanding Q&A. In the answer text, italicized words represent placeholders that will be dynamically substituted with actual values during task execution.
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%2081.png)

**世界模型** for AD：基于当前观察，理解车辆状况和环境，进行预测未来和作出规划；

场景生成通过LLM+Diffusion（理解、预测），规划（下游任务）和表示学习（理解输入）归属LLM；

> **[图片提取文字 (image.png)]:**
> ## 2 Related work
> 
> ## 2.1 Unified multimodal understanding and generation
> 
> VILA-U [73] employ VQ-VAE [61] to transform images into discrete tokens while training LLMs to predict them. However, these methods suffer from insufficient semantic information preservation, often leading to performance degradation in downstream understanding tasks. Alternative methods [57] [11] [48] [9] [82] utilize ViT [12]-based vision encoders (e.g., CLIP [51]) to encode images into continuous feature maps. Nevertheless, such methods typically depend on external diffusion models for image generation or use different training objectives (i.e. diffusion and autoregression) for the two tasks, further complicates the infrastructure design with overall lower efficiency. Moreover, the aforementioned methods usually require massive billion-scale datasets for extensive training from scratch, which results in prohibitively high computational costs when disseminating explorations in this form. In this work, we demonstrate that the visual generative capabilities of existing MLLMs can be directly activated through minimal training costs (approximately 0.3% of previous methods [70] [58] [42] [8])
> 
> Recent research efforts [38] 70, 49, 68 have increasingly focused on unifying multimodal under-
> 
> standing and visual generation within a single LLM. On one front, methods like Show-o [74], and
> 
> ## 2.2 Vision-language models for autonomous driving
> 
> without requiring sophisticated architectural designs.
> 
> Given the superior capabilities of large language models (LLMs) in world knowledge, reasoning, and interpretability, recent researches [2] [83] [39] [85] increasingly integrate Vision-Language Models (VLMs)/LLMs with autonomous driving systems to address limitations in end-to-end approaches.
> 
> DriveGPT4 [76] employs LLMs through iterative question-answering interactions to explain vehicle behaviors and predict control signals. DriveVLM [60] synergizes LLMs with end-to-end architectures, where LLMs predict low-frequency trajectories that are subsequently refined by the end-to-end model for final planning. Doe-1 [95] reformulates autonomous driving as a next-token prediction task using Lumina-mGPT's [37] multimodal generation capabilities, executing diverse tasks through multimodal
> 
> token processing. EMMA [27] leverages Gemini's multimodal foundation by encoding all non-sensor
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%2082.png)

> **[图片提取文字 (image.png)]:**
> language text, fully exploiting pre-trained LLMs' world knowledge. In this work, we propose a spatio-temporal chain of thought (CoT) reasoning method that unifies the form of images, allowing the model to think visually about trajectory planning.
> 
> inputs (navigation instructions, vehicle status) and outputs (trajectories, 3D positions) as natural
> 
> ## 2.3 World models for autonomous driving
> 
> vations to enable accurate future prediction and planning. Current applications of world models in autonomous driving primarily focus on driving scenario generation [47, 16, 32], planning [66, 41], and representation learning [45, 79, 84]. For driving scenario generation, most prior works are built upon diffusion models, with the exception of GAIA-1 [18] which incorporates a progressive
> 
> World models 66, 45, 90, 89 aim to infer ego status and dynamic environments from past obser-
> 
> next-token predictor and an additional diffusion image decoder. Recent DrivingGPT [5] leverages existing vision generation LLM LlamaGen [56] while simultaneously outputting predictions for future states and actions. However, such VQ-VAE based visual tokens lack semantic information,
> 
> future states and actions. However, such VQ-VAE based visual tokens lack semantic information, often leading to performance degradation in downstream visual understanding tasks [74, 40, 59]. In this work, we propose to directly activate the visual generation capabilities of existing multimodal large language models, enabling VLMs to act as world models and predict future frames.
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%2083.png)

## VAE

**VE是变量分布估计**，比如贝叶斯估计中max_i{似然概率i /（似然概率1 + … + 似然概率N）}进行分类和估计，预测思路是根据后验概率**预测现有结果的内在原因**。VE结合马尔科夫链模型用于**Diffusion和变分自编码器（VAE）**的构建和训练；

**Diffusion**的**BackBone（UNet、DiT）的智能**是预测当前时间步的Xt如何从过去时间步Xt-1添加噪声得到（类似计算后验概率的动机），**推理**时BackBone**预测**输入X（未来时间步N）如何从N-1…1（先验噪声）**逐步添加噪声**得到（去噪过程**denoise**），最后将先验噪声叠加所有噪声还原原始输入X，由于高斯噪声的叠加保持高斯分布，denoise中预测高斯噪声来一次性**加噪**；

训练和推理的目标：**训练原生包含推理过程**，因为训练就是为了推理性能更强；

**VQ-VAE**：VAE是变分自编码，作用是将输入**表达成连续型**隐变量，VQ是将隐变量**量化**到codebook中的向量（codebook是隐向量的取值空间），从而将输入**表达成离散型**隐变量；

隐变量表示学习的逻辑：**encoder**通过**输入x**的高层特征（结果）和**隐变量z**（原因）的先验概率**prior**，计算后验概率**posterior**后估计输入的**隐变量表征**（找到原因），**decoder**通过隐变量表征和先验概率计算似然概率**likelihood来预测输出y**，来**评估**表征学习的效果（训练）或**应用**到下游任务（推理）；

> **[图片提取文字 (image.png)]:**
> Perhaps the work most related to our approach are VAEs. VAEs consist of the following parts: an encoder network which parameterises a posterior distribution q(z|x) of discrete latent random variables z given the input data x, a prior distribution p(z), and a decoder with a distribution p(x|z)over input data.
> 
> Typically, the posteriors and priors in VAEs are assumed normally distributed with diagonal covariance, which allows for the Gaussian reparametrisation trick to be used [32, 23]. Extensions include autoregressive prior and posterior models [14], normalising flows [31] [10], and inverse autoregressive posteriors [22]. In this work we introduce the VQ-VAE where we use discrete latent variables with a new way of
> 
> training, inspired by vector quantisation (VQ). The posterior and prior distributions are categorical, and the samples drawn from these distributions index an embedding table. These embeddings are then used as input into the decoder network.
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%2084.png)

**表示学习**的衡量：表示学习是Model学习信息的表示方法（图像、文本、隐变量、特征、函数等），算法对输入图像进行**表示**，基于表示**重建**输出图像的**还原度**；

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 1: Left: A figure describing the VQ-VAE. Right: Visualisation of the embedding space. The output of the encoder z(x) is mapped to the nearest point  $e_2$ . The gradient  $\nabla_z L$  (in red) will push the encoder to change its output, which could alter the configuration in the next forward pass.
> 
> During forward computation the nearest embedding  $z_q(x)$  (equation 2) is passed to the decoder, and during the backwards pass the gradient  $\nabla_z L$  is passed unaltered to the encoder. Since the output representation of the encoder and the input to the decoder share the same D dimensional space, the gradients contain useful information for how the encoder has to change its output to lower the reconstruction loss.
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%2085.png)

> **[图片提取文字 (image.png)]:**
> $\frac{128 \times 128 \times 3 \times 8}{32 \times 32 \times 9} \approx 42.6$  in bits. We model images by learning a powerful prior (PixelCNN) over z. This allows to not only greatly speed up training and sampling, but also to use the PixelCNNs capacity to capture the global structure instead of the low-level statistics of images.
> 
> In this experiment we show that we can model  $x = 128 \times 128 \times 3$  images by compressing them to a
> 
>  $z = 32 \times 32 \times 1$  discrete space (with K=512) via a purely deconvolutional p(x|z). So a reduction of
> 
> Figure 2: Left: ImageNet 128x128x3 images, right: reconstructions from a VQ-VAE with a 32x32x1 latent space, with K=512.
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%2086.png)

## AAAI24：ResDiff（Conv、Attention、FFT、DWT）

**任务**：超分辨率图像重建ISR，image super resolution；

**GAN、Flow、Diffusion**方法；

> **[图片提取文字 (image.png)]:**
> to generate high-quality images. Similarly, Kim et al. (Wang et al. 2018b) introduced ESRGAN, which adopted an enhanced super-resolution GAN and a superior loss function to improve the perceptual quality. GAN-based methods combine content losses with adversarial losses, allowing them to generate sharp edges and richer textures. However, they are prone to mode-collapse, which decreases diversity in the generated SR samples. Moreover, training GANs is challenging and may lead to unexpected artifacts in the generated image. Flow-based methods Lugmayr et al. (Lugmayr et al. 2020) proposed SRFlow, which is a flow-based method that learns the conditional distribution of high-resolution images given their low-resolution counterparts, enabling highquality image super-resolution with natural and diverse out-
> 
> puts. Flow-based methods map HR images to flow-space la-
> 
> tents using an invertible encoder and connect the encoder
> 
> and decoder with an invertible flow module, which avoids
> 
> training instability but requires higher training costs and pro-
> 
> vides lower perceptual quality.
> 
> **GAN-based methods** Ledig et al. (Ledig et al. 2017) pro-
> 
> posed SRGAN, which employs a perceptual loss function
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%2087.png)

> **[图片提取文字 (image.png)]:**
> troduced SrDiff, the first diffusion-based model for SISR, demonstrating that using the diffusion model for SISR tasks is feasible and promising. Saharia et al. proposed Sr3 (Saharia et al. 2022c), which adapts Denoising Diffusion Probabilistic Models (DDPM) to perform SISR tasks, yielding a competitive perceptual-based evaluation value. Diffusionbased methods utilize a diffusion process that simulates noise reduction, resulting in sharper and more detailed images. However, a high computational cost is needed due to multiple forward and backward passes through the entire network during the training process. Our proposed ResDiff, though without improving the training speed of a single iteration, accelerates convergence, which can alleviate this issue from another perspective.
> 
> **Diffusion-based methods** Li et al. (Li et al. 2022) in-
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%2088.png)

CNN推理一次，Diffusion迭代推理（Conv、Attention、Residual Block、FFT/-1傅里叶变换、DWT小波变换）；

> **[图片提取文字 (image.png)]:**
> ## Algorithm 1: ResDiff Inference
> 
> **Input**: low-resolution image  $x_{LR}$  and pre-trained CNN **Parameter**:  $\mu_{\theta}$  and  $\Sigma_{\theta}$  same as in DDPM
> 
> - 1:  $x_{cnn} = \text{CNN}(x_{LR})$ 
>   - 2:  $x_T \sim \mathcal{N}(0, I)$
>   - 3: **for** t = T : 1 **do**
>   - 4:  $\epsilon \sim \mathcal{N}(0, I)$  if t > 1, else  $\epsilon = 0$
> - $x_{t-1} = \frac{\mu_{\theta}}{\mu_{\theta}}(x_t, t, x_{cnn}) + \sqrt{\sum_{\theta}}(x_t, t, x_{cnn}) \epsilon$
> - 6: end for
>   - 7: return  $x_0 + x_{cnn}$
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%2089.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 3: Depiction of the three loss functions utilized in CNN pre-training. A spatial domain loss (GT Loss) and two frequency domain losses (FFT Loss and DWT Loss) are computed.
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%2090.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 4: An overview of the model architecture in proposed FD-guided diffusion. The pre-trained CNN prediction and the noisy image  $x_t$  from step t are fed into the FD-info-Splitter, and its output is then passed on to a U-net, which is equipped with HF-guided cross-attention.
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%2091.png)

**GAN方法**：Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network

generator网络基于LR生成SR（造假），discriminator网络讲SR从HR（真图）中鉴别，训练好的生成网络用于SISR；

**dense layer**是FC layer；

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 4: Architecture of Generator and Discriminator Network with corresponding kernel size (k), number of feature maps (n) and stride (s) indicated for each convolutional layer.
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%2092.png)

GAN网络的loss：通过loss关于模型参数的梯度，反向传播梯度值来优化参数；

> **[图片提取文字 (image.png)]:**
> $l^{SR} = l_{X}^{SR} + 10^{-3} l_{Gen}^{SR} \tag{2}$ 
> 
> and an adversarial loss component as:
> 
> In the following we describe possible choices for the content loss  $l_{\rm X}^{SR}$  and the adversarial loss  $l_{\rm Gen}^{SR}$ .
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%2093.png)

> **[图片提取文字 (image.png)]:**
> is defined based on the probabilities of the discriminator  $D_{\theta_D}(G_{\theta_C}(I^{LR}))$  over all training samples as:  $l_{Gen}^{SR} = \sum -\log D_{\theta_D}(G_{\theta_G}(I^{LR}))$ (6)
> 
> fool the discriminator network. The generative loss  $l_{Gen}^{SR}$ 
> 
> n=1structed image  $G_{\theta_G}(I^{LR})$  is a natural HR image. For better
> 
> Here,  $D_{\theta_D}(G_{\theta_G}(I^{LR}))$  is the probability that the recongradient behavior we minimize  $-\log D_{\theta_D}(G_{\theta_C}(I^{LR}))$  instead of  $\log[1 - D_{\theta_D}(G_{\theta_G}(I^{LR}))]$  [22].
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%2094.png)

> **[图片提取文字 (image.png)]:**
> described in Simonyan and Zisserman [49]. With  $\phi_{i,j}$  we indicate the feature map obtained by the j-th convolution (after activation) before the i-th maxpooling layer within the VGG19 network, which we consider given. We then define the VGG loss as the euclidean distance between the feature
> 
> similarity. We define the VGG loss based on the ReLU
> 
> activation layers of the pre-trained 19 layer VGG network
> 
> representations of a reconstructed image  $G_{\theta_C}(I^{LR})$  and the reference image  $I^{HR}$ :
> 
> reference image 
> $$I^{HR}$$
> : 
> $$l_{VCC/i,j}^{SR} = \frac{1}{W_{i,j}} \sum_{j=1}^{W_{i,j}} \sum_{j=1}^{H_{i,j}} (\phi_{i,j}(I^{HR})_{x,y})$$
> 
>  $l_{VGG/i.j}^{SR} = \frac{1}{W_{i,j}H_{i,j}} \sum_{x=1}^{W_{i,j}} \sum_{y=1}^{H_{i,j}} (\phi_{i,j}(I^{HR})_{x,y})$ (5)
> 
> $$l_{VGG/i.j}^{SR} = \frac{1}{W_{i,j}H_{i,j}} \sum_{x=1} \sum_{y=1} (\phi_{i,j}(I^{HR})_{x,y})$$
> (5)
> 
>  $-\phi_{i,j}(G_{\theta_G}(I^{LR}))_{x,y})^2$ 
> 
> Here  $W_{i,j}$  and  $H_{i,j}$  describe the dimensions of the respective feature maps within the VGG network.
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%2095.png)

**Flow方法**：Hierarchical Conditional Flow: A Unified Framework for Image Super-Resolution and Image Rescaling

假设HR图像x经过一系列双射f得到隐变量z和LR，x经过f得到的中间结果是LR（image rescaling），LR和隐变量z经过一系列f-1得到SR（image super resolution），f是不同layer；

> **[图片提取文字 (image.png)]:**
> aim to learn a bijective mapping between the target space and the latent space. For a high-dimensional random variable (e.g., an image)  $\mathbf{x}$  with distribution  $\mathbf{x} \sim p(\mathbf{x})$  and a latent variable  $\mathbf{z}$  with simple tractable distribution  $\mathbf{z} \sim p(\mathbf{z})$ 
> 
> Flow-based models [7, 8, 23, 37, 13, 22, 2, 16, 12, 36, 28]
> 
> latent variable  $\mathbf{z}$  with simple tractable distribution  $\mathbf{z} \sim p(\mathbf{z})$  (e.g., multivariate Gaussian distribution), flow models generally use an invertible neural network  $f_{\theta}$  to transform  $\mathbf{x}$  to  $\mathbf{z}$ :  $\mathbf{z} = f_{\theta}(\mathbf{x})$ . Conversely,  $\mathbf{x}$  can be recovered from  $\mathbf{z}$  by the inverse mapping  $\mathbf{x} = f_{\theta}^{-1}(\mathbf{z})$ .
> 
> formations:  $f_{\theta} = f_{\theta}^1 \circ f_{\theta}^2 \circ \cdots \circ f_{\theta}^K$ . The intermediate variables are defined as  $\mathbf{h}^k = f_{\theta}^k(\mathbf{h}^{k-1})$  for  $k \in \{1, ..., K\}$ . The input  $\mathbf{h}^0$  and output  $\mathbf{h}^N$  of  $f_{\theta}$  are  $\mathbf{x}$  and  $\mathbf{z}$ , respectively. Concretely,  $f_{\theta}^k$  are flow layers such as squeeze layer, batch normalization layer, affine coupling layer, etc.
> 
> Generally,  $f_{\theta}$  is composed of a series of invertible trans-
> 
> According to the change of variable formula and the chain rule, for a sample  $\mathbf{x}$ , the log probability  $\log(\mathbf{x})$  can be calculated as
> 
> $$\log p(\mathbf{x}) = \log p(f_{\boldsymbol{\theta}}(\mathbf{x})) + \sum_{k=1}^{K} \log \left| \det \frac{\partial f_{\boldsymbol{\theta}}^k(\mathbf{h}^{k-1})}{\partial \mathbf{h}^{k-1}} \right|, (1)$$
>  where  $\log \left| \det \frac{\partial f_{\boldsymbol{\theta}}^k(\mathbf{h}^{k-1})}{\partial \mathbf{h}^{k-1}} \right|$  is the logarithm of the absolute value of the determinant of the Jacobian of  $f_{\boldsymbol{\theta}}^k$  at  $\mathbf{h}^{k-1}$ . The flow model can thereby be optimized by minimizing the negative log-likelihood loss.
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%2096.png)

> **[图片提取文字 (image.png)]:**
> Both image SR and image rescaling try to reconstruct the HR image x given a LR image. Since the image degradation process (or image downscaling) is the inverse of image super-resolution (or image upscaling), we can model these two processes with an invertible bijective transformation:  $x \leftrightarrow [y, a]$ , where y and a are the generated LR image and
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%2097.png)

> **[图片提取文字 (image.png)]:**
> it is reasonable to design a flow model conditional on the ground-truth LR image y\* as,  $p(\mathbf{x}|\mathbf{y}^*) \leftrightarrow p(\mathbf{y}, \mathbf{a}|\mathbf{y}^*) = p(\mathbf{y}|\mathbf{y}^*)p(\mathbf{a}|\mathbf{y}, \mathbf{y}^*).$ (2) Ideally, we hope the model can generate exactly the same LR image as the ground-truth LR image. This can be formulated as a Dirac delta function  $\delta(\mathbf{y} - \mathbf{y}^*)$  and further approximated by a multivariate Gaussian distribution as,  $p(\mathbf{y}|\mathbf{y}^*)p(\mathbf{a}|\mathbf{y},\mathbf{y}^*) = \delta(\mathbf{y} - \mathbf{y}^*)p(\mathbf{a}|\mathbf{y})$ 
> 
> (3)
> 
> the rest high-frequency component, respectively. As mod-
> 
> elling the probability of natural images is a non-trivial task,
> 
> where 
> $$\Sigma$$
>  is a diagonal covariance matrix with all diagonal elements close to zero. Note that  $\mathbf{y}$  is nearly equal to  $\mathbf{y}^*$  in this case. By further mapping  $p(\mathbf{a}|\mathbf{y})$  to a standard multivariate Gaussian distribution  $p(\mathbf{z}) = \mathcal{N}(\mathbf{z}|\mathbf{0},\mathbf{I})$ , the flow model is defined as, 
> $$p(\mathbf{x}|\mathbf{y}^*) \leftrightarrow \lim_{\Sigma \to \mathbf{0}} \mathcal{N}(\mathbf{y}|\mathbf{y}^*,\Sigma) \mathcal{N}(\mathbf{z}|\mathbf{0},\mathbf{I}). \tag{4}$$
>  As we can see, part of the latent space is constrained to
> 
>  $= \lim_{\Sigma \to 0} \mathcal{N}(\mathbf{y}|\mathbf{y}^*, \Sigma) p(\mathbf{a}|\mathbf{y}),$ 
> 
> (4)As we can see, part of the latent space is constrained to be the LR image space. In particular, decomposed highfrequency component a is conditional on another decomposed component y. Once trained, following the forward direction, HCFlow can decompose the HR image x into LR image y and latent variable z that follows a simple distribution. Following the inverse direction, HCFlow can generate x given the LR image input  $y^*$  and a random sample z from the latent distribution, as it is an invertible bijective model.
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%2098.png)

**forward推理**是image rescale，**inverse推理**是image super resolution；

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 2: Schematic computational graphs of the hierarchical conditional flow (HCFlow) with 3 flow levels. On level l,  $\mathbf{y}_{l-1}$  (note that  $\mathbf{y}_0 = \mathbf{x}$ ) is decomposed to low-frequency component  $\mathbf{y}_l$  and high-frequency component  $\mathbf{a}_l$ . The transformation between  $\mathbf{a}_l$  and  $\mathbf{z}_l$  is conditional on  $[\mathbf{y}_L, \mathbf{y}_{L-1}, ..., \mathbf{y}_l]$ , as indicated by the blue arrows. The computation orders in forward and inverse propagation are shown on the top of each node.
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%2099.png)

HR图像x前向推理得到LR图像和**隐变量z（NGS分布）**，隐变量z（NGS分布）和LR图像逆向推理得到SR图像；

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 3: The architecture of the hierarchical conditional flow (HCFlow) with 2 flow levels. For a HR image  $\mathbf{x}$ , we first squeeze, transform and split it to low-frequency component  $\mathbf{y}_1$  and high-frequency component  $\mathbf{a}_1$ . Similarly,  $\mathbf{y}_1$  is decomposed to  $\mathbf{y}_2$  (*i.e.*, the LR image in this case) and  $\mathbf{a}_2$  in the next level.  $\mathbf{a}_1$  and  $\mathbf{a}_2$  are transformed to latent variables  $\mathbf{z}_1$  and  $\mathbf{z}_2$ , conditional on  $\phi_1([\phi_2(\mathbf{y}_2), \mathbf{y}_1])$  and  $\phi_2(\mathbf{y}_2)$  (note that  $\phi_1$  and  $\phi_2$  are feature extractors, *e.g.*, CNN) respectively, in a hierarchical manner. The model is trained by negative log-likelihood loss, and can be further enhanced by pixel loss, perceptual loss and GAN loss.
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%20100.png)

## AAAI24：SCTNet（Conv+Transformer）

任务：语义分割；

Transformer效果好但代价高，单branch架构的思路是轻量化Transfomer的输入尺寸或分尺寸级联，双branch架构融合Conv和Transfomer特征来保证速度和准确率；

双分支的Transformer使用代价依然高，部分双分支网络的**交互**开销高（数据依赖的等待）；

> **[图片提取文字 (image.png)]:**
> rell 2015) leads to the tendency to utilize CNN for semantic segmentation. Following FCN, a series of improved CNN-based semantic segmentation methods are proposed. DeepLab (Chen et al. 2017) enlarges the receptive field with dilated convolution. PSPNet (Zhao et al. 2017), U-Net (Ronneberger, Fischer, and Brox 2015), and RefineNet (Lin et al. 2017) fuse different level feature representations to capture multi-scale context. Some methods (Fu et al. 2019; Huang et al. 2019; Yuan et al. 2018; Zhao et al. 2018b)propose various attention modules to improve segmentation performance. In recent years, transformer has been adopted for semantic segmentation and shows promising performance. SETR (Zheng et al. 2021) directly applies the vision transformer to image segmentation for the first time. PVT (Wang et al. 2021) introduces the typical hierarchical architecture in CNN into the transformer-based semantic segmentation model. SegFormer (Xie et al. 2021) proposes an efficient multi-scale transformer-based segmentation model.
> 
> Semantic Segmentation. FCN (Long, Shelhamer, and Dar-
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%20101.png)

> **[图片提取文字 (image.png)]:**
> tic segmentation methods (Paszke et al. 2016; Wu, Shen, and Hengel 2017) usually accelerate inference by compressing channels or fast down-sampling. ICNet (Zhao et al. 2018a) first introduces a multi-resolution image cascade network to accelerate the speed. BiSeNetV1 (Yu et al. 2018) and BiSeNetV2 (Yu et al. 2021) adopt two-branch architecture and feature fusion modules to achieve a better tradeoff between speed and accuracy. STDC (Fan et al. 2021) rethinks the two-branch network of BiSeNet, removes the spatial branch, and adds a detailed guidance module. DDR-Nets (Pan et al. 2022) achieves a better trade-off by sharing branches in the early stages. Very recently, some efficient transformer methods for real-time segmentation have been proposed, but they still have unresolved problems. Top-Former (Zhang et al. 2022) only uses transformer on 1/64 scale of the feature maps, leading to low accuracy. RT-Former (Wang et al. 2022) and SeaFormer (Wan et al. 2023) need frequent interaction between the two branches. This additional computation slows down the inference speed. In addition, there are also some single-branch and multi-branch
> 
> methods in real-time segmentation.
> 
> **Real-time Semantic Segmentation.** Early real-time seman-
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%20102.png)

双分支架构（一种trade-off）：CNN识别分割图像，Transformer对齐实体（语义）；

pointwise conv替换gemm，去除（3D>2D）flatten和reshape的开销；

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> a spatial branch at the early stage. (b) Feature sharing bilateral network separates the two branches at the latter stage and adopts dense fusion modules. (c) Our SCTNet applies a single hierarchy branch with a semantic extraction transformer, free from the extra branch and costly fusion module in inference. FM: Fusion Module, SIAM: Semantic Information Alignment Module. Dashed arrows and boxes denote training-only.
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%20103.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 4: Design of Conv-Former Block (left) and the details of convolutional attention (right). GDN means Grouped Double Normalization.  $\otimes$  means convolution operations,  $\oplus$ 
> 
> stands for addition, and k means the kernel size.
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%20104.png)

**Backbone Feature Alignment**让CNN Branch中的CF Block（Conv计算Attention）对齐Transformer branch中的block能力；

**Shared Decoder Head Alignment**让CNN Branch中Decoder对齐Tranformer branch中的Decoder能力；

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 3: The architecture of SCTNet. CFBlock (Conv-Former Block, detailed in Figure 4) takes advantage of the training-only Transformer branch (greyed-out in the dashed box) via SIAM (Semantic Information Alignment Module) which is composed of BFA (Backbone Feature Alignment) and SDHA (Shared Decoder Head Alignment).
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%20105.png)

NIPS（CCF-B）：HR和LR图像的两个Branch之间的张量交互，引入**数据依赖**，需要调度来减小数据同步的等待时间；

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 2: Illustration of RTFormer block. For low resolution, GPU-Friendly Attention is applied. And for high resolution, we use Cross-resolution Attention which draws K and V from low resolution branch. Besides, we make up FFN with two  $3 \times 3$  convolution layers.
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%20106.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 4: Illustrating the RTFormer architecture. We place RTFormer block at the last two stages which indicated by pink block and use convolution blocks at the earlier stages which indicated by blue block. Besides, we add a DAPPM module for segmentation head, drawing on the successful
> 
> experience from [17].
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%20107.png)

## TGRS2022（CCF-B）：CNN + Transformer for semantic segment of ISR

# 方法/Acc

## NV MPS、MIG、vGPU

GPU VIRTUALISATION

VIRTUALIZING HARDWARE PROCESSINGRESOURCES INA PROCESSOR

## VLSI2023：Model & HW Co-optimize for sparse CNN+SA

Sense: Model-Hardware Co-design for Accelerating Sparse CNNs on Systolic Array

SA架构作sparse CNN推理：

channel cluster，完成IFM和weight在PE的负载均衡；

负载均衡的weight prune，保证kernel的稀疏比和模型精度；

自适应数据流配置计算策略，根据IFM和weight的存储需求；

> **[图片提取文字 (image.png)]:**
> neural network(CNN) and worth exploiting for CNN accelerators, but extra processing comes with hardware overhead, causing many architectures suffering from only minor profit. Meanwhile, systolic array has been increasingly competitive on CNNs acceleration for its high spatiotemporal locality and low hardware overhead. However, the irregularity of sparsity induces imbalanced workload under the rigid systolic dataflow, causing performance degradation. Thus, this paper proposed a systolicarray-based architecture, called Sense, for sparse CNN acceleration by model-hardware co-design, achieving large performance improvement. To balance input feature map(IFM) and weight loads across Processing Element(PE) array, we applied channel clustering to gather IFMs with approximate sparsity for array computation, and co-designed a load-balancing weight pruning method to keep the sparsity ratio of each kernel at a certain value with little accuracy loss, improving PE utilization and overall
> 
> Abstract—Sparsity is an intrinsic property of convolutional
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%20108.png)

> **[图片提取文字 (image.png)]:**
> applied to determine the computing strategy based on the storage ratio of IFMs and weights, lowering  $1.17 \times 1.8 \times DRAM$  access compared with Swallow and further reducing system energy consumption. The whole design is implemented on ZyngZCU102 with 200MHz and performs at 471-, 34-, 53- and 191-image/s for AlexNet, VGG-16, ResNet-50 and GoogleNet respectively. Compared against sparse systolic-array-based accelerators, Swallow,
> 
> performance. Additionally, Adaptive Dataflow Configuration is
> 
> FESA and SPOTS, Sense achieves  $1 \times 2.25 \times$ ,  $1.95 \times 2.5 \times$  and  $1.17 \times 2.37 \times$  performance improvement on these CNNs respectively with reasonable overhead.
> 
> Index Terms—systolic array, hardware accelerator, sparsity, weight pruning, convolutional neural network.
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%20109.png)

加速器设计目标：增大（有效）计算并行度来降低延迟，增大数据复用来提高能效；

相关工作对模型作硬件友好的稀疏prune，基于稀疏方式设计加速器，挑战是：

因为稀疏的随机性导致**PE负载不均衡**，无法**同时利用weights和IFM**的稀疏性，IFM和weights的随机稀疏导致的**内存低效访问**，稀疏格式的处理需要消耗特定硬件资源而**性能受限**（FPGA Acc的LUT、BRAM）；

> **[图片提取文字 (image.png)]:**
> CNNs [10], [11], [26], [27] for higher energy efficiency and reource efficiency compared with other architectures as shown in Tab. Thus, researchers try to process sparsity with systolic architecture to improve the overall benefits. Swallow [28] overcomes the inability to exploit the sparsity of weights and IFMs, CONV layers and FC layers of CNNs with limited resource in a systolic array, and introduce a sparse-aware dataflow to boost PE utilization, achieving higher bandwidth saving and energy efficiency compared with previous sparse accelerators. However, the structured systolic dataflow essentially contradicts with the irregularity of sparsity, causing imbalanced PE loads. Considering that, FESA [29] pruned the kernels to 2~7 formalized zero distribution patterns and left IFM
> 
> Since systolic array [25] is widely applied to accelerate
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%20110.png)

> **[图片提取文字 (image.png)]:**
> achieving lower sparsity processing overhead. But this pruning method is only implemented on Cifar-10 and Cifar-100 [30] currently. Thus, to balance workload with higher versatility in systolic array, SPOTS [31] designed a group-wise pruning method to divide weights into groups and prune some elements of the same position in each group, which achieves similar versatility with shape-wise pruning method [32] and improves compatibility with systolic array. Accordingly, SPOTS applied Image to Column (Im2Col) transformation of IFMs coupled with general matrix-matrix multiplication (GEMM) to better fit its pruning scheme into systolic array by skipping the weight rows and IFM columns with all zeros. However, since its pruning method is too fine-grained, the sparsity of weights after pruning is bounded by accuracy. Besides, SPOTS fails to exploit the sparsity in those rows and columns with some zeros, causing inefficient acceleration.
> 
> unprocessed to regularize dataflow as the dense systolic tempo,
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%20111.png)

> **[图片提取文字 (image.png)]:**
> of system energy consumption, making it critical to further reduce memory access through dataflow. But sparse IFMs and weights can be irregular and fragmented, which leads to lower memory access efficiency. SCNN [22] employed a novel dataflow to eliminate unnecessary data transfers, but access contentions occurred when routing the products to accumulator buffer due to irregular sparse patterns. Lu et al [24] proposed a weight layout to enable efficient memory access without conflicts, but huge LUT consumption blocked the performance. Swallow harnesses a sparsity-aware dataflow with matrix multiplication tiling to promote data reuse within each channel, reducing DRAM access with little overhead. However, Swallow always preferentially reuse IFMs, while DRAM access can be variable if we choose different reuse
> 
> Additionally, memory access occupies a huge proportion
> 
> IFMs and weights in each layer.
> 
> These previous sparse systolic accelerators suffered from imbalanced workload, lacking versatility or low sparsity of weight pruning. Besides, the dataflow is inflexible for the variable ratio of IFM and weight in each layer. Thus, this paper aims to balance workload to fit with sparse systolic array, while maintaining the sparsity and versatility of weight pruning with reasonable overhead, and further
> 
> optimize DRAM access. A model-hardware co-design of
> 
> sparse CNN accelerator based on systolic array is proposed
> 
> to improve system performance and energy efficiency. Our
> 
> main contributions are as follows:
> 
> strategies. Thus, there is still room to further lower DRAM
> 
> access by choosing dataflow according to the storage ratio of
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%20112.png)

## SPOTS：An Accelerator for Sparse Convolutional Neural Networks Leveraging Systolic General Matrix-matrix Multiplication（Sparse、IM2COL）

**动机**：im2col耗时间且访问耗能，软件完成不能利用im2col和GEMM的pipeline；

> **[图片提取文字 (image.png)]:**
> volutional layer as a large, single **General Matrix-Matrix Multiplication (GEMM)** using a data reorganization transformation called Image-to-Column (IM2CoL). Unsurprisingly, many mainstream frameworks use this approach since highly optimized GEMM primitives are available (e.g., BLAS [4] or CuBLAS [30]). One method to accelerate the convolution computation is to offload the GEMM operation to a hardware accelerator. However, the Im2CoL operation accounts for a sizable fraction of the execution time (29% of the total time). Further, IM2Col performs many redundant memory accesses, which contributes to the overall energy consumption. Further, offloading only the GEMM operation to a hardware accelerator and doing the IM2CoL operation in software pre-
> 
> vents fine-grained pipelining of the IM2CoL transformation and the matrix multiplication opera-
> 
> tion. Thus, performing the IM2CoL operation in hardware avoids significant data transfer between
> 
> the CPU and the hardware accelerator.
> 
> Convolution as matrix multiplication. One approach to implement CNNs is to realize a con-
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%20113.png)

> **[图片提取文字 (image.png)]:**
> creates a set of linearized patches. The IM2CoL unit consists of PUs where each PU is responsible for constructing a linear patch. As values are streamed in, the PU constructing the patch will forward overlapped elements to neighboring PUs. Once the PU collects all the values in a patch, it forwards in-order partial patches to the GEMM unit. This approach allows the IM2Col unit to read in values from the input feature map only once and reuse the values avoiding redundant memory accesses. We design a dynamically reconfigurable GEMM unit with a systolic-array-based design. It can be configured as a tall array to balance the work between IM2CoL and GEMM computation. To maintain a high PE utilization with CNN layers with varying shapes, the GEMM units can be configured as small GEMM units (Section 3.4). This dynamic reconfigurability enables our hardware to adapt to CNN layers with varying dimensions and shapes. Further, it also helps with sparsity
> 
> We propose a hardware unit for the IM2Col transformation that is synergistic and pipelined
> 
> with the hardware unit for GEMM. The IM2CoL unit reads the input feature map, a 3-D array, and
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%20114.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> (c) The steps to generate the patches with two PUs
> 
> Fig. 4. Illustration of patch generation using the PUs in the IM2CoL unit. We show an IM2CoL unit with two PUs for exposition. (a) The input feature map with one channel. We show the sliding windows used to generate patches with a stride of 1. (b) The two PUs are interconnected by a ring network. (c) There are two rounds. Round 1 corresponds to patches belonging to the first row of sliding windows over the input feature
> 
> map. Similarly, round 2 corresponds to patches belonging to the second row of sliding windows.
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%20115.png)

im2col模块，和我的设计原理类似，但我的存储Buffer粒度更粗，不需要分布式buffer；

im2col过程只读取一次SRAM的特性不同，SPOTS作了H和W两个方向overlap的reserve，我只作了W方向的reserve，多任务场景下我作一个**row-wise的切换**（正好需要重复读取）？；

SPOTS中im2col模块的每个PU需要一个控制器来**分布式**控制3个buffer行为，而我是一个整体控制器控制top buffer的行为；

每个PU中的reserve buffer就是我的slab buffer，但我能**统一每个tile的overlap的大小**（不多消耗buffer），因此控制和设计更简单，pipeline粒度更细；

im2col+GEMM的pipeline相同；

SPOTS中im2col模块的PU分布式控制的对上接口的粒度是layer，即PUs动态协同控制完成layer输入的im2col，我的im2col设计的控制和模块很简单，更适合**多任务场景**？

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> (a) Overall architecture of SPOTS
> 
> (b) Overall IM2CoL architecture and patch unit
> 
> Fig. 3. (a) The overall architecture of our accelerator with the IM2CoL unit and a systolic-array-based GEMM unit. (b) The overall IM2CoL architecture and patch unit internals.
> 
> awareness by enabling our design to detect and skip zeros in the input feature map (Section 3.3). Figure 3(a) shows the overall architecture of our accelerator. The two main components are the unit for the IM2Col transformation and the GEMM unit. They are connected by two buffers that allow effective pipelining of the operations between the IM2Col unit and the GEMM unit. The compress unit detects and skips the zero blocks in the feature map and weights before they are sent to the GEMM unit. Next, we describe the details of each component.
> 
> The IM2CoL transformation creates a 2-D matrix from the 3-D input feature map, which reduces
> 
> ## 3.1 The IM2Col Unit
> 
> output controller.
> 
> convolution to matrix multiplication (Section 2.2). The IM2Col transformation is challenging because it inherits a part of the complexity of convolution, has complex memory access patterns, and results in redundant accesses. We propose a distributed hardware structure consisting of a series of PUs to both accelerate IM2Col and minimize the number of accesses to the elements of the input feature map. The key insight in our IM2Col unit is to exploit the localities resulting from the overlap between the patches as we slide the filters across the input feature map both vertically and horizontally. Each PU is responsible for building one patch at a time. One of our design goals is to read the input feature map only once from SRAM. To accomplish this goal, each patch unit has small local buffers that store some values that will be useful for building future patches. The PUs are also connected using a ring network, which allows the PUs to communicate elements locally and avoid redundant accesses to the input feature map in SRAM. Figure 3(b) shows the overall architecture of our IM2Col unit that consists of three main components: input controller, PUs, and
> 
> The input controller reads the input feature map from SRAM and forwards them to the appropriate PUs. Apart from sending values from the input feature map to the respective PUs, the input controller maintains extra metadata for every scheduled patch. This metadata carries information about the position of the current patch. For some convolution layers, the stride size is the same as the kernel size. In those cases, there is no overlap between the patches. For those scenarios, the input control forwards its output directly to the output controller by skipping the PUs.
> 
> Our IM2Col unit has multiple PUs within it. The PUs are the main components of the IM2Col unit for generating patches. Figure 3(b) shows the internals of the PU. Each PU has three buffers: the new buffer, the neighbor buffer, and the reserved buffer. The new buffer (N) maintains the newly
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%20116.png)

> **[图片提取文字 (image.png)]:**
> fetched element received from the input controller. The neighbor buffer (G) stores the elements received from the neighboring PU. The reserved buffer (R) stores some of the elements previously received at that PU in the previous rounds. We store the row and column indices (i.e., coordinates) along with the value for each element. The control unit within each PU manages the buffer and generates patches. It decides whether an element needs to be forwarded to the neighboring PU and whether it should be maintained in the reserve buffer for future use.
> 
> A unique identifier identifies each patch (i.e., row and column index of top-left element). The control unit in a PU uses the patch identifier, the filter size, and the stride size to determine which elements need to be (1) fetched from the input feature map, (2) forwarded to the neighboring PUs, and (3) stored in the reserve buffer for future rounds. For example, all elements need to be fetched from the input feature map when a PU processes the first patch in the first round.
> 
> All elements that are necessary for adjacent patches in a given round are provided by the neighboring PUs. A PU typically receives  $K^2 - K \times S$  elements from the neighboring patches as long as it is not the first patch in a given round, where K is the size of the kernel and S is the stride size. We assign all patches that belong to the same column (i.e., column index of the top-left element) in different rounds to the same PU. Hence, the PUs also store some elements that may be useful to build patches in subsequent rounds in the reserved buffer. This procedure is repeated for all C channels in the feature map.
> 
> The total number of elements that are overlapped between the vertical patches for a given filter
> 
> size is  $C \times W \times (K-S)$ , where W is the width of the input feature map. This is the maximum data reuse that can be attained with the reserve buffer. Further, the width and the channel size are inversely proportional to each other. For example, the first few layers of a CNN often have a small number of channels that are wider. In contrast, the later layers of the CNN have larger channels of smaller width. Thus, a small reserve buffer can provide significant data reuse even for larger layers. When the number of overlapped elements between the vertical patches is larger than the size of the reserved buffer, the input controller skips the reserved buffer and fetches the element again from SRAM. In such cases, data reuse is restricted to horizontally adjacent patches. Finally, the output controller organizes patches formed by each PU and manages communications with the GEMM unit. It coordinates double buffering that enables the overlapped execution of the IM2CoL unit and the GEMM unit.
> 
> example, PU1 receives four elements (A1, A6, A2, A7) from the input controller and stores them in the new buffer in step 1. Similarly, PU2 receives two new elements (A3, A8). PU2 will receive other elements from the PU1 in subsequent steps (i.e., step 2).
> 
> In summary, our hardware IM2Col unit provides two benefits: energy efficiency and perfor-
> 
> mance. Accessing the smaller SRAM and performing integer operations (for computing on row and column indices) consumes significantly less energy than accessing DRAM and large SRAMs. Hence, our design provides significant energy benefits. Further, our distributed collection of PUs unlocks extra parallelism beyond parallelism among the channels, allowing multiple patches to be built simultaneously by different PUs in the IM2Col unit that boosts performance.
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%20117.png)

GEMM提供**两种dataflow**（tall tile、sub-tall-tiles），利用稀疏CNN提供的特性；

tall-shape GEMM的设计是为了降低IM2COL吞吐的需求，从而减少IM2COL的资源，意味着论文PU的**资源消耗较大**；

> **[图片提取文字 (image.png)]:**
> one of the inputs of the GEMM unit comes from the IM2Col unit. Using a tall-shaped array reduces the memory bandwidth requirement for the input arriving from the IM2Col unit. Thus, we can attain high PE utilization in the GEMM unit with less throughput from the Iм2Col unit. This helps us to build an IM2Col unit with fewer resources and memory bandwidth requirements. Second, the tall array helps our design to exploit sparsity in the output of the IM2Col unit to skip zeros and increase performance. As the width of the tall array is smaller than its height, fewer columns from the IM2Col transformation enter the systolic array at any instant of time, which increases the opportunity for detecting and skipping entire rows of inputs with zeros before entering the
> 
> There are two main benefits in using a tall systolic-array-based architecture for GEMM. First,
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%20118.png)

output stationary的特点：128 rows，4 cols；

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> (c) Cycle by cycle execution of GEMM with an output-stationary dataflow
> 
> Fig. 5. Illustration of our GEMM unit. (a) Inputs to the GEMM unit. (b) A tall array for the GEMM unit. (c) Illustration of GEMM computation at various steps. We show the current inputs and the partial results computed till a step for each PE. We demonstrate the output-stationary attribute of our design.
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%20119.png)

> **[图片提取文字 (image.png)]:**
> sult by accumulating the partial products for a particular output element. This output-stationary dataflow ensures maximum reuse of the output data. Besides, with a tall array, SPOTS can attain high data reuse for the result of the IM2Col transformation (i.e., feature map input). More importantly, with output-stationary dataflow, there is no need for separate multiplication and accumulation units. This eliminates multiple levels of multiplication and addition and the routing logics between the two units (Section 2.3). Figure 5(a) shows the weight matrix from the filter and the output of the IM2Col transformation that forms the input to the GEMM unit. The values of
> 
> the filter matrix enter the GEMM unit's systolic array from left to right, while the result of the
> 
> IM2Col unit enters the systolic array from top to bottom. Figure 5(c) shows the various steps and
> 
> Our GEMM unit uses an output-stationary dataflow, where a given PE computes the final re-
> 
> partial results computed in the GEMM unit. Our design is parameterizable with *M* rows and *N* columns in the systolic array. In our design, each row handles multiple rows of the filter matrix. Our specific prototype used 128 rows of PEs and 4 columns. These numbers are chosen based on the characteristic of common CNN layers. Further, each row of the systolic array can be assigned multiple rows of the filter matrix depending on the scheduling mode. The majority of layers in state-of-the-art CNNs have fewer than 512 rows of the filter matrix in each convolution layer.
> 
> Each PE has a single **multiply-accumulate** (MAC) unit that uses two 16-bit fixed-point inputs and accumulates the result in a 24-bit register. To handle multiple rows of the filter matrix, each PE has K registers to compute the final result (e.g., in our design, we use K = 4). Each PE has three FIFOs. Two FIFOs are for each arriving input. The other FIFO works as the work queue for the MAC unit. In GEMM, the coordinates of the elements of the two input matrices should
> 
> for the MAC unit. In GEMM, the coordinates of the elements of the two input matrices should match before multiplying the inputs. In the fetch unit, we ensure that the inputs are sent to the PEs in the proper order; thus, we do **not** need additional logic to perform index matching inside a PE. Additionally, our output-stationary dataflow ensures all the partial products produced in a PE belong to the same output element. Next, we describe how to support sparsities in both inputs without requiring any index matching units inside the PEs.
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%20120.png)

**Sparse handling**

以block为粒度的结构化稀疏，column-block的稀疏Map；

Sparse Weight——Buffer——Compress（skip）——GEMM；

Dense Input——Buffer——IM2COL——Compress（bitmap、skip）——GEMM；

> **[图片提取文字 (image.png)]:**
> unit in our accelerator (Figure 3(a)) identifies a block of zeros in the result of the Iм2Col transformation. It creates a bitmap for every block coming out of the IM2Col unit. If all elements in a block in the output of the IM2Col unit are zeros, the bit is set to zero for that block; otherwise, the bit is set to one. Subsequently, the input controller of the GEMM unit uses this bitmap and M1 level bitmaps for the weights (Figure 7(a)) to skip blocks of the input feature map and weights on the fly when they are all zeros. One unique feature of our approach is that we skip MAC operations involving zeros outside
> 
> the PEs and in the input controller. These have two advantages. First, we avoid the unnecessary
> 
> **Skipping zeros in the feature map and weights.** The *compress* component before the GEMM
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%20121.png)

> **[图片提取文字 (image.png)]:**
> Second, detecting and skipping zeros centrally (inside the input controller) relieves the PEs from storing and processing any metadata, which reduces area and power consumption. Besides, our approach does not require any costly hardware units inside every PE to detect and match the nonzero pairs, unlike some prior work (Section 2.3). Figure 7(b) illustrates how the zero columns in the weight matrix and the zero rows in the output of the IM2Col unit are skipped. In addition to the zero blocks that we skip in the control unit, some PEs may still receive zero blocks (the gray blocks in C1, C2, and C4 columns in Figure 7(b)). This happens when a column of the weight matrix is partially zero. For those cases, the input controller sends one bit to the PE to indicate a zero block. The PEs will then ignore the blocks with all zeros, and the MAC units are gated to reduce energy consumption.
> 
> data traffic to stream the rows of feature maps and columns of filters to PEs when they are zeros.
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%20122.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> (a) Our custom sparse format to store filters
> 
> (b) Skip rows and columns with all zeros
> 
> Fig. 7. (a) Our custom sparse format to store filters. (b) Illustration of how our design skips rows and columns with all zeros. (1) Weight matrix with the metadata about columns with all zeros. (2) The IM2Col result with the metadata about rows with all zeros. (3) If a row or a column is all zeros, all such rows and columns can be skipped (i.e., *and* operation of the row and column metadata). (4) GEMM computation when rows and columns are skipped. For example, the first element of column C4 will be fetched by the first PE in cycle 2 (skipping columns C2 and C3).
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%20123.png)

**GEMM reconfiguration**

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Fig. 8. (a) Enhancements to reorganize the tall systolic array (SA) as multiple GEMM units. (b) Illustration of how inputs are distributed in the configuration with multiple GEMM units.
> 
> enhancement allows our design to be more **adaptive** to different layer shapes and thus maintains high PE utilization under different conditions. Figure 8(a) demonstrates how a tall array can be used as two smaller arrays using the multiplexers. Hence, the PEs now either can receive the input from the PEs above (i.e., it forms a tall array) or can get the input from a different IM2Col unit. These multiplexers can be configured based on the mode register dynamically depending on the structure of a layer. The weight matrix is broadcast to all small systolic arrays when the GEMM unit is configured as smaller systolic arrays. Each small GEMM unit receives the feature map input from their assigned IM2Col units. The two GEMM units compute two independent groups of columns of the final result matrix (i.e., GEMM 1 computes result columns from 0 to N/2, GEMM computes the columns from N/2+1 to N). In our prototype, we have four IM2Col units. There is one main IM2Col and three smaller IM2Col units to support the two configurations. The main IM2Col unit is used for the tall array configuration. For the other configuration, all four IM2Col units are being used. This dynamic reorganization of the GEMM unit's systolic array coupled with the multiple IM2Col units enables our hardware to maintain high PE utilization for various CNN layers with
> 
> different shapes.
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%20124.png)

Load Balance

## Swallow：

## DAC24：window-based attention Acc

swin Transformer：sparse attention

[https://zhuanlan.zhihu.com/p/362672090](https://zhuanlan.zhihu.com/p/362672090)

## ITSC24：Inter-Operator Schedule for CNN on GPU

## ICS24：HW-aware(quantization) NAS for ViT

## ISCA23：Eagar Correlation Prediction based FFN-Attention Co-optimized Transformer Acc

## DAC22：length-adaptive sparse transformer Acc

## MICRO16：fused CNN Acc