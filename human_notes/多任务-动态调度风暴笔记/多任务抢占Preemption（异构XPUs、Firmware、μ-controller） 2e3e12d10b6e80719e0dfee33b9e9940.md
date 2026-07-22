# 多任务抢占Preemption（异构XPUs、Firmware、μ-controller）

## idea

目前：给一个conv算子，加载-计算-存储和固定tiling全在片上，没有ld、st指令，img2col fused在片上。

场景：多任务，VAE（conv）、Diffusion（conv、attention）、VAE在LDM（VAE+Dif）中时间不长。

修改：我实现的conv是一个编程时算子，分块-加载-计算-传输只由硬件控制完成，将硬件控制改为参数化的**粗粒度XPU指令**。

### 背景

云边场景下，可以用**边缘加速卡**抢占式调度解码和编码和DiT，云端负责llm生成修饰词，由于云和网络延迟，边缘需要抢占式DiT调度。

为什么不用GPU？吞吐率不如SA更高。

云边DiT不常见，找并行。

边端小模型，MoE的小专家并行（粗粒度），和多任务，

稀疏激活，动态激活（多任务），SpMM和SpMV，算子抢占（时间抢占、算力抢占）

抢占：preemption、flush、Spatial Acc的多任务并行（片上资源的动态分配）类似RDU？

任务的CFG重排能力：片上资源的弹性变换支撑多任务。抢占开销，结合LLM特点。

抢占：进一步利用性能，动态插入算子、让算子并行，不浪费acc的空间资源，只算一个warp，动态插入算子，（类似GPU的多warp并发，不浪费memory时间）。

根据资源匹配插入的任务。

而不是任务抢占并发。

LLM的可分拆点。

Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models

**什么算子抢占什么算子？**

DiT、ViT中Transformer Decoder比VAE的负载更重，因此**attention抢占Conv和attention之间抢占的可能性更大。**

LLM、Transformer的KV Cache（PD）计算需要MMU和MVU，多卡调度时，attention的offload schedule（W和KVCache分GPU存放和运算，GPU之间传输QKV向量，缓解显存压力）更常见，因为attention的负载很大，每个decoder块有一个KV Cache（token loop）。

ViT、DiT的T steps和llm的逐个输出token很相似，是否意味着DiT的BackBone也能利用KV-Cache的策略来减少计算复杂度？不行❌，ViT和DiT的推理模式不属于自回归，由于DiT的time loop，可以设置Cache来替换ViT BackBone。

### 相关工作

相同抢占场景下，指令粒度的影响和区别？

GPU线程块的细粒度kernel指令，计算尺度等价于spatial acc的张量粗粒度指令。但指令级抢占粒度，GPU比ACC更灵活。

抢占的核心是满足运行时的QoS，以及整体的QoS，功耗等，优化目标是特定kernel尽快部署计算。

领域spatial acc的粗粒度指令，将new kernel和old kernel也作为pipeline，进一步压榨计算能力，相当于粗指令之间中断，粒度等价于GPU的细粒度指令抢占。

### 设计考虑

观察1：GPU支持指令level抢占，其他Acc一般不支持。

指令级抢占的过程：XPU中断执行，保存上下文，（恢复上下文），新指令执行。

观察2：host命令XPU（可能包含μ-controller）是异步的，host提交多个（TH）命令给XPU。

host和XPU异步，即host一般提交多条指令给XPU，因此XPU保存还需要保存这些接受的指令，多任务时，XPU的（部分）**指令序列需要保存**，**指令粒度不宜过细**。

μ-controller和XPU同步，辅助XPU完成通用计算和控制。

策略交给软件决定，硬件需要提供给软件所需的接口。比如保存上下文时反映资源使用率？设计合适的硬件接口。

观察3：img2col重排算子需要和GEMM计算融合，否则重排后的数据保存时浪费带宽（比原来输入更大）。

img2col+GEMM需要作融合算子的tiling+中断设计，即**指令设计+保存/恢复上下文的格式**。

img2col本身是融合GEMM的tiling设计，tile重排时不中断。

脉动阵列具备很高自由度的pipeline设计，能在GEMM被抢占时，将中间结果传出。保持MAC之间的内部位宽24bit而不需要拓宽，每个MAC两个cycle传出32bit结果，需要2*SA-size个cycle，硬件资源开销不大。

观察4：SA的中间结果，Conv输入的Overlapped data（input row、slab），权重都需要保存，开销很大。

**计算传输覆盖pipeline**是计算时**写回**上一个tile的结果，**加载**下一个tile的输入和权重。

XPU收到中断后，恢复新指令的上下文（加载），同时完成当前指令的（部分？）计算，之后新指令执行，进行被抢占指令的上下文保存（写回），类似的**计算传输覆盖pipeline。**（新指令在SA计算的同时，SA传出被抢占指令的中间结果，能利用SA的输入输出脉动特性）

观察5：接受抢占信号后，XPU直接中断需要保存的上下文可能很多，影响多任务整体吞吐，如何决定**什么时候中断（动态中断？）**，达到整体性能和高优先级任务抢占延迟的甜点？

Conv、GEMM等locality强的算子被抢占时，输入Overlapped data无法复用（若存在），设计用其**“偷偷”**完成下一个tile的部分计算，**保存中间结果**后直接丢弃。

img2col在重排后数据进入GEMM作reduction后才执行中断。

上下文不保存全部而是压缩后重新计算（传输延迟？）

当新指令的数据加载完成后，上述行为若完成，则继续pipeline，若未完成，根据新指令的优先级程度，决定强行中断还是算完才中断。

观察6：**Conv flexible tiling**和Decoder static tiling

高of高ox，高of高if在conv中能取得较高的使用率：涉及layout transformation（if紧密和ix紧密的im2col，sa逐行或者列输出到输出buffer）和double-dataflow，两种parallel的计算模型和抢占设计。

ix-img2col：输入是列Banks，每行ix，每次左移s个数，每次输出前k个数。

**if-**img2col：输入是if Banks，每行if，每次一行，若干行对应x，按列输出是of，转为ix需要sa输出一个就写一个O Bank，每个O Bank是一个x的of列，维持if需要集齐一列一起写入一组O Banks。

原生DiT参数量没那么大，只涉及**MM和重排**，所以能否设计抢占式DiT或UNet。patchify是原生的tiling，DiT是静态分辨率，因此只需要**参数化静态tiling**，但需要设计重排和MM的融合。

### 实现和实验

RV SoC DLA模拟，Verilog实现关键部件（重排、MMU、抢占）。

RV Core作为DLA的控制，协同完成抢占和指令集。

## 多任务抢占

[https://www.usenix.org/conference/osdi25/presentation/shen-weihang](https://www.usenix.org/conference/osdi25/presentation/shen-weihang) 

XPU Preemption：异构XPUs的统一调度框架能**避免重复开发**、多任务、调度需求

> **[图片提取文字 (image.png)]:**
> ## XPUs are Widely Adopted
> 
> ![](_page_0_Figure_1.jpeg)
> 
> Typical scenarios:
> 
> ![](_page_0_Picture_3.jpeg)
> 
> ![](_page_0_Picture_4.jpeg)
> 
> ![](_page_0_Picture_5.jpeg)
> 
> Autonomous Vehicles
> 
> ![](_page_0_Picture_7.jpeg)
> 
> ![](_page_0_Picture_8.jpeg)
> 
> ![](_page_0_Picture_9.jpeg)
> 
> Cloud Services
> 
> Edge Devices
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2010.png)

> **[图片提取文字 (image.png)]:**
> ## XPU Multitasking is Common
> 
> Cloud: improve utilization, save cost
> 
> ![](_page_0_Picture_2.jpeg)
> 
> Edge: limited XPU resources
> 
> 30+ tasks are deployed on  $1\sim2$  Orin GPU
> 
> One XPU can serve tens of job instances
> 
> ![](_page_0_Picture_6.jpeg)
> 
> ![](_page_0_Picture_7.jpeg)
> 
> Perception
> 
> Planning & Control
> 
> Fatigue Detection
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2011.png)

> **[图片提取文字 (image.png)]:**
> ## Diverse Scheduling Requirements of Apps
> 
> ![](_page_0_Figure_1.jpeg)
> 
> ![](_page_0_Figure_2.jpeg)
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2012.png)

> **[图片提取文字 (image.png)]:**
> ## Diverse Scheduling Requirements of Apps
> 
> ![](_page_0_Picture_1.jpeg)
> 
> Task preemption is a crucial mechanism to meet diverse application needs
> 
> Tail HC33 / THI Oughput
> 
> ![](_page_0_Picture_4.jpeg)
> 
> ![](_page_0_Picture_5.jpeg)
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2013.png)

### 硬件调度、host+XPU调度

NPU执行AI会议：语音转录任务低频长持续（一秒转录一句话，每次800ms）、虚拟背景任务高频短持续（一秒25帧，40ms一次，每次几ms）。**需要抢占机制**，但XPU不支持。

> **[图片提取文字 (image.png)]:**
> ## Issues of Native Hardware Scheduling
> 
> ## #1 Most XPUs lack preemption support
> 
> Non-preemptive FCFS policy is naturally embedded in the HW, FW, and drivers.
> 
> ![](_page_0_Figure_3.jpeg)
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2014.png)

> **[图片提取文字 (image.png)]:**
> ## Example: Al Conference on Intel NPU
> 
> ![](_page_0_Figure_1.jpeg)
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2015.png)

硬件实现简单的**静态调度**，没有**高级调度策略**。高级调度策略（如MLFQ）在XPU上的运行和硬实现开销高。

不能利用运行时信息（统计QoS、ddl）**灵活切换调度策略**（FCFS+优先调度）。

> **[图片提取文字 (image.png)]:**
> ## Issues of Native Hardware Scheduling
> 
> ## #2 Hardware schedulers are not flexible
> 
> ![](_page_0_Figure_2.jpeg)
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2016.png)

> **[图片提取文字 (image.png)]:**
> ## Issues of Native Hardware Scheduling
> 
> #2 Hardware schedulers are not flexible
> 
> ## XPU hardware scheduling mismatches application requirements
> 
> ![](_page_0_Figure_3.jpeg)
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2017.png)

调度方式应该是**CPU调度+XPU执行**，但是现有框架不通用：

1、**抢占式调度面向GPU设计**，其他XPU不支持。> **通用的硬件抢占设计**？支持异构硬件？

2、无法利用硬件新特性，因为**特性绑定特定软件栈（运行时）**才可使用。> ？？？

3、不同XPU的**算子抽象和约束不同**，**难以统一调度**异构算力。> ？？？

**host+XPU现有框架的实现**：

GPU可编程性实现**低开销抢占**（设置timeslice、设计**钩子kernel** ），利用嵌入式GPU的**ioctl接口**。

XPU的**软件栈复杂**，调度框架的调度策略受**特定XPU的细节限制**（TimeGraph绑定DRI驱动、REEF绑定AMD的运行时和驱动）。

> **[图片提取文字 (image.png)]:**
> ## Approach: Host-Managed Scheduling
> 
> ![](_page_0_Figure_1.jpeg)
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2018.png)

> **[图片提取文字 (image.png)]:**
> ## Existing Solutions are not GENERAL
> 
> ![](_page_0_Picture_1.jpeg)
> 
> **CFS** 
> 
> SJF
> 
> STCF
> 
> **MLFQ** 
> 
> ![](_page_0_Picture_6.jpeg)
> 
> ## #3 Not Uniform
> 
> hinder hardware-agnostic policies & cooperative scheduling
> 
> ![](_page_0_Picture_9.jpeg)
> 
> Existing Solutions, e.g., Effisha [PPoPP' 17], FLEP [ASPLOS' 17], REEF [OSD1'22], ...
> 
> ![](_page_0_Picture_11.jpeg)
> 
> ![](_page_0_Picture_12.jpeg)
> 
> ![](_page_0_Picture_13.jpeg)
> 
> ![](_page_0_Picture_14.jpeg)
> 
> ![](_page_0_Picture_15.jpeg)
> 
> ## #1 Not Portable
> 
> designed for certain GPU, while new XPUs emerging
> 
> ![](_page_0_Picture_18.jpeg)
> 
> ## #2 Not Evolvable
> 
> for new / deprecated HW features, while XPUs keep advancing
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2019.png)

host+XPU虚拟化的实现挑战：每个XPU有**自己的垂直软件栈**，**XPU能力不同（可编程性、timeslice、抢占）**。

> **[图片提取文字 (image.png)]:**
> ## Existing Solutions are not GENERAL
> 
> ![](_page_0_Picture_1.jpeg)
> 
> ![](_page_0_Picture_2.jpeg)
> 
> ![](_page_0_Picture_3.jpeg)
> 
> STCF
> 
> ![](_page_0_Picture_5.jpeg)
> 
> ![](_page_0_Picture_6.jpeg)
> 
> #3 Not Uniform
> 
> hinder hardware-agnostic policies & cooperative scheduling
> 
> ## Our Goal: **GENERAL** & **PREEMPTIVE**Host-managed XPU Scheduling
> 
> ![](_page_0_Picture_10.jpeg)
> 
> ![](_page_0_Picture_11.jpeg)
> 
> ![](_page_0_Picture_12.jpeg)
> 
> ![](_page_0_Picture_13.jpeg)
> 
> # I Not Portable designed for certain GPU, while new XPUs emerging
> 
> ![](_page_0_Picture_15.jpeg)
> 
> ## #2 Not Evolvable
> 
> for new / deprecated HW features, while XPUs keep advancing
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2020.png)

> **[图片提取文字 (image.png)]:**
> ## Challenges for General XPU Scheduling
> 
> scheduling could be tightly coupled with XPU details **CUDA Libs ROCm Libs CUDA RT** HIP RT oneAPI Libs **CUDA RT** User-space Libs ZE GPU ZE NPU **CUDA** Driver HIP Driver cuDLA VPI Kernel Modules **NV KMD** AMD KMD i915 **iVPU** NV L4T **XPUs** 
> 
> #2 Diverse HW Capabilities
> 
> #1 Complex SW Stack
> 
> scheduling could rely on specific functionalities e.g., programmability, CU reset, timeslice control
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2021.png)

### XQueue模型（XPU的线程模型）

CPU（进程）提交命令（指令）到**hwQueue**，XPU基于hwQueue串行执行命令，CPU命令提交后无法干涉命令执行，CPU进程**等待**XPU完成命令而**同步**。

> **[图片提取文字 (image.png)]:**
> ## Opportunity: XPU Programming Paradigm
> 
> ```
> def infer_task():
> memcpyH2D(...) # memcpy cmd
>          # kernel cmds
> conv(...)
> relu(...)
> softmax(...)
> memcpyD2H(...) # memcpy cmd
> sync(...) # wait for completion
> ```
> 
> ![](_page_0_Figure_2.jpeg)
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2022.png)

> **[图片提取文字 (image.png)]:**
> ## Opportunity: XPU Programming Paradigm
> 
> ![](_page_0_Picture_1.jpeg)
> 
> ## An XPU task can be abstracted as a sequence of commands executed on a command queue
> 
> memcpyD2H(...) # memcpy cmd
> sync(...) # wait for completion
> 
> Tensor Ops Algorithms
> 
> ![](_page_0_Picture_5.jpeg)
> 
> Levelzero Command Queue
> 
> OpenCL Command Queue
> 
> . . .
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2023.png)

**XQueue**≈thread是进程的多任务调度模型，便于实现不同调度策略。

XQueue是可被CPU抢占的hwQueue，CPU提交命令到XQueue A后，可以暂停XQueue A的命令执行，恢复XQueue B的命令执行，并将XPU的hwQueue指针引导到XQueue B。

> **[图片提取文字 (image.png)]:**
> ## XQueue Abstraction: XPU Command Queue
> 
> ![](_page_0_Figure_1.jpeg)
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2024.png)

> **[图片提取文字 (image.png)]:**
> ## XQueue: Preemptible XPU Command Queue
> 
> ![](_page_0_Figure_1.jpeg)
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2025.png)

> **[图片提取文字 (image.png)]:**
> Table 1: Interfaces of the XQueue abstraction.
> 
> XQueue Interface Description
> 
> submit (xq, cmd) Submit a command (cmd) to XQueue (xq)
> 
> wait (xq, cmd) Wait for a given *cmd* in *xq* to complete
> 
> suspend (xq) Suspend xq to pause task execution resume (xq) Resume xq to continue task execution
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2026.png)

**异构XPUs的抢占**

调度模型和XPU之间插入**硬件模型层**，**面向XPU模型进行调度**，**不直接面对不同XPU的个性化接口**：XPU+软件栈支持硬件模型定义的接口，调度模型利用硬件模型提供的接口进行调度。

不同硬件抢占能力不同，调度假设单一的硬件抢占能力（模型）不合适：假设统一低的抢占能力，会损失高抢占能力硬件的性能；假设统一高的抢占能力，会不适配能力不足的硬件。

**分层硬件抽象**，不同抢占能力的硬件划分到不同抽象模型：GPU可编程因此调度能力最强（timeslice，priority，线程blk or instr level Preemption），NPU部分可编程，ASIC和FPGA几乎不可编程（没有合适的接口）。

> **[图片提取文字 (image.png)]:**
> ## XQueue: Preemptible XPU Command Queue
> 
> ![](_page_0_Figure_1.jpeg)
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2027.png)

> **[图片提取文字 (image.png)]:**
> ## Implementing XQueue on Diverse XPUs
> 
> ![](_page_0_Figure_1.jpeg)
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2028.png)

> **[图片提取文字 (image.png)]:**
> ## Dilemma: Compatibility or Performance?
> 
> ![](_page_0_Figure_1.jpeg)
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2029.png)

> **[图片提取文字 (image.png)]:**
> ## Our Solution: Multi-level Hardware Model
> 
> ![](_page_0_Figure_1.jpeg)
> 
> - Categorize preemption-related capabilities into three incremental levels (LvI—Lv3)
> - ➤ Higher levels posses all capabilities of lower ones and introduce advanced capabilities
> - ➤ The lowest level (LvI) is supported by ALL XPUs
> - ➤ The highest level (Lv3) unlocks full potentials of advanced XPUs
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2030.png)

hwQueue（待提交pending > 已提交in-flight > 执行中running）、XPU的launch+sync接口

> **[图片提取文字 (image.png)]:**
> ## Basic Assumption: hwQueue
> 
> ![](_page_0_Figure_1.jpeg)
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2031.png)

> **[图片提取文字 (image.png)]:**
> ## From the Perspective of Command States
> 
> ![](_page_0_Picture_1.jpeg)
> 
> - Pending commandsCreated and waiting on the host side
> - ➤ In-flight commands
> 
>   Launched to a hwQueue, escapes host control
> - Running commandsBeing executed on the XPU
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2032.png)

**lv1抢占**：抢占pending state的命令，即进程停止提交当前**XQueue的下一条**指令，XPU都有该接口。

> **[图片提取文字 (image.png)]:**
> ## Level-I: Preempt Pending Command
> 
> ![](_page_0_Figure_1.jpeg)
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2033.png)

> **[图片提取文字 (image.png)]:**
> ## **4.1** Level 1: Pending Command Preemption Level 1 (Lv1) preemption targets commands in the pending
> 
> state before launch. Once a command is launched, it is enqueued to the hardware queue (hwQueue), becoming in-flight and escaping host control. Therefore, the host can preempt pending commands by simply blocking their launch. Lyd only
> 
> pending commands by simply blocking their launch. Lv1 only requires capabilities to **launch** and **synchronize** commands, which all XPUs provide. The preemption latency is the total execution time of all launched commands, as shown in Fig. 3.
![image.png](%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%8A%A2%E5%8D%A0Preemption%EF%BC%88%E5%BC%82%E6%9E%84XPUs%E3%80%81Firmware%E3%80%81%CE%BC-controller%EF%BC%89/image.png)

**lv2抢占**：抢占in-flight state的命令，即**XPU μ-controller支持停止派发指令**，类似GPU的线程块level Preemption，停止SM的dispatch Unit派发指令。

stall-based deactive让**μ-controller停止当前queue的命令派发，选择派发其他queue命令。（GPU、Ascend、iGPU）**

flush-based deactive让**μ-controller清空当前queue的命令，r**eactive**时重新填充queue。（GPU的钩子kernel）**

> **[图片提取文字 (image.png)]:**
> ## Level-2: Preempt In-flight Command
> 
> ![](_page_0_Figure_1.jpeg)
> 
> - Deactivate an hwQueue so that its commands will not be executed further
> - Significantly reduced preemption latency
> - Approach I: instruct the  $\mu$ -controllers in modern XPUs, e.g., Intel NPU, NV & AMD GPU to stall command dispatching
> - Approach 2: leverage command programmability to abort in-flight commands
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2034.png)

> **[图片提取文字 (image.png)]:**
> running command, as shown in Fig. 3. The key capability is to deactivate and reactivate the hwQueue. Once deactivated, no new commands from this hwQueue will execute until reactivation, enabling Lv2 preemption. This deactivation can be implemented by stalling-based or flushing-based approaches. Stalling-based deactivation. The hwQueue can be deactivated by stalling command dequeuing, which prevents new commands from being fetched for execution. This approach requires advanced XPUs with integrated microcon-
![image.png](%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%8A%A2%E5%8D%A0Preemption%EF%BC%88%E5%BC%82%E6%9E%84XPUs%E3%80%81Firmware%E3%80%81%CE%BC-controller%EF%BC%89/image%201.png)

> **[图片提取文字 (image.png)]:**
> mands based on their attributes. The host can either instruct the microcontroller directly or modify command attributes to control dequeuing, achieving deactivation and reactivation. **Flushing-based deactivation.** Another approach is to flush all in-flight commands in the hwQueue and relaunch them upon reactivation. Prior work [19, 39, 129] proposes that commands could be retrofitted to flush themselves, which solely requires the commands to be programmable. This approach works on all programmable XPUs, e.g., GPUs and many NPUs [43].
> 
> trollers [30, 44, 46, 55, 72] that can selectively dequeue com-
![image.png](%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%8A%A2%E5%8D%A0Preemption%EF%BC%88%E5%BC%82%E6%9E%84XPUs%E3%80%81Firmware%E3%80%81%CE%BC-controller%EF%BC%89/image%202.png)

**lv3抢占**：抢占running state的命令，保存或恢复上下文（GPU的指令level Preemption）。

[GPU的抢占Preemption和Timeslice（WFI/drain、switch）](GPU%E7%9A%84%E6%8A%A2%E5%8D%A0Preemption%E5%92%8CTimeslice%EF%BC%88WFI%20drain%E3%80%81switch%EF%BC%89%20262e12d10b6e8048b20cf29d04938602.md)

[https://docs.nvidia.com/cuda/pascal-tuning-guide/index.html#compute-preemption](https://docs.nvidia.com/cuda/pascal-tuning-guide/index.html#compute-preemption)

> **[图片提取文字 (image.png)]:**
> ## 1.4.7. Compute Preemption
> 
> Compute Preemption is a new feature specific to GP100. Compute Preemption allows compute tasks running on the GPU to be interrupted at nstruction-level granularity. The execution context (registers, shared memory, etc.) are swapped to GPU DRAM so that another application can be
> 
> - swapped in and run. Compute preemption offers two key advantages for developers:
> 
>   > Long-running kernels no longer need to be broken up into small timeslices to avoid an unresponsive graphical user interface or kernel timeouts
> - when a GPU is used simultaneously for compute and graphics.
> - Interactive kernel debugging on a single-GPU system is now possible.
![image.png](%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%8A%A2%E5%8D%A0Preemption%EF%BC%88%E5%BC%82%E6%9E%84XPUs%E3%80%81Firmware%E3%80%81%CE%BC-controller%EF%BC%89/image%203.png)

> **[图片提取文字 (image.png)]:**
> ## Level-3: Preempt Running Command
> 
> ![](_page_0_Figure_1.jpeg)
> 
> - **Interrupt** the running command
> - Ultra-low & stable preemption latency
> - Supported in modern GPUs
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2035.png)

> **[图片提取文字 (image.png)]:**
> ## 4.3 Level 3: Running Command PreemptionLv2 preemption still requires waiting for the running com-
> 
> tion latency and may not meet strict real-time requirements for applications like automation [2, 51, 134] and networking [33, 59]. In contrast, Level 3 (Lv3) preemption targets the running command, aiming to achieve ultra-low and stable preemption latency, as illustrated in Fig. 3. This requires hardware capabilities that can **interrupt** and **restore** the run-
> 
> mand to complete, which leads to unpredictable preemp-
> 
> ning command. Once interrupted by the host, the running command is instantly paused and preempted for the execution of another command. The interrupted command is later restored to continue its execution. These capabilities are already present in modern GPUs [39, 44, 79].
![image.png](%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%8A%A2%E5%8D%A0Preemption%EF%BC%88%E5%BC%82%E6%9E%84XPUs%E3%80%81Firmware%E3%80%81%CE%BC-controller%EF%BC%89/image%204.png)

### XSched Lib、Xscheduler**实现**

**运行时库**：**XShim**为应用提供API，**XPreempt**实现XQueue接口，**XAL**实现分层硬件模型接口。

**调度器**：后台进程XScheduler通过Agent报告的XQueue行为，监控运行时状态，按policy决策调度。

> **[图片提取文字 (image.png)]:**
> ## XSched: A Preemptive XPU Scheduling Framework
> 
> based-on the XQueue Abstraction and the Multi-level Hardware Model
> 
> ![](_page_0_Figure_2.jpeg)
> 
> tasks [3]). This may lead to unfairness and priority inver-
> 
> sions [60, 61, 100, 103, 113], which manifest as service-level
> 
> objective (SLO) violations in data centers or missed deadlines
> 
> in latency-critical autonomous systems. For example, in a
> 
> video conferencing application (see §8), the tail latency of a
> 
> with flexible policies. XSched provides unified interfaces
> 
> for scheduling XPU tasks through a preemptible command
> 
> queue abstraction (XQueue). The key challenge in imple-
> 
> menting the abstraction is adapting to XPUs with diverse and
> 
> evolving hardware capabilities and software stacks. XSched
> 
> 27
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2036.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Fig. 4: Architecture and workflow of XSched.
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2037.png)

**XPreempt的设计**

host提交命令比XPU执行命令快很多，因此host等待XPU完成的**性能浪费大**，如果host提交过多命令到XPU（in-flight），**Lv1的XPU抢占延迟会很高**。因此host XQueue设计**可配置in-flight threshold**，当host提交命令达到threshold时，调用sync()来等待XPU完成或被抢占。

> **[图片提取文字 (image.png)]:**
> mand launching mechanism to balance preemption latency and runtime overhead, as shown in Fig. 6. This approach maintains a small number of in-flight commands, preventing pipeline stalls while enabling fine-grained preemption. When suspending the XQueue, the scheduler waits only for these few in-flight commands to complete, rather than all submitted commands (usually hundreds of commands).
> 
> Each XQueue contains a worker thread, a pending command buffer an in-flight command log, and a corresponding
> 
> Our solution: progressive command launching. Inspired by
> 
> our prior work [39], XPreempt introduces a progressive com-
> 
> mand buffer, an in-flight command log, and a corresponding hwQueue. When a command is submitted to the XQueue, it is first pushed into the pending command buffer (Line 1). The worker thread then *progressively* launches it to the hwQueue and records these in-flight commands in the log. Specifically, the worker checks the count of in-flight commands against a user-defined threshold (Line 12). If the count exceeds this
![image.png](%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%8A%A2%E5%8D%A0Preemption%EF%BC%88%E5%BC%82%E6%9E%84XPUs%E3%80%81Firmware%E3%80%81%CE%BC-controller%EF%BC%89/image%205.png)

> **[图片提取文字 (image.png)]:**
> the worker pauses to block new command launches (Lines 17–18). Once the XQueue is resumed, the worker continues to launch pending commands (Lines 19–20). Lv2 and Lv3 preemption. For XPUs supporting Lv2 interfaces, XPreempt additionally deactivates the hwQueue after the worker is paused to accelerate task preemption when suspending the XQueue. Also, the hwQueue is reactivated when resuming. For XPUs supporting Lv3 interfaces, XPreempt further interrupts the currently running command in the hwQueue to preempt the task instantly when suspending the XQueue, and restores the command when resuming.
> 
> The design of XPreempt is compatible with all three levels
> 
> and can fully leverage the hardware capabilities thanks to the
> 
> multi-level hardware model implemented in the XAL.
> 
> threshold, the worker invokes sync to wait for half of in-
> 
> flight commands to complete (Lines 13–15). This threshold
> 
> can be adjusted based on workloads to trade off preemption la-
> 
> tency and runtime overhead. When the XQueue is suspended,
![image.png](%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%8A%A2%E5%8D%A0Preemption%EF%BC%88%E5%BC%82%E6%9E%84XPUs%E3%80%81Firmware%E3%80%81%CE%BC-controller%EF%BC%89/image%206.png)

> **[图片提取文字 (image.png)]:**
> ```
> # XQueue APIs
> void submit(xq, cmd)
>    push pending cmd(xq, cmd)
> 1
> void wait(xq, cmd)
>     if find_pending_cmd(xq, cmd) then
> 2
> 3
>            # wait to launch cmd
>     sync(xq.hwq, cmd) # sync for completion
> 4
> void suspend(xq)
> 5
>     xq.mode = SUSPENDED # suspend XQueue
> 6
>     if LEVEL >= 2 then deactivate(xq.hwq)
>     if LEVEL == 3 then interrupt(xq.hwq)
> 7
> void resume(xq)
>     xq.mode = RUNNING # resume XQueue
> 8
> 9
>     if LEVEL >= 2 then reactivate(xq.hwq)
>     if LEVEL == 3 then restore(xq.hwq)
> 10
> # progressive command launching
> void worker_thread(xq)
>     while TRUE
> 11
>       if get_inflight_cnt(xq) >= THRESHOLD then
> 12
>         # wait until half of launched cmds complete
>         mid = get_inflight_middle(xq)
> 13
> 14
>         sync(xq.hwq, mid) # sync for completion
> 15
>         pop completed cmds(xq)
> 16
>       cmd = pop pending cmd(xq)
> 17
>       while xq.mode == SUSPENDED
> 18
>         pause() # pause cmd launching
> 19
>       launch(xq.hwq, cmd) # launch cmd to hwqueue
> 20
>       push_inflight_cmd(cmd)
> ```
> 
> **Fig. 6:** *Pseudocode of* XQueue *APIs and progressive command launching implementation in* XPreempt.
![image.png](%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%8A%A2%E5%8D%A0Preemption%EF%BC%88%E5%BC%82%E6%9E%84XPUs%E3%80%81Firmware%E3%80%81%CE%BC-controller%EF%BC%89/image%207.png)

**XScheduler的设计**

Xscheduler是event驱动的调度器，根据agent发送信息，维持XQueue状态，根据policy做决策（抢占或者恢复），policy可用户配置。

> **[图片提取文字 (image.png)]:**
> ## 5.3 XScheduler Design
> 
> application process.
> 
> As depicted in Fig. 4, the XScheduler daemon is an event-driven service that coordinates all XQueues from different processes in the system, in cooperation with the XPreempt agents. Each agent monitors the change events of XQueue states, e.g., become *ready* when new commands are submitted
> 
> and idle when all commands are completed. These events are
> 
> sent to the XScheduler daemon to maintain global XQueue
> 
> status (each XQueue: ready or idle, XPU device ID, process ID, etc.) and trigger the policy upon status changes. The policy decides which XQueues to suspend or resume based on current status. The decisions are sent back to the agents and applied by calling suspend and resume interfaces of these XQueues. Messages between XScheduler and the agents are passed via shared-memory IPC, enabling XSched to schedule XQueues across both processes and containers. For cross-VM scheduling, message passing can alternatively be implemented over network. This distributed XQueue design separates the control and data planes, minimizing com-
> 
> mand submission overhead and isolating errors within the
![image.png](%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%8A%A2%E5%8D%A0Preemption%EF%BC%88%E5%BC%82%E6%9E%84XPUs%E3%80%81Firmware%E3%80%81%CE%BC-controller%EF%BC%89/image%208.png)

> **[图片提取文字 (image.png)]:**
> tomizable. XSched provides a send\_hint API for applications and a command-line tool (XCLI) for users to provide hints to the policy, which set policy-specific parameters (e.g., priority, bandwidth, and deadline) of an XQueue. This is similar to how setpriority syscall and nice command set the priority in Linux. In addition to several built-in policies (e.g., fixed priority, bandwidth partition), users are free to customize policies optimized for their application scenarios. The policy should implement two basic interfaces: schedule and recv\_hint, which are called when XQueue status changes and a new hint is given, respectively. As mentioned in Fig. 2, schedule uses suspend and resume to schedule XQueues, and add\_timer to trigger itself after an interval.
> 
> The policy in XSched is designed to be flexible and cus-
![image.png](%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%8A%A2%E5%8D%A0Preemption%EF%BC%88%E5%BC%82%E6%9E%84XPUs%E3%80%81Firmware%E3%80%81%CE%BC-controller%EF%BC%89/image%209.png)

**XAL的实现（XPU相关）**

**LV1 XAL：抢占host未提交的XQueue，支持所有XPU。**

所有XPU的driver API会提供launch和sync接口（完成后通知）。

> **[图片提取文字 (image.png)]:**
> ## 6.1 Level 1 (Lv1) PreemptionImplementing Lv1 preemption is straightforward on XPUs
> 
> CUDA [93] (for NVIDIA GPUs), hipStream in HIP [5] (for AMD GPUs), ze\_command\_queue in LevelZero [47] (for Intel GPUs and NPUs), aclrtStream in ACL [43] (for Ascend NPUs), VPIStream in VPI [91] (for vision process-
> 
> ing ASICs), and cl\_command\_queue in OpenCL [62] (for
> 
> Xilinx FPGAs). The launch interface is implemented by
> 
> calling the appropriate driver API corresponding to the com-
> 
> since their drivers natively support hwQueue to launch and
> 
> synchronize commands. Examples include CUstream in
> 
> mand type, e.g., cuLaunchKernel for launching kernels on a CUstream and cuMemcpyAsync for memory copy commands. These drivers also support events, e.g., CUevent in CUDA and cl\_event in OpenCL, which are fine-grained synchronization points that can be recorded on the hwQueue. The sync interface is implemented by synchronizing with an extra event recorded after a given command. If the driver
> 
> does not support events, it can alternatively be implemented by synchronizing with the hwQueue. Since Lv1 implementation relies only on basic driver APIs, it can be shared across XPUs that use the same software platform. For example, the OpenCL implementation supports GPUs, FPGAs and even CPUs from various manufacturers.
![image.png](%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%8A%A2%E5%8D%A0Preemption%EF%BC%88%E5%BC%82%E6%9E%84XPUs%E3%80%81Firmware%E3%80%81%CE%BC-controller%EF%BC%89/image%2010.png)

> **[图片提取文字 (image.png)]:**
> ## Supported & Evaluated XPUs
> 
> ![](_page_0_Picture_1.jpeg)
> 
> ![](_page_0_Picture_2.jpeg)
> 
> ![](_page_0_Picture_3.jpeg)
> 
> NV K40m GPU
> 
> ![](_page_0_Picture_5.jpeg)
> 
> AMD MI50 GPU
> 
> ![](_page_0_Picture_7.jpeg)
> 
> Ascend 910b NPU
> 
> ![](_page_0_Picture_9.jpeg)
> 
> Xilinx VU9P FPGA \*integrated with AWS-F1
> 
> ![](_page_0_Picture_11.jpeg)
> 
> ![](_page_0_Picture_12.jpeg)
> 
> NV DLA (NPU) + OFA (ASIC) + PVA (ASIC)
> \*integrated with Jetson AGX Orin
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2039.png)

> **[图片提取文字 (image.png)]:**
> ## Level-I: General Design for Effortless Integration
> 
> ![](_page_0_Picture_1.jpeg)
> 
> ![](_page_0_Picture_2.jpeg)
> 
> ![](_page_0_Picture_3.jpeg)
> 
> ![](_page_0_Picture_4.jpeg)
> 
> NV GV100 GPU
> 
> AMD MI50 GPU
> 
> Ascend 910b NPU
> 
> Xilinx VU9P FPGA \*integrated with AWS-F1
> 
> ![](_page_0_Picture_9.jpeg)
> 
> ![](_page_0_Picture_10.jpeg)
> 
> ![](_page_0_Figure_11.jpeg)
> 
> ![](_page_0_Figure_12.jpeg)
> 
> ![](_page_0_Figure_13.jpeg)
> 
> NV DLA (NPU) + OFA (ASIC) + PVA (ASIC) \*integrated with Jetson AGX Orin
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2040.png)

**LV2 XAL：抢占in-flight但未执行的命令，支持GPU和Intel NPU。**

Stall-based基于**XPU的μ-controller**提供的硬件接口。

Flush-based基于**动态二进制插桩DBI**，插入**Guardian kernel**，让in-flight kernel执行时先执行Guardian来检测是否被抢占（Xsche设置flag），如果被抢占，这些kernel被Guardian保存信息，并强制退出（flush）。

DBI方法不需要修改XPU的PTX，而是基于CUDA编程即可。

> **[图片提取文字 (image.png)]:**
> ```
> exit_thread() # abort kernel execution
>  3
> # host CPU code in XAL to implement Level 2 interfaces
> void deactivate(hwq)
>        int *flag ptr = get deactivation flag(hwq)
> 4
>  5
>        *flag ptr = DEACTIVATED # set per-hwqueue flag
> void reactivate(hwq)
>        int *flag_ptr = get_deactivation flag(hwq)
> 6
> 7
>        *flag ptr = NORMAL # clear per-hwqueue flag
>        # relaunch aborted cmds
> 8
>        while cmd = pop_aborted_cmd(hwq)
> 9
>          launch(hwq, cmd)
> Fig. 7: Pseudocode of flushing-based preemption implementation.
> ```
> 
> # GPU code injected at the beginning of kernel binary
> 
> void guardian(flag ptr, hwq id, cmd id)
> 
> if \*flag ptr == DEACTIVATED then
> 
> push aborted cmd(hwq id, cmd id)
> 
> 1
> 
> 2
![image.png](%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%8A%A2%E5%8D%A0Preemption%EF%BC%88%E5%BC%82%E6%9E%84XPUs%E3%80%81Firmware%E3%80%81%CE%BC-controller%EF%BC%89/image%2011.png)

> **[图片提取文字 (image.png)]:**
> ## 6.2 Level 2 (Lv2) Preemption
> 
> XSched implemented hwQueue deactivation and reactivation
> 
> using a hardware-assisted stalling approach on Intel NPUs and a software-based flushing approach on NVIDIA GPUs.
> 
> Stalling-based preemption. On-chip microcontrollers have
> 
> been widely integrated into XPUs to selectively dispatch commands to execution units. Examples include Falcon mi-
> 
> crocontrollers in NVIDIA GPUs [10, 30, 55, 108], Command Processors in AMD GPUs [39], Graphics microcontrollers
> 
> (GuCs) in Intel GPUs [44], LeonRT cores in Intel NPUs [46],
> 
> and Taishan cores in Ascend NPUs [72]. These XPUs can preempt in-flight commands by instructing their microcontrollers to stall command dequeuing for deactivation. Recently, Intel
> 
> released a new NPU firmware that supports these capabilities [48]. We modified the driver [116] to expose them to the
> 
> host and implemented the Lv2 interfaces on Intel NPUs [45]. Flushing-based preemption. For XPUs without microcontroller support or exposed interfaces, we implement a software approach for flushing-based deactivation on programmable XPUs, demonstrating this on NVIDIA GPUs. XSched lever-
> 
> ages command programmability, instead of hardware micro-
![image.png](%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%8A%A2%E5%8D%A0Preemption%EF%BC%88%E5%BC%82%E6%9E%84XPUs%E3%80%81Firmware%E3%80%81%CE%BC-controller%EF%BC%89/image%2012.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Fig. 8: Binary instrumentation for flushing-based preemption.
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2038.png)

> **[图片提取文字 (image.png)]:**
> As shown in Fig. 7, XSched prepends a *guardian* code snippet to each GPU kernel (i.e., command) that checks a perhwQueue deactivation flag in GPU global memory. When this flag is set, the guardian code records the command ID and exits this command immediately. To deactivate the hwQueue, the host sets the flag, prompting all in-flight kernels to abort themselves. When reactivating, the host clears the flag and relaunches the aborted kernels.
> 
> XSched injects the guardian code snippet into each GPU kernel at runtime using dynamic binary instrumentation (DBI)
> 
> controllers, to flush all in-flight commands in the hwQueue.
> 
> technique [118]. This guardian code is compiled to binary using the NVIDIA compiler (NVCC) [83] and loaded into GPU instruction memory at process startup. As shown in Fig. 8, XSched rewrites the first instruction of each kernel with a JMP instruction to a per-kernel helper snippet. This snippet first loads the arguments of the guardian code from GPU constant memory, where kernel arguments are stored, and then calls the guardian code. After that, the snippet exe-
> 
> and then calls the guardian code. After that, the snippet executes the original replaced instruction and returns to the next instruction in the kernel. Note that XSched leverages the hidden functions in the CUDA export table to allocate and access GPU instruction and constant memory.
![image.png](%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%8A%A2%E5%8D%A0Preemption%EF%BC%88%E5%BC%82%E6%9E%84XPUs%E3%80%81Firmware%E3%80%81%CE%BC-controller%EF%BC%89/image%2013.png)

**LV3 XAL：抢占运行中指令，支持NV GPU。**

TSG-based：通过ioctrl设置线程块的**timeslice**，被中断的线程块设置timeslice为0，**GPU自动切换超时的线程块**，恢复的线程块设置正常timeslice，包括**instr-level和Thread Blk(CTA)-level切换**。

Queue-based：调用**ioctrl会中断GPU所有kernel**，**通过DBI在trap handler segment中注入代码**，检测Xsche是否要求抢占，**被Xsche要求抢占的kernel退出执行**，其余kernel中断后恢复执行。

> **[图片提取文字 (image.png)]:**
> (TSGs) for NVIDIA GPUs [80, 123]. Each process is assigned a CUDA context corresponding to a TSG. When a TSG's timeslice expires, the GPU interrupts all running kernels in this TSG and switches to the next TSG in a roundrobin manner. A previous study [123] found that TSGs of NVIDIA Tegra embedded GPUs could be adjusted through driver ioctl for task preemption. We implemented a similar approach on desktop and server GPUs (e.g., GV100), which dynamically adjusts TSG timeslices for Lv3 preemption. Specifically, the interrupt interface sets the times-
> 
> **TSG-based preemption.** The interrupt mechanism on GPUs
> 
> is designed for scheduling processes, or timeslice groups
> 
> empted, and the restore interface resets the timeslice to its original value. Since this interrupt affects the entire TSG, this approach is only capable of inter-process scheduling. Note that the TSG-based approach achieves both Lv2 and Lv3 preemption by preempting both in-flight and running commands.
> 
> lice to zero for the TSG containing the hwQueue to be pre-
![image.png](%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%8A%A2%E5%8D%A0Preemption%EF%BC%88%E5%BC%82%E6%9E%84XPUs%E3%80%81Firmware%E3%80%81%CE%BC-controller%EF%BC%89/image%2014.png)

> **[图片提取文字 (image.png)]:**
> at the hwQueue granularity. Although GPU interrupts are undocumented [75, 111], by tracing CUDA application syscalls, we discovered a specific ioctl that triggers GPU interrupts by writing to a GPU control register—a feature originally intended for kernel debugging. When the GPU is interrupted, it immediately stalls all running kernels and invokes the trap handler to save context and check trap reason. Using the DBI-based guardian technique for flushing-based preemption, XSched extends the trap handler to detect interrupts triggered by XSched. When detected, the target kernel is aborted for preemption, and the rest ones restore the context and continue execution. Unfortunately, NVIDIA GPUs do not expose support for resuming an aborted kernel from its interruption point. Inspired by prior work [39], XSched only preempts idempotent kernels and restarts them from the beginning.<sup>4</sup>
> 
> Queue-based preemption. We devise the first fine-grained
> 
> interrupts on NVIDIA GPUs to implement Lv3 preemption
![image.png](%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%8A%A2%E5%8D%A0Preemption%EF%BC%88%E5%BC%82%E6%9E%84XPUs%E3%80%81Firmware%E3%80%81%CE%BC-controller%EF%BC%89/image%2015.png)

> **[图片提取文字 (image.png)]:**
> ## Level-2 Preemption Support
> 
> ![](_page_0_Picture_1.jpeg)
> 
> Pure software approach by modifying GPU kernel binary
> 
> ![](_page_0_Picture_3.jpeg)
> 
> NV GVI00 GPU
> 
> AMD MI50 GPU
> 
> Ascend 910b NPU
> 
> inx VU9P **FPGA** 
> 
> INVIDIA TESLA
> 
> Hardware-assisted approach by modifying driver to leverage the  $\mu$ -controller
> 
> Intel NPU3720 + Arc iGPU
> 
> \*integrated with Core Ultra 185H
> 
> ![](_page_0_Picture_13.jpeg)
> 
> NV DLA (NPU) + OFA (ASIC) + PVA (ASIC)
> \*integrated with Jetson AGX Orin
> 
> NV K40m GPU
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2041.png)

> **[图片提取文字 (image.png)]:**
> ## Level-3 Preemption Support
> 
> ![](_page_0_Picture_1.jpeg)
> 
> Modify GPU trap handler and proactively trigger GPU interrupts
> 
> ![](_page_0_Picture_3.jpeg)
> 
> NV GV100 GPU
> 
> ![](_page_0_Picture_8.jpeg)
> 
> ![](_page_0_Picture_9.jpeg)
> 
> ![](_page_0_Picture_10.jpeg)
> 
> ![](_page_0_Picture_12.jpeg)
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2042.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%8A%A2%E5%8D%A0Preemption%EF%BC%88%E5%BC%82%E6%9E%84XPUs%E3%80%81Firmware%E3%80%81%CE%BC-controller%EF%BC%89/image%2016.png)

### 实验

开销很低

> **[图片提取文字 (image.png)]:**
> ## Is XSched Effective?
> 
> - Policy: Fixed-priority policy
> - Result: With XSched, latency close to standalone, reduce the P99 latency by up to 2.11 x compared with native
> 
> ## Latency CDF of the high-priority task
> 
> ![](_page_0_Figure_4.jpeg)
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2043.png)

> **[图片提取文字 (image.png)]:**
> ## Is XSched Effective?
> 
> - ➤ Policy: Bandwidth partition policy, partition ratio = 75% : 25%
> - > Result: Guarantee desired throughput ratio with 1.5% overhead
> 
> ## Normalized throughput of the two process
> 
> ![](_page_0_Figure_4.jpeg)
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2044.png)

> **[图片提取文字 (image.png)]:**
> ## Application: Al Video Conference with XSched
> 
> FPS: 24.18
> 
> ig Systems
> 
> **Native** 
> 
> fundamental Al Video Conference on Intel NPU w/ XSched 1000 FPS: 24.04 Frame Latency (ms) 19th USENIX Symposium g Systems Design and Implementation Test preemption is a critical and fundamental
> 
> Al Video Conference on Intel NPU
> 
> 19th USENIX Symposiur
> 
> Design and Implemental
> 
> Test preemption is a critical and
> 
> ![](_page_0_Figure_3.jpeg)
> 
> Frame Latency Monitor
> 
> **XSched** Laxity-based policy
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2045.png)

> **[图片提取文字 (image.png)]:**
> ## Application: Inference Serving with XSched
> 
> - ➤ Integrated with Triton
> - > 1.41 × reduction on P99
> 
> ➤ Comparable with ad-hoc solutions, Paella [SOSP'23]
> 
> Latency CDF of the high-priority requests.
> 
> ![](_page_0_Figure_5.jpeg)
> 
> ![](_page_0_Figure_6.jpeg)
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2046.png)

> **[图片提取文字 (image.png)]:**
> ## Conclusion
> 
> ![](_page_0_Picture_1.jpeg)
> 
> ![](_page_0_Picture_2.jpeg)
> 
> ![](_page_0_Picture_3.jpeg)
> 
> XSched — A preemptive scheduling framework that supports diverse XPUs
> 
> ## Challenges
> 
> - Complex XPU software stacks
> - Diverse and evolving hardware capabilities
> 
> ## Key Idea
> 
> ![](_page_0_Picture_9.jpeg)
> 
> ![](_page_0_Picture_10.jpeg)
> 
> • Multi-level hardware model (optimal scheduling performance and compatibility)
> 
> ![](_page_0_Picture_12.jpeg)
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2047.png)

> **[图片提取文字 (image.png)]:**
> ## **XSched**: Preemptive Scheduling for Diverse XPUs
> 
> Weihang Shen Mingcong Han Jialong Liu Rong Chen Haibo Chen
> 
> IPADS, Shanghai Jiao Tong University
> 
> ![](_page_0_Picture_3.jpeg)
> 
> ![](_page_0_Picture_4.jpeg)
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2048.png)

## Intel & NV μ-controller

ref：**NVIDIA RISC-V Story**

ref：[https://envytools.readthedocs.io/en/latest/hw/falcon/index.html](https://envytools.readthedocs.io/en/latest/hw/falcon/index.html)

ref：**Exploring microcontrollers in GPUs**

ref：[https://docs.kernel.org/gpu/nova/core/falcon.html](https://docs.kernel.org/gpu/nova/core/falcon.html)

ref：**Enabling the GuC/HuC Firmware for Linux* on New Intel GPU Platforms**

ref：[https://riscv.org/blog/how-nvidia-shipped-one-billion-risc-v-cores-in-2024/](https://riscv.org/blog/how-nvidia-shipped-one-billion-risc-v-cores-in-2024/)

### μ-controller for GPU

NV的μ-controller之前通过**Falcon实现**。目前设计**RV-Cores实现μ-controller功能，同时完成虚拟化等高级功能且向后兼容。**

μ-controller专为XPU服务，和XPU同步。易混淆的是host CPU和XPU是异步的，host CPU提交任务给XPU完成。

> **[图片提取文字 (image.png)]:**
> stances (e.g., GSP (the GPU system processor) and SEC2 (the security engine)) and also may integrate a RISC-V core. This core is capable of running both RISC-V and Falcon code.
> 
> The code running on the Falcon cores is also called 'ucode', and will be referred to as such in the following sections.
> 
> NVIDIA GPUs embed small RISC-like microcontrollers called Falcon cores, which handle secure firmware
> 
> tasks, initialization, and power management. Modern NVIDIA GPUs may have multiple such Falcon in-
> 
> Falcons have separate instruction and data memories (IMEM/DMEM) and provide a small DMA engine (via the FBIF - "Frame Buffer Interface") to load code from system memory. The nova-core driver must reset and configure the Falcon, load its firmware via DMA, and start its CPU.
![image.png](%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%8A%A2%E5%8D%A0Preemption%EF%BC%88%E5%BC%82%E6%9E%84XPUs%E3%80%81Firmware%E3%80%81%CE%BC-controller%EF%BC%89/image%2017.png)

> **[图片提取文字 (image.png)]:**
> ## **NVIDIA Falcon overview**
> 
> Falcon = FAst Logic CONtroller
> 
> Introduced over 10 years ago, and used in >15 different hardware engines today
> 
> Design for flexibility
> 
> Design for long memory latency
> 
> Design for low area
> 
> Design for security
> 
> ![](_page_0_Figure_7.jpeg)
![image.png](%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%8A%A2%E5%8D%A0Preemption%EF%BC%88%E5%BC%82%E6%9E%84XPUs%E3%80%81Firmware%E3%80%81%CE%BC-controller%EF%BC%89/image%2018.png)

> **[图片提取文字 (image.png)]:**
> ## Why Falcon Next Gen?
> 
> - New use cases requiring more horsepower & feature
>   - Wide addressing range
>   - More performance
>   - Not limit to code size
>   - Rich OS support
> - Falcon has limits
>   - Small addressing range
>   - Poor performance (0.67DMIPS/Mhz, 1.4Coremark/Mhz)
>   - No D\$
>   - No rich OS support
![image.png](%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%8A%A2%E5%8D%A0Preemption%EF%BC%88%E5%BC%82%E6%9E%84XPUs%E3%80%81Firmware%E3%80%81%CE%BC-controller%EF%BC%89/image%2019.png)

> **[图片提取文字 (image.png)]:**
> from G98. Originally developed as the controlling logic for VP3 video decoding engines as a replacement for xtensa used on VP2, it was later used in many other places, whenever a microprocessor of some sort was needed.
> 
> falcon is a class of general-purpose microprocessor units, used in multiple instances on nvidia GPUs starting
> 
> ## A single falcon unit is made of:
> 
> - the core microprocessor with its code and data SRAM [see Processor control]
>   - an IO space containing control registers of all subunits, accessible from the host as well as from the code running on the falcon microprocessor [see IO space]
>     - common support logic:
>       interrupt controller [see Interrupt delivery]
>       - o interrupt controller [see interrupt delivery
>       - o periodic and watchdog timers [see Timers]
>       - scratch registers for communication with host [see Scratch registers]
>         PCOUNTER signal output [see Performance monitoring signals]
>       - PCOUNTER Signal output [s
>         some unknown other stuff
>   - optionally, FIFO interface logic, for falcon units used as PFIFO engines and some others [see FIFO interface]
>   - optionally, common memory interface logic [see Memory interface]. However, some engines have their own type of memory interface.
>   - optionally, a cryptographic AES coprocessor. A falcon unit with such coprocessor is called a "secret ful" unit. [see Cryptographic coprocessor]
>     - any unit-specific logic the microprocessor is supposed to control
![image.png](%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%8A%A2%E5%8D%A0Preemption%EF%BC%88%E5%BC%82%E6%9E%84XPUs%E3%80%81Firmware%E3%80%81%CE%BC-controller%EF%BC%89/image%2020.png)

> **[图片提取文字 (image.png)]:**
> ## Falcon Next Gen with RISCV
> 
> - RISCV plugged-in as 2<sup>nd</sup> core
>   - Back compatibility on interface, easy to integrate
>   - Isolation between security and non-security applications
> 
> ![](_page_0_Figure_4.jpeg)
![image.png](%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%8A%A2%E5%8D%A0Preemption%EF%BC%88%E5%BC%82%E6%9E%84XPUs%E3%80%81Firmware%E3%80%81%CE%BC-controller%EF%BC%89/image%2021.png)

**GPC、Hub是GPU中的μ-controller**。

> **[图片提取文字 (image.png)]:**
> Table 1 illustrates the details of the GF100 microcontroller. There are two types of microcontrollers called HUB and GPC. HUB is broadcasting the access to all GPC's, while the GPC represents a specific microcontroller for each GPC engine. Since the maximum code size is limited to 16KB, developers should carefully design firmware code.
![image.png](%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%8A%A2%E5%8D%A0Preemption%EF%BC%88%E5%BC%82%E6%9E%84XPUs%E3%80%81Firmware%E3%80%81%CE%BC-controller%EF%BC%89/image%2022.png)

> **[图片提取文字 (image.png)]:**
> Table 1: Details of the GF100 microcontroller.
> 
> |                 | HUB | GPC |
> |-----------------|-----|-----|
> | Number of units | 1   | 4   |
> 
> 32 bits
> 
> 16,384 bytes
> 
> 4,096 bytes
> 
> 32 bits
> 
> 8,192 bytes
> 
> 2,048 bytes
> 
> Addressing
> 
> Code section
> 
> Data section
![image.png](%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%8A%A2%E5%8D%A0Preemption%EF%BC%88%E5%BC%82%E6%9E%84XPUs%E3%80%81Firmware%E3%80%81%CE%BC-controller%EF%BC%89/image%2023.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 2: Overview of compiler implementation.
![image.png](%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%8A%A2%E5%8D%A0Preemption%EF%BC%88%E5%BC%82%E6%9E%84XPUs%E3%80%81Firmware%E3%80%81%CE%BC-controller%EF%BC%89/image%2024.png)

**GPC和Hub执行的firmware code**

> **[图片提取文字 (image.png)]:**
> flow. (1) initialize The firmware configures the interrupt handler, and
> 
> receives the default set of data when started.
> 
> cution. The following are the details of each block in the
> 
> Figure 4 shows the basic control flow of firmware exe-
> 
> ## (2) sleep
> 
> The firmware enters the standby mode in the main event loop, waiting for the next command issued by the device driver or the debugging tool. Upon every arrival of the command, an interrupt is generated on the microcontroller, awakening the firmware in the
> 
> "ihbody" function.
> 
> (3) ihbody This is an interrupt handler invoked by the command. All we have to do here is to enqueue the corresponding command, and releases the standby mode to resume firmware execution.
> 
> (4) work This is a main body of the firmware. It is called when the firmware is released from the standby
> 
> mode. The basic procedure of this function is to dequeue a pending command one by one, and call the function corresponding to the command. If the
> 
> specified flag is cleared, we destroy the firmware.
![image.png](%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%8A%A2%E5%8D%A0Preemption%EF%BC%88%E5%BC%82%E6%9E%84XPUs%E3%80%81Firmware%E3%80%81%CE%BC-controller%EF%BC%89/image%2025.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 4: Flowchart of firmware execution.
![image.png](%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%8A%A2%E5%8D%A0Preemption%EF%BC%88%E5%BC%82%E6%9E%84XPUs%E3%80%81Firmware%E3%80%81%CE%BC-controller%EF%BC%89/image%2026.png)

**Intel iGPU Firmware**

Intel的**iGPU和CPU集成在一个SoC中**，**CPU兼任μ-controller的职能**来执行GPU的**Firmware**。**Firmware**调度GPU硬件，提高硬件性能和降低功耗，引入新特性。

> **[图片提取文字 (image.png)]:**
> ## 1.2 Background
> 
> The new generations of the Intel graphics hardware use firmware that have power, performance benefits, and functionalities, such as scheduling and media offloading. Some advanced GPU features (e.g., low power/EU- less than H.264 encoding in Gen9 and higher GPU platforms) cannot be achieved if the GuC and HuC lack support.
> 
> and higher GPU platforms) cannot be achieved if the GuC and HuC lack support.
> 
> It is important for users to understand how to enable and check the firmware status before using it. The 9<sup>th</sup> platform of the Intel® Core™ i5-6600K Processor will be used in this paper to enable the GuC HuC on an Ubuntu\* 16.04 system.
![image.png](%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%8A%A2%E5%8D%A0Preemption%EF%BC%88%E5%BC%82%E6%9E%84XPUs%E3%80%81Firmware%E3%80%81%CE%BC-controller%EF%BC%89/image%2027.png)

**GuC（GPC-used Code？）**负责在不同并行图形引擎上，调度图形负载。

**HuC（Hub-used Code？）**负责将CPU的媒体功能卸载到GPU，减少CPU和GPU同步的等待开销。

> **[图片提取文字 (image.png)]:**
> ## 2.1.1 What is GuC/HuC
> 
> GuC is designed to perform graphics workload scheduling on the various graphics parallel engines. In this scheduling model, the host software submits work through one of the 256 graphics doorbells. This invokes the scheduling operation on the appropriate graphics engine. Scheduling operations include determining which workload to run next, submitting a workload to a command streamer, pre-empting existing workloads running on an engine, monitoring progress, and notifying the host software when work is done.
> 
> HuC is designed to offload some of the media functions from the CPU to GPU. These include bitrate control and header parsing. For example, in the case of bitrate control, the driver invokes the HuC in the beginning of each frame encoding pass. The encode bitrate is adjusted by the calculation from HuC. Both the HuC hardware and the encode hardcode reside in the GPU. Using the HuC will save unnecessary CPU-GPU synchronization.
![image.png](%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%8A%A2%E5%8D%A0Preemption%EF%BC%88%E5%BC%82%E6%9E%84XPUs%E3%80%81Firmware%E3%80%81%CE%BC-controller%EF%BC%89/image%2028.png)

### NV RV-Cores扩展μ-controller

NV的RV Cores支撑：**功能控制**（图形功能、内存、GPU间传输、上下文切换）、**芯片/系统控制**（资源管理、功耗、安全）、**数据处理**（收发网络数据包）。

> **[图片提取文字 (image.png)]:**
> house and, although important, tends to be related to the non-customer-facing aspect of their products These can be divided into three key areas where RISC-V plays a significant part in the NVIDIA product portfolio:
> 
> 1. Function Level Controllers including Video Codecs, Displays, Cameras, Memory Controllers (training), Chip2Chip
> 
> Nevertheless, NVIDIA is not often associated with RISC-V and this is probably because much of its work is done in-
> 
> - Interfaces, and Context-switch
> - 2. Chip/System Level Control including resource management, power management, and security
> 
> GPU)
> 
> 3. Data Processing including packet routing in networking and activation and other DL network layers in DLA (not
![image.png](%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%8A%A2%E5%8D%A0Preemption%EF%BC%88%E5%BC%82%E6%9E%84XPUs%E3%80%81Firmware%E3%80%81%CE%BC-controller%EF%BC%89/image%2029.png)

**RV Cores**替代支撑原有简单功能的Falcon。

> **[图片提取文字 (image.png)]:**
> ## NVIDIA's RISC-V Cores
> 
> NVIDIA's transition to RISC-V from the 32-bit only Falcon core was initially driven by a need for 64-bit capability. Their first RISC-V development was a fairly ordinary dual issue out-of-order RISC-V core with standard extensions that is deployable as a multi-processor version. Subsequently, they added a 32-bit version for area-constrained applications plus a vector processor with a 1024-bit vector unit.
![image.png](%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%8A%A2%E5%8D%A0Preemption%EF%BC%88%E5%BC%82%E6%9E%84XPUs%E3%80%81Firmware%E3%80%81%CE%BC-controller%EF%BC%89/image%2030.png)

> **[图片提取文字 (image.png)]:**
> ## **NVIDIA RISC-V cores**
> 
> | NV-RISCV32                                         | NV-RISCV64                                                                         | NV-RVV                                                       |
> |----------------------------------------------------|------------------------------------------------------------------------------------|--------------------------------------------------------------|
> | RV32I-MU<br>Multiplication<br>Compression<br>Float | RV64I-MSU<br>Multiplication<br>Compression<br>Float<br>Bit manipulation<br>Atomics | RV32I-MU<br>Multiplication<br>Compression<br>Float<br>Vector |
> | In Order<br>Single Issue<br>1.8 CM/MHz<br>1.8 GHz  | Out of Order<br>Dual Issue<br>5 CM/MHz<br>2 GHz<br>SMP                             | NV-RISCV32 +<br>vector extension<br>(1024-bit)               |
> 
> | Examples of (20+) NVIDIA custom extensions |                           |  |
> |--------------------------------------------|---------------------------|--|
> | functional                                 | 64-bit PA/VA extension    |  |
> |                                            | 2KB page sizes            |  |
> | security                                   | Secure debug with ICD     |  |
> |                                            | ROM memory protection     |  |
> | performance                                | Cache operation extension |  |
> 
> ![](_page_0_Figure_3.jpeg)
![image.png](%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%8A%A2%E5%8D%A0Preemption%EF%BC%88%E5%BC%82%E6%9E%84XPUs%E3%80%81Firmware%E3%80%81%CE%BC-controller%EF%BC%89/image%2031.png)

### NV RV Cores for SoC

扩展**RV Cores**作为μ-controller的用途，设计**RV Cores对上提供屏蔽外设细节的更简单的接口**。

RV Core和常见外设构建的**Peregrine子系统参数化管理**任意组合的外设（普通外设、RV扩展、Cache和TCM大小可配置），通过**模块化策略来最大化软硬件复用。**

> **[图片提取文字 (image.png)]:**
> ## A Subsystem enabled by RISC-V
> 
> NVIDIA's SoCs use their own RISC-V subsystem, named Peregrine. On top of RISC-V cores, it includes other peripherals such as DMA and security IP. Peregrine is of crucial importance for NVIDIA as it allows them to pick and reuse any of the 30+ system control and management applications they want to include without needing specific independent development effort each time. The RISC-V architecture allows NVIDIA the flexibility and modularity to configure the subsystem depending on the requirements. for example, they can choose a 32-bit or 64-bit core followed by the specific extensions required for the workload thus maximizing development reuse and return on investment.
![image.png](%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%8A%A2%E5%8D%A0Preemption%EF%BC%88%E5%BC%82%E6%9E%84XPUs%E3%80%81Firmware%E3%80%81%CE%BC-controller%EF%BC%89/image%2032.png)

> **[图片提取文字 (image.png)]:**
> ## Peregrine subsystem
> 
> ## Reuse
> 
> - · Peregrine combines RISCV core with common peripherals
> - · Maximizes HW and SW reuse
> 
> ## Building block strategy
> 
> - · Peripherals may be present or not
> - · RISC-V extensions may be present or not
> - · Cache and TCM sizes parameterized
> 
> ![](_page_0_Figure_8.jpeg)
![image.png](%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%8A%A2%E5%8D%A0Preemption%EF%BC%88%E5%BC%82%E6%9E%84XPUs%E3%80%81Firmware%E3%80%81%CE%BC-controller%EF%BC%89/image%2033.png)

基于Peregrine子系统（RV Cores+外设IP）构建统一的软硬件栈，**显著复用boot、OS、分离kernel和应用库**。

> **[图片提取文字 (image.png)]:**
> ## One Core Strategy - Peregrine Ecosystem
> 
> - Unified embedded HW and SW across all NVIDIA products
>   - Eliminates replication in basic primitives (isolation, crypto etc.)
>   - Maximizes SW/HW leverage across NVIDIA
> - Configurability & Extensibility
>   - Configurable architecture, easily adapted to different products, feature and deployments
> - Security
>   - Uniform physical attack mitigations
>   - In-depth offensive security efforts investments
>   - Peregrine/NVIRSCV architecture foundation for GPU SW Security
> 
> ![](_page_0_Figure_10.jpeg)
![image.png](%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%8A%A2%E5%8D%A0Preemption%EF%BC%88%E5%BC%82%E6%9E%84XPUs%E3%80%81Firmware%E3%80%81%CE%BC-controller%EF%BC%89/image%2034.png)

安全功能：分离kernel是底层软硬件栈的虚拟化，类似VMM，也可用于调度。

> **[图片提取文字 (image.png)]:**
> A core component of the Peregrine subsystem is the separation kernel, which can be thought of as a very basic hypervisor system. It divides the system into different pieces that are independent of each other and can be separately verified. This allows the user to run different pieces of software on separate partitions. For example, a safety-compliant application with ASIL-D certification can be run on one partition while another non-certified application can be run independently on another.
![image.png](%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%8A%A2%E5%8D%A0Preemption%EF%BC%88%E5%BC%82%E6%9E%84XPUs%E3%80%81Firmware%E3%80%81%CE%BC-controller%EF%BC%89/image%2035.png)

> **[图片提取文字 (image.png)]:**
> ## Peregrine / NVRISCV Multi-Partition Software Architecture
> 
> ![](_page_0_Picture_1.jpeg)
> 
> ![](_page_0_Figure_2.jpeg)
> 
> ![](_page_0_Figure_3.jpeg)
> 
> - Foundation for mixed-criticality applications on a single core
> - Partitions are isolated execution environments
>   - · Each component can be developed and analyzed independently
> - Fine-grained access control to HW defined by manifest and partition policies
>   - Manifest and policies are signed static configuration sets
> - Core SW formally verified to be free of runtime errors
>   - Written in Ada/SPARK
> 
> ![](_page_0_Figure_11.jpeg)
![image.png](%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%8A%A2%E5%8D%A0Preemption%EF%BC%88%E5%BC%82%E6%9E%84XPUs%E3%80%81Firmware%E3%80%81%CE%BC-controller%EF%BC%89/image%2036.png)

### NV RV Cores for GPU

**GPU的GSP代表GPU和host交互**，而不是host通过kernel driver直接访问GPU内硬件。

GSP承担kernel driver职能、管理部分MMIO、隐藏GPU底层细节，简化host和GPU的接口。

> **[图片提取文字 (image.png)]:**
> Firstly, the GPU system processor (GSP) has created fundamental changes for NVIDIA as to how they approach their software. The GSP is a processor that sits at the top of the GPU where it creates abstractions of what can be done in the GPU. Instead of the host processor and the kernel driver using individual control registers inside the GPU, they
> 
> simply talk to the GSP and it translates those higher-level commands into lower-level control register rates.
![image.png](%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%8A%A2%E5%8D%A0Preemption%EF%BC%88%E5%BC%82%E6%9E%84XPUs%E3%80%81Firmware%E3%80%81%CE%BC-controller%EF%BC%89/image%2037.png)

> **[图片提取文字 (image.png)]:**
> ## **Application: GPU System Processor**
> 
> Hardware Overview
> 
> ![](_page_0_Figure_2.jpeg)
> 
> GPU System Processor (GSP) is an embedded RISCV processor
> 
> - GSP offloads Kernel Driver functions
> - · GPU MIMO exposure to CPU reduced
> - · GSP SW encapsulates low-level GPU details
> 
> Enables PC / GPU in cloud, shared by multiple remote users
![image.png](%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%8A%A2%E5%8D%A0Preemption%EF%BC%88%E5%BC%82%E6%9E%84XPUs%E3%80%81Firmware%E3%80%81%CE%BC-controller%EF%BC%89/image%2038.png)

host的多VM访问GPU，每个VM使用**GSP管理的vGPU**，分离kernel提供vGPU之间的隔离性。

> **[图片提取文字 (image.png)]:**
> The GSP Peregrine has a 64-bit RISC-V processor, available in single-hart and multi-hart versions. Most importantly, GSP has full access to everything in the GPU including the memory and display controllers which need to be very carefully managed in the software. From a software perspective, the user can deploy a host processor that has a kernel driver and multiple quest virtual machines. Guest virtual machines have corresponding vGPU runtime partitions on GSP and the separation kernel ensures those are isolated and do not interfere with each other. <mark>The</mark> resource manager swaps in and out different guests and ensures that allocation is fair. This capability enables specific use cases such as confidential computing where the GPU is handed over to a guest without any impact from the hypervisor. In this case, RISC-V architecture is fundamental to security because of its specific isolation capabilities
> 
> coupled with NVIDIAs own extension properties.
![image.png](%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%8A%A2%E5%8D%A0Preemption%EF%BC%88%E5%BC%82%E6%9E%84XPUs%E3%80%81Firmware%E3%80%81%CE%BC-controller%EF%BC%89/image%2039.png)

### NVIRV Cores for DLA

DLA compiler将layer graph的node变为的kernel code，使用RV ISA是使用RV的编译器将kernel code编译成RV指令。

> **[图片提取文字 (image.png)]:**
> The second NVIDIA RISC-V enabled application described is a Deep Learning Accelerator which forms part of some Al-specific SoCs. This is essentially an inferencing engine that is programmed in graph processing. An example would be an ONNX program that represents a graph of layers processed in a deep learning network. It then uses standard RISC-V compilers that take the kernel code and compile them into an executable. On top of that there is an RVV compiler which turns it into a loadable. It is also possible to combine different kernels into a single kernel for the runtime to achieve the fastest execution.
![image.png](%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%8A%A2%E5%8D%A0Preemption%EF%BC%88%E5%BC%82%E6%9E%84XPUs%E3%80%81Firmware%E3%80%81%CE%BC-controller%EF%BC%89/image%2040.png)

RV32控制，RVV作向量计算，结合Conv等专用张量核组成RV SoC。

> **[图片提取文字 (image.png)]:**
> DLA does not run everything on the RISC-V processor, the main convolutional cores and the matrix multipliers are a separate entity. In the hardware diagram below there are two RISC-V processors one being the control; a simple 32bit unit, and then there is the vector which is the NVRVV, a 1024bit vector unit. There is a convolutional core and in total six hardware engines. As an example, the Rubik is a smart DMA data transformer that moves the data around while the RISC-V RVV vector processor is used for most of the layers that are not Matrix multipliers. In short it is essentially a full ONNX implementation running on the DLA.
![image.png](%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%8A%A2%E5%8D%A0Preemption%EF%BC%88%E5%BC%82%E6%9E%84XPUs%E3%80%81Firmware%E3%80%81%CE%BC-controller%EF%BC%89/image%2041.png)

> **[图片提取文字 (image.png)]:**
> ## **Application: Deep Learning Accelerator**
> 
> ![](_page_0_Figure_1.jpeg)
> 
> ![](_page_0_Figure_2.jpeg)
![image.png](%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%8A%A2%E5%8D%A0Preemption%EF%BC%88%E5%BC%82%E6%9E%84XPUs%E3%80%81Firmware%E3%80%81%CE%BC-controller%EF%BC%89/image%2042.png)

## HW Ascend

ref：**DaVinci: A Scalable Architecture for Neural Network Computing**

ref：**Atlas AI Computing Platform**

### AI Core和多线程并行

Cube、Vector（SP）的AI Core，DSA的多层编译。

> **[图片提取文字 (image.png)]:**
> ## **DaVinci Core**
> 
> ![](_page_0_Figure_1.jpeg)
> 
> - Cube: 4096(16^3) FP16 MACs + 8192 INT8 MACs
> - Vector: 2048bit INT8/FP16/FP32 vector with special functions (activation functions, NMS- Non Minimum Suppression, ROI, SORT)
> - Explicit memory hierarchy design, managed by MTE
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2049.png)

> **[图片提取文字 (image.png)]:**
> ## **Overview of the DSA Developer Stack**
> 
> Level 3 Library (written by novice programmer)
> 
> Level 3 Compiler (mathematical programming model)
> 
> Level 2 Library (written by skilled programmer)
> 
> Level 2 Compiler (parallel/kernel programming model)
> 
> Level 1 Library (written by expert)
> 
> Low Level 1 Compiler (Intrinsic C) (Architecture defined programming)
> 
> TBE LIB
> 
> TVM/XLA
> 
> TBE
> 
> CudaNN/ CuBLAS
> 
> TIK LIB
> 
> Cuda/OpenCL
> 
> TIK
> 
> CCE Lib
> 
> CCE C
> 
> Instruction Set Architecture
> 
> **GPU** 
> 
> NPU
> 
> Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on January 11,2026 at 08:31:15 UTC from IEEE Xplore. Restrictions apply.
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2050.png)

硬件实现多线程并行，共享资源的线程之间串行（可**静态调度**），没有共享资源的线程之间同样静态调度，因此没必要增加硬件开销。

> **[图片提取文字 (image.png)]:**
> ## Challenge 1: How to Enable Parallelism with Single Thread
> 
> ![](_page_0_Figure_1.jpeg)
> 
> - Programmer is comfortable with the sequential code
> - Davinc's C like programming interface (CCE) let programmer to control the parallelism explicitly.
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2051.png)

> **[图片提取文字 (image.png)]:**
> ## Solution with Multi-thread?
> 
> How about support hardware multi-thread feature?
> 
> - The code in each thread is sequential
> - CUBE is a share resource between threads
> - It has hardware cost
> 
> ![](_page_0_Figure_5.jpeg)
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2052.png)

软件编译生成静态多线程序列

> **[图片提取文字 (image.png)]:**
> ## How does it work - TIK
> 
> - Typical sequential Davinci code is a combination of nested FOR loops
> - Software multi-thread can be added to any FOR loop body (iterator kernel).
> 
> ```
> 1 #preload
> 2 mov out to ub(deq scale)
> 3 mov out to ub(l0c offset)
>  4 duplicate(loc offset)
> 5 load2d(weight_matrix)
>  6 #burst leavel
>  7 for(burst level)
>        brc()
>        #pipe level
>        for(pipe_level)
>     mov_out_to_ll(ll_fmi, out_fmi)
>             load3d(l0a u8, l1 fmi, weight matrix)
> 12
>            mmad(l0c s32,
> 13
>        mov_l0c32_to_ub(ub_fp16, l0c_s32)
>        vconv(ub_u8, ub_fp16)
> 15
>        mov ub to out(out fmo, ub u8)
> 16
> 17
> ```
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2053.png)

> **[图片提取文字 (image.png)]:**
> ## Programmer view of multi-thread
> 
> Kernel2 – Two threads, original M iteration is divided by 2
> 
> ![](_page_0_Figure_2.jpeg)
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2054.png)

task scheduler（**μ-controller**）将指令序列派发给AI Core。

> **[图片提取文字 (image.png)]:**
> ## **Advanced Compiler Techniques**
> 
> - Architecture independent DSL→ C → Binary lowering process
> - Traversal order determines data reuse factor
> - Millions of legitimate mappings
> - Find optimal mapping to
>   - bridge the 2,000x memory bandwidth gap
> 
> ![](_page_0_Picture_6.jpeg)
> 
> ![](_page_0_Figure_7.jpeg)
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2055.png)

> **[图片提取文字 (image.png)]:**
> ## **Putting All This Together**
> 
> ![](_page_0_Figure_1.jpeg)
> 
> - User program AI model using familiar frameworks
> - Extends operator library when necessary
> - The tasks are executed in a single node, or over a network cluster
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2056.png)

### Ascend SoC和3D IC

AP SoC中，或许Little CPU作为NPU和GPU的μ-controller，类似Intel的iGPU。

> **[图片提取文字 (image.png)]:**
> ## Mobile AP SoC
> 
> ![](_page_0_Figure_1.jpeg)
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2057.png)

> **[图片提取文字 (image.png)]:**
> ## **Automotive SoC**
> 
> ![](_page_0_Figure_1.jpeg)
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2058.png)

AI Infer SoC中A55 Task Scheduler（CPU Core）是AI-Core的μ-controller。

> **[图片提取文字 (image.png)]:**
> ## Al Inference SoC
> 
> ![](_page_0_Figure_1.jpeg)
> 
> Chip
> 
> Chip
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2059.png)

> **[图片提取文字 (image.png)]:**
> ## **AI Training SoC**
> 
> ![](_page_0_Figure_1.jpeg)
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2060.png)

> **[图片提取文字 (image.png)]:**
> ## Al Training SoC: Logic + 3DSRAM + 12 HBM
> 
> ![](_page_0_Figure_1.jpeg)
> 
> ![](_page_0_Figure_2.jpeg)
> 
> - Customized HBM2E with two Stacks to increase HBM bandwidth
> - Large 3D-SRAM as AI core cache
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2061.png)

> **[图片提取文字 (image.png)]:**
> ## Mobile AP: LoL + MoL
> 
> ![](_page_0_Picture_1.jpeg)
> 
> ## Step 1
> 
> - One logic die + 3D DRAM
> - 3DM+POP LPDDR
> 
> ![](_page_0_Figure_5.jpeg)
> 
> ## Step 3:
> 
> - Multi-layer 3D DRAM (remove POP LPDDR)
> - Multi-layer Logic die
> 
> ## Step 2
> 
> - Two logic die + 3D DRAM
> - POP LPDDR
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2062.png)

> **[图片提取文字 (image.png)]:**
> ## **Ascend910 Die Shot**
> 
> - Total 8 Dies integrated
>   - Two dummy dies are added to ensure mechanical uniformity
> - Total size:
>   456+168+96x4+110x2=1228mm<sup>2</sup>
> 
> ![](_page_0_Figure_4.jpeg)
> 
> ![](_page_0_Picture_5.jpeg)
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2063.png)

> **[图片提取文字 (image.png)]:**
> ## **Ascend310 Die Shot**
> 
> ![](_page_0_Picture_1.jpeg)
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2064.png)

### Da Vinci Core

> **[图片提取文字 (image.png)]:**
> ## Da Vinci Architecture (Al Core)
> 
> - Main components of the Da Vinci architecture:
>   - Computing unit: It consists of the cube unit, vector unit, and scalar unit.
>   - Storage system: It consists of the on-chip storage unit of the Al core and data channels.
>   - Control unit provides instruction control for the entire computing process. It is equivalent to the command center of the Al core and is
>     responsible for the running of the entire Al core.
> 
> ![](_page_0_Figure_5.jpeg)
![image.png](%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%8A%A2%E5%8D%A0Preemption%EF%BC%88%E5%BC%82%E6%9E%84XPUs%E3%80%81Firmware%E3%80%81%CE%BC-controller%EF%BC%89/image%2043.png)

计算：cube、vector、scalar

> **[图片提取文字 (image.png)]:**
> ## Da Vinci Architecture (Al Core) - Computing Unit
> 
> - Three types of basic computing units: cube, vector, and scalar units, which correspond to matrix, vector and scalar computing modes respectively.
> - Cube computing unit: The matrix computing unit and accumulator are used to perform matrix-related operations. Completes a matrix (4096) of 16x16 multiplied by 16x16 for FP16, or a matrix (8192) of 16x32 multiplied by 32x16 for the INT8 input in a shot.
> - Vector computing unit: Implements computing between vectors and scalars or between vectors. This function
>   covers various basic computing types and many customized computing types, including computing of data
>   types such as FP16, FP32, INT32, and INT8.
> - Scalar computing unit: Equivalent to a micro CPU, the scalar unit controls the running of the entire AI core. It implements loop control and branch judgment for the entire program, and provides the computing of data addresses and related parameters for cubes or vectors as well as basic arithmetic operations.
![image.png](%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%8A%A2%E5%8D%A0Preemption%EF%BC%88%E5%BC%82%E6%9E%84XPUs%E3%80%81Firmware%E3%80%81%CE%BC-controller%EF%BC%89/image%2044.png)

> **[图片提取文字 (image.png)]:**
> ## Da Vinci Architecture (Al Core) - Computing Unit
> 
> ![](_page_0_Figure_1.jpeg)
![image.png](%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%8A%A2%E5%8D%A0Preemption%EF%BC%88%E5%BC%82%E6%9E%84XPUs%E3%80%81Firmware%E3%80%81%CE%BC-controller%EF%BC%89/image%2045.png)

存储：IO buffer、Regs、Convention Unit（**img2col**、unzip、transpose）

> **[图片提取文字 (image.png)]:**
> ## Da Vinci Architecture (Al Core) - Storage System (1)
> 
> - The storage system of the AI core is composed of the storage unit and corresponding data channel.
> - The storage unit consists of the storage control unit, buffer, and registers:
> - Storage control unit: The cache at a lower level than the AI core can be directly accessed through the bus interface. The memory can also be
>   directly accessed through the DDR or HBM. A storage conversion unit is set as a transmission controller of the internal data channel of the AI
>   core to implement read/write management of internal data of the AI core between different buffers. It also completes a series of format
>   conversion operations, such as zero padding, Img2Col, transposing, and decompression.
> - Input buffer: The buffer temporarily stores the data that needs to be frequently used so the data does not need to be read from the AI core
>   through the bus interface each time. This mode reduces the frequency of data access on the bus and the risk of bus congestion, thereby
>   reducing power consumption and improving performance.
> - Output buffer: The buffer stores the intermediate results of computing at each layer in the neural network, so that the data can be easily
>   obtained for next-layer computing. Reading data through the bus involves low bandwidth and long latency, whereas using the output buffer
>   greatly improves the computing efficiency.
> - Register: Various registers in the AI core are mainly used by the scalar unit.
![image.png](%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%8A%A2%E5%8D%A0Preemption%EF%BC%88%E5%BC%82%E6%9E%84XPUs%E3%80%81Firmware%E3%80%81%CE%BC-controller%EF%BC%89/image%2046.png)

> **[图片提取文字 (image.png)]:**
> ## Da Vinci Architecture (Al Core) - Storage System (2)
> 
> - Data channel: path for data flowing in the AI core during execution of computing tasks
>   - A data channel of the Da Vinci architecture is characterized by multiple-input single-output. Considering various types and a large quantity of input data in the computing process on the neural network, parallel inputs can improve data inflow efficiency. On the contrary, only an output feature matrix is generated after multiple types of input data are processed. The data channel with a single output of data reduces the use of chip hardware resources.
> 
> ![](_page_0_Figure_3.jpeg)
![image.png](%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%8A%A2%E5%8D%A0Preemption%EF%BC%88%E5%BC%82%E6%9E%84XPUs%E3%80%81Firmware%E3%80%81%CE%BC-controller%EF%BC%89/image%2047.png)

控制：task块（tensor算子）同步、指令Prefetch、指令译码queue、指令执行queue。

> **[图片提取文字 (image.png)]:**
> ## Da Vinci Architecture (Al Core) - Control Unit (1)
> 
> - The control unit consists of the system control module, instruction cache, scalar instruction processing queue, instruction transmitting module, matrix operation queue, vector operation queue, storage conversion queue, and event synchronization module.
>   - System control module: Controls the execution process of a task block (minimum task computing granularity for the AI core). After the
>     task block is executed, the system control module processes the interruption and reports the status. If an error occurs during the\nexecution, the error status is reported to the task scheduler.
>   - Instruction cache: Prefetches subsequent instructions in advance during instruction execution and reads multiple instructions into the cache at a time, improving instruction execution efficiency.
>   - Scalar instruction procession queue: After being decoded, the instructions are imported into a scalar queue to implement address
>     decoding and operation control. The instructions include matrix computing instructions, vector calculation instructions, and storage
>     conversion instructions.
>   - Instruction transmitting module: Reads the configured instruction addresses and decoded parameters in the scalar instruction queue,
>      and sends them to the corresponding instruction execution queue according to the instruction type. The scalar instructions reside in the scalar instruction processing queue for subsequent execution.
![image.png](%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%8A%A2%E5%8D%A0Preemption%EF%BC%88%E5%BC%82%E6%9E%84XPUs%E3%80%81Firmware%E3%80%81%CE%BC-controller%EF%BC%89/image%2048.png)

> **[图片提取文字 (image.png)]:**
> ## Da Vinci Architecture (Al Core) - Control Unit (2)
> 
> - Instruction execution queue: Includes a matrix operation queue, vector operation queue, and storage conversion queue. Different instructions enter corresponding operation queues, and instructions in the queues are executed according to the entry sequence.
> - Event synchronization module: Controls the execution status of each instruction pipeline in real time, and analyzes dependence relationships between different pipelines to resolve problems of data dependence and synchronization between instruction pipelines.
> 
> ![](_page_0_Figure_3.jpeg)
![image.png](%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%8A%A2%E5%8D%A0Preemption%EF%BC%88%E5%BC%82%E6%9E%84XPUs%E3%80%81Firmware%E3%80%81%CE%BC-controller%EF%BC%89/image%2049.png)

### Ascend 软件栈和推理流程

> **[图片提取文字 (image.png)]:**
> ## Logic Architecture of Ascend Al Processor Software Stack (1)
> 
> - L3 application enabling layer: It is an application-level encapsulation layer that provides different processing algorithms for specific application fields. L3 provides various fields with computing and processing engines. It can directly use the framework scheduling capability provided by L2 to generate corresponding NNs and implement specific engine functions.
>   - Generic engine: provides the generic neural network inference capability.
>   - Computer vision engine: encapsulates video or image processing algorithms.
>   - Language and text engine: encapsulates basic processing algorithms for voice and text data.
> 
> ![](_page_0_Figure_5.jpeg)
![image.png](%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%8A%A2%E5%8D%A0Preemption%EF%BC%88%E5%BC%82%E6%9E%84XPUs%E3%80%81Firmware%E3%80%81%CE%BC-controller%EF%BC%89/image%2050.png)

> **[图片提取文字 (image.png)]:**
> ## Logic Architecture of Ascend AI Processor Software Stack (2)
> 
> - L2 execution framework layer: encapsulates the framework calling capability and offline model generation capability. After the application algorithm is developed and encapsulated into an engine at L3, L2 calls the appropriate deep learning framework, such as Caffe or TensorFlow, based on the features of the algorithm to obtain the neural network of the corresponding function, and generates an offline model through the framework manager. After L2 converts the original neural network model into an offline model that can be executed on Ascend Al chips, the offline model executor (OME) transfers the offline model to Layer 1 for task allocation.
> - L1 chip enabling layer: bridges the offline model to Ascend AI chips. L1 accelerates the offline model for different computing tasks via libraries. Nearest to the bottom-layer computing resources, L1 outputs operator-layer tasks to the hardware.
> - L0 computing resource layer: provides computing resources and executes specific computing tasks. It is the
>   hardware computing basis of the Ascend AI chip.
![image.png](%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%8A%A2%E5%8D%A0Preemption%EF%BC%88%E5%BC%82%E6%9E%84XPUs%E3%80%81Firmware%E3%80%81%CE%BC-controller%EF%BC%89/image%2051.png)

> **[图片提取文字 (image.png)]:**
> ## Neural Network Software Flow of Ascend AI Processors
> 
> - The neural network software flow of Ascend AI processors is a bridge between the deep learning framework and Ascend AI chips. It realizes and executes a neural network application and integrates the following functional modules.
> - Process orchestrator: implements the neural network on Ascend AI chips, coordinates the whole process of effecting the neural network, and controls the loading and execution of offline models.
> - Digital vision pre-processing (DVPP) module: performs data processing and cleaning before input to meet format requirements for computing.
> - Tensor boosting engine (TBE): functions as a neural network operator factory that provides powerful computing operators for neural network models.
> - Framework manager: builds an original neural network model into a form supported by Ascend AI chips, and integrates the new model into Ascend AI chips to ensure efficient running of the neural network.
> - Runtime manager: provides various resource management paths for task delivery and allocation of the neural network.
> - Task scheduler: As a task driver for hardware execution, it provides specific target tasks for Ascend AI chips. The operation manager and task
>   scheduler work together to form a dam system for neural network task flow to hardware resources, and monitor and distribute different types
>   of execution tasks in real time.
![image.png](%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%8A%A2%E5%8D%A0Preemption%EF%BC%88%E5%BC%82%E6%9E%84XPUs%E3%80%81Firmware%E3%80%81%CE%BC-controller%EF%BC%89/image%2052.png)

> **[图片提取文字 (image.png)]:**
> ## Neural Network Software Flow of Ascend AI Processors
> 
> ![](_page_0_Figure_1.jpeg)
![image.png](%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%8A%A2%E5%8D%A0Preemption%EF%BC%88%E5%BC%82%E6%9E%84XPUs%E3%80%81Firmware%E3%80%81%CE%BC-controller%EF%BC%89/image%2053.png)

> **[图片提取文字 (image.png)]:**
> ## Data Flowchart of the Ascend AI Processor - Facial Recognition Inference Application (1)
> 
> - Camera data collection and processing
>   - Compressed video streams are transmitted from the camera to the DDR memory through PCIe.
>   - DVPP reads the compressed video streams into the cache.
>   - After preprocessing, DVPP writes decompressed frames into the DDR memory.
> 
> ![](_page_0_Figure_5.jpeg)
![image.png](%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%8A%A2%E5%8D%A0Preemption%EF%BC%88%E5%BC%82%E6%9E%84XPUs%E3%80%81Firmware%E3%80%81%CE%BC-controller%EF%BC%89/image%2054.png)

> **[图片提取文字 (image.png)]:**
> ## Data Flowchart of the Ascend AI Processor - Facial Recognition Inference Application (2)
> 
> - Data inference
>   - The task scheduler (TS) sends an instruction to the DMA engine to pre-load the AI resources from the DDR to the on-chip buffer.
>   - The TS configures the AI core to execute tasks.
>   - □ The AI core reads the feature map and weight, and writes the result to the DDR or on-chip buffer.
> - Facial recognition result output
> - After processing, the AI core sends the signals to the TS, which checks the result. If another task needs to be allocated, the operation in step (4) is performed.
>   - □ When the last AI task is complete, the TS reports the result to the host.
![image.png](%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%8A%A2%E5%8D%A0Preemption%EF%BC%88%E5%BC%82%E6%9E%84XPUs%E3%80%81Firmware%E3%80%81%CE%BC-controller%EF%BC%89/image%2055.png)

## RV-Arch SoC

[https://riscv.org/blog/risc-v-ai-native/](https://riscv.org/blog/risc-v-ai-native/)

[https://riscv.org/blog/design-approaches-and-architectures-of-risc-v-socs/](https://riscv.org/blog/design-approaches-and-architectures-of-risc-v-socs/)

[https://riscv.org/blog/semidynamics/](https://riscv.org/blog/semidynamics/)

### RV Architecture

x-bit kernel、x-bit OS、x-bit CPU、x-bit SoC是指支持x-bit的地址空间，**每个x-bit（虚拟地址）编码一个字节/entry**。

为了向后兼容，64-bit地址空间的**低32bit部分和32-bit地址空间一致**。

> **[图片提取文字 (image.png)]:**
> | OXFFFF | FFFF |          | DRAM                                                                                                       |
> |--------|------|----------|------------------------------------------------------------------------------------------------------------|
> | 0X8000 | 0000 |          | 2GB address space for DRAM<br>Subset of 64-bit system<br>[Must be used before<br>utilizing higher address] |
> |        |      |          |                                                                                                            |
> | 0X7FFF | FFFF |          | MAPPED I/O                                                                                                 |
> |        |      |          | MAPPED I/O: PCIe                                                                                           |
> |        |      |          | 64KB Pages                                                                                                 |
> | 0X4000 | 0000 |          | 64KB Pages                                                                                                 |
> | 0X3FFF | FFFF | ROM      | ROM, RAM, I/O                                                                                              |
> |        |      | RAM      | BOOT ADDRESS: 0x0000 0000/                                                                                 |
> | 0X0001 | 0000 | SOC I/O  | 0xFFFF 0000                                                                                                |
> |        |      |          | ROMs, RAMs, Static Memory                                                                                  |
> | 0X0000 | FFFF |          | SoC Peripheral Registers                                                                                   |
> |        |      | BOOT ROM | Dynamically Mapped I/O                                                                                     |
> | 0x0000 | 0000 |          | 64KB pages                                                                                                 |
> 
> ĕ
![image.png](%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%8A%A2%E5%8D%A0Preemption%EF%BC%88%E5%BC%82%E6%9E%84XPUs%E3%80%81Firmware%E3%80%81%CE%BC-controller%EF%BC%89/image%2056.png)

> **[图片提取文字 (image.png)]:**
> Let's understand the 32-bit memory map shown above. We design a chip with 4GB address space for the bare-metal coding. Within 4GB, we map everything: ROM, I/Os, SRAM, DRAM, external memories, etc., as shown in the figure. In this case, you can have only limited DRAM (2 to 3 GB), as we need a minimum of 1 GB for other components like ROM and I/O interfaces. The bare-metal software application will follow physical addressing with RAM. Usually, we follow this framework and approach for designing 32-bit embedded microcontrollers.
> 
> RISC-V ISA also offers Sv32 address translation to realize 32-bit systems with a 32-bit OS. The Sv32 address translation scheme generates a 32-bit virtual address with 4GB of virtual address space and a 34 bits physical address with 16GB of physical address space. Now, with this 16GB address space, we can comfortably increase RAM, ROM, and I/Os. This explains how we usually scale up a hardware system even with 32-bit RISC-V chips. Still, 4GB of virtual address space is limited for desktop/server OS and applications.
> 
> Also, general purpose 32-bit operating systems struggle to manage bigger DRAM chips. A 32-bit OS kernel can comfortably manage a DRAM chip of maximum size 32GB; hence, we prefer 64-bit operating systems on 64-bit SoCs. So, let's explore the 64-bit memory map.
![image.png](%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%8A%A2%E5%8D%A0Preemption%EF%BC%88%E5%BC%82%E6%9E%84XPUs%E3%80%81Firmware%E3%80%81%CE%BC-controller%EF%BC%89/image%2057.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%8A%A2%E5%8D%A0Preemption%EF%BC%88%E5%BC%82%E6%9E%84XPUs%E3%80%81Firmware%E3%80%81%CE%BC-controller%EF%BC%89/image%2058.png)

> **[图片提取文字 (image.png)]:**
> Using 36 bits (address bus), we can design a 64-bit SoC comfortably with all I/O interfaces, controllers, and memories within a 64GB address space. This allows us to expand the DRAM more than 30 to 40 GB and realize any kind of bare-metal complex embedded and OS-based desktop applications.
> 
> Usually, we prefer a 36/40-bit memory map to design 64-bit RISC-V SoCs, rather than simply using a 64-bit address space (address bus). Also, RISC-V offers various address translation schemes – Sv39, Sv48, Sv57, and Sv64 – to increase the virtual and physical address space for 64-bit SoCs, as shown below.
![image.png](%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%8A%A2%E5%8D%A0Preemption%EF%BC%88%E5%BC%82%E6%9E%84XPUs%E3%80%81Firmware%E3%80%81%CE%BC-controller%EF%BC%89/image%2059.png)

x-bit虚拟地址经过MMU翻译成y-bit的内存物理地址，y>x说明**内存同时驻留多个进程的地址空间**，翻译模式记为Sv-y。

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%8A%A2%E5%8D%A0Preemption%EF%BC%88%E5%BC%82%E6%9E%84XPUs%E3%80%81Firmware%E3%80%81%CE%BC-controller%EF%BC%89/image%2060.png)

> **[图片提取文字 (image.png)]:**
> compatibility—porting 32-bit software (OSes, bootloaders & drivers) to 64-bit systems. Generally, the firmware, bootloaders and test codes are 32-bit software even for 64-bit SoCs. It executes without MMUs; hence all peripherals must be implemented within the physical address space - following physical addressing, as shown in the above figure. Innovating high-performance electronic products requires more than just creativity – it demands a deep understanding of design approaches like 32-bit and 64-bit SoCs, along with modern system-level design
> 
> strategies. System level designers must know the nuts and bolts of processor technologies like RISC-V ISA,
> 
> and SoC and Software design methodologies as well.
> 
> The Sv39 address translation scheme generates a 39-bit virtual address with 512GB of virtual address space
> 
> for software applications and a 56-bit physical address with 64PB of physical address space, as shown
> 
> above. The 32-bit memory map is always a subset of the 36/40-bit memory map for backward
![image.png](%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%8A%A2%E5%8D%A0Preemption%EF%BC%88%E5%BC%82%E6%9E%84XPUs%E3%80%81Firmware%E3%80%81%CE%BC-controller%EF%BC%89/image%2061.png)

### RV as AI platform

RV ISA设计**模块化、可定制和标准化**，作为集成NPU、TPU、CiM和类脑计算等各种架构的**SoC** **ISA**。每种架构处理器被视为RV ISA的扩展。

> **[图片提取文字 (image.png)]:**
> The RISC-V ISA is designed for modularity. Its dual advantage of standardization and customization makes it ideal for domain-specific compute<mark>, such as:</mark>
> 
> **Neural Processing Units (NPUs):** NPUs are specialized processors optimized for running neural network operations. With RISC-V's modular ISA, designers can tightly integrate NPUs to accelerate inference, cut energy use, and boost real-time AI responsiveness across industries from mobile to automotive. They're a great example of full-stack optimization for inference speed – a key topic at this year's AI Infra Summit.
> 
> Tensor Acceleration Engines: Tensor accelerators handle large matrix multiplications at the heart of AI training and inference. RISC-V enables custom tensor extensions, letting vendors optimize throughput for specific workloads like recommendation engines, speech recognition, or large language models—critical for efficient scaling.
> 
> Compute-in-Memory (CiM): CiM architectures process data directly where it is stored, reducing costly movement between memory and processor. RISC-V's extensibility allows seamless integration of CiM approaches, vital for edge AI and IoT devices constrained by power and latency requirements.
> 
> Optical or Neuromorphic Paths: Optical and neuromorphic computing paths mimic the brain or use light to move and process information with extreme efficiency. RISC-V provides the flexibility to connect these unconventional paradigms with mainstream compute, enabling breakthrough architectures for next-generation AI workloads.
![image.png](%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%8A%A2%E5%8D%A0Preemption%EF%BC%88%E5%BC%82%E6%9E%84XPUs%E3%80%81Firmware%E3%80%81%CE%BC-controller%EF%BC%89/image%2062.png)

**RV SoC是可编程的异构AI SoC。**

GPU是**CUDA SoC**，其中RV Cores支撑计算之外的**控制和管理功能**。

> **[图片提取文字 (image.png)]:**
> - Andes Technology, a founding member of RISC-V International, provides a comprehensive family of RISC-V IP cores with DSP, vector, and extensibility features supported by automation tools for custom instruction extensions powering AI in SoCs across applications from ultra-low-power sensor nodes to the data center.
>   Codasip empowers SoC developers with customizable RISC-V cores using its CodAL design language. It claims
> - over 2 billion cores shipped, including configurations tailored for AI/ML edge use cases.
> - NVIDIA, which provides much of the accelerator technology that has empowered the AI boom, shipped over Ibn RISC-V cores in 2024, as well as announcing its intention to port its CUDA AI acceleration stack to the RVA23 profile. This underscores that RISC-V's relevance extends beyond the open software ecosystem to mainstream AI applications, serving as the orchestrator of the world's leading proprietary GPU architecture.
> - Semidynamics, a European Integrated Matrix Extensions (IME) pioneer and supplier of IP cores, recently introduced a RISC-V Tensor Unit that supports streaming workloads, sparse/dense tensor ops, and AI dataflow processing. By embedding vector and tensor capability into the CPU, Semidynamics is tackling energy efficiency and PPA challenges central to the AI data centers track at Infra Summit 2025.
> - SiFive, a commercial vendor formed by RISC-V's inventors, delivers CPU cores in IP form for AI use cases from minimally configured edge sensors up to enterprise-grade cloud infrastructure systems.
> - SpacemiT develops RISC-V processors for AI CPUs—its Muse Book features the K1 chip, as does Deep Computing's laptops. The upcoming 64-core VitalStone V100 targets server-grade AI workloads using the upcoming RVA23 standard.
> - Tenstorrent builds high-performance AI processors using RISC-V CPU cores and chiplet architectures, focusing on scalable compute from edge to data center. It collaborates with Japan's LSTC on a 2nm AI accelerator and has
![image.png](%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%8A%A2%E5%8D%A0Preemption%EF%BC%88%E5%BC%82%E6%9E%84XPUs%E3%80%81Firmware%E3%80%81%CE%BC-controller%EF%BC%89/image%2063.png)

> **[图片提取文字 (image.png)]:**
> ## **Tensor Unit**
> 
> Ultra-Fast Al Power efficiency Supports transformers
> 
> ![](_page_0_Picture_2.jpeg)
> 
> ![](_page_0_Picture_3.jpeg)
> 
> ![](_page_0_Picture_4.jpeg)
> 
> ## **Vector Unit**
> 
> Up to 2048-bits RISC-V OOO Customizable
> 
> ## **CPU Cores**
> 
> 64-bit Cores RISC-V OOO AXI / CHI Fast unaligned
> 
> ![](_page_0_Picture_9.jpeg)
> 
> ![](_page_0_Picture_10.jpeg)
> 
> ## Gazzillion Misses<sup>™</sup>
> 
> Maximizes bandwidth For Big Data, AI, HPC
![image.png](%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%8A%A2%E5%8D%A0Preemption%EF%BC%88%E5%BC%82%E6%9E%84XPUs%E3%80%81Firmware%E3%80%81%CE%BC-controller%EF%BC%89/image%2064.png)