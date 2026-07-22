# FEATHER: A Reconfigurable Accelerator with Data Reordering Support for Low-Cost On-Chip Dataflow Switching

# Abstract

Feather：加速器架构，ZCU104 FPGA实现；

LayoutLoop：基于TimeLoop（dataflow cost model）改进的layout access model，比较不同架构的性能；

将实现的加速器和DPU/Gemmini比较；

> **[图片提取文字 (image.png)]:**
> Abstract—The inference of ML models composed of diverse structures, types, and sizes boils down to the execution of different dataflows (i.e. different tiling, ordering, parallelism, and shapes). Using the optimal dataflow for every layer of workload can reduce latency by up to two orders of magnitude over a suboptimal dataflow. Unfortunately, reconfiguring hardware for different dataflows involves on-chip data layout reordering and datapath reconfigurations, leading to non-trivial overhead that hinders ML accelerators from exploiting different dataflows, resulting in suboptimal performance. To address this challenge, we propose FEATHER, an innovative accelerator that leverages a novel spatial array termed NEST and a novel multi-stage reduction network called BIRRD for performing flexible data reduction with layout reordering under the hood, enabling seamless switching between optimal dataflows with negligible latency and resources overhead. For systematically evaluating the performance interaction between dataflows and layouts, we enhance Timeloop, a state-of-theart dataflow cost modeling and search framework, with layout assessment capabilities, and term it as Layoutloop. We model FEATHER into Layoutloop and also deploy FEATHER end-to-end on the edge ZCU104 FPGA. FEATHER delivers  $1.27 \sim 2.89 \times$ inference latency speedup and  $1.3 \sim 6.43 \times$  energy efficiency improvement compared to various SoTAs like NVDLA, SIGMA and Eyeriss under ResNet-50 and MobiletNet-V3 in Layoutloop. On practical FPGA devices, *FEATHER* achieves  $2.65/3.91 \times$ higher throughput than Xilinx DPU/Gemmini. Remarkably, such
![image.png](FEATHER%20A%20Reconfigurable%20Accelerator%20with%20Data%20Reo/image.png)

# Intro/概述

**最优条件下，每个layer的dataflow不同；**

> **[图片提取文字 (image.png)]:**
> The mechanism for orchestrating a DNN layer over the accelerator's on-chip compute and memory resources is called dataflow. It can be precisely defined by transformations of the loop nest, as shown in Fig. 1. Several prior works [33], [41] have demonstrated that dataflows can lead to significant differences in compute utilization and up to two orders of magnitude variance in latency and energy, and thereby motivated the need to support per-layer dataflow flexibility.
![image.png](FEATHER%20A%20Reconfigurable%20Accelerator%20with%20Data%20Reo/image%201.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Fig. 1: Terminology of convolution workload and dataflow
![image.png](FEATHER%20A%20Reconfigurable%20Accelerator%20with%20Data%20Reo/image%202.png)

**改变dataflow需要：**

**1、重配置片上的compute、distribute和reduction网络；**

**2、修改片上Buffer的layout；**

**layout reorder很重要，因为SRAM Bank conflict；但reorder的延迟和能耗代价很高；**

**片外reorder需要在片外存储和computation（计算地址）之间来回数据移动；**

**片上reorder需要额外的存储和关键路径的延时；**

> **[图片提取文字 (image.png)]:**
> Changing dataflows on accelerators requires (a) reconfiguring datapaths in computation, distribution, and reduction networks, and (b) modifying data layout in on-chip buffers. Almost all prior works have focused on the first aspect, and several clever interconnect topologies for data distribution and reduction have been proposed that activate subset of paths at runtime through reconfiguration depending on the dataflow being run [42], [44]. However, data layout in the on-chip buffer is a critical and often overlooked in past work.
> 
> In this work, we demonstrate that the high performance of dataflows is unachievable in practice without layout reordering capability. This is because, without a suitable data layout, the required data may be located in the same SRAM banks and compete at the same SRAM reading ports. Such bank conflict slows down the delivery of data to computation engines, leading to stalling and computation underutilization. Overlooking layout reordering thus introduces a significant 128× performance gap between theory and practice as quantified in Fig. 2. We discuss this with more depth in §II.
> 
> Unfortunately, layout reordering comes with severe latency and energy overheads. Off-chip layout reordering requires back-and-forth data movement between off-chip DRAM/HBM and computation, while on-chip layout reordering requires additional intermediate storage and extra latency in the critical path. In fact, these costs can outweigh the benefits of switching dataflows, leading existing ML accelerators to compromise settling on a single dataflow (e.g., Xilinx DPU, Gemmini, NVDLA, Eyeriss in Table I) that provides good average utilization across all layers, but sub-optimal performance.
![image.png](FEATHER%20A%20Reconfigurable%20Accelerator%20with%20Data%20Reo/image%203.png)

加速器将layout reorder过程隐藏在reduction过程中，按最优dataflow完成片上数据的reorder；

加速器能每层切换一次dataflow和layout；

评估不同layout切换配置的加速性能的tool；

> **[图片提取文字 (image.png)]:**
> To fully explore the potential of *FEATHER*, we also developed a tool that facilitates: (a) dataflow evaluation factoring in data layout, and (b) (layout, dataflow) co-exploration.
![image.png](FEATHER%20A%20Reconfigurable%20Accelerator%20with%20Data%20Reo/image%204.png)

# BG、动机

**dataflow定义为n-loop（tiling、ordering、parrallel、shape）；**

vPE shape，即每个维度parrallel定义的并行shape，映射到实际PE array；

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Fig. 1: Terminology of convolution workload and dataflow
![image.png](FEATHER%20A%20Reconfigurable%20Accelerator%20with%20Data%20Reo/image%202.png)

> **[图片提取文字 (image.png)]:**
> ## A. Dataflow Space in Convolution
> 
> - Fig. 1 depicts a convolution operation with seven dimensions with various shapes. Dataflows can be represented as a nested loop with four types of optimizations [24], [34].
> - (T)iling breaks down dimensions of iActs N, C, H, W into smaller chunks, and enables executing workloads in tile granularity as on-chip storage is limited.
> - (O)rdering allows arbitrary loop reordering (aka "stationarity" [13]) to reuse more data since dimensions N, M, C, P, Q, R, S do not come with loop-carried dependencies except reduction-dependencies over C, R, and S.
> - (P) arallelism allows for arbitrary parallelism over any dimensions as all dependencies are loop-independent, leading to different spatial reuse opportunities.
> - (S)hape defines the virtual grouping of the physical PE array.
> 
> These dataflow flexibility (TOPS) [34] create an extremely large dataflow design space with a complexity of  $O(10^{36})$  for a single convolution layer [27]. The choice of the dataflow affects both runtime performance (as it affects overall compute utilization) and energy efficiency (as it affects the number of accesses across the memory hierarchy). Not surprisingly, no single dataflow is generally optimal for all types of layers given their diverse sizes and shapes [33], [41]. This can be seen by comparing the first two bars (blue and green bars) in Fig. 2.
![image.png](FEATHER%20A%20Reconfigurable%20Accelerator%20with%20Data%20Reo/image%205.png)

> **[图片提取文字 (image.png)]:**
> output-stationary dataflow + fixed layout, error bar shows impact of diff layout (Fixed dataflow-layout)
> searched dataflow w/o layout consideration (theoretical best)\nevaluate theoretical best dataflow under various layouts, error bar shows impact of diff layouts (practice)
> 
> flexible dataflow with data layout switching support (FEATHER, this work)
> 
> ![](_page_0_Figure_1.jpeg)
![image.png](FEATHER%20A%20Reconfigurable%20Accelerator%20with%20Data%20Reo/image%206.png)

 **片上存储逻辑上看作2D-buffer，word字长固定，line-width是buffer一次读写的word最大个数，line-depth是buffer中line的个数；**

物理实现是FPGA芯片中的BRAM或CPU、GPU中的SRAM（cache）；更底层由Bank组实现，line通过bank的读写port传输，同时读写的line个数有上限（port个数）；

> **[图片提取文字 (image.png)]:**
> words a buffer could supply per cycle) and the depth represents the total number of buffer row entries as shown in Fig. 1.
> 
> Physically, on-chip storage is implemented by BRAM/U-RAM in FPGA and SRAM in ASICs, which come with a *fixed number (often two) read or write ports*. Therefore, once arranged into the logical 2D buffer, the number of lines being concurrently accessed is limited by the number of ports. A request that accesses more lines than the available ports will lead to bank conflicts, resulting in a slowdown from the reading/writing delay (resource hazard).
> 
> **Data Layout Terminology.** In this paper, data layout is represented as "(Inter-line dimension order)\_(Intra-line dimension order interleaved with sizes)" with one example shown in Fig. 3. For instance, two commonly used PyTorch data layouts, channel-last [18] and row-major [38], can be interpreted as Channel (C) or Width (W) being the innermost dimension in both inter and intra-line orders, separately.
![image.png](FEATHER%20A%20Reconfigurable%20Accelerator%20with%20Data%20Reo/image%207.png)

> **[图片提取文字 (image.png)]:**
> ## TABLE II: On-chip memory terminology
> 
> T----
> 
> Port
> 
> | 1erm   | Meaning                                                                                                                                         |
> |--------|-------------------------------------------------------------------------------------------------------------------------------------------------|
> | Buffer | A logical 2D on-chip memory (num_line $\times$ line_size) stacking multiple SRAM banks both vertically (num_line) and horizontally (line_size). |
> |        |                                                                                                                                                 |
> 
> Buffer SRAM banks both vertically (num\_line) and horizontally (line\_size).
> 
> Bank A physical 2D SRAM (entries × io) with address/data ports.
> 
> Line/Row A buffer line (line\_size = accumulated IO of horizontal SRAM banks).
> 
> An input/output port, each bank has at most two ports in TSMC 28nm.
![image.png](FEATHER%20A%20Reconfigurable%20Accelerator%20with%20Data%20Reo/image%208.png)

**片上存储layout的表达方式是inter-line dimension order，intra-line dimension order，分别表示line间的layout和line内的layout；**

**（line-width=16）**

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](FEATHER%20A%20Reconfigurable%20Accelerator%20with%20Data%20Reo/image%209.png)

**同时考虑dataflow和layout是因为bank conflict；**

**没有conflict的dataflow和layout称为concordant，否则discordant；**

**concordant dataflow space of layout：某layout下所有concordant的dataflow；**

> **[图片提取文字 (image.png)]:**
> Insight 1: Discordance between dataflow and data layout leads to bank conflicts and results in performance degradation. A discordance between dataflow and data layout leads to slowdown because compute units have to stall and wait for data to arrive, as illustrated by the slowdown from green bar to
> 
> yellow bar in Fig. 2. Taking ResNet-50 layer 47 as an example (Fig. 4-M7), the channel-parallel dataflow requires concurrent access to iActs (H0W0C0:3), which are distributed across four separate lines, including line 0, r4, r5 and r6, in the row-major
![image.png](FEATHER%20A%20Reconfigurable%20Accelerator%20with%20Data%20Reo/image%2010.png)

> **[图片提取文字 (image.png)]:**
> Insight 3: Systematic layout modeling should be factored into dataflow exploration for bridging the theory-practice gap. Dataflow has a huge space, which requires systematic modeling and searching algorithms to identify the optimum. However, many dataflow exploration frameworks [33], [41] and algorithms [9], [27], [28], [30] purely model on-chip storage as bandwidth, often assuming ideal data layouts, which could lead to significant theory-practice performance gap. For instance, all layouts in Fig. 4 possess identical bandwidth, but they result in markedly different compute utilization and energy efficiency for two workloads, which is not the case in the existing frameworks as they do not model layout. In Fig. 2, we find that the best dataflow reported by a mapper from an
![image.png](FEATHER%20A%20Reconfigurable%20Accelerator%20with%20Data%20Reo/image%2011.png)

> **[图片提取文字 (image.png)]:**
> in more dimensions and requires more concurrent data.
> 
> Insight 2: Co-switching (dataflow, layout) for different layers
![image.png](FEATHER%20A%20Reconfigurable%20Accelerator%20with%20Data%20Reo/image%2012.png)

**channel-p dataflow**

PE阵列每行存放不同卷积核（不同输出通道）的多个输入通道权重并固定在原地；

不同输入通道的iAct从上到下流动进行传播，每个cell计算的中间结果从左向右移动进行累加；

输入和权重的读取order完成滑窗卷积；

**sliding-window-p dataflow**

每个cell计算的中间结果留在原地进行累加；

不同宽度坐标（不同位置滑窗）的iAct从上到下流动进行传播，不同卷积核（不同输出通道）的权重从左到右流动进行传播；

输入和权重的读取order完成滑窗卷积；

**channel-last layout：优先排布相邻channel上数据，连续存储channel；**

**row-major：优先排布同一行的数据，连续存储row；**

**和我设计的区别**

**M4/8的dataflow和layout和我设计类似，但M4/8通过地址计算完成滑窗卷积，而我额外设计explicit的img2col模块；**

**固定dataflow下，地址+img2col模块相比纯地址的优势是什么？**

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](FEATHER%20A%20Reconfigurable%20Accelerator%20with%20Data%20Reo/image%2013.png)

**data reorder**

DL模型一般是静态的，部署运行前确定的每层的最优dataflow和layout；

**权重训练得到因此能提前reorder到最优layout，但iAct运行时产生，因此iAct需要运行时reorder；**

> **[图片提取文字 (image.png)]:**
> 1) Reorder Target (iActs): As established above, both weights and input activations (iActs) necessitate layout reordering within the on-chip memory when switching dataflows. For ML inference, the structure and weights of ML models are established prior to deployment, enabling the offline optimal dataflow-layout determination for each layer and offline reordering of all weights. Consequently, an optimal layout for weights within the on-chip scratchpad is assured. However, iActs are generated in real-time, so that iActs reordering happens online. Therefore, this work focuses on layout reordering of iActs.
![image.png](FEATHER%20A%20Reconfigurable%20Accelerator%20with%20Data%20Reo/image%2014.png)

**相同reorder pattern有不同实现和延迟，将pattern和implement解耦，来简化硬件设计和延迟；**

**不同reorder得到的不同layout有不同的concordant dataflow space；**

> **[图片提取文字 (image.png)]:**
> concurrently access up-to two rows within a single bank, such as  $(0,1,2,\cdots,7)$ . This restricts concordant dataflow space to limited T,O,P,S flexibility (see purple quadrilateral in Fig. 5f). • Line Rotation (Fig. 5b) arguments concordant dataflow space
> 
> • Fixed layout (Fig. 5a) is only concordant to dataflows which
![image.png](FEATHER%20A%20Reconfigurable%20Accelerator%20with%20Data%20Reo/image%2015.png)

> **[图片提取文字 (image.png)]:**
> to concurrently access up-to **three** rows within a single bank by storing a copy of a row in other banks. For example, to access three rows including data  $(0, 1, \dots, 7, C, D, E, F)$  from bank 0 in Fig. 5b, row (C,D,E,F) is moved to bank such that it provides  $(0,1,\cdots,7)$  from bank 0 and (C,D,E,F) from bank 1 to avoid bank conflicts. However, line rotation comes at the price of (1) extra bandwidth: it employs three ports for reading data that could be accessed with up-to two ports under concordant layout, (2) storage: it stores a copy of (C, D, E, F). Such price could have been used for supporting more parallelism under arbitrary reordering to improve performance.
![image.png](FEATHER%20A%20Reconfigurable%20Accelerator%20with%20Data%20Reo/image%2016.png)

> **[图片提取文字 (image.png)]:**
> transformation, such as changing layout from HWC\_W2C3 (Fig. 4, L1) to HWC\_W8 (Fig. 4, L2)
> 
> • Row Reorder (Fig. 5d) does not support more concurrent access within a single bank, but enables arbitrary order within each row, hence supporting dataflows with higher O flexibility. Further, row reorder also supports im2col [11], which does
> 
> not reduce bank conflicts because it still accesses the same
> 
> • Arbitrary Reorder (Fig. 5e) enables arbitrary layout trans-
> 
> number of rows from on-chip buffers.
> 
> • Transpose (Fig. 5c) enables concurrently access to up-to two
> 
> rows or columns within a bank, hence augmenting concordant
> 
> dataflow choice with higher P flexibility than fixed layout.
> 
> But pure transpose falls short of supporting tiled layout
![image.png](FEATHER%20A%20Reconfigurable%20Accelerator%20with%20Data%20Reo/image%2017.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Fig. 5: Overview of reordering patterns. The 2D layout without any reordering is shown in 5a, which only allows reading two rows concurrently, assuming true dual-port SRAM. Line Rotation (5b, e.g., Medusa [48]) moves a row from bank 0 to bank 1 prior to reading, enabling simultaneous access to at most three rows from bank 0 through dual-bank ports. This technique, however, utilizes additional port from bank 1, potentially limiting access to other data in bank 1. Transpose (5c, e.g., MTIA [19] and TPUv4i [26]) could swap rows with columns. Row Reorder (5d, e.g., TPUv4i [26]) permutes data within each row. Arbitrary reorder (5e, proposed in this work) enables arbitrary permutation for data within the entire 2D buffer. Line Rotation, Transpose and Row-Reorder are done by prior works by reading at most two rows per bank, leverage Transpose/Permute unit to reorder and then write data back in concordant order (On-chip RAR in 6b). In contrast, FEATHER's BIRRD network (§III-B) performs the Arbitrary-Reorder during the reduction phase of the matrix multiplication or convolution computation (RIR in Fig. 6c). The concordant dataflow space supported by each layout reorder pattern is shown in 5f. Reordering enables a given layout to alter the order of data it could provide per cycle and across cycles. Among four dimensions (T,O,P,S) of concordant dataflow space, reordering enlarges O,P,S by supporting dataflows to read from or write to layout in different order. Note that reordering by itself cannot enlarge T dimension flexibility because higher Tiles flexibility requires accessing more data per cycle.
![image.png](FEATHER%20A%20Reconfigurable%20Accelerator%20with%20Data%20Reo/image%2018.png)

**TPU中row reorder支持im2col？基于im2col的conv和distrbute-compute-reduce的区别是什么？**

？？？

**我设计的加速器也能采用文中的reorder&reduce，但我的创新点在哪？im2col的作用？**

> **[图片提取文字 (image.png)]:**
> ## TPU Design choice #2: Ahead of Time (AoT) Compilation + Less Reliance on Caches
> 
> 本节解答了 TPU 如何通过避免缓存来实现高能效 TPU + XLA 编译器的硬件-软件协同设计。
> 
> 传统的缓存旨在处理不可预测的内存访问模式。一个应用程序的内存访问模式可能与其他应用程序的内存访问模式截然不同。本质上,缓存使硬件更加灵活,能够适应各种应用程序。这也是 GPU 非常灵活 (注意:与 TPU 相比)的一个重要原因。
> 
> 然而,缓存访问(以及一般的内存访问)会消耗大量的能耗。 以下是对芯片(45nm, 0.9V; [18])上操作能耗的粗略估 算。这里的关键点在于,内存访问和控制消耗了大部分能量, 而运算消耗的能量则要少得多。
> 
> ## Instruction Energy Breakdown
> 
> ![](_page_0_Figure_5.jpeg)
![image.png](FEATHER%20A%20Reconfigurable%20Accelerator%20with%20Data%20Reo/image%2019.png)

> **[图片提取文字 (image.png)]:**
> 但是,如果应用程序非常具体,并且其计算/内存访问模式高度可预测,该怎么办?
> 
> 举一个极端的例子,如果我们的编译器可以提前找出所有需要的内存访问,那么我们的硬件就可以只使用暂存器作为缓冲区,而不需要缓存。
> 
> 这正是 TPU 的目标,也是 TPU 与 XLA 编译器协同设计以实现这一目标的原因。XLA 编译器通过提前分析计算图来生成优化的程序。
> 
> JAX 也能与 TPU 很好地配合,但它们使用 @jit 吗?
> 
> TPU 上的 JAX+XLA 处于 JIT 和 AOT 的混合空间,因此容易造成混淆。当我们首次在 JAX 中调用一个已编译的函数时,JAX 会跟踪它并创建一个静态计算图。该计算图会被传递给XLA 编译器,在那里被转换为适用于 TPU 的完全静态二进制文件。在最后的转换阶段,会进行针对 TPU 的优化(例如,最小化内存访问),以使流程适应 TPU。
> 
> 但需要注意的是:如果使用不同的输入形状运行已编译的函数,则必须重新编译和缓存。这就是为什么 JAX 无法很好地处理动态填充或长度取决于输入的 for 循环层。
> 
> 当然,这种方法听起来不错,但也存在一些不方便的缺点。它 缺乏灵活性,而且严重依赖编译器,这是一把双刃剑。
![image.png](FEATHER%20A%20Reconfigurable%20Accelerator%20with%20Data%20Reo/image%2020.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 3. Unrolling the convolution operations in a convolutional layer (biases, sub-sampling, and non-linearity omitted), to produce a matrix-matrix product version.
![image.png](FEATHER%20A%20Reconfigurable%20Accelerator%20with%20Data%20Reo/image%2021.png)

**reorder implement**

off-chip reorder：通过CPU对内存的操作完成，之后加载到片上；

on-chip reorder（RAR）：reduct之后的激活值，通过layout transform模块完成；

本文的reorder（RIR）：reduct过程中，按照reorder后layout将激活值输出到对应位置；

> **[图片提取文字 (image.png)]:**
> - The layout reorder patterns described in Fig. 5 could have different implementations with *different critical-path latency*.
> - 1) Existing Implementations: We classify existing reordering implementations into three categories.
> - a) No Reordering: If there is no reordering, either the accelerator needs to run a fixed dataflow or a subset of dataflows that are concordant to the fixed layout, or pay the cost of bank conflicts due to discordant accesses. This can lead to suboptimal performance (as shown by blue bar in Fig. 2).
> - b) Off-chip Reordering: SoTA that support dataflow switching (Tab. I) require iActs to move to off-chip DRAM, get reordered there by CPU, and then move back to the accelerator. This naturally incurs extra latency and energy costs (Fig. 6a).
![image.png](FEATHER%20A%20Reconfigurable%20Accelerator%20with%20Data%20Reo/image%2022.png)

> **[图片提取文字 (image.png)]:**
> - c) On-chip Reorder After Reduction (RAR): Existing onchip reordering techniques essentially perform reordering after reduction. The post-reduction oActs are first written to the on-chip buffer, then read and sent to a separate unit to perform a layout transformation, and then fed back to compute unit as iActs of the next layer. This puts reordering in the critical path, as shown in Fig. 6b. Previous arts all fall into this bucket with explicit reordering latency, as listed in Tab. III. For example, Medusa [48] proposes dedicated hardware between on-chip buffer to compute unit to implement line rotation (Fig. 5b); Meta's MTIA [19] proposes a Memory Layout Unit (MLU) to implement transpose; Google's TPUv4 [26] also supports row-reordering (Fig. 5d) to facilitate im2col.
> - 2) Proposed Implementation On-Chip Reorder In Reduction (RIR): This work proposes to perform reordering on output during reduction phase of computation, such that oActs are written in the layout concordant with the dataflow of the next layer. We call this Reorder in Reduction (RIR). RIR implicitly modifies the layout during the reduction process when generating oActs instead of transforming iActs from one layout to another, as depicted inFig. 6c. This approach (i) removes reordering from critical path, (ii) reduces the total number of partial sums into fewer final sums, reducing buffer access and effectively minimizing potential bank conflicts. §IV provides more details.
![image.png](FEATHER%20A%20Reconfigurable%20Accelerator%20with%20Data%20Reo/image%2023.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> (a) Off-chip Data Reorder. (b) Reorder after Reduction (prior works). (c) Reorder in Reduction (RIR, this work).
> 
> Fig. 6: Comparison of data reordering *implementations*. This work proposes RIR that eliminates reorder latency and bank conflicts. We discuss on-chip reorder patterns, including transpose, line rotation, row-reorder and arbitrary reorder, in Fig. 5.
![image.png](FEATHER%20A%20Reconfigurable%20Accelerator%20with%20Data%20Reo/image%2024.png)

**不同dataflow本质是spatial reuse和temporal reuse的trade-off；**

**weight-stationary对权重temporal reuse，对输入和中间结果spatial reuse；**

**output-stationary对中间结果temporal reuse，对权重和输入spatial reuse；**

> **[图片提取文字 (image.png)]:**
> define two additional terms: temporal reuse happens when both use and reuse access exactly the same address and spatial reuse occurs when its use and reuse access different addresses that are located in the same cache line. Consider a sequence of memory accesses shown in Figure 8: a1,b1,e1,b2,c1,d1,a2, where locations a, b, and c occupy cache line N, and locations d and e reside on subsequent cache line N+1. In this example, the temporal reuse distance of access a2 is four, because there are four unique locations accessed between the two consecutive accesses to a, namely, b, c, d, and e. Access d1 is not a temporal reuse, however, it is a spatial reuse since we previously accessed
> 
> location e, which resides on the same cache line as d. The spatial
> 
> reuse distance of access d1 is two.
> 
> Since a unit of memory access in a modern processor is a cache line, we
![image.png](FEATHER%20A%20Reconfigurable%20Accelerator%20with%20Data%20Reo/image%2025.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](FEATHER%20A%20Reconfigurable%20Accelerator%20with%20Data%20Reo/image%2026.png)

# Arch设计特点

2D PEs vs 1D PEs

**2D的PE组织将数据流固化在PE的连接中，牺牲了数据流的灵活性，换取更大规模的扩展scalability；**

**1D的PE组织设置distribution和reduction的片上网络NoC，有更好的数据流灵活性，但NoC设计复杂、消耗资源，牺牲更大规模的扩展scalability；**

> **[图片提取文字 (image.png)]:**
> have better scalability but are limited in their dataflow options due to their rigid structure, leading to suboptimal utilization due to mismatch of layer shapes and array aspect ratios, as prior works have shown [35], [45]. 1D arrays with flexible distribution and reduction NoCs [32] have been shown to support arbitrary dataflows with full-range of TOPS (§II-A), specifically flexible parallelism and shape. However, they suffer from scalability issues due to their all-to-all NoCs.
![image.png](FEATHER%20A%20Reconfigurable%20Accelerator%20with%20Data%20Reo/image%2027.png)

**weight-stationary的例子：每行4个PE映射Pco=2和Pci=2，每列4个PE映射Pco=4，每个PE存储im2col后的一个通道的权重；注意Pci映射的2个PE输入需要同步，不同于传统weight-stationary；**

取消原本**weight-stationary中单输入通道卷积结果在行内PE间传播并累加得到部分和，即将行内PE映射到*Pci的乘法累加*；**

**改为将行内PE映射到Pco×*Pci的乘法，*完成Pco个输出通道的单输入通道卷积（Phase 1），单输入通道卷积结果的累加统一由reduction模块完成（Phase 2）；**

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Fig. 9: Illustration of the *FEATHER* with *NEST* and *BIRRD* employing a convolutional operation with a  $2 \times 2$  weights featuring 2 input channels (C = 2) and generating 16 output channels (M = 16) across a  $4 \times 4$  iAct with 2 input channels. The depicted dataflow utilizes a weight-stationary approach, where each PE has a local register file containing a channel of weights  $(2 \times 2)$ . The dataflow is parallelized for two input channel and two output channel across four PE columns, and for four kernels across four PE rows. In each row, four PEs generate 4 partial sums, contributing to 2 final sums, which thus necessitates a 4:2 spatial reduction in the BIRRD to produce two outputs. We assume the weights are already preloaded into NEST before the first cycle in this illustration. The iActs are streamed from the top, undergo multiplication with corresponding weight values (e.g., w0 in the top-left PE at cycle-0), and are locally accumulated for the next set of inputs (e.g., until cycle-3 in the top-left PE). Following this initial phase of local temporal reduction, the top row transmits the locally reduced result to the BIRRD for the second phase of spatial reduction. In the steady state, BIRRD reduces data from one NEST row per cycle (cycles 4-6). In steady state, all PEs are working and there is no output bus conflict for PEs of the same column. This is because, during phase-2 of spatial reduction in one PE, remaining PEs of the same column perform local reduction. In general, AW × AH NEST takes  $AH^2$  cycles to load weights, and ping-pong local registers are instantiated to hide such latency behind computation. BIRRD could reduce results from PEs at different rows as long as only one PE per column uses the output bus. Takeaway: NEST utilizes local temporal and global spatial reduction to (i) ensure all PEs of the same column share the same output bus without competition while achieving full utilization, and (ii) hide weight loading latency in steady phase.
![image.png](FEATHER%20A%20Reconfigurable%20Accelerator%20with%20Data%20Reo/image%2028.png)

> **[图片提取文字 (image.png)]:**
> Phase 1: Local Temporal Reduction. NEST involves local registers in each PE for temporal (local) reduction of partial sums. This is then followed by a phase of global reduction via the reduction network (described in §III-B). Phase 2: Interleaved Spatial Forwarding and Reduction. However, unlike prior works where all PEs participate simultaneously in the spatial reduction, the PE rows in FEATHER perform spatial reduction one after another, temporally mul-
> 
> tiplexing on the reduction network. Further, while each PE
![image.png](FEATHER%20A%20Reconfigurable%20Accelerator%20with%20Data%20Reo/image%2029.png)

**灵活dataflow：reduction&reorder模块将任意形状的中间结果reduce到最终结果，而2D的PEs的因映射由数据排布决定（reorder）；**

> **[图片提取文字 (image.png)]:**
> arbitrary dataflow parallelism strategies and shapes (§II-A). This is because Phase 2 can be configured to create arbitrarysized reduction groups (i.e., all outputs can be unique or any combinations can be reduced) enhancing mapping flexibility. FEATHER supports inter-layer pipelining. We deploy distinct
> 
> Flexible Dataflow: FEATHER retains the ability to support
> 
> FEATHER supports inter-layer pipelining. We deploy distinct computation engines for ReLU, BatchNorm, and MaxPooling. For AvgPooling layers, they are transformed into convolution operations and executed within the NEST. When there is a sole
![image.png](FEATHER%20A%20Reconfigurable%20Accelerator%20with%20Data%20Reo/image%2030.png)

> **[图片提取文字 (image.png)]:**
> requirement for reorder and reduction, the PE Array can be bypassed, directing inputs from NEST directly to the BIRRD. To optimize storage utilization and reduce data movement costs, all computation engines utilize the same on-chip storage.
![image.png](FEATHER%20A%20Reconfigurable%20Accelerator%20with%20Data%20Reo/image%2031.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Fig. 8: Micro-architecture of *FEATHER*'s datapath for convolution/GEMM. For convolution, the NEST reads iActs from StaB and weights from StrB, streaming both in a top-to-bottom pipeline. PEs in a column time-multiplex a common output bus. *BIRRD* conducts global spatial reduction and reorders results for targeted StaB banks during reduction, altering data layout in StaB. NEST facilitates inter-layer pipelining by reading iActs from StaB Ping (or Pong) and writes oActs (next-layer iActs) back to StaB Pong (or Ping). Note: *FEATHER* is scalable architecture and we show 8-input *BIRRD* as an example.
![image.png](FEATHER%20A%20Reconfigurable%20Accelerator%20with%20Data%20Reo/image%2032.png)

**reduction&reorder模块Bird**

**Bird作为PE和OBuffer之间的接口，接收PE的中间结果，进行reduce后按照输出layout对齐并输出到OBuffer的指定位置；**

> **[图片提取文字 (image.png)]:**
> to reorganize data during the reduction phase. It receives computation results from the previous stage and directs them to new positions in the output buffer while concurrently reducing the data. This process aligns the data in the format needed for the subsequent dataflow, enabling *FEATHER* to seamlessly co-switch (dataflow, layout) for each layer.
> 
> ## Algorithm 1: Inter-stage Connectivity for AW-input BIRRD
> 
> ```
> 1: output[i][id]/input[i][id] (id \in [0,AW)) refers to id-th output/input port of BIRRD switches at the stage i.
> ```
> 
> - 2: **FUNCTION** reverse\_bits(data, bit\_range)
> - 3:  $mask = (1 \ll bit\_range) 1$
> - 4:  $reversed\_bits = 0$
> - 5: **for** i FROM 0 TO bit\_range 1
> - 6: **if** (data  $(1 \ll i)$ )
> - 7: reversed\_bits  $= (1 \ll (bit\_range 1 i))$
> - 8: **return** (data & ∼mask) | reversed\_bits
> - 9: **for** i in  $[0, 2 \times log_2(AW))$  // i is stage\_id
> - 10: **for** j in [0, AW) // j is port\_id
> - output[i][j]-input[i+1][reverse\_bits(j, min( $log_2(AW)$ , 2+i,  $2 \times log_2(AW) i$ ))] (- indicates output connects to input)
![image.png](FEATHER%20A%20Reconfigurable%20Accelerator%20with%20Data%20Reo/image%2033.png)

> **[图片提取文字 (image.png)]:**
> 1) BIRRD Topology: The BIRRD topology is interfaced with NEST engine one side and output buffer on the other side, and is composed of two butterfly networks back-to-back with log(AW)-bit bit reverse connections [16]. This topology grants symmetry with respect to the middle, enabling the construction of each half separately. Each input of BIRRD receives data from one column-wise bus of the NEST while each output of BIRRD forwards the result to one output buffer and eventually back to one bank of stationary buffer (StaB, refer to Fig. 7). For NEST with AW columns in total (AW must be a power of 2), the BIRRD encompasses  $2 \times log(AW)$  stages with AW/2switches located at every stage. The inter-stage connections of
> 
> The topology of *BIRRD* has been proven to be strictly non-blocking for unicast (any single data point among concurrent inputs sent to a single output) [5] and rearrangeably non-blocking for multicasting (at least one data point among all concurrent inputs sent to multiple output ports) [8], [16], [36]. We found no multicasting case that it cannot accommodate.
> 
> BIRRD are outlined in Alg. 1.
![image.png](FEATHER%20A%20Reconfigurable%20Accelerator%20with%20Data%20Reo/image%2034.png)

**Bird是由Eggs组成的多层网络，每个Eggs两输入两输出，完成4种运算（=、swap、+=、=+）；**

**reduce group定义为归约到1个最终输出的中间结果组；**

**Bird能将多个reduce group规约到最终结果，同时将最终结果route到多个输出port；**

> **[图片提取文字 (image.png)]:**
> - Pass (=) / Swap (×): directly pass left (right) input data to left (right) output port, or swap them.
>   Add-Left (\(\pi\)) / Add-Right (\(\pi\)): Accumulates data from input
> - ports and transmits results to the left/right output port, with the secondary output inheriting the input from the same direction. Extra broadcast functions could be added in the Eggs to duplicate accumulated results in multiple banks of StaB.
![image.png](FEATHER%20A%20Reconfigurable%20Accelerator%20with%20Data%20Reo/image%2035.png)

> **[图片提取文字 (image.png)]:**
> of inputs that get reduced into one output. AW-input BIRRD supports arbitrary number of reduction groups (up to AW). • Arbitrary Reordering: The rearrangeably multicasting capability enables BIRRD to route results from many reduction groups to many arbitrary output ports concurrently.
![image.png](FEATHER%20A%20Reconfigurable%20Accelerator%20with%20Data%20Reo/image%2036.png)

**reduce&reorder的例子**

**去除行内的spatial reuse&reduce，全部在Bird中进行，提高dataflow的灵活性；**

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Fig. 10: Comparison between per-layer flexible dataflows in *FEATHER* and fixed-dataflow in the systolic array under GEMM. *FEATHER* dynamically alters layout by redirecting oActs to various banks with distinct writing addresses, exemplified by rerouting a blue result from bank 0 (Workload A) to bank 2 (Workload A Change oAct Layout). *FEATHER* consistently outperforms SA in irregular-sized GEMM (Workload B, C, D), achieving near full utilization. Enhanced utilization arises from (1) enabling cross-column spatial reduction using *BIRRD* in *FEATHER*, e.g. *FEATHER* maps K dimension across the entire 2D array instead of a single PE in SA under workload D. (2) Eliminating SA's horizontal rigid reuse links, thereby enabling independent mappings across columns, e.g. (Workload C) adopting iAct stationary in first three columns and weights stationary in the last column. *BIRRD* could perform pure reordering to change the layout when no spatial reduction is required (e.g. *BIRRD* reordering all incoming results to target banks directly under workload B). **Takeaway:** BIRRD's flexible reduction enhances compute utilization across diverse skewed shapes, expanding the range of dataflows that NEST can efficiently support.
![image.png](FEATHER%20A%20Reconfigurable%20Accelerator%20with%20Data%20Reo/image%2037.png)

**架构是一种dataflow灵活性和设计复杂度的折中、逻辑自洽，但感觉扩展性、reduce&reorder网络的完备性仍然不会理想；**

**配置Birds中Eggs的路由算法**

> **[图片提取文字 (image.png)]:**
> From a routing perspective, reduction can be viewed as a reverse multicasting operation, where multiple input data points target the same output port and are reduced upon encountering each other at BIRRD Eggs. Thus, we adopt the multicasting routing algorithm [4] to establish paths and configurations for BIRRD Eggs, enabling reordering during reduction. If a
![image.png](FEATHER%20A%20Reconfigurable%20Accelerator%20with%20Data%20Reo/image%2038.png)

> **[图片提取文字 (image.png)]:**
> certain input-output connection cannot be established by the algorithm [4], we will brute force all possible configurations. Fig. 10 showcases how BIRRD supports arbitrary dataflows and layout switching requirements.
![image.png](FEATHER%20A%20Reconfigurable%20Accelerator%20with%20Data%20Reo/image%2039.png)

**其他工作的distribution设计**

> **[图片提取文字 (image.png)]:**
> thereby minimizing control, resource, and latency expenses.
![image.png](FEATHER%20A%20Reconfigurable%20Accelerator%20with%20Data%20Reo/image%2040.png)

**片上存储和后处理**

> **[图片提取文字 (image.png)]:**
> As for convolution/GEMM (Fig. 8), iActs are kept stationary within StaB Ping (or Pong), and the resulting oActs are written back into StaB Pong (or Ping) with a new layout. Meanwhile, weights are streamed via StrB (Ping/Pong). StaB requires a multi-bank organization (AW banks), with each bank storing a single data piece, to accommodate the varied write addresses in different banks necessitated by layout changes in *FEATHER*. Conversely, StrB adopts a simplified single-bank structure with an AW-data bandwidth to conserve area, because weights do not need layout reordering.
> 
> 2) Instruction Buffer (IB): The configurations for BIRRD are generated offline and get fetched into IB to configure the reduction networks at run-time.
![image.png](FEATHER%20A%20Reconfigurable%20Accelerator%20with%20Data%20Reo/image%2041.png)

> **[图片提取文字 (image.png)]:**
> 3) Output Buffer (OB): enables in-situ temporal reduction of partial sums when the reduction size of workloads exceeds the overall reduction capacity of both NEST and BIRRD. OB has AW banks, and each equipped with a 32-bit adder. 4) ZP/Scale Buffer and Quantization Module (QM): employing quantization schemes from PyTorch FBGEMM [31] and QNNPACK [17], with 8-bit zero points and 32-bit scales (housed in ZP/Scale Buffer). The quantization module rescaled
> 
> down 32-bit oActs and then quantized to 8-bit oActs.
![image.png](FEATHER%20A%20Reconfigurable%20Accelerator%20with%20Data%20Reo/image%2042.png)

# Arch的运行时

**reorder前后layout示例**

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> | Cycle | iActs<br>Required by<br>Mapping | StaB Ping<br>Read Trace<br>(Line#, Bank#) |
> |-------|---------------------------------|-------------------------------------------|
> | 0     | H0W0 <b>C0:3</b>                | (0, 0:3)                                  |
> | 1     | H0W1 <b>C0:3</b>                | (1, 0:3)                                  |
> | 2     | H1W0 <b>C0:3</b>                | (r0, 0:3)                                 |
> | 3     | H1W1C0:3                        | (r0+1, 0:3)                               |
> | 4     | H0W1 <b>C0:3</b>                | (1, 0:3)                                  |
> | 5     | H0W2C0:3                        | (2, 0:3)                                  |
> | 6     | H1W1C0:3                        | (r0+1, 0:3)                               |
> | 7     | H1W2 <b>C0:3</b>                | (r0+2, 0:3)                               |
> |       |                                 |                                           |
> 
> | Cycle<br># | Generated oActs            | StaB Pong<br>Write Trace<br>(Line#, Bank #) |
> |------------|----------------------------|---------------------------------------------|
> | 0~2        | Temporal Reduction in NEST |                                             |
> | 3~5        | Spatial RIR in BIRRD       |                                             |
> | 6          | M0P0Q0                     | (0, 0)                                      |
> | 7          | M1P0Q0                     | (r1, 0)                                     |
> | 8          | M2P0Q0                     | (r2, 0)                                     |
> | 9          | M3P0Q0                     | (r3, 0)                                     |
> | 10         | M0P0Q1                     | (0, 1)                                      |
> | 11         | M1P0Q1                     | (r1, 1)                                     |
> | 12         | M2P0Q1                     | (r2, 1)                                     |
> | 13         | M3P0Q1                     | (r3, 1)                                     |
> |            |                            |                                             |
> 
> Row-major MPQ\_Q4 (oActs) (CHW\_W4, iActs for next layer)
> 
> ![](_page_0_Figure_4.jpeg)
![image.png](FEATHER%20A%20Reconfigurable%20Accelerator%20with%20Data%20Reo/image%2043.png)

> **[图片提取文字 (image.png)]:**
> Fig. 11: Example of *FEATHER* switching from channel-last layout  $(HWC_C4)$  to a row-major format  $(MPQ_Q4(CHW_W4))$ during reduction without incurring bank conflicts. This is because multiple iActs are reduced into fewer oActs, thereby reducing accesses within each bank. In this example, NEST leverages parallelism along the kernel M and channel C dimensions, reading and vertically streaming four iActs of four input channels from top to bottom. Specifically, at cycle 0, NEST fetches H0W0C0:3 from (line 0, banks 0:3), as recorded in the StaB Ping read trace. Subsequent cycles involve a two-stage reduction: temporal reduction within the PE for cycles 0 to 2, and spatial reduction within BIRRD for cycles 3 to 5, culminating in a single oAct M0P0Q0. This oAct is reordered to bank 0 during reduction and written to line 0 in the StaB Pong during cycle 6. FEATHER's pipelined processing of following iActs is further exemplified in the read/write trace. M0: 3P0Q0 target bank 0 and use connectivity of BIRRD as shown in the left while M0:3P0Q1 use the right. For brevity, the notation of R0:1S0:1 is omitted, which indicates that each PE in *NEST* holds four weights of one channel. **Takeaway:** FEATHER reorders oActs into next layer's desirable layout during reduction, enabling dataflow/layout co-switching.
![image.png](FEATHER%20A%20Reconfigurable%20Accelerator%20with%20Data%20Reo/image%2044.png)

**RIR避免bank conflict**

**channel last到row major，直接reorder意味着从bank内一行存储的多通道数据reorder到单bank多行存储的多通道数据，发生bank conflict；RIR在每个通道数据reduce产生时就reorder到目标bank，将reduce和reorder进行流水，隐藏bank conflict带来的延迟；**

**row major到channel last，直接reorder意味着从bank内一行存储的多列数据reorder到单bank多行存储的多列数据；**

> **[图片提取文字 (image.png)]:**
> ## A. RIR for Bank Conflicts Mitigation and Layout Transform
> 
> In the example shown in Fig. 11, the layout conversion from iActs to oActs is realized via RIR, thereby avoiding the explicit latency in reorder after reduction. This efficiency stems from the key insight that RIR reorders post-reduction oActs into a new layout, rather than directly transforming iActs from one layout to another.
> 
> Specifically, in the reduction phase, numerous iActs naturally get accumulated into fewer oActs and consequently target fewer banks. For example, four iActs get accumulated to one
![image.png](FEATHER%20A%20Reconfigurable%20Accelerator%20with%20Data%20Reo/image%2045.png)

> **[图片提取文字 (image.png)]:**
> oAct that targets a single line in Fig. 11. Conversely, if we directly transform the layout of iActs from channel-last to row-major, four iActs (H0W0C0:3) would target four different lines within the same bank under row-major layout, leading to bank conflicts.
![image.png](FEATHER%20A%20Reconfigurable%20Accelerator%20with%20Data%20Reo/image%2046.png)

**当同时产生的激活值超过memory的写入port数时，仍然会产生bank conflict，因为PE算力和memory的读写能力不一定匹配；此时调整dataflow（比如增大Pci）,消除bank conflict；**

> **[图片提取文字 (image.png)]:**
> While the strategy of 'reordering post-reduction oActs' aids in reducing bank conflicts, conflicts may still arise when the number of partial sums to write into memory exceeds the number of writing ports of the memory. This scenario is particularly common in scaled-up 128 × 128 compute array (Google TPUv4 [26]), as it generates more oActs concurrently.
> 
> FEATHER fully eliminates conflicts with the second key insight that FEATHER picks the dataflow with the number of oActs (partial sums) matching with the number of memory write ports. In essence, FEATHER employs dataflows free from bank conflicts, and the flexible reduction of the BIRRD consistently allows FEATHER to identify such dataflows with high performance and efficiency.
> 
> In summary, *RIR* together with flexible dataflows selection enable *FEATHER* to switch among arbitrary layouts without incurring bank conflicts.
![image.png](FEATHER%20A%20Reconfigurable%20Accelerator%20with%20Data%20Reo/image%2047.png)

**感觉论文并不完美和完善，但逻辑自洽，方方面面都考虑到了，侧重方法；**

我设计的im2col模块功能等效论文的reorder模块，结构更简单、扩展性更好但不可配置为其他数据流；文中的OS数据流下所需的im2col隐藏在前一个算子的reorder中；

我需要找到我的故事线；

论文的不同dataflow的多样性如何？似乎不支持Pow？若支持Pow，则不存在reduce，reorder无法隐藏在reduce中；