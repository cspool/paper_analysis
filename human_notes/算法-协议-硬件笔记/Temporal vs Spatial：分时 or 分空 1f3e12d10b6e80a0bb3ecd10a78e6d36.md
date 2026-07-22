# Temporal vs Spatial：分时 or 分空

# 1、Temporal vs. Spatial Arch

## 0、并行和架构

架构支持任务的并行方式，架构划分本质是任务的不同并行方式。

对n-loop的kernel，**不同loop-index的kernel并行**（展开unrolling）**是时间并行**，因为不同loop-index代表kernel的不同执行轮次（时间）。每个loop-index的**kernel直接作为一个CPU线程**执行，warpSz个loop-index的kernel直接作为一个GPU的warp执行，因此CPU和GPU是**时间架构**。

时间架构下，n-loop的kernel经编译后是指令流，一个kernel线程的指令流在时间架构下一般**串行执行**，无Reg依赖时**可能乱序发射**。

将n-loop的kernel，**kernel内计算节点和数据依赖flow进行loop-index并行是空间并行**，构建成数据流dataflow，因为数据流代表kernel的多个执行步骤（空间）。kernel执行是数据流执行（的迭代），kernel数据流通过映射到**空间架构加速器**上执行。

空间架构下，kernel数据流的数据依赖flow映射到分块加速器的Tile间路由，计算节点映射到Tile内指令流或硬件datapath。

## 1、“分时”

> **[图片提取文字 (image.png)]:**
> ## Temporal Architecture (SIMD/SIMT)
> 
> ![](_page_0_Picture_1.jpeg)
> 
> ## Spatial Architecture (Dataflow Processing)
> 
> ![](_page_0_Figure_3.jpeg)
![image.png](Temporal%20vs%20Spatial%EF%BC%9A%E5%88%86%E6%97%B6%20or%20%E5%88%86%E7%A9%BA/image.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 6. Comparison between the temporal and spatial architectures.
![image.png](Temporal%20vs%20Spatial%EF%BC%9A%E5%88%86%E6%97%B6%20or%20%E5%88%86%E7%A9%BA/image%201.png)

现有的CPU和GPU都有官方的加速库来加速矩阵运算。CPU和GPU为了支持DLP行为，有专门的向量指令集，但CPU的SIMD指令并行度不太高；GPU的SIMT的并行度远超CPU的SIMD指令。

**SIMD和SIMT本质上是一种时间上的并行展开,属于Temporal架构**。比如NV GPU：并行单元只有计算或存储功能（如FP/INT、LD/ST、SF等**core**），控制器**统一**控制所有并行单元，寄存器文件**统一**分派和接收所有的并行单元的数据。类似于一条指令多个数据，即相同的控制，不同的数据映射到并行单元。

分时架构的并行单元PE**只负责计算或存储**，**不同功能的数据流统一**通过寄存器和Buffer传输，并行度更高。

分时架构的优势是通用性和高并行度，算法到硬件的**Mapping**是基本运算DFG >> **基本指令**序列。

## 2、“分空”

**数据流架构**. 相比Temporal Arch的不同之处就是，数据流架构中PE与PE之间可以相互通信的，**数据可以在PE间流动**，这一周期计算出来的结果可以在下一周期流到其他PE参与运算，而**不用发生访存动作**。所以，这样带来的好处是有更大的数据吞吐量，适合具有很好**数据重用特性**的应用。

空间架构的数据流动是为了**复用中间结果或输入**，减少不必要的读写，并且降低瞬时读写压力。

分空架构的并行单元PE需要完成计算或存储，还需实现设计的片上数据流，不同功能的**片上数据流可能不同**，理论吞吐量更大；

分空架构的优势是通过**硬件数据流**实现tiling后算法流程来降低传输压力，实际提高应用性能；但硬件数据流比基本运算的粒度更粗，导致**通用性较差**，拥有硬件数据流过程的算法Mapping到Acc的本质是在**资源约束下将算法拆分成硬件数据流**；

## 3、设计思想

并行的本质是**不同数据**在**不同功能单元**上**同时执行**，由于存储容量限制，算法需要**tiling**成tile后加载到Acc上，tile在Acc上受到并行方式限制，按照**parallel-shape轮流并行**；

并行性能的评估是算法端到端运行过程的分析，硬件对性能的影响有传输速度和处理速度（**影响延迟的硬指标**），存储容量和并行方式（**影响schedule和Mapping的自由度**）；

temporal架构的**整体pipeline**实现不同的**单条指令**。temporal架构的“分时”，因为Acc通过简化并行单元功能来**提高处理速度**，算法tiling后的**tiles在不同时间分别并行**，降低PE复杂度，增大PE并行度；

spatial架构的**数据流pipeline**实现**指令融合后的算子**。spatial架构的“分空”，因为Acc通过片上并行数据流来**降低算法对传输速度的压力**，算法tiling后的**tiles在不同空间分别并行**，增加PE功能性，降低带宽压力；

硬件支持的**算子粒度**越粗越复杂，**特定应用**的性能越高，但算法到硬件Mapping空间的**约束更多**，更难找到最优解，比如**spatial Acc**支持的**张量代数运算**；反之，硬件支持的算子粒度越细越简单，**通用应用**的性能越好，算法到硬件Mapping空间的**约束更少**，容易找到最优解，比如**temporal Acc**支持的**标量/向量代数运算**；

# 2、我的设计

那么我设计的加速器是什么架构？**似乎是spatial arch。**

## 1、RS vs OS的功耗

> **[图片提取文字 (image.png)]:**
> ## (1) Conv Layers
> 
> ![](_page_0_Figure_1.jpeg)
![image.png](Temporal%20vs%20Spatial%EF%BC%9A%E5%88%86%E6%97%B6%20or%20%E5%88%86%E7%A9%BA/image%202.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](Temporal%20vs%20Spatial%EF%BC%9A%E5%88%86%E6%97%B6%20or%20%E5%88%86%E7%A9%BA/image%203.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](Temporal%20vs%20Spatial%EF%BC%9A%E5%88%86%E6%97%B6%20or%20%E5%88%86%E7%A9%BA/image%204.png)

**感觉我设计的OS的数据复用和RD的数据复用程度类似，并且同样对重叠的窗口数据进行了复用的优化（相比shidiannao），并且由于提前排列好数据，PE的结构简单，T/W可能更高？**

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> 图 12 ShiDianNao
![image.png](Temporal%20vs%20Spatial%EF%BC%9A%E5%88%86%E6%97%B6%20or%20%E5%88%86%E7%A9%BA/image%205.png)