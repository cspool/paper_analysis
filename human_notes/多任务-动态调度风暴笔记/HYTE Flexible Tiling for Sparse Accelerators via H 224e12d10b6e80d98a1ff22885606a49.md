# HYTE: Flexible Tiling for Sparse Accelerators via Hybrid Static Dynamic Approaches

# Intro

**稀疏张量计算的数据分布不规则，通用处理器计算稀疏发生load imbalance，加速效果不好。**

Sparse tensor data are prominently used in many domains including graph processing, high-performance computing, and machine learning. **Due to their irregular data distributions, sparse tensor computations are usually inefficient on general-purpose processors, causing numerous random data accesses with little locality in the memory hierarchy, as well as severe load imbalance among parallel computing cores.** Consequently, special-purpose sparse tensor accelerators have been proposed to **optimize critical sparse kernels such as sparse-sparse matrix multiplications [4, 12, 14, 17, 20, 22, 24, 26, 31, 40, 41]. These accelerators typically contain an array of multiply-accumulate processing elements and a hierarchy of SRAM buffers. They use dedicated dataflow schemes that correspond to various iteration orders among tensor dimensions, such as Inner Product (IP), Outer Product (OP), and Gustavson’s.**

**片上存储不能放下完整大张量，需要tiling，tiling需要尽可能复用数据，减少稀疏随机访问带来的fetch数据。tiling尺寸被片上存储限制。**

For large sparse tensors, the on-chip buffer in the accelerator may be insufficient to fit all data, and there would still be substantial random data accesses to the expensive off-chip memory. In such cases, tiling becomes an attractive solution, where the tensor is split into multiple smaller tiles that each fit in the buffer and are maximally reused on-chip before moving to the next tile. However, the **irregular distribution of sparse data makes it difficult to identify the optimal tile shapes and sizes.** A large tile with many non-zero elements may overflow the SRAM buffer and sacrifice data reuse, while a small tile with few non-zero elements would underutilize the buffer space and lead to many tiles which cause unnecessary refetches of the other operand tensors.

**SOTA的tiling方式是：动态改变分块尺寸；静态启发式使用更大tiling。动态tiling的尺寸选择有限（实现复杂度高），静态tiling容易收到稀疏影响导致资源使用率很低。**

State-of-the-art sparse accelerators try to address this difficulty through either dynamic runtime tiling that flexibly changes the tile size [19, 25], or using static heuristics to slightly overbook the buffer space to improve utilization [38]. Unfortunately, purely dynamic tiling has to limit its tiling decisions to a small number of choices due to high implementation complexity, and purely static tiling is usually less efficient when data sparsity varies significantly. In addition, we find that these prior designs have not thoroughly explored the full design space of tiling. Many of their design parameters, including the tile shape, the inter-tile iteration order, and the relative space of SRAM buffers allocated among different operand tensors, are fixed and sub-optimal, especially when the tensors have diverse sparse patterns. Moreover, the metadata to support tiling,e.g., the begin and end locations of the compressed non-zero data in a tile, may also become a significant overhead and require careful management by the hardware accelerator.

**框架特点：加速器支持若干参数下的动态tiling，比如tile大小、尺寸和tiling次序、片上存储分配策略。编译时（offline）基于模型对稀疏张量采样、评估并决策出最合适的tiling。**

In this paper, we take a holistic approach to study the tiling strategies of sparse tensor accelerators and propose HYTE, a hybrid static-dynamic framework for flexible and efficient sparse tiling.
HYTE **supports a rich set of flexible tiling parameters, including the tile size (number of non-zero elements within a tile), the tile shape (coordinate range along each tensor dimension), the iteration order of dimensions across adjacent tiles, and the SRAM buffer allocation policies.** At the static offline phase, HYTE relies on a **scheduler to analyze the sparsity patterns of the operand tensors, using effective yet lightweight sampling approaches to estimate several key metrics. With the help of a performance model, the scheduler then generates a near-optimal tiling scheme with initial values for the above parameters.** Our sampling method is more comprehensive than previous static heuristics [38], and gives more efficient tiling results with only minor offline overheads.

**在基于建模的静态tiling基础上，运行时设计动态tuning，伸缩tile尺寸来保证较高的存储使用率。**

**为辅助tile间执行在片外存储设置元数据，为辅助tile内执行在片上存储设置元数据。**

With the initial tiling scheme, the HYTE hardware further applies dynamic tuning, which shrinks or extends the tile size to always ensure maximum buffer utilization even with highly varying local data sparsity patterns. Because the statically scheduled scheme is near-optimal, dynamic tuning can be much simplified. Besides, HYTE efficiently manages the metadata in both the off-chip memory (for inter-tile execution) and the on-chip buffer (for intra-tile execution), and flexibly shares the buffer space between data and metadata to alleviate the metadata complexity.

**评估：在稀疏数据集上，和SOTA的稀疏加速器比较。**

We evaluate HYTE by comparing it with the state-of-the-art sparse accelerators [19, 25, 38] on a diverse range of sparse datasets. On the representative sparse-sparse matrix multiplication kernel with the Gustavson’s hardware dataflow, HYTE is on average 3.3× to 6.2× faster than the baselines, and performs very close to the exhaustively searched static optimal schemes. Most of the benefits are enabled by the flexible tiling parameter choices and the effective static scheduling, while our dynamic features in hardware can also boost performance for certain pathological cases when the static scheduler fails to find a good scheme. We also show the performance gains of HYTE are consistent across various sparse computation kernels and different hardware dataflows. The offline scheduling cost is minor even though it executes on the CPU, thanks to our effective sampling method.

**贡献：1、探索更多的tiling空间；2、基于建模的静态（offline）决策稀疏tiling方式；3、对shape进行动态tuning。4、集成所有技术到框架。**

We make the following contributions in this paper.
• We demonstrate that existing sparse accelerators have not extensively explored the full design space of tiling, including the tile size, tile shape, inter-tile iteration order, and buffer allocation policies.
• We propose a static offline scheduler for sparse accelerators, which uses lightweight sampling to adaptively identify near-optimal tiling schemes for various sparsity patterns.
• We design a hardware architecture for sparse accelerators, which supports dynamic tuning on the tile shape to ensure high buffer utilization, and efficiently manages the tiling metadata in both the off-chip memory and the on-chip buffer.
• We integrate the above techniques into a hybrid static-dynamic framework, which enables flexible and efficient tiling on sparse accelerators, and significantly outperforms previous approaches on diverse sparse matrices.

# BG & RW

## 稀疏张量代数

> **[图片提取文字 (image.png)]:**
> dominantly consist of elements with zero values. As commonly termed [35], elements in a tensor are referred to as points with a tuple of *coordinates*, e.g.,  $X_{i,j,k}$  at (i, j, k) in a 3D tensor X. Following previous studies [25, 38], we use the Einsum notation [8] to articulate operations on sparse tensors. For example, the widely used sparse-sparse matrix multiplication (SpMSpM) between tensors A
> 
> Tensors are multi-dimensional data arrays, and sparse tensors pre-
> 
> sparse-sparse matrix multiplication (SpMSpM) between tensors A  $(I \times K)$  and B  $(K \times J)$  is written as  $C_{i,j} = A_{i,k} \times B_{k,j}$  over (I,J,K). Here, k represents a contracted dimension that aggregates values across iterations (i, j, \*) to output  $C_{i,j}$ .
![image.png](HYTE%20Flexible%20Tiling%20for%20Sparse%20Accelerators%20via%20H/image.png)

稀疏张量使用压缩的存储格式，如坐标COO、压缩行CSR/列CSC、块压缩行等。

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> ## 适用场景
> 
> - 主要用来创建矩阵,因为coo\_matrix无法对矩阵的元素进行增删改等操作
> - 一旦创建之后,除了将之转换成其它格式的矩阵,几乎无法对其做任何操作和矩阵运算
![image.png](HYTE%20Flexible%20Tiling%20for%20Sparse%20Accelerators%20via%20H/image%201.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> - 对于矩阵第 ø 列, 我们需要先得到其非零元素行索引
>   - 由 indptr[0] = 0 和 indptr[1] = 1 可知, 第 0 列行有1个非零元素。
>   - 它们的行索引为 indices[0:1] = [0] , 且存放的数据为 data[0] = 8
>   - 因此矩阵第 ø 行的非零元素 csc[ø][ø] = 8
> - 对于矩阵第 3 列,同样我们需要先计算其非零元素行索引
>   - 由 indptr[3] = 4 和 indptr[4] = 6 可知, 第 4 行有2个非零元素。
>   - 它们的行索引为 indices[4:6] = [4, 6] , 且存放的数据为 data[4] = 1 , data[5]
>     - = 9
>   - 因此矩阵第 i 行的非零元素 csr[4][3] = 1 , csr[6][3] = 9
![image.png](HYTE%20Flexible%20Tiling%20for%20Sparse%20Accelerators%20via%20H/image%202.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> - 当 offsets[0] = 0 时,表示该对角线即是主对角线,相应的值为 [1,2,3,4,5]
> - 当 offsets[2] = 2 时,表示该对角线为主对角线向上偏移2个单位,相应的值为 [11, 12, 13, 14, 15]
> - 但该对角线上元素仅有三个 ,于是采用先出现的元素无效的原则
> - 即前两个元素对构造矩阵无效,故该对角线上的元素为 [13,14,15]
![image.png](HYTE%20Flexible%20Tiling%20for%20Sparse%20Accelerators%20via%20H/image%203.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> - 对于矩阵第 0 行,我们需要先得到其非零元素列索引
>   - 由 indptr[0] = 0 和 indptr[1] = 2 可知, 第 0 行有两个非零元素。
>   - 它们的列索引为 indices[0:2] = [0, 2] , 且存放的数据为 data[0] = 8 , data[1] = 2
>   - 因此矩阵第 0 行的非零元素 csr[0][0] = 8 和 csr[0][2] = 2
> - 对于矩阵第 4 行,同样我们需要先计算其非零元素列索引
>   - 由 indptr[4] = 3 和 indptr[5] = 6 可知, 第 4 行有3个非零元素。
>   - 它们的列索引为 indices[3:6] = [2, 3, 4] , 且存放的数据为 data[3] = 7 , data[4] = 1 , data[5] = 2
>   - 因此矩阵第 4 行的非零元素 csr[4][2] = 7 , csr[4][3] = 1 和 csr[4][4] = 2
![image.png](HYTE%20Flexible%20Tiling%20for%20Sparse%20Accelerators%20via%20H/image%204.png)

> **[图片提取文字 (image.png)]:**
> ## Block Sparse Row Matrix 分块压缩稀疏行格式
> 
> - 基于行的块压缩,与CSr类似,都是通过 data , indices , indptr 来确定矩阵 • 与csr相比,只是data中的元数据由0维的数变为了一个矩阵(块),其余完全相同
> - 块大小 blocksize
>   - 块大小 (R, C) 必须均匀划分矩阵 (M, N) 的形状。
>   - R和C必须满足关系: M % R = 0 和 N % C = 0
> 
> • 适用场景及优点参考csr
![image.png](HYTE%20Flexible%20Tiling%20for%20Sparse%20Accelerators%20via%20H/image%205.png)

> **[图片提取文字 (image.png)]:**
> ## Linked List Matrix 链表矩阵
> 
> - 使用两个列表存储非0元素data
> - rows保存非零元素所在的列
> - 可以使用列表赋值来添加元素, 如 lil[(0,0)] = 8
> 
> ![](_page_0_Figure_4.jpeg)
> 
> • lil[(0, -1)] = 4 : 第0行的最后一列元素为4
> 
> • lil[(4, 2)] = 5 : 第4行第2列的元素为5
![image.png](HYTE%20Flexible%20Tiling%20for%20Sparse%20Accelerators%20via%20H/image%206.png)

> **[图片提取文字 (image.png)]:**
> 于是我们引导出高效的CSF压缩方式(Compressed Sparse Fibers)或者称之为Fiber Tree<sup>+</sup>
> 
> ![](_page_0_Figure_1.jpeg)
> 
> Coordinate format.
> 
> ![](_page_0_Figure_3.jpeg)
> 
> CSF: conce與語: @树哥谈芯
> 
> ![](_page_0_Figure_5.jpeg)
> 
> 知乎 @树哥谈芯
![image.png](HYTE%20Flexible%20Tiling%20for%20Sparse%20Accelerators%20via%20H/image%207.png)

**压缩格式是将张量维度组织成fibers（坐标-数据）的分层结构，即坐标和非零数值的有序列表。**

**position通常指数据在压缩格式中的存储位置。**

To avoid ineffectual operations on zeros, sparse tensors are often stored in various compressed formats like the coordinate (COO) and compressed sparse row/column (CSR/CSC) formats, and their block variants like block CSR. These formats generally **organize tensor dimensions into a hierarchical structure with levels consisting of fibers** [35], which are sequential lists of coordinates and associated non-zero values. The term position indicates the actual storage location of a point within these formats, often differing from its coordinates due to the compression of zeros and null pointers.

## 稀疏加速器

3种主流的数据流使用不同矩阵乘的i、j、k的循环顺序：Inner Product（i>j>k）、Outer Product（k>i>j）、Gust（i>k>j）。

不同数据流有不同的数据复用方式。数据复用效率依照不相关维度（比如数据C和loop维度k）在多重loop中的排布决定，**比如：和张量C维度（i，j）不相关的loop维度k在维度i/j的内部，则说明Ci，j会在loop-k中被复用。**

设计运行时配置硬件来支持不同数据流，以适应输入张量的不同稀疏模式。

> **[图片提取文字 (image.png)]:**
> sidering SpMSpM with three dimensions (i, j, k), there are three mainstream dataflow choices adopted by recent sparse accelerators, namely Inner Product (IP) [4, 12, 31], Outer Product (OP) [14, 26, 41], and Gustavson's (Gust) [17, 20, 40], corresponding to the loop orders (outer  $\triangleright$  inner) of  $i \triangleright j \triangleright k$ ,  $k \triangleright i \triangleright j$ , and  $i \triangleright k \triangleright j$ , respectively. These dataflow schemes exhibit different data reuse friendliness for the three tensors A, B, C, as summarized in Table 1. Specifically, the effectiveness of reuse is contingent on the placement of the irrelevant dimensions, such as k for C, within the loop order. When the irrelevant dimension is at the inner loop, the data can be reused across these iterations, resulting in excellent reusability. Conversely, if the irrelevant dimension is at the outermost level, the whole data are repetitively scanned and would thrash the limited on-chip buffer. Recognizing these tradeoffs, several designs [20, 24]
> 
> supported multiple dataflows using runtime configurable hardware,
> 
> aiming to adapt to the diverse sparse patterns of the input tensor.
> 
> **Sparse dataflow.** Similar to the dense scenario [16, 39], the
> 
> dataflow of sparse accelerators can also be represented as multi-
> 
> level loop nests iterating over the multi-dimensional space. Con-
![image.png](HYTE%20Flexible%20Tiling%20for%20Sparse%20Accelerators%20via%20H/image%208.png)

> **[图片提取文字 (image.png)]:**
> Table 1: Data reuse tradeoffs of different dataflow schemes for sparse accelerators.
> 
> |                 | $\mathrm{IP}(i\triangleright j\triangleright k)$ | $OP(k \triangleright i \triangleright j)$ | Gust $(i \triangleright k \triangleright j)$ |
> |-----------------|--------------------------------------------------|-------------------------------------------|----------------------------------------------|
> | Reuse $A_{i,k}$ | Good                                             | Good                                      | Good                                         |
> | Reuse $B_{k,j}$ | Poor                                             | Good                                      | Poor                                         |
> | Reuse Ci i      | Good                                             | Poor                                      | Good                                         |
![image.png](HYTE%20Flexible%20Tiling%20for%20Sparse%20Accelerators%20via%20H/image%209.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> |                    | InP         | OutP      | ROW       |
> |--------------------|-------------|-----------|-----------|
> | Input reuse (B)    | Poor        | Excellent | Poor      |
> | Output reuse (C)   | Excellent   | Poor      | Good      |
> | Index intersection | Inefficient | Efficient | Efficient |
> | Psum granularity   | Scalar      | Matrix    | 知学领特行小家   |
![image.png](HYTE%20Flexible%20Tiling%20for%20Sparse%20Accelerators%20via%20H/image%2010.png)

**Buffer management.** 

**显式解耦的数据排布（EDDO）将计算单元和不同层Buffer解耦，每层buffer通过地址生成器尽快取数据，每层Buffer使用独立地址空间显式编址。**

能复用的张量缓存在片上以免片下访问，其余张量时间局部性差，流式加载和刷新。

Conventional hardware-managed caches are considered inefficient for specialized accelerators. The approach of explicit decoupled data orchestration (EDDO) [21, 27, 30] thereby emerges as a buffer management idiom tailored for accelerators. Specifically, EDDO decouples the computation units and the different levels of buffers, and fetches data as far in advance as possible at each buffer level using dedicated address generators. Each buffer is explicitly addressed with an independent address space, avoiding the overheads of cache tags.Depending on the dataflow, some tensors are accessed in irregular patterns. They should be buffered and reused in the on-chip buffers to reduce expensive accesses to the off-chip memory. Other tensors may exhibit simple streaming patterns with little temporal locality. They only need small buffer space.

## 稀疏加速器的tiling

片上存储有限，因此需要tiling；tiling影响tile内和tile间的数据复用，影响片外访问次数，因此需要tiling策略。

**tile是迭代计算空间的子空间（片上每次执行的计算），tile shape被每维的坐标范围定义，tile size是tile中非零值个数。tile size表示计算和存储的数据量。**

**当Buffer空间能放下tile size时，能最大化tile内的数据复用。**

tiling可能产生对其余张量/tile的重复访问、对部分和的合并开销和复杂的数据访问模式（加载、片上flow和存储）。

大部分稀疏方式中，每个维度的fibers需要适应tile size进行分段（segment），分段的fiber引入额外元数据。

To further improve data reuse, tiling becomes a promising technique for sparse accelerators. **A tile is a logically continuous sub-space of the full iteration space (𝐼, 𝐽, 𝐾), representing a subset of the overall computations. The tile shape is defined as its coordinate spread of each dimension**, denoted as 𝑇𝑖,𝑇𝑗,𝑇𝑘. **The tile size is the number of non-zero points in it.** By restricting the **tile size to be no larger than the on-chip buffer capacity, we can maximize data reuse** within the tile before moving to the next tile. On the other hand, tiling also incurs repetitive accesses to other tensors and/or additional partial result merging cost, and complicates data access patterns. Notably, with **most sparse formats, the fibers of a dimension would need to be segmented according to the tile shape, and these fiber segments further increase the metadata overheads.**

**稀疏张量的tiling分为两类，坐标tiling和位置tiling。**

坐标tiling在每个维度上划分相同的坐标范围，简化计算时的坐标匹配，因为两个tile要么坐标范围相同要么不会在某一维重叠。由于不同tile稀疏性不同，但占用一致的buffer空间，导致buffer使用率低或溢出。

位置tiling按照非零数据的个数划分，但每个tile稀疏性不同，坐标范围不对齐，导致实现难度高。

ref:[Accelerating Sparse Data Orchestration via Dynamic Reflexive Tiling]

ref:[Tailors:Accelerating Sparse Tensor Algebra by Overbooking Buffer Capacity]

Tiling for sparse tensors typically follows two categories: **coordinate tiling and position tiling** [25, 38]. **Coordinate tiling divides data with the same coordinate spans along each dimension, which simplifies coordinate matching during the computation as two tiles either have exactly the same coordinate range or do not overlap along a certain dimension**. However, the tiles may have different sizes due to the varying local sparsity, causing potential buffer underutilization or overflow. On the other hand, **position tiling divides data into tiles with the same size based on the actual data amounts in the specific format**, but the resultant unaligned coordinate ranges between tiles are quite challenging to manage.

**大部分设计使用坐标tiling来简化硬件控制，tile的非零值个数和分配的buffer容量（估计的tile size）不匹配，导致buffer使用率低，需要动态和灵活的tiling策略：**

**Tailor通过预采样稀疏性（稀疏比）估计每个tile（坐标范围）的tile size，基于估计的tile size确定存储空间；划分出部分tile，其估计tile size超过分配空间（多估一些），即静态tiling；**

**DRT和HARP在运行时调整tile shape来调整估计的tile size，直到占满分配到空间，即动态tiling。**

**State-of-the-art tiling techniques for sparse accelerators.** Currently, most existing designs adopt coordinate tiling to simplify hardware control. At the same time, they also recognize the inefficiency of mismatched data size and buffer capacity, and introduce more dynamic and flexible tiling strategies [19, 25, 38]. For example, **Tailors [38] adopted a speculative strategy to determine the tile size, by pre-sampling the data sparsity and allowing a small portion of tiles (e.g., 10%)** to overbook the buffer capacity. On the other hand, **DRT [25] and HARP [19] both used dynamic approaches, adjusting the tile size at runtime to fully utilize the buffer even with varying data sparsity characteristics.**

Tailors根据估计tile size等于或超过10%Buffer空间来静态划分每个tile（coordinate spread），DRT和HARP按照用完Buffer动态调整出每个tile（coordinate spread）。

**Tailors优先沿着loop-k划分出tile，之后是loop-j、loop-i，以此划分每个tile。**

**DRT采用运行时贪心算法来划分出tile，按k、i、j的顺序依次扩展tile，直到Buffer完全占满。实现上，以micro-tile为最小扩张单位。**

**HARP使用OP数据流，只沿着loop-i作tiling，将loop-i分成小的伪tile（暂不执行），执行时根据buffer使用统计动态合并伪tile成大tile进行计算。**

Table 2 summarizes the details of these three designs from various perspectives. First, **the tile size of Tailors is only statically determined with 10% overbooking, while both DRT and HARP dynamically decide the tile size to fully utilize the buffer capacity based on the current local data sparsity**. 

To construct the tile with a concrete shape along each dimension, **Tailors prioritizes expanding along the contracted dimension *k* in order to maximize the reuse for tensor *C*, followed by *j* and *i*.** 

**DRT employs an online greedy algorithm to select the tile shape. It iteratively grows each dimension in the order of *k*,*i*,*j*, until the buffer is fully occupied. Consequently, the tile shape in DRT resembles a cube with similar spans along all dimensions. To facilitate such flexible tiling across all dimensions under a compressed format, DRT needs to first preprocess the original tensor into micro-tiles (of 32 × 32), as the smallest unit for tiling.** 

**HARP, in contrast, focuses on tiling along *i* only, which is specialized for its OP dataflow and aims to improve data reuse of tensor *C*. It segments dimension *i* into small pseudo-tiles, and dynamically merges them into super-tiles during execution, based on specific buffer usage statistics.**

> **[图片提取文字 (image.png)]:**
> Tailors [38]
> 
> DRT [25]
> 
> HARP [19]
> 
> HYTE
> 
> (ours)
> 
> ## Decision
> 
> Static
> 
> Dynamic
> 
> Dynamic
> 
> Static
> 
> + dynamic
> 
> Tile size
> 
> Fit 90% tiles
> 
> Exact fit
> 
> Exact fit
> 
> if beneficial
> 
> ![](_page_0_Figure_2.jpeg)
> 
> Tile shape
> 
>  $k \rightarrow j \rightarrow i$ 
> 
> Table 2: Tiling scheme comparison between state-of-the-art designs and ours.
> 
> Inter-tile order
> 
> *j* first
> 
> *i* first
> 
> *i* first
> 
> Flexibly
> 
> scheduled
> 
> ![](_page_0_Figure_3.jpeg)
> 
> **Buffer allocation** 
> 
> Unspecified ratios among A, B, C
![image.png](HYTE%20Flexible%20Tiling%20for%20Sparse%20Accelerators%20via%20H/image%2011.png)

**For the processing order between tiles, Tailors goes along dimension *j* first, while DRT follows ExTensor [12] to go along *i* first. HARP also goes along *i* first for inter-tile execution.** Finally, these three designs are also equipped with **separate dedicated buffers with different capacities for different tensors.** DRT and HARP require the tensor tiles to be fully buffered in the corresponding buffers (buffering), while Tailors could additionally support streaming data in and out of a small space when the tiles overflow (streaming).

# Motivation

**稀疏tiling的优化方式**

Despite the above recent efforts making tiling more efficient on sparse accelerators [19, 25, 38], they are still **limited in the abilities to fully adapt to various complicated sparse data distributions.** In this section, we discuss several critical aspects of sparse tensor tiling that exhibit drastically different optimal decisions in various scenarios, motivating even more flexible approaches.

## Tile Parameters

The most **critical parameters in tiling are the tile size (number of non-zero points) and shape (coordinate spread along each dimension)**. Making a tile small enough to fit in the on-chip buffer improves data locality of the current tensor, but it also results in more tiles, requiring repetitive fetches of the other tensors irrelevant to the tiling dimension to multiply with each of these tiles. Figure 1a illustrates the performance impact of tile sizes with different sparse matrices from SuiteSparse [7] doing self-multiplication using the Gust dataflow. We use an on-chip buffer of 16 MB. The **horizontal axis lists different tile sizes as the overflow ratios compared to the buffer capacity**, e.g., “1” means the tile size is 2× larger than the tile size with no overflow. As shown in the figure, **ML_Laplace achieves the best performance at the tile size without overflow; mouse_gene excels at a tile size of 2×; and ldoor and cit-Patents prefer much larger tile sizes close to no tiling.** These different behaviors are due to the diverse data patterns. For example, **cit-Patents is quite sparse, with only 4 non-zeros per column on average, indicating limited data reuse opportunities. If applied a large tiling factor beyond 4, the repetitive access cost would outweigh the reuse benefit**. The star mark on each line shows the choice of Tailors [38] with 10% overbook tiles, which is suboptimal for most matrices shown.

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> (a) Tile size. Shown as the overflow ratio compared to the buffer capacity. Stars are Tailors' choices.
![image.png](HYTE%20Flexible%20Tiling%20for%20Sparse%20Accelerators%20via%20H/image%2012.png)

**根据估计tile size等于或超出部分Buffer容量进行tiling不一定最优，当重复加载开销超过数据复用收益时，比如稀疏分布导致低复用。**

**Takeaway 1 (tile size): Using tile sizes that match the buffer capacity or follow a fixed overflow ratio is not always optimal, e.g., when repetitive accesses outweigh tiling reuse.**

Furthermore, **even with the same tile size, the concrete shape of the tile may also significantly affect performance.** Figure 1b shows the off-chip memory access amounts with different tile shapes while the tile’s coordinate size (product of the two dimensions) remains unchanged. In the almost-**diagonal matrix TSOPF_FS_b300_c3, the generated 𝐶 is relatively small, and thus tiling 𝑘 and fetching 𝐶 repetitively would be the best**. However, in the **power-law graph kron_g500-logn18, 𝐶 dominates the accesses, favoring only tiling 𝑗 and fetching 𝐴 repetitively**. Finally, the **structured mycielskian16 matrix resides in between, where the best performance is achieved by balancing the tile shape along both dimensions.**

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> (b) Tile shape. A label of "aT,bT" means to tile dimension k by a factor of a and tile j by b.
![image.png](HYTE%20Flexible%20Tiling%20for%20Sparse%20Accelerators%20via%20H/image%2013.png)

**相同的tile size，不同tile shape的片外访问次数不同。**

**Takeaway 2 (tile shape): With the same tile size, the best coordinate spread in each dimension would depend on the specific tensor data patterns.**

## Control Schemes

Prior designs have used either purely static [38] or purely dynamic [19, 25] approaches to determine their tile sizes and shapes, each with certain drawbacks. **Purely static approaches can only optimize for the average case with heuristic parameter choices.** For example, the 10% overbooked tiles in Tailors [38] may not always lead to the best performance, as shown in Figure 1a. **Purely dynamic tiling, on the other hand, would either require significant metadata overheads to make the data format suitable for tiling (e.g., micro-tiles [25]), or limit to simple tiling schemes that lose efficiency** [19], as discussed in Section 3.1.

**纯静态tiling中“经验”参数确定的tile size和shape难以一直得到最优或次优解；纯动态tiling中动态调整tile size和shape引入许多元数据辅助，过于昂贵和复杂，可能限制tiling的方式。**

**Takeaway 3 (static vs. dynamic): Purely static tiling is less optimal while purely dynamic tiling is too expensive. A combination of the two may be desired.**

**数据复用一般完全复用一种张量，将其tile缓存在片上，其余张量tile进行加载、丢弃、重新加载。**

这里存在理论上更优的复用调度策略。比如计算（1，2，3，4）和（A，B，C，D）的全连接计算，完全复用一个张量tile的加载次数是20次/16edge，如果按照b4-a1-b1-a2-b2-a3-b3-a4-b4-a3-a2-a2-b3-a1-b2-a4-b1-a3的调度顺序则是17次/16edge。

**不同数据复用方式复用的张量不同，流式加载不同张量tile，需要不同张量的inter-tile order调度。**

Another critical control scheme is the inter-tile execution order, which affects the reuse of the fetched tiles. Note that **this inter-tile order differs from the intra-tile order. The latter is usually designed by the fixed dataflow implemented by a specific hardware chip (Table 1) for computations within a single tile, while the former is across different tiles.** Again because of the diverse sizes and sparsity degrees, in different cases we would need to **prioritize the reuse of one specific tensor over the others, thus requiring support of different inter-tile orders**. In Figure 1c, both matrices use the Gust dataflow and are evenly tiled along each dimension. However, reusing 𝐶 by putting the irrelevant 𝑘 dimension at the innermost (i.e., 𝑘 first) is more critical in the matrix filter3D, but less beneficial for rail2586.

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> (c) Inter-tile order. "i/j/k first" means to put i/j/k at the innermost loop of the inter-tile level.
![image.png](HYTE%20Flexible%20Tiling%20for%20Sparse%20Accelerators%20via%20H/image%2014.png)

**不同的稀疏模式有不同的reuse方式，对应不同的tile间执行次序。**

**Takeaway 4 (inter-tile order): The hardware should support configurable inter-tile orders that could adapt to different tensor data patterns.**

## Buffer Management

**不同复用方式将不同的张量tile缓存在Buffer以复用，因此Buffer需要灵活分配来缓存不同张量。**

动态tiling需要大量的元数据，包含稀疏压缩引入的格式地址数据，也包含tiling的关联数据（tile size运行时确定）。元数据和非零数据的比例可能很大，也可能很小，和稀疏性有关。

稀疏张量分类：1、结构性稀疏（行、列、对角），元数据开销小；2、非结构性稀疏，非零数据分散随机，元数据开销大。

Finally, to realize the flexible tiling schemes and the relevant control as mentioned above, the **hardware buffers must be accordingly enhanced**. First, because **each of the different inter-tile orders mainly buffers one of the three operand tensors and stream the others, the buffer capacity allocation among them should be flexible.** Previous designs with rigid capacity partitioning (Table 2) cannot support such diverse requirements. In addition, with **dynamic tiling, the metadata size and management cost could be substantial, not only including the original compressed format indexing structure, but also the new metadata associated with tiling (e.g., the begin and end positions of a tiled fiber segment)**. Actually, **the ratio between metadata and data varies significantly for different matrices, with the mean and variance as 0.27 and 0.51 across our evaluated matrices. Sparse and heavily tiled matrices could have excessive metadata, e.g., up to 3.2× of actual data in kron_g500-logn18.** **Dense or mildly tiled matrices, such as nd24k and TSOPF_FS_b300_c3, only incur < 0.02× overheads. Thus it would be more efficient to also use the same buffer to flexibly store and manage the metadata in tandem with the data.**

**Buffer容量需要灵活分配，以适应不同复用方式对应的不同加载过程；Buffer需要高效管理，以管理元数据和非零数据的存储。**

**Takeaway 5 (buffer management): The buffer capacity needs to be flexibly allocated and efficiently managed among different tensors as well as between data and metadata.**

# HYTE Overview

Based on the key takeaways in Section 3, we propose HYTE, a hybrid static-dynamic framework for flexible and efficient tiling on sparse accelerators. HYTE follows previous work [19, 25, 38] to adopt coordinate tiling and **focuses on selecting the best tiling configuration**. It is able to support various **tiling schemes and identify the best one according to the specific data tensor characteristics and hardware dataflow implementations**, therefore enabling fully adaptive execution with high performance and efficiency.

offline和online类似cpu和gpu的关系，cpu offline交付任务，gpu online执行任务。

**线下静态分析稀疏模式，决定初始执行状态（tile size→tile shape、tile顺序）和分配Buffer空间。**

**先根据容量确定tile size，根据tile size和稀疏性并最大化数据复用确定tile shape和tile顺序。**

Table 2 compares HYTE with the previous designs. HYTE uses novel hybrid static-dynamic approaches to achieve the above flexibility and efficiency (Takeaway 3). **At the static offline phase, HYTE relies on a lightweight scheduler to analyze the data patterns and to decide an initial execution scheme**. It extensively yet efficiently explores the design space, and determines **optimized tile sizes, shapes, and inter-tile orders** (Takeaways 1, 2, 4). Notably, it may **choose larger tile sizes or even disable tiling** if it finds that the data reuse benefit does not justify the repetitive data fetch cost. It will also **select the tile shape and the inter-tile order** that result in the best overall data reuse based on the tensor size and sparsity information. During such exploration, **the unknown information such as the output tensor sparsity is estimated through effective yet lightweight sampling.** Moreover, based on the inter-tile reuse analysis, the scheduler also **flexibly allocates the buffer space to each tensor to match their reuse requirements, e.g., using most buffer capacity for reused tensor tiles while keeping minimum space for streaming ones (Takeaway 5).** Overall, such an initial scheme serves as a reasonably near-optimal schedule for hardware execution.

> **[图片提取文字 (image.png)]:**
> Inter-tile order: a permutation of i, j, k
> Buffer allocation: S<sub>A</sub>, S<sub>B</sub>, S<sub>C</sub>
> 
> • Initial tile shape:  $T_i, T_j, T_k$ 
> 
> Figure 2: Initial tiling scheme generated by HYTE scheduler.
![image.png](HYTE%20Flexible%20Tiling%20for%20Sparse%20Accelerators%20via%20H/image%2015.png)

**运行时按照静态确定顺序加载tile到分配的空间，并动态调整shape来利用Buffer。**

At the **dynamic online phase, the hardware processes each tile one after one following the given inter-tile order, while the data within each tile are fetched to its statically allocated buffer space.** Due to the need of flexible tiling schemes, HYTE requires more complex **metadata for bookkeeping the storage positions of tiled fiber segments**. HYTE enables coordinated management of data and metadata (Takeaway 5) in both the on-chip buffer and the off-chip memory, to facilitate efficient intra-tile and inter-tile execution, respectively. In addition, it also applies **dynamic tuning based on the real-time, local data patterns, to ensure maximum buffer utilization**. **Specifically, the statically determined tile size may underutilize or overflow the buffer when the local data region is sparser or denser than expected. The hardware would dynamically adjust the tile shapes to better match the buffer capacity (Takeaways 1, 2).** Such dynamic tuning is implemented with cheap hardware counters and simple heuristic rules, in contrast to complex designs in previous work [25]. This is **sufficient because the initial scheme is already close to optimal, and the dynamic tuning is only needed in occasional cases to correct the small estimation errors.**

**线下scheduler接收输入张量的格式、大小和稀疏性（非零比），在内部评估结果张量C的信息。**

**scheduler也接收tile计算方式（reuse pattern）、计算吞吐、片上容量和DDR带宽。**

**假设tile内数据流固定为OP，对应tile内loop顺序是k→i→j或k→j→i，进行线下调度。**

Workflow and interface. **The offline scheduler of HYTE takes in the information for input tensors 𝐴 and 𝐵, including their formats, sizes, and sparsities (percents of non-zeros). The scheduler internally estimates the information for the result tensor 𝐶.** The hardware specifications are also provided, such as the **intra-tile dataflow (IP/OP/Gust), computation throughput, SRAM buffer capacity, and DRAM bandwidth**. We consider **the intra-tile dataflow as a fixed hardware setting**, because different sparse dataflows require quite different hardware implementations. It is not difficult to extend our scheduler to search for flexible dataflow choices [20, 24].

**scheduler生成初始的tiling状态，指导硬件执行。tiling状态包含默认tile shape（硬件可tuning）、tile执行次序、每个张量占据的Buffer容量。**

The scheduler eventually generates an initial tiling scheme as in Figure 2, which is then used to guide the execution on the hardware. The scheme specifies the default tile shape to use (upon which the hardware may further fine-tune), the inter-tile order to traverse the tiles, as well as the allocated buffer capacity for each tensor. In the subsequent Sections 5 and 6, we introduce the detailed designs of the scheduler (offline phase) and the hardware architecture (online phase) of HYTE, respectively.

# Scheduling Algorithms

**The offline scheduler of HYTE aims to determine a relatively optimized initial tiling scheme** for the given sparse data. This scheme should include the parameters in Figure 2. Similar to existing schedulers for dense computations [13, 16, 28, 39], the scheduler **searches the configuration space of different tile shapes, inter-tile orders, etc., assesses each scheme with a hardware cost model, and identifies the best scheme with the minimum cost** (i.e., the best performance).

> **[图片提取文字 (image.png)]:**
> ## **Algorithm 1:** Overall workflow of HYTE scheduler. // Sample and estimate
> 
> - 1  $S_I \leftarrow$  Sample a fraction sp of values from 0 to I;
> - 2  $S_I$  ← Sample a fraction sp of values from 0 to J;
> - 3 effMAC ← EstEffMAC( $A, B, S_I, S_I, sp$ ); 4 {nnzCTk} $_{T_k=1,2,4,...,K} \leftarrow \text{EstNnzCTk}(A, B, S_I, S_J, \text{sp, sk});$
> 
> 6 **foreach** s in PrunedTilingSchemeSpace() **do** 
> 
>  $c \leftarrow \text{CostModel}(s, \text{effMAC}, \text{nnzCTk}_{T_k});$ 
> 
> if  $c < c_{\min}$  then  $c_{\min} \leftarrow c$ ;  $s_{\min} \leftarrow s$ ;
> 
> 5  $c_{\min} = \infty$ ;  $s_{\min} = \bot$ ;
> 
> 9 return s<sub>min</sub>;
> 
> - // Search the tiling scheme with the minimum cost
![image.png](HYTE%20Flexible%20Tiling%20for%20Sparse%20Accelerators%20via%20H/image%2016.png)

**scheduler采样确定张量稀疏分布并评估延迟，tiling的评估延迟对比延迟仿真结果有平均15%、最高43%的偏差。**

The overall workflow is summarized in Algorithm 1.The key challenge that the scheduler needs to address is that, unlike the dense case where the computation load and data sizes are all directly known, **in the sparse scenario both the required computation amount and the output tensor size would depend on the input tensors' data distributions as well as their correlation in a complex way. HYTE proposes effective yet lightweight sampling and estimation methods(Algorithm 1 Lines 1 to 4;Section 5.1)to derive these required statistics without the need to investigate the full input tensors.** We find that such sampling and estimation may introduce 15% on average and up to 43% errors against the cycle-accurate simulation results, mostly occurring in irregular tensors. These inaccuracies can be fixed by the online dynamic fine-tuning phase.

**评估延迟前搜索tiling参数，先prune space，再通过Cost Model进行评估。**

To efficiently explore the search space and reduce the search cost, HYTE leverages several observations to **prune unnecessary and sub-optimal schemes (Line 6;Section 5.2).The remaining ones are then fed to the hardware cost model (Line 7;Section 5.3)for assessment to determine the best one.** We provide several example cases in Section 5.4 to show how the scheduler works.

## Sampling and Estimation

**effMAC ：effective MAC, 有效MAC次数.**

**nnzCTk ：non_zero access of tensor C in tiling Tk,按Tk in loop-K作Tiling后的access traffic size。**

To accurately assess the performance of each tiling scheme, we need to know the computation amount and the data sizes. Besides the easily known non-zero sizes of the input tensors, we mainly need to estimate two critical metrics: **effMAC as the expected effectual (i.e., non-zero)MAC number,** and **nnzCTk as the non-zero access traffic size of the output tensor C under each tiling factor Tk of dimension k when repetitive accesses are accounted for.**

**输出C的稀疏性取决于A、B的稀疏性，以及A、B间稀疏分布的关系。**

**采样A的列、B的行,采样比是sp。**

Note that both the effectual MACs and **the non-zero size of *C* depend not only on the sparsities of the two inputs *A* and *B*, but more importantly also on the correlation between their non-zero distributions.** Due to the complex relationship, we adopt sampling-based approaches to efficiently estimate these metrics. We **sample a small fraction (denoted as *sp*) of rows and columns from *A* and *B* (Algorithm 1, Lines 1 to 2).** If the compressed formats are compatible, e.g., getting rows/columns from CSR/CSC, the sampling is straightforward. If the formats mismatch, e.g., extracting a column from CSR, we sample a set of points uniformly from the entire matrix, and re-group these points into the desired rows/columns.

> **[图片提取文字 (image.png)]:**
> ## **Algorithm 2:** Estimating effMAC and nnzCTk.
> 
> 1 **function** ESTEFFMAC(
> $$A, B, S_I, S_J, \text{sp}$$
> ):  
> 2 |  $sum \leftarrow 0$ ;
> 
> for 
> $$k \leftarrow 0$$
>  to  $K$  do
> 
> $$szA \leftarrow Number of non-zeros in  $A[S_I][k]$ ;$$
> 
> $$szA \leftarrow \text{Number}$$
> 
> return
> 
> $$szB \leftarrow Number of$$
> 
> $$szB \leftarrow \text{Number of non-zeros in } B[k][S_J];$$
> 
> $$B \leftarrow \text{Number o}$$
> 
> $$sum += szA \times szB;$$
> 
> $$k$$
> ][ $S_{\tau}$ ]:
> 
> ![](_page_0_Picture_14.jpeg)
![image.png](HYTE%20Flexible%20Tiling%20for%20Sparse%20Accelerators%20via%20H/image%2017.png)

在loop-k中统计采样后A列和B行中非零值个数szA、szB，szA*szB是其外积中有效MAC个数，sum(szA * szB)/sp2得到对整体MAC次数的估计值。

sp是行/列中的采样比例。对szA*szB求和来估计整体MAC，是假设外积矩阵之间稀疏分布不同而不需要累加操作。

With the sampled matrices, our estimation methods are summarized in Algorithm 2. For effMAC, we **iterate over the contracted dimension *k*, and multiply the non-zero sizes of the sampled *A* column and *B* row pair, to get the MAC number of their outer-product. The total estimated MAC number is the sum scaled by *sp*2.**

**数据库中估计自然连接result size的近似算法来估计张量C的非零个数nnzC或所有tile C的非零值总数nnzCTk。**

**算法思路：**

$E_{ST}N_{NZ}C$函数用于计算张量C作tiling后的$N_k$个tile的非零值总数，对张量C中每个Tile t内的(i，j)对进行Hash排序、计算Hash Value并分别保存到R[t]，之后dedup & merge到$R_{all}$（global var），最后计算估计量$sk/(v_{sk}\cdot sp^2)$。

$E_{ST}N_{NZ}CT_K$调用函数$E_{ST}N_{NZ}C$ ，分别计算不同$T_k$值作Tiling时的所有tile C的非零值总数，即不同$T_k$下的nnzCTk。

**算法优化：**

不同Tiling不影响(i，j)对的Value计算和$R_{all}$中vsk计算，因此Tk=K时通过Hash、sk-th in $R_{all}$筛选出的(i，j)对是其他Tk筛选时的必要条件（v≤vsk）：

先计算Tk=K，即no tiling时筛选出的(i，j)对保存在queue中；遍历保存的(i，j)对，构建Tk=K/Nkmax时的Nkmax个去重的小顶堆R[Nkmax]。由于loop-k的tiling是2的幂次，对Tk*2大小的tiling，将Tk对应的R[Nk]中相邻小顶堆去重并合并（dedup & merge）得到R[Nk*2]。

注：算法中loop-k中对t的赋值应该添加一个下取整符号，因为t是整数tile index。

> **[图片提取文字 (image.png)]:**
> To estimate nnzCTk, we borrow from previous techniques [2, 3]. We first consider nnzCTk for  $T_k = K$ , i.e., no tiling. As in Algorithm 2 Lines 15 to 17, for each non-zero element pair A[i][k] and B[k][j] in the sampled matrices, we add (i, j) into a min-heap with the value  $h(i, j) = (h_1(i) - h_2(j))$  mod 1, where  $h_1$  and  $h_2$  are two hash functions to fixed-point numbers, and "mod 1" extracts the fractional part of the result. The same (i, j) pairs for different k are deduplicated in the heap. Then  $\operatorname{nnzCTk}_{T_k=K}$  can be estimated as  $\operatorname{sk}/(v_{\operatorname{sk}} \cdot \operatorname{sp}^2)$ , where  $v_{\operatorname{sk}}$  is the sk-th smallest h(i, j) value.
> 
> The estimation of nnzCTk for other values of  $T_k$  can be done similarly as in Algorithm 2. However, naively doing so would incur  $O(\log K)$  invocations of the above procedure (Lines 23 to 24). We apply two optimizations. First, we use the  $v_{\rm sk}$  value obtained in the  $T_k = K$  case to filter out most of the (i,j) pairs, and only process the pairs within the sk-th smallest. These pairs are kept in a queue during the  $T_k = K$  invocation. Second, we only iterate these (i,j) pairs once, to build  $N_{k,\rm max} = 4096$  deduplicated heaps  $R[N_{k,\rm max}]$  for  $T_{k,\rm min} = K/N_{k,\rm max}$ . For  $T_{k\times 2}$ , we merge each pair of adjacent heaps of  $T_k$  with deduplication. We do this recursively to calculate nnzCTk for all  $T_k$  values. With these optimizations, the extra cost
> 
> of estimating for other  $T_k$  values can be reduced to 10% of that for
> 
>  $T_k = K$  for most matrices.
![image.png](HYTE%20Flexible%20Tiling%20for%20Sparse%20Accelerators%20via%20H/image%2018.png)

> **[图片提取文字 (image.png)]:**
> ```
> function EstNNzCT\kappa(A, B, S_I, S_I, sp, sk):
>          function EstNnzC(N_k, A, B, S_I, S_I, sp, sk):
>  9
>                R[N_k] \leftarrow A list of empty min-heaps, each of size sk;
> 10
>                for k \leftarrow 0 to K do
> 11
>                      t \leftarrow \frac{k}{K/N_L};
>                                                                                 // tile index
> 12
>                      a \leftarrow \text{Non-zero indices of } A[S_I][k] \text{ sorted by } h_1();
> 13
>                      b \leftarrow \text{Non-zero indices of } B[k][S_J] \text{ sorted by } h_2();
> 14
>                      foreach i, j in a, b do
> 15
>                            v \leftarrow (h_1(i) - h_2(j)) \mod 1; // fractional part
> 16
>                          Add ((i, j), v) to R[t], if not already existing;
> 17
>                R_{\text{all}} \leftarrow \text{Deduplicate and merge sort } R[N_k];
> 18
>                v_{\rm sk} \leftarrow The sk-th smallest value in R_{\rm all};
> 19
>                sum \leftarrow \sum_{i} \left| \left\{ v \le v_{\rm sk} | v \in R[t] \right\} \right|;
> 20
>                return \frac{sum}{v_{\rm sk} \cdot {\rm sp}^2};
> 21
>          nnzCTk \leftarrow [];
> 22
>          foreach T_k do
> 23
>                nnzCTk.Append(EstNnzC(K/T_k, A, B, S_I, S_I, sp, sk));
> 24
>          return nnzCTk;
> 25
> ```
![image.png](HYTE%20Flexible%20Tiling%20for%20Sparse%20Accelerators%20via%20H/72c0c494-624a-4ec5-9ef6-04b6372d7019.png)

**超参数sp和sk的默认值是*sp*=1/*N* and *sk*=*N*(*I*, *J*, or *K*)。sp是采样比，sk表示近似算法中计算到第sk小的value；*sp*⋅*sk*≤1。**

The above estimation involves two hyperparameters, whose default values are chosen as ***sp*=1/*N* and *sk*=*N* (where *N* represents the corresponding dimension *I*, *J*, or *K*), to strike a balance between estimation accuracy and computational cost. This follows the theoretically proved suggestion of *sp*⋅*sk*≤1** [2]. The empirical time cost is further evaluated in Section 8.4.

**ref：[Better Size Estimation for Sparse Matrix Products]**

**估计思路：构造目标值的估计量，设计兼顾时空复杂度、近似概率和近似比的近似算法。**

> **[图片提取文字 (image.png)]:**
> ## 2 Our algorithm
> 
> The task is to estimate the size z of  $Z = \pi_{ac}(R_1 \bowtie R_2)$ . We may assume that attribute values are  $\mathcal{O}(\log n)$ -bits integers, since any domain can be mapped into this one using hashing, without changing the join result size with high probability. When discussing I/O bounds, B is the number of such integers that fits in a disk block. In linear expected time (by hashing) or sort(n) I/Os we can cluster the relations according to the value of the join attribute b. By initially eliminating input tuples that do not have any matching tuples in the other relation we may assume without loss of generality that  $z \geq n/2$ .
> 
> In what follows, k is a positive integer parameter that determines the space usage and accuracy of our method. The technique used is to compute the kth smallest value v of a hash function h(x,y), for  $(x,y) \in Z$ . Analogously to the result by Bar-Yossef et al.  $\P$  we can then use  $\tilde{z} = k/v$  as an estimator for z.
> 
> Our main building block is an efficient iteration over all tuples  $(x, \cdot, y) \in R_1 \bowtie R_2$  for which h(x, y) is smaller than a carefully chosen threshold p, and is therefore a candidate for being among the k smallest hash values. The essence of our result lies in how the pairs being output by this iteration are computed in expected linear time. We also introduce a new buffering trick to update the sketch in expected amortized  $\mathcal{O}(1)$  time per pair. In a nutshell, each time k new elements have been retrieved, they are merged using a linear time selection procedure with the previous k smallest values to produce a new (unordered) list of the k smallest values.
![image.png](HYTE%20Flexible%20Tiling%20for%20Sparse%20Accelerators%20via%20H/image%2019.png)

> **[图片提取文字 (image.png)]:**
> **Theorem 1.** Let  $R_1(a,b)$  and  $R_2(b,c)$  be relations with n tuples in total, and define  $z = |\pi_{ac}(R_1 \bowtie R_2)|$ . Let  $\varepsilon$ ,  $0 < \varepsilon < \frac{1}{4}$  be given. There are algorithms that run in expected  $\mathcal{O}(n)$  time on a RAM, and expected  $\mathcal{O}(\operatorname{sort}(n))$  I/Os in the cache-oblivious model, and output a number  $\tilde{z}$  such that for  $k = 9/\varepsilon^2$ :
> 
> - $-\mathbf{Pr}[(1-\varepsilon)z<\tilde{z}<(1+\varepsilon)z]\geq 2/3 \ when \ z>k^2, \ and$
> - $\mathbf{Pr}[\tilde{z} < (1+\varepsilon)k^2] \ge 2/3 \text{ when } z \le k^2.$
> 
> Observe that for  $\varepsilon > 4/\sqrt[4]{z}$  we will be in the first case, and get the desired  $1 \pm \varepsilon$  approximation with probability 2/3. The error probability can be reduced from 1/3 to  $\delta$  by the standard technique of doing  $\mathcal{O}(\log(1/\delta))$  runs and taking the median (the analysis follows from a Chernoff bound). We remark that this can be done in such a way that the  $\mathcal{O}(\log(1/\delta))$  factor affects only the RAM running time and not the number of I/Os. For constant relative error  $\varepsilon > 0$  we have the following result:
> 
> **Theorem 2.** In the setting of Theorem 1, if  $\varepsilon$  is constant there are algorithms that run in expected  $\mathcal{O}(n)$  time on a RAM, and expected  $\mathcal{O}(\operatorname{sort}(n))$  I/Os in the cache-oblivious model, that output  $\tilde{z}$  such that  $\operatorname{\mathbf{Pr}}[(1-\varepsilon)z < \tilde{z} < (1+\varepsilon)z] = 1 - \mathcal{O}(1/\sqrt{n})$ .
> 
> The error probability can be reduced to  $n^{-c}$  for any desired constant c by running the algorithms  $\mathcal{O}(c)$  times, and taking the median as above.
![image.png](HYTE%20Flexible%20Tiling%20for%20Sparse%20Accelerators%20via%20H/image%2020.png)

**估计的本质是采样并计算估计量，优化估计量的计算过程来降低算法时空复杂度、缩小近似比$\varepsilon$和提高近似概率p。**

> **[图片提取文字 (image.png)]:**
> ## 2.1 Finding pairs
> 
> (see Algorithm 1 for pseudocode).
> 
> For  $\mathcal{B} = \pi_b(R_1) \cup \pi_b(R_2)$  and each  $i \in \mathcal{B}$  let  $\mathcal{A}_i = \pi_a(\sigma_{b=i}(R_1))$  and  $\mathcal{C}_i = \pi_c(\sigma_{b=i}(R_2))$ . We would like to efficiently iterate over all pairs  $(x, y) \in \mathcal{A}_i \times \mathcal{C}_i$ ,  $i \in \mathcal{B}$ , for which h(x, y) is smaller than a threshold p. This is done as follows
![image.png](HYTE%20Flexible%20Tiling%20for%20Sparse%20Accelerators%20via%20H/image%2021.png)

R1/2是关系矩阵的二维表（row | col），矩阵乘法可通过$Z=\pi_{ac}(R_1\bowtie R_2)$进行表达。

i是关系矩阵R1列、R2行可能连接的坐标，Ai是A中若干列，Bi是B中若干行，**Ai×Ci是矩阵乘中外积过程的“潜在”非零位置（a，c）。**

> **[图片提取文字 (image.png)]:**
> **Definition 1.1** (Pairwise independent hash functions). A family  $\mathcal{H} = \{h : U \to R\}$  is
> 
> said to be pairwise independent, if for any two distinct elements  $x_1 \neq x_2 \in U$ , and any two
> 
>  $\Pr_{h \in \mathcal{H}}[h(x_1) = y_1 \text{ and } h(x_2) = y_2] = \frac{1}{|R|^2}.$ 
> 
> (possibly equal) values  $y_1, y_2 \in R$ ,
![image.png](HYTE%20Flexible%20Tiling%20for%20Sparse%20Accelerators%20via%20H/image%2022.png)

> **[图片提取文字 (image.png)]:**
> For a set U, let  $h_1, h_2: U \to [0; 1]$  be hash functions chosen independently at random from a pairwise independent family, and define  $h: U \times U \to [0; 1]$  by
> 
> $$h(x,y) = (h_1(x) - h_2(y)) \mod 1.$$
> 
> algorithm still outputs all pairs with hash value at most p.
> 
> eliminating the  $\log k$  factor implied by the heap data structure.
> 
> property we will utilize later. Now, conceptually arrange the values of h(x, y) in an  $|\mathcal{A}_i| \times |\mathcal{C}_i|$  matrix, and order the rows by increasing values of  $h_1(x)$ , and the columns by increasing values of  $h_2(y)$ . Then the values of h(x, y) will decrease (modulo 1) from left to right, and increase (modulo 1) from top to bottom.
> 
> It is easy to show that h is also a pairwise independent hash function — a
> 
> For each  $i \in \mathcal{B}$ , we traverse the corresponding  $|\mathcal{A}_i| \times |\mathcal{C}_i|$  matrix by visiting the columns from left to right, and in each column t finding the row  $\bar{s}$  with the smallest value of  $h(x_{\bar{s}}, y_t)$ . Values smaller than p in that column will be found in rows subsequent to  $\bar{s}$ . When all such values have been output, the search proceeds in column t + 1. Notice, that if  $h(x_{\bar{s}}, y_t)$  was the minimum value in column t, then the minimum value in column t + 1 is found by increasing  $\bar{s}$  until  $h(x_{\bar{s}}, y_{t+1}) < h(x_{(\bar{s}-1) \mod |\mathcal{A}_i|}, y_{t+1})$ . We observe that the algorithm is robust to decreasing the value of the threshold p during execution, in the sense that the
> 
> ## 2.2 Estimating the size
> 
> tain the k smallest hash values in an unordered buffer instead of using a heap data structure (lines  $\boxed{14}$ - $\boxed{18}$  in Algorithm  $\boxed{1}$ ). In this way we are able to maintain the k smallest hash values in constant amortized time per insertion in the buffer,
> 
> While finding the relevant pairs, we will use a technique that allows us to main-
> 
> Let S and F be two unordered sets containing, respectively, the k smallest hash values seen so far (all, of course, smaller than p), and the latest up to k elements seen. We avoid duplicates in S and F (i.e., the sets are kept disjoint) by using a simple hash table to check for membership before insertion. Whenever
> 
> |F| = k the two sets S and F are combined in order to obtain a new sketch S. This is done by finding the median of  $S \cup F$ , which takes  $\mathcal{O}(k)$  time using either deterministic methods (see [8]) or more practical randomized ones [12]. At each iteration the current kth smallest value in S may be smaller than
> 
> the initial value p, and we use this as a better substitute for the initial value of p. However, in the analysis below we will upper bound both the running time and the error probability using the initial threshold value p.
![image.png](HYTE%20Flexible%20Tiling%20for%20Sparse%20Accelerators%20via%20H/image%2023.png)

> **[图片提取文字 (image.png)]:**
> ## 3 Distinct sketches
> 
> A well-known approach to size estimation in, described in generality by Gibbons [II] and explicitly for join-project operations in [I0]3], is to sample random subsets  $R'_1 \subseteq R_1$  and  $R'_2 \subseteq R_2$ , compute  $Z' = \pi_{ac}(R'_1 \bowtie R'_2)$ , and use the size of Z' to derive an estimate for z. This is possible if  $R'_1 = \sigma_{a \in S_a}(R_1)$ , where  $S_a \subseteq \pi_a(R_1)$  is a random subset where each element is picked independently with probability  $p_1$ , and similarly  $R'_2 = \sigma_{c \in S_c}(R_2)$ , where  $S_c \subseteq \pi_c(R_2)$  includes each element independently with probability  $p_2$ . Then  $z' = |Z'|/(p_1p_2)$  is an unbiased estimator for z. The samples can be obtained in small space using hash functions whose values determine which elements are picked for  $S_a$  and  $S_c$ . The value |Z'| can be approximated in linear time using the method described in section 2 if the samples are sorted — otherwise one has to add the cost of sorting. In either case, the estimation algorithm is I/O-efficient.
> 
> Below we analyze the variance of the estimator z', to identify the minimum sampling probability that introduces only a small relative error with good probability. The usual technique of repetition can be used to reduce the error probability. Recall that we have two relations with  $n_1$  and  $n_2$  tuples, respectively, and that  $n_a$  and  $n_c$  denotes the number of distinct values of attributes a and c, respectively. Our method will pick samples  $R'_1$  and  $R'_2$  of expected size s from each relation, where  $s = p_1 n_1 = p_2 n_2$  is a parameter to be specified.
> 
> **Theorem 4.** Let  $R'_1$  and  $R'_2$  be samples of size s, obtained as described above. Then  $z' = |\pi_{ac}(R'_1 \bowtie R'_2)|/(p_1p_2)$  is a  $1 \pm \varepsilon$  approximation of  $z = |\pi_{ac}(R_1 \bowtie R_2)|$  with probability 5/6 if  $z > \beta$ , where  $\beta = \frac{14}{\varepsilon^2} \left( \frac{n_c n_1 + n_a n_2}{s} \right)$ . If  $z \leq \beta$  then  $z' < (1 + \varepsilon)\beta$  with probability 5/6.
![image.png](HYTE%20Flexible%20Tiling%20for%20Sparse%20Accelerators%20via%20H/image%2024.png)

> **[图片提取文字 (image.png)]:**
> **Algorithm 1** Pseudocode for the size estimator. 1: **procedure** DISITEMS $(p, \varepsilon)$ 2:  $k \leftarrow \lceil 9/\varepsilon^2 \rceil$  $F \leftarrow \emptyset$ 3: for  $i \in \mathcal{B}$  do 4:  $x \leftarrow \mathcal{A}_i$  sorted according to  $h_1$ -value 5:  $y \leftarrow C_i$  sorted according to  $h_2$ -value 6: 7:  $\bar{s} \leftarrow 1$ 8: for t := 1 to  $|\mathcal{C}_i|$  do while  $h(x_{\bar{s}}, y_t) > h(x_{(\bar{s}-1) \mod |\mathcal{A}_i|}, y_t)$  do  $\triangleright$  Find  $\bar{s}$  s.t.  $h(x_{\bar{s}}, y_t)$  is min. 9:  $\bar{s} \leftarrow (\bar{s} + 1) \mod |\mathcal{A}_i|$ 10: 11: end while 12:  $s \leftarrow \bar{s}$  $\triangleright$  Find all s where  $h(x_s, y_t) < p$ 13: while  $h(x_s, y_t) < p$  do  $F \leftarrow F \cup \{(x_s, y_t)\}$ 14: if |F| = k then  $\triangleright$  Buffer filled, find smallest hash values in  $S \cup F$ 15:  $(p, S) \leftarrow \text{Combine}(S, F)$ 16:  $F \leftarrow \emptyset$ 17: end if 18:  $s \leftarrow (s+1) \mod |\mathcal{A}_i|$ 19: 20: end while end for 21: end for 22: 23:  $(p,S) \leftarrow \text{Combine}(S,F)$ 24: if |S| = k then **return** " $\tilde{z} = \frac{k}{p}$  and  $\tilde{z} \in [(1 \pm \varepsilon)z]$  with probability 2/3" 25: 26: else **return** " $\tilde{z} = k^2$ ,  $z \le k^2$  with probability 2/3" 27: end if 28: 29: end procedure 30: **procedure** Combine (S, F) $v \leftarrow \text{Rank}(h(S) \cup h(F), k)$  $\triangleright$  Rank $(\cdot, k)$  returns the kth smallest value 31:  $S \leftarrow \{x \in S \cup F | h(x) \le v\}$ 
> 
> 32:
> 
> 33:
> 
> return (v, S)
> 
> 34: end procedure
![image.png](HYTE%20Flexible%20Tiling%20for%20Sparse%20Accelerators%20via%20H/image%2025.png)

**ref：[Balanced Hashing and Efficient GPU Sparse General Matrix-Matrix Multiplication]**

## Search Space Pruning

**loop transformation是多重loop的次序排列，其构造/生成逻辑是：**

**1、片上数据流决定tile内数据流（IP、OP和Gust），对应内层3-loop。**

**2、确定tile之间的处理顺序，对应外层3-loop。**

**3、tile内数据流计算子矩阵，tile间数据流计算分块矩阵得到大矩阵。**

**tile内复用是指缓存张量供当前tile内的后续计算使用，后续计算由数据流定义（内层3-loop）；**

**tile间复用是指缓存张量供后续tile的计算使用，后续tile由tile执行次序定义（外层3-loop）。**

As in Figure 2, the search space consists of three categories of parameters: **tile shapes, inter-tile orders, and allocated buffer capacities.** Below we analyze the candidate choices for each category and prune the unnecessary ones.

**划分尺度是2pow值；**

**对tile内数据流最外层loop对应的原始loop作tiling不改变数据访问和计算顺序。比如片上OP数据流时不必要对loop-K作tiling；**

**tile内数据流最外层loop关联的张量复用性强，如OP数据流的张量A和B，对A、B作tiling只是把tile内对A和B的数据复用，转移到tile之间A和B的数据复用，不会增加复用机会。**

**比已有tile shape更小的tile无需考虑（类似分支定界）。**

First, for the **tile shapes, we only consider power-of-two sizes along each dimension**, resulting in (log*N*)^3 choices where *N*∼*I*,*J*,*K*. Furthermore, we notice that it is not beneficial to **tile the outermost loop of the intra-tile dataflow, e.g., *i* in Gust and IP, or *k* in OP (Table 1).** This is because data are already sequentially processed along this dimension and **tiling it will not affect the access and computation flow.** We hence reduce the space to (log*N*)^2. We can do **further pruning if a tile shape is smaller than another feasible** (i.e., fitting in the buffer size) shape along all dimensions, as it would surely underutilize the buffer capacity.

**tile间loop（外层3-loop）中只有最内层loop影响tile间的tile复用，比如 i▹j▹k和 j▹i▹k的tile次序对tile C复用相同。**

**tile间顺序只需要考虑最内层loop是什么，即i▹j▹k是完全复用tile C（i&j）、k▹j▹i是完全复用tile B（k&j）、i▹k▹j是完全复用tile A（i&k）。**

tile次序是基于完全复用/缓存一种张量tile，其余张量tile流式加载的模式。比如N个tile和N个tile全连接计算的加载次数是（N+1）*N，但复用模式的理论最低加载次数是N*N+1。   

Second, we note that **only the choice of the innermost loop in the inter-tile order affects the inter-tile data reuse characteristics.** Recall that given an inter-tile order such as *i*▹*j*▹*k*, the tensor irrelevant to the innermost *k* dimension, i.e., *C*, is reused. Exchanging the order of the outer two loops does not change this result. In summary, we only need to consider 3 instead of all 6 inter-tile orders.

**对Buffer空间分配，考虑复用tile内张量和复用tile间张量的分配策略。**

比如片上OP数据流，对应内层3-loop是**k▹i▹j顺序，tile内复用张量C需要大量缓存空间。不同tile间次序对应外层3-loop的不同次序，在tile间缓存并复用不同张量。**

**Buffer分配只缓存tile内张量或分区缓存，则张量复用模式是4种：1）只满足片上数据流的张量复用；2）~4）Buffer分区，同时满足tile内和tile间的张量复用，tile间有三种复用方式。**

Third, for buffer allocation, we consider both **inter-tile reuse and intra-tile reuse**. First, the fixed intra-tile dataflow could exhibit a poor access pattern on one tensor (e.g., *C* for OP in Table 1) that needs to be buffered. We choose to **always satisfy this requirement of the hardware dataflow to ensure its efficiency.** Second, as described above, the **inter-tile order designates one tensor that could enjoy inter-tile reuse and would request buffer space to keep its tile.** Consequently, we consider two choices: **either only using the buffer for intra-tile reuse, or dividing the buffer space between inter-tile and intra-tile reuse.** The first choice adds 1 more case to the 3 inter-tile orders, making it 4 in total. Unbuffered tensors use the streaming mode with minimum space.

In summary, the total search space is no larger than 4×(log*N*)^2, which is just a few hundred different schemes and can be explored in a short time in practice, as further shown in Section 8.4.

## Cost Model

**执行时间建模：块的总数×每块的执行时间；**

**每块的执行时间等于PE time, DRAM time（片外内存）, SRAM（片上缓存） time的最大值；**

**PE time = effMAC/throughput，DRAM/SRAM time=total nnz*CTk*  /BW。**

For each candidate scheme from Section 5.2, we use a relatively straightforward cost model to evaluate its performance, similar to previous work [13, 16, 28, 39]. **The total time is modeled as the total number of tiles (as *I*/*Ti*×*J*/*Tj*×*K*/*Tk*) multiplied by the per-tile execution time, which is max{PE time,DRAM time,SRAM time}. Here, PE time = effMAC/throughput, and DRAM/SRAM time is the total access amounts of all tensors (e.g., nnz*CTk* for *C*) divided by the corresponding bandwidth.**

**SRAM的访问次数受到tile内数据流和tiling模式影响，**假设张量格式排列好，进行下列分析：

**1）不作tiling**

**OP数据流上张量A、B、C的访问数据量是nnz*A*, nnz*B*（读取一次即完成相关计算）, 和effMAC（外积 &求和对应先读后写SRAM）；**

**Gust数据流上张量访问数据量是nnz*A*, effMAC（C中每行数据计算需要依次读取B中所有行并写回SRAM，读/写数据量都是effMAC/2）, and nnz*C*（过程数据缓存/流动在PE中累加，最终结果写入SRAM一次）；**

**IP数据流上张量B中每列需要加载I次，总计I*nnzB。**

**2）对张量无关的loop进行tiling**

**SRAM访问按照tile数量增加，OP数据流进行Tk的Tiling，访问数据量是nnz*A*×*J*/*Tj* and nnz*B*×*I*/*Ti，*张量C是运行时累加产生的，每个tile的SRAM访问数据量包含在nnz*CTk*（列表）中 。**

It is worth noting that the **total DRAM/SRAM access amounts are sometimes non-trivial to calculate.** For SRAM, **the accesses are affected by both the intra-tile dataflow and the tiling scheme.** We assume the tensors are already in the desired sparse formats of the hardware dataflow. First, we consider the case without tiling. For example, for OP, the access amounts of matrices *A*, *B*, and *C* are nnz*A*, nnz*B*, and effMAC, respectively; and for Gust, they become nnz*A*, effMAC, and nnz*C*. **Repetitive accesses may be needed, e.g., IP accesses *B* for *I* times.** Then, if tiling is applied to the irrelevant dimension of a tensor, its SRAM accesses are amplified by the number of tiles, e.g., nnz*A*×*J*/*Tj* and nnz*B*×*I*/*Ti*. However, ***C* needs special treatment because it is generated and accumulated on the fly**; the access amount is estimated as nnz*CTk* in Section 5.1.

**DRAM访问数据量考虑Buffer分配和数据访问模式（Buffer还是Stream）。**

**Stream的数据访问模式下，DRAM的流量等于SRAM的流量，因为使用才加载，用完即丢弃。**

**Buffer的数据访问模式下，若tile size远小于分配空间，数据访问量按照复用次数减少。hit rate是Buffer中存储数据和分配空间的比，即实际tile size和估计tile size的比。**

The DRAM access amounts need to further consider the buffer allocation and access mode of each tensor. **In the streaming mode, the DRAM traffic is equal to the SRAM traffic. In the buffering mode, if the tile size is smaller than the allocated buffer space, the traffic is reduced by the reuse times.** Otherwise, the **hit rate** is the ratio between the two.

**每个tile中fiber段的实际存储地址离散且无规律。**

**SRAM、DRAM的访问数据量考虑元数据，比如CSR-format *A* tile of *Ti* ×*Tk* has *Ti* fiber segments。**

Besides the data, we also need to consider the **metadata access cost.** With the flexible tiling schemes supported in HYTE, we need to maintain a non-negligible amount of metadata (Section 6.2), e.g., **to specify the actual storage positions of the fiber segments in the current tile, which are irregular and differ significantly from the regularly tiled coordinates.** When a tile is highly sparse, the metadata overhead can be substantial compared to the data access cost, e.g., reading/writing the begin position of a fiber segment vs. only a few non-zeros in this segment. **To account for their accesses, for each tile we calculate how many individual fiber segments it has, e.g., a CSR-format *A* tile of *Ti* ×*Tk* has *Ti* fiber segments.** This determines the size of metadata. Their access counts follow those of the corresponding data.

## **Case Studies**

**稀疏矩阵样本：mouse_gene、dielFilterV2real，稀疏分布：**

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> ![](_page_0_Figure_1.jpeg)
> 
> (a) mouse\_gene
> 
> (b) dielFilterV2real
> 
> Figure 3: Non-zero distributions of two example matrices that prefer no tiling and extensive tiling, respectively.
![image.png](HYTE%20Flexible%20Tiling%20for%20Sparse%20Accelerators%20via%20H/image%2026.png)

**按sp采样、小于sk-th Value求和，估计$effMAC$、$nnzCTk_K$（no tiling）、$nnzCTk_{K/128}$；**

mouse_gene的$nnzC$是比输入矩阵中的非零值数量大18倍，$nnzCTk_{K/128}$比$nnzC$大8倍；

dielFilterV2real的$nnzC$是比输入矩阵中的非零值数量大4倍，$nnzCTk_{K/128}$近似于$nnzC$。

**mouse_gene数据密度高，按照K/128进行tiling比no tiling大幅增加数据访问量，因此no tiling；**

**mouse_gene数据很稀疏，按照K/128进行tiling的数据访问量较低，因此extensive tiling；**

> **[图片提取文字 (image.png)]:**
> We illustrate how the HYTE scheduler works on two example matrices: mouse\_gene and dielFilterV2real, whose non-zero distributions are shown in Figure 3. The mouse\_gene matrix has dimensions  $45101 \times 45101$  with 14,506,196 non-zeros, while dielFilterV2real has dimensions  $1157456 \times 1157456$  with 24,848,204 non-zeros.
> 
> By sampling sp =  $1/\sqrt{N}$  = 0.005 of mouse\_gene and tracking the top sk =  $\sqrt{N}$  = 212 hash values, we estimate effMAC, nnzCTk<sub>K</sub>, nnzCTk<sub>K/128</sub> as 7,442,882,727, 262,241,518, and 2,084,207,396, respectively, while the actual values are 7,971,580,000, 237,833,954, and 2,065,359,984. For dielFilterV2real, sampling with sp = 0.0009 and sk = 1075 yields the estimated values of 449,092,928, 95,012,185, and 125,005,246, compared to the actual 435,260,000, 105,679,996, and 121,610,583. The errors are only about 5% to 10%.
> 
> Notably, with mouse\_gene, nnzC is  $18 \times$  larger than the non-zero size of the input matrix, and nnzCTk $_{K/128}$  is another  $8 \times$  larger than nnzC. In contrast, in dielFilterV2real, nnzC is only  $4 \times$  the input, and nnzCTk $_{K/128}$  approximates nnzC.
> 
> After the tiling space exploration, the HYTE scheduler decides not to tile dimension k for mouse\_gene. This is due to its relatively dense distribution and the high nnzC value, which would result in significant redundant accesses to C after tiling (i.e., nnzCTk $_{K/128}$  vs. nnzCTk $_{K}$ ). Conversely, the sparsity and low nnzCTk values of dielFilterV2real favor extensive tiling of k.
> 
> Our scheduler is general and can easily discover more patterns. Matrices with similar characteristics to mouse\_gene — such as kron\_g500-logn18, ship\_001, and human\_gene — show similar variance and power-law distribution. Large and structured matrices — like 1door and fem\_hifreq — perform comparably to dielFilterV2real. Additional patterns are presented in Section 8.1.
![image.png](HYTE%20Flexible%20Tiling%20for%20Sparse%20Accelerators%20via%20H/image%2027.png)

# **Hardware Architecture**

## Overview

Figure 4 illustrates the overall hardware architecture of HYTE.

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 4: Hardware architecture of HYTE. The tiling controller and the accessors at each buffer level are newly added.
![image.png](HYTE%20Flexible%20Tiling%20for%20Sparse%20Accelerators%20via%20H/image%2028.png)

**多PE、一层SRAM和片外DRAM的加速器模型。**

**忽略片上数据流的特化支撑模块，因为和Buffer层的tiling策略无关，考虑片上数据流对性能的影响。**

Without loss of generality, we **assume a multi-PE accelerator with one level of SRAM buffer and the off-chip DRAM memory**, similar to most prior designs [20, 25, 38, 40]. Here in the figure we omit any dedicated units to **support a specific intra-tile dataflow, such as index intersectors for IP and partial sum mergers for Gust and OP, since they do not affect our tiling designs at the global buffer level. But we consider their performance impact in the evaluation.**

**tiling controller控制tiling模式，tensor accessor负责将稀疏tensor分块后的fiber segment加载到Buffer，并管理元数据。两个模块只涉及buffer控制和片外数据访问，不影响片上数据流。**

HYTE mainly introduces two new hardware components which are highlighted in Figure 4: **the tiling controller that controls the overall tiling scheme, and the accessor of each tensor (e.g., “A/B/C acc”) that is in charge of fetching the tiled fiber segments into the buffer and managing the corresponding metadata.** Note that these modifications are only to the logic for buffer control and data access, without altering the PE datapaths.

**tiling controller加载tiling模式，由线下scheduler静态决定。tiling模式中的tile间次序和tile形状来决定下一个处理的tile。**

**tensor accessors将下一个处理的tile关联的张量加载到片上Buffer，根据buffer分配模式分配buffer空间，buffer分配模式由线下scheduler静态决定。**

The high-level workflow is as follows. **The global tiling controller first loads the initial tiling scheme statically determined by the offline scheduler.** **The inter-tile order and the tile shapes are used by the tiling controller to determine which tiles to process next after each inter-tile iteration** (Section 6.2). **This information is sent to the accessors, who fetch the corresponding tiles of the multiple tensors into the buffer, and manage the buffer space according to the buffer allocation in the offline scheduled scheme (Section 6.3).**

tensor accessor基于**Buffers，**改变控制和访问粒度，从单元素改为fiber segment的开始和结束坐标。**accessor设计难点在于高效管理元数据，以便通过fiber segment的开始和结束坐标来支持任意tile间次序的执行。**

Our accessor design is extended from **Buffets** [30], with the main difference as **changing the control and access granularity from a single element to a fiber segment with specified begin/end coordinates.** A specific design challenge is to effectively manage the metadata, so that with an arbitrary inter-tile order, we can derive the positions (i.e., the actual storage locations) of the fiber segments given their begin/end coordinates from the tiling controller. Note that **previous tiling designs have overlooked this issue, either only supporting tiling along fixed dimensions with simple metadata [19, 38], or relying on expensive preprocessing [25].** Section 6.2 describes how we maintain the necessary metadata in the memory across tiles, while Section 6.3 discusses how the metadata within a tile are managed in coordination with the tensor data.

**硬件动态微调tile shape，修正静态scheduler对tile size估计误差导致的tiling对Buffer的underutilized或overflow。** 

**accessor设置计数器统计运行时特征，tiling controller使用这些特征动态调整tile shape。**

Finally, HYTE supports **dynamic tuning of tile shapes at runtime in hardware** (Section 6.4), in order to correct the estimation errors of the static scheduler and to better adapt to the local sparse patterns. A few **hardware counters are added to the accessor to collect the runtime statistics, and the tiling controller uses such information to dynamically adjust the tile shape using a simple model.**

## Inter-Tile Management

**tiling controller向accessor发送信号：Begin、T、Change，指示下一个加载的tile；**

Begin是tile的起始坐标（ $Begin_{i},Begin_{j},Begin_{k}$）；

T是当前tile的形状（ $T_{i},T_{j},T_{k}$），可动态调整；

Change指示tile的坐标变化行为（$Change_{i},Change_{j},Change_{k}$）：0表示开始/结束坐标和上一次迭代保持不变；1表示开始/结束坐标增加对应的$T_{i}/T_{j}/T_{k}$，以去到下一个segment；2表示置0.

Following the statically scheduled inter-tile iteration order, **the tiling controller tells each accessor which tile to fetch next through the following control signals** (Figure 4 top right). (1) $Begin_i, Begin_j, Begin_k$ as the begin coordinates. (2) $T_i, T_j, T_k$ as the current tile shape, which could **differ from the statically scheduled one after dynamic tuning**. (3) $Change_i, Change_j, Change_k$, which can take three values: **0 indicates the begin/end coordinates remain the same as the last iteration; 1 means moving to the next segment by increasing the begin/end coordinates by the corresponding T value; and 2 resets to 0. For example, with an inter-tile order of  *i*▹*j*▹*k*** **, the tiling controller would send 0, 0, 1, until the last innermost iteration that sends 0, 1, 2.**

**accessor需要维护元数据，比如fiber segments的位置：加载fiber segment后，自增位置寄存器，直到遇到tile的边界。**

**元数据存储数量取决于tile间次序和张量压缩格式。**

如果tile间次序是***i*▹*j*▹*k，***A和B都按CSR存储，那么去往下一个tile时，A保持当前tile的Ti个位置，因为迭代顺序和fiber格式匹配，即**tile A在k维度增加时需维护Ti个位置自增（loop-k）。**但B需要维护tile所在列的**K个位置自增（loop-k）**，**因为tile B在k维度增加的访问方式在CSR存储格式下需要K行。**

**元数据的访问开销在Cost模型中参与Tiling的决策。**

Besides **fetching the data fiber segments, the accessor also maintains the necessary metadata, i.e., the positions of the fiber segments. As the accessor fetches the fiber segment, it sequentially increments the position, until encountering a coordinate exceeding the tile boundary.** The current position would become the begin position of the next fiber segment. **The exact amount of metadata needed to store depends on both the inter-tile order and the compressed format of the tensor.** For example, assume the ***i*▹*j*▹*k*** inter-tile order and both tensors 𝐴 and 𝐵 are in CSR. As shown in **Figure 5(a), 𝐴 only needs to keep 𝑇𝑖 positions of the current tile, because the iteration direction matches its fiber format. But 𝐵 would need to keep 𝐾 positions for the whole tile column. Because HYTE supports flexible inter-tile orders, both cases must be considered.** Due to the various demands and the potentially large size (e.g., 𝑂(𝐾) for 𝐵 above), **HYTE keeps these metadata in the memory**. The **cost of reading/writing these metadata from/to memory has been considered in our cost model in Section 5.3. If the cost is too high, HYTE can flexibly and automatically adapt to less aggressive tiling.**

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 5: Example hardware behaviors with  $i \triangleright j \triangleright k$  inter-tile order and Gust dataflow. (a) Current tiles' sparse patterns, and metadata in the memory. (b) Streaming and buffering modes, and coordination of data and metadata in the buffer. (c) Counters for dynamic tuning of tile shapes.
![image.png](HYTE%20Flexible%20Tiling%20for%20Sparse%20Accelerators%20via%20H/image%2029.png)

## Intra-Tile Management

**accessor根据指定的坐标范围（tile shape）和开始位置，加载张量的fiber segments到静态分配的Buffer空间。每个张量支持两种加载模式，Buffer for Reuse和Stream and Refetch。**

**Buffer填入fiber segment时，表示segment location的元数据也被记录。**

With the **specified coordinate ranges and the begin positions, the accessor fetches the fiber segments of the tensor into its statically allocated buffer space.** We support two modes for each tensor [38], **buffering if the tile should be buffered for reuse, or streaming otherwise.** Figure 5(b) shows the two modes. In this example, we buffer tensor 𝐵’s tiles while streaming 𝐴. We **manage the data and metadata in coordination**, **each growing from one end of the buffer space towards the middle until met.** When filling in the data of a fiber segment, the accessor also records its begin location in the buffer as the metadata. For example, in 𝐴’s buffer, **at time 0 and 1 the first fiber segment (𝑎, 𝑏) is in, with a metadata 0 denoting its location. The second segment is empty. The third segment 𝑐 starts at location 2. Similarly for 𝐵, e.g., the last segment 𝑠 is at location 3. The difference between two adjacent metadata entries is the (non-zero) size of the corresponding fiber segment.**

**Stream模式下，head和occupancy寄存器记录环形元数据Buffer的开头和已用大小。用完的数据直接覆盖。**

For the streaming mode, we follow the Buffets-like circular buffer design [30], with **a head and an occupancy register denoting the head of the circular metadata buffer and its size.** In Figure 5(b), at time 3 the 𝐴 buffer is full. Then we evict one fiber segment (i.e., 𝑎 and 𝑏) as well as its metadata at time 4. Now the head moves by 1 and the occupancy reduces by 1, and we have available space to read in the last segment 𝑑 at time 5.

**Buffer模式下，tile data超过buffer space时不覆盖原有数据，bypass该Buffer。**

In the buffering mode, if the tile data exceed the allocated buffer space, e.g., due to a locally dense region, we do not evict previous data since they are expected to be reused. **The future data of this tile will just bypass the buffer.** This is a result of inaccurate static estimation that misses local sparse pattern variations. We use dynamic tuning in Section 6.4 to alleviate it.

## Dynamic Tuning

Due to variations in the distribution of non-zeros across different tiles, **the tile that should be buffered may sometimes overflow or underutilize the statically allocated buffer space.** We may need to **decrease or increase the tile size in these cases, respectively.** HYTE applies dynamic tuning for this purpose. The accessor collects a few runtime statistics and transfers them to the tiling controller, which adjusts the tile shapes based on certain simple rules.

**计数器统计加载tile切4份后每份的非零个数，基于统计结果计算9种tile shape的Buffer的hit rate；**

**9种tile shape是当前tile shape在2个维度上分别缩小一半、不变和放大一倍的叠加变换。**

nnztotal是tensor中非零值总数，(Tj*Tk)表示计算所有该shape下tile的非零值总数，E[nnztile]是调整后tile中非零值个数的估计值，min(E[nnztile],bufsize)计算bufsize限制下的tile的非零值个数；

通过cnt计算的所有tile非零值总数和张量非零值总数的比是hit rate。

**hit rate公式的含义是tiling设置的buffer size能否容纳tile size；若buffer size可容纳任意tile size，则hit rate是100%，同时tile size需尽可能大。**

Specifically, **when an accessor fetches a tile, e.g., a 𝐵 tile of shape 𝑇𝑘×𝑇𝑗 in Figure 5(c), we use four counters, cnt00, cnt01, cnt10, cnt11, to count the numbers of non-zero elements in the four quadrants of the tile.** This gives us a more accurate view of the local non-zero distribution of the accessed tile. We then use these counter values to **estimate the buffer hit rates for nine potentially adjusted tile shapes, in which each of the two dimensions can increase by 2×, decrease by 2×, or keep the same spread. That is, the new tile shape 𝑇𝑘′×𝑇𝑗′ follows 𝑇𝑘′ ∈ {𝑇𝑘/2, T𝑘, 2𝑇𝑘}, 𝑇𝑗′ ∈ {𝑇𝑗/2, 𝑇𝑗, 2𝑇𝑗}.** The average buffer hit rate is estimated as

> **[图片提取文字 (image.png)]:**
> ## $T_k' \times T_i' \times \min(\mathbb{E}[\text{nnz}_{\text{tile}}], \text{bufsize})$ nnz<sub>total</sub>
![image.png](HYTE%20Flexible%20Tiling%20for%20Sparse%20Accelerators%20via%20H/image%2030.png)

where **nnztotal is the total tensor size, and nnztile is the number of non-zeros in the adjusted tile, estimated through the counters.** 

**通过原来tile的四分统计，估计改变形状后的tile在bufsize限制下的tile size。运行时，张量非零值总数nzztotal和其余nzztile未知，因此hit rate无法通过公式计算；但是，对tensor来说，每个tile size都能在buffer中放下，tile的hit rate为100%，否则是二者之比。**

**For example, the adjusted tile 𝑇𝑘′ × 𝑇𝑗′ = 𝑇𝑘/2 × 2𝑇𝑗 would have
nnztile = 1/2 × (min(2×cnt00 + 2×cnt01, bufsize) + min(2×cnt10 + 2×cnt11, bufsize)). If the buffer size is larger than the tile size, the hit rate is 100%; otherwise it is the ratio between the two.**

图5中4*4的张量B包含4个非零值，使用2*2的tile shape来tiling，Buffer size是1，则：

平均1个tile包含1个非零值，等于bufsize，因此hit rate估计值是100%。但每个tile受bufsize限制，实际hit rate是50%。

Using the **dynamic counters is more accurate than the static uniform estimation.** Suppose we have a small buffer that can hold only one element. In Figure 5 if we use a tile shape of 2 × 2 for 𝐵, each tile contains one element on average, and thus we would assume 100% hits in the static scheduler. However, the actual hit rate is 1/4 × (min(3, 1) + min(0, 1) + min(0, 1) + min(1, 1)) = 50%.

**选择hit rate提高超过5%的tile shape；考虑tile shape和tensor size的对齐，保证恰好切分，而不多出几块。**

We empirically decide that **if the best hit rate among the nine shapes is more than 5% better than the existing shape, we will adjust the tile shape.** However, this adjustment may not be immediately applied. **If the current inter-tile iteration direction is different from the dimension to be adjusted, changing the tile shape would result in misaligned tiles compared to previous ones. Therefore, the adjustment is delayed until the next inter-tile iteration that requires modification of that specific dimension.** For example, in Figure 5(a) we are iterating along 𝑘 in 𝐵. Changing 𝑇𝑘 does not affect the inter-tile iteration, but shrinking or extending 𝑇𝑗 would make the end coordinates of 𝑗 misaligned, complicating the processing when we start a new 𝑗 iteration.

# **Methodology（实现）**

**和baseline相同的配置（TOps、Freq、SRAM、DDR）；**

**开发cycle-wise的模拟器，显式建模关键部件的行为，面对不同输入进行模拟。**

We compare HYTE with three previous sparse accelerators that support tiling, namely Tailors [38], DRT [25], and HARP [19]. The characteristics of these four designs are summarized in Table 2. All the designs **use the same default hardware configuration, with 32 MAC PEs running at 1 GHz, and a 4 MB global SRAM buffer realized in 32 banks. The off-chip DRAM uses four DDR4 channels with 68 GB/s in total.** These configurations mostly follow the prior works [20, 33, 40]. We later assess performance with different PE counts and buffer capacities. We assume the PE array follows the Gust dataflow by default, but also study the performance under other dataflow schemes like IP and OP. **We implement a cycle-accurate simulator in C++ to measure the performance of the above designs when processing different matrix data.** Our simulator accurately captures the accesses of in dividual non-zero matrix elements, in order to reflect the actual influence of the input data pattern. This is more detailed than previous models [37]. In particular, **various key components like the index intersector, index selector, and partial sum merger for the IP, Gust, and OP dataflow schemes are explicitly modeled.** **The real input sparse matrix is fed to them to determine which data elements are actually accessed and processed in the PEs, and thus affect the compute and memory timing results.** The simulator is open-sourced at **https://github.com/tsinghua-ideal/HYTE-sim.**

**global tiling controller、tensor accessors的RTL级别实现。**

In addition, we **implement the RTL designs of the key components introduced by HYTE, including the global tiling controller and the tensor accessors.** We synthesize them using **Synopsys DC** on the TSMC 28 nm technology. We use CACTI 7.0 [5] to model the SRAM buffers. The area numbers are listed in Table 3. We see that HYTE incurs minor area cost of 3.7%, where the chip area is dominated by the large SRAM buffer.

**scheduler部署在CPU。**

**Our static scheduler, including the sampling process, runs on an Intel Xeon Gold 6248R processor at 3 GHz, compiled with g++ -O3.**

**benchmark采用SuiteSparse Matrix Collection，进行不同的矩阵操作。**

We select **real-world sparse matrices from the SuiteSparse Matrix Collection [7] as our datasets. These matrices are diverse, with varying densities (from 0.0006% to 0.356%), non-zero sizes (from 1.5M to 25M), and sparsity patterns.** Tiling is irrelevant for smaller matrices with our 4 MB buffer. For better comparison, we include several matrices used in the baseline papers, e.g., filter3D, web-Google, pwtk, kkt_power, kron_g500-logn18, cit-Patents. We mainly evaluate the performance of SpMSpM with self-multiplication of square matrices, i.e., $𝑆 × 𝑆$, following prior studies. In addition, we also test several other irregular sparse kernels, including (1) $𝐹^𝑇 × 𝐹$ with a tall-skinny sparse matrix 𝐹 ; (2) $𝐹 × 𝐷$ where 𝐷 is a random dense matrix, i.e., SpMM; (3) $𝐹^𝑇 ×𝑆$ as one iteration of multi-source breadth-first search (MS-BFS) in graph analytics [1, 6], where 𝑆 is the graph and 𝐹 represents the initial source nodes [25].

实验如何做？

动态tiling：张量内不同tile大小、shape不同；不同张量不同tiling。

实验，idea是否相似。