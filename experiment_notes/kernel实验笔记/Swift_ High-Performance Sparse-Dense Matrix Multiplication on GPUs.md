## Swift: High-Performance Sparse-Dense Matrix Multiplication on GPUs

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  提出Swift，面向GPU上SpMM (Sparse-Dense Matrix Multiplication, C=A×B)的双路径kernel系统。核心设计：(1) Sparsity-based sorting：按稀疏矩阵A每列NNZ升序排序列，同步重排稠密矩阵B的对应行，使warp内线程处理相邻列时访问B中连续地址。(2) Blocking：按warpSize=32将排序后A划分为regular block（列宽=32的完整block）和irregular block（不足32列或长列残留），生成blkPtr/blkColIdx/value/rowIdx/positionIdx/offsetIdx（regular）和irrPtr/irrValue/irrRowIdx/colIdxIndex/blkStart/blkStop（irregular）两套索引结构。(3) Dual-kernel路径：regular kernel以32×8 thread block处理8个sparse block+B中连续32列，warp lane读sparse value/rowIdx后访问B连续位置，乘积写入shared memory，用positionIdx/offsetIdx做segment sum降低atomicAdd开销；irregular kernel将长短列统一按sub-column/block拆分，warpId先通过colIdxIndex判断独立短列还是长列子块，lane以步长32遍历范围做atomicAdd写回。实验比较：在2757个SuiteSparse矩阵上对比ASpT、cuSPARSE v12.2、RoDe、Sputnik四种SOTA baseline，覆盖FP32/FP64精度和dense矩阵B的N=32/128/48/96/182/384/768等列数。RTX 4080s上FP64/N=32相对ASpT/cuSPARSE/RoDe/Sputnik分别为2.22×/59.19×/5.16×/10.92×；FP64/N=128相对1.79×/27.02×/3.62×/6.53×；FP32/N=128相对ASpT 1.19×。消融实验：regular part coalesced B access带来1.32×(N=32)/1.38×(N=128) speedup；irregular part load-balancing优化带来2.26×(N=32)/2.69×(N=128) speedup。

- 后端平台是什么，配置是什么。
  RTX 4080 SUPER (i9-14900K CPU)、RTX 3090Ti (i9-12900K CPU)、Tesla V100 (Xeon Gold 6151 CPU)、NVIDIA A100 (Xeon Gold 5120 CPU)。软件环境：Ubuntu 22.04.4、GCC 9.5.0、CUDA 12.2（NVCC编译），Matplotlib+Numpy用于绘图脚本。

- 评估性能的软件/脚本是什么。修改了什么。
  自研CUDA kernel实现Swift SpMM双路径系统。baseline使用公开代码：ASpT (adaptive tiling SpMM)、cuSPARSE v12.2 (通用SpMM库)、RoDe (CSR row decomposition SpMM)、Sputnik (ROMA+vector memory instruction SpMM)。性能指标：相对baseline的几何平均speedup。使用NVIDIA profiling工具比较memory bandwidth utilization、memory coalescing、L2 hit rate、SM occupancy。实验覆盖2757个SuiteSparse矩阵，dense matrix B随机生成。所有kernel在同一软硬件环境下编译执行保证公平性。

- 开源情况。评估软件/脚本如何使用？基于开源文档和论文，使用例子解释。
  开源：https://github.com/MinttHu/Swift.git（论文首页脚注和Artifact Appendix均确认）。仓库含CUDA源码、编译脚本、数据下载脚本、FigurePlot原始数据处理脚本和SOTA方法复现实验入口。CUDA kernel使用流程：
  1. 预处理阶段（CPU）：读取CSC格式稀疏矩阵A→统计每列NNZ→按NNZ升序排序A的列并同步重排B的行→按warpSize=32划分regular block（列宽=32，blkPtr记录value/rowIdx起点，blkColIdx记录起始列）→剩余不足32列或长列残留元素归入irregular part（irrPtr/irrValue/irrRowIdx+colIdxIndex/blkStart/blkStop描述拆分的sub-column范围）→为regular part生成positionIdx/offsetIdx（segment sum前缀和索引）。
  2. Regular kernel执行：thread block=32×8线程，每warp(32 lane)处理一个sparse block+B中连续32列。每个lane读一个sparse value/rowIdx→根据blkColIdx+colIdx访问B连续位置（column-major layout保证coalesced）→乘积写入shared memory→利用positionIdx/offsetIdx对rowIdx相同的partial sum做segment sum→用较少的atomicAdd写回C。
  3. Irregular kernel执行：warpId通过colIdxIndex判断任务类型（独立短列或长列子块）→blkStart/blkStop或irrPtr定位irrValue/irrRowIdx范围→lane以步长32遍历该范围→对每个非零元循环访问B的N个元素→atomicAdd写回C。
  4. 例如：对一个M=10K、K=10K、NNZ=500K的SuiteSparse矩阵，预处理排序后将相邻NNZ数相似的列聚集成regular block（如NNZ≈64的32列），同一warp的32个线程同时取B的连续地址；长列残余（如NNZ=512的单列）被拆为irregular sub-block，由warp内各lane分担处理。

