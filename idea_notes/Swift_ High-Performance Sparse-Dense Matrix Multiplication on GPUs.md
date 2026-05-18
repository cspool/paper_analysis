## Swift: High-Performance Sparse-Dense Matrix Multiplication on GPUs

- baseline方法是什么？
  Baseline是已有GPU SpMM方法：Sputnik（ROMA格式+vector memory instruction处理未对齐稀疏访问）、ASpT（adaptive tiling处理矩阵不规则性）、RoDe（CSR行拆分为regular block和residual part优化pipeline）、cuSPARSE v12.2（通用稳定实现）。这些baseline在不同稀疏分布下可以提升格式效率或负载均衡，但均未同时解决warp内sparse矩阵A和dense矩阵B的coalesced memory access问题。

  全栈执行例子（以cuSPARSE CSR-based SpMM + N=128 + A100为例）：
  - 算法层：SpMM C=A×B，A为M×K稀疏矩阵(CSR格式)，B为K×N稠密矩阵(row-major)，C为M×N稠密输出。CSR格式下每个warp处理一行或多行，通过rowPtr定位value/colIdx范围。
  - 系统框架/Serving层：论文未明确说明（直接CUDA kernel调用，无上层框架）。
  - 编译框架层：论文未明确说明（CUDA 12.2 NVCC默认编译路径）。
  - kernel调度层：cuSPARSE CSR SpMM kernel——warp内线程按CSR行分配，每个线程迭代rowPtr[start]到rowPtr[end]范围的非零元。读取A的value和colIdx后，colIdx作为索引访问B的行（B为row-major时colIdx×N+jump导致不连续地址），warp内不同线程的colIdx分布随机→B的访问地址跳跃→接近warpSize次memory transaction→数据加载开销平均超过整体性能32%。
  - 硬件架构层：NVIDIA A100 GPU，128 SM，40/80GB HBM2e，1555 GB/s bandwidth。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出Swift，通过"sparsity-based sorting + dense row rearrangement + warp-size blocking + dual-kernel path (regular coalesced + irregular load-balanced) + segment-sum atomic reduction"系统性地解决baseline的memory coalescing缺失问题。

  **缺陷1：CSR/常规格式下warp内线程的colIdx分布随机→访问dense B时地址不连续→memory coalescing差→数据加载占>32%时间**
  → Swift第一步sparsity-based sorting：按稀疏矩阵A每列NNZ升序排序列，并同步重排B的行。排序后相邻列NNZ相近、colIdx连续性增强，配合column-major B布局使warp内线程访问B地址更连续。

  **缺陷2：稀疏矩阵非零元分布高度不规则→单一路径处理浪费coalescing机会或负载不均**
  → Swift第二步blocking：按warpSize=32将排序后A划分为regular block（列宽恰好=32的完整block）和irregular block（不足32列或长列残留）。为两类block生成独立索引结构（regular: blkPtr/blkColIdx/value/rowIdx/positionIdx/offsetIdx; irregular: irrPtr/irrValue/irrRowIdx/colIdxIndex/blkStart/blkStop）。

  **缺陷3：regular block的coalesced访问仍需要高效的partial sum reduction→直接atomicAdd写回C产生大量atomic冲突**
  → Regular kernel使用segment sum优化：warp内各lane将乘积写入shared memory→用positionIdx/offsetIdx前缀和索引对rowIdx相同的partial sum做local segment sum→仅发生少量atomicAdd写回。这在不破坏coalesced加载的前提下大幅降低了global atomic竞争。

  **缺陷4：长列（高NNZ列）直接分配整个warp导致warp间负载不均**
  → Irregular kernel将长列按sub-column/block拆分：warpId先通过colIdxIndex判断任务类型（独立短列或长列子块）→blkStart/blkStop定位范围→lane以步长32遍历→对每个非零元循环访问B并atomicAdd写回C。长列被拆解为多个均匀子块由不同warp分担，消除单warp阻塞其他warp的问题。

  论文方法全栈执行例子（以Swift CSC-based SpMM + N=128 + A100为例）：
  - 算法层：SpMM C=A×B，A为M×K稀疏矩阵(CSC格式，经Swift预处理)，B为K×N稠密矩阵(column-major+行重排后)，C为M×N稠密输出。
  - 系统框架/Serving层：论文未明确说明（直接CUDA kernel launch，无上层框架）。
  - 编译框架层：论文未明确说明（CUDA 12.2 NVCC编译，生成test可执行文件）。
  - kernel调度层：预处理——CPU端CSC A→统计列NNZ→升序排序A的列并重排B行→warpSize=32划分regular/irregular→生成双套索引。Regular kernel——thread block=32×8线程，每32-lane warp处理一个sparse block(32列)+B中连续32列，lane读sparse value/rowIdx→colIdx=blkColIdx+laneID→B[colIdx:k:k+N]连续加载（coalesced column-major access）→乘积入shared memory→segment sum归并同rowIdx的partial→atomicAdd写回。Irregular kernel——warp通过colIdxIndex查任务→blkStart/blkStop取子块→lane stride=32遍历→每非零元访问B→atomicAdd写回C。
  - 硬件架构层：NVIDIA A100 GPU，利用shared memory(每thread block 48KB可用)做segment sum buffer，warp scheduler通过dual-path减少stall。关键trade-off：预处理（sorting+blocking+索引生成）有额外开销，NNZ>10^6时Swift预处理成本大于Sputnik/RoDe、接近ASpT；但当非零元分布较均匀时memory coalescing收益远大于预处理开销。
