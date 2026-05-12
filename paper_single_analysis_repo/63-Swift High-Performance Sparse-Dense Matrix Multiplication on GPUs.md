论文标题：Swift: High-Performance Sparse-Dense Matrix Multiplication on GPUs

SpMM的GPU kernel优化。

开源仓库确认：
    - 状态：已找到
    - 链接：https://github.com/MinttHu/Swift.git
    - 说明：论文首页脚注明确写出 Swift 可在该 GitHub 仓库获得，Artifact Appendix 也给出同一仓库地址，并说明仓库包含 CUDA 源码、编译脚本、数据下载脚本、FigurePlot 原始数据处理脚本和 SOTA 方法复现实验入口。GitHub 页面可访问，因此可判断为作者提供的公开 Artifact/官方仓库。

1、论文工作：
    - 论文要解决的核心问题：该论文针对 GPU 上 Sparse-Dense Matrix Multiplication（SpMM）的数据加载低效问题。已有 GPU SpMM 方法通常关注稀疏矩阵存储格式、线程负载均衡或局部 tiling，但论文指出它们没有充分利用 GPU warp 内相邻线程访问连续地址时的 coalesced memory access，尤其无法同时让稀疏矩阵 A 和稠密矩阵 B 的加载都具备高度合并访问。论文的实验观察表明，数据加载相关开销平均可超过整体性能的 32%，因此 memory coalescing 是一个被低估的主要瓶颈。
    - 论文的主要贡献：论文提出 Swift，一种面向 GPU SpMM 的算法和数据布局扩展。它先按稀疏矩阵每列非零元数量对列排序，并同步重排稠密矩阵 B 的对应行，使同一 warp 处理的稀疏列与 B 中访问地址更连续；然后用 blocking 将排序后的稀疏矩阵划分为 regular block 和 irregular block，用不同 kernel 路径处理，以同时提升内存访问合并度和负载均衡；最后对 regular part 使用 segment sum 和 shared memory 降低 atomic add 开销。
    - 论文所处背景：SpMM 是图神经网络、稀疏深度学习和科学计算中的基础算子。GPU 的高吞吐能力依赖高效内存访问，但 SpMM 的稀疏性带来非零元分布不规则、索引间接访问和线程间工作量不均。已有方法如 Sputnik、ASpT、RoDe、cuSPARSE 分别使用向量化访存、adaptive tiling、CSR 行分解或通用库实现提升性能，但论文认为这些方案没有从 global memory 到线程这一层同时解决 sparse A 和 dense B 的访问合并问题。

2、相对 Baseline 解决的问题与设计方法：
    - Baseline 的具体问题：Sputnik 通过 ROMA 和 vector memory instruction 处理 sparse data structure 的未对齐访问，ASpT 用 adaptive tiling 处理矩阵不规则性，RoDe 基于 CSR 将行拆成 regular block 和 residual part 并优化 pipeline，cuSPARSE 提供稳定的通用实现。这些 baseline 都能在某些矩阵分布下提升负载均衡或格式效率，但论文指出它们无法让 warp 内线程在访问稀疏矩阵 A 的同时，也连续访问稠密矩阵 B。对于 CSR 或常规 row/column-major B，当 warp 内线程拿到的 sparse column index 不连续时，B 的访问地址会跳跃，可能需要接近 warpSize 次 memory transaction。
    - 论文的设计方法：Swift 采用 column-major 方向的 CSC 实现作为 case study。第一步是 sparsity-based sorting：按每列 NNZ 升序排序稀疏矩阵 A 的列，并按同样顺序重排 B 的行，使 warp 内线程处理相邻列时能访问 B 中连续或近连续位置。第二步是 blocking：根据排序后每列高度，把 A 划分为 block set layer，每个 regular block 的宽度匹配 warpSize；不能整除 warpSize 或长度不规则的部分进入 irregular part。第三步是针对 regular 和 irregular 分别执行两套 SpMM algorithm，regular part 优先保持 memory coalescing，irregular part 优先用批处理和子列划分降低 warp 间负载不均。
    - 方法如何对冲 Baseline 缺陷：相对只改善格式或 load balance 的方法，Swift 把 sparse-column 排序、dense-row 重排和 warp-size blocking 绑定在一起，使 sparse A 的 value/rowIdx 读取和 dense B 的对应元素读取更容易合并为少量内存事务。对于 regular block，同一 warp 的 32 个线程处理连续列，访问 B 时使用 column-major 布局，减少非连续访存；对于 irregular block，Swift 不把一个完整长列简单交给一个 warp，而是把长列拆成多个 sub-column/block，用 colIdxIndex、blkStart、blkStop、irrPtr 等数组记录范围，让 warp 处理粒度更均衡。
    - 关键 trade-off：Swift 接受了预处理开销和数据结构复杂度，包括对 A 的列排序、对 B 的行重排、生成 regular/irregular 两套索引数组、为 segment sum 维护 positionIdx 和 offsetIdx。论文评估显示，当 NNZ 较大时 blocking 的数据移动会使预处理成本上升，NNZ 超过 10^6 时 Swift 的预处理开销会大于 Sputnik 和 RoDe、接近 ASpT。此外，Swift 在非零元分布更均匀时收益更明显；当非零元高度集中或 FP32、N=128 等配置下，ASpT 在部分矩阵上可能排名更好。

3、论文实现：
    - Baseline 如何实现：实验比较四类 SOTA baseline：ASpT、cuSPARSE v12.2、RoDe 和 Sputnik。论文以 GPU 上运行 SpMM 的时间作为主要性能指标，覆盖 FP32/FP64 两种精度和 dense matrix B 的 N=32、N=128 两种列数，并额外测试 N=48、96、182、384、768 等非 32 倍数场景下相对 cuSPARSE 的速度。论文还用 NVIDIA profiling 工具比较 memory bandwidth utilization、memory coalescing、L2 hit rate、SM occupancy 等指标。
    - 新设计如何实现：Swift 使用 CUDA 实现，基础格式选择 CSC，并扩展出 regular part 与 irregular part 两套结构。regular part 包含 blkPtr、blkColIdx、value、rowIdx、positionIdx、offsetIdx，其中 blkPtr 记录 block 在 value/rowIdx 中的起点，blkColIdx 记录 block 起始列，positionIdx/offsetIdx 支撑 segment sum。irregular part 用类似 CSC 的 irrPtr、irrValue、irrRowIdx 等数组存储未进入 regular block 的元素，并配合 colIdxIndex、blkStart、blkStop 描述长列拆分后的 block 范围。Artifact Appendix 说明程序为 CUDA code，使用 NVCC 编译，编译后生成名为 test 的可执行文件。
    - 实验 / 实现平台：论文使用 2757 个 SuiteSparse Matrix Collection 稀疏矩阵，dense matrix B 随机生成。平台包括 RTX 4080 SUPER/4080s、RTX 3090Ti、Tesla V100 和 NVIDIA A100；Artifact Appendix 中列出 CPU/GPU 组合，如 i9-14900K + RTX 4080 SUPER、i9-12900K + RTX 3090Ti、Xeon Gold 6151 + V100、Xeon Gold 5120 + A100。软件环境说明为 Ubuntu 22.04.4、GCC 9.5.0、CUDA 12.2，并需要 Matplotlib 和 Numpy 运行脚本。
    - 关键实验设置与指标：整体性能使用相对 baseline 的几何平均 speedup。以 RTX 4080s 为例，Swift 在 FP64/N=32 下相对 ASpT、cuSPARSE、RoDe、Sputnik 分别为 2.22x、59.19x、5.16x、10.92x；FP64/N=128 下分别为 1.79x、27.02x、3.62x、6.53x；FP32/N=128 下相对 ASpT 的平均优势缩小到 1.19x。A100 和 V100 上也保持相似趋势。消融实验显示，regular part 的 coalesced B access 相对 non-coalesced 版本在 N=32/N=128 下分别带来 1.32x/1.38x 几何平均 speedup；irregular part 的 load-balancing 优化分别带来 2.26x/2.69x speedup。论文还指出 Swift 在 2757 个矩阵中多数配置排名第一或第二，但当非零元集中在较少 32x32 block 中时收益下降。

4、pipeline/kernel解析：
    - 新 pipeline/kernel 是什么：论文没有给一个单独命名的 CUDA kernel 名称，但明确提出 Swift SpMM 的双路径执行流：regular-part Swift SpMM kernel 和 irregular-part Swift SpMM kernel。核心 pipeline 可以概括为“排序 + 阻塞预处理 + regular coalesced SpMM + irregular load-balanced SpMM + atomic/segment-sum 写回”。regular path 追求 warp 内 sparse/dense 双输入的 coalesced memory access；irregular path 追求把不规则剩余元素拆得更均匀，避免某些 warp 处理过长列而其他 warp 空闲。
    - 新 pipeline/kernel 的执行流例子：给定 SpMM C=A×B，其中 A 是 M×K 稀疏矩阵，B 是 K×N 稠密矩阵。预处理阶段先统计 A 每列 NNZ，按 NNZ 升序重排 A 的列，并同步重排 B 的行；随后按 warpSize=32 划分 regular block，剩余不足 32 列或长列残留元素进入 irregular part。执行 regular kernel 时，一个 thread block 配置为 32×8 线程，负责 8 个 sparse block 和 B 中连续 32 列；每个 warp lane 读取一个 sparse value/rowIdx，并根据 blkColIdx 访问 B 的连续位置，把乘积写入 shared memory。随后利用 positionIdx 和 offsetIdx 把 rowIdx 相同的 partial sum 在 shared memory 中先做 segment sum，再用较少的 atomicAdd 写回 C。执行 irregular kernel 时，warpId 先通过 colIdxIndex 判断当前任务是独立短列还是长列子块，再用 blkStart/blkStop 或 irrPtr 找到 irrValue/irrRowIdx 的范围；lane 以步长 32 遍历该范围，对每个非零元循环访问 B 的 N 个元素，并 atomicAdd 到 C。这样，一个稀疏 tile 或列残块会先被映射到 regular/irregular 两条路径之一，再分别按“连续加载、局部规约、原子写回”或“拆分负载、遍历残块、原子写回”的方式完成。
