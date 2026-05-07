论文标题：Scaling GPU-to-CPU Migration for Efficient Distributed Execution on CPU Clusters

    本地条目说明：
        - 本地编号：paper_2026 第 39 篇
        - 本地 PDF：paper_2026/39-Scaling GPU-to-CPU Migration for Efficient Distributed Execution on CPU Clusters.pdf
        - 本地文本抽取：paper_2026/39-Scaling GPU-to-CPU Migration for Efficient Distributed Execution on CPU Clusters.txt
        - 发表信息：PPoPP 2026，DOI：https://doi.org/10.1145/3774934.3786435

    原文与开源仓库确认：
        - 原文状态：已找到本地 PDF 全文；作者个人页面 / 实验室页面也提供论文 PDF 或论文条目。
        - 原文链接：https://drcut.github.io/assets/papers/ppopp26.pdf
        - 实验室论文页：https://sites.gatech.edu/hparch/publication/
        - 开源状态：未找到明确官方开源仓库
        - 仓库链接：N/A
        - 说明：论文没有在正文中声明 CuCC 代码开源；作者主页、HPArch publication 页面和公开搜索结果未发现与该论文直接对应的官方 CuCC GitHub / artifact 链接。论文说明 CuCC 是扩展 CuPBoP 代码库实现的，单节点性能等价于 CuPBoP 并将其作为 baseline，但这不等于 CuCC 本身已公开。

    1、论文工作：
        - 论文要解决的核心问题：
          这篇论文解决的是 GPU 资源紧缺时，如何把原本写给单 GPU 的 CUDA 程序迁移到多节点 CPU cluster 上执行，并让这些 CPU 节点真正贡献可观吞吐，而不是只作为很慢的 fallback。背景观察来自 Georgia Tech PACE cluster：CPU partition 的等待时间显著短于 GPU partition，说明用户排队等 GPU 时，数据中心里仍有大量 CPU 资源可用。已有 GPU-to-single-CPU 迁移工作可以把 CUDA block 的工作包装成 CPU function，但单个 CPU 与 GPU 的算力差距很大，例如论文引用 A100 FP32 19.5 TFLOP/s，而 AMD EPYC 7713 约 4.096 TFLOP/s，因此单 CPU 迁移仍很难接近 GPU。
        - 瓶颈来源：
          主要瓶颈不是单个 CPU kernel 的计算表达能力，而是从 GPU shared-memory programming model 到 CPU cluster distributed-memory model 的语义落差。GPU global memory 对所有 GPU thread 可见，硬件自然维护一致性；CPU cluster 中每个节点有独立内存，跨节点一致性必须靠网络通信维护。若直接用 DSM / PGAS 之类通用分布式共享内存方案，把每个 GPU thread 的细粒度 global memory 读写映射为 remote memory access，会把大量 byte-level 或 small-object 访问变成碎片化网络通信。例如论文 Listing 3 的 vec_copy 例子会产生 1200 次 cluster-level remote put，每次只有 1 byte，通信开销迅速吞掉分布式执行收益。
        - 论文的主要贡献：
          论文提出 CuCC（CUDA on CPU Clusters），一个把 CUDA 程序编译成 CPU cluster executable 的端到端框架。CuCC 的核心贡献包括：第一，提出针对 GPU SPMD 程序规律的三阶段 CPU cluster 执行流，避免通用 DSM 的碎片化通信；第二，提出 Allgather distributable analysis，用编译期分析判断哪些 GPU blocks 可以被均匀分配到 CPU 节点并通过 balanced-in-place Allgather 一次性同步；第三，实现 LLVM IR 级编译分析、模板化 host module 生成和 MPI-based runtime library；第四，在 CPU cluster 与 A100 / V100 GPU 上评估迁移效果，报告 CuCC 相比 PGAS / single-CPU baseline 有明显加速，并在 cluster-wide throughput 视角下让 CPU 资源达到平均 2.59x 高于 GPU-only 的吞吐。
        - 论文所处背景：
          论文属于 compiler/runtime migration 与 heterogeneous / distributed execution 交叉方向。它关注的不是重写算法，也不是把 GPU kernel 手动改成 MPI 程序，而是尽量保留 CUDA 编程模型，把 single-GPU 程序自动迁移到 CPU cluster 上。目标 workload 包括 AI/HPC 中的 GPU kernels，覆盖 Triton 生成的 BERT / ViT kernels 以及 Hetero-Mark 中手写 CUDA kernels；重点场景是 batch-processing 或对单任务 latency 不极端敏感、但可利用 idle CPU 节点提升总体吞吐的数据中心环境。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：
          论文对比的 baseline 可分为两层。第一层是 GPU-to-single-CPU 系统，例如 CuPBoP 这类把 CUDA block workload 转成 CPU function、把 GPU thread 转成 loop iteration 的系统。它能解决 CUDA 到 CPU 的语义迁移，但只能用单节点 CPU 资源，受限于单 CPU 算力，难以接近 GPU。第二层是把单 CPU 迁移结果进一步放到 CPU cluster 上的通用 DSM / PGAS 思路，例如 UPC++。这类方法提供灵活的 global address space，但灵活性的代价是细粒度 remote access：GPU 程序通常有海量 thread，每个 thread 发起少量 global memory access，映射到 PGAS 后会形成大量小消息，导致网络通信比本地计算更贵。论文 Figure 4 显示 PGAS 迁移在 32-node cluster 上多数程序扩展性差，部分甚至扩展后变慢。
        - 论文的设计方法：
          CuCC 的设计抓住 GPU 程序的 SPMD regularity：所有 GPU blocks / threads 执行同一段程序，差异主要来自 blockIdx / threadIdx，因此许多 global memory 写入呈现连续、对称、可合并的区间。CuCC 不为每个 thread 的 remote write 单独通信，而是让每个 CPU 节点执行一段 GPU blocks，本地写入自己的内存副本；随后用 balanced-in-place Allgather 把各节点写出的连续区间拼接并同步到所有节点；最后，对不适合在第一阶段分布执行的 callback blocks，在所有节点上独立重复执行以恢复一致状态。
        - 方法如何对冲 Baseline 缺陷：
          对 single-CPU baseline，CuCC 将 GPU block-level parallelism 扩展到多个 CPU 节点，让原本单节点难以承受的 GPU-sized workload 被多个 CPU 节点分摊。对 PGAS / DSM baseline，CuCC 避免 per-thread remote access，把大量细粒度访问合并为一个或少数 collective communication。关键是 Allgather distributable analysis 只把满足 balanced 与 in-place 条件的 blocks 放入 partial block execution：各节点写入量相同，写入区间互不重叠且按 cluster rank 排列，从而可以用一次 balanced-in-place Allgather 高带宽同步。对尾部不规则 block 或无法静态证明规则性的 block，CuCC 将其作为 callback block 延后到同步后由每个节点重复执行，牺牲部分重复计算换取语义正确性。
        - 关键 trade-off：
          CuCC 接受了更强的程序结构假设和编译分析复杂度，以换取远低于 PGAS 的通信开销。它特别适合写区间规则、GPU block 数足够多、local computation 足够重的 kernels；对于 block 数少、global write 区间重叠、间接写地址、数据竞争或大量 callback blocks 的程序，扩展性会下降。CuCC 的 sufficient-but-not-necessary 静态分析可能产生 false negative，被误判为非 Allgather distributable 的 kernel 会退化为所有节点重复执行，保证正确但损失性能。另一个 trade-off 是 latency vs throughput：CPU cluster runtime 平均仍慢于 A100/V100 单 GPU，但数据中心 CPU 节点数量远多于 GPU 时，cluster-wide throughput 可能更高。

    3、论文实现：
        - Baseline 如何实现：
          单节点 baseline 来自 CuPBoP，论文说明 CuCC 是扩展 CuPBoP 代码库开发的，因此单节点性能等价于 CuPBoP，并把它作为 GPU-to-single-CPU baseline。PGAS baseline 使用 UPC++ 实现，把迁移后的 GPU memory variables 替换成 PGAS global variables，并把相应 read/write 替换为 remote memory access。GPU baseline 则运行原始 GPU 程序，在 NVIDIA A100 与 V100 上测量原始 CUDA 程序 execution time。
        - 新设计如何实现：
          CuCC 由编译组件和 runtime 组件构成。编译流程先将 GPU source code 编译为 LLVM IR，然后在 LLVM IR 层执行 Allgather distributable analysis，收集生成 CPU executable 所需 metadata，包括是否 tail divergent、需要通信的 memory variables、每个 block 写入的 byte 数等。随后 CuCC 用模板化方法生成 CPU host module，host module 对应三段代码：Partial Block Execution、Balanced-In-Place Allgather、Callback Block Execution。CPU kernel module 的生成沿用 GPU-to-single-CPU 的转换思路：同一 GPU block 内的所有 GPU threads 总是在同一个 CPU node 上执行，GPU thread 变成 CPU function 内 loop iteration，GPU block 变成可由 CPU thread 调度的 function call。runtime library 提供 cluster operations，论文实验中使用 MPI primitives 实现这些操作。
        - Allgather distributable analysis 如何实现：
          CuCC 定义 Write Interval 表示 GPU thread / block 写入 global memory 的地址范围。一个 GPU kernel 对 N-node cluster 是 Allgather distributable，需要存在 callback block 集合 C，使 B-C 可以划分为 N 个 disjoint subsets，并满足三点：每个 subset 的 write interval 长度相等；任意两个 subset 的 write interval 不重叠；所有 subsets 的 write interval 合起来没有 gap，覆盖 B-C 的整体写区间。编译实现中，CuCC 检查所有写 GPU global memory 的指令：写地址相对 thread index 是 affine function；写指令不在 thread-variant condition 中，除非是 tail divergent；写地址相对 block index 是正系数 affine function。tail divergence 用来处理常见边界检查，例如输出大小不是 block size 倍数时，只有最后一个 block 的 if(id<N) 可能 diverge，这个 tail block 可以转成 callback block。
        - 实验 / 实现平台：
          论文使用两个 CPU clusters。SIMD-Focused cluster 有 32 个节点，每节点 2 x Intel 6226，24 cores，约 4.15 TFLOPs，100 Gbps InfiniBand with RDMA。Thread-Focused cluster 有 4 个节点，每节点 2 x AMD 7713，128 cores，约 8.19 TFLOPs，同样 100 Gbps IB。GPU 对照包括 NVIDIA A100（108 SMs，19.5 TFLOPs）和 V100（80 SMs，15.7 TFLOPs）。覆盖率实验分析 Triton 生成的 BERT / ViT NVVM IR kernels，以及 Hetero-Mark 的手写 CUDA kernels。性能实验选取 8 个此前 GPU migration 工作使用过的 GPU programs，并过滤掉 A100 上 kernel execution time 小于 100 ms 的程序；每个实验运行 7 次，报告 median。
        - 关键实验设置与指标：
          主要指标包括 Allgather distributable 覆盖率、strong scalability、network overhead、相对 PGAS 的 speedup、CPU cluster vs GPU runtime、cluster-wide throughput，以及不同 CPU 架构对迁移程序的影响。覆盖率上，BERT / ViT 中 21 个 Triton-generated kernels 全部可 Allgather distributable；Hetero-Mark 中 13 个手写 CUDA kernels 有 8 个满足，剩余主要因为写区间重叠或间接内存访问无法静态分析。性能上，CuCC 在多数 benchmark 上随节点数提升有扩展性；相对 PGAS，过滤 Transpose outlier 后，CuCC 在 2-node cluster 平均 4.09x faster，在 32-node cluster 平均 12.81x faster。CPU vs GPU runtime 上，SIMD-Focused cluster 几何平均比 V100 慢 2.55x、比 A100 慢 4.14x；Thread-Focused cluster 比 V100 慢 1.57x、比 A100 慢 2.54x。吞吐分析中，TACC Lonestar6 这类 CPU 节点远多于 GPU 节点的集群上，论文估计 CPU migration 使 CPUs 达到平均 2.59x GPU throughput，GPU+CPU 相比 GPU-only 平均提升 3.59x throughput。
        - 局限与实验解释：
          CuCC 并不宣称 CPU cluster 总能替代单 GPU 的低延迟执行。Transpose 这类 memory movement-heavy kernel 在 CPU cache / SIMD 上甚至可接近或超过 GPU，但 EP、GA 等 block 数较少且 SIMD 不友好的程序仍被 GPU 以 5x-10x 优势超过。Kmeans 在 32-node SIMD-Focused cluster 上会因每节点分到的 GPU blocks 过少、callback blocks 增多而变慢。论文讨论指出，迁移收益依赖两个条件：足够多 GPU blocks 用于分布到 C 个节点、每节点 T cores 时最好至少有 C x T blocks；local execution overhead 足够重，使通信占比不主导。CPU 架构方面，Thread-Focused CPU 通常优于 SIMD-Focused CPU，即使理论 FLOPs 接近，也说明迁移后的 GPU program 更容易利用 thread-level parallelism，而不是自动获得理想 SIMD 向量化。

    4、pipeline/kernel 解析：
        - 新 pipeline/kernel 是什么：
          论文引入的核心是 CuCC GPU-to-CPU-cluster migration pipeline，而不是一个单独的 GPU kernel。它包含一条编译期 pipeline 和一条运行期 kernel execution workflow。编译期 pipeline 是：CUDA source -> LLVM IR -> Allgather distributable analysis -> metadata collection -> template-based CPU host module generation -> CPU object files -> link with CuCC MPI runtime。运行期 workflow 是三阶段：Partial Block Execution、Balanced-In-Place Allgather、Callback Block Execution。
        - 运行期三阶段执行流：
          第一阶段 Partial Block Execution 中，CuCC 将满足规则的 GPU blocks 按连续 block 区间分给不同 CPU nodes，每个节点只执行自己的 block subset，并写入本地 memory copy 中对应的 non-overlapping interval。第二阶段 Balanced-In-Place Allgather 中，每个节点把自己本地写出的连续区间贡献给 collective communication，Allgather 将所有节点的区间按顺序拼接，并在所有节点形成一致的 memory copy。第三阶段 Callback Block Execution 中，那些尾部不规则、写入量不均衡或无法满足 Allgather 条件的 callback blocks 在所有 CPU nodes 上重复执行；因为 Allgather 之后所有节点 memory copy 已一致，而 callback blocks 的执行集合也一致，所以执行后各节点仍保持一致。
        - 一个具体 kernel 如何流过 CuCC：
          以论文 Listing 1 的 vec_copy 为例，原始 GPU kernel 有 5 个 GPU blocks，每个 block 256 threads，N=1200。blocks 0-3 各写 256 个元素，block 4 只写 176 个元素，属于 tail divergent。若在 2-node CPU cluster 上执行，CuCC 在第一阶段把 blocks 0-1 分给 Node 0，把 blocks 2-3 分给 Node 1；两个节点分别本地写入 512 个元素。block 4 不在第一阶段执行，而是标记为 callback block。第二阶段，CuCC 对 dest 数组中两个 512-element 区间执行 balanced-in-place Allgather，使 Node 0 和 Node 1 都获得 blocks 0-3 的完整写入结果。第三阶段，两个节点都独立执行 block 4，写入最后 176 个元素。最终每个 CPU node 的 memory space 都等价于 GPU 执行 blocks 0-4 后的 global memory 结果。
        - 编译期 analysis 如何支撑 pipeline：
          CuCC 的 compiler analysis 是 pipeline 能成立的关键。它必须确认每个节点在第一阶段写入的数据量相同，这样 Allgather 是 balanced；必须确认不同节点写入区间不重叠，这样无需解决 write conflict；还必须确认 write interval 的顺序与 rank 对齐，这样可以使用 in-place Allgather，避免额外 output buffer 和本地 memory movement。对常见边界检查，CuCC 用 tail divergence 放宽条件：最后一个 block 写入量不同不破坏整体 pipeline，因为该 block 可延后为 callback。对写区间重叠、间接地址、复杂 thread-variant branch 的 kernel，CuCC 要么退化为 callback-heavy 执行，要么需要更昂贵的 peer-to-peer / DSM 式通信，论文没有将其作为主要优化路径。
        - 与传统 DSM / PGAS 路径的区别：
          PGAS 路径把 GPU global memory 映射成 distributed global pointer，kernel 中每次 global write 都可能变成 remote put / get。例如 vec_copy 会产生 1200 次 1-byte remote put。CuCC 路径则把这些 thread-level writes 在 block / node 级别合并为连续 write interval，再用一次 collective Allgather 同步。也就是说，CuCC 的 pipeline 把通信粒度从“每个 thread 的一次内存访问”提升到“每个节点的一段连续写区间”，把通信模式从 flexible remote access 改为高带宽 collective communication。这正是论文性能收益的核心来源。
        - 适用条件与失效模式：
          CuCC pipeline 适合 SPMD、block-level parallelism 充足、global memory 写入连续且无跨 block 写冲突的 GPU kernels。Triton 生成的 AI kernels 在论文中覆盖率高，是因为 Triton 抽象较高，不支持 inter-block barrier，倾向生成规则 memory access 和无 block 间 data race 的程序。手写 CUDA kernels 则更容易出现 overlap write interval 或 indirect memory access，导致静态分析无法证明 Allgather distributable。即使 kernel 可迁移，若计算太轻、通信占比高，或者 GPU blocks 数不足以填满 CPU cluster，也会出现扩展性平台期甚至变慢。论文提出的后续方向包括 workload redistribution、可调 block size 的编程框架，以及面向 GPU-to-CPU 转换程序的 SIMD 优化。

