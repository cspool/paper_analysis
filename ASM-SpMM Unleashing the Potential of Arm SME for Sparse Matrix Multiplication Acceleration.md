论文标题：ASM-SpMM: Unleashing the Potential of Arm SME for Sparse Matrix Multiplication Acceleration

    原文来源：
        - 状态：已找到本地全文
        - 链接：paper_2026/34-ASM-SpMM Unleashing the Potential of Arm SME for Sparse Matrix Multiplication Acceleration.pdf；DOI: https://doi.org/10.1145/3774934.3786422
        - 版本说明：本地 PDF 为 PPoPP 2026 论文，13 页；已用 pdftotext 抽取为 paper_2026/34-ASM-SpMM Unleashing the Potential of Arm SME for Sparse Matrix Multiplication Acceleration.txt 供分析。

    开源仓库确认：
        - 状态：未找到明确开源仓库
        - 链接：N/A
        - 说明：论文正文没有给出 GitHub、artifact URL 或开源说明；基于 PPoPP 官方页面、DOI 信息和公开 GitHub 搜索，未确认作者发布的官方实现。因此本文实现细节只依据论文正文描述，不假定存在可复现实验仓库。

    1、论文工作：
        - 论文要解决的核心问题：SpMM 是科学计算、图分析和 GNN 中的核心稀疏算子，但 Armv9 SME 的矩阵单元主要面向 dense outer-product。直接把传统 CPU SpMM、GPU Tensor Core SpMM 或普通稀疏格式搬到 SME 上，会同时遇到稀疏存储浪费、outer-product 执行不匹配、矩阵单元和向量单元协同不足，以及 ARM 异构核上的负载不均衡。
        - 论文的主要贡献：提出 ASM-SpMM，一个面向 ARM SME 的高性能 SpMM 库。它包含 SME-adapted 压缩格式 OP-MCF、outer-product-oriented SME kernel、多 tile 并发和显式 prefetch pipeline、SME 与 SVE/Neon 的混合 matrix-vector kernel，以及面向异构 CPU core 的动态 work stealing 调度。
        - 论文所处背景：ARM CPU 正从移动端扩展到桌面和高性能计算场景，Armv9 SME 提供 ZA matrix register 和 MOPA outer-product 指令。Apple M4 是论文中提到的公开可用 SME 平台之一，论文还评估了新发布的 LX2 ARM processor。目标 workload 包括 SuiteSparse 稀疏矩阵、TC-GNN/SNAP/OGB/DGL 图矩阵，以及 GCN/GIN inference 中的 SpMM。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：第一类 baseline 是 ArmPL、Armadillo、Eigen、Cholmod、MP-SpMM 等 ARM CPU SpMM 库，它们主要依赖通用多核和 vector unit，无法使用 SME matrix unit 的 outer-product 算力。第二类 baseline 是 TCF、ME-TCF、DTC-LSH、AccOrder、TC-GNN 等 GPU Tensor Core 思路，它们假设 inner-product、warp 调度和 GPU memory hierarchy，并常依赖固定 block alignment、left-aligned tile 和 zero padding；这些约束与 SME 的 predicate-driven vector outer-product 不一致。第三类 baseline 是静态负载均衡，它按行、非零元或 core 能力预分配任务，在 Apple M4 这类 P-core/E-core 且 SME 单元按 cluster 共享的架构上容易失衡。
        - 论文的设计方法：ASM-SpMM 先把 sparse matrix A 转换成 OP-MCF，即按 SME vector length 切成 row window，并在 window 内做 column compaction 和 masked multi-column merging。运行时 kernel 对每个 compressed slot 加载 sparse vector 和 dense matrix B 的 tile，用 predicate mask 控制有效非零位置，通过 svmopa 类 outer-product 指令累加到 ZA tile。kernel 同时使用多 ZA tile 并发、软件 prefetch 和 pipeline 隐藏访存延迟。对于低密度或碎片化 block，ASM-SpMM 使用 SVE/Neon vector path 处理，并通过 interleaved instruction scheduling 与 SME path 重叠。多核执行时，runtime 先做 hardware-aware task mapping，再用 progress monitoring 和 work stealing 动态再平衡。
        - 方法如何对冲 Baseline 缺陷：OP-MCF 去掉 Tensor Core 格式中的硬性 block padding，把非重叠稀疏列合并为一个 physical column，并用 bitmask 还原每个原始列的有效行，使 SME predicate register 可以只参与有效非零计算。outer-product kernel 让稀疏值向量和 dense tile 直接形成外积并写入 ZA，避免把 SME 当成普通 SIMD 使用。多 tile 并发提高 ZA/Z register 占用率，prefetch pipeline 缓解 SpMM 不规则访存带来的 cache miss。混合 matrix-vector kernel 把稀疏尾部和碎片 block 交给 vector unit，减少 SME 在低利用率 block 上浪费周期。动态调度则弥补不同 ARM core 和不同 row window 非零分布造成的运行时偏差。
        - 关键 trade-off：ASM-SpMM 需要一次性格式转换，适合同一 sparse matrix 被重复执行的场景，如 GNN inference、迭代 solver、超参搜索；若矩阵频繁变化且复用次数很少，OP-MCF 转换成本可能更难摊销。混合 SME/SVE 调度提升吞吐，但会引入 register partition、资源竞争和 workload partition 开销，论文报告 hybrid/theory 只有 0.78 到 0.90。设计还依赖 ARM SME 可用性，不同 SME 平台需要调整 tile size、resource allocation 和调度参数。

    3、论文实现：
        - Baseline 如何实现：operator-level baseline 包括 ArmPL v24.10、Armadillo v14.6.0、SuiteSparse Cholmod v5.3.3、Eigen v3.4.0 和 MP-SpMM。单核 ablation 以不使用多线程的 MP-SpMM 作为基础 baseline，并逐步加入 naive SME、multi-tile、format、prefetch pipeline、vector co-execution。GNN case study 将 ASM-SpMM 接入 PyTorch CPU Extension，实现 ASM-GCN 和 ASM-GIN，并与 PyG v2.6.1、DGL v1.1.2 对比。
        - 新设计如何实现：OP-MCF 使用 RowWindowOffset、ColumnOfRowWindow、SparseAtoB、ColumnPositionMaskBit 四类数组记录 row window 的起始位置、压缩列数、原始列索引和列内位置 mask。kernel 使用 SME intrinsics 完成 svld1_f32 加载、svmopa_za32_f32_m outer product accumulate、svst1_hor_za32 写回，并使用 _svprfw 类 prefetch 指令显式预取 sparse/dense operand。Apple M4 上 FP64 的 SVL512 对应 8 行 row window，双精度 ZA 可划分为 8 个 tile；kernel 可把多个 independent outer products 映射到不同 ZA tile/slice，并用剩余 Z register 做 operand streaming。hybrid path 根据 SME/SVE microbenchmark 的延迟估计，优先把最稀疏 block 分配给 vector unit，要求 vector 工作量能隐藏在 SME 固定执行窗口内。
        - 实验 / 实现平台：主要平台为 Mac M4 CPU 和 LX2 ARM processor，二者均为 512-bit vector length，一次处理 8 个 double。M4 最多 10 核，包括 4 个 performance core 和 6 个 efficiency core，但只有两个 SME compute unit，分别服务 P-core cluster 和 E-core cluster；LX2 最多 12 核且所有 core 都配备 SME unit。编译器为 Clang 16.0。benchmark 包括 12 个代表性真实图/稀疏矩阵和 SuiteSparse 中按规模、形状、稀疏度分层抽样的 80 个矩阵，dense matrix B 的列数评估 512 和 1024。
        - 关键实验设置与指标：总体性能以 Cholmod、ArmPL、Armadillo、Eigen、MP-SpMM 为对照，报告 speedup 和 GFLOPS。M4 上相对 Cholmod 在 12 个代表矩阵上取得 3.5x 到 7.9x speedup；SuiteSparse 分布实验显示相对 ArmPL、Armadillo、Eigen、Cholmod、MP-SpMM 的 geomean speedup 在 LX2 上分别为 9.69、16.43、19.53、4.32、2.62，在 M4 上分别为 11.81、15.12、18.62、4.78、2.94。OP-MCF 的 mean NNZ per slot 达到约 4 到 6，明显高于 CSR 和 Tensor Core oriented 格式。prefetch/pipeline 将 LLC miss rate 从约 30%-61% 降到 23%-48%。hybrid kernel 在 rCA、FY-RSR、ddi、ppi 上比 matrix-only 约快 8%-18%。LX2 12 线程相对 2 线程达到 8x 到 11x scaling；M4 因 SME 单元数量有限，多线程增益更受约束。GNN inference 中，在 FY-RSR 上 GCN 端到端速度约提升 1.3x-1.6x，在 ddi 上因 SpMM 占比更高可达 2.0x-2.9x。

    4、pipeline/kernel 解析：
        - 新 pipeline/kernel 是什么：论文的核心新 kernel 是 ASM-SpMM runtime kernel，可概括为 OP-MCF format conversion + SME outer-product SpMM microkernel + heterogeneous matrix-vector co-execution + hetero-core dynamic scheduling。它不是单一算子替换，而是把稀疏格式、SME ZA tile 数据流、SVE/Neon 辅助路径和多核调度放在同一个执行栈里协同设计。
        - 新 pipeline/kernel 的执行流例子：给定 sparse matrix A 和 dense matrix B，系统先按 SME vector length 把 A 的连续行划成 row window。每个 window 内删除空列，并把非零位置不重叠的列合并成一个 compressed slot，同时记录原始列索引和 bitmask。执行某个 row window 时，kernel 清空 ZA tile，遍历该 window 的 compressed slots：读取 SparseAtoB 找到 B 的对应列，加载 compressed sparse values 到 Z register，把 ColumnPositionMaskBit 转成 predicate register，再 vectorized load B 的多个 dense tile。随后对每个有效 mask 调用 SME outer-product 指令，把 sparse vector 与 dense tile 的外积累加到对应 ZA tile/slice。循环期间，kernel 预取下一 slot 的 sparse data、mask、column index 和下一列 B 的 dense fragments，使访存与当前 ZA 计算重叠。若某些 block 太稀疏，runtime 将其交给 vector unit 处理，并把 vector 结果与 SME 累加路径交错安排，避免 SME 因 padding/低密度而空转。所有 slots 处理完成后，ZA tile 被写回输出 matrix C 的对应 row window。多线程时，每个 core 初始获得若干 row window；当某个 core 提前完成，调度器根据滑动窗口中的剩余非零元和进度信息，从负载较重的 core 窃取 row window，直到全局任务完成。
