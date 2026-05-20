## ASM-SpMM: Unleashing the Potential of Arm SME for Sparse Matrix Multiplication Acceleration

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  提出ASM-SpMM，一个面向ARM SME的高性能SpMM库。核心kernel/runtime设计：(1) OP-MCF (Outer-Product-friendly Masked Column-merging Format)：按SME vector length划分row window，window内做column compaction将非零位置不重叠的列合并为一个compressed slot，记录RowWindowOffset/ColumnOfRowWindow/SparseAtoB/ColumnPositionMaskBit四类数组，消除Tensor Core格式的硬性block padding；(2) SME outer-product SpMM microkernel：对每个compressed slot加载sparse vector和dense matrix B tile，用predicate mask控制有效非零位置，通过svmopa类outer-product指令累加到ZA tile，把稀疏值向量和dense tile外积写入ZA而非当作普通SIMD；(3) 多ZA tile并发与显式prefetch pipeline：多个independent outer products映射到不同ZA tile/slice，剩余Z register做operand streaming，_svprfw类prefetch指令显式预取sparse/dense operand；(4) SVE/Neon混合matrix-vector kernel：对低密度或碎片化block，将稀疏尾部交给vector unit处理，通过interleaved instruction scheduling与SME path重叠，要求vector工作量能隐藏在SME固定执行窗口内；(5) hetero-core动态work stealing调度：runtime先做hardware-aware task mapping，再用progress monitoring和work stealing动态再平衡。实验比较：(a) operator-level baseline对比ArmPL v24.10、Armadillo v14.6.0、SuiteSparse Cholmod v5.3.3、Eigen v3.4.0、MP-SpMM；(b) 单核ablation对比naive SME、multi-tile、format、prefetch pipeline、vector co-execution逐项贡献；(c) GNN case study将ASM-SpMM接入PyTorch CPU Extension实现ASM-GCN/ASM-GIN，对比PyG v2.6.1、DGL v1.1.2。M4上12个代表矩阵相对Cholmod取得3.5x-7.9x speedup；SuiteSparse分布实验相对ArmPL/Armadillo/Eigen/Cholmod/MP-SpMM的geomean speedup在LX2上为9.69/16.43/19.53/4.32/2.62，M4上为11.81/15.12/18.62/4.78/2.94。

- 后端平台是什么，配置是什么。
  Apple M4 CPU（最多10核：4 P-core + 6 E-core，2个SME compute unit分别服务P-core cluster和E-core cluster，512-bit vector length，一次处理8个double）；LX2 ARM processor（最多12核，所有core配备SME unit，512-bit vector length）。编译器Clang 16.0，使用ARM SME/SVE/Neon intrinsics实现。

- 评估性能的软件/脚本是什么。修改了什么。
  Benchmark脚本执行SpMM operator-level latency和GFLOPS测量。稀疏矩阵来自SuiteSparse（按规模/形状/稀疏度分层抽样80个矩阵）和12个代表性真实图/稀疏矩阵（含TC-GNN/SNAP/OGB/DGL图矩阵）。dense matrix B列数评估512和1024。ASM-SpMM kernel使用ARM SME intrinsics实现：svld1_f32加载sparse data、svmopa_za32_f32_m outer product accumulate、svst1_hor_za32写回ZA到output、_svprfw类prefetch指令显式预取。GNN case study中ASM-SpMM通过PyTorch CPU Extension接入，实现ASM-GCN和ASM-GIN的custom SpMM operator。

- 开源情况。评估软件/脚本如何使用？基于开源文档和论文，使用例子解释。
  论文未找到明确开源仓库（正文无GitHub/artifact URL，PPoPP 2026未确认官方实现发布）。kernel使用流程（基于论文描述）：
  1. OP-MCF格式转换：输入sparse matrix A→按SME vector length划分row window→每个window内删除空列→非零位置不重叠的列合并为compressed slot→生成RowWindowOffset/ColumnOfRowWindow/SparseAtoB/ColumnPositionMaskBit四数组
  2. SME SpMM kernel执行：清空ZA tile→遍历row window的compressed slots→根据SparseAtoB找到B的对应列→svld1_f32加载compressed sparse values到Z register→ColumnPositionMaskBit转predicate register→vectorized load B的dense tile→svmopa_za32_f32_m执行outer product accumulate（sparse vector × dense tile→ZA tile/slice）→循环期间_svprfw预取下一slot的sparse data/mask/column index/dense B fragments→所有slots完成后svst1_hor_za32写回output matrix C
  3. hybrid路径：对低密度block分配SVE/Neon vector path→interleaved instruction scheduling隐藏vector工作量于SME执行窗口内→结果与SME累加路径合并
  4. 多核调度：每core初始获得若干row window→core提前完成时根据滑动窗口剩余非零元和进度信息→从负载较重core窃取row window→全局任务完成

