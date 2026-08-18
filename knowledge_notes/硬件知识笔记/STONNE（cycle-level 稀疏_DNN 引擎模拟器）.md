## STONNE（cycle-level 稀疏/DNN 引擎模拟器）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
STONNE（A Simulation Tool for Neural Networks Engines）是开源 cycle-level 微架构模拟器，建模可重构空间 DNN 加速器（MAERI、SIGMA 等架构族）在卷积/稠密 GEMM/稀疏 GEMM（SpGEMM）/稀疏-稠密（SpMM）等算子上的执行，支持 MK/KN-stationary dataflow 与 bitmap/CSR 稀疏压缩，输出硬件统计 JSON、组件计数器与能耗估算（calculate_energy.py）。SegFold 论文用它建模 Flexagon baseline 的片上组件：Flexagon 的片上部分用 STONNE 模拟、与同一 Ramulator 内存后端集成以保证一致性；Flexagon 原为 1D 128-PE 阵列，被扩展为 2D 2×128 PEs（复制片上 cache、共享 offchip DRAM、按 M 维 tiling 分配、每 1D 阵列保留 128 元素/cycle 的归约与分发网络带宽）以匹配 SegFold 的计算资源。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
使用例子（STONNE 通用流程）：`./stonne -SparseGEMM -M=20 -N=20 -K=256 -num_ms=128 -dn_bw=64 -rn_bw=64 -MK_sparsity=80 -KN_sparsity=10 -dataflow=MK_STA_KN_STR`——命令行指定矩阵形状、PE 数、分发/归约网络带宽、双端稀疏度与数据流，模拟器按 cycle 推进分发-归约数据通路，输出周期/组件统计。SegFold 评估中：Flexagon 的每个 1D 阵列跑 STONNE，内存侧统一走 Ramulator2 HBM2，产出 Flexagon 各静态数据流配置（IP/OP/Gustavson per-tile 选择）的 cycle 数，作为 SegFold 的静态 baseline。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源（https://github.com/stonne-simulator/stonne）。实现：模拟 MAERI/SIGMA 式可重构分发/归约网络（Benes/merge tree），稀疏控制器支持 bitmap 与 CSR；STONNE 还集成 PyTorch 前端（torch_stonne）与 SST 集成（sstStonne）。使用：作为多数据流稀疏加速器评估的标准模拟底座；SegFold 用它保证 Flexagon baseline 与 SegFold 在相同内存后端下公平比较。

涉及论文标题：
- SegFold: Accelerating Sparse GEMM with a Fine-Grained Dynamic Dataflow
