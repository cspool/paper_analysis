## PIMBench（PIM 基准测试套件与 bank-PIM kernel 评估）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
PIMBench（Siddique et al., IISWC'24，UVA 团队）是数字 DRAM PIM 的架构建模与基准测试套件，提供面向 bank 级 PIM 的通用 kernel 集合，开源于 PIMeval-PIMbench（https://github.com/UVA-LavaLab/PIMeval-PIMbench）。论文用它评估 reliable bank-PIM 在 GEMV 之外的通用 PIM kernel 上的表现，只取原作者报告在 all-bank PIM 上有正加速的 benchmark，并按公开实现移植进 Ramulator2 模拟器。为支持这些应用，论文给模拟架构补充了几条与乘加同延迟的简单指令（绝对值、小于、clamp、位操作），跨 bank 通信指令不用（reduction 在 host 做）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
kernel 调度视角的分层：读为主 kernel（结果存 PIM SRAM、host 后读）与写密集 kernel（PIM 直接写 DRAM）在 reliable bank-PIM 下行为不同。读为主例子（Linear Regression，论文实测 7.9× vs rank-PIM，接近理想）：host 把长输入向量流式分布到各 bank → 每个 bank 旁 PIM 单元算部分和（4 FPU 乘加）→ 结果写 PIM 本地 SRAM（仅 4 个归约和、输出大小与向量长度无关）→ host 读回做最终归约。写密集例子（K-means，1.6× vs rank-PIM）：论文加了 K-means Optimized 优化——每个 PIM 单元用本地 SRAM 缓冲跟踪最小距离与质心，每样本只写质心归属，跨 bank 归约由 host 在 kernel 结束时做；未优化的 K-means 因每样本都要 host 读回（KNN 1.6× 最低即此因，数据复用少）而降速。写路径代价：reliable bank-PIM 每次写要更新 rank 级 ECC（等效 CPU 写、约 8× 慢于非可靠 bank-PIM），写密集 kernel（vector add/AXPY，0.7×）反而不如 rank-PIM；读-执行比高的 kernel（K-means、Image Downscaling 1.3×）仍胜出。VRT 纠错开销经 Codeword Flip + 硬件纠错控制 <2.1%（Filter by Key 最大，因读 PIM 操作占比高）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：PIMeval-PIMbench 开源（https://github.com/UVA-LavaLab/PIMeval-PIMbench），论文按公开实现把 kernel 移植到 Ramulator2（含 PIM 单元指令与 SRAM 缓冲建模），对每 kernel 收集 PIM 命令 trace 喂给模拟器估执行时间。使用：评估 bank-PIM 架构（性能、写开销、纠错开销）时作为 GEMV 之外的通用 workload 补充；比较可靠/非可靠 bank-PIM 时按读-执行比到写比分组看差异（读为主近乎无差、写密集是可靠性的主要代价）。

涉及论文标题：
- ECC Enabled Reliable and Performant Processing-in-Memory
