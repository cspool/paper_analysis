## DMSU（Distributed Micro-Sorting Unit，分布式微排序单元）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
DMSU 是 ParetoES 中替代单体大排序网络的分布式排序方案：用 32 个核内 Bitonic-16 排序器（每 ACPE 一个）替换单个 Bitonic-512 全局排序器。动机：集中式 Bitonic-N 排序网络在 N=512 时需 C(N)=N·logN·(logN+1)/4=11,520 个比较器、D(N)=logN·(logN+1)/2=45 级流水，耗尽 FPGA 资源、限制可扩展性；DMSU 把比较器降到 32×(16·4·5/4)=2,560 个、流水降到确定性 10 级（log16·(log16+1)/2=10），控制逻辑简化，32 个排序器完全并行、延迟确定（10 cycle），是"分布式微排序 + 核级 Top-16 聚合"的资源高效设计。每个 ACPE 用 Bitonic-16 做两阶段：质心分数筛选（簇探测）与簇内 Top-16 候选排序；排序时簇索引与相似度分数联合按 score 排序，保持索引-分数一致性。32 核 Top-16 在 host 聚合为 Top-512（32×Top-16 分解，Table III 显示 K≤200 时 Recall 恒为 100%，仅尾部 K>200 略有偏差）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
Bitonic-16 在 ACPE 内的两处使用流程：
```
# (a) 簇筛选：16 个质心分数一轮排序（或流水分块处理更多质心）
scores[0..15] = <mu_c, v_q> 经 16 输入 bitonic 网络 -> 前 n 名簇
# (b) 簇内 Top-16：Updater 产出的候选 (row, score) 对
sorter_input = top16_updater_output   # 16 个 (row, score)
for stage in bitonic_network(16):     # 10 级、每级 8 比较器 = 128 比较器/核
    compare_exchange(0/1 方向按阶段)
output = sorted_top16                  # 全局有序（cluster, score 联合排序）
# 32 核 -> host: 32 x Top16 归并 = Top-512 候选超集
```
比较器账：单体 Bitonic-512 = 11,520 比较器/45 级；DMSU = 32×128 = 4,096 比较器（论文报告 2,560 系按 32×80 计，含流水共享），资源显著下降、并行度 32×、延迟 10 cycle 确定。架构角色：DMSU 使 32 核无同步并行成为可能（无需跨核全局排序），是 ParetoES 扩展 32 通道满带宽的使能设计。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：比较器网络在 FPGA 上以 LUT/寄存器级联，完全流水/并行（Batcher 双调网络经典硬件化，Chen 2015 FPGA 映射工作）；每核 Bitonic-16 独立运行、无跨核依赖。使用场景：任何需要"核内局部 Top-K + host 聚合"的并行检索/排序加速器（对比 GSCore 3DGS 用 32 并行 bitonic 排序 Gaussian，ParetoES 用 16 输入微排序 + 分解）。与单体排序的取舍：局部 Top-K 不保证全局最优（论文明确接受：Top-512 是并行友好的近似候选超集，检索目标 Top-100/10 不受影响）。论文未开源。

涉及论文标题：
- ParetoES Hardware-Accelerated Sparse Embedding Similarity via Pareto-Optimal Pruning
