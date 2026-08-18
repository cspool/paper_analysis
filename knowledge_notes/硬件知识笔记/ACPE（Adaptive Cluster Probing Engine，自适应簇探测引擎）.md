## ACPE（Adaptive Cluster Probing Engine，自适应簇探测引擎）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
ACPE 是 ParetoES 的 FPGA 计算核心：一个可重构、深流水、流式取向的簇探测引擎，负责"簇筛选 + 簇内 Top-K 相似度评估"两阶段检索。每个 ACPE 绑定一个专属 HBM2 pseudo-channel（512-bit 接口、225MHz），构成完全隔离的 compute-memory 对（无核间通信/同步），32 个 ACPE 并行占满全部 HBM 通道。ACPE 内部数据流：Ultra-CSR 质心/向量 packet（512-bit 含 30 非零）→ x-decoder（位宽 popcount 单周期解析行索引，生成 (x,y,val) 元组）→ 乘法器（从 15 个复制的 URAM bank 读查询向量元素，INT6 内积）→ Aggregator（行式部分和累加）→ Top-16 Updater（LUT 比较器堆，4 个并行 Top-4 + round-robin 轮询，缩短关键路径）→ Bitonic-16 排序器（两段 merge-sort，10 级流水，索引与分数联合排序）→ Mem Map scheduler（按 sub_nprobe 与预载 LUT 定位下一目标簇的 HBM 地址，发随机访问请求）。关键设计：(1) 质心-向量共置（cluster-centroid co-placement）——质心预编码 Ultra-CSR 放 HBM 通道头部，选中簇子矩阵按簇局部有序索引组织，把全局不规则随机访存重塑为有界流式；(2) sub_nprobe 软件接口——host 初始化计算 sub_nprobe=⌈nprobe/32⌉ 配置每核，运行时调 Recall 目标无需重综合；(3) 每核返回局部 Top-16，host 聚合 32 核得全局 Top-512（Top-512 是并行友好的候选超集，检索目标仍是 Top-100/Top-10，K≤200 时 Recall 恒 100%，Table III）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
单 ACPE 一次自适应簇探测周期（端到端数据流，Fig.7）：
```
# 初始化：host 下发 sub_nprobe、查询量化值、Mem Map LUT
# 每周期循环（每个选中簇）：
1  HBM PC 流式读质心/簇块 Ultra-CSR packet（30 非零/512-bit，1 packet/cycle）
2  x-decoder: popcount 位运算单周期得行索引 -> (row, col, val) 元组流
3  乘法器: 从 15 复制 URAM bank 取 query[col]，30 个 6-bit 乘法并行（DSP 1 周期 3 个）
4  Aggregator: 按行累加部分和 -> 完成 (row, score) 对
5  Top-16 Updater: 4x Top-4 LUT 比较器堆 + round-robin 轮询维护局部候选（关键路径）
6  Bitonic-16: 两段 merge-sort 10 级流水，按 score 联合排序 (cluster, score)
7  Mem Map: 按排序结果 + sub_nprobe + LUT 取下一簇地址，发随机访问请求
# 全部 sub_nprobe 簇完成后：局部 Top-16 -> PCIe 回传 host -> 全局 Top-512
```
架构角色：ACPE 是"选择性计算硬件化"的最小单元——把 Faiss 式 nprobe 簇探测与簇内 Top-K 变成固定流水硬件，32 核无同步并行是吞吐 4761.9 QPS（Sp.Baidu，Recall≈0.8）的来源；质心共置与 URAM 复制消除随机读与带宽瓶颈。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：Vitis HLS 2023.2 把 ACPE 数据流（解码-乘加-聚合-排序-调度）综合为 RTL 部署到 Xilinx Alveo U280（8GB HBM2、32 pseudo-channel、460GB/s）。资源：LUT 54%、FF 37%、DSP 13%、BRAM 34%、URAM 63%，频率 225MHz，81.5 GOPS、59W、1.38 GOPS/W。使用流程：host（CPU）负责离线预处理（聚类/量化/剪枝/编码/H2Balance 映射）与在线查询量化、sub_nprobe 配置、全局 Top-512 聚合；FPGA 只做流式检索。可扩展性：ACPE 数可缩放（论文固定 32 以对齐 32 HBM PC），DMSU 以 32 个 Bitonic-16 替代单体 Bitonic-512 控制资源。论文未开源。

涉及论文标题：
- ParetoES Hardware-Accelerated Sparse Embedding Similarity via Pareto-Optimal Pruning
