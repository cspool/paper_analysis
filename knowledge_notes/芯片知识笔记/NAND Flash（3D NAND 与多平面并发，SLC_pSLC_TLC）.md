## NAND Flash（3D NAND 与多平面并发，SLC/pSLC/TLC）

术语解释
- NAND Flash 是 SSD 的存储介质，数据以"块(block)-页(page)-平面(plane)-die"的层次组织；一次读（sensing）或写（program）以物理页为单位，擦除以块为单位，且必须先擦除才能重写（out-of-place update）。本文按单元存储位数区分 SLC（1bit/cell，低延迟高 IOPS，如 Kioxia XL-Flash、Samsung Z-NAND）、pSLC（TLC 以伪 SLC 模式运行）、TLC（3bit/cell，容量大但 sense/program 延迟高）。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- NAND 芯片由多个 die 构成，每个 die 内含多个 plane，每个 plane 有独立的读出放大器与页缓冲。一次读/写命令可以同时在多个 plane 上并行执行（multi-plane operation），是 die 内并行（intra-die parallelism）的主要来源。论文（Sec. III-B）用第一性原理把 NAND 设备性能建模为：单 die 峰值 IOPS = 读部分 R_r·N_Plane/τ_sense + 写部分 R_w·N_Plane·l_PG/(τ_prog·l_blk)，其中 τ_sense（如 SLC 5μs、pSLC 20μs、TLC 40μs）是 sensing 延迟、τ_prog（SLC 50μs、TLC 1ms）是 program 延迟、N_Plane 是平面数、l_PG 是物理页大小、l_blk 是主机访问块大小。物理页必须整页编程，所以控制器把主机随机写合并为整页顺序写，写 IOPS 以 l_PG/l_blk 个块折合。3D NAND 大页/大块（如 16KB 页、768 页/块 vs 2D SLC 的 4KB/128 页）会放大 GC 代价与写放大（网络来源证实）。
- 从芯片设计角度拆解术语：SLC 配置（τ_sense=5μs、N_Plane=6、N_CH=20、N_NAND=4、B_CH=3.6GB/s、τ_CMD=150ns、l_PG=4KB）下，模型给出 512B 随机读 ~57M IOPS、4KB ~11M IOPS：小块的超高 IOPS 来自短 sensing 延迟 + die 内多平面并行 + 通道带宽近似按 B_CH/l_blk 缩放（SCA 把 τ_CMD 降到 100-200ns 后命令开销不再主导）。TLC 因 τ_sense/τ_prog 长，设备端（NAND 时序）始终是瓶颈，IOPS 随块大小几乎不变；SLC 小块是设备端受限、大块变通道受限。Storage-Next SSD 正是利用这一 regime：SLC/pSLC 在 ≤4KB 上提供可扩展的小块 IOPS，而传统 SSD 因 4KB 取向的 ECC/控制器架构在 ≤4KB 保持平直。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 现代 3D NAND 通过增加层数（如 144/176/332 层）与每单元 bit 数提升容量密度；独立多平面读（如 176 层 4 平面 CMOS-under-the-array 设计）与 multi-plane program 是提升吞吐的关键实现手段（网络来源：多平面命令可把分散 4KB 内容性能提升到 1600%）。使用上，SSD 控制器按读写比 Γ_RW 与写放大 Φ_WA 把请求分发到各通道/plane，配合 cache read/program（双内部缓冲隐藏通道搬运时间）与 transfer-sense overlap 最大化吞吐。论文把 SLC/pSLC/TLC 的时序参数视为可再参数化的输入，随技术演进更新。
- GRAINS 补充（基因组图 SCC 的 NAND 组织依赖）：GRAINS 评估用 TLC NAND SSD 配置为 16 channel × 8 die/channel × 4 plane/die、4-KiB 页、4 TB 容量（SSD-G4/G5，channel I/O 1.2/2.4 GB/s）；依赖 multi-plane 操作——GST 调度表的 one-hot plane 位图使同一 die 的多个 plane 并发服务不同访问，图数据与查询以块粒度 round-robin 布放且 active block 跨 plane 页偏移对齐以支持多平面操作；die 页缓冲是 IFP 的处理位置——页读入页缓冲后 on-die PE（ECC_LITE + selection/comparison）直接在缓冲内容上操作，只回传目标窗口/比较结果，避免整页经 channel 搬出 die；ECC_LITE 为轻量 on-die ECC（占 flash 芯片面积 0.2%）。

涉及论文标题：
- Five-Minute Rule 40 Years Later A First-Principles Revisit for Modern Memory Hierarchy
- GRAINS: Enabling High-Performance and Low-Cost Graph-Based Genome Analysis via Storage-Aware Algorithm-Architecture Co-Design
