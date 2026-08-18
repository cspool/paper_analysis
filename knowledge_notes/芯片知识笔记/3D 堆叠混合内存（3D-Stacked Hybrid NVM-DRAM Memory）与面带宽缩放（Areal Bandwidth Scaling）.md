## 3D 堆叠混合内存（3D-Stacked Hybrid NVM-DRAM Memory）与面带宽缩放（Areal Bandwidth Scaling）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 3D 堆叠混合内存 = 把 NVM（如 PCM）与 DRAM 的存储 die 通过 TSV 垂直堆叠在计算 die 之上，组成同一芯片内的异构主存。SHyLA 采用并行堆叠：4-Hi DRAM（25nm、cell density 1×）与 4-Hi PCM NVM（cell density 4×）各自独立优化带宽-容量 trade-off、经 buffer die 重路由，避免垂直堆叠共享 TSV 网格的工艺/IO 对齐约束。面带宽缩放（areal bandwidth scaling）是核心机会：TSV 在二维平面铺开、带宽随 die 面积缩放（2D/2.5D 只能沿封装边缘 PHY 一维缩放，外部带宽远低于内部 cell 带宽）；NVM 高密度让"把 cell 面积换 TSV/外围电路提带宽"几乎不损容量，从而打开 2D/2.5D 无法触及的带宽-容量联合设计空间。LLM 线性层顺序访问命中 row buffer，使 NVM 读利用率达 >70%（一般负载 <10%），是该带宽可被利用的前提。

从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 运转流程（芯片级设计空间）：固定 400mm² 内存 die 面积 → 按 Area-Ratio（如 1.6:0.4）分配 NVM/DRAM 面积 → 各自在带宽-容量曲线上取点（Table III：NVM (138,0.5)/(112,1)/(59,2) GB/s·GB、DRAM (138,0.125)/(112,0.25)/(59,0.5) per plane）→ 有效带宽 = 硬件带宽 × 利用率（DRAM 90%、NVM 读 70%、NVM 写 10%）→ DSE 选 4:1 NVM:DRAM 面积比 + NVM 平衡带宽-容量 + DRAM 峰值带宽 → 每 die 64GB PCM/2GB DRAM、1792/552GB/s。数据放置：IA→DRAM、Weight/KVCache→NVM（溢出入 DRAM；KV 向量 ≥256B ≥ NVM 物理编程单元、写放大 ≈1.0）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 带宽/容量由 CACTI-3DD（[11] CACTI 的 3D 堆叠内存扩展，论文未给公开链接）按共享外围电路 + NVM 4× DRAM 密度推导；性能由 GPGPU-Sim（channel 数按推导带宽配置）评估；热由 3D-ICE 分析（液冷 h=2×10^-7 W/(μm²·K)，稳态温度 315-345K 安全）。与 2D/2.5D NVM-DRAM 混合（DRAM 作缓存或 Optane DIMM 约束）相比，SHyLA 用静态、面向 LLM 数据异质性的放置 + 3D 面带宽取得 up to 5.84×（DRAM-only）/6.03×（NVM-only）系统吞吐。SHyLA 架构参数未开源（联网未找到）。

涉及论文标题：
- SHyLA 3D-Stacked NVM-DRAM Hybrid LLM-Inference Architecture Exploiting Data and Memory Heterogeneity
