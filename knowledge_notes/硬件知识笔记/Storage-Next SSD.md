## Storage-Next SSD

术语解释
- Storage-Next 是 NVIDIA 发起的存储倡议（GTC 2025 起，40+ 厂商加入）与一类 SSD 的统称：面向 GPU（而非 CPU）作为存储客户端，针对 512B 扇区/小块随机访问优化，IOPS 随访问粒度缩小而可扩展（如 50M IOPS@512B vs 10M IOPS@4KB），达到"每美元 IOPS 提升 10 倍"量级，把 NAND flash 从容量层升格为内存的主动扩展层。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 论文把 Storage-Next SSD 定义为"IOPS 随块大小缩小而增长的 SSD"（对比 Normal SSD 在 ≤4KB 上 IOPS 平直），并强调其 IOPS 值并非厂商投影的数据表数字，而是由 Sec. III-B 的第一性原理模型推导（SLC 配置 512B≈57M IOPS、4KB≈11M IOPS；敏感性扫描 512B 落在 39.4M-79.3M）。其物理来源：短 sensing 延迟（SLC 5μs）+ die 内多平面并行（N_Plane=6）+ SCA 通道协议（τ_CMD=100-200ns）+ 扇区级 ECC（512B BCH 内码，跳过 4KB LDPC）共同使小块 IOPS 按 ~B_CH/l_blk 缩放。网络来源证实：NVIDIA 正式启动 Storage-Next（含 DDN/KIOXIA/Micron 等 40+ 厂商），核心问题是"GPU 作为存储客户端时 SSD 应如何表现"，聚焦 512B 扇区读写的优化；Micron 参考设计 3×H100 驱动 44 块 PCIe Gen6 SSD 达 2.3 亿随机读 IOPS、118GB/s。
- 从硬件架构角度拆解术语：Storage-Next SSD 是"设备-主机协同"概念——设备侧提供可扩展小块 IOPS（NAND/channel 物理 + SCA + 扇区级 ECC），主机侧需 GPU 提供足够 IOPS 预算（~4M IOPS/SM，NVIDIA SCADA 平台）才能榨出设备潜力。论文定量显示：CPU+DDR（100M IOPS 上限）下 512B 即使配 Storage-Next 也受主机限制（83s→47s 随 CPU 预算 40M→100M），而 GPU+GDDR（400M IOPS）几乎全在设备受限 regime，break-even 压到 <7s 甚至 ~5s。这是"存储设备主动参与内存层次"的架构级范式。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现路径：设备侧重写 SSD 控制器/固件为 512B 小包场景（ECC 粒度、队列、调度），主机侧用 GPU 直接发起存储访问（SCADA/BaM、cuFile 开源到 XIO-SIG）；KIOXIA GP Series/XL-FLASH 等为 Storage-Next 共研，目标 ~200M IOPS/GPU、Gen6/Gen7。论文使用方式：把 Storage-Next SSD 作为解析框架与 MQSim-Next 的输入配置（SLC/pSLC/TLC 参数），对比 Normal SSD 评估 break-even、可行性阈值与 KV/ANN 案例吞吐；结论是 GPU+Storage-Next 使 DRAM↔flash 缓存阈值从分钟压到秒级。

涉及论文标题：
- Five-Minute Rule 40 Years Later A First-Principles Revisit for Modern Memory Hierarchy
