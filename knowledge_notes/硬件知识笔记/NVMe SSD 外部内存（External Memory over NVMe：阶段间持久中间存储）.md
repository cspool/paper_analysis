## NVMe SSD 外部内存（External Memory over NVMe：阶段间持久中间存储）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- NVMe（Non-Volatile Memory express）是挂接 PCIe 的闪存存储协议/设备：以提交/完成队列命令 + DMA 实现高带宽低延迟块访问（单条 M.2 NVMe 顺序读 GB/s 级）。Lembas（ISCA'26）把它当作**外部内存（external memory）**使用：数据规模超出 FPGA HBM 与 host DRAM 容量时把数据"溢出"到 NVMe 持久化、按需经 PCIe 流式搬回，配合外部内存算法（external-memory columnsort）完成原本需 TB 级内存驻留的计算；论文明确"NVMe 是阶段间持久中间存储而非加速器缓存"，使 seed 阶段可超过 host DRAM 与 FPGA HBM 容量上限。已有外存基因组工具 SMUFIN-F（Frontiers in Genetics 2021）实测 NVMe 相对 SATA 把读带宽从 ~800 MiB/s 提到 ~1.5 GiB/s、用 32 GB DRAM 达 512 GB DRAM 服务器 53–81% 性能，佐证该路线。
- 从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- Lembas 的数据流（III/IV 章）：① seed——minimizer 流（16 B 〈minimizer,index〉 元组）溢出存 NVMe → columnsort 加速器 4 轮把 256 MB 列从 NVMe 搬入 HBM、16 个 16-to-1 merger 排序、转置写回 NVMe（PCIe 8 GB/s 双工 → 有效 ~2 GB/s）→ anchors 存 NVMe → 按 idxR 二次排序；② chain——anchor 流从 NVMe 顺序读回、chaining kernel（4×单 PE）输出 chains 存 NVMe；③ extend——chains 读回、SWG 分数矩阵 + 8×8 tile traceback，输出 SAM。四个 1 TB NVMe 提供双 FPGA 所需带宽（3-FPGA 配置时 NVMe 成瓶颈，加 SSD 即可缓解）。NVMe 三角色：阶段间持久中间存储（比缓存持久、容量大）、外部排序的列/块存储（256 MB 列 = 1 HBM PC）、顺序流喂下游（局部性由排序保证）。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：M.2 NVMe 经 PCIe 接 host；数据搬运用异步 I/O（libaio/io_uring/SPDK）或 DMA 引擎；host 软件 orchestrator 负责 NVMe↔FPGA 的搬运调度与列转置重组（多 KB 大块 memcpy）。Lembas 系统内存恒定 ~8 GB（7× 降低 vs Minimap2 384 GB 级），把基因组可扩展性从 DRAM 容量缩放中解耦（DRAM 容量/美元增速放缓背景下）。典型应用：任何"中间数据超内存容量、可用外部排序/流式算法重构"的数据密集型工作负载（基因组比对/组装、图分析、日志分析）。

涉及论文标题：
- Lembas: Cost-Efficient Genome Alignment with External Memory and FPGA Acceleration
