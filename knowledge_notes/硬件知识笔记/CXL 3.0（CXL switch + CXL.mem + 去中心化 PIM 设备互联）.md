## CXL 3.0（CXL switch + CXL.mem + 去中心化 PIM 设备互联）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
CXL（Compute Express Link）是基于 PCIe 物理层的缓存/内存一致性互连标准；MERIDIAN 用 CXL 3.0 构建去中心化 PIM 系统：多个 PIM 设备（CXL Type-3 模块，仅支持 CXL.mem、无 CXL.cache）经 CXL switch 挂到 host，标准 load/store 语义访问；CXL switch 同时提供 host-device 连接与 device-to-device 直连（绕开 host 降低延迟）。相比 HBM-PIM（base die 集成逻辑、容量有限）与 DIMM-PIM（DDR 带宽/仲裁受限），CXL 提供模块化、可扩展的内存解耦基板，LPDDR5X-backed 原型兼顾高带宽与大内存足迹；容量随设备数线性扩展（32 设备 16 TB）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
MERIDIAN 的 CXL 3.0 运转流程：host 收 query 并协调分布式执行（统一 host 侧调度器）→ 预计算文档 KV 经 CXL.mem load/store 写入 head-sharded PIM 位置（标准 load/store，文档更新无需系统级重排）→ 推理时 query token 下发 CEC、DAC 就地算文档注意力 → 设备间交换紧凑统计量走 CXL switch 的 device-to-device 直连（绕开 host）→ host 返回/回收输出。建模参数：CXL 3.0 over PCIe Gen5 ×16 峰值 128 GB/s/链路，端到端访问延迟 165ns（25ns 端口往返 + 10ns retimer + 70ns switch + 60ns 内存控制器/DRAM），switch 处按活跃设备共享链路带宽、超限传输串行化（contention 建模）。对比集中式（文档 KV 经 PCIe 搬上设备）：MERIDIAN 每步只传 query 向量（~d_model×2 bytes FP16）并回收紧凑摘要，通信占比 ≤6.34%（baseline 最高 93.40%）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：CXL Type-3 设备 = CXL 控制器（集成 PCIe 物理/链路/事务层）+ 内存子系统；MERIDIAN 设备 = CXL 控制器 + 8 个 LPDDR-PIM package（每 package 8-channel PIM 控制器、128-bit 宽、每 channel 4 个 16-bit die），每设备 512 GB/32 TFLOPS。系统侧 Linux 经 daxctl + mmap 把设备内存映射为字节可寻址区域。使用方式：大容量 RAG KV 库的弹性扩展（每设备 8×64GB LPDDR5X package）、跨设备去中心化推理的通信基板；与 CXL Type-3 分离式 KV 缓存（KV cache offload）相比，MERIDIAN 不止把 KV 放 CXL 内存，还在设备上就地计算（CCM/计算内存路线）。

涉及论文标题：
- MERIDIAN: In-Memory Acceleration for RAG with Document Attention Decomposition
