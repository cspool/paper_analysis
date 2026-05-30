## Three-Tier Memory System for AI Accelerators（AI 加速器三级存储体系：SRAM + HBM + DDR）

术语是什么？
三级存储体系是 SN40L RDU 的内存层次设计，将三种不同特性和用途的内存集成到同一加速器上：(1) Tier 1 — 520 MiB 片上分布式 PMU SRAM（~12.5 TB/s 聚合带宽），最低延迟，用于 streaming dataflow 的 stage buffers 和中间结果，编译器显式管理（scratchpad，非 cache）；(2) Tier 2 — 64 GiB co-packaged HBM（~1.8 TB/s per socket），中带宽/中容量，用于活跃 expert 权重、KV cache 的常驻，利用自回归解码中的 temporal locality 重复读取；(3) Tier 3 — 最高 1.5 TiB off-package DDR DRAM（~200 GB/s per socket, 8-socket 聚合 >1 TB/s），高容量/低带宽，用于存储 CoE 中不活跃 expert 的全部权重。三级存储的关键创新在于 DDR 直接连接加速器（而非通过 PCIe 走 host），使模型切换带宽比 GPU（32-64 GB/s through PCIe）高 15×-31×。所有三级内存空间由软件显式管理，编译器通过符号生命周期分析和 temporal locality 估算决定数据驻留位置。

从芯片设计角度拆解：
三级存储在 CoE 推理中的物理数据流（以单 socket 为例）：
```
DDR (1.5 TiB, 200 GB/s)
  │  存储150个7B expert的全部权重
  │  ↓ DDR→HBM copy (按需)
HBM (64 GiB, 1.8 TB/s)
  │  存储: Router权重(常驻) + KV cache(常驻) + 1-4个活跃expert(LRU缓存)
  │  ↓ HBM→SRAM streaming (每个decoder layer一次)
PMU SRAM (520 MiB, ~12.5 TB/s aggregate)
  │  存储: GEMM weight tiles + activation tiles + intermediate stage buffers
  │  ↓ PCU streaming compute
PCU → PMU → PCU (片上dataflow, 无片外访问)
```

芯片物理特性：TSMC 5nm 工艺，2.5D CoWoS chiplet 封装（HBM 与 compute die 同 package），DDR 接口通过插拔式 DIMM 连接（方便扩容）。与 NVIDIA GH200（仅 HBM + host DDR via NVLink-C2C）相比，SN40L 每个 socket 的聚合内存容量高 ~2.5×。

术语一般如何实现？如何使用？
实现需要：(1) 芯片级 DDR PHY/controller 集成；(2) 编译器中的多地址空间管理（设备虚拟地址、符号到内存层级的映射）；(3) 运行时 CoE Runtime 管理 DDR↔HBM 数据移动和 LRU 缓存策略；(4) 编译器分析 temporal locality 决定各符号的最佳内存层级（权重优先 HBM，activations/intermediate 可溢出 DDR）。使用时，应用开发者通过标注/API 指定模型和数据位置偏好，编译器自动处理具体分配和数据移动。

涉及论文标题：
- SambaNova SN40L: Scaling the AI Memory Wall with Dataflow and Composition of Experts
