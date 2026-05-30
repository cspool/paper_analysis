## Intel Optane Persistent Memory (Intel Optane PMem / 傲腾持久内存)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Intel Optane Persistent Memory（PMem，傲腾持久内存）是 Intel 推出的基于 3D XPoint 存储介质的新型非易失性内存模块。它融合了 DRAM 的字节级寻址能力和 SSD 的非易失性持久存储能力。PMem 通过 DIMM 接口直接连接到 CPU 的集成内存控制器（IMC），使用 DDR-T 协议（基于 DDR4 电气/机械接口的通信协议），允许 CPU 通过 load/store 指令直接访问（字节级寻址），无需经过传统的 block I/O 和 page cache 路径。相比 NVMe SSD，PMem 提供更高的带宽（约 6-8 GB/s per DIMM 读带宽）和更低的延迟（~350ns 对比 SSD ~10μs）；相比 DRAM，PMem 容量更大（单条 128/256/512GB vs DRAM 64/128GB）且数据持久（断电不丢失）。

从硬件架构角度拆解术语：
Optane PMem 在 MoESys 的 Hierarchical Storage 中的角色：
```
┌──────────────────────────────────────────┐
│ GPU HBM (80GB/A100)                       │  ← 延迟 ~ns, 带宽 2TB/s
│ Dense params + 激活批次的 sparse params    │
├──────────────────────────────────────────┤
│ CPU DRAM (DDR4)                           │  ← 延迟 ~100ns, 带宽 ~200GB/s
│ Sparse params LFU cache                   │
├──────────────────────────────────────────┤
│ Optane PMem (FSDAX mode)                  │  ← 延迟 ~350ns, 带宽 ~8GB/s/DIMM
│ 全部 sparse optimizer states (master fp32, │     字节级寻址, 非易失
│   momentum fp32, variance fp32)           │
└──────────────────────────────────────────┘
```

MoESys 选择 Optane PMem 替代传统 SSD 存储 sparse parameter states 的原因：
- **低延迟**：PMem ~350ns vs NVMe SSD ~10μs — 对于频繁读写的 expert 参数（每个 training step 都有 expert 激活/淘汰），延迟敏感。
- **字节寻址**：DAX（Direct Access）模式允许 `memcpy` 级别的直接 load/store，避免 kernel page cache 和 block I/O 栈。
- **长寿命**：3D XPoint 的写入耐久度远高于 NAND Flash（传统 SSD 在 MoE 训练的频繁写入场景下寿命严重受损）。
- **容量充足**：12S bytes 的 sparse optimizer states 对 200B+ MoE model 可达 TB 级别，需要 PMem 的大容量（每 DIMM 512GB，同一 CPU socket 支持多 DIMM）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- PMem 有两种操作模式：(1) **Memory Mode**——PMem 作为系统 DRAM 的扩展（DRAM 作为 PMem 的 cache），应用程序无感知，但失去持久性；(2) **AppDirect Mode**——PMem 作为独立的持久内存设备，应用程序通过 filesystem DAX（FSDAX）或 devdax 直接访问。MoESys 选择 AppDirect + FSDAX，在 Ext4 文件系统上以 DAX 方式打开文件，执行 mmap 后通过 CPU load/store 直接读写，无需 read/write 系统调用。
- 限制：Intel 已于 2022 年宣布停止 Optane 产品线的开发和生产。替代方案包括 CXL-attached memory（Compute Express Link 连接的内存扩展，类似 PMem 的字节寻址特性但基于 PCIe/CXL 协议）和 Samsung/其他厂商的类似产品。
- 在 MoESys 的 Hierarchical Storage 公式中，SSD-Node（即 PMem AppDirect）的容量约束为 12S ≤ M_SSD·N，其中 S 为 sparse 参数总量、N 为节点数。

涉及论文标题：
- MoESys: A Distributed and Efficient Mixture-of-Experts Training and Inference System for Internet Services
