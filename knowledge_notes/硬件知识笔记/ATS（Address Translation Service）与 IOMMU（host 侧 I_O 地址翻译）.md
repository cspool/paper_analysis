## ATS（Address Translation Service）与 IOMMU（host 侧 I/O 地址翻译）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
ATS 是 PCI-SIG 规范定义的地址翻译服务：设备（如 GPU）在本地 TLB/MMU miss 且本地页表无映射时，向 host 侧 IOMMU 发起翻译请求，IOMMU 查集中式页表返回翻译结果，设备缓存并使用该翻译（PCIe ATS 1.1 [56]，配套 PRI 处理页错误重放）。IOMMU（I/O Memory Management Unit）是 host 侧管理设备 DMA/地址翻译的 MMU，维护集中式页表并含多个页表走查器（PWQ/PTW/PWC 结构与 GPU GMMU 同构）。ShadowUpdate（ISCA'26）中，多 GPU UVM 的 page fault 路径即：GPU GMMU 本地走查失败 → 发 ATS 请求经 CPU-GPU 互联（32GB/s）到 host IOMMU → IOMMU PWQ 排队 + PTW 集中式走查（16 个共享 PTW，每级 100 cycle）→ 返回远程映射。论文实测该 GPU 外路径占翻译总延迟的 78.32%，是 re-fault 的性能放大器。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
运转流程（ShadowUpdate）：CU 访存 → L1/L2 TLB miss → GMMU 本地 page table walk 失败（page fault）→ GPU 经互联发 ATS 请求 → host IOMMU PWQ（64 项）排队 → PTW 走查集中式页表（查 PWC，每级 100 cycle）→ 翻译结果经互联返回 → GPU 更新本地页表并重放请求 → 用远程映射做远端访问直到访问计数达阈值。规模效应：GPU 数增多时 host IOMMU 集中式翻译压力增大（contention），ShadowUpdate 在 32 GPU 时收益增至 1.57× 即源于消除 re-fault 后 host 侧翻译压力大幅下降。注意别名歧义：数据中心语境中 ATS 常指"自动转换开关（Automatic Transfer Switch）"（vault 证据：repos/repo_paper_isca26_full/knowledge_repo/知识库_系统架构.md 的供电拓扑条目），与本文的 Address Translation Service 是不同概念，需按上下文区分。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：PCIe/PCIe-ATS 规范定义的设备-主机翻译协议（设备侧 ATC/缓存 + host IOMMU 侧 PTW）；NVIDIA 多 GPU 系统（DGX）中 GPU 经 NVLink/PCIe 接入，UVM 页错误由 GPU GMMU 经 ATS 请求 host 的 nvidia-uvm 驱动/IOMMU 处理。使用上，ATS 延迟 = interconnect 往返 + host PWQ 排队 + 集中式走查，是 page fault 开销的主要构成；ShadowUpdate 用"本地提前装映射 + IfMT 短暂阻塞"把翻译留在 GPU 内完成，从而把 ATS/IOMMU 从 re-fault 路径上彻底移除（对非迁移页仍保留 ATS 路径）。

涉及论文标题：
- Reducing Page Faults via Invalidation-based Mapping Propagation in Multi-GPU Systems
