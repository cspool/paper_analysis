## Address Allocation Unit（AAU，地址分配单元）与 RAT/APT

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
AAU 是 MoE-Hub 放在消费者 GPU hub 中的硬件模块，负责把 destination-agnostic 请求的逻辑地址（MallocID, RowID）按到达顺序翻译成具体的物理内存地址，实现"按需、无冲突"的地址分配，从而消除软件地址解析阶段。它由两个结构组成：(1) **Row Allocation Table (RAT)**——tag-RAM，缓存 (MallocID, RowID) → 已分配 LocalRowID 的映射；(2) **Allocation Pointer Table (APT)**——小型 CAM，为每个 MallocID 保存一个 RowPointer（下一个可分配的 LocalRowID）。二者共同完成消费者侧地址分配，无需设备间同步。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
运转流程：目标 hub 收到 st.rowsp 包 → AAU 查 RAT：若 (MallocID, RowID) 已有有效条目，复用缓存的 LocalRowID（同行的后续包）；RAT miss 且从未分配过 → 用 APT 中该 MallocID 的 RowPointer 分配新 LocalRowID 并原子自增（保证连续无冲突分配）；被驱逐的条目映射从内存恢复（late packet 到达时按需恢复，保证整行所有包解析到同一 LocalRowID）。最终地址 = BaseAddr + LocalRowID × RowSize + RowOffset。驱逐策略：RAT 满且需新分配时触发 FIFO 驱逐——行的所有包到达后 (MallocID, RowID)→LocalRowID 映射即可安全删除（RPM 按最小 RowID 先发保证同行的包连续快速到达，支持 FIFO 驱逐），被驱逐映射 spill 到设备内存；RAT/APT 仅在 rowspMalloc 区域释放或重初始化时 flush。这些操作全部限定在 hub 元数据，不涉及全局 coherence 或内存一致性机制改动。效果：来自所有生产者的 token 按到达顺序在消费者内存中密集打包，且全程无设备间同步。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现为 hub 内的小型表格硬件：RAT 是 16-bank 双口 SRAM tag-RAM（占 MoE-Hub 面积大头），APT 是 CAM。硬件开销评估：全部硬件支持（含 AAU/RPM/DAM）TSMC 7nm 下 0.49 mm²、<H800 die 0.06%。使用上对软件透明：路由 kernel 发 st.rowsp 即可，AAU 自动分配；rowspMalloc 负责在分配前注册区域（BaseAddr/RowSize）并返回 MallocID。该机制与 NVLink 128B 缓存行粒度、IOMMU 地址翻译流程兼容（AAU 分配后仍走 IOMMU 翻译经 crossbar 写内存）。

涉及论文标题：
- MoE-Hub Taming Software Complexity for Seamless MoE Overlap with Hardware-Accelerated Communication on Multi-GPU Systems
