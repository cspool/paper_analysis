## Destination-Agnostic Communication（目标无关通信范式）与 st.rowsp 指令

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Destination-agnostic communication 是 MoE-Hub 提出的新通信抽象：把"数据搬运"与"地址分配"解耦，生产者只用一个**逻辑目的地**（Expert ID / MallocID）发起传输，而不再需要知道接收方内存中的确切物理/虚拟地址；地址分配由接收方硬件的 AAU 在数据到达时按到达顺序动态完成。论文的核心洞察（Insight-1）：MoE 通信的真正依赖是 expert ID 而非内存地址——路由结果一旦确定，token 应该发给哪个 expert、在哪个 GPU 上就确定了，但它在 expert 输入张量中的精确行位置（地址）却是运行时才知道的动态信息；address-centric 模型迫使生产者必须先做软件地址解析才能发数据，这正是软件复杂性与无法重叠的根源。该范式是对 GPU 现有 UVA（统一虚拟地址）address-centric 模型的直接对立面。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
落地需要新的 ISA 指令 `st.rowsp`（row-sparse store，一行 = 激活张量中的一个 token 行）与运行时 API `rowspMalloc`。st.rowsp 与传统 store 语义相同、共享同一 datapath（SIMD lane 合并、TLB 翻译），但目标寄存器含三个逻辑字段：RowID（跨 GPU 全局唯一 token 行索引，同时编码期望的传输顺序）、RowOffset（行内偏移，行按 ≤128B 的 NVLink 包拆分传输）、MallocID（目标 GPU 标识 + 区域标识）。指令带 `.nop` 后缀（st.rowsp.nop）标记非关键路径数据（如 combine 用的源信息）。rowspMalloc 在目标 GPU hub 经 MMIO 注册区域元数据（BaseAddr、RowSize）并返回 MallocID。运转流程：生产者路由 kernel 得到 token→(expert, GPU) 后立即发 st.rowsp → TLB 用轻量逻辑（指令 flag 门控）把 MallocID 解析为目标 GPU ID，不干扰标准地址翻译 → 传输到目标 hub → AAU 查 RAT/自增 APT 分配 LocalRowID → 地址 = BaseAddr + LocalRowID×RowSize + RowOffset → 写内存。本地 expert 的 token 也走同一映射策略（统一本地/远程寻址）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
软件侧使用：编程时路由 kernel 对每个被选中 expert 的 token 发一条 st.rowsp（目标 = 该 expert 的 MallocID + 该 token 的 RowID），完全省去 baseline 中数百至数千行的地址解析/同步/重排代码（MoE-Hub 调度代码 0 行、通信指令 <10 条）。硬件侧：AAU 的 RAT/APT 实现"按到达分配、无冲突、连续密集打包"；RPM 实现传输整形；DAM 实现消费者就绪信号。论文指出该范式可扩展到 TP+EP 混合并行（每个 TP shard 一个 MallocID）、动态 expert 放置（round-robin/负载感知端点选择）与分布式 KV-cache 交换等运行时稀疏通信负载。局限：目前采用静态保守的专家输入区域预分配（防溢出），paging 式更高级内存管理留作未来工作。

涉及论文标题：
- MoE-Hub Taming Software Complexity for Seamless MoE Overlap with Hardware-Accelerated Communication on Multi-GPU Systems
