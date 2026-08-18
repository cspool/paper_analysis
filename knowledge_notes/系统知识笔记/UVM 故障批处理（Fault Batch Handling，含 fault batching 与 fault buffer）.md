## UVM 故障批处理（Fault Batch Handling，含 fault batching 与 fault buffer）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
UVM 故障批处理是 CPU 侧 UVM driver（nvidia-uvm.ko）处理 GPU 页故障的核心流程：GPU 高并行（多 SM 的 warp 同时在不同虚拟地址上故障）导致故障量大，driver 不逐条服务而是**批量**处理——GMMU 把故障聚合进 GPU 侧 ring fault buffer，中断 CPU 后 driver ISR 取批（fault batching，fault batching count Bf 默认 256，LÆGIS 实验用 128），按虚拟地址分组识别约 2 MB 迁移区域，随后执行 fault preparation（取 fault 信息、预处理）与 fault servicing（加密、DMA、更新页表、重放）。batch 处理中断驱动、服务链经 driver 管理的工作队列单线程执行（影响加密并行性）。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
（近似分层：本术语是 OS driver 级调度机制，非 serving 系统；归入系统架构记录 driver 侧调度行为。）LÆGIS 剖析的 batch 处理流程（图 3 时间分解）：GPU 缺页 → GMMU 聚合 fault buffer → GPU 中断 CPU → ISR 取批 → **fault preparation**（取 fault 信息 + 预处理，含 TBNp 预取决策）→ **fault servicing**（对每个 4 KB 页做 AES-GCM 加密、CE copy 命令入 pushbuffer、DMA、更新页表、信号重放）→ 批次间 driver 进入 true idle（平均 87% 时间）等待下一批。关键观测：(1) 加密占 batch 处理 CPU 时间可超 70%（aggressive 预取下 CNN 实测），是首要瓶颈；(2) GMMU 调度批次时并不服务全部 pending fault，而是跨批拆分，产生自然 idle 窗口；(3) driver 读取 fault buffer 内容即可预取下一批候选（Opportunity 2）。LÆGIS 利用这两类空闲（false：fault preparation 期；true：批次间）做预加密，把加密移出关键路径。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：NVIDIA open-gpu-kernel-modules 的 nvidia-uvm 模块（中断驱动 ISR + 工作队列）；参数 Bf（fault batching count）与 Pt（TBNp 预取阈值）可调。使用：UVM/oversubscription 场景的页迁移性能优化对象——aggressive 预取（Pt=1%）减少 batch 交互次数但增加每批加密量（LÆGIS 观测 1），LÆGIS 的预加密调度（IFN）消除该权衡。评估：在 GPGPU-Sim+UVMSmart 中建模 fault batching、可调预取阈值、tree-based 预取，并用真实硬件 profile（fault prep 时间、idle 窗口、加密吞吐）注入。

涉及论文标题：
- LÆGIS: Pinpointing and Addressing Performance Overheads of GPU-based Confidential Computing
