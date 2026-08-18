## Page Fault（页错误：cold-fault 与 re-fault）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
GPU 地址翻译中，本地页表无法解析某虚拟地址时发生的缺页事件：GMMU 本地页表走查失败后，GPU 经互联向 host 侧发 ATS 请求，由 host IOMMU 做集中式页表走查并返回翻译结果，GPU 更新本地页表后重放原访存请求。ShadowUpdate（ISCA'26）把多 GPU UVM 下的页错误分成两类：① **cold-fault**——GPU 首次访问某页、本地页表本就没有该映射；② **re-fault**——GPU 以前已解析过该页映射，但页迁移过程中该映射被 invalidation 从所有 GPU 页表清除、且新映射只装到迁移目的 GPU，其他 GPU 再次访问时旧映射已失效而重新触发缺页。论文实测 re-faults 平均占全部 page faults 的 73.59%（14 个 workload），且 78.32% 的翻译延迟来自 GPU 之外的 host 侧处理（interconnect + IOMMU），说明 re-fault 是 access counter-based migration 下被忽视的主要瓶颈。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
运转流程（ShadowUpdate 论文）：CU 访存 → L1 TLB miss → L2 TLB miss → GMMU（PWQ 排队 → PTW 多级页表走查，每级 100 cycle，查 PWC）→ 本地页表有映射则直接返回；无映射则发 ATS 请求经 32GB/s CPU-GPU 互联到 host IOMMU（16 个共享 PTW）→ 集中式走查返回远程映射 → GPU 装本地页表并重放请求。cold-fault 与 re-fault 走同一路径，区别只在 re-fault 是"以前已翻译过、因迁移失效而重翻"，属于冗余翻译开销。ShadowUpdate 通过把新映射 piggyback 到 invalidation 广播中，使所有 GPU 在 invalidation 阶段同步更新 PTE，从而消除 re-fault（页错误总数平均降 73.83%）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现为 GPU GMMU 硬件（页表走查失败检测 + ATS 请求发起）+ host IOMMU/驱动（集中式页表、翻译返回）；NVIDIA Volta 起用每页访问计数器（阈值如 256）触发迁移，Ampere 后 UVM 驱动（nvidia-uvm 模块）负责迁移处理。使用上以 PFPKI（Page Fault Per Kilo Instruction，页错误/千指令）作为 workload 指标（ShadowUpdate 论文 Table II 给出 14 个 workload 的 PFPKI，如 SpMV 26.63、MIS 21.30、FW 21.62），用于比较迁移/翻译压力；re-fault 的量化（73.59% 占比）驱动了 ShadowUpdate 的机制设计。

涉及论文标题：
- Reducing Page Faults via Invalidation-based Mapping Propagation in Multi-GPU Systems
