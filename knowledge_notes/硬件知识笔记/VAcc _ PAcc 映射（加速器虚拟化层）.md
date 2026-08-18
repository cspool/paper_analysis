## VAcc / PAcc 映射（加速器虚拟化层）

术语解释
UTE 的核-NCA 接口虚拟化：核用虚拟加速器（VAcc）类型 ID 编程，UTE 通过 VAcc→PAcc 映射把任务动态调度到具体物理加速器（PAcc）实例上。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
每个任务类型有一个 VAccId，作为 ATX 指令第一个输入操作数的元数据；每个物理 NCA 实例有一个 PAccId。程序执行前，核用普通寄存器写配置 UTE 的两个 CAM：VAcc→PAcc 映射（哪些 PAcc 能执行哪些任务类型）与 VAcc→Streams 映射（每个任务类型的流数、依赖、beexp）。运行时核只需要知道"任务类型"，无需知道有几个物理实例、由哪个执行、状态如何——UTE 的 PAcc Allocator 负责按映射找空闲实例。这带来两个收益：(1) 新增/更换 NCA 不修改核流水线或端口——核永远只对一个 ATX Port 编程；(2) 二进制兼容——程序先 `UTE_check(NCA_Type)` 查询硬件表（只读，记录 UTE 挂接的物理 NCA 类型），不支持则 fallback 到普通 CPU 代码，同一二进制可在不同 NCA 配置的 CPU 上运行。CAM 容量有限：配置超出容量 UTE 发异常；VAcc 映射（0.5KB）+Streams 映射（4KB）是 UTE 仅有的进程架构状态，OS 用 trap 懒保存/恢复（类似 DECA）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
任务派发路径：ATX 指令入 InTaskQ → PAcc Allocator 用 VAccId 查 VAcc→PAcc 映射得到候选 PAcc 集合、查 PAcc Status 选空闲实例；Stream Unit Allocator 用 VAccId 查 VAcc→Streams 映射确定所需流数与配置，检查空闲 Stream Units → 两资源齐备才派发；头部任务缺资源时可跳过，允许后续任务先行（乱序派发）。映射中无该 VAcc 或无兼容 PAcc 时向核报异常，由 OS 处理。同核多个 VAccId 可并发活跃。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现要点：内容可寻址存储器（CAM）做映射表、PAcc Status/Stream Unit Status 位图做空闲跟踪、只读硬件类型表支撑软件兼容性探测。类似机制见于设备虚拟化（IOMMU 页表、virtio 队列映射）与 GPU 流处理器调度（逻辑 work item → 物理 SIMD lane）。使用方式见论文图 12 的配置期代码（`UTE_check`、`UTE_cfg_VAcc_to_Type`、`UTE_cfg_num_streams`）；配置开销按任务类型摊销，每类型一次配置、可服务每核数千任务。

涉及论文标题：
- ATX: Accelerator Task Extensions
