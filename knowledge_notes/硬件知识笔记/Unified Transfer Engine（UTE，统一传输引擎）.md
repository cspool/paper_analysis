## Unified Transfer Engine（UTE，统一传输引擎）

术语解释
ATX 中每核一个的硬件模块，接口核、多个 NCA 与缓存子系统：一是把任务输入数据从内存系统取到 NCA scratchpad（fetch/prefetch），二是作为核-NCA 虚拟化层，对核透明地调度、重叠、流水化多 NCA 上的任务。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
UTE 由前端（InTaskQ、PAcc Allocator、Stream Unit Allocator、VAcc→PAcc 映射 CAM、VAcc→Streams 映射 CAM、Task Predictor/Prefetcher）与后端（若干 Stream Units、LDQ、Stream Scheduler、Common Bus、每 NCA 一个 PAcc Port、PDQ、OutQ）组成。两个 CAM 是可编程的配置寄存器组，存放"哪些物理 NCA（PAcc）能执行哪些任务类型（VAcc）"与"每个任务类型需要哪些流及其依赖/beexp"。UTE 用虚拟地址访存、复用核的 L2 TLB/MMU 做翻译，只读不写 L2/内存。论文参数：32 Stream Units、128-entry LDQ、Common Bus 128B(data)+32B(addr)、每 Stream Unit 1KB PDQ、每 NCA 2 个 32KB 输入缓冲。开销：单 UTE <128KB 存储，64 个 UTE <1% SPR 面积（CACTI 估算）、4.37% TDP；VAcc 映射表只需 0.5KB+4KB 架构状态（对比 AMX 8KB tile 状态）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
任务旅程：①ATX 指令进 InTaskQ → ②PAcc Allocator 查 VAcc→PAcc 映射找空闲物理加速器、Stream Unit Allocator 查 VAcc→Streams 映射分配所需 Stream Units（头部任务资源不足时可跳过，任务乱序派发）→ ③派发到后端 → ④Stream Scheduler 每周期选一个 Stream Unit 发请求（age-based，同老 round-robin）→ ⑤请求经 LDQ 发向 L2（LDQ 深度限定在途请求数，即 MLP 上限）→ ⑥返回数据经 Common Bus 送到 PAcc Port 写入 NCA 输入缓冲；父流数据同时转发给子流 Stream Unit 算地址 → ⑦全部流数据到齐，通知 NCA 启动 → ⑧NCA 完成，输出进 OutQ、经 ATX Port 写回核寄存器 → 释放 PAcc/Stream Units。若配置了不存在的 VAcc，UTE 向核发异常由 OS 处理。squash 路径：核通知 UTE → 中断任务、清输入缓冲、释放资源（无状态 NCA 直接可接新任务）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现要点：CAM 存映射表（进程状态，OS 用 trap 机制懒保存/恢复以降低上下文切换开销）、每 Stream Unit 的地址生成器做简单增量算术、Access Queue 合并同缓存行的连续访问、PDQ 限制父流领先子流的距离。程序员使用方式（论文图 12）：任务配置期用普通读写 UTE 配置寄存器完成 `UTE_check`（探测 NCA 类型、不支持则 fallback 非 ATX 代码，保证二进制兼容）、`UTE_cfg_VAcc_to_Type`、`UTE_cfg_num_streams`、`UTE_cfg_stream_size/parent/bexp_beg/bexp_end`；执行期每个任务一条 ATX 指令。设计空间：默认 UTE {32 Stream Units, 128 LDQ, 128B CB, 1KB PDQ} 达 Inf UTE {64,512,512B,4KB} 的 80% 性能、1/3 面积、1/2.1 功耗；PDQ 大小与 CB 宽度是最关键的两项资源。

涉及论文标题：
- ATX: Accelerator Task Extensions
