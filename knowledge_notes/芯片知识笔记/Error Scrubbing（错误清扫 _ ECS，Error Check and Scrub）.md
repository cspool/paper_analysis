## Error Scrubbing（错误清扫 / ECS，Error Check and Scrub）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Error Scrubbing（错误清扫）是内存系统主动的可靠性维护机制：周期性地扫描存储空间、读出每个码字、用 ECC 检测并纠正错误、把纠正后的数据写回，从而在错误累积成不可纠（DUE）或静默损坏（SDC）之前就把它们清除。逻辑链：DRAM 保持性失效（retention failure）与瞬时故障随时间累积 → 单次纠错能力有限的 ECC 只能救"当下"，救不了"积累" → 周期性全空间清扫把 transient 错误在变严重前清除（每清扫周期一次"重置"），使长期 DUE/SDC 率被清扫频率钳制而非线性累积。HBM3 中即 JEDEC 的 ECS（Error Check and Scrub）模式：在 REFab（带自动刷新的 activate）与 Self-Refresh 期间后台运行，读整码字→检测→纠正→写回，单比特错（CEs）与可纠多比特错（CEm）被纠正，不可纠错（UE）保留原样并通过 IEEE 1500 寄存器/SEV pin 上报（NE/CEs/CEm/UE 分级）；tECS（全扫周期，如 24h）与 tECSint（命令间隔，如 16Gb 配置约 10.3ms）由 mode register（MR9 OP4/OP5/OP6/OP7）控制。
从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
Error Scrubbing 的芯片设计要素：扫描引擎（ECS engine）+ 可编程周期 + 错误日志/透明上报。运转流程（HBM3）：控制器/片上调度器在 refresh 空闲期插入 scrub 命令 → 每命令读一个码字（含 ODECC parity）→ 片上 ODECC 解码器生成 syndrome → s≠0 则纠正并写回 → 错误计数/地址写入 error log（IEEE 1500 TAP 可读）→ SEV pin 按严重级别编码对外报警。Micron 的 adaptive ECS（专利 US20240211344A1）按 error counter 反馈自动调 scrub 频率。HBM-CASO 的关联（ISCA'26）：其 Decoupled Read Protection 把片上读路径降级为"只检测"，因此必须保证与 ECS 兼容——scrub 扫描读到错误时 HBM 不做纠正、而是发 alarm 给处理器，由控制器取回全部 parity 后分层纠正；同时论文的六年前景生命周期评估把 scrubbing 建模为 12 小时一次，证明清扫频率是长期可靠性的决定性参数（permanent-only p=10^-4 下 G-mode Year 6 DUE 1.6×10^-3、SDC 1×10^-10；mixed p=10^-5/t=10^-5 下 SDC 进一步降到 5×10^-12，正是因为 transient 被清扫周期性清除）。
术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：片上或控制器内的扫描状态机 + ECC 编解码器复用（scrub 读走正常读路径的 syndrome 生成）+ 后台调度（与 refresh 共时隙）。使用场景：(1) 数据中心/HPC 内存 RAS——把散落的单比特错在变为多比特错前清除；(2) HBM/HBM3 ECS——refresh 期间自动执行（Auto ECS via REFab），错误统计经 ECC transparency 寄存器暴露；(3) 学术评估——论文级 lifetime 模拟把 scrubbing 周期作为输入参数（HBM-CASO 用 FaultSim 操作模型：3 小时注入间隔、12 小时 scrubbing，transient 被清扫清除、permanent 持续累积）。实现代价：scrub 占用 refresh/空闲带宽与命令槽位，频率越高 DUE/SDC 越低但性能/功耗开销越大，需按错误率自适应（adaptive ECS）。论文未明确说明 scrub 具体电路面积。

涉及论文标题：
- HBM-CASO: A Coordinated Approach to HBM System-Level and On-Die ECC
