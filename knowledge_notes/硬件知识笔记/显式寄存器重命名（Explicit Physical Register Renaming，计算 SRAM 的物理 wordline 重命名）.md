## 显式寄存器重命名（Explicit Physical Register Renaming，计算 SRAM 的物理 wordline 重命名）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
显式寄存器重命名是 PipeIMC 首次引入 in-SRAM 计算设备的机制，解决双端口计算 SRAM 的端口冲突与数据依赖。传统重命名（如超标量处理器的 RAT，见本库 Rename Map Table 条目）把逻辑寄存器映射到物理寄存器；PipeIMC 把"寄存器"落到计算 SRAM 的**物理 wordline**上：每个 warp 的线程映射到 SRAM 阵列（每 4 条 wordline 一个物理寄存器），重命名即运行时为逻辑寄存器分配不同的物理 wordline 组。硬件组成：renaming unit 内的 alias table（逻辑→物理映射）、wordline usage counter（统计引用某物理寄存器的在飞操作数）、free list（空闲物理寄存器）。流程：rename phase——为操作从 free list 分配物理寄存器、更新 alias table/arch flag（物理 wordline 是否被用作架构寄存器）/usage counter；commit phase——提交单元更新源物理寄存器 usage counter、释放目的寄存器映射到的旧物理寄存器；物理寄存器既不在用也不是架构寄存器时即释放回 free list。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
重命名后的性质（同 issue window 内）：(1) 两个操作永不写同一物理寄存器；(2) 操作不写重命名时刻为架构寄存器的物理寄存器；(3) 只有数据依赖的操作写其他操作的源物理寄存器。由此 dispatcher 的端口冲突检查只需查数据依赖（Algorithm 1 第 17 行），另外两类冲突（双端口写、read-first read-write）被消除。例子（论文 Fig.5 第二行）：冲突情形 1——乘法（port 1）结果写入物理寄存器 P1，加法（port 2）想读 P1 且写 P2，重命名让加法等乘法的结果就绪（数据依赖检查）即可；冲突情形 2——两个操作写同一架构寄存器 R1，重命名分配不同物理寄存器 P1/P2，两端口可并行写；冲突情形 3——乘法（port 1）在 32 次迭代中反复读源 wordline P3，加法（port 2）想写 P3，重命名后加法写的是新分配的物理寄存器，不触碰 P3。操作表以 cycle-by-cycle 图（Fig.6）跟踪两个数据依赖操作的完成/在飞/跳过 phase，全部 phase 完成后操作退出操作表。代价：重命名需要每线程寄存器数翻倍（hybrid-4→hybrid-8，64 GPR/线程），降低执行单元并行度（Pipe-1r 在计算密集 kernel 上略逊 Pipe-1），但换来端口冲突消除与更大的活动操作窗口，使额外计算端口（Pipe-2r）收益显著（重命名单独带来 30.7% 平均提速）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：alias table + free list + usage counter 的硬件单元集成进 IMC controller（renaming unit 接收解码后的操作、分配 wordline、把重命名后操作插入操作表）；没有重命名的配置用 scoreboard 机制（在 rename 单元阻塞有写冲突与潜在 read-first read-write 冲突的操作，只允许无冲突操作进操作表）作对照。使用：PipeIMC 用 16 条目操作表，实验中 rename 单元不因 free list 空而拥塞。Web 证据：经典寄存器重命名概念见通用乱序处理器（Tomasulo/ROB+RAT）；PipeIMC 首次将其应用于 in-SRAM 计算的物理 wordline。Vault 无专门笔记证据（omnisearch 对"寄存器重命名"无命中；本库 Rename Map Table 条目是通用处理器 RMT 视角）。

涉及论文标题：
- PipeIMC a Pipelined In-SRAM Computing Architecture
