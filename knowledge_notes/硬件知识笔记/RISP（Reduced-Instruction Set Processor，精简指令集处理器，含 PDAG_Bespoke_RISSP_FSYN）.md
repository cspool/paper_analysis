## RISP（Reduced-Instruction Set Processor，精简指令集处理器，含 PDAG/Bespoke/RISSP/FSYN）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
RISP（精简指令集处理器）是通过"减法"——移除 ISA 指令而非添加——从通用处理器裁剪出来的单应用/单领域处理器，是 ASIP 的"减性"特例。核心观察：真实 workload 只用 ISA 的一小部分子集（实证研究 [11][58] 显示大多数应用只使用少量指令），对应硬件可安全移除而不影响正确性。代表工作：Bespoke microprocessors [22]（单应用自动化硬件移除）、PDAG（Property-Driven Automatic Generation of reduced-ISA hardware）[11]（LTL model checking 支持任意 ISA/微架构组合）、RISSP [58]（RISC-V 子集、限于单周期微架构）、FSYN [34]（PDAG 的开源工具链移植）、以及后续的近似计算/符号执行/加速器子集化/形式化 CPU profiling 扩展 [66][67][70][81]。与传统 ASIP 加指令相反，RISP 用 profiling 得指令使用约束后剪除未使用硬件，可系统性缩小芯片面积/功耗。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
RISP 的硬件架构运转流程（以 PDAG 为例，æSIP 的 SOTA baseline）：profiling 目标应用收集指令使用统计 → 翻译成 ISA 级约束（decoder 只允许用到的 opcode）→ LTL model checking 在 baseline netlist 上验证哪些门在约束下冗余/不可达 → 剪除冗余门 → 得单应用 ASIP。æSIP 论文以 PDAG 为归一化 baseline（Table III/IV：PDAG 各 benchmark 面积 ~9.05×10^4–1.31×10^5 µm²、功耗 3.3–4.5 mW @ SKY130；nsichneu 只用 16 条无复杂指令受益最大）。论文归纳 RISP 类方法三大局限：①约束由 profiling 得到，但编译器不知道减少指令使用、不会改写程序（bitcnts 用几次 mul 就保留整个乘法器）；②约束粗粒度、微架构无关（只有 ISA 级 opcode，漏掉操作数宽度/立即数/时序级机会）；③专一化 vs 泛化权衡未解决（per-application 专一化 NRE 高，共享又使 ISA 取并集膨胀）。æSIP 在 RISP 思想上加"硬件感知程序重写"：先把 mul/mulh/div 等重写为更简单指令序列再裁剪，配合数据/时序级 EDC，使乘法器整体可删（对比：bitcnts 在 PDAG 面积 ~1.16×10^5 µm²，æSIP 1.2× 时延约束下面积再降 21.9%、功耗降 16.4%）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：Bespoke 用符号执行/语义约束推导硬件不变量；PDAG 用 LTL model checking（状态元素指数膨胀，规模受限——论文称"PDAG-based pruning under the RV32I baseline times out after 5 hours"）；RISSP 面向 RISC-V 子集与单周期微架构；FSYN 将 PDAG 移植到开源工具链。使用：作为"单应用 ASIP"生成器，输入应用+baseline，输出裁剪核；在 æSIP 评估中 PDAG 方法学充当 baseline（软件 profiling → ISA 约束 → 裁剪），对比 æSIP 的重写+EDC 裁剪。可复现资源：PDAG（Bleier, Sartori, Kumar, DAC 2021）相关方法与 Ibex/Rocket 开源核均在 æSIP artifact 的 Docker 镜像内。

涉及论文标题：
- æSIP μArch-aware ASIP-ISA Co-Design via Program Synthesis, Equality Saturation, and External Don't Cares
