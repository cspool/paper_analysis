## Intrinsic transmitter / transponder（内禀发射机 / 转发器）与微架构执行可变性

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 这是硬件侧信道分析的通用模型（telecommunication 类比，论文引 [57]）：transmitter（发射机）——不安全指令，其操作数调制（modulate）一个硬件资源（channel）；receiver（接收器）——攻击者，观测通道调制推断操作数值；transponder（转发器）——通道调制最终表现为一条或多条指令的微架构执行可变性（执行时间、资源争用、功耗等）。transmitter 按相对 transponder 的微架构行为分类（RTL2µpath [48]）：intrinsic（内禀）——为自身产生执行可变性，即同时是 transponder（如受 zero-skip 影响的算术指令，在自己功能单元驻留 0 或非零周期）；dynamic（动态）——为并发执行的 transponder 产生可变性（如 store buffer 中的 store 影响后续同地址 load 的转发/L1 命中）；static（静态）——为之后任意时刻执行的 transponder 产生可变性。
- 逻辑链：Helium 聚焦 intrinsic transmitter——算术指令历史上被认为安全、广泛用于处理秘密（密码学），是 cio 等新软件缓解的目标（§III-A）；动态/静态 transmitter 需要 Tracer 维护抽象微架构状态（§VIII 未来工作）。µobs functions 正是为 intrinsic transmitter 建模：映射发射机（同一条指令）的不安全操作数到 transponder 自己的 µobs。
- Web 证据：leakage functions / transmitter 分类出自 RTL2µpath（Hsiao et al., MICRO 2024，https://doi.org/10.1109/MICRO61859.2024.00033，论文引 [48]）；"执行可变性"与 SynthLC 泄漏函数签名相关。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 在硬件架构中，一条 intrinsic transmitter 的运转：MUL 指令进入功能单元，检测到操作数含 0 → 单元内驻留 0 周期（快速 µobs1）→ 提交；否则驻留 N 周期（慢速 µobs2）→ 提交。攻击者测量该指令执行时间（receiver 观测 channel 调制）→ 推断"至少一个操作数为 0"。Helium 把这种可变性形式化为 µobs function（操作数等价类 → µobs），并沿程序把逐指令 µobs 组装成 µtrace 计算概率。
- 具体例子（论文图 2 的优化 2 + §IV-B）：32-bit 均匀秘密 x，transmitter 暴露 y₁（x=0）或 y₂（x≠0）；y₁ 的 PML=32 bit（全泄漏）、y₂ 的 PML≈3.36×10⁻¹⁰——互信息只有 7.786×10⁻⁹（掩盖 y₁），最大泄漏恒 1（不区分输入分布）；PML 正确刻画逐观测泄漏。这展示"同一 intrinsic transmitter、不同观测"的泄漏差异是硬件侧信道量化的核心难点。
- Annotations：transmitter 可同时具多类型（一条 store 可同时是 static 到后续 load、dynamic 到并发 load）；µobs function 的"抽象掉微架构细节"使其比 leakage functions（source→destination 控制流状态元组）更粗、专为泄漏量化设计（比 LM-SPEC 的泄漏子句更不夸大观测）；transmitter 可为硬件发起的非指令操作（硬件预取、页表访问，§VIII）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：intrinsic transmitter 的可变性来自硬件数据相关优化（计算简化、narrow-width 操作数优化等）；leakage functions 可从 SystemVerilog RTL 经 SynthLC [48] 形式化模型检验自动合成签名（source、destination 集、transmitter 操作数、影响可变性的微架构状态），µobs functions 由其合并而来。Helium 假设确定性 µobs（保守覆盖无噪攻击者）；非确定性（噪声、共驻进程）可由用户在每个 µobs function 内指定观测上的概率分布，TracerSim 无需修改即可用（§VIII）。
- 使用：安全工程师用 µobs functions 在硬件设计早期评估优化；Helium 目前只量化 intrinsic transmitter 的泄漏（含历史上"安全"的算术指令），动态/静态 transmitter、非推测执行是当前范围，推测性（Spectre 类）攻击明确排除（§III-A 引 [24][25][96]）。

涉及论文标题：
- Helium: Quantifying Microarchitectural Side-Channel Leakage with Probabilistic Guarantees
