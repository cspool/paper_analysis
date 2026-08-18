## Stateful Emulation（有状态仿真，in-field 运行时硬件仿真）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Stateful Emulation（有状态仿真）是论文演示的 IPU 第一大能力：让通用 IPU 在真实部署、主处理器全速运行时，以 line-rate 执行复杂、有状态的历史依赖算法（如预取器），相当于"在硅上运行时仿真一个候选硬件模块"。背景逻辑链：预取器等硬件是复杂历史状态机（历史表、状态转移图），必须订阅处理器实时取指流、维护内部状态、对每条指令做计算且不能 stall 主处理器——现有工具全部做不到：PMU 是无状态计数器、JTAG/trace 不跑在实时部署上、硬件仿真器（Cadence Palladium/Synopsys Zebu）耗资数百万且 2-5× 慢、RTL/cycle 级模拟再慢几个数量级。IPU 用其紧耦合集成 + 低延迟状态跟踪做到：introspection 程序订阅 HIT 信号（如 fetch-PC）实时运行候选逻辑并回报统计，实现 in-field A/B 测试（在同一真实负载上对比不同预取器二进制）。论文选择 entangled 指令预取器（ISCA 2021，Ros & Jimborean）作为演示：订阅 CPU 前端 fetch-PC（1 个数据信号=64 bits、2 个 HIT 信号），在 IPU_pro eFPGA 上实现（RISC-V 软件跑会丢维持 cache 状态的数据，故用软逻辑，每地址 1 cycle、近满 eFPGA 利用率），累计覆盖/准确/缺失计数器（每 2³¹ cycles 上报避免溢出），ChampSim 上跑 135 条 CVP traces 与原实现对比（always-hit-in-L2 假设使各指标平均优于实际 <5%）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
有状态仿真在 IPU 中的运转流程（entangled 预取器 emulation，ChampSim co-simulation）：CPU 前端（HIT）每周期把当前 fetch-PC 送入 IPU（64-bit，全程序分析故不配 ADDR/TS/TE）→ IPU_pro 的 eFPGA 软逻辑每地址一个 cycle 运行 entangled 状态机：维护历史表、按历史判定 entangling 对（预取对象），并假设所有 L1 miss 都是 L2 hit 以确定 entangling 对（IPU 不能注入 HIT，此假设不改变 L1 内容故访存执行仍正确）→ 预取决策与统计返回，覆盖/准确/缺失计数器在硬件中累计 → 周期性（每 2³¹ cycles）把统计发主机避免溢出 → 主机与原始 entangled 实现（作者在 ChampSim 的实现）对比：误差仅来自 always-hit-in-L2 假设，每个统计指标平均优于实际 <5%（高预取统计时初始 miss 率 <0.25% 故指标意义有限）；不建模 cache 污染（结果证明影响小）。IPU_pro 面积 0.22 mm²=0.7% CPU 参考、20.8mW≈0.5%；单芯片单 IPU_pro 则 0.175%/0.125%。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：状态机逻辑用 Verilog（300 行）映射到 FABulous 生成的 eFPGA 软逻辑（590 CLB + 470 AIO tiles + 8 BRAM），与 RISC-V 核经内存映射寄存器接口协作；模拟验证用自研 IPU 模拟器加进 ChampSim（135 条 CVP-1 traces，https://microarch.org/cvp1/）做 co-simulation。使用方式：作为"可部署的硬件仿真平台"——预取器作者/芯片团队把候选算法编译成 introspection 二进制，部署到真实负载上做 A/B 测试（同一 IPU 硬件先后跑不同二进制），无需重新流片；相比百万美元级硬件仿真器与 2-5× 慢的商用 emulation（Palladium/Zebu），IPU 以 <1% 面积开销在真实硅上做到近似 line-rate 仿真。局限：不能注入信号（只能观测+本地计算），故需对"被测模块与真实模块交互"做假设（如 always-hit-in-L2），假设引入的近似误差需量化验证。

涉及论文标题：
- Enabling Continuous, In-Field Introspection: The Programmable IPU Architecture
