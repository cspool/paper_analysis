## Introspection Binary（内省二进制）与数据驱动执行模型（data-driven execution，丢数据语义）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Introspection Binary（内省二进制）是运行在 IPU 上的小签名分析程序：相对"用户程序"（跑在芯片主处理器上的应用，如浏览器、DL 推理负载），内省二进制专门分析用户程序的硬件级行为。它与普通内核/算子程序的关键差异：①输入不是内存数据而是 HIT 的微架构信号（每次执行 32 个命名输入寄存器，各带 valid 位指示新数据到达）；②按数据驱动执行模型运行——新数据到达且 IPU 处于 ACTIVE-PAUSED 状态时调用 _main，处理期间处于 ACTIVE-RUNNING，新数据被丢弃；③输出经逻辑 FIFO 或主机内存映射区域发出（IPU 可发简单访存指令透明路由到主机内存区域）；④受代码签名与策略模式（closed/restrictive/permissive）约束，经 app-store 式分发部署。程序用 RISC-V 工具链 C 编写（直方图/hash/循环指令作 intrinsics），IPU_pro 程序附带 Verilog 软逻辑；三段式结构（init 固定 0x0、_main、end 固定 0x7F0），8KB 指令内存=2048 条指令。丢数据是核心语义：IPU 不能 stall HIT，introspection 程序处理速率 < 数据到达率时（Little's Law 下必然）丢数据，程序必须用采样、聚合或事件稀疏性适配（随机化采样窗口长度可打破与周期硬件行为的病理性对齐）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
内省二进制的执行模型与丢数据语义（示意）：
```
// 配置阶段（主机 API 经 MMIO）
IPU_CONFIG_IMAGE("PICS-generation")   # 加载签名二进制
IPU_CONFIG_START(ROI_BEGIN)           # TS：区域开始 PC
IPU_CONFIG_STOP(ROI_END)              # TE：区域结束 PC

// 数据驱动执行（每新 IORegs 数据触发一次 _main）
状态机: PAUSED --(配置完成)--> ACTIVE-PAUSED
        ACTIVE-PAUSED --(新数据到达 valid=1)--> 调用 _main --> ACTIVE-RUNNING
        ACTIVE-RUNNING --(新数据到达)--> 丢弃（IORegs 保持旧值）
        _main 返回 --> ACTIVE-PAUSED（等待下一数据）
        FINALIZE 状态 --> 执行 end 清理代码 --> PAUSED

// 示例（PICS generation 的 itlb miss 处理，论文汇编片段）
_main:
    regtimer 50000, psv_loop        # 50000-cycle 定时器驱动 psv_loop
psv_loop:
    beq x0, 1, itlbm_m               # x0=itlb-miss 信号
    beq x1, 1, icache_miss           # x1=icache-miss 信号
    ...
itlbm_m:
    hash r1, x12                     # x12=PC 硬件输入，hash 得表索引
    ld   r2, r1, 0                   # load PSV
    addi r2, r2, 0x40                # 置事件位
    sw   r2, r1, 0                   # store PSV
    ret
```
数据率适配三模式（IPU 2GHz vs HIT 3-4GHz 不同步）：①亚采样（1-in-2 cycles）接受降保真；②快 buffer 异步窗口处理；③只限低频相或优化 IPU 速度。输出节奏：PICS 每 400k cycles 发几字节 PSV 数据；预取器 emulation 每 2³¹ cycles 发统计避免计数器溢出；GPU 直方图每 256 cycles 发 3 字节。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：标准 RISC-V 工具链交叉编译 C 代码 + 专用指令 intrinsics（histogram/hash/loop）；开发环境含 IPU emulator（RISC-V 功能模拟 + IPUpro Verilog co-sim）+ HIT traffic injector 测试台（按 ABI Spec 的时序/数据率生成信号流，忠实复现丢数据行为，验证近似误差并迭代采样窗口长度/分析粒度）；部署经签名+策略模式。使用方式：开发者（或芯片设计者）把分析算法写成内省二进制（如 PICS 75 行、Gaze 根因分类 50 行、GPU 直方图 2 行、预取器 emulation 300 行 Verilog），按 ABI Spec 引用信号输入寄存器；用户程序侧用 API 配置区域与触发；运行结束后主机后处理输出。局限：处理速率不足会丢数据（需采样/聚合），不能注入 HIT（只能观测+本地计算），二进制受 8KB 指令内存与 32 输入限制。

涉及论文标题：
- Enabling Continuous, In-Field Introspection: The Programmable IPU Architecture
