## Sigries（混合 Rowhammer 防御：欠配置 Misra-Gries + 行采样）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Sigries 是微软部署于 Azure Cobalt 200 SoC（Arm 云原生 SoC）内存控制器的生产级 Rowhammer 防御（ISCA 2026 Industry Track，首个在论文中详细披露的生产实现），混合两类方案：light mode 用欠配置的 Misra-Gries 计数器表跟踪 hot 行；heavy mode 用行采样。三个核心洞察：(1) 欠配置的 MG 只要 spillover 低于 Rowhammer 阈值仍阻止全部攻击；(2) 计数器表按 **sub-bank**（bank 的一部分）组织，k 个 c 项小表比单个 k×c 大表更可实现；(3) spillover 接近阈值作为"可能受攻击"预警（gatekeeper），只让被攻击的 sub-bank 切 heavy mode，其余保持 light。切回 light 需 heavy mode 停留至少若干 refresh window（32ms），期间用 **shadow counting** 监控 spillover 饱和率，饱和率低于阈值才回切；切换在 refresh window 边界进行，暴露限于一个 tREFW，worst-case 累计暴露 < 1 小时/年。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 内存控制器内运转流程（memcached 负载下一次行激活）：ACT 到达 → 定位 sub-bank → set-associative SRAM 计数器表查找（命中 +1 / 未命中与 spillover 比较替换）→ 计数达阈值发 DDR5 DRFM 刷 victim 行、复位计数、置 lock bit → spillover 达阈值-1 时该 sub-bank 切 heavy mode：每次激活以 p 采样发 DRFM，shadow counting + heavy mode countdown 决定回切 → 全程遵守 DDR5 7.8µs DRFM 速率限制（per-bank DRFM 地址表）。SRAM 状态用 parity + 列交织保护，parity 错误按表 IV 做保守修复（优先保安全、牺牲瞬时性能）。配置项：计数器表大小（几十项）、Rowhammer 阈值（DRAM qualification 实测）、采样率 p（按逃逸概率推导）、heavy mode 最短时长、shadow 饱和率阈值。评估（QEMU→Asim 内部 SoC 模拟器→RHSim 三层仿真）：所有 commodity workload 下零 DRFM 带宽开销、不切 heavy mode；decahammer（10 面）被 light mode 遏制；megahammer（k 面）/omni-mega（全 bank）只让被攻击 sub-bank 切 heavy（omni-mega 最坏 6.8% 带宽开销）；面积 < 内存控制器 10%。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：RTL 落地于 Cobalt 200 内存控制器（SRAM 计数器表 + DRFM 逻辑 + mode 状态机），Dafny 形式化验证算法，firmware 支持受限运行时重配置（confidential computing 下只允许更保守的改动，如降低阈值）；telemetry 只上报 mode-transition 事件（不含触发行地址）以兼容机密计算隐私。使用场景：DRAM 内置 TRR/PRAC 之外的系统侧叠加防御——即便 PRAC-enabled DRAM 上市，Microsoft 仍保留 Sigries 作 fallback 与独立 telemetry 信号。评估框架（微软内部）：QEMU 内部 emulator 生成指令 trace → Asim 内部 cycle-level SoC 模拟器（乱序执行/分支预测/TLB/cache）出带时间戳的 memory request trace → RHSim（内部 timing-accurate 内存控制器+Rowhammer 模拟器，只模拟 row activation、tRC 约束、含 REF/RFM/DRFM）。论文 PDF：https://stefan.t8k2.com/publications/isca/2026/sigries.pdf；Sigries 与全部模拟器未开源。

涉及论文标题：
- From Lab to Fleet: Building and Deploying a Practical Rowhammer Defense in Cloud SoCs
