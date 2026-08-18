## SimpleScalar（执行驱动架构模拟器）

术语解释
SimpleScalar 3.0 是 Todd Austin 团队（Wisconsin）开发的开源执行驱动架构模拟器工具集，其 sim-outorder 建模乱序超标量+推测执行处理器；本论文在其 3.0a 基础上自建现代乱序核模拟器（移除 RUU、加入 ROB/PRF/IQ/RMT、TAGE、L3、stream prefetch 与 ChampSim DRAM 时序）来评估 HWL。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
逻辑链：(1) 执行驱动模拟器随程序执行推进模型微架构各部件（对比 trace-based）；(2) sim-outorder 建模乱序发射、推测执行、两级 cache、多类分支预测器（bimod/2lev/comb）、RUU（reorder+reservation station 合一，默认 16 项）与 LSQ（默认 8 项），关键参数 -ruu:size/-issue:width/-lsq:size/-bpred/-cache:* 等可配置；(3) 学术非商业用途免费（www.simplescalar.com，simplesim-3v0d.tgz），GitHub 镜像 github.com/toddmaustin/simplesim-3.0；(4) 论文修改（[33] 引用）：移除原 Register Update Unit [34]，换成独立的 reorder buffer + 物理寄存器堆 + IQ（random queue with age matrix）+ RMT；加 TAGE 预测器 [35]、L3 cache、stream prefetcher；集成取自 ChampSim [36] 的 DRAM 时序。Web 证据：多个课程/研究使用 sim-outorder 做乱序超标量研究（UW CSE 471 指南、TAMU CSCE614、Pittsburgh 的 8-way OoO 研究等）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
运转流程：输入 = Alpha ISA 二进制（SPEC2017 基准，gcc 用 SPEC2006；每个基准用 SimPoint 从 reference inputs 选单个代表性 100M 条指令区域）+ 微架构配置（Table I：6-wide 前端/12-wide 后端、512 ROB、12-issue 200×200 IQ、200 LSQ、512+512 PRF、TAGE 64KB、L1 32KB×2、L2 1MB、L3 4MB、3200MT/s、stream prefetch；Table II HWL 参数）；逐周期推进 fetch→decode→rename（含 segment 分配电路行为建模）→dispatch（HSD+混合模式）→issue（L1/L2 唤醒+select）→execute→commit；输出 IPC/周期数与相对基线退化。作用：在无硬件条件下验证 HWL 的 IPC 影响（0.9% 退化）与 scaling 收益（1.5× 配置 +17.2%）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：C 语言模拟器（sim-alpha/sim-outorder 等目标变体），可改核心模型；本论文即深度修改（新 OoO 核心 + 预测/预取/DRAM）。使用：编译目标模拟器 → 提供二进制与配置 → 运行取 IPC 等统计。开源（github.com/toddmaustin/simplesim-3.0）；HWL 自身未开源，但可基于该模拟器框架复现论文的 HWL/HSD/混合派发建模。

涉及论文标题：
- Hierarchical Wakeup Logic of the Issue Queue for High Scalability
