## 逻辑综合与 Place-and-Route 物理实现流程（ASIC Synthesis & PnR，含工艺节点缩放与 SRAM 宏）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
逻辑综合（logic synthesis）把 RTL 网表映射到标准单元库并满足时序/面积/功耗约束，Place-and-Route（PnR）再做标准单元布局（placement）、时钟树综合（CTS）与详细布线（routing）生成可流片版图；配合门级仿真与功耗分析（基于 VCD 切换活动）构成完整 ASIC 物理设计流程。逻辑链：微架构创新（如 GPU OoO 执行）在软件仿真器中无法暴露电路级效应（关键路径/频率退化、互连非线性扩展、切换活动功耗），只有走综合+PnR 才能在真实 PDK 上量化 PPA（Power/Performance/Area）。sCROOGe 的流程：SystemVerilog → Synopsys Design Compiler v2022.12 综合（GlobalFoundries 22nm FDSOI，TT corner / 0.9V / 25°C）→ GF Memory Compiler 生成 SRAM 宏（RF、local memory、cache 模块）→ 门级仿真验证功能 → Synopsys PrimeTime 分析 post-synthesis VCD 切换活动估功耗（含 UPF 功耗意图建模）→ Cadence Innovus v23.11 全 PnR（400 MHz 保守时钟）交叉验证 → 扩展至 IMEC N2 2nm GAA Nanosheet pathfinding PDK（1.2 GHz）确认趋势。关键量化结果：面积随频率 5× 缩放仅增 ~6%，功耗增 4×（综合工具靠加大标准单元驱动强度满足时序）；PnR 后线负载电容使开关功耗占总量 ~20%（综合阶段仅 ~4%），但各设计一致，功率开销偏差 frontend 0.8% / backend 1.7%。

从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
在 sCROOGe 的运转流程（Fig.19-24、Table III/IV 数据来源）：① RTL 输入（Vortex+sCROOGe 的 frontend/backend 扩展）→ DC 以目标频率约束（最高 1GHz，400MHz 用于保守 PnR）综合到 GF 22nm FDSOI 标准单元 → 输出门级网表 + 面积/时序报告（Table III：pipeline 各阶段关键路径延迟——Schedule 634ps、Issue baseline 295ps / frontend 600ps / backend 901ps、Commit→Issue 353/390/466ps、Execute 993ps 为全局关键路径且不因 OoO 修改变化）；② SRAM 部分用 Memory Compiler 宏实例化（RF/本地内存/cache），保证物理布局真实性；③ 门级仿真 + PrimeTime 按 VCD 切换活动算功耗，并对 {voltage, frequency} 有效工作点（UPF + GF 22nm PVT corner 库插值）测 GOPS/W 趋势（电压平方增功耗、频率线性增吞吐 → GOPS/W 近线性下降）；④ Innovus 做 placement/CTS/routing，与综合结果对比验证功率/面积趋势稳定（PnR 后开关功耗占比从 ~4% 升至 ~20%，源于时钟树缓冲与金属寄生 RC）；⑤ 2nm N2（GAA Nanosheet）重综合验证：线负载开关功耗相对总功耗 +17%（互连 RC 不随晶体管等比例缩小），但 OoO 方案无宽线主导数据通路，相对功耗增幅略低于 22nm。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现/工具：Synopsys Design Compiler（综合）、Synopsys PrimeTime（静态时序与功耗，配合 VCD/UPF）、Cadence Innovus（PnR）、GF 22nm FDSOI 与 IMEC N2 PDK、GF Memory Compiler（SRAM 宏）。使用：sCROOGe 对单 SM（1 processing block + 16KB L1）配置做 PPA 评估并扫描 {warp=4..64, thread=4..32} × {IsB/CU/RRS} 设计空间（综合为主，PnR 只验证 Fig.23 精选点，因全 PnR 周期过长）；DRAM 行为用 Ramulator 建模（与芯片流程互补）。关键约束与限制：PDK/标准单元库/Memory Compiler 受 NDA 无法开源，sCROOGe 只发布 RTL 源码 + 面积/功耗 .csv（Fig.19-24），有授权的审稿人可用相同工具版本复验；论文明确指出综合 vs 仿真（GPUWattch/McPAT/AccelWattch）的面积功耗评估差距（LOOG 面积开销仿真 1.28% vs RTL 21.50%，GhOST 功率 0.67% vs 6.70%），即合成流程是揭示真实芯片成本的必要手段。

涉及论文标题：
- sCROOGe Circuit-level Design and Optimization Framework for RISC-V Out-of-Order GPUs
