## Sniper 模拟器（interval simulation）

术语解释
Ghent 大学与 Intel ExaScience Lab 联合开发的开源 x86 多核模拟器，以 interval core model（interval 仿真）为核心抽象；ATX 论文在其内部 silicon-validated 版本上扩展 NCA/ICA/OCA/ATX/UTE 模型。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Sniper 用"interval"（区间）而非逐周期推进核模型：把执行时间切成由长延迟事件（cache miss、分支误预测、窗口满等）分隔的区间，区间内按 IPC 解析模型推进，从而在保持机械式（mechanistic）微架构行为的同时获得数量级加速（相对 cycle-accurate 快，主机 8 核时约 2 MIPS 量级）。特色输出是 CPI stacks——把损耗周期按原因（分支误预测、cache miss、同步等）分解成柱状图，天然适合做瓶颈定位。开源（MIT/Interval Academic License），支持 x86-64、多线程（pthread/OpenMP/TBB）、与 McPAT 联用估功耗，官方站点 snipersim.org。ATX 论文的版本是 Intel 内部"silicon-validated"扩展：L1/L2 空间预取器对齐真实 SPR 硅片预取器行为，并增补了 ATX 所需的所有模块。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
ATX 论文的使用方式：在 Sniper 上配置 64 核 SPR-like 系统（2.5GHz、48KB L1/2MB L2/1.875MB LLC、DDR 1TB@270GB/s + HBM 64GB@4TB/s、L2 MSHR=128），扩展核模型加入 ATX 执行端口/ATX Queue/Reservation Station，扩展内存子系统加入 UTE（InTaskQ/Stream Units/LDQ/Common Bus/PDQ/Task Predictor）与 NCA/ICA/OCA 模型；跑 SpMM/SDDDMM/GeMM/解压 kernel 二进制，输出加速比、memory/core-bound 周期分解、roofline 点；UTE 面积/功耗另用 CACTI + 缩放方程离线估算。评估方法学要点：ICA 用"perfect ICA"（零计算时间）做 MLP 上限分析，OCA 两种形态分离"位置"与"调用模型"变量。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：Pin 动态插桩 + Graphite 并行基础设施 + interval core model；配置文件（sniper/config）用 Python 描述核心/缓存/预取器参数，`run-sniper` 跑应用并产 sim.out/sim.stats。扩展点：Rob 模型、cache 模型与新增功能模块通常以 C++ 组件挂接。适用场景：体系结构研究的快速设计空间探索（几十核规模、多核共享内存负载）；不适用于需要精确核内时序（流水线级）的研究——interval 抽象在此有精度上限（论文同作者的 HPCA'10 interval 论文给出与 cycle-accurate 的精度对照）。

涉及论文标题：
- ATX: Accelerator Task Extensions
