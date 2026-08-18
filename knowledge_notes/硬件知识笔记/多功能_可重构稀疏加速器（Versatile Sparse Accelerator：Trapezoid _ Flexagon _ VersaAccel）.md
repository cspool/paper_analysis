## 多功能/可重构稀疏加速器（Versatile Sparse Accelerator：Trapezoid / Flexagon / VersaAccel）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
多功能/可重构稀疏加速器是一类在单一硬件基板上支持多种稀疏执行数据流（InP/Row/OutP）的加速器，处于"固定数据流专用加速器（SIGMA、HighLight，窄场景高利用率但泛化差）"与"全可重构灵活加速器（Flexagon，多模板但重互连高开销）"之间的设计空间（Harmonia Fig.1）。代表：Trapezoid——保持同构 PE 阵列，只重配置 DN（输入广播）与 MRN（psum 归约）两个部件，在 InP/Row 之间切换，从稠密 DNN 层到高稀疏 SpGEMM 保持近常效；VersaAccel——同类同构阵列+轻量重配置的中间路线。Harmonia 明确指出这类硬件的局限：灵活性仅停留在结构层（可切换模板但缺系统机制在运行时按 tile 选择数据流），导致负载不均、访存不协调、复用退化——这正是它要补的"运行时调度层"。Harmonia 本身即基于 Trapezoid 平台（32 行、1MB SRAM、HBM 2TB/s）扩展反馈计数器/重构引擎/tiling 控制器而成。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
架构运转：32 PE 行半独立（semi-independent）执行——每行一条水平 PE 链处理独立非零对，经可配置 MRN 累加；DN 把操作数按模板路由到 PE，支持 InP/Row/OutP（tile 边界重编程路由表与归约模式）；on-row buffer 暂存操作数与 psum，spill 计数器与占用监视器把压力事件暴露给运行时；全局 SRAM 存 tile 与元数据（含逐 tile nnz）支持执行前 profiling。Harmonia 的加法：Feedback Counters + Reconfiguration Engine + Tiling Controller 构成 SW-HW 调度接口（合计 3.3% 面积），把"能切模板"升级为"按 tile 自动切模板"（逻辑异构）。能耗结构：SRAM 访问 ~50%、数据路由与稀疏控制 ~20%、计算 20–30%；相比全灵活架构省掉重型结构功耗。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
评估/使用方式：Harmonia 在所有 baseline（Trapezoid 上分别装 Vesper/Misam/HYTE 调度）运行在同一 32-row 配置上，用自研 cycle-accurate 模拟器（建模 MAC、MRN、DN、本地 buffer、SRAM、HBM；显式注入 20–50 cycle 重构开销）+ TSMC 28nm RTL 综合（面积 7.51 mm²）+ CACTI 7（SRAM）+ HBM datasheet（能耗）。workload：SuiteSparse SpMSpM（bcsstk10、email、orani678、rajat19 等）+ 剪枝 DNN（LLaMA-7B/OPT-1.3B 经 SparseGPT 剪 0.2/0.4/0.6，ResNet-50 经 STR 剪 0.1/0.2，VGG-16 幅度剪枝 0.1/0.32）。结果：平均 1.75× 加速（orani678 3.46×）、SRAM 流量降 32%、能耗 -40%、端到端 DNN 1.87×。论文未提供开源实现；Trapezoid 为 ISCA'23 工作。

涉及论文标题：
- Harmonia: A Unified Hierarchical Scheduling Framework for Sparse Matrix Multiplication

Spada 与 SegFold 补充视角（ISCA'26，可重构稀疏加速器的两条"自适应"路线）：Spada（ASPLOS'23，Tsinghua）是 runtime-adaptive 的 SpGEMM 加速器，提出 window-adaptive (WA) dataflow——在 tile 粒度按矩阵 A 的稀疏度动态调整 window 的高度与宽度，配合相邻 lane 的机会性 work stealing 提供局部负载均衡；但其 tile 内调度仍由静态循环结构决定，自适应粒度限于 tile 级。SegFold 把自适应推到 sub-tile 细粒度：不再切换"window 形状"模板，而是逐周期动态重排 (m,k) 选择（SELECTA）并在 PE 间动态重映射部分和（SEGMENTBC + 自适应 merge network + folding），即"动态数据流"而非"动态模板选择"。SegFold 以 Spada 为最强 baseline（直接使用其开源模拟器 spada-sim 不变），在高度稀疏矩阵上 1.08×–5.75×、整体 geomean 1.95× over Spada；ca-GrQc（0.59×）例外——无标度图的极密行压垮 SegFold 的 per-row PE 分配，而 Spada 的 tile 级 window 适应反而更稳，说明动态粒度过细也有代价。Flexagon 在 SegFold 评估中作为静态多数据流 baseline（经 STONNE 模拟 + Ramulator 内存后端，1D 128-PE 扩展为 2D 2×128），其最佳 per-tile 配置仍比 SegFold 慢 5.3× geomean。

涉及论文标题：
- Harmonia: A Unified Hierarchical Scheduling Framework for Sparse Matrix Multiplication
- SegFold: Accelerating Sparse GEMM with a Fine-Grained Dynamic Dataflow
