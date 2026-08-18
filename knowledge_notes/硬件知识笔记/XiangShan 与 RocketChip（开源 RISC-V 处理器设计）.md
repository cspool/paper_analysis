## XiangShan 与 RocketChip（开源 RISC-V 处理器设计）

术语解释
本文验证工作负载：XiangShan（RVA23 SoC，3,451,036 LOC，Spec2006 负载，compute 压力）与 RocketChip（RV64GC 核，51,721 LOC，compute 压力），另有 CoupledL2（86,104 LOC，PIPT 1MB 缓存子系统，memory 压力）。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
XiangShan（香山）是中科院计算所 OpenXiangShan 团队开发的高性能开源 RISC-V 处理器，Chisel 编写，三代微架构（雁栖湖/南湖/昆明湖，Gen3 支持乱序执行与 RISC-V 向量扩展），Hot Chips 2024 上被定位为"处理器界的 Linux"，Mulan PSL v2 许可——本文作者单位（ICT CAS 处理器国家重点实验室、北京开源芯片研究院）即其开发方；RocketChip 是 UC Berkeley 的 RISC-V SoC 生成器（Chisel，Diplomacy/TileLink 总线框架），RV64GC 标量核是其核心配置之一；CoupledL2 是香山项目的 L2/L3 缓存子系统（huancun）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
作为验证负载的架构特征：XiangShan 是大设计/compute 压力——内存负担主要来自仿真逻辑（代码段），多实例共享文本段收益大（UCV 单进程多线程比多进程内存降 52%）；CoupledL2 是 memory 压力——状态存储管理主导内存，多实例共享收益微弱（图 Fig.12 的差异化归因：文本段 vs 数据段占比不同）。验证难点：BPU 流水多级重叠预测且预测器表在 SRAM（波形调试不可见，需 MemD 表视图）、NoC/ICache 需复用 UVM VIP 事务交互、Router/RAS/Decoder 为社区任务载体（6 个月社区研究在 XiangShan 发现 30 个此前未知 bug）。生态：XiangShan 复用 RocketChip 的 Diplomacy/TileLink 与 berkeley-hardfloat，difftest 做同仿对比。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
获取：XiangShan https://github.com/OpenXiangShan/XiangShan （文档 https://xiangshan-doc.readthedocs.io ）、RocketChip https://github.com/chipsalliance/rocket-chip （Chipyard 集成）。本文用法：作为 UCV 评估负载与社区任务平台——经 Verilator 编译为 C++ 模型（8 线程）、由 Picker 打包为软件包后以 pytest 等驱动；XiangShan 全量评估在 2×EPYC 7773X 上约 23 小时。Web 证据：Hot Chips 2024（https://arxiv.org/abs/2407.18765 相关介绍见项目文档）、半导体系介绍（https://semiiphub.com/pulse/technical-articles/an-open-source-approach-to-developing-a-risc-v-chip-with-xiangshan-and-mulan-psl-v2 ）。

涉及论文标题：
- Democratizing and Accelerating Hardware Verification with Software-Native Optimization
