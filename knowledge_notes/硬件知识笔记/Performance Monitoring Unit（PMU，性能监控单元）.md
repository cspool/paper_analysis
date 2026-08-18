## Performance Monitoring Unit（PMU，性能监控单元）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
PMU 是硬件性能监控单元（Performance Monitoring Unit），现代 CPU/GPU 普遍内置的固定功能计数器硬件，是 in-field 性能分析的主力工具。其工作方式：把"数据源"和"分析"硬编码进单个计数器（如"数 L2 miss"），每个 PMU 事件对应一个预设答案，近乎零开销、无处不在。核心局限是刚性（inflexibility）：只能数设计者事先预料有用的事件，无法适配部署后新出现的问题，且隐藏真实片上事件的细节（如 miss 的根因、内部决策点状态）。在论文的四轴分类（Speed/Transparency/Programmability/Accessibility）中，PMU 具备 Speed 但 Programmability 受限（仅可配置预设事件）、Transparency 受限（粗粒度计数）、无法做有状态仿真（无状态计数器）。论文把 PMU 定位为 IPU 的对照 baseline 之一：固定 PMU 数 N 个事件得 N 个答案，而 IPU 的 32 个 HIT 信号可组合出大量可编程分析程序。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
PMU 在硬件架构中的运转流程：设计时——芯片设计者为每种想监控的事件（L2 miss、TLB miss、分支误预测等）定义事件选择与计数通路（事件源信号 → 事件选择逻辑 → 计数器寄存器）；运行时——软件（Linux perf_event/perf、CUPTI、NVIDIA Nsight Compute）经 MSR 或 MMIO 配置要计数的事件，PMU 每周期对匹配事件加 1，计数器溢出触发中断或采样（PEBS/IBS 类），主机读取计数做分析。论文的对照例子：perf stat 读 PMU 计数器只能得到"L2 miss 总量"这类聚合值，无法回答"这个 PC 的 miss 是冷启动还是预取失败"，也无法做 GPU TensorCore 的 256-cycle 细粒度时间窗直方图（聚合 PMU 只报告整个 kernel 的误导性平均）。IPU 与 PMU 的关系：IPU 把数据源选择（32 个原始信号）与分析方法（任意软件）解耦，PMU 的定位被泛化为"IPU 可编程实现的一种特例"；IPU 的信号选择方法论（三轴覆盖）也与 PMU 事件选择不同——不预设问题、只暴露原语状态。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：PMU 是标准芯片 IP（Intel/AMD x86、ARM、RISC-V 均有实现），由计数器阵列、事件选择寄存器、溢出中断逻辑组成；Linux perf_event 子系统（https://perf.wiki.kernel.org）与 NVIDIA CUPTI/Nsight 是主流使用接口。使用方式：开发者在热点分析中配置事件并读取计数（如 perf stat -e cache-misses,cycles），采样工具（Intel PEBS、AMD IBS、ARM SPE）附加调用栈；局限在论文的 Capability 2/4 中具体化——perf 采样无法生成 per-instruction cycle stacks（PICS 需专用 TEA 硬件），PEBS 无预取器内部状态可见性（无法做逐 miss 根因分类）。论文未把 PMU 作为开源项目（其为通用现有技术）。

涉及论文标题：
- Enabling Continuous, In-Field Introspection: The Programmable IPU Architecture
