## 层次化两级 2D Mesh（Hierarchical Two-Level 2D Mesh）与 Die-to-Die（D2D）链路

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
层次化两级 2D mesh 是 wafer-scale chiplet 集成系统（BusyBarn 目标平台）的芯片级互联组织：上层为 die 级 2D mesh（多个已知良好 die 经硅 interposer 互连成规则网格），下层为每个 die 内部的 core 级 2D mesh NoC。相邻 die 通过 Die-to-Die（D2D）链路通信，每条 D2D 链路由多个 SerDes lanes 组成并做协议翻译（如 AXI→UCIe）[64]；每个 die 内部的 NoC 经路由器互连同构 core 与外部 I/O 组件，部分路由器直连 HBM 接口、其余连 D2D 链路，HBM 与 D2D 接口的放置需平衡 die-to-HBM 与 die-to-die 带宽（论文采用 HBM 置于 die 角落的拓扑 Fig.2a）。die 内 core 经片上硅与金属层互联，不同 die 与 HBM 经 interposer 上的 D2D 链路相连；先进封装（CoWoS：interposer 金属布线 + TSV）增强 D2D 互连与 die-封装基板连接。该设计根本区别于 Cerebras WSE（单片晶圆 + 片内 SRAM 空间并行）：BusyBarn 系统每 die 有本地 DDR/HBM，通过时空复用扩展调度空间并满足 LLM 大内存需求，同时减少服务单一大模型所需设备总数。

从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
在芯片设计中的运转流程（Fig.2 硬件结构 + Table I 参数）：die 级 2D mesh 上每条 D2D 链路延迟 20 ns、带宽 256 GB/s（多 SerDes lanes 聚合）；die 内 core 级 mesh 的 on-chip 链路 1 ns、256 GB/s；off-chip HBM 100 ns 延迟、256 GB/s 每 die、8 GB 容量。一次跨 die 通信的数据路径：源 die 内 core → 片上 NoC 路由器（AXI 域）→ die 边界 D2D 接口做 AXI→UCIe 协议转换 → SerDes lanes 跨硅 interposer 传输 → 目标 die D2D 接口反向转换 → 目标 die NoC 路由器 → 目标 core。端到端评估使用三个 wafer 拓扑：HW1 5×5 mesh（类 Tesla Dojo）、HW2 7×12 mesh（类 Cerebras 外形）、HW3 8×8 mesh（自设计）；ablation 用 6×8 die 阵列、每 die 16×16 core、1.02 TFLOPs/core、每 die 边 1.5 TB/s D2D（DOJO 风格 WSC-LLM）。层次化网格的不对称性（非均匀节点度、异构片内/片间带宽）破坏对称 collective 的对称性假设，是 BusyBarn 层次化映射与 BALD 通信调度的直接动机。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现与使用：die 级 mesh 由先进封装工艺支撑——TSMC CoWoS 类 interposer 布线 + TSV；D2D 链路用多 SerDes lanes 与 UCIe 协议（UCIe 是 chiplet 间 die-to-die 互联的开放标准，见"UCIe"条目）；die 内用 2D mesh NoC（路由器 + 链路，见硬件架构层"2D Mesh NoC"条目）。评估通过自研事件驱动模拟器（10K+ 行 Python，开源 artifact：https://github.com/redbird-arch/isca2026-busybarn-artifact.git）建模两层 mesh 的带宽/延迟并做映射与路由调度；相关工程实践可参考 Tesla Dojo（25 die 5×5 mesh、每 die 1000 TFLOPS + 80GB HBM）、TSMC SoW（24 compute die + 96 HBM die 单晶圆 >200,000 mm²）。使用场景：LLM 推理（TP/PP 混合并行跨 die 组、每 die 内 SP/CP/TP 混合并行）、故障容错部署（D2D 链路或 die 故障时重路由）。

涉及论文标题：
- Mapping and Communication Optimizations with Fault Tolerance for Wafer-Scale LLM Inference
