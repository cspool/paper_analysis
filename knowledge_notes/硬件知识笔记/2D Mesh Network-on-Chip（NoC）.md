## 2D Mesh Network-on-Chip（NoC）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
2D mesh NoC 是一种规则网格拓扑的片上网络：每个 tile（计算节点+本地存储）位于网格交叉点，通过路由器连接东/西/南/北四个邻居，数据包经路由算法沿链路逐跳传输。在 BusyBarn 的 wafer-scale 系统中，NoC 出现在 die 内层级——每个 die 内用 2D mesh NoC 互连同构 core 与外部 I/O 组件（路由器连 HBM 接口或 D2D 链路），core 数据包在 mesh 上按最短路径（XY routing 或其变体）转发；die 级网格（D2D 链路）与 die 内 NoC 构成层次化两级 2D mesh。Table I 中 on-chip link 延迟 1 ns、带宽 256 GB/s。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
在硬件架构中的运转流程（BusyBarn 的 mesh 通信调度）：通信任务（point-to-point 或 multicast）由数据依赖生成→路由算法为每个任务选择路径→数据包逐跳经过 NoC 路由器占用链路→链路共享导致争用（如 Fig.7 4×4 mesh 上两个并发 multicast 共享 (9,10)(10,11)(11,7) 形成热点）→BALD 算法在调度时为每条链路分配流量以均衡负载并缩短距离。mesh 上同源到同目标常有多个等长最短路径（如 8→5 有 ((8,9),(9,5)) 与 ((8,4),(4,5)) 两条），BALD 的 path profiling 记录这种非唯一性供调度利用；故障链路/节点会破坏 mesh 对称性并产生 detour，XY 等固定顺序路由在故障下表现差（见"XY 路由与 XY-YX-FT"条目）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现与使用：NoC 由路由器 + 双向链路组成，路由可用确定性（如 XY/DOR）或自适应算法；mesh 拓扑在学术/工业加速器中广泛使用（Cerebras WSE 的 core mesh、Tesla Dojo、HD-MoE 的 3D NMP 节点网）。评估工具：事件驱动模拟器（BusyBarn 自研、HD-MoE 的 discrete-event simulator）、BookSim2、gem5-Garnet、ASTRA-sim（后者常用于 ring all-reduce 延迟校验）。使用场景：LLM 推理的 TP 部分和归约、PP 层间激活传输、MoE 的 all-to-all dispatch/combine。

涉及论文标题：
- Mapping and Communication Optimizations with Fault Tolerance for Wafer-Scale LLM Inference
