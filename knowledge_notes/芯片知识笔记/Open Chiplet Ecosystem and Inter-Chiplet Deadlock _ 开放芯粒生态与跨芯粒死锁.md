## Open Chiplet Ecosystem and Inter-Chiplet Deadlock / 开放芯粒生态与跨芯粒死锁

术语是什么？

Open Chiplet Ecosystem 是一种模块化芯片设计范式，目标是将传统 monolithic SoC "解耦"为多个可复用的 chiplet（芯粒），不同供应商的 chiplet 可通过标准化 die-to-die 接口（如 UCIe）直接互连，无需了解彼此内部实现细节即可组成完整系统。UCIe 等标准定义了 physical layer 和 link layer，但不保证互连网络在拓扑组合后仍然 deadlock-free。Inter-Chiplet Deadlock 指的是：即使单个 chiplet 内部的 2D NoC 已经是 deadlock-free 的，当多个 chiplet 通过 active interposer 互连后，chiplet NoC 与 interposer NoC 之间可能形成跨 chiplet 的 cyclic channel dependency（循环通道依赖），导致包在 vertical channel、边界路由器和片内 NoC 之间互相等待。

从芯片设计角度拆解术语：

在 open chiplet ecosystem 中，集成方希望像插积木一样组合不同供应商的 chiplet。但 deadlock 问题破坏了这一愿景：已有的 inter-chiplet deadlock 处理方法（MTR 的 turn restriction、DeFT 的 VC isolation、RC 的 permission network）都将 deadlock 处理逻辑与 chiplet 内部 NoC 实现绑定——要么需要特定转向限制，要么要求额外 VC 划分，要么需要片内控制网络。这意味着集成方必须知道每个 chiplet 内部 NoC 的拓扑、路由算法、VC 分配，供应商也可能需要为每种互连 floorplan 修改 NoC 设计，这与"plug-and-play chiplet"的模块化目标矛盾。本文的 DFBM 通过将 deadlock 处理逻辑外置到 bridge module（位于 interposer 侧），使 chiplet 供应商只需提供少量协议参数（coherence state machine 依赖、最大 outstanding request 数、VC 数量），无需暴露或修改内部 NoC 细节。

术语一般如何实现？如何使用？

UCIe 标准（Universal Chiplet Interconnect Express）自 2022 年发布，2025 年已演进到 3.0 版本（支持 64 GT/s），定义了 die-to-die 物理层、链路层和协议层。Arm 的 Chiplet System Architecture (CSA) 补充了系统级设计原则。但 deadlock 这一上层问题仍需架构级解决：本文的 DFBM 是一种 avoidance-based 方案，通过 credit management + shared deadlock buffer 消解跨 chiplet 的循环依赖；MTR（turn restriction）、DeFT（VC isolation）、RC（injection control）是同一问题的其他 avoidance 方案；UPP、Steered Bubble 则是 recovery-based 方案（允许死锁先发生再恢复）。这些方案的 trade-off 取决于对面积、性能、portability 和供应商集成的综合考量。

涉及论文标题：
- Deadlock-Free Bridge Module for Inter-Chiplet Communication in Open Chiplet Ecosystem
