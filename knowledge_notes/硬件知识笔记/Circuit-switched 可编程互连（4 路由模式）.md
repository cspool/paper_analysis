## Circuit-switched 可编程互连（4 路由模式）

术语解释
- Morphatron 的 32 B 宽全局片上互连：每个 switch 5 端口（N/S/W/E/Local）+ 5×5 crossbar，仅支持 4 种固定路由模式（2 个配置位），graph/tree morpha 用地址比较器按地址范围动态选向；不设通用 packet-switched 网络，也无独立第二层。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 设计取舍：CGRA/数据流架构需要任意空间映射 → 互连复杂、布局布线困难；本文因只支持五种 morphas 而只需要四种路由模式（每 switch 2 配置位确定整个 switch 的路由模式）。静态路由用于数据流确定性的 morphas（systolic/vector/queue：MORPHA_STATIC_CONFIG 编程 crossbar 方向位）；动态路由用于 graph/tree（MORPHA_DYN_CONFIG 编程地址比较器：给定输出端口 + 地址范围 [lower, upper]，落在范围的访问走该端口——支持按顶点→核静态映射寻址）。冲突处理：systolic 需多端口对同时通信时用少量附加线 L_sys_in/out 与 mux 解决 crossbar 冲突；collective 模式由 2-bit mode register 激活（BROADCAST/COLUMN_REDUCE）复用同一数据通路。代价：互连 + Controller 合计 11% 面积、7% 能耗；重配延迟 Systolic/Queue 165 cycle、Graph/Tree 288 cycle。
- 从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 运转流程例子（graph morpha 顶点更新）：静态顶点→Morpha Core 映射 → 某核产生给邻居顶点的更新 → 该核经 REMOTE_STORE 发出（数据/队列 ID + 远端地址）→ switch 的地址比较器按 [lower, upper] 判定出口方向（N/S/W/E/Local）→ 逐跳转发到目标核的共享队列；相间阶段全部核 standby，互连充当"二维内存子库集合"供更新写入传播。shape-shifting：Controller 经 32-bit 控制网络写配置 → 2 位模式切换路由 → power-gate 未用端口。DMA 单元各自经自己的 switch 接入互连，Global Buffer 1/2 作 systolic 的输出/输入缓冲。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- Web 证据：circuit-switched NoC 每跳免路由/仲裁、能效高（Linköping 大学 FPGA 对比：4 端口 CS switch 约 4700 gates vs packet-switched 约 29000），但建立电路有延迟、大数据量（>~33 words）才划算、随规模增大电路竞争加剧；packet-switched 灵活但缓冲（FIFO）面积功耗大。本文用法：以"负载数据流可枚举"为前提，用 4 种固定模式换取极简硬件，再以 power gating 覆盖未用部分。

涉及论文标题：
- Accelerator Polymorphism: Transcending Domain-Specific Architectures with Robotics
