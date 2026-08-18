## BPU（Brain Processing Unit）与事件驱动神经形态 BPU 流水线

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
BPU（Brain Processing Unit）是 WaferBRAIN 晶圆级芯片的基本处理节点：每个 die（23mm×32mm）集成 4×4=16 个 BPU，每个 BPU 含一个轻量 NoC 路由器（支持 unicast + broadcast 投递 spike 事件）与四个功能模块，构成事件驱动 SNN 计算流水线：(1) **axon-in**——摄取本地/全局 AER spike，在 3D-DRAM 做直接 pointer/fan-out 查表，把 spike 映射到目标突触列表（本地或全局），提供发起访存所需元数据；(2) **dendrite**——DMA 引擎从 3D-DRAM 拉取邻接表格式突触数据，每个突触权重与元数据解码后按目标神经元 ID 分发到不同 FIFO，硬件实现稀疏突触索引与事件驱动、异步并行的跨 FIFO 突触电流累加，结果交给 soma；(3) **soma**——SRAM 中维护神经元状态（膜电位等），按神经元模型更新膜电位，阈值穿越产生新 spike 转发给 axon-out；(4) **axon-out**——查 axon-pointer/fan-out 表确定 BPU/chiplet 内外的目标，打包并分发 spike（驱动 NoC unicast 或 broadcast）。对应的流水线语义：Axon.in 汇集输入事件 → Synapse 稀疏权重查找/衰减（memory-bound）→ Dendrite 分段归约累加突触电流 → Soma 膜电位积分与阈值发放（compute-bound）→ Axon.out 广播/边界触发散射 + NoC 路由（communication-bound）——每步在 1ms 实时预算内执行 trillions 级事件操作，异质瓶颈并存。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
BPU 的硬件运转流程（一个输入 spike 到输出 spike）：axon-in 接收 AER 包并查 3D-DRAM pointer/fan-out 表 → 得到本地/全局突触列表（本地：LNid 索引 (L.Syn.Pointer, Fanout)；全局：GAid 索引）→ dendrite 的 DMA 引擎按 contiguity-aware 邻接块批量拉取 (DstNeuron, Weight) 条目（coalesced DMA、compact pointer/length 头）→ 解码后按目标神经元分发到各 FIFO、异步并行累加 → 累加结果传 soma → soma 按神经元模型（如 LIF）更新 SRAM 中的膜电位/refractory、阈值穿越产生 spike → axon-out 打包（mode 0 广播 LNid 或 mode 1 单播 <POD,Wafer,Node>+GAid）→ NoC 路由器按模式位转发。存储分工：热状态（膜电位、refractory、临时累加器、事件队列）在片上 SRAM，突触/连接元数据在 3D-DRAM（每 die 40GB）——"compute-communication-memory" 三者在 BPU 内被协同平衡，DRAM 延迟由 coalesced DMA 与流水隐藏。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：BPU 是通用神经形态 core 的可定制形态（类比 ELSA 的 neural core：定制路由器 + 4 PE、128 ST-BIF 神经元电路、N 路权重 SRAM；TrueNorth 的 neurosynaptic core；Loihi 的 manycore 核），WaferBRAIN 把四模块以硬件流水 + 3D-DRAM 邻接表寻址落地，评估用自研分析模拟器（未开源）。使用场景：大规模事件驱动 SNN 仿真（1B 神经元单 wafer、100B 全脑多 wafer）——axon-in/dendrite/soma/axon-out 四模块对应事件驱动管线四阶段，soma 计算密度与 dendrite 访存、NoC 通信异质瓶颈由 NAHP 范式 + SRAM/3D-DRAM 层次 + mesh/dragonfly 拓扑联合消解。

涉及论文标题：
- WaferBRAIN: Whole-Brain Scale Neuromorphic Architecture Based on Wafer-Scale Integration
