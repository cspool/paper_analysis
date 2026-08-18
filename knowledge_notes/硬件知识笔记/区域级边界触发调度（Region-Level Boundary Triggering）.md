## 区域级边界触发调度（Region-Level Boundary Triggering）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
区域级边界触发是 WaferBRAIN NAHP 范式的关键机制：跨区域（inter-region）单播只在区域边界节点发起，绝不由区域内部节点直接发起，从而避免单播集中在放电源附近（hotspot）并共享区域内广播路径。机制（Algorithm 1 Deterministic Axon-to-Boundary Assignment）：对源节点 (i,j) 在 N×M 区域中的每个跨区域轴突，从 4 个边界候选 {(i,0), (i,M-1), (0,j), (N-1,j)} 中选唯一 owner——取到路由锚点 (i_a,j_a) 的最小曼哈顿距离（同 wafer 单播锚点=目标节点；跨 wafer/POD 单播锚点=所选 WC egress 所在节点），平局按字典序保证唯一。owner 持有该轴突的 axon-out 元数据并负责发起远程单播。效果：源节点只发 mode=0 区域内广播；边界 owner 收到后发 mode=1 单播到 <POD,Wafer,Node>。全局传播 = 两阶段（区域内广播 + 边界触发单播），把单播热点从源扩散到区域边界；每个边界节点神经元放置减 20%、被置换神经元区域内均匀重分布以保持存储与广播扇出对称；每个神经元 k 个跨区域轴突时，平均累计路由距离每神经元减少 (N+M-2)·k/4。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
硬件运转流程（一次跨区域 spike）：① 源 BPU 的 axon-out 只发 mode=0 广播包（LNid=<LNode,Neuron>），区域路由表（even/odd 源列/行扩展）把事件送达区域内全部节点；② 区域内的边界 owner（配置期由 Algorithm 1 确定性分配、元数据预存于 owner 的 SRAM/DRAM）在收到事件后查其持有的 axon-out 条目（<DstNode, GAid>），发 mode=1 单播包；③ 单播走 egress-WC 选择 → WC 确定性路由 → dragonfly → 目标 wafer mesh-XY；④ 接收端按 GAid 索引全局突触块。Algorithm 1 伪代码：输入区域 N×M、源 (i,j)、目标；同 wafer 时锚点=(i_d,j_d)，否则锚点=intra-wafer exit point（WC egress 节点）；边界集 B={(i,0),(i,M-1),(0,j),(N-1,j)}；遍历 B 求 d=|i_a-i_c|+|j_a-j_c| 最小者（字典序破平）返回 (i_b,j_b)。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：配置期确定性分配（Algorithm 1 离线执行，结果 = 每轴突 → 边界 owner 的映射，写入 owner 的 axon-out 元数据）；硬件只需"边界节点可发单播、内部节点只发广播"的路由器规则 + 单播目标物理地址。使用场景：大脑尺度（95% 本地 + 5% 长程连通）的跨区域 spike 投递——替代 axon-centric 的"每个源独立远距离单播"，把单播注入点挪到离目标最近的区域边界；效果：NAHP 全局流量比 neuron-centric 降 1.3-360×、比 axon-centric 更低的 R_Global（式 5），且因减少源附近单播拥塞直接贡献 1ms 实时界内更高可持续 firing rate。局限（论文未明确说明）：边界节点承担额外 axon-out 元数据存储与单播注入负载（以 20% 神经元放置折中），区域/目标分布改变时需重新做确定性分配。

涉及论文标题：
- WaferBRAIN: Whole-Brain Scale Neuromorphic Architecture Based on Wafer-Scale Integration
