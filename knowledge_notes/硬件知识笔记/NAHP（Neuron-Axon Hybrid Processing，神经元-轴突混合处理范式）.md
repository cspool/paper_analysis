## NAHP（Neuron-Axon Hybrid Processing，神经元-轴突混合处理范式）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
NAHP 是 WaferBRAIN（ISCA 2026）提出的神经形态通信/处理范式，把既有 neuron-centric 与 axon-centric 两种范式的优势合二为一：本地区域内用神经元驱动广播（neuron-driven broadcast，mode=0，包只携带 LNid=<LNode, Neuron>，区域内路径复用）、跨区域用轴突驱动单播（axon-driven unicast，mode=1，包携带 <POD, Wafer, Node> 目标物理 ID + GAid，无冗余远场投递）。关键设计：(1) 连接按区域组织为层次——区域内部（~95%）稠密本地 fan-out 走广播，区域间（~5%）稀疏长程投射走单播；(2) 包格式压缩——LNid 27bit（区域神经元 ID）比 neuron-centric 的 FNid 37bit 少 10bit，GAid 33bit（NodeID 17 + GAid 23）比 axon-centric 的 NodeID&FAid 46bit 少 4bit；(3) 配套存储——本地/全局突触分开布局（见 BPU 与 3D-DRAM 条目），本地条目 2^27 个（区域神经元数）、全局条目 2^23 个，索引开销降 1.2-7,400×；(4) 边界触发（Region-Level Boundary Triggering，见专有条目）让单播只从区域边界节点发起。路由器按 0/1 模式位分类（见 NoC 条目）。评估（自研分析模拟器）：100B dragonfly 配置下平均流量比 neuron-centric 降 300×、比 axon-centric 降 6.9×（峰值 2.5-140×）；本地流量降 1.4-32×、全局流量降 1.3-360×；1ms step 界内可持续 firing rate 比 neuron-centric 高 14×、比 axon-centric 高 4.7×（NAHP 达 3.8%）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
NAHP 在硬件架构中的运转流程（一次 spike 的本地+全局两阶段传播）：源 BPU 的 soma 触发 spike → axon-out 打包：若目标全部在本区域则发 mode=0 广播包（LNid=<LNode, Neuron>），各路由器按 LNode 索引 5-bit 端口掩码表单周期转发（even/odd 源列/行扩展），区域内全部节点收到并按 LNid 索引本地突触邻接块（3D-DRAM，coalesced DMA 取 <DstNeuron, Weight> 列表）→ dendrite FIFO 累加 → soma 更新膜电位；若有跨区域目标，源节点只发区域内广播，由 Algorithm 1 选出的边界 owner（4 候选边界节点中最小曼哈顿距离到路由锚点）查 axon-out 元数据并发 mode=1 单播包（<POD,Wafer,Node>+GAid）→ egress-WC 选择 → WC 确定性路由 → dragonfly → 目标 wafer mesh-XY → 接收端按 GAid 索引全局突触块。边界 owner 机制让单播在离源最近的边界发起，共享区域内广播路径、减少源附近单播热点；每个边界节点神经元放置减 20% 并在区域内均匀重分布以容纳额外 axon-out 元数据。路由成本建模：广播 L_b=|V_b|-1（NAHP 的 V_b=区域节点，neuron-centric=全系统节点）、单播 L_u=源到目标跳距和（NAHP 由边界节点注入），平均每节点路由负载 R=R_Local+R_Global（λ=firing rate），流量 T=R_Local×W_L+R_Global×W_G。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：硬件层 = NoC 路由器模式位分类 + 表驱动广播/坐标 mesh-XY 单播 + 边界触发逻辑（Algorithm 1 的确定性 axon-to-boundary 分配）+ SRAM/3D-DRAM 双布局存储（紧凑 pointer/length 头 + 连续邻接表、coalesced DMA）；软件/配置层 = 区域分配时指定 LNode、离线规划广播路径表与确定性单播路由表、加载 router-local SRAM。使用场景：面向大脑结构连通性（区域模块化、稠密本地 + 稀疏长程）的大规模神经形态仿真——比 neuron-centric（全局广播冗余流量 + 全局目录索引墙）与 axon-centric（无路径复用的独立单播拥塞）更适合 1B/16B/100B 规模的实时（1ms step）spike 通信。评估工具：自研 topology-aware analytical simulator（未开源）。

涉及论文标题：
- WaferBRAIN: Whole-Brain Scale Neuromorphic Architecture Based on Wafer-Scale Integration
