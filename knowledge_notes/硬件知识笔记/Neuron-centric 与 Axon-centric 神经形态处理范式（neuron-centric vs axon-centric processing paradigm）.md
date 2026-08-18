## Neuron-centric 与 Axon-centric 神经形态处理范式（neuron-centric vs axon-centric processing paradigm）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
神经形态系统两大类分布式通信范式的分类学（WaferBRAIN 论文 II-B 系统归纳）：(1) **Neuron-centric（神经元中心）**——以神经元 spike 为路由实体，事件用完整神经元 ID（FNid，37bit）编码、用 FNid 直接寻址突触；代表系统 SpiNNaker、Darwin 从源节点组播/广播 FNid，目标节点按 FNid 索引本地突触。优点：广播/组播共享路径复用；缺点：全局广播洪泛冗余流量、组播需逐节点大路由表、FNid 寻址要求每节点复制全局神经元目录（100B 模型 ~2^37 条目 → TB 级索引）。(2) **Axon-centric（轴突中心）**——以轴突 spike 为路由实体，事件用完整轴突 ID（FAid）编码、按 FAid 寻址突触；代表系统 Loihi、Darwin3 由源节点存 fan-out FAid 并单播给目标，接收者按 FAid 索引突触。优点：FAid 寻址目录小（~2^27 条目）、无冗余投递；缺点：独立单播丧失路径复用、密集本地连通下拥塞。另有 TrueNorth/Tianjic/PAICORE 的 crossbar 突触阵列，fan-in 受硬件上限限制、不适应稀疏不规则连通。CPU/GPU 软件模拟器（DFMG 的 local-regular/global-flush 分组、Xin Du 的桥接 GPU 两级路由）也属 neuron-centric 语义，但以 MPI collectives 点对点消息实现、无 NoC 硬件路径复用。关键 trade-off：路径复用 vs 目录/流量开销，两种范式都无法同时匹配大脑"稠密本地 + 稀疏长程"的结构。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
两范式在硬件中的运转流程对比（同一神经元 n_i 放电 → 目标节点集合 V_d(n_i)）：neuron-centric——axon-out 把 spike 编码为 FNid（37bit）→ 路由器按广播/组播规则转发（广播：向全部节点扩散，每节点收冗余包；组播：逐芯片查大路由表）→ 目标节点按 FNid 索引本地突触邻接表；路由成本 L_b=|V|-1（广播域=全系统）。axon-centric——源节点存 fan-out FAid 列表 → 逐目标发独立单播（NodeID&FAid 46bit）→ 接收者按 FAid 索引突触；路由成本 L_u=Σ Dist(v(n_i), v_j)（每对目标独立点对点，无复用）。WaferBRAIN 论文的定量对比（Table III）：事件表示 FNid 37bit vs NodeID&FAid 46bit；寻址条目 2^37（全局目录）vs 2^27（fan-in 对应条目）；100B 规模 neuron-centric 索引开销随模型增长到 TB 级、axon-centric 更紧凑但在 ~1% firing rate 就破 1ms 实时界。这正是 NAHP（见专有条目）要调和的两个极端。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：neuron-centric 靠 NoC 广播/组播（SpiNNaker 的 multicast 路由、Darwin 的组播）与全局神经元目录；axon-centric 靠每源节点存 fan-out FAid 表 + 单播注入（Loihi 的树状单播、Darwin3 的 ISA）。使用场景：选择范式取决于连通性——稠密均匀连接宜广播（复用收益高）、稀疏不规则连接宜单播（免冗余）；大脑尺度（95% 本地 + 5% 长程）两者都不够，需 NAHP 混合。评估：WaferBRAIN 用自研分析模拟器按式 L_b/L_u、R、T 量化两者在 1B/16B/100B × mesh/dragonfly 下的流量与时延，作为 NAHP 的 baseline。

涉及论文标题：
- WaferBRAIN: Whole-Brain Scale Neuromorphic Architecture Based on Wafer-Scale Integration
