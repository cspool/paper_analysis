## WaferBRAIN: Whole-Brain Scale Neuromorphic Architecture Based on Wafer-Scale Integration

- baseline方法是什么？
  - baseline 分两个维度：(1) 处理范式——neuron-centric（SpiNNaker [29]、Darwin [26]：FNid 37bit 广播/组播、FNid 寻址全神经元目录 ~2^37 条目，路径复用好但全局广播冗余流量 + 大路由表/全局目录，索引开销随模型规模增长到 100B 下每节点 TB 级）；axon-centric（Loihi [5]、Darwin3 [25]：NodeID&FAid 46bit 单播、FAid 寻址 ~2^27 条目更紧凑，但独立单播无路径复用、密集本地连通下拥塞；TrueNorth/Tianjic/PAICORE 的 crossbar 限制 fan-in）。(2) 互连/集成——PCB 多芯片集成（D2D ~100ns、router ~100Gbps 量级、长 PCB 走线）+ mesh/torus 拓扑（10×10 mesh 端到端 18 hops），带宽与 hop 数随规模急剧恶化；SRAM/crossbar 突触存储限制容量。超大算例（14,012 GPU 慢 65-118.8×、82,944 CPU 节点慢 578×）都远离生物实时。
  - baseline 全栈执行例子（一个源神经元 spike 传播到 100B 规模多个跨区域目标）：
    ```
    算法pipeline层：论文未明确说明（本文为神经形态硬件架构、无模型推理加速算法；工作负载为 SNN 事件驱动管线
               Axon.in→Synapse→Dendrite→Soma→Axon.out，全步 trillions 级事件操作）
    系统框架层：论文未明确说明（无开源 serving 框架；CPU/GPU 软件模拟器如 DFMG [21] / 桥接 GPU 分组 [7] 用
               MPI collectives 点对点消息，无 NoC 路径复用）
    编译框架层：论文未明确说明（无编译框架；神经形态工具链离线分区/映射 [16][41]）
    kernel调度层：neuron-centric：源节点把 spike 按 FNid 编码 → 每路由器按全局目录转发（广播：所有节点收冗余包；
               组播：逐芯片查大路由表）→ 每 hop 5ns 转发 + 链路时延 → 目标节点按 FNid 索引本地突触邻接表
    硬件架构层：PCB 多芯片：D2D ~100ns、router ~100Gbps → 流量与 hop 数随规模爆炸、1ms 步内无法完成
               （100B 下 0.1% firing rate 就超 1ms）；存储：neuron-centric 每节点复制全局目录（100B 下 TB 级）、
               axon-centric 每节点存 fan-in 条目（索引冗余）
    ```
  - 缺失层次说明：算法pipeline/系统框架/编译框架三层论文未明确说明（论文聚焦通信范式-路由-存储-拓扑的硬件协同）。
- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文方法 = WaferBRAIN：在 3D-WSI 基座上对通信范式、路由、存储、拓扑四者协同设计。(1) **NAHP 混合范式**：本地区域 neuron-driven 广播（LNid 27bit、区域内路径复用）+ 跨区域 axon-driven 单播（GAid 33bit、无冗余远场投递）+ 区域级边界触发（单播只在 Algorithm 1 选出的边界 owner 发起，源附近无热点、共享区域广播路径）——直接对应 neuron-centric 冗余广播流量与 axon-centric 无路径复用两个缺陷；(2) **wafer-native 存储层次**：热状态在 SRAM、突触/连接在 3D-DRAM（紧凑 ID + contiguity-aware 邻接块 + coalesced DMA）——对应 neuron-centric 全局目录与 crossbar 的索引/容量墙；(3) **switchless dragonfly scale-out**：NC/WC/PC 三级无交换机通道 + WC-to-NC sharding + 预规划确定性单播路径——对应 PCB+mesh 的 hop 数与热点问题；(4) **3D-WSI 集成**：D2D ~1ns/>1Tbps、每 wafer 1.92TB DRAM——对应 PCB 100ns 级长链路与 SRAM 容量墙。
  - 论文方法全栈执行例子（同一 spike 的 NAHP 传播）：
    ```
    算法pipeline层：论文未明确说明（SNN 事件驱动管线不变：dendrite 分段归约累加突触电流→soma 膜电位更新→
               阈值触发 spike；无推理算法改动）
    系统框架层：论文未明确说明（无 serving 框架；脑模型按区域分区映射到 8×8 节点网格，95% 本地+5% 长程连接）
    编译框架层：论文未明确说明（离线预计算：区域广播路径规划生成 router-local 路由表、even/odd 源分集、
               确定性单播路径与 WC-to-NC sharding 表，系统配置时加载，无运行时计算）
    kernel调度层：源 BPU 发 mode=0 区域广播（LNid）→ 各路由器按 LNode 索引 5-bit 端口掩码表单周期转发
               （even/odd 列/行扩展平衡链路）→ 边界 owner（Algorithm 1 最小曼哈顿距离）持有 axon-out 元数据
               → 发 mode=1 单播（<POD,Wafer,Node>+GAid）→ egress WC 选择 → WC 索引确定性路由表沿预规划路径
               → WC 出 wafer（8 NC sharding 均衡）→ dragonfly（同 POD 1 hop/跨 POD 3 hops，wafer 0-6/7-13
               平衡负载）→ 目标 wafer mesh-XY → 接收端按 GAid 索引 3D-DRAM 全局突触邻接块（coalesced DMA）
    硬件架构层：3D-WSI：6×8 die × 4×4 BPU、每 die 40GB 3D-DRAM（1B 神经元/256B 突触/wafer）、D2D ~1ns/>1Tbps、
               router 1Tb/s/5ns、L_n=1ns/L_d=8ns/L_w=493ns（Lyra X 实测标定）；结果：100B dragonfly 下 1ms 界内
               可持续 3.8% firing rate（比 neuron-centric 高 14×、比 axon-centric 高 4.7×）、流量降 2.6-300×、
               索引降 1.2-7,400×、dragonfly 比 mesh 峰值 inter-wafer 流量降 3.4-3.7×、WSI 比 PCB 可持续
               firing rate 高 13×
    ```
