## NoC（Network-on-Chip，片上网络）与 SWIFT 优化路由器

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
NoC 是片上多核/多 bank 的互连：报文切成 flit（32–128 bit）在路由器间逐跳传输，经典路由器为 5 级流水（RC 路由计算→VA 虚通道分配→SA 交换分配→ST 交换遍历→LT 链路传输），每跳流水延迟是低延迟互连的主要开销。SWIFT（Kumar 等 ISCA'07 "Express Virtual Channels: Towards the Ideal Interconnection Fabric"）以 lookahead（提前 1 周期发路由信号）+ bypass（express flit 不经缓冲与仲裁直穿中间路由器交换机）+ EVC（1-hop..n-hop 分层虚通道）把每跳压缩到 1–2 周期：相对全互连理想网仅多 2 周期零负载延迟，报文延迟最高降 84%、吞吐 +23%、路由器能耗 -38%。CompAir-NoC 基于 SWIFT：4×16 2D-mesh、每 bank 4 router、flit 72b。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
CompAir 在 SWIFT 路由器上增加"flit compute"级：与 switch traversal 并行，Curry ALU 就地替换 flit 数据，不增加任何流水级。这是可行性的关键——SWIFT 每跳只有 1–2 周期预算，任何串入关键路径的计算都会破坏低延迟。执行例子（RoPE 重排）：4 个 router 五阶段交换完成相邻元素对调与奇数位取反（NoC_Exchange(R-, SrcRow, DstRow, 1, 2)）；NoC 天然把向量序列化为 flit，使标量级操作（对调、累加、迭代）可以在传输途中进行——这是 CompAir-NoC 的设计出发点。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：VC 划分、lookahead 路由、bypass 通路、credit/token 流控；SWIFT 的 EVC 让 flit 跳过中间路由器并在下一路由器才重新入队。仿真用 Booksim（cycle-accurate，https://github.com/booksim/booksim2）。使用方式：多 bank/多核片内互连；在途计算（归约树、广播树、非线性迭代）挂接在 ST 并行级上。注意：路由计算与仲裁仍占关键路径，计算逻辑必须旁路化而非串入流水。

GenZA 补充视角（ISCA'26，ZKP 加速器的 2D-mesh NoC）：GenZA 用 16×8 PE 阵列的 2D mesh NoC（32×64-bit links @ 2 GHz），PE 与 DRAM 间以轻量 packet 按静态确定的映射路由（类比 TANGRAM 类空间加速器）。与 CompAir 的"在途计算"不同，GenZA NoC 只做数据搬运：NTT 折叠流水线的 scratchpad 借贷——FIFO 访问模式使同时至多一对借/贷 PE 活跃、NoC 流量 ≤2× 前向数据，worst-case 256-bit 152 GB/s = per-hop 容量 30%（384-bit 22%、768-bit 15%），确定性流量下几乎无拥塞；MSM 分发——decoder&dispatcher 在进入 NoC 前按窗口/桶复制点并注入对应 PE 行（BN128 c=16 每点平均到 ~4 PEs，标量均匀分布故注入天然均衡），packet-level NoC 模拟显示 dispatch stall 仅 3.97%、平均 link 利用 5.9%、最热 link 峰值 44.51%；64-bit NTT 的 MDC 管线合并进单 PE 后完全不产生 inter-PE 流量。

MTIA 300 补充视角（ISCA'26，训练芯片主网格 NoC）：MTIA 300 的 NoC 是 2D mesh 路由器网络，连接 compute chiplet 主网格内的 72 PE 与 16 ME，并把 compute chiplet 连到北/南边的控制/主机接口与 chiplet interface IP（挂网络 chiplet）。设计要点：(1) **cluster router**——本地连 6 个 PE，降低总跳延迟（MTIA-2i 有 memory crossbar，MTIA 300 改为 NoC 处理 bank 选择路由）；(2) **L-routing**——先沿一维（如 X）再沿另一维（Y）路由，均匀分布流量到网格；(3) **虚拟通道（virtual lanes）**防死锁。NoC 提供 data/control/utility（寄存器访问、debug）/synchronization/reduction 通道；reduction 通道与 PE 内 RE 的专用归约网络配合，NMC 归约在网格边缘的 HBM 旁进行以减跨网格拥塞。与 CompAir/SWIFT 的"每跳低延迟优化/在途计算"不同，MTIA 300 NoC 的重点是"compute 与 collective 流量隔离"（ME 放边缘、L-routing 均匀化）而非逐跳计算。

Omelet 补充视角（ISCA'26，2.5D/3D chiplet 分层互连模拟器）：NoC 是 Omelet 分层互连层次中的最低层（intra-chiplet on-die 网络），与 interposer 级 NoI、垂直层间 NoL 统一进 Network-on-X（NoX）单一 flit-level cycle-level 框架。NoC 配置：2D Mesh、router 4 级流水、4 VC、4 flits/VC 缓冲、link 1 cycle/16B、NoC unit latency 10/50/100 ps/mm。设计要点：NoC 用密集金属走线、短低延迟链路；跨 chiplet 流量在 chiplet 边界经 adapter（PHY transition + SerDes + CDC）进入 NoI，NoC 注入带宽（16B/cycle 量级）与 NoI 技术感知带宽 W=floor(λ·R_lane(ℓ)/(f_clk·F_bits)) 之间的失配在边界形成打包/去打包与队列累积——这正是 Omelet 揭示的"cross-layer backpressure 在 NoI 饱和时回压 NoC 注入端、推迟饱和"与"isolated 模型（NoC/NoI 独立仿真后相加）更早饱和"现象的载体。Omelet 在 gem5 Garnet（v21.1）基础上扩展实现。

  - SHyLA 补充（面向 LLM 推理的专用连接 NoC）：片上 tile 按内存关联分 DTile/NTile 并配对成 tile group，组内用专用高速链路（NTile 供 NVM 的 Weight/KVCache 给 DTile、DTile 供 DRAM 数据给 NTile），跨组用 AXI fabric 提供全片连接——层次化连接匹配混合内存诱导的片上 LLM 流量，是带宽利用中心数据流的基础。chiplet 间经 die-to-die 接口 + ICNT_BW 429GB/s（A100 缩放 NVLink 模型）通信，通用处理器管理 chiplet 间链路、类 Bluefield 网络处理器处理跨板通信与归约。

WaferBRAIN 补充视角（ISCA'26，神经形态 NoC 路由器与混合广播/单播数据通路）：WaferBRAIN 的 BPU NoC 路由器支持 spike 事件的 unicast + broadcast 两种投递，按包 0/1 模式位分类（Fig.7）：(1) 广播（mode=0，本地区域 neuron-driven）——路由器用 router-local SRAM 中按 LNode 索引的 5-bit {N,E,S,W,L} 端口掩码路由表，单周期查表转发且无死锁；广播路径按区域形状离线规划（even/odd 源分别列扩展/行扩展以平衡两维链路、交替扩展避免热点），改变区域大小/平铺只需重新生成表并加载，无包格式改动与运行时计算。(2) 单播（mode=1，跨区域 axon-driven）——同 wafer 用无表 mesh-XY（坐标比较器逐跳比较目标 <POD,Wafer,Node> 坐标）；跨 wafer/POD 先经 egress-WC 选择逻辑（跨 POD 由目标 POD 驱动、同 POD 由目标 wafer 驱动）选 WC，再索引 WC 确定性路由表（WC-to-NC sharding 预规划，见芯片设计库 Switchless Dragonfly 条目）沿确定性路径到 WC egress，过 dragonfly 后目标 wafer 内 mesh-XY 递送。对比经典 5 级流水（RC/VA/SA/ST/LT）与 SWIFT 的逐跳低延迟优化，WaferBRAIN 路由器是"表驱动广播 + 坐标比较 mesh-XY + WC 索引确定性单播"的事件驱动轻量路由器（router forward latency 5ns、throughput 1Tb/s、实测 intra-die 1ns/inter-die 8ns/inter-wafer 493ns 跳时延）。
涉及论文标题：
- Bridging Efficiency and Scalability in LLM System via 3D Hybrid PIM with 2D In-Transit Computation
- GenZA: A General and Efficient Accelerator for Diverse Zero-Knowledge Proof Protocols
- MTIA 300: Meta's First Training Chip Featuring Built-in NICs and Collective Offloading Engines
- Omelet: A Packaging-Aware Hierarchical Interconnect Simulator for 2.5D/3D Chiplet Architectures
- SHyLA 3D-Stacked NVM-DRAM Hybrid LLM-Inference Architecture Exploiting Data and Memory Heterogeneity
- WaferBRAIN: Whole-Brain Scale Neuromorphic Architecture Based on Wafer-Scale Integration
