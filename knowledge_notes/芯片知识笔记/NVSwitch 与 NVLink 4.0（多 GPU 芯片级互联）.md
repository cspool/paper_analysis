## NVSwitch 与 NVLink 4.0（多 GPU 芯片级互联）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
NVLink 是 NVIDIA 的 GPU 直连高速串行链路（点对点、含 PHY/数据链路层，支持地址翻译与远端访存语义），NVSwitch 是把多条 NVLink 端口汇聚成交换的独立芯片，二者共同构成多 GPU 节点内（及超级芯片间）的芯片级互联网络。NVLink 4.0（Gen4，本论文建模版本）：每链路双向 900 GB/s 聚合带宽（每 GPU 18 链路）、单跳延迟约 250ns（论文建模值，往返 1µs）、flit 16B。NVLink 3.0（Gen3，A100 世代）：每链路 300 GB/s 理论峰值，DGX A100（8×A100 SXM4）实测单个 NCCL Broadcast 约 262 GB/s、8 GPU AllReduce 聚合带宽最高 1878 GB/s；实测呈非线性延迟-大小缩放（4KB 与 32MB 传输延迟相当；8 GPU 下 4KB/64KB 仅 1.12/17.12 GB/s）与可忽略的链路内/间争用（7 接收者延迟仅 +13.27%），CDFD 据此采用 32MB 粗粒度页复制。NVSwitch Gen3：64 端口/块，NVL32 以 9 块（物理部署 9 tray×2 芯片）把 32 个 GH200 的 576 个 NVLink 端口连成全连接 fat-tree；Blackwell 世代升级为 NVLink Gen5（100 GB/s/链路）与 GB200 NVL72。

从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
芯片级互联决定跨 GPU 数据路径：GH200 superchip 把 H100 级 GPU + Grace CPU 以片间 NVLink-C2C 封装，外部经 18 个 NVLink 4.0 端口上连 NVSwitch；NVSwitch 芯片内含交叉开关 + 每输入端口 16 条 256 深 VC（8 请求 + 8 响应，本论文建模值）+ 端口 reduction buffer（64KB）与 NVLS 归约逻辑。DySHARP 的芯片级改动沿 NVLink 数据链路层与 NVSwitch 转发路径展开：flit0 重定义为 48-bit multimem 地址 + 1-bit stage + 15-bit target count，后接 target extension flits（每 flit 8 个 16-bit expert ID）；交换机 Route 按 OutPort^i = Target^i / 每 GPU 专家数逐端口复制裁剪转发，Reduction Logic 随部分响应递减计数、归零返回归约结果。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
商用系统：DGX H100/H200（NVLink4 + NVSwitch）、GH200 NVL32、GB200 NVL72（NVLink5 + 18 NVSwitch）；软件经 CUDA 统一地址空间点对点访问远端 GPU 内存，经 NVLS/NCCL 做集合通信。研究者用 Accel-Sim + BookSim2 建模：NVLink 参数（900 GB/s、250ns、16B flit）与 NVSwitch 参数（VC 数/深度、reduction buffer）注入 BookSim2 网络模型。本论文还模拟 64-GPU 扩展节点 = 18 NVSwitch（64 端口/块、每端口直连一 GPU，为 NVL32 的双倍互联）。CDFD 论文的用法：在 DGX A100（NVLink 3.0）/DGX H100（NVLink 4.0）真机上用 NCCL 与 cudaMemPrefetchAsync 实测延迟-大小曲线，注入 MGPUsim 4-GPU 模型作为 inter-GPU 网络参数，支撑 32MB 粗粒度页复制策略。

LIBRA 补充视角（ISCA'26，NVLink 作为多 GPU 页面迁移的权衡参数）：LIBRA 建模 NVLink 3.0 为 inter-GPU 300GB/s（CPU-GPU 为 PCIe-v4 32GB/s），其核心洞察"远程访问可能比迁移更优"正源于 NVLink 的低延迟高带宽——式(2) 用跨 GPU 平均远程访问延迟 lat_remote 与 page_migration_overhead 做权衡，模拟器标定一次页面迁移开销≈200 次远程 GPU 访问（与 NVIDIA UVM 访问计数迁移阈值 256 吻合）。敏感性实验：NVLink 200/300/400 GB/s 下 LIBRA 相对 TBNP-O 提升 64.7%/46%/36.8%——带宽越高迁移开销占比越小、预取收益递减。multi-rack 扩展：rack 间用 NDR 400Gb/s InfiniBand（NVIDIA Quantum-2 平台、ConnectX-7 400Gb/s NIC），成本模型需区分 intra-rack/inter-rack 远程访问延迟，并把跨 rack 页表更新、跨 rack TLB 失效、inter-rack 数据传输延迟计入迁移开销（2 rack×8 GPU 下相对 TBNP-EA/Forest/HOPP/GRIT 提升 74%/65%/60%/56%）。

MoE-Hub 补充视角（ISCA'26，NVLink/NVSwitch 作为 destination-agnostic 通信的传输层）：baseline 配置为 8 GPU 经 4 个 NVSwitch 的 DGX-H800 拓扑，BookSim2 修改版复刻 NVLink 设计（全双工链路、16B flit、单 flit 头、full-to-full 路由与 switch 级转发），每 GPU 400 GB/s、GPU↔switch 单向 250ns（往返约 1µs），带宽/延迟按真实硬件校准（0.5MB-256MB 下模拟 All-to-All 与物理系统误差 4.36%）。MoE-Hub 把 NVLink 视为"128B 缓存行粒度"的传输通道：st.rowsp 的 token 行按 ≤128B 拆包传输（RowOffset 字段指定行内偏移），RPM 在 hub egress 按目标 GPU 分区合并成 interconnect 友好的 128B 包、round-robin 平滑突发（防单消费者↔switch 链路拥塞反压）、consumer-aware 优先级最小 RowID 先发保证整行连续到达。芯片级设计启示：MoE-Hub 不修改 NVLink/NVSwitch 本身（互连物理层与交换机保持不变），只在 GPU 侧 hub（crossbar 与互连之间的位置）加入 AAU/RPM/DAM——即"通信控制平面下沉到 GPU 内、互连物理层原样复用"，这与 DySHARP（改 flit0 头与交换机 Route/Reduction 逻辑的 in-switch 计算）形成对比：前者在端侧 hub 编排、后者在交换机内计算。

涉及论文标题：
- Accelerating MoE with Dynamic In-Switch Computing on Multi-GPUs
- Coarse-Grained Duplication First, Fine-Grained Deduplication Later: Duplication-Centric Multi-GPU Memory Management
- LIBRA: A High-Accuracy, Cost-Aware, and Coordinated Multi-GPU Page Prefetcher
- MoE-Hub Taming Software Complexity for Seamless MoE Overlap with Hardware-Accelerated Communication on Multi-GPU Systems
