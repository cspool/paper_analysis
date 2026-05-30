## Co-Packaged Optics (CPO)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Co-Packaged Optics (CPO，共封装光学) 是一种将光收发器（optical transceiver）模块直接封装在与计算芯片（GPU/TPU/CPU/NPU）相同的 substrate 或 interposer 上的先进封装技术。与传统的 pluggable optical transceiver（通过面板插入 NIC 的 QSFP/OSFP 端口，中间经过 PCB trace + SerDes + retimer）不同，CPO 将光电器件（激光器、调制器、光电探测器、波导）与计算 die 通过硅 interposer、EMIB 或玻璃基板集成在同一 package 内。CPO 消除了芯片到面板光纤连接的多道电气-光学-电气转换，显著降低：(1) SerDes 功耗（每链路节省 3-5 pJ/bit）；(2) 信号衰减和 PCB trace 带宽限制；(3) 系统体积和光纤管理复杂度。CPO 被认为是下一代 >100 Tbps GPU I/O 带宽的关键使能技术。

从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
CPO 的芯片级实现方案：
- Ayar Labs TeraPHY：使用硅光子 microring resonator modulator + photodetector array，通过 EMIB（Embedded Multi-die Interconnect Bridge）与 FPGA/GPU die 相连。每个 TeraPHY chiplet 提供 2 Tbps 光 I/O 带宽。光信号经单模光纤输出（可长距离传输至架顶 OCS）。
- Lightmatter Passage：使用硅光子 Mach-Zehnder interferometer array 实现 chip-to-chip 光通信，每个 Passage chiplet 的 32×32 光 switch matrix 可在 chip 级别实现拓扑重配置。
- NVIDIA CPO 路线图：NVIDIA 已展示 CPO GPU 原型，将 optical engine（OE）与 GPU die 通过 NVLink-C2C（chip-to-chip interconnect）连接，每 GPU 目标 4-8 Tbps 光 I/O。

在 MixNet 前瞻分析（§8）中的 CPO 应用：
- 当前 MixNet：光信号通过 NIC + pluggable transceiver 进出 GPU server，NIC 通过 PCIe 连接到 GPU（多道电气转换，延迟和功耗增加）。
- CPO-based MixNet：光信号直接从 GPU die 的 CPO module 发出 → 通过光纤连接到 OCS → OCS 交换到目标 GPU die 的 CPO module。消除 NIC + PCIe + pluggable transceiver 的多道转换。
- 性能模拟：2048 GPU cluster（NVL72 系统），MixNet with optical I/O（CPO）训练 DeepSeek-V3（EP=128, PP=16, micro-batch=240），比 NVL72（7.2 Tbps NVLink + 800 Gbps Ethernet）降低 iteration time 1.3×。当 GPU I/O 带宽扩至 16 Tbps 时，MixNet 优势进一步扩大。
- 关键洞察：CPO 使 OCS 能直接连接到 GPU die 的 scale-up 域边界（而非 NIC 侧的 scale-out 域边界），将 OCS 的区域可重构能力延伸到 chip-level。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 封装方案：(a) 2.5D 集成（硅 interposer，如 Ayar Labs + Intel EMIB）；(b) 3D 堆叠（将 photonic die 直接堆叠在 logic die 上）；(c) 玻璃基板（glass interposer，Bunandar et al. 2024 US Patent）。
- 光 I/O 模块类型：(a) Micro-ring modulator（Ayar Labs, Lightmatter）——小尺寸、低功耗但温度敏感；(b) Mach-Zehnder interferometer（Lightmatter Passage）——宽带大、温度不敏感但尺寸大；(c) EAM (Electro-Absorption Modulator)——高速（>100 GHz）但 extinction ratio 有限。
- 部署准备度：CPO 仍在 pre-production 阶段。主要挑战：(a) 封装良率（光子 die + 电子 die 共封装的制造复杂性）；(b) 热管理（激光器发热和温度对波长的敏感性）；(c) 标准化（多厂商互操作的 CPO interface standard 仍在制定中）；(d) 测试和维护（光纤连接的面板可测试性）。
- MixNet 兼容性：MixNet 的区域可重构 OCS 设计与 CPO 的演进方向自然对齐——CPO 简化了 MixNet 的光纤连接（减少 NIC + pluggable 中间环节），使 MixNet 能更直接地利用 OCS 的高带宽和低延迟优势。

涉及论文标题：
- MixNet: A Runtime Reconfigurable Optical-Electrical Fabric for Distributed Mixture-of-Experts Training

---
