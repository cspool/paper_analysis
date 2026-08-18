## Die-to-die SerDes PHY 与 PAM4 调制

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Die-to-die（D2D）SerDes 是 chiplet 之间高速物理层链路的发送/接收电路：发送侧把并行数字位（如 128-bit flit）经并行-串行转换（P2S）串成高速波形，接收侧采样、均衡、解调后串行-并行转换（S2P）还原位流。与片上 NoC 的 CMOS 数字走线不同，D2D 链路是模拟信号链路，承载于 silicon interposer、EMIB/Foveros 或 InFO-oS 等先进封装。PAM4（4 电平脉冲幅度调制）是 D2D SerDes 的主流调制格式之一（与 NRZ 并列被 UCIe 采纳）：每符号携带 2 bit，Gray 映射 [00,01,11,10]→[-3d,-d,+d,+3d]，在相同波特率下把带宽翻倍，代价是眼图电压裕度缩小、对噪声更敏感。DICE 采用 interposer 级低摆幅 ±50/±150 mV（d=50 mV），与 UCIe 2.0 最高 32 GT/s/SerDes lane 的规格对齐；商业 UCIe PHY IP（Alphawave AresCORE、Cadence 32GT/s UCIe、Qualitas）与 BoW 是 D2D 链路的落地实现（Web 证据）。

从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
DICE 中的运转流程（Fig. 2/Fig. 4）：CCD/IOD 边界路由器把 FEC 编码后的 flit（128-bit 数据 + 16-bit 奇偶校验）送入发送缓冲与调制仲裁器 → PAM4 Gray 映射把每 2 bit 编为 {-150,-50,+50,+150} mV 四电平符号 → 串行波形经有损信道（注入 AWGN 噪声）→ 接收侧逐符号计算位 LLR（软判决解调）→ FEC 解码/纠错 → flit 重组后进入对端 NoC。P2S 串行化延迟与符号率（2–32 symbols/cycle）直接决定 D2D 链路的传输延迟-带宽折衷：符号率↑ 平均包延迟↓，但 >16 symbols/cycle 后收益递减（串行化与排队不再是主导）。芯片设计含义：D2D SerDes 的符号率、摆幅、均衡与 FEC 是芯片封装设计的核心旋钮——短距低摆幅降低 I/O 功耗，但眼图裕度变小、需要更强 FEC/重传兜底，形成"信号完整性×可靠性×带宽×功耗"的耦合设计空间。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：UCIe 标准包（standard package，55µm bump，4/8/16/32 GT/s，NRZ 或 PAM4）与先进包（<25µm bump，翻倍数据率）定义 D2D PHY 规格；厂商 IP（Cadence/Qualitas/Alphawave/StarIC）提供 CTLE/DFE 均衡、每 lane 校准、eye monitor；BoW（OCP）用单端信号更低功耗但数据率上限更低。使用方式（DICE）：在仿真器里以 1-cycle 编码 + 1-cycle 调制 + 符号率（默认 32 GT/s）+ PAM4 电平来标定 SerDes 阶段延迟；参数对齐 IEEE HIR 2024 与公开 datasheet；并暴露 jitter/crosstalk/SNR/符号率旋钮供 DSE。注意：DICE 把 SerDes 阶段建模为逐 flit 的延迟组件而非带宽限流（HeteroGarnet 只做限流）。

MTIA 300 补充视角（ISCA'26）：MTIA 300 的 compute chiplet 与 2 个网络 chiplet 之间以 die-to-die 接口 + 112G SerDes 实现高带宽密度互联（每网络 chiplet 600 GB/s、两 chiplet 共 1.2 TB/s I/O），支撑 12 个 800 Gbps RoCE RDMA NIC 的 scale-up/scale-out 流量；相比 PCIe（MTIA 300 仅保留 16× Gen5 64 GB/s 作 host 接口），die-to-die 带宽高一个数量级且无协议转换/主机参与开销。SerDes 速度与每 lane 带宽密度是网络 chiplet 数据路径的物理基础（配合 RDMA NIC 的 express doorbell 省 800 ns HBM ring 读）。

涉及论文标题：
- DICE: Detailed Inter-Chiplet End-to-End PHY Modeling for Accurate Chiplet Simulation
- MTIA 300: Meta's First Training Chip Featuring Built-in NICs and Collective Offloading Engines
