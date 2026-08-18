## UCIe（Universal Chiplet Interconnect Express）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
UCIe 是 2022 年发布的开放 die-to-die 互联标准（Intel/AMD/ARM/TSMC/Samsung/Qualcomm 等组成的 UCIe Consortium 制定），定义 chiplet 之间通信的物理层（PHY）、D2D Adapter 层与协议层：物理层规定 bump 间距（标准包 25–55µm、先进包 <25µm）、单 lane 数据率（UCIe 2.0 最高 32 GT/s，NRZ 或 PAM4）、信号完整性预算（32 GT/s 下 crosstalk ≈ 20 dB 的 SI 指导）；D2D Adapter 负责 CRC 插入/校验、flit 打包/重传与链路状态管理；协议层承载 PCIe/CXL/流式协议。目标是把异构小片按"乐高"方式组合，让多厂商 chiplet 互操作（Web 证据：HotChips 2023 UCIe tutorial、UCIe 2.0 文档）。

从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
DICE 与 UCIe 的接口关系：DICE 的 flit 结构与 UCIe 68B flit 格式（Format 2）兼容——68B = 64B 协议负载 + 2B flit header + 2B CRC，DICE 的 FEC 奇偶校验字节可注入 68B 格式的未使用字节位（如 header/CRC 之外的填充位），从而在遵守 UCIe 帧格式的前提下叠加 LDPC FEC（论文 Fig.5 与 Table III：2B parity/16B flit，兼容 68B 格式）。UCIe 的规格参数是 DICE 标定的来源：符号率上界 32 GT/s（UCIe 2.0）、crosstalk ≈20 dB（UCIe SI 指导）。芯片设计含义：UCIe 把 D2D 链路的电气规格与帧格式标准化后，chiplet DSE（符号率、FEC 强度、lane 数）可在标准约束内量化探索；DICE 论文指出 UCIe 生态的实测数据公开后，可进一步逐参数校准 PHY 模型（当前缺实际实现数据是校准的主要限制）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：物理层 PHY IP（Cadence/Qualitas/Alphawave 等已 tapeout 32 GT/s UCIe IP，含 lane repair、去加重、EOM 眼图监控）；D2D Adapter 处理 68B flit 的 barrel shift（68B 非 lane 数整数倍，连续 flit 间 4B 移位、PDS token 终止流）与 CRC/重传；封装载体为 CoWoS/EMIB/Foveros/InFO。使用方式：作为 D2D 接口标准选型与 PHY 参数来源；在 DICE 类模拟器中，UCIe 决定符号率、crosstalk 预算与 flit 帧格式三类约束，FEC 编码以 UCIe 兼容格式承载。

CASCADE 补充视角（ISCA'26，TFHE 流水线的 D2D 载体）：CASCADE 用 UCIe Advanced 规范建模 12 个 HMUX Chiplet（HC）之间的 D2D 互连——16 GT/s 传输率、64-bit 数据宽 → 1024 Gbps D2D 带宽，D2D PHY 面积/功耗按 [27]（UCIe Advanced，arXiv:2510.06513）估计。D2D 链路构成 HC 的环形拓扑（HC_{C-1} 回传 HC_0），支撑跨 HMUX 流水线（inter-HC PCG 流水：上游 HC 算完多项式系数即经 D2D 送下游，不等整个 RLWE）。关键设计权衡：D2D 时延 > HMUX 计算时间时 HC 欠利用（naive 每 HMUX 跨 chiplet 的映射使 D2D 成为瓶颈）→ Interleaved-Fusion 把连续 HMUX 融合在本地、仅组间经 D2D 传输（ACC 经输入/输出 double buffer 隐藏 D2D 时延）。评估：DeepCNN-50 参数集 I 下 D2D 带宽利用率 76.8%（OIFS）vs 7.7%（Segmented 映射）——D2D 带宽利用率是 CASCADE 评估的核心指标之一。对比 DICE 关注的 PHY 信号完整性（SNR/FEC），CASCADE 以固定 16 GT/s/1024 Gbps 参数建模 D2D 时延与带宽，不建模 PHY 级动态行为。

涉及论文标题：
- DICE: Detailed Inter-Chiplet End-to-End PHY Modeling for Accurate Chiplet Simulation
- Unlocking Pipeline Parallelism for Bootstrapping: A Pipelined Multi-Chiplet TFHE Accelerator
