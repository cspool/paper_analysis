## LPDDR5X（大容量低功耗内存，低功耗 DRAM 堆叠封装）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
LPDDR5X 是 JEDEC LPDDR5 的低功耗演进版，最初面向移动/边缘设备，因容量-带宽-功耗均衡而扩展到 AI 推理与数据中心场景。JEDEC 规格：单 pin 速率 8.5Gb/s 以上（商用 8533-9600 Mbps），x16/x32 数据宽度，1.05V/0.5V IO；HybridSpec 引用 [54] 的封装形态：每包 8 层堆叠、每层 4×16Gb die = 64GB，128 DQ 引脚 @8.5Gb/s/pin = 136GB/s 单包带宽；XPU 配 8 包共 512GB 容量、1.1TB/s 聚合带宽（trace routing 在 PCIe 卡外形内可行已获验证）。相对 HBM：带宽低一个数量级但容量/成本/功耗友好，且无需 TSV + interposer。

从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
HybridSpec 用它做 XPU 的目标模型主存：target 模型权重 + 全部 KV cache 常驻 512GB LPDDR5X，容量支撑高并发（KV cache 不再封顶 batch）；因 SD 把 target 的算术强度抬高（一次验证前向处理多个 draft token），1.1TB/s 的相对低带宽对 target 影响有限（Fig.19 实证：低请求率下 HBM 更快，但请求率升高后 LPDDR 的大容量反超）。数据流：target prefill/verification 每轮从 LPDDR5X 流式读权重、KV cache 按需分配/增长，经 wire-bonding 封装接 XPU 计算 die；XPU-HB 栈间只传 draft KV cache（几 MB）与 token 列表（几百 B）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：8 层 DRAM die 堆叠封装（每层多 die 并列），wire bonding 到 substrate/封装基板；无需 TSV/interposer，装配成本低。使用方式：作"大容量、中等带宽、低成本"的内存基板，匹配计算密集（高算术强度）或容量敏感负载（大模型权重 + 大 KV cache）；与 HB/HBM 等"高带宽-小容量"基板异构搭配，各自承接带宽型与容量型需求。局限：绝对带宽远低于 HBM，纯 decode 密集且容量不紧的负载会受带宽限制。

MERIDIAN 补充视角（ISCA'26，LPDDR5X-PIM 的 DRAM die 物理组织与容量）：MERIDIAN 把 LPDDR5X 作为 PIM 存储介质与计算基板——每设备 8 个 LPDDR-PIM package（64 GB/package、8.5 Gb/s/pin、128 channels、8-channel PIM 控制器 128-bit 总宽、每 channel 4 个 16-bit DRAM die）；每个 die 16 Gb、16 banks/4 bank groups（每 group 4 bank），每设备最高 512 GB、32 设备共 16 TB（支撑 TB 级文档 KV 库）。与 HybridSpec 的"纯存储"用法不同，MERIDIAN 在每 bank 旁放 PIM Unit（PU：16 FP16 比较/乘法/加法器 + 4KB buffer，1 GHz，16-lane）实现 All-Bank-Mode 存内计算，并在 die 内集成 PIM 逻辑——面积开销按 10nm-class（1z-nm）DRAM 工艺核算：每 PU 0.15mm²（算术 50.5%/buffer 34.9%/控制 14.6%），16 个 PU/die 共 2.41mm²，仅占 47.53mm² LPDDR5X die 的 5.07%；时序参数 t_RC=60/t_RAS=40/t_CL=23/t_RP=20/t_RCDRD=17/t_RCDWR=8，外部带宽 1.1 TB/s、内部带宽 16 TB/s（LPDDR5X 选型理由：带宽/容量/成本/功耗均衡，设计可泛化到 HBM/GDDR/DDR）。

涉及论文标题：
- HybridSpec: Exploiting Hybrid-Bonding Memory to Accelerate LLM Serving through Heterogeneous Architecture and Speculative Decoding
- MERIDIAN: In-Memory Acceleration for RAG with Document Attention Decomposition
