## 操作数上转 / 空间复制 / 时间共享（Operand Upcasting / Spatial Replication / Temporal Sharing）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
FPGA 上支持混合精度与运行时数据类型切换的三类既有 MAC 微架构策略（XtraMAC 的 baseline，Fig.2/Table II）：
① 操作数上转（Operand Upcasting）：把低精度操作数 padding/提升到固定高精度格式，在固定高精度 datapath 上执行——典型是 AMD Xilinx Floating-Point Operator IP [1]（PG060）：低精度 workload 下平均 DSP 位利用仅 32.4%（Fig.3 量化 INT×BF16/FP×FP 组合）；② 空间复制（Spatial Replication）：为每个数据类型实例化独立 MAC datapath + MUX 运行时选择（Tensor Slices [5] 类设计），如 INT8/BF16 配置同时放两套 MAC——零时延切换、无 pipeline bubble，但一套活跃其余闲置，平均 DSP 利用 26.7%，资源随格式数线性翻倍；③ 时间共享（Temporal Sharing）：复用一个整数 MAC 单元，把高精度操作分解为多个低精度微操作跨 cycle 执行——典型 TATAA [38] 把 BF16 MAC 分解为 4 个顺序 INT8 微操作，避免重复 BF16 逻辑但每个 BF16 操作独占 4 个 PE/流水级，BF16 峰值吞吐只有 INT8 的 1/4，BF16 MAC 有效 DSP 利用仅 8.9%。根因：DSP 被当作不透明单 lane 原语，数据类型切换被当作整个 datapath 的粗粒度控制。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
三策略在混合精度 GEMV 的运转对比（Table V，每操作资源，II=1、4 cycle）：
- Vendor IP 空间复制：同时实例化 INT8 与 BF16 两套 FP Operator pipeline，MUX 按控制信号选活跃 datapath——每 BF16 操作 220.0 LUT/310.5 FF/DSP 1，每 INT8 操作 110.0/155.3/0.5；混合精度需配开源 int-to-float 转换模块（dawsonjon fpu [10]/M-Machine [17]）。
- TATAA 时间共享：把 BF16 乘分解为 4 个 INT8 子乘（位分解），INT8 datapath 复用于 BF16——每 BF16 操作 352.0 LUT/467.0 FF/DSP 4（含额外控制与操作数调整硬件），每 INT8 操作 22.0/29.2/0.25。
- XtraMAC（对照）：共享位映射前端 + 单一乘法器-加法器流水线，每 BF16 与每 INT8 操作都 142.0 LUT/128.3 FF/DSP 0.25——相对 TATAA 降 LUT 59.7%/FF 72.5%/DSP 93.8%，相对 vendor IP 降 LUT 35.5%/FF 58.7%/DSP 75.0%。
要点：上转浪费 DSP 位空间（低精度操作数只占低位）、空间复制按格式数线性放大资源、时间共享以吞吐换面积——三者都无法在同一 DSP 位空间内跨数据类型共享乘法器。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：upcasting = 操作数位扩展 + 固定高精度 FP operator（PG060 参数化、综合期选格式）；spatial replication = 多 datapath 例化 + 运行时 MUX；temporal sharing = 位分解微操作序列 + 复用窄 datapath（TATAA 为 INT8 优化、含近似 sqrt/div fapp 单元）。使用：作为混合精度 MAC 的对照 baseline——XtraMAC 论文用 vendor IP（配 int-to-float 转换）做混合精度对比、用 vendor IP 空间复制与 TATAA 做运行时切换对比，均按 lane 数归一化资源、相同接口配置、II=1、4-cycle 时延公平比较。局限（论文指出）：upcasting 无法处理异构格式的高效执行、空间复制扩展性差、时间共享牺牲空间并行与吞吐。

涉及论文标题：
- XtraMAC An Efficient MAC Architecture for Mixed-Precision LLM Inference on FPGA
