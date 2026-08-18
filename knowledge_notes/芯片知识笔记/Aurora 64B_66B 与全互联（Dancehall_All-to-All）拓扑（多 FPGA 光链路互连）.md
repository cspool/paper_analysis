## Aurora 64B/66B 与全互联（Dancehall/All-to-All）拓扑（多 FPGA 光链路互连）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Aurora 64B/66B 是 AMD/Xilinx 的免费 LogiCORE IP 与轻量链路层协议（spec v1.3，64B/66B 编码），用于 FPGA 间高速串行点对点链路：支持 500 Mb/s 到 400+ Gb/s（最高 16 个绑定收发器 GTX/GTH/GTY/GTM）、约 3% 开销、AXI4-Stream 帧/流控接口、自动通道初始化与维护、32-bit CRC；典型应用是 chip-to-chip、板对板、背板与短距离光互连。Web 证据：AMD Aurora 64B/66B 产品页（https://www.amd.com/en/products/adaptive-socs-and-fpgas/intellectual-property/aurora64b66b.html ）、Alveo 上 Aurora 的 Vitis 教程（https://xilinx.github.io/Vitis-Tutorials/master/docs-jp/docs/Hardware_Acceleration/Design_Tutorials/08-alveo_aurora_kernel/README.html ）、开源实现 AuroraFlow（4 条 QSFP28 lane 100Gb/s，https://github.com/pc2/auroraflow ）。
- 在 Lotus（ISCA'26）中：8 块 AMD Alveo U55C FPGA 用 QSFP28 光口经 1:4 光分线（breakout）电缆按**全互联（dancehall/all-to-all）**拓扑互连——每块 FPGA 的两个 QSFP28 接口内部含 4 条独立 lane，7 条 lane 用 breakout 接到其余 7 块 FPGA（外加 patch panel 管理走线），无需网络交换机。网络对分带宽 350 GB/s，FPGA 间延迟 200ns（与现代 CPU 服务器 socket 间延迟相当）。跨 FPGA 通信经自研 Aurora shim：把 token 串行化到每条 64-bit lane、缓冲传输、错误检测（链路偶发 bitflip）+ 类似 TCP 的滑动窗口算法（既做错误重传又做流控）。

从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 在 Lotus 系统中的运转：①FPGA 内 token 交换机把要跨芯片的 token 路由到 8 个出口端口之一；②Aurora shim 把 token 打包成 64B/66B 帧、串行化到目标 FPGA 对应的 lane；③经光分线电缆（200ns）到达对端 FPGA 的 Aurora shim；④对端解帧、错误检测（CRC/滑动窗口序号），按需重传，再注入对端 token 交换机；⑤送达目标 tile 的任务单元。滑动窗口机制保证无丢包与背压流控——token 流量是变长的、突发性强的任务间通信。
- 设计取舍：选择全互联而非环形（AWS/其他 Alveo 系统常用的 2×QSFP28 环形连接）：全互联提供任意两 FPGA 直接通路（无多跳、无中继带宽争用），但每 FPGA 只有 7 个直接邻居端口（8 FPGA 系统刚好）；更大系统（如 128 FPGA）则需换用亚微秒延迟的商用以太网交换机（论文指出其可行性）。成本：互连仅 $1,200（总系统 $86K），是 emulator 无法企及的性价比。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：AMD Aurora 64B/66B IP（底层），Lotus 在其上实现自定义 shim（token 打包/解包、滑动窗口重传与流控、位翻转错误检测）；物理层用 Alveo U55C 的 QSFP28 光口 + 1:4 breakout 光缆 + patch panel。Web 证据：QSFP28 lane 到 GTY quad 的物理映射需逐卡核对（非顺序连接），Aurora 跨光学（AOC/SFP28/QSFP28）已验证可用（https://adaptivesupport.amd.com/s/question/0D54U00008QoC89SAF/ ）、AuroraFlow 证明 4×QSFP28 lane 100Gb/s 的可行封装（https://github.com/pc2/auroraflow ）。
- 使用：多 FPGA 仿真/原型平台的芯片间数据通路；在 Lotus 中是"时间映射"的关键使能——跨芯片 token 异步传递与计算重叠，200ns 延迟被任务缓冲吸收，使 8 FPGA 系统达到 emulator 级性能（而 emulator 的跨芯片锁步通信正是其瓶颈）。

涉及论文标题：
- Lotus A Multi-FPGA Task Dataflow Architecture to Accelerate Cycle-Level Simulation
