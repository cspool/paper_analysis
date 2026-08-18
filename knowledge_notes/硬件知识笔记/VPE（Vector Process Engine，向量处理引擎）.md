## VPE（Vector Process Engine，向量处理引擎）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
VPE 是 NASZIP 在 DIMM 每个 sub-channel 内集成的近存向量计算引擎，把 FEE-sPCA 与 Dfloat 两种软件优化落成硬件模块：内含 4 条并行处理路径（每条对应一个 DRAM device），路径上依次为 Dfloat 解码模块（解码压缩的向量数据）、query buffer（预存查询向量元素）、距离计算模块（L2/IP 共享数据通路）；4 路部分距离经 accumulator 合并，结果动态喂给 FEE 模块触发早退。VPE 是 NDP 上"取数→解码→算距→早退判断"流水线的核心。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
VPE 运转流程（图 10c-e）：① Dfloat 处理模块：DRAM 突发读入后，每个 device 每 cycle 供 8 bit、16 cycle 经 16-to-1 MUX 填满 128-bit 寄存器；barrel shifter 按预设偏移寄存器逐段抽出 n-bit Dfloat 元素并零填充为 FP32；② query buffer：wrap-counter 驱动的 MUX 每 cycle 输出一个 query 元素；③ 距离计算模块：L2（Σ(x_i−q_i)²）或 IP（Σx_i·q_i），MUX 切换两种模式；④ accumulator 累加 4 路部分距离；⑤ FEE 模块：每次累加器更新即用 α_k/β_k 缩放估计 d_est 并与 threshold 比较，超阈值立即丢弃该向量。例子（SIFT 128 维，Dfloat 18/14/16 bit 三段）：一次向量访问需 16 个 burst，4 device 并行、每 burst 后 FEE 判断一次是否继续。面积分解（Fig.27）：Query Buffer 与 FEE 模块占面积大头（存储 query 与参数），Multiplier/Adder 占能耗大头。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：RTL 设计（论文开发 RTL 并 FPGA 验证功能），Synopsys Design Compiler 28nm 综合 + Cadence Innovus P&R 评估面积/功耗（VPE 面积 144.6K μm²，总 NASZIP 附加 709.1K μm²）；系统性能由 UniNDP 周期精确模拟器评估（每 rank 2 个 VPE，1.2 GHz）。使用：作为 NDP 的通用"向量距离引擎"，配合 DaM 保证数据在本 sub-channel、LNC 缓存邻居表，实现无跨通道的并行 BFS 距离计算。开源实现见 NasZip 仓库 simulate/ 与 RTL。

涉及论文标题：
- NasZip Software and Hardware Co-design to Accelerate Approximate Nearest Neighbor Search with DIMM-based Near-Data Processing
