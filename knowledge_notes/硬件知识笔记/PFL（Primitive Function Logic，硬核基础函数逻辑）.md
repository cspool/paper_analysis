## PFL（Primitive Function Logic，硬核基础函数逻辑）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
PFL 是 CCM 硬件原型中固化基础功能的硬核逻辑块：把常用计算模式（MAC 乘累加、ACC 累加、CMP 比较等）做成专用硬件 IP 集成进 PNM 引擎，避免通用处理器的指令开销，为特定应用（如 KNN 向量距离计算）提供接近 ASIC 的效率。SK hynix 的 CCM 原型（Xilinx Versal VP1502 FPGA）的 PNM 引擎即由 PFL 硬件 IP + 一个 Cortex-A72 通用核组成。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
KNN 卸载例子：数据库向量驻留设备内存，查询向量广播到 PFL 数据路径 → MAC 单元做逐维乘累加得到部分距离（对应向量距离计算）、CMP 单元比较候选结果（对应 top-K 筛选）→ 每个输入向量只产生 4 字节 float 距离值回传主机。PFL 承担数据面计算，通用核承担控制/新增操作扩展。局限：PFL 面向单一应用定制，换负载要重造硬件——这是 M²NDP/AXLE 转向通用 PNM 的原因。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：FPGA 综合的硬核 IP（MAC/ACC/CMP），也可在 ASIC 中固化；与 CXL 内存控制器、DMA 引擎一起集成在 add-in card。使用方式：固定功能加速（数值/字符串过滤、向量距离、embedding 查找），一般配合设备固件调用。早期 CCM 原型（KNN 专用）依赖 PFL 获得高加速比，但通用性差。

涉及论文标题：
- AXLE: Coordinated Offloading with Asynchronous Back-Streaming in Computational Memory Systems
