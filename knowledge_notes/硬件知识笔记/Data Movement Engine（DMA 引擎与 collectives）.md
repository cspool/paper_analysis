## Data Movement Engine（DMA 引擎与 collectives）

术语解释
- Morphatron 底部的可编程数据搬运引擎：多个 DMA 单元（各经自己的 switch 接入全局互连）+ 两个 Global Buffer（片上存储与数据 staging）+ 两种 collective 通信模式 BROADCAST / COLUMN_REDUCE，负责全部 off-chip 数据进出与跨核分发归约。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 编程方式分两种：(1) vector/systolic morphas——DMA 由编译器静态编程（LD_CONFIG_BASE/STRIDE/ITER 描述连续块维度），实施 prefetch 与 double-buffering；(2) queue-centric 执行——DMA 用"小指令模板"动态编程，模板参数由各 Queue Manager 中保存的队列统计填充（数据量运行时才知道）。collectives：BROADCAST 向参与核分发输入、COLUMN_REDUCE 收集结果，通过连接 DMA 引擎与各 switch 的控制网络路由控制/同步信号，复用既有数据通路（不建独立网络）；编译器插入 barrier 指令保证顺序。Global Buffer 1/2：systolic morpha 中分别作输出/输入缓冲，其他场景作 staging 区。
- 从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 运转流程例子（systolic morpha 一次 GEMM）：编译器静态配置 DMA 迭代参数 → DMA 把权重块从 off-chip 预取进 Global Buffer 2（输入缓冲）→ BROADCAST 沿控制网络激活各 switch 的 collective 模式、把输入分发到各核 → 列方向核流水执行乘加（水平走流水寄存器、垂直走互连）→ 部分和经 COLUMN_REDUCE 归约收集到 Global Buffer 1 → DMA 写回 off-chip；整个过程 prefetch/double-buffering 与计算重叠。配置延迟：on-chip collective 配置 129 cycle。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 通用 DMA 引擎（旁路 CPU 的块拷贝、散列-聚集描述符）在这里被扩展为"多单元 + 双缓冲 + 可静态/动态编程 + collective 模式"的加速器数据平面。使用：所有 morphas 的 off-chip 流量必经此引擎；队列溢写（spill）也经由 DMA 路径执行。论文未明确说明该引擎是否开源。

涉及论文标题：
- Accelerator Polymorphism: Transcending Domain-Specific Architectures with Robotics
