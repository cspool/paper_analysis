## Computational Introspection（计算型内省 / 片内数据聚合 On-Chip Data Aggregation）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Computational Introspection（计算型内省）是论文对 IPU 片内分析模式的统称：IPU 不止是"看"硬件信号，而是在片上本地执行分析程序（聚合、过滤、统计），只把高层洞察导出片外，而不是把原始数据流导出。背景逻辑链：trace 类方案（JTAG）失败的根本原因是数据洪流——原始微架构信号可达 TB/s 级，片上 buffer 放不下、I/O 带宽饱和、应用被拖慢；IPU 把"在数据产生处就近处理"（near-data processing 思想用于观测域）：introspection 程序在 HIT 旁边的 IPU 上就地计算（如 1024-bin 直方图、256-cycle 窗口活跃率、每 PC 失败模式计数），输出降到 KB/s 级（论文四大演示的输出规模：预取器 3B/程序结束、PICS 15KB/s、GPU 利用率 15.6MB/s（108 SM 全采样）/0.16GB/s（10 SM 采样）、逐 miss 诊断 18.75KB/s），把瓶颈从数据传输移到本地计算。这是 IPU 三大属性之一"On-Chip Data Processing"的实现机制，也是 Capability 3（Scalable On-Chip Data Aggregation，可扩展片内数据聚合）的核心：可编程时间窗聚合（如 256-cycle 窗口直方图），捕捉 bursty/ephemeral 硬件行为——聚合 PMU 只报整个 kernel 的平均，掩盖亚 kernel 级动态。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
计算型内省在 IPU 硬件中的运转流程（Capability 3 为例，AccelSim 验证）：HIT=GPU SM scoreboard，3 个 1-bit 信号（SIMT/TensorCore/L1-MSHRS）每 cycle 进 IORegs → IPU_lite 执行单条直方图指令（loop directive 优化），把 3 信号各当一个桶做 3-bucket 直方图，256-cycle 窗口内逐 cycle 累加活跃计数 → 窗口结束写 3 字节统计（可跨窗口批量）→ 经逻辑 FIFO 发主机内存 → 主机后处理：图 7(c) 上图为按时间排序的窗口活跃率曲线（可见 SIMT/TC/L1 互斥行为），下图为按利用率排序的运行平均；图 10 把窗口分成 4 类（≥2 信号高/1 高/全低，高=窗口内 >25% 活跃）并按 gemm shape 堆叠，揭示大段时间至少一个 SM 组件空闲。数据降维量化：3×108 SM /256 cycles = 1.7 GB/s @1.4GHz 全采样，10/108 采样 0.16 GB/s——远低于原始逐 cycle 信号率；对比聚合 PMU 只能给整个 kernel 一个平均活跃率。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：IPU_lite 的直方图加速器 + 循环指令 + 32KB scratchpad 构成聚合执行环境；API（IPU_CONFIG_*）配置触发区（可选把 retiring PC 接 ADDR、TS/TE 限定 kernel 内区域，或 untriggered 全 kernel）；输出经 IPU 访存指令透明路由到主机内存区域。使用方式：把任意微架构事件集合映射为窗口化直方图/平均/最大争用统计（论文列举"compute histogram/find average/report max contention"等任意聚合逻辑），跨 SM 规模化（每 SM 一个 IPU，覆盖率 <1% 面积开销）；对数据洪流类问题（如 TensorCore 输出值分布、GPU 利用率）是通用解法。局限：分析粒度受 IPU 处理速率与窗口长度约束，超快事件流需采样或接受丢数据近似（论文用随机化采样窗口长度打破与周期硬件行为的病理性对齐）。

涉及论文标题：
- Enabling Continuous, In-Field Introspection: The Programmable IPU Architecture
