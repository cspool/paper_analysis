## Wafer-scale NPU core（tensor core + vector unit + private buffer + crossbar + NoC router）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Wafer-scale NPU core 是 BusyBarn 目标 wafer-scale chiplet 系统中每个 die 内阵列的基本计算单元：一个通用神经网络处理单元（NPU），内部用高速、低延迟 crossbar 连接四类组件——(1) tensor core：加速矩阵乘与卷积（高算术强度算子）；(2) vector unit：加速算术强度较低的一般计算，如激活函数与非线性变换；(3) private buffer：暂存中间计算结果；(4) NoC router：把 core 连入更广的系统网络。配置参数（Table I）：每 core 1 tensor core + 1 vector unit、16 MB SRAM，计算逻辑与链路均 1 GHz，baseline peak compute 16 TFLOPs/core（BF16，敏感实验覆盖 8/16/32 TFLOPs/core）。在层次化网格中，core 位置决定算子间通信距离，是 intra-die 映射优化（BusyBarn 第二个 SA 迭代器）的对象。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
在硬件架构中的运转流程（一次 TP 注意力算子映射到 2×2 core 的 Fig.6 例子）：LR 记号把每层算子切成数据切片→算子被分配到具体 core 的 tensor core（矩阵乘/卷积）或 vector unit（Softmax/激活）→ private buffer 暂存中间结果→数据依赖触发通信事件，经 NoC router 把数据切片发给消费方 core（如 K 与 K^T 若放错 core 会产生长距离通信，Fig.6d 放对位置则消除长路径）→ tensor core 与 vector unit 的负载、NoC 链路的负载共同决定执行时间。BusyBarn 的 intra-die 四元混合损失（总通信距离、最大链路负载、最大 tensor workload、最大 vector workload）正是为同时均衡这四类资源而设计——单看通信距离（Gemini 的做法）会留下 tensor/vector 计算热点。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现与使用：作为加速器 IP 在 wafer-scale 芯片上实现（tensor core + SIMD 风格 vector unit + 片上 SRAM buffer + 路由器），运行 LLM 推理的 GEMM/激活/归一化等算子；评估经事件驱动模拟器建模（计算事件按输入数据 shape 计时）。使用场景：LLM 混合并行推理（die 组间 PP、die 组内 SP/CP/TP）、故障容错（某 core 故障后重映射）。与商用对照：Cerebras WSE-2/3 的 PE（waferscale engine 的简化 AI core，局部存储 + 低精度脉动式计算）、Tesla Dojo 的训练 tile core；本论文 core 的通用 NPU 形态（tensor core + vector unit + 私有 buffer）介于 GPU SM 与 WSE PE 之间。

涉及论文标题：
- Mapping and Communication Optimizations with Fault Tolerance for Wafer-Scale LLM Inference
