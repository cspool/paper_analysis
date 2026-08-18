## NPU 固件 JIT 编译（JIT Compilation in NPU Firmware，M100）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- M100 运行时软件栈的组成部分：AI inference runtime + NPU driver 运行在 SoC 的 ARM Cortex-A78 上；NPU firmware 运行在 NPU RISC-V 核上，采用 just-in-time（JIT）编译技术，基于 M100 编译器工具链生成的二进制动态生成优化的 TPB 指令，并实时计算 tensor shape 与存储地址，再把 TPB 指令下发给任务分配的 TPB 组。runtime 负责输入准备、加载模型、分配资源踢发任务、结果后处理与错误/异常监控（满足车规功能安全 FuSa）；driver 是应用软件的硬件抽象层。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 流程：runtime 加载模型并分配 NPU 资源 → NPU firmware JIT 把离线工具链二进制翻译为 TPB 指令流（动态解析 tensor shape/地址，生成含通信需求元数据的指令）→ 经 ICB 广播到任务分配的 TPB 组 → TPB 指令队列按数据就绪执行。JIT 使运行时能适应动态 shape 与动态内存分配，比纯静态 AOT 灵活（论文未明确说明 JIT 的具体 IR/优化内容）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：固件内 JIT（论文未明确说明细节）；运行时按 automotive FuSa 要求监控 NPU 错误。使用：垂直集成部署场景（AD 车规芯片），动态 shape 推理；与离线工具链（space-time scheduler/graph compiler/backend）构成两级编译（AOT + JIT）。未开源。

涉及论文标题：
- M100: An Orchestrated Dataflow Architecture Powering General AI Computing
