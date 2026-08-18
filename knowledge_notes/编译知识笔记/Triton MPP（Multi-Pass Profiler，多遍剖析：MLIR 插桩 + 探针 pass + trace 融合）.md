## Triton MPP（Multi-Pass Profiler，多遍剖析：MLIR 插桩 + 探针 pass + trace 融合）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Triton MPP（Multi-Pass Profiler）是 Meta 内部的编译器中心联邦 profiling 框架，解决现代 GPU profiling 的碎片化问题（IR 级 tracer、汇编级 profiler NCU、二进制插桩 NVBit 接口不兼容、输出面向人读的文本/仪表盘难以自动化）。MPP 以 job graph 组合分析：compiler transforms 插入 MLIR 级插桩、profiling passes 采集指标、trace synthesis 产出结构化输出。对现代 Triton kernel 的关键价值：传统 profiler 粗略暴露异步行为（TMA、warp-specialized 流水、overlapped data movement），而直接插入 wait 会扰动执行（初始同步破坏 overlap，级联改变后续时序）；MPP 以最小侵入方式——捕获未修改的 base trace、用定向 probe pass 隔离特定迭代中的单条指令、加 guard 防干扰、融合多次结果归因——在 TTGIR 级 profile warp-group 操作、async copy、TMA 传输。注：MPP 为 Meta 内部工具，公开资料未见；最接近的开源等价物是 Triton 的 Proton profiler（TTGIR 级 instrumentation mode）与 KPerfIR（arXiv:2505.21661，TTIR/TTGIR profiling dialect）。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 在编译框架中作为"profile 编译 pass 流水线"运转：①原始 Triton kernel 编译 → TTGIR；②MPP 第一遍不插桩，捕获 unmodified base trace（异步执行的真实行为基线）；③第二遍插入定向 probe pass，只隔离目标迭代中的单条指令（如某次 WGMMA/TMA）加计时；④instrumentation guard 确保 probe 不干扰其他指令的 overlap；⑤多遍结果按指令/迭代融合归因，产出结构化（机器可读）的指令级延迟与 overlap 数据 → 供 context memory sub-agent 分析瓶颈。相比单遍插桩，多遍方案避免"同步破坏 overlap → 级联时序漂移"的系统性误差。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：作为 KernelEvolve 评估框架的一部分集成（论文 3.4.4），与 NCU、Proton、NVBit、MTIA Insight 组合成多粒度 profiling（系统级 Torch Profiler / kernel 级 NCU / intra-kernel MPP+Proton）；由 evaluation code generator 确定性生成调用 MPP API 的脚本，在解释器环境执行，输出结构化数据直接喂给搜索。因为无公开版本，复现只能依赖论文描述或用 Proton（github.com/triton-lang/triton PR #6505 的 instrumentation mode）作为功能等价物。

涉及论文标题：
- KernelEvolve: Scaling Agentic Kernel Coding for Heterogeneous AI Accelerators at Meta
