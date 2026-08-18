## 检索增强动态 Prompt 合成与持久知识库（RAG + Knowledge Injection，context memory / deep search sub-agent）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 检索增强生成（RAG）指不把全部历史知识放 LLM 工作记忆，而在运行时按需检索相关文档注入上下文。KernelEvolve 将其用于 kernel 生成：两级 sub-agent 流水线——context memory sub-agent 分析运行时产物（kernel 实现、profile 测量、错误诊断、正确性结果）诊断瓶颈并合成优化指令；deep search sub-agent 根据这些指令从持久知识库检索对应硬件约束与优化模式，二者把检索到的内容参数化进 LLM 的 prompt，实现"运行时上下文决定检索目标"的自管理上下文窗口（token 预算 64K-1M 按 LLM 后端）。知识库按分层文件系统组织：constraints/（anti-cheating 规则：禁止跨平台抽象、外部库依赖、直接 CUDA API、测试覆盖不足；确保生成的是真实 Triton 实现）、guidance/（平台无关调试方法论、性能调优、Triton 语言惯用法）、hardware/（NVIDIA/AMD/MTIA 各自 15-40 篇文档，共 ≥100 篇，含架构、调试、优化技巧），配 index.md 导航与 BigGrep/Glean 代码搜索（STRMATCH/REGEX/FILENAME 三种模式，fbsource monorepo）。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 在编译框架中作为"prompt 编译器"运转：输入 = 运行时反馈（profile 指标、错误诊断）→ ①deep search 两级检索：先按平台/瓶颈类型/优化阶段查 index.md 定位模块，再取目标内容（如 H100 内存带宽瓶颈 → 返回 hardware/nvidia/optimization/{tma, shared_memory, on_device_tma}.md）→ ②知识注入教育 LLM 私有架构知识（MTIA 不在 LLM 训练语料：检索 libdevice API 映射、cross-PE 通信、cb_multiplier 等）→ ③组合成动态 prompt：当前 kernel 实现+执行历史 + LLM 分析报告 + 检索内容 + 硬件约束 → ④LLM 生成 kernel 候选。渐进式专业化：GEMM on H100 先从 hardware/nvidia/arch/tensor_cores.md（能力）→ hardware/nvidia/tlx/{overview,warp_specialization,async_tensor_core_operations}.md（细粒度控制）→ code_samples/{hopper-gemm-pipelined,hopper-gemm-ws}.py（完整参考实现）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：知识库是普通分层文件系统（文件夹/命名/文件关系作为检索信号，无需语义标注），index.md 双重用途（人读参考 + 机器导航）；检索经 Model Context Protocol（MCP）工具暴露统一代码搜索接口（STRMATCH/REGEX/FILENAME），自动 dereference 知识库内 fbcode 路径触发二次搜索拉取生产实现代码。效果：无 MTIA 知识时 LLM 生成标准 GPU 语义 Triton 在 MTIA 编译失败或功能错误；注入后能生成利用 SFU、inter-PE 通信、dual-core 的 MTIA kernel。局限：知识库需人工维护、检索质量依赖 index 结构。

涉及论文标题：
- KernelEvolve: Scaling Agentic Kernel Coding for Heterogeneous AI Accelerators at Meta
