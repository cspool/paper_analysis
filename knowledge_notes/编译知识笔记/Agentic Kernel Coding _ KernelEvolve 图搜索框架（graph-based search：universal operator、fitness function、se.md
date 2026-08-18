## Agentic Kernel Coding / KernelEvolve 图搜索框架（graph-based search：universal operator、fitness function、selection policy、termination rule）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Agentic Kernel Coding 指用 LLM agent 自动完成 kernel 从合成、编译、profile、正确性验证到性能基准的全生命周期，替代手工 kernel 开发。KernelEvolve 把 kernel 优化形式化为图搜索算法，元组 (F, π_sel, O, τ)：fitness 函数 F(v)=t_pytorch/t_triton（正确性失败或编译/运行错误置 0，F(v)=0）；selection policy π_sel 用启发式 h(v) 选择扩展节点（greedy 选最高分、MCTS 用 UCT 平衡探索-利用、进化算法维护种群做 crossover/mutation）；universal operator O: S×C→S 是"单一变换算子"，从现有实现生成新候选（C 为运行时上下文：profile 结果、错误信息、硬件约束、历史优化）；termination rule τ 在预算耗尽/进度停滞/达标时停止。与传统的多 operator 框架（Draft/Debug/Improve 各自固定 prompt）不同，KernelEvolve 的 universal operator 通过检索增强动态 prompt 合成（RAG）让 LLM 同时推理正确性、性能与架构权衡。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 作为"编译栈上层的生成服务"运转：每次迭代 = ①选择节点（如 greedy 选 fitness 最高）→ ②context memory sub-agent 分析该节点运行时产物（profile/错误/正确性）诊断瓶颈、生成优化指令 → ③deep search sub-agent 按瓶颈从持久知识库检索硬件约束/优化模式 → ④LLM synthesizer 合成动态 prompt → ⑤外部（Claude 4.5、GPT-5）或内部（Meta CWM、Llama on Twine）LLM 生成 Triton kernel 候选（新节点）→ ⑥评估：TritonBench 正确性+speedup、Torch Profiler 系统级、NCU/Proton/MTIA Insight kernel 级、Triton MPP intra-kernel → ⑦结果写 metadata store（id/pid/score/is_buggy/path_ref），生成 overview.md → ⑧下一轮按 fitness 选择。例子：conv1d 300 步搜索，draft 阶段（step 0-10 独立采样）fitness ~2000，树扩展阶段（step 10-300 带执行反馈）逐步到 4000→5000→收敛 6889，最终产出融合 2-kernel 实现。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：KernelEvolve 本体 Meta 内部部署，2026 年 4 月以 KernelAgent 开源（https://github.com/meta-pytorch/KernelAgent，Apache 2.0，arXiv:2512.23236）；核心设计是硬件引导的迭代优化流水线：任务分解、确定性编排、带 early stopping 的并行搜索、结构化状态持久化。搜索策略可插拔（greedy/MCTS/进化算法）；节点持久化用 metadata store（关系数据库）+ object store（kernel 文件、overview.md）分离，支持分布式并发探索、SQL CTE 图查询、跨 session 复用（新 GEMM 变体可从历史 15 个 GEMM kernel 中选最高分实现起步）、崩溃 checkpoint 恢复。局限：LLM 生成成本（token 消耗）、单算子粒度（论文未来方向扩展到模型级跨层融合）。

涉及论文标题：
- KernelEvolve: Scaling Agentic Kernel Coding for Heterogeneous AI Accelerators at Meta
