## Agentic Kernel Coding (LLM-based Automated Kernel Generation)

术语是什么？
Agentic Kernel Coding是指利用LLM agent自动生成和优化高性能计算kernel的编程范式——将kernel优化建模为LLM驱动的iterative graph search，而非传统手动开发（每算子-平台组合2-8周专家工作量）或传统编译器优化（固定heuristics/pass pipeline）。Agent通过迭代：生成kernel候选 → 执行correctness验证和multi-level profiling → 根据执行反馈优化prompt → 生成更优kernel，最终收敛到expert-level实现。

从编译框架角度拆解术语：
KernelEvolve的kernel优化formalization为graph search四元组(F, π_sel, O, τ)：
```
F: Fitness Function  F(v) = t_pytorch / t_triton
   正确性失败或编译/运行时错误 → F(v) = 0
π_sel: Selection Policy (greedy/MCTS/evolutionary)
O: Universal Operator  单一context-aware transformation
τ: Termination Rule  (time/max artifacts/stall/fitness threshold)
```

整个agent loop:
```
G = ({v0}, ∅)  # v0为baseline specification
for t in 0..T:
    U_t = π_sel(V_t)  # select nodes to expand
    for v in U_t:
        context = ContextAnalysis(v) + KnowledgeRetrieval(v)
        prompt = DynamicSynthesis(v.source + context + hw_constraints)
        v_new = LLM(prompt)  # Universal Operator
        fitness = RemoteEvaluate(v_new)  # FaaS dispatch
        V_t+1.add(v_new), E_t+1.add(v → v_new)
        Persist(v_new.id, v.id, fitness, is_buggy)
    if τ satisfied: break
```

关键设计是使用single universal operator替代传统多算子（Draft/Debug/Improve各含static prompt template）——消除predefined operator categories强加的cognitive constraints，让LLM基于实际runtime context自由推理优化策略。

术语一般如何实现？如何使用？
KernelEvolve是agentic kernel coding的首个生产部署系统，使用Claude 4.5/GPT-5/CWM/Llama作为LLM backends。通过FaaS platform解耦generation (CPU-bound)和evaluation (accelerator-bound)；通过persistent metadata store + object store支持distributed concurrent exploration、cross-session knowledge reuse和fault-tolerant checkpointing。生产环境中自动生成优化Triton kernel服务数百个模型，将开发时间从数周降至数小时，在异构硬件上实现1.25-17× speedup。

涉及论文标题：
- KernelEvolve: Scaling Agentic Kernel Coding for Heterogeneous AI Accelerators at Meta

---
