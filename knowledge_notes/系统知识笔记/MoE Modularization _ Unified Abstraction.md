## MoE Modularization / Unified Abstraction

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

MoE Modularization 是 FSMoE 的系统架构核心设计——将 MoE 层分解为 Gate/Order/I-Order/Dispatch/Combine/Expert 六个独立可替换子模块，通过抽象基类（GateBase, OrderBase, ExpertBase 等）定义接口，Hook 机制（BeforeMoeStartHook 等 6 种 hook）支持非侵入式扩展。前端 API 定义与后端任务调度完全隔离——调度器通过 online profiler 自动适配任意子模块组合。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

```
┌─────────────────────────────────────────┐
│        FSMoE Scheduler (Back-end)        │
│  Profiler(α-β) → SLSQP/DE Optimizer     │
│              → Task Orchestrator         │
└──────────────────┬──────────────────────┘
                   │ schedule
┌──────────────────▼──────────────────────┐
│      MoE Layer (Front-end)               │
│  Gate → Order → Dispatch → Expert        │
│    → I-Order → Combine                   │
│  All via abstract base classes           │
│  + 6 hooks for non-invasive extension    │
└──────────────────────────────────────────┘
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

FSMoE 基于 PyTorch + C/C++/CUDA 扩展（https://github.com/xpan413/FSMoE）。预实现 4 种 Gate（GShard/Sigmoid/X-MoE/EC）、2 种 Order、4 种 AlltoAll 算法。新增路由仅需继承 GateBase 并实现 forward()，调度器无需修改。

涉及论文标题：
- FSMoE: A Flexible and Scalable Training System for Sparse Mixture-of-Experts Models
