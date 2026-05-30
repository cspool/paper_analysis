## LLM Agentic Workflow for Kernel Optimization (Planner-Executor-Summarizer)

术语是什么？
LLM Agentic Workflow for Kernel Optimization 是 AccelOpt 提出的三代理（Planner-Executor-Summarizer）协作系统，模仿人类专家优化 kernel 的工作流。Planner agent 分析 profiling 数据识别性能瓶颈并提出 1-step 优化计划；Executor agent 将优化计划转化为具体 NKI kernel 代码实现；Summarizer agent 从 slow-fast kernel pairs 中提炼通用优化策略和关键代码变更，存入 optimization memory。三个 agent 各使用不同的 LLM（可异构，如 Planner 用 gpt-oss-120b、Executor 用 Qwen3-Coder-480B），通过精心设计的 prompt template（包含 NKI API 知识、profiling 术语、编程指南）引导行为。

从编译框架角度拆解术语：
三代理协作的完整流程：
```
┌──────────────────────────────────────────────────────┐
│                   AccelOpt Agentic Workflow           │
│                                                      │
│  Baseline Kernel + Profile ──► Planner               │
│       │                        │                     │
│       │  Past Experiences       │ 分析 bottleneck     │
│       │  (from Memory)    ◄─────┤ 提出 1-step plan    │
│       │                        │                     │
│       ▼                        ▼                     │
│   1. "Hoist LHS Transpose Out of Reduction Loop"     │
│       │                        │                     │
│       ▼                        ▼                     │
│   Executor ◄── Plan ──► 实现具体优化:                 │
│   - 将 v7/v8/v9 transpose 外提到 i1 循环外              │
│   - 分配 global buffer v9_global 存储预计算结果         │
│   - 调整 loop ordering                                │
│       │                                              │
│       ▼                                              │
│   Neuron Compiler → Trainium Hardware → Profiling     │
│       │                                              │
│       ▼                                              │
│   Slow kernel A vs Fast kernel B (speedup 1.2x)      │
│       │                                              │
│       ▼                                              │
│   Summarizer: "Loop Invariant Code Motion for LHS    │
│   Matrix Transposition: The computation of LHS       │
│   transposition is invariant w.r.t loop index i1..."  │
│       │                                              │
│       ▼                                              │
│   Optimization Memory ← Append experience item       │
└──────────────────────────────────────────────────────┘
```

Executor 是性能瓶颈 agent（Table 1）：切换 executor 模型对 speedup 影响显著（Qwen3-Coder-30B: 1.144→1.197, gpt-oss-120b: 1.228→1.235），而切换 planner 模型基本不影响（Table 2, 1.234 vs 1.235 vs 1.234）。

术语一般如何实现？如何使用？
Agent 通信通过 prompt 模板实现：Planner 的 output pattern 稳定，可直接作为 Executor 的输入（无需额外格式化）。Planner prompt 包含 NKI API 基础、profiling 术语（20 项指标）和优化计划指导（5 条原则）；Executor prompt 包含 NKI 编程指南（output dependencies, tensor indexing, tensor usage scope, access variables）；Summarizer prompt 要求输出 "Short description + Full description + Original code + Optimized code" 格式。为增加多样性，profiling items 的展示顺序在每次采样时随机变化。Cost 使用 token-based 计费模型计算。

涉及论文标题：
- AccelOpt: A Self-Improving LLM Agentic System for AI Accelerator Kernel Optimization
