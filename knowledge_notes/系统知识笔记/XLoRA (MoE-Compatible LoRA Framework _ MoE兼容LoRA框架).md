## XLoRA (MoE-Compatible LoRA Framework / MoE兼容LoRA框架)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

XLoRA (Buehler & Buehler, 2024) 是一个将 Mixture-of-Experts 架构与 LoRA 参数高效微调相结合的开源框架。它允许在 HuggingFace Transformers 模型中，将多个 LoRA adapter 作为 expert 模块部署，并通过路由机制动态选择和组合。核心特性：(1) 兼容标准 HuggingFace PEFT——每个 expert 是一个标准 LoRA adapter；(2) 支持 per-layer 的 expert 路由——可在每个 Transformer layer 处插入路由层；(3) 灵活的 expert 数量配置——用户可指定每个 layer 使用的 expert 数量和路由策略；(4) 推理时动态 expert 选择——根据输入 hidden state 实时计算路由权重。Self-MoE 论文使用 XLoRA 作为 MiXSE 的底层 MoE 框架实现。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

XLoRA 在 Self-MoE MiXSE 系统架构中的角色：

```
┌─────────────────────────────────────────────┐
│           MiXSE Runtime (基于 XLoRA)          │
├─────────────────────────────────────────────┤
│  Input Token x                               │
│       ↓                                      │
│  ┌─────────────────────────────────────┐     │
│  │  Base LLM Forward Pass (θ_0)        │     │
│  │  For each Transformer Layer:        │     │
│  │    ├─ Attention (standard)          │     │
│  │    ├─ Router θ_r (linear layer)     │     │
│  │    │   α = top-k(softmax(θ_r @ x))  │     │
│  │    ├─ LoRA Expert 0 (knowledge)     │     │
│  │    ├─ LoRA Expert 1 (reasoning)     │     │
│  │    ├─ LoRA Expert 2 (math)          │     │
│  │    ├─ LoRA Expert 3 (coding)        │     │
│  │    └─ Combine: h = θ_0x + Σα_i·Δθ_i│     │
│  └─────────────────────────────────────┘     │
│       ↓                                      │
│  Output Logits                                │
└─────────────────────────────────────────────┘

组件管理:
- XLoRA 管理 4×L 个 LoRA adapter (L=layers)
- Router 在所有 LoRA 层间共享（1 个线性层）
- HuggingFace PEFT 负责 adapter 加载/卸载
- XLoRA 负责 per-layer forward 时的 expert 选择和组合
```

关键系统设计：(1) Router 共享——同一 router 在模型所有 LoRA 层中复用，大幅减少额外参数；(2) LoRA adapter 的按需激活——仅 top-k 选中的 expert 执行 LoRA 前向计算；(3) base model θ_0 始终活跃，确保基础能力不受 expert 选择影响。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

XLoRA 开源实现 (Buehler & Buehler, 2024)，基于 HuggingFace PEFT 库构建。典型使用流程：(1) 独立训练各领域 LoRA adapter（HuggingFace PEFT）；(2) 使用 XLoRA 加载所有 adapter 并配置 router；(3) 训练 router（可选冻结 adapter）；(4) 推理时 XLoRA 自动管理 expert 路由和组合。Self-MoE 中 XLoRA 用于实现 token-level 的 LoRA expert 动态选择，训练仅需约 1 GPU天（A100-80GB）。

涉及论文标题：
- Self-MoE Towards Compositional Large Language Models with Self-Specialized Experts
