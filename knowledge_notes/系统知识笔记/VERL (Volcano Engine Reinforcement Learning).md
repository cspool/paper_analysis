## VERL (Volcano Engine Reinforcement Learning)

术语解释
VERL (Volcano Engine Reinforcement Learning for LLMs) 是字节跳动 Seed 团队发起的开源 LLM 强化学习框架，基于 HybridFlow 架构（EuroSys 2025 接收）。提供统一的 RL 训练平台，支持 PPO、GRPO、ReMax、REINFORCE++、RLOO、PRIME、DAPO、DrGRPO、VAPO、GSPO 等多种 RL 算法。通过 hybrid-controller 编程模型实现灵活的算法扩展，支持 FSDP/FSDP2/Megatron-LM 训练后端和 vLLM/SGLang/HuggingFace 推理后端的无缝切换。2025 年关键里程：Doubao-1.5-pro 基于 VERL 训练达到 AIME 70.0% pass@1；DAPO 算法在 Qwen2.5-32B 上实现 AIME 50 分。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
VERL 是一个 "通用 LLM-RL 训练操作系统"：将 RL 训练的 actor rollout（推理引擎）、reward computation（奖励计算）、policy update（训练引擎）三个阶段解耦为独立的、可扩展的组件。核心组件：(1) Rollout Engine —— 使用 vLLM 或 SGLang 进行大规模 policy rollouts（生成候选响应）；(2) Training Engine —— 使用 FSDP/FSDP2/Megatron 进行 actor 模型参数更新；(3) Hybrid Engine —— 在训练和推理模式间无缝切换，避免 GPU 资源浪费（actor model resharding）；(4) Reward Manager —— 支持 verifiable rewards (accuracy) 和 learned reward models。v0.3.0+ 引入 3D-HybridEngine 实现约 1.4× 加速，v0.6.0 引入 Model Engine 和 Rollout Server 原生 server 模式。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
VERL 在 Mirage 中的 RL 训练流程：
```
┌──────────────────────────────────────────────────────┐
│                    VERL Training Loop                 │
├──────────────────────────────────────────────────────┤
│                                                      │
│  1. Actor Model (Qwen2.5-VL-7B after Stage 2 SFT)   │
│     ├── 加载到 GPU memory (单 H100)                  │
│     └── Reference Model (同结构, frozen)             │
│                                                      │
│  2. Rollout Phase (per input query x_i):             │
│     ├── Actor 生成 num_rollouts=5 个候选响应          │
│     │   y_i^(1), y_i^(2), y_i^(3), y_i^(4), y_i^(5) │
│     ├── 每个响应: max 1024 output tokens             │
│     └── 包含: <think> latent_tokens </think> \boxed{}│
│                                                      │
│  3. Reward Computation:                              │
│     r_acc = 1 if answer correct else 0               │
│     r_fmt = 0.1 if format valid else 0                │
│     R = σ_c·r_acc + σ_f·r_fmt (σ_c=0.9, σ_f=0.1)    │
│                                                      │
│  4. GRPO Update (per group of 5 rollouts):           │
│     ├── μ = mean(R^(1..5)), σ = std(R^(1..5))        │
│     ├── A^(k) = (R^(k) - μ) / σ                      │
│     ├── L_GRPO = -E[min(ratio·A, clip(ratio,1-ε,1+ε)·A)]│
│     │            + λ_kl·KL(π_θ||π_ref)               │
│     ├── λ_kl=0.01 (excludes latent visual tokens)    │
│     └── Optimizer step (lr=1e-6, mini_batch=8)       │
│                                                      │
│  Config: batch_size=32, grad_accum=4, epochs=15      │
│  Hardware: single NVIDIA H100 GPU                     │
└──────────────────────────────────────────────────────┘
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：开源地址 https://github.com/volcengine/verl。使用方式：(1) 配置 YAML/Hydra 训练参数（algorithm, model, reward, trainer）；(2) 指定 rollout engine (vLLM/SGLang) 和 training backend (FSDP/Megatron)；(3) 定义自定义 reward function（Mirage 中: accuracy + format rewards, σ_c=0.9, σ_f=0.1）；(4) 启动分布式训练（verl 支持 PP/TP/DP/EP 等多维并行）。在 Mirage 中，VERL 的关键配置：(a) KL 正则仅应用于 text tokens，latent visual tokens 排除——这是对标准 GRPO 实现的扩展；(b) rollout num=5, mini batch=8, lr=1e-6；(c) prompt length limit=1024, response length limit=1024；(d) entropy regularization disabled (λ_en=0.0)。适用于任何需要对 LLM/VLM 进行 RL fine-tuning 的场景（reasoning, alignment, agent training）。

涉及论文标题：
- Machine_Mental_Imagery__Empower_Multimodal_Reasoning_with_Latent_Visual_Tokens
- LongVT__Incentivizing__Thinking_with_Long_Videos__via_Native_Tool_Calling
- ResAdapt__Adaptive_Resolution_for_Efficient_Multimodal_Reasoning
