## VeRL (Volcano Engine RL)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
VeRL（Volcano Engine RL, https://github.com/volcengine/verl）是由字节跳动火山引擎开发的开源RLHF/RL训练框架，专为LLM的强化学习后训练设计。HybridFlow架构（Sheng et al., 2024）：将训练流程分为rollout generation和actor update两个解耦阶段，支持灵活的资源分配和异构硬件调度。核心特性：(1) 支持多种RL算法（PPO/GRPO/DPO等）；(2) 与vLLM深度集成用于高效rollout生成；(3) 支持FSDP分布式训练；(4) 支持CUDA graph加速生成。在M1论文中，VeRL被用作GRPO RL训练的基础框架。M1对VeRL的关键贡献是修复了Mamba模型在CUDA graph+PyTorch FSDP组合下的兼容性问题——此前CUDA graph与Mamba的recurrent计算模式不兼容导致训练速度慢，修复后Mamba生成速度提升5x（CUDA graph disabled→enabled）。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
```
VeRL在M1 GRPO训练中的系统架构流程：
┌─────────────────────────────────────────────────────┐
│                    VeRL Training Loop                │
├─────────────────────────────────────────────────────┤
│ 1. Rollout Generation (使用vLLM/vLLM-like引擎)       │
│    - 128个question → 每个生成8个rollout (batch=1024) │
│    - CUDA graph enabled (M1修复后)                   │
│    - Mamba recurrent decode: O(1) per token          │
│    - max_new_tokens = 32k                            │
│                                                       │
│ 2. Reward Computation                                 │
│    - 验证每个rollout的\boxed{}答案 vs ground truth     │
│    - 组内(per-question)计算relative advantage         │
│                                                       │
│ 3. Actor Update (FSDP + Adam)                        │
│    - PPO iterations: µ=2                             │
│    - Mini-batch size: 64                             │
│    - Gradient accumulation + FSDP all-reduce         │
│    - 仅更新actor参数 (无需critic model)              │
│                                                       │
│ 4. Checkpoint Management                              │
│    - 每步保存checkpoint                              │
│    - 选择highest critic reward的checkpoint            │
└─────────────────────────────────────────────────────┘
```

M1论文指出DeepScaleR的时间分析显示RL训练中生成阶段耗时超过actor更新（forward+backward）的3倍。Mamba的3x生成加速直接缓解这一瓶颈，使整体RL训练时间大幅缩短。CUDA graph修复进一步带来5x生成速度提升（在VeRL框架内）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
安装：`pip install verl`，需要PyTorch+vLLM环境。使用方式：通过YAML配置文件指定模型路径、RL算法类型（GRPO/PPO等）、训练超参数。M1的关键配置：algorithm=GRPO, batch_size=128, ppo_mini_batch_size=64, rollout_n=8, max_prompt_length=1024, max_response_length=32768。VeRL的设计优势：rollout和update可部署在不同GPU组（异构调度），支持多节点分布式RL训练。开源：https://github.com/volcengine/verl。适用于需要RL post-training的LLM项目，特别是有verifiable rewards的任务。

涉及论文标题：
- M1__Towards_Scalable_Test-Time_Compute_with_Mamba_Reasoning_Models
