## verl (Volcano Engine RL)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
verl (Volcano Engine Reinforcement Learning) 是字节跳动开源的 RLHF/RL 训练框架，专注于使用强化学习对齐 LLM。基于 Ray 分布式计算框架和 vLLM/SGLang 推理引擎，实现 PPO、GRPO、DPO、RLOO 等主流 RL 算法的分布式训练。核心架构：训练 actor (policy model) + rollout engine (vLLM/SGLang) + reward model + reference model，通过 Ray 协调各组件间的数据流动。verl 的关键设计特性包括：(1) 通过 GPU resource sharing 将推理引擎和训练模型共享 GPU 资源，避免空置；(2) 支持混合引擎模式——多种推理后端可混合部署；(3) 3D-HybridEngine：同时管理 actor model、reference model、reward model 和 rollout engine。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
verl 在 LongVT RL 训练中的完整流程：
```
初始化 (64 GPU cluster via Ray):
  Actor Model (π_θ)     ← 16 GPU (TP=4, DP=4)
  Rollout Engine (SGLang) ← 32 GPU 
  Reference Model (π_ref) ← 8 GPU
  Reward Engine           ← 8 GPU (Judge LLM + IoU计算)

每次 RL iteration:
1. Rollout 阶段 (SGLang):
   for prompt in batch:
       for k in range(16):  # 16 rollouts/prompt
           # SGLang 支持 multi-turn generation:
           y_k = sglang_engine.generate(prompt,
                                         tools=[crop_video_def],
                                         max_turns=5,
                                         temperature=1.0)
           # SGLang 内部使用 RadixAttention prefix caching
           # 在多次 crop_video 调用间复用共享 prompt prefix

2. Reward 计算 (Reward Engine):
   for each rollout:
       # 解析 <answer> 和 tool_call 参数
       R_acc = judge_llm(answer, answer_gt)
       R_fmt = check_format_schema(y_k)
       R_time = IoU(extracted_timespan, gt_timespan)
       R = R_acc + R_fmt + R_time

3. Advantage 计算:
   baseline = mean(R[1..16])
   A = [r - baseline for r in R]

4. 策略更新 (Actor Model):
   for mini_batch in rollouts:
       # KL-constrained policy gradient
       loss = -mean(A * log_prob_ratio) + β * KL(π_θ || π_ref)
       optimizer.step(loss)
```

LongVT 对 verl 的扩展：(1) 支持多轮 multimodal tool-augmented rollouts——在模型生成和工具执行间切换；(2) 扩展 SGLang 后端以处理视觉 token 输入和工具返回的 vision tokens；(3) 支持自定义联合奖励函数（accuracy + format + temporal IoU）。verl 使用 TP=4 data parallel 配置训练 7B 模型，SGLang 提供 16 并发 rollouts 生成能力。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
verl 通过 Python API 使用，核心流程：(1) 定义训练配置 (YAML/JSON: model path, training steps, rollout settings, reward function)；(2) 启动 Ray cluster；(3) verl 自动管理 actor/reference/rollout/reward 组件的生命周期；(4) 通过 Ray Dashboard 监控训练。verl 是 2025 年最活跃的 RL 训练框架之一，与 TRL、OpenRLHF、slime 等并列。其设计灵感来自 DeepSpeedChat，但通过 GPU resource sharing 可达到 2× 更高的训练吞吐量。verl 同时支持 FP8 training、Tensor/Pipeline/Expert 并行等分布式训练特性。

涉及论文标题：
- LongVT__Incentivizing__Thinking_with_Long_Videos__via_Native_Tool_Calling
