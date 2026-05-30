## Data-Enhanced GRPO (数据增强 GRPO)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Data-Enhanced GRPO 是 EVA 对标准 GRPO 的改进。标准 GRPO 依赖静态训练数据集，模型仅在固定 query-video pairs 上通过有限 epoch 迭代学习——当模型发现自身在某一能力维度（如计数）薄弱时，只能从有限的失败 query 和视频中学习改进。Data-Enhanced GRPO 引入动态数据增强机制：在 GRPO 训练若干步后，收集当前 policy 的 failure cases，以这些 failures 作为 in-context examples 提供给 teacher MLLM (Qwen2.5-VL-72B)，让 teacher 为 HD-VILA 中的新视频生成新的 open-ended QA pairs（条件于 failure examples），然后将新 QA pairs 合并到训练集中继续训练。此机制不断扩展训练数据的多样性，帮助 agent 在更广泛的挑战中学习。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Data-Enhanced GRPO Training Loop
EVA_RL = {9.6k OE + 1.1k MCQ}  # 初始 RL 数据集

for grpo_step in range(total_steps):
    # 1. 标准 GRPO 训练一步
    batch = sample(EVA_RL, bs=64)
    for each (q, v) in batch:
        responses = π_θ.sample(q, v, n=8)  # 8 rollouts
        rewards = [calc_reward(r) for r in responses]
        advantages = (rewards - mean(rewards)) / std(rewards)
        update π_θ with GRPO loss
    
    # 2. 每 N 步：收集 failures + 数据增强
    if grpo_step % N == 0:
        # 收集 KTO 训练后模型的 failure cases
        failures = collect_failures(π_θ, EVA_RL)
        # Teacher MLLM 以 failures 为 in-context examples
        # 为 HD-VILA 新视频生成 open-ended QA pairs
        new_QA = teacher_MLLM(
            video=HD_VILA[v_new],
            in_context_examples=failures,
            prompt="Generate open-ended QA with concise answers"
        )
        # 增强训练集
        EVA_RL = EVA_RL ∪ new_QA
        # 在新增强的数据集上继续训练
```

选择 open-ended QA 而非 MCQ 生成的原因：(1) 缓解 reward hacking（答案猜测）；(2) 对 teacher model 更高效——设计平衡的 MCQ 选项常引入非预期的信息线索和额外复杂度。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Data-Enhanced GRPO 依赖于：(1) 一个强大的 teacher MLLM (Qwen2.5-VL-72B) 用于生成高质量新数据；(2) 未标注的新视频源 (HD-VILA) 提供视频多样性；(3) in-context learning 机制让 teacher 针对模型当前的薄弱环节生成针对性训练数据。这种方法的优势在于持续扩展训练数据分布，打破标准 GRPO 的静态数据限制。实现复杂度较高，需要 teacher model inference 与 RL training 交替进行。

涉及论文标题：
- EVA__Efficient_Reinforcement_Learning_for_End-to-End_Video_Agent
