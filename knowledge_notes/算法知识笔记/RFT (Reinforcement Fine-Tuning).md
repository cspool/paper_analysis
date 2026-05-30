## RFT (Reinforcement Fine-Tuning)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
RFT (Reinforcement Fine-Tuning)，也称 ReFT (Reinforced Fine-Tuning)，是一种后训练范式：在 RL 阶段之后，将模型自身 RL policy 生成的高质量 rollout trajectories 转回监督数据，以 SFT 方式进一步微调模型。在 LongVT 的三阶段 pipeline 中，RFT 是第三阶段：从早期 RL 运行的 rollouts 中筛选同时满足 (1) final answer 正确 AND (2) predicted temporal span 与 ground-truth 的 IoU ≥ 0.3 的高质量 trajectories，转换为 <think> + <tool_call> + <tool_response> + <think> + <answer> 结构的监督训练样本，用标准 next-token prediction loss 微调。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
RFT 数据构造和训练流程：
```
# Step 1: 从 RL rollouts 筛选高质量轨迹
rft_data = []
for rollout in early_rl_rollouts:
    pred_answer = extract_answer(rollout)
    pred_span = extract_time_span(rollout)
    gt_answer = rollout.ground_truth.answer
    gt_span = rollout.ground_truth.time_window
    
    # 双重过滤标准
    answer_correct = (judge_llm(pred_answer, gt_answer) == "Fully Consistent")
    span_accurate = (IoU(pred_span, gt_span) >= 0.3)
    
    if answer_correct and span_accurate:
        # 保留完整交互轨迹作为训练样本
        rft_data.append({
            "prompt": rollout.prompt,
            "completion": rollout.full_text  # 含 think/tool_call/answer
        })

# Step 2: RFT 训练（与 SFT 相同格式，但初始化自最佳 RL checkpoint）
model = load(best_rl_checkpoint)
for step in range(1600):  # 64 GPU, lr=5e-5, cosine schedule
    batch = stream_packing(rft_data, buffer_size=51200)
    loss = -Σ log P(completion_t | prompt, completion_<t)
    update(model, AdamW(loss, lr=5e-5))
```
RFT 的独特价值：RL 阶段通过 exploration 找到好的策略方向，但 RL 训练通常不稳定（reward hacking、policy collapse）；RFT 将成功的探索结果蒸馏为稳定的监督信号，巩固 RL 阶段获得的 temporal grounding 和 tool-calling 模式，使性能超越 RL-only plateau。在 VideoSIAH-Eval 上，RFT 相比 RL-only 有显著提升（42.0 vs 35.9）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
RFT 已在多个场景中被验证有效：(1) OpenAI 的 o1 系列使用 RFT 作为后训练组件；(2) Fireworks.ai 提供 RFT API 服务；(3) 2025 年多篇论文证明 RFT 的 on-policy 数据生成本质上是其抵抗灾难性遗忘的关键因素，而非 KL 正则化或特定 RL 算法。在 LongVT 中，RFT 使用与 SFT 相同的训练基础设施（LMMs-Engine + stream packing），但 compute 从 32 GPU 扩展到 64 GPU，训练约 1600 steps。

涉及论文标题：
- LongVT__Incentivizing__Thinking_with_Long_Videos__via_Native_Tool_Calling
