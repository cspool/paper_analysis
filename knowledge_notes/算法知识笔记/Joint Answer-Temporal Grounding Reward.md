## Joint Answer-Temporal Grounding Reward

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Joint Answer-Temporal Grounding Reward 是 LongVT 在 Agentic RL 阶段提出的三方联合奖励函数，用于同时优化模型的答案正确性、输出格式合规性和时间定位精度。奖励分解为三个独立可加组件：(1) Answer Accuracy R_acc：使用 LLM-as-a-Judge 将模型回答与 ground-truth 比较，分为 Fully Consistent (1.0)、Partially Consistent (0.5)、Inconsistent (0.0) 三档；(2) Format Compliance R_fmt：检查输出是否符合 <think>...</think><tool_call>...</tool_call><answer>...</answer> 的结构化模板，符合则 1、不符合则 0；(3) Temporal Overlap R_time：计算预测时间窗口 [t_s, t_e] 与 ground-truth 窗口 [t_s', t_e'] 的 Intersection over Union (IoU)，R_time = IoU ∈ [0, 1]。最终奖励 R = R_acc + R_fmt + R_time ∈ [0, 3]。

从算法pipeline角度拆解术语。通过联网搜索让回答具体和精准。
奖励计算流程：
```
def compute_reward(rollout, ground_truth):
    # rollout = <think> time window hypothesis </think>
    #           <tool_call> crop_video </tool_call>
    #           <tool_response> frames </tool_response>
    #           <think> verify evidence </think>
    #           <answer> final answer </answer>
    
    # 1. 答案准确性 (LLM-as-a-Judge)
    answer_pred = extract_answer_tag(rollout)
    verdict = judge_llm.predict(answer_pred, ground_truth.answer)
    # verdict ∈ {"Fully Consistent", "Partially Consistent", "Inconsistent"}
    R_acc = {"Fully Consistent": 1.0, 
             "Partially Consistent": 0.5, 
             "Inconsistent": 0.0}[verdict]
    
    # 2. 格式合规性
    R_fmt = 1.0 if matches_schema(rollout) else 0.0
    
    # 3. 时间重叠 (Temporal IoU)
    t_s, t_e = extract_time_window(rollout)  # 从 tool_call 参数提取
    t_s_gt, t_e_gt = ground_truth.time_window
    intersection = max(0, min(t_e, t_e_gt) - max(t_s, t_s_gt))
    union = max(t_e, t_e_gt) - min(t_s, t_s_gt)
    R_time = intersection / union  # IoU ∈ [0, 1]
    
    return R_acc + R_fmt + R_time  # ∈ [0, 3]
```
关键设计选择：(1) 使用 IoU 而非 Recall 作为时间奖励：Recall 允许 policy 放大预测窗口来作弊（span inflation），而 IoU 通过分母 union 项隐式惩罚过度扩展；(2) 解耦 temporal grounding reward：不与 accuracy 耦合，使奖励信号更清晰可解释。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在 RL 训练循环中，每个 rollout 生成后立即计算奖励。LLM-as-a-Judge 使用 Qwen3 作为评判模型，通过严格协议（输出 1/0.5/0）避免对模糊案例的奖励。时间窗口解析从 tool_call JSON 的 start_time/end_time 参数提取。该联合奖励设计避免了 prior work 中 accuracy-only 或 IoU-only 的局限，证明单一任务 RL（仅 video QA）配合 decoupled temporal grounding reward 即可在长视频推理上达到 SOTA。

涉及论文标题：
- LongVT__Incentivizing__Thinking_with_Long_Videos__via_Native_Tool_Calling
