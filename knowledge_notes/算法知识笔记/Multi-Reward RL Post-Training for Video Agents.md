## Multi-Reward RL Post-Training for Video Agents

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Multi-Reward RL Post-Training 是 SAGE 为训练 any-horizon video agent 提出的 GRPO 多奖励设计方案。单一 accuracy reward（如 Video-R1 的 ROUGE/option-matching）对 open-ended 问题无效，而 naive 实现常导致工具调用过度或崩溃。SAGE 的解决方案：R_i = Σ step_rewards + accuracy_reward，uniformly 赋给 trajectory 中所有 actions。Step rewards 分解为四个可加组件：(1) s_format: +0.05 若 JSON 仅含必需字段，否则 -0.10；(2) s_reasonable-tool: GPT-4o 判断当前 tool call 合理→+0.10，否则 -0.10；(3) s_args-repeat: -0.05·√num_repetitions 惩罚重复参数；(4) s_args-valid: -0.10 惩罚无效参数。Accuracy reward a_N 由 GPT-4o binary judge 决定：正确+visual tools→+1.25, 正确无tools→+1.0, 错误→-0.5, JSON无效→-2.0。Step reward 总值被设计为与 accuracy reward 可比（10步累积 ≈ 1.25）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
def compute_sage_reward(trajectory, Q, ground_truth):
    step_rewards = []
    for j, step in enumerate(trajectory):
        s_format = +0.05 if valid_json(step.action) else -0.10
        s_reasonable = +0.10 if GPT4o_judge_tool(step, Q, trajectory[:j]) else -0.10
        s_repeat = -0.05 * sqrt(count_repetitions(trajectory, step.action))
        s_valid = -0.10 if invalid_args(step.action) else 0
        step_rewards.append(s_format + s_reasonable + s_repeat + s_valid)
    
    # 仅在 trajectory 结束时计算 accuracy reward
    final_action = trajectory[-1]
    if invalid_json(final_action):
        a_N = -2.0
    else:
        verdict = GPT4o_judge(final_action.answer, ground_truth)  # True/False
        if verdict:
            used_visual = any(tool in {'extract-video-parts','ground-event'} 
                            for step in trajectory)
            a_N = +1.25 if used_visual else +1.0
        else:
            a_N = -0.5 if len(trajectory) >= 1 else -2.0
    
    R_i = sum(step_rewards) + a_N  # uniform for all actions
    return R_i
```

关键设计决策：(1) 正确回答 + visual tools 额外 +0.25 bonus（鼓励视觉信息利用）；(2) 错误回答 + tool calls 惩罚 -0.5（补偿正向 step rewards，防止 tool overcalling）；(3) JSON 无效直接 -2.0（强制格式合规）。前100步 N_max=6 避免长 trajectory 的方差过大导致训练不稳定。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
该 multi-reward 设计的关键工程选择：(a) Step-level rewards 累积值需与 accuracy reward 量级匹配（10步 maximal step rewards ≈ 1.25，等于 accuracy reward 上限）；(b) GPT-4o 用于 reasonable-tool 和 accuracy 评判（需 carefully designed prompt 协议）；(c) RL 数据构造：7680 样本，half tool-calls half single-turn，确保 any-horizon 学习信号平衡。与 LongVT 的 Joint Reward（R_acc + R_fmt + R_time）相比，SAGE 的奖励设计更细粒度（4种 step rewards vs 2-3种），且加入了 tool overcalling 惩罚和 visual tool bonus。SAGE 的奖励仅需 QA accuracy judge（无需 temporal IoU），简化了 reward computation pipeline。

涉及论文标题：
- SAGE__Training_Smart_Any-Horizon_Agents_for_Long_Video_Reasoning_with_Reinforcement_Learning
