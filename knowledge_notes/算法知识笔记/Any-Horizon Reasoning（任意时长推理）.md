## Any-Horizon Reasoning（任意时长推理）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Any-Horizon Reasoning 是 SAGE 提出的视频推理范式，指模型能够根据任务难度和视频时长自适应地选择推理策略：对简单/短视频问题采用单轮推理直接输出答案（DIRECT模式），对复杂/长视频问题采用多轮工具调用逐步聚合信息（AGENT模式）。该概念受人类行为启发——人类看短视频会完整观看后回答，看2小时长视频则会迭代式地定位关键信息。核心机制：orchestrator VLM (SAGE-MM) 在 Stage-1 (Context VLM) 输出 video_context + query_intent + 首步action，若当前信息充足则直接输出 final_answer（单轮），否则进入 Stage-2 (Iterative Reasoner) 多步 tool-calling（最多11轮）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Any-Horizon Reasoning 决策流程
def any_horizon_reason(SAGE_MM, F, M, T, Q):
    # Stage-1: Context VLM — 首步判断
    action_1 = SAGE_MM(prompt=[T, F, Q, M])
    if action_1.final_answer is not None:
        return action_1.final_answer  # 单轮推理（短/简）
    
    # Stage-2: Iterative Reasoner — 多轮工具调用（长/难）
    history = [action_1]
    C = action_1.video_context
    for step in range(2, N_max+1):  # N_max=11
        S_j = {T, Q, M, C, history}
        action_j = SAGE_MM(prompt=S_j)
        if action_j.final_answer is not None:
            return action_j.final_answer
        tool_result = execute(action_j.recommended_tool)
        history.append(tool_result)
```

训练中通过 GRPO 多轮 reward 塑造 any-horizon 行为：单轮正确回答 +1.0，多轮+工具正确回答 +1.25（额外奖励鼓励视觉工具使用），多轮错误回答 -0.5（惩罚不必要的 tool overcalling）。RL 前100步使用 N_max=6 稳定训练，之后扩大至11步。SFT 是必须的——直接 RL 导致 collapse to single-turn（base model 的训练目标强烈偏向直接产出答案）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Any-Horizon 的核心在于训练数据设计：RL 数据中一半样本需要 tool calls，一半是 single-turn，使模型学会判别两者。表 9 展示了 any-horizon 效果：SFT 模型从 expert Gemini-2.5-Flash 蒸馏获得强 single-turn 能力（79.0% accuracy）但 overcall tools（1038 multi-turn vs expert's 885），RL 后 single-turn 升至 948 样本（+242）、multi-turn 降至 796（-242），分布更接近 expert，且 multi-turn accuracy 从 53.7% 升至 54.3%（+0.6%）。推理时模型可自主决策推理步数——表 15 显示随着视频时长增加（0-60s → 2400+s），平均 turns 从 1.74 渐变至 2.77。

涉及论文标题：
- SAGE__Training_Smart_Any-Horizon_Agents_for_Long_Video_Reasoning_with_Reinforcement_Learning
