## LLM-as-a-Judge

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
LLM-as-a-Judge 是一种使用大型语言模型作为自动评估器的方法，用于评判模型生成文本的质量、正确性或一致性。在 LongVT 的 RL 训练中，LLM-as-a-Judge 用于评估 open-ended QA 的答案准确性：给定问题 Q、标准答案 A* 和模型回答 A_hat，Judge LLM 输出三级判定——Fully Consistent (语义等价，得 1.0)、Partially Consistent (包含部分正确信息但不完整/不精确，得 0.5)、Inconsistent (错误或矛盾，得 0.0)。使用严格评判协议（图 6）：仅输出 1/0.5/0 数字，避免对模糊案例提供奖励，以确保 RL 奖励信号的可靠性。

从算法pipeline角度拆解术语。通过联网搜索让回答具体和精准。
LLM-as-a-Judge 的评判流程：
```
def judge_answer(model_answer, ground_truth, question):
    prompt = f"""
    Below are two answers to a question.
    Question is: {question}
    [Standard Answer] is: {ground_truth}
    [Model_answer] is: {model_answer}
    
    Judge how consistent the two answers are.
    Scoring rules:
    - 1 Fully consistent: convey the same meaning
    - 0.5 Partially consistent: overlap on some key points but not all
    - 0 Inconsistent: they conflict or share no essential overlap
    
    Output **only** one of: 1, 0.5, or 0.
    """
    verdict = judge_llm(prompt)  # 使用 Qwen3 作为评判模型
    return float(verdict)
```
在 GRPO 训练流程中，每个 rollout 的 answer 部分被提取后送入 Judge LLM 获得分数，该分数直接作为 R_acc 组件进入联合奖励函数 R = R_acc + R_fmt + R_time。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
LLM-as-a-Judge 已被广泛用于 RLHF/RL 训练中的自动评估（取代人工标注），特别适用于 open-ended 生成任务（长文本 QA、摘要、翻译）无法用规则匹配评估的场景。实现注意事项：(1) 评判模型应比训练模型更强或至少相当；(2) prompt 设计需包含明确的分级标准和输出格式约束；(3) 存在 position bias、verbosity bias 等已知偏差，可通过多轮评判或位置随机化缓解；(4) 在 LongVT 中，RL 训练设置恒定 temperature=1.0 以鼓励探索，评判严格性对防止 reward hacking 至关重要。

涉及论文标题：
- LongVT__Incentivizing__Thinking_with_Long_Videos__via_Native_Tool_Calling
- SAGE__Training_Smart_Any-Horizon_Agents_for_Long_Video_Reasoning_with_Reinforcement_Learning

SAGE 中的 LLM-as-a-Judge 使用方式（binary correctness verdict）：
SAGE 使用 GPT-4o 作为 LLM-Judge，但采用简单的 binary verdict（True/False）而非三级评分。Judge prompt: "Compare the model prediction and the ground truth and determine if they convey the same meaning for the question... respond with the verdict as 'True' if they match semantically or 'False' if they don't match." 该 binary 判定直接作为 GRPO accuracy reward a_N 的核心输入。与 LongVT 的三级评分 (1/0.5/0) 相比，SAGE 的二元判定更激进——完全正确得正奖励，部分正确也可能被判为 False 得负奖励——这迫使模型追求完整正确的答案。SAGE 同时将 LLM-as-a-Judge 用于 evaluation（对所有 DIRECT 和 AGENT baselines 统一评估），保持 training 和 evaluation 的一致性。
