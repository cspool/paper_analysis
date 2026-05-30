## ReAct / Think-Act-Observe Loop（推理-行动-观察循环）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

ReAct（Reasoning + Acting）是 2022 年 Yao et al. 提出的 LLM agent 范式，将推理（thought）与行动（action）交替交织。一个 ReAct step 包含三个子步骤：(1) Thought — LLM 基于当前 context（prompt + 历史 trajectory）推理当前状态、评估信息是否充足、规划下一步行动；(2) Action — LLM 选择并调用一个工具，指定工具名称和参数；(3) Observation — 工具执行结果被追加到 trajectory context。这三个子步骤构成的 triplet 被反复执行，直到 LLM 判断信息充足并调用 final answer action，或达到最大轮次限制。原始 ReAct 允许多工具并行调用，而 VideoSeek 约束为每轮仅一个工具，以避免 context 跳跃和过早终止。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

VideoSeek 中 ReAct 风格的 agent 算法（Algorithm 1）：

```
Input: query Q, video X, system instruction I, thinking model θ, toolkit T, max_turns N

τ ← [I, Q]                    // trajectory 初始化
T ← T ∪ {answer}

for t = 1 to N:
    (z_t, a_t) ← θ(τ)         // thinking model 读完整 trajectory，输出推理 trace 和工具计划
    if a_t == [answer]:
        Y ← parseAnswer(a_t)
        break
    o_t ← callTools(a_t, X, T) // 执行工具，获取 observation
    τ ← τ ∪ [z_t, a_t, o_t]   // 追加到 trajectory

if Y == null:
    Y ← θ(τ + I_answer)       // 强制回答指令
return Y
```

VideoSeek 的关键变化：工具集为视频专用的 overview/skim/focus 三粒度工具；trajectory 的文本 token 数随轮次线性增长（LVBench 平均 49K tokens, 4.42 turns）。消融：去掉中间推理步骤（直接将 VideoSeek 选中的帧喂给单次 GPT-5）导致 LVBench 从 68.4% 降至 63.9%，说明多轮 reasoning 贡献约 4.5 pp 增益。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现要点：(1) System prompt 定义 Thought/Action/Observation 格式和可用工具集；(2) 工具定义包含名称、用途、参数格式、使用约束；(3) Output parser 提取 action 并路由到对应工具执行器；(4) 视频帧的视觉解释由 LMM API 完成，工具返回的 observation 为文本描述。VideoSeek 的 prompt 分为六部分：Role, Environment, State, Workflow, Toolkit, Operational Rules。开源参考：github.com/jylins/videoseek（ReAct agent 框架 + video toolkit）。

涉及论文标题：
- VideoSeek__Long-Horizon_Video_Agent_with_Tool-Guided_Seeking
