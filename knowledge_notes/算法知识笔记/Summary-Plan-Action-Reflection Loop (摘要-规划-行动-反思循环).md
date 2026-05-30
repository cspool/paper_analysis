## Summary-Plan-Action-Reflection Loop (摘要-规划-行动-反思循环)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Summary-Plan-Action-Reflection 是 EVA 提出的迭代视频理解推理循环范式，受 Zhang et al. (2025) 的 Agent Learning via Early Experience 启发。每个循环包含四个阶段：(1) Summary —— MLLM 对当前返回的帧生成详细内容描述，显式推动模型关注返回的视觉证据并更好理解 tool 参数和输出之间的关系；(2) Planning —— 基于当前信息（query + 历史 + summary）推理潜在 actions，估算每个 action 的 token cost 和 expected outcome，从大 action space 中选择最优策略；(3) Action —— 生成具体的 frame_select tool call JSON（start_time, end_time, nframes, resize）；(4) Reflection —— 评估当前视觉信息是否充足，若不足则生成下一步策略，若充足则终止循环并生成最终答案。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
EVA 的 SFT 数据实例格式精确对应这四个阶段：
```
Round 1:
  Summary: "The low-resolution frames show various gameplay scenes involving
            characters and text references..."
  Planning: "To pinpoint the exact moment, I will select a segment around
             frames showing scoring attempts. The frame at [03:24] shows a
             character near a goal post. Focusing on [03:20]-[04:10] with
             higher resolution makes sense."
  Action: {"tool": "frame_select", "arguments": {
           "start_time": 200, "end_time": 250, 
           "nframes": 100, "resize": 0.4}}
  Reflection: "The increased resolution provides clearer visuals...
               Based on the gameplay analyzed so far, the trigger is evident.
               No further analysis is necessary."

Round 2:
  Planning: (evaluates that evidence is sufficient)
  Action: (no tool call — generates final answer)
  Answer: "The GOAAAAAL!! animation is triggered when the ball..."
```

循环终止条件：Reflection 阶段判断视觉信息已充足 → 直接生成答案（不再 tool call）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在 EVA 中通过 Multi-Agent Data Pipeline 构造训练数据：Executor agent 分析 context 并评估 actions → Reflective Thinker 审计 tool call 参数合理性（检查 fps 是否 >1、visual budget 是否太小等规则）→ 成功轨迹存入 Experience Bank 供未来检索引导 Executor。各阶段对最终性能的贡献通过 SFT→KTO→GRPO 消融实验间接验证：SFT 学习格式但低效（多帧多轮低 accuracy）→ KTO 减少帧数和轮数提升 accuracy → GRPO 增加轮数但更精准分配 token。

涉及论文标题：
- EVA__Efficient_Reinforcement_Learning_for_End-to-End_Video_Agent
