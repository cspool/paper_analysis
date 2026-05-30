## Planning-before-Perception (先规划后感知)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Planning-before-Perception 是 EVA 提出的视频 Agent 核心范式，与传统的 perception-first（先感知后推理）相对。在 perception-first 范式下，MLLM 先接收均匀采样帧或完整视频作为视觉输入，然后基于这些固定视觉信息进行推理或 tool call——这使得模型被动消费可能无关的视觉 token，且早期视觉噪音可能误导后续规划。Planning-before-Perception 翻转这一流程：agent 在初始状态仅接收 textual query（无视觉输入），先基于 query 进行文本推理生成 explicit plan（明确要观察什么、何时观察、如何观察），再通过 frame_select tool 有针对性地获取视觉信息。通过迭代 summary-plan-action-reflection 循环逐步完善感知。这一范式使 MLLM 从"被动视频识别器"进化为"主动自适应自主 agentic watcher"。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Planning-before-Perception 执行流程
s_0 = {q, h=[], F=[]}  # 初始状态：仅 query，无视觉信息

for round in 1..max_rounds:
    # Step 1: 基于已有信息进行 Planning
    # 无需先看视频，从 query 推理需要什么信息
    plan = MLLM.plan(q, h, F)
    # plan = {
    #   "hypothesis": "需要先获取视频全貌",
    #   "strategy": "low_res_global_scan",
    #   "estimated_action": {start:0, end:T, nframes:10, resize:0.1}
    # }
    
    # Step 2: 执行 Action（选择性获取视觉信息）
    new_frames = frame_select(V, plan.action)
    F = F ∪ new_frames
    
    # Step 3: Summary + Reflection
    summary = MLLM.summarize(new_frames)
    sufficient = MLLM.reflect(q, F, summary)
    if sufficient: break

answer = MLLM.answer(q, h, F)
```

Planning-before-Perception 相比 perception-first 的优势：
1. 避免视觉误导：uniform frames 可能包含不相关/噪音内容误导 planner
2. 节省 visual tokens：仅获取 query 真正需要的视觉信息
3. 主动感知而非被动观察：agent 显式决定需要什么、如何获取、选择性交互

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在 EVA 中通过三阶段训练实现：(1) SFT stage: 用 teacher MLLM (Qwen2.5-VL-72B) 构造 planning-before-perception 格式的训练数据（Summary → Planning → Action → Reflection），冷启动训练 agent 的 tool-call 和推理格式；(2) KTO stage: 纠正 planning 策略中的典型错误（如计划获取不足 visual tokens 但仍强行回答）；(3) GRPO stage: 在线优化 exploration-exploitation 平衡，让 agent 学会根据 query 自适应调整 planning 策略。EVA 的 frame_select tool 提供 start_time/end_time/nframes/resize 四参数灵活控制。使用 vLLM 部署推理。

涉及论文标题：
- EVA__Efficient_Reinforcement_Learning_for_End-to-End_Video_Agent
