## Query Typology: Global Query vs Localized Query

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Query Typology 是 DIG 提出的视频问答查询分类框架，将查询分为：(1) Global Query (GQ) —— 需要理解和综合整个视频内容，不含特定时空指代词（如 "What title best summarizes this video?", "What is the primary focus?"），回答需要 holistic understanding；(2) Localized Query (LQ) —— 可通过关注特定时间段来回答，包含具体 referents（实体、动作、时间标记，如 "What color is the man's bike at 3:15?"）。这一分类的核心价值在于指导帧选择策略：DIG 证明 uniform sampling 在 GQ 上已足够有效且高效（Figure 5 右侧），而对 LQ 则需专门的 keyframe selection（Figure 5 左侧）。分类方法：LLM（Qwen3-Next-80B-A3B）通过 CoT 4 步推理（理解意图 → 推断视频风格 → 识别 referents → 综合判断）输出 isGlobal: true/false。LQ accuracy >90%，GQ accuracy 38-75%（误分类代价低，走错 branch 最多增加计算开销）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Query Classification + Branch Routing
classification = LLM_classify(Q)  # CoT prompt
if classification.isGlobal:
    frames = uniform_sample(V, N)  # 高效路径
else:
    r_frames = CAFS(V)             # 精准路径
    rewards = LMM_reward(r_frames, Q)
    refined = video_refinement(V, r_frames, rewards)
    frames = uniform_sample(refined, N)
answer = LMM(frames, Q)
```
判断标准：Global = 缺乏具体 referent 或虽有但需 holistic understanding；Localized = 有具体 referent 且可通过关注相关片段回答。效率增益（Table 11）：Query Identification 使 MLVU 节省 13.3% 总时间，VideoMME 节省 19.9%。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Query Typology 分类由任意具备推理能力的 LLM 执行，DIG 使用 Qwen3-Next-80B-A3B-Instruct。分类 prompt 采用 CoT 策略（论文 Figure 11）。benchmark 的 ground truth 标注：MLVU 通过任务结构映射（holistic→GQ, single/multi-detail→LQ），LVB 全为 LQ（referring reasoning 设计），VideoMME 通过人工标注 majority vote。DG 的误分类影响：GQ 误分为 LQ → 多耗计算但 accuracy 接近持平（Figure 5）；LQ 误分为 GQ → 回到 uniform sampling，可能丢失关键信息。

涉及论文标题：
- Divide__then_Ground__Adapting_Frame_Selection_to_Query_Types_for_Long-Form_Video_Understanding
