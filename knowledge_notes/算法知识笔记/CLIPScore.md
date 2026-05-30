## CLIPScore

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
CLIPScore 是基于 CLIP 模型的跨模态相似度评估指标（Hessel et al., 2021）：图像经 CLIP 视觉编码器得 v，文本得 t，CLIPScore = cosine_sim(v, t)。在视频理解中广泛用于 query-aware frame selection——对候选帧计算与查询文本的 CLIPScore，选 top-K 高分帧。DIG 揭示其关键局限：(1) 表面特征匹配——无法捕捉多步推理或世界知识；(2) 视觉偏差——倾向给含常见物体的帧高分；(3) 缺乏上下文推理——无法评估帧间关联。DIG Table 2 证明 LMM-based reward 在所有 frame count 和 benchmark 上一致优于 CLIPScore。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# CLIPScore in query-aware selection
v_i = CLIP_vision(f_i)  # L2 normalized
t = CLIP_text(Q)        # L2 normalized
CLIPScore_i = dot(v_i, t)
selected = TopK(candidates, key=CLIPScore, k=N)
```
典型失效（DIG 分析）："Why did the character leave?" → CLIPScore 给含"人物+房间"的帧高分，但答案取决于时序因果推理而非单帧内容。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
使用 OpenAI CLIP 或 OpenCLIP 预训练权重，单帧评分仅需 ms 级别。DIG 将 CLIPScore 作为 LMM Reward 的对比 baseline，证明在复杂推理上 LMM Reward 显著优于 CLIPScore（LVB 128 frames: +4.2%）。

涉及论文标题：
- Divide__then_Ground__Adapting_Frame_Selection_to_Query_Types_for_Long-Form_Video_Understanding
