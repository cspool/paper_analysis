## Video Segment-In-A-Haystack (VideoSIAH)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
VideoSIAH (Video Segment-In-A-Haystack) 是 LongVT 提出的数据套件和评估范式。它模拟长视频理解中的关键挑战：答案所需的关键视觉证据（"needle"）仅存在于视频若干小时的极窄时间窗口（"haystack"）中。与传统的针-in-a-干草问题（Needle-In-A-Haystack, NIAH，测试 LLM 在长文本中检索特定事实的能力）不同，VideoSIAH 将概念扩展至视频领域：(1) 问题证据稀疏且时间上分散；(2) 证据以视觉形式存在，需要模型具备 temporal localization 和 visual reasoning 能力；(3) 采用开放式 QA 格式（而非 MCQ）避免选项记忆偏差。VideoSIAH 包含训练数据（247.9K SFT + 1.6K RL + 15.4K RFT）和评估基准 VideoSIAH-Eval（244 视频、652 QA 对，平均时长 1688s）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
VideoSIAH 数据构造 pipeline（半自动 + human-in-the-loop）：
```
# 1. 场景检测与分割
segments = pixel_level_scene_detection(long_video)
segments = merge_short_segments(segments, min_duration=10s)

# 2. 视频片段描述生成
for segment in segments:
    caption = Qwen2.5-VL-72B.describe(segment)
    segment.caption = caption  # 物体、空间关系、事件演变

# 3. QA 对生成（基于 captions）
for segment in segments:
    qa_pairs = generate_qa_from_caption(segment.caption)
    # 覆盖: temporal events, spatial layouts, motion, 
    #       object attributes, scene transitions

# 4. 两阶段 QA 过滤
# Stage 1: Text-based filtering
qa_pairs = filter_by_linguistic_heuristics(qa_pairs)
qa_pairs = filter_by_model_agreement(qa_pairs)
# Stage 2: Multimodal filtering
for qa in qa_pairs:
    if not GLM-4.5V.verify(qa.answer, segment.video):
        discard(qa)

# 5. Human-in-the-loop refinement
# 人工检查少量代表性失败案例 → 改进 QA 生成/过滤 prompt 规则
# Prompt-feedback refinement loop: 提升可靠性无需全量人工标注

# 6. iMCoTT trace 生成（仅 SFT 阶段）
for qa in filtered_qa_pairs:
    imcott_trace = generate_multiround_tool_trace(
        qa, video, 
        P_multi = 1 - (L_max - clip(L_video, L_min, L_max))/(L_max - L_min)
    )
```
RL 数据额外经过 difficulty-aware filtering：对每个问题采样 K 个 rollouts，若全部正确（太易）或全部失败（太难）则丢弃，仅保留混合结果的 middle-band 样本。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
VideoSIAH 数据集的构造利用了多个 SOTA LMMs（Qwen2.5-VL-72B 描述视频、GLM-4.5V 验证答案、Gemini 2.5 Flash 蒸馏 iMCoTT traces），通过半自动 + human-in-the-loop pipeline 在质量和规模间取得平衡。评估基准 VideoSIAH-Eval 通过 contamination study 验证了零泄漏（Qwen3-VL 在 "No Visual" 设置下得分为 0.00），且开放式 QA 格式天然免疫 MCQ option bias。这种 segment-in-a-haystack 范式特别适合评估需要长视频中精确定位稀疏证据的推理能力。

涉及论文标题：
- LongVT__Incentivizing__Thinking_with_Long_Videos__via_Native_Tool_Calling
