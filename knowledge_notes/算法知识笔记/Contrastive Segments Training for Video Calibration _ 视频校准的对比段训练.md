## Contrastive Segments Training for Video Calibration / 视频校准的对比段训练

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
ReVisionLLM 提出的改善 VLM 置信度校准的训练策略。传统 VLM 仅用正样本（包含目标事件的视频段）训练，从未被训练判断"事件不存在"，导致 ECE=0.62，在小时级视频中产生大量高置信度假阳性。Solution: Stage 1 从同视频中随机采样不含目标事件的段（负样本），与正样本 1:1 混合训练，目标简化为 "Does <event> happen? Yes/No."，迫使模型学习区分视觉输入中的存在与不存在。ECE 降至 0.46，+Contrastive Segments 使 R1@.1 从 1.4% 提升至 4.8%。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Stage 1 adapter微调阶段
pos_seg = segment_containing_event(video, gt)
neg_seg = random_non_overlapping_segment(video, gt)

pos_prompt = [SparseFeatures(pos_seg), "Does <event> happen? Yes/No."]
neg_prompt = [SparseFeatures(neg_seg), "Does <event> happen? Yes/No."]

loss = CE(LLM(pos_prompt), "Yes.") + CE(LLM(neg_prompt), "No.")
# 仅更新 Hierarchical Adapter, LLM LoRA 冻结
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
负样本必须从同视频中采样（确保视觉上下文相似），不与 ground truth 时间边界重叠。与 LLM 校准论文 "LLMs must be taught to know what they don't know" (Kapoor et al. 2024) 理念一致。Stage 1 的校准效果传递到 Stage 2 和推理——良好校准的 sparse features 在上层 hierarchy 减少假阳性，使底层的 dense features 处理更高效。

涉及论文标题：
- ReVisionLLM__Recursive_Vision-Language_Model_for_Temporal_Grounding_in_Hour-Long_Videos
