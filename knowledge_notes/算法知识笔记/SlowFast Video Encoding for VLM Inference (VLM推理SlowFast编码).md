## SlowFast Video Encoding for VLM Inference (VLM推理SlowFast编码)

术语解释
VLM推理时对不同视频帧使用不同空间pooling ratio: 关键帧用fine-grained pooling (3×3→~81 tokens/frame)，非关键帧用coarse pooling (9×9→~9 tokens/frame)，在固定total vision token budget下覆盖更多帧。Molmo2在training-free + query-based frame selection模式下实现。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
借鉴SlowFast-LLaVA，在Molmo2 connector的MH attentional pooling层改变window size: default 3×3, fast=9×9。Frame selection策略：(1) periodic (every p-th frame slow)；(2) diff-based (相邻帧feature差异)；(3) query-based (SigLIP 2 embedding cosine similarity between query and each frame)。p selection: 动态选择p∈{1,2,3,4}使total tokens≈10.6K。Key finding: training-free query-based SlowFast在~43% fewer tokens下匹配224 frames全分辨性能；甚至training with SlowFast 10%后反而不如training-free query-based（说明Molmo2可zero-shot generalize到9×9 pooling）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。
```
# Dynamic p selection based on sampled frame count
if F_t <= 128: p = 1; elif F_t <= 224: p = 2; elif F_t <= 300: p = 3; else: p = 4

# Query-based frame selection for slow pathway:
scores = cosine_sim(SigLIP2(query), SigLIP2(frames))  # [F_t]
slow_frames = select_top_global(scores, F_s//2) + select_best_per_group(scores, F_s//2 groups)

# Encode with different pooling:
slow_tokens = connector(slow_frames, pool=3)  # 81 tok/frame
fast_tokens = connector(fast_frames, pool=9)  # 9 tok/frame
# Interleave by temporal order, total ≈ 10.6K tokens
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
当FPS≥2时fallback到periodic（高FPS下frame selection不必要）。与connector的MH attentional pooling天然兼容——仅改变window size。Molmo2 training code支持此功能。适用于>2min长视频VLM推理，需在fixed token budget下覆盖更多视觉信息。

涉及论文标题：
- Molmo2__Open_Weights_and_Data_for_Vision-Language_Models_with_Video_Understanding_and_Grounding
