## VTG (Video Temporal Grounding，视频时间定位)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Video Temporal Grounding (VTG) 是视频理解领域的核心任务之一：给定一段未裁剪视频 v 和一个自然语言查询 q（描述某个特定事件），模型需输出该事件在视频中发生的具体时间片段 S = (t_start, t_end)。形式化定义：输入 (v, q)，输出 S = (t_start, t_end) 使得视频在 [t_start, t_end] 区间内的视觉内容与查询 q 的描述语义匹配。VTG 的核心难点：(1) 需要细粒度的时序感知能力，而非粗粒度的语义聚合；(2) 需要建模长时序视觉动态（appearance-centric features 难以标注和学习）；(3) 需要精确的边界定位（start/end boundary precision）。评估指标：R1@m（top-1 预测与 ground-truth 的 IoU 超过阈值 m 的比例，m ∈ {0.3, 0.5, 0.7}）和 mIoU（所有测试样本的平均 IoU）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
VTG 的 MLLM-based 处理流程（TimeLens 的 thinking-free RLVR paradigm）：
```
# Input: video v, query q (e.g., "When does the person turn off the light?")
# Output: timestamps (t_start, t_end)

# Step 1: 视频帧采样与时间戳编码
frames = sample_frames(v, fps=2)  # T frames
for i, frame in enumerate(frames):
    t_i = i / 2.0  # 绝对时间（秒）
    text_token = tokenizer(f"{t_i:.1f}s")  # e.g. "10.2s"
    visual_token = vision_encoder(frame)  # frozen ViT
    # Interleaved: timestamp before visual
    sequence.append([text_token, visual_token])

# Step 2: 追加 prompt 和 query
sequence = [prompt_tokens, query_tokens] + sequence

# Step 3: MLLM 生成时间片段
output = LLM(sequence)  # autoregressive generate
# Output: "The event happens in 5.2 - 12.7 seconds"
(t_start, t_end) = parse(output)

# Step 4: 评估（仅在训练/测试时）
IoU = intersection(t_start, t_end, t*_start, t*_end) 
    / union(t_start, t_end, t*_start, t*_end)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
VTG 的主流实现方式经历了三代演进：(1) 传统方法：基于 proposal-based 或 proposal-free 的专用 VTG 模型（如 2D-TAN、Moment-DETR），使用预提取的视频特征和文本特征，通过 proposal ranking 或 span prediction 定位时间边界；(2) MLLM-based 方法：利用预训练 MLLM（Qwen2.5-VL、InternVideo2 等）的多模态理解能力，通过 timestamp encoding 将时间信息注入模型，使用 SFT 或 RLVR 进行后训练优化（TimeLens、Time-R1、TRACE 等）；(3) 自监督方法：VideoSSR 等通过自监督 pretext tasks（anomaly grounding、temporal jigsaw）无需人工标注即可提升 VTG 能力。TimeLens 的核心贡献：(a) 数据质量保证（TimeLens-Bench 手动重标注 + TimeLens-100K 自动化重标注）；(b) thinking-free RLVR 训练范式；(c) interleaved textual timestamp encoding。

涉及论文标题：
- TimeLens__Rethinking_Video_Temporal_Grounding_with_Multimodal_LLMs
