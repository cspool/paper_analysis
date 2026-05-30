## Needle-in-a-Haystack Evaluation (for Long Video)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Needle-in-a-Haystack（大海捞针）评估是测试长上下文模型在极长序列中检索特定信息能力的实验方法。对于长视频VLM：构建一个极长的视频序列（如6000帧），在其中特定深度位置（如0%、25%、50%、75%、100%位置）插入"needle"（特殊设计的图像），要求模型回答与该图像相关的问题。评估指标为正确检索的准确率。LongVILA训练于2048帧，但在6000帧（超过1M tokens）测试中达99.8%准确率，证明了其上下文能力的有效扩展。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Long Video Needle-in-a-Haystack
eval_result = {}  # (depth, num_frames) → accuracy

for num_frames in [32, 64, 128, ..., 6000]:
    for depth in [0.0, 0.1, 0.2, ..., 1.0]:  # relative position
        # Create haystack video with num_frames
        base_frames = sample_video_frames(num_frames)
        
        # Insert needle image at depth position
        insert_pos = int(depth * num_frames)
        needle_image = create_needle_image(question_id)
        test_frames = base_frames[:insert_pos] + [needle_image] + base_frames[insert_pos+1:]
        
        # Query model about needle content
        prompt = f"<video>{test_frames}</video> What was shown at position {depth*100}%?"
        answer = model.generate(prompt)
        
        correct = evaluate_answer(answer, ground_truth[question_id])
        eval_result[(depth, num_frames)] = correct

# Plot heatmap: depth (y-axis) × num_frames (x-axis) × accuracy (color)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
源自LLM长上下文评估方法，扩展到视频领域。LongVILA的needle使用特殊设计的图像（如红色圆点或特定文字），模型需描述看到的内容。用于量化评估长视频VLMs在各深度位置的检索能力，是验证模型有效上下文窗口的关键实验。LongVILA是首个在1M+ token上下文中达到99%+准确率的VLM。

涉及论文标题：
- LongVILA__Scaling_Long-Context_Visual_Language_Models_for_Long_Videos
- TSPO__Temporal_Sampling_Policy_Optimization_for_Long-form_Video_Language_Understanding
- VideoRoPE__What_Makes_for_Good_Video_Rotary_Position_Embedding

TSPO 中的 Video Needle-in-a-Haystack 使用方式：TSPO 创新性地将 NIAH 从**评估范式**转化为**训练数据构建范式**。

**VideoRoPE 提出的 V-NIAH-D (with Distractors)**：VideoRoPE 发现标准 V-NIAH 的检索任务过于简单，即使位置编码有缺陷的模型也可以通过 spatial dimension（而非 temporal dimension）定位 needle。V-NIAH-D 在 V-NIAH 基础上（3000 帧 haystack, needle 插入于随机位置），在距 needle 约 200 帧处周期性插入 semantic distractor——与 needle 语义相似但与问题无关的图像（通过 Google Image Search 或 Flux 生成）。distractor 的插入周期由 RoPE 频率特性计算：2·π·10000^(32/128) ≈ 198.7 ≈ 200。该任务暴露了高频 temporal allocation 的缺陷：distractor 帧在 temporal 维度的高频旋转角下与 needle 产生相同的 temporal embedding（"hash collision"），使仅依赖 temporal 维度的模型被误导。MRoPE 在 V-NIAH→V-NIAH-D 上从 78.67% 降至 74.67%（-4.0），而 VideoRoPE 通过低频 temporal allocation 保持在 87.11%（-4.0）。传统 Video NIAH 仅用于评估模型的长程检索能力，TSPO 将其改造为 RL 训练数据管道：(1) 从 LLaVA-Video-178K 采样目标视频，使用 Qwen2.5-VL 生成详细事件描述并重格式化为多选题；(2) 在 segment 级别将目标视频与无关视频拼接/打乱，合成 10∼60 分钟超长训练视频；(3) 这些合成视频自动带有伪标签（目标视频时间边界），用于计算 Temporal Localization Reward R_T = T_t/T_a（采样帧中目标帧占比）。与评估用 NIAH 的关键区别：评估 NIAH 使用合成 needle（图像/文字）插入，TSPO 使用真实视频段作为 needle，保持自然视频分布。该管道产出的数据与 Comprehensive Temporal Data 合并为 TSPO-10K 训练集。
