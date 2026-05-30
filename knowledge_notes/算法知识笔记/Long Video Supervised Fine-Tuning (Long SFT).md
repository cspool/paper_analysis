## Long Video Supervised Fine-Tuning (Long SFT)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Long Video SFT（长视频监督微调）是LongVILA的Stage5训练阶段，在完成上下文扩展后，使用长视频指令数据对VLM进行全参数监督微调。关键创新：(1) 数据生成pipeline——长视频先被切分为约10秒的短片段，每片段由VILA-1.5模型独立生成描述性caption，然后由LLM基于所有片段caption生成问答对（涵盖总结、空间、属性、动作、对象、OCR、时序等7类问题）；(2) LongVILA_SFT数据集——来自Shot2Story20k的15,292个长视频，涵盖Travel/Sports/Education等12个类别，每个视频配有1个caption question和1个QA question；(3) 使用MM-SP系统进行分布式训练，因为单个样本可达1400帧（约274K tokens），远超单GPU显存。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Long Video Data Generation Pipeline
def generate_long_video_sft_data(video_path):
    # Step 1: Segment long video into 10-second clips
    clips = segment_video(video_path, clip_duration=10)  # 10s each
    
    # Step 2: Generate captions per clip using VILA-1.5
    captions = []
    for clip in clips:
        frames = sample_frames(clip, num_frames=8)
        caption = VILA_15.generate_caption(frames)  # short-context VLM
        captions.append(caption)
    
    # Step 3: Generate QA pairs using LLM from all captions
    prompt = f"Based on these clip descriptions:\n{captions}\nGenerate questions about: summary, spatial relations, attributes, actions, objects, OCR, temporal events."
    qa_pairs = LLM.generate(prompt)  # text-only LLM
    
    return {"frames": sample_frames(video_path, num_frames=256), 
            "qa_pairs": qa_pairs,
            "captions": captions}
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现依赖MM-SP系统进行分布式训练（SP degree ≥ 4）。全参数微调（所有视觉编码器+投影器+LLM参数可训练），与Stage3短SFT的区别在于数据是长视频（数百到数千帧）且需要SP系统支持。Batch size设为1（受限于单序列超长）。适用于需要让VLM理解数十分钟到数小时长视频内容的应用，如体育赛事分析、电影理解、监控视频摘要等。

涉及论文标题：
- LongVILA__Scaling_Long-Context_Visual_Language_Models_for_Long_Videos
