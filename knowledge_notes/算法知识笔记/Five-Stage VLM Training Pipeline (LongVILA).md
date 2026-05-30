## Five-Stage VLM Training Pipeline (LongVILA)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
五阶段VLM训练pipeline是LongVILA提出的长上下文视觉语言模型训练流程，将VLM训练从传统的3阶段（对齐→预训练→SFT）扩展为5阶段：(1) Stage1 多模态对齐——冻结LLM和视觉编码器，仅训练多模态投影器（linear/MLP层），桥接视觉与语言模态；(2) Stage2 大规模预训练——冻结视觉编码器，在COYO-25M等大规模图文数据集上训练LLM和投影器，使用VILA-1.5-40B重标注数据提升质量；(3) Stage3 短监督微调（Short SFT）——全参数微调，混合图像和短视频数据（如YouCook2、ShareGPTVideo）；(4) Stage4 上下文扩展（Context Extension）——在进入长视频SFT之前，先用纯文本数据（SlimPajama 17B tokens）对LLM进行持续预训练以扩展上下文窗口，采用渐进式训练调度（8K→65K→262K tokens），配合RoPE基频增大和LoRA微调；(5) Stage5 长监督微调（Long SFT）——全参数训练，使用MM-SP系统，在15,292个长视频的LongVILA_SFT数据集上进行指令微调。Ablation证明Stage4必须在Stage5之前执行才能获得最佳性能（57.5 vs 55.3-56.0 VideoMME average w/o subtitle）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
Algorithm: Five-Stage VLM Training Pipeline
Input: vision_encoder Φ, projector Ψ, LLM Θ

# Stage 1: Multi-modal Alignment
Freeze(Φ); Freeze(Θ)
for (img, caption) in D_alignment:
    v = Ψ(Φ(img))                 # vision features → projector
    loss = CE(Θ([v; text(caption)]), labels)
    Update(Ψ)                     # only projector trained

# Stage 2: Large-scale Pre-training
Freeze(Φ)
for (img, text) in D_coyo_relabeled:
    v = Ψ(Φ(img))
    loss = CE(Θ([v; text]), labels)
    Update(Ψ, Θ)                  # projector + LLM trained

# Stage 3: Short Supervised Fine-Tuning
for (img_or_frames, text) in D_short_mixed:
    v = Ψ(Φ(img_or_frames))
    loss = CE(Θ([v; text]), labels)
    Update(Φ, Ψ, Θ)               # all params

# Stage 4: Context Extension (text-only, LoRA)
SetRoPEBase(Θ, base_freq × scale) # 增大RoPE基频
for text in ProgressiveSchedule(D_slimpajama):
    # 8K → 65K → 262K progressive
    loss = CE(Θ_LoRA(text), labels)
    Update(LoRA_params)

# Stage 5: Long Video SFT (full params, MM-SP)
for (long_video_frames, text) in D_longvila_sft:
    # MM-SP distributes across GPUs
    loss = CE(Θ_SP([Ψ(Φ(frames)); text]), labels)
    Update(Φ, Ψ, Θ)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现基于VILA框架（HuggingFace Transformers），开源在github.com/NVlabs/VILA/tree/main/longvila。Stage4使用LongLoRA方法进行LoRA微调，约需336 GPU hours on 80GB A100。Stage5需要MM-SP系统支持，因为单个长视频样本可达1400帧（约274K tokens），超出单卡内存。数据生成方面，长视频被切分为10秒片段，每片段用VILA-1.5生成caption，再由LLM基于所有片段caption生成QA对。适用于需要处理时长数十分钟到数小时的长视频理解任务。

涉及论文标题：
- LongVILA__Scaling_Long-Context_Visual_Language_Models_for_Long_Videos
