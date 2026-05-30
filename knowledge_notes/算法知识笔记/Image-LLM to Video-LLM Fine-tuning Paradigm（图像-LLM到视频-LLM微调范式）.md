## Image-LLM to Video-LLM Fine-tuning Paradigm（图像-LLM到视频-LLM微调范式）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Image-LLM to Video-LLM Fine-tuning 是当前开发视频-LLM 的两条主流路线之一：基于预训练的 Image-MLLM（通常已完成 vision-text 对齐预训练和 instruction tuning），通过额外的视频数据 fine-tuning 使其具备视频理解能力，无需从头进行 vision-text 对齐。区别于另一条路线（从 pretrained LLM 开始，先做 vision-text 对齐再做 video instruction tuning，如 VideoLLaMA 2、VITA），此路线利用 Image-LLM 中已内置的丰富视觉知识（来自大规模 image-text 数据的预训练），仅需较少的视频数据即可激活时序理解能力。Sparrow 论文使用的两个 base model 均属此范式：(1) Mini-InternVL-Chat-4B-V1.5（基于 InternLM2, 3.8B，支持最多 13 子图 patch，每子图 256 visual tokens）；(2) MiniCPM-Llama3-8B-V2.5（基于 LLaMA3-8B，最多 10 patch，每 patch 96 visual tokens）。InternVL 训练时冻结 vision encoder（保留预训练视觉知识），MiniCPM-8B 全量训练，lr=5e-6。训练时关闭动态分辨率 patchifying。关键发现：Image-LLM 的 zero-shot 视频理解能力已很强（InternVL-4B: Video-MME 52.5），甚至超过部分专用 Video-LLMs（VideoChat2 7B: 39.5），归功于大规模 image-text 预训练。Video fine-tuning 可在 zero-shot 基础上额外提升 ~3.8 points。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# === Image-LLM → Video-LLM Fine-tuning Pipeline ===
# 模型架构: ViT(vision_encoder) + MLP(projector) + LLM(backbone)
# 标准 MLLM 架构天然支持多帧——视频仅是多帧扩展

def video_finetune_step(image_llm, video_sample, config):
    """
    config:
      - InternVL: freeze_vit=True,  max_frames=64, tokens_per_frame=256
      - MiniCPM:  freeze_vit=False, max_frames=24, tokens_per_frame=96
    """
    # 1. 帧提取（FPS=1，超出则均匀降采样）
    frames = extract_frames(video_sample['video'], fps=1)
    if len(frames) > config.max_frames:
        frames = uniform_downsample(frames, config.max_frames)

    # 2. 逐帧视觉编码
    visual_tokens = []
    for frame in frames:
        if config.freeze_vit:
            with torch.no_grad():
                tokens = ViT(frame)  # [H_patch*W_patch, C]
        else:
            tokens = ViT(frame)
        visual_tokens.append(tokens)
    # visual_tokens: [T, num_tokens_per_frame, C]

    # 3. Projector 映射到 LLM embedding space
    visual_emb = Projector(visual_tokens)  # [T*num_tokens, d_llm]

    # 4. 与 text token 拼接 + 自回归生成
    text_emb = LLM.embed_tokens(tokenize(video_sample['instruction']))
    input_emb = concat([visual_emb, text_emb], dim=0)
    logits = LLM(input_emb)
    loss = -log P(answer | frames, instruction)

    return loss

# 训练数据: video-caption (ShareGemini 100K) + video-instruction (Video-ChatGPT 100K)
# 数据处理: 1:1 采样 video-caption 和 video-instruction
# GPU: 200K 全量 276.8 GPU hours，30K Sparrow 33.6 GPU hours
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现关键点：
1. **架构零修改**：Image-LLM 的 ViT+Projector+LLM 架构无需任何修改——视频仅是多帧输入，每帧独立通过 ViT 编码后按时间顺序拼接。这与多图 Image-LLM 推理完全一致。数据加载层将视频帧序列打包为 multiple images 即可。
2. **训练策略选择**：
   - 冻结 vision encoder（InternVL 方案）：利用预训练视觉知识，减少视频数据需求，降低训练成本。vision encoder 参数不更新，仅训练 projector + LLM。
   - 全量训练（MiniCPM 方案）：允许视觉特征适配视频域，获得更定制化的视频表示，但需要更多数据和计算。
3. **帧数与效率的平衡**：短视频为主的训练数据中，超出 24-64 帧可能引入冗余而非新信息（Sparrow 验证 48 帧无助于长视频理解）。关闭动态分辨率 patchifying 可固定每帧 token 数提升训练效率。
4. **评估**：Video-MME（短/中/长三段式评估）、MVBench（20 个视频任务）、TempCompass（时序理解）、LongVideoBench/MLVU（长视频）。评估方法推荐 exact matching + LLM matching 组合，因部分模型不严格遵守格式要求。
5. **Sparrow 增强**：在维持此范式不变的前提下，通过 text-to-image 数据增强改进数据质量（而非模型架构或训练协议），实现了 8.2× 训练效率提升。

Image-LLM zero-shot baseline 已超过部分专用 Video-LLM 的发现说明：视频理解的很大一部分基础能力（目标识别、OCR、场景理解）来自 image-text 预训练；视频 fine-tuning 主要注入时序/因果推理能力。这解释了为何 Instruction Diversity 比数据量更重要——因为模型真正从视频数据中学到的是"如何跨帧推理"，而非"如何理解单帧内容"（那已经在 image 预训练中完成了）。

涉及论文标题：
- T2Vid__Translating_Long_Text_into_Multi-Image_is_the_Catalyst_for_Video-LLMs
