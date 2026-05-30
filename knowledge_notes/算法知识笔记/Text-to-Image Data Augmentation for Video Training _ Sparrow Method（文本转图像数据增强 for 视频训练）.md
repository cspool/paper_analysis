## Text-to-Image Data Augmentation for Video Training / Sparrow Method（文本转图像数据增强 for 视频训练）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Sparrow Method 是一种无需调用视觉 API 的纯工程化数据增强方法，核心是将纯文本 instruction 数据转化为"类视频"的多图像序列，以桥接文本-视觉模态差异并丰富训练数据的 instruction 多样性。方法流程：(1) 从文本 instruction 数据集（LongAlpaca, LongQLora）取 (long_context, instruction, answer) 三元组；(2) NLTK 按 ~115 词分割 long_context 为多段；(3) 每段用 Pillow ImageFont 渲染为 448×448 白底黑字图像（20pt Arial Regular 字体，黑色，左右 20px margin）；(4) 生成 (images[], instruction, answer) 序列，格式与真实视频样本完全一致，可直接混合训练。与 TOPA/T3 等文本辅助方法不同：Sparrow 不提取视觉信息再转文字（信息损失），不调用 LLM API（零额外成本），而是将文字直接转为视觉表示。Sparrow 用 30K 混合数据（20K video + 10K synthetic）达到了 200K 纯视频数据相当的 Video-MME 性能（56.7 vs 56.3），GPU hours 从 276.8 降至 33.6（8.2× efficiency）。长视频理解额外提升 6.6 points（100K 规模）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# === Sparrow 数据增强 Pipeline ===

# Step 1: 文本 → 图像合成
def text_to_video_like(text_sample):
    # 1.1 NLTK 分词并按词数分割（~115 words/segment）
    import nltk
    from PIL import Image, ImageDraw, ImageFont
    words = nltk.word_tokenize(text_sample['long_context'])
    chunks = []
    for i in range(0, len(words), 115):
        chunks.append(" ".join(words[i:i+115]))

    # 1.2 PIL 渲染每个 chunk 为 448×448 图像
    images = []
    font = ImageFont.truetype('arial.ttf', 20)
    for chunk_text in chunks:
        img = Image.new('RGB', (448, 448), color='white')
        draw = ImageDraw.Draw(img)
        # 逐词绘制，控制换行（可用宽度 = 448 - 40 = 408px）
        y, current_line = 20, []
        for word in chunk_text.split():
            test_line = " ".join(current_line + [word])
            if len(test_line) * 10 <= 408:  # ~10px/char for 20pt
                current_line.append(word)
            else:
                draw.text((20, y), " ".join(current_line),
                          fill='black', font=font)
                y += 24
                current_line = [word]
        if current_line:
            draw.text((20, y), " ".join(current_line),
                      fill='black', font=font)
        images.append(img)
    return {'images': images, 'instruction': text_sample['instruction'],
            'answer': text_sample['answer']}

# Step 2: 混合训练（与视频数据共用同一 ViT encoder）
def sparrow_training_step(sample, image_llm):
    # 视觉编码（video frames 和 synthetic images 共用同一编码路径）
    visual_tokens = ViT(sample['images/frames'])  # [K, H*W, C]
    visual_emb = Projector(visual_tokens)          # MLP 投影
    text_emb = LLM.embed_tokens(tokenize(sample['instruction']))
    input_seq = concat([visual_emb, text_emb])
    logits = LLM(input_seq)
    loss = CE(logits[answer_pos], answer_tokens)   # 仅 answer token 计算 loss
    return loss

# 数据组织
# 文本来源: LongAlpaca (5K) + LongQLora (5K) = 10K synthetic
# 视频来源: ShareGemini (10K) + Video-ChatGPT (10K) = 20K video
# 混合比例: video:synthetic = 2:1
```

关键张量维度: InternVL-4B 每帧 256 visual tokens（单 tile 模式，关闭 patchify），max 64 frames；MiniCPM-8B 每帧 96 visual tokens，max 24 frames。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源实现：https://github.com/VITA-MLLM/Sparrow (Python 3.9 + PyTorch + Flash-Attention 2)。合成数据集：https://huggingface.co/datasets/xjtupanda/Sparrow-Synthetic。

使用方法：
1. 准备文本数据（LongAlpaca, LongQLora 或任意含 long_context + QA 的数据集）
2. 运行合成脚本：NLTK 分割 → PIL 渲染 → 输出 (images[], QA) 对
3. 按 1:2 比例与真实视频数据混合
4. 标准 MLLM fine-tuning 协议训练（与 baseline 完全相同）

关键发现与约束：
- **纯文本混合失败**：直接用原始文本（不转为图像）混入训练导致 Video-MME 仅 55.8（vs Sparrow 56.7），Long 视频从 48.1 降至 47.7。Text-to-image 转换通过统一的 ViT 编码路径消除了 training-inference modality gap。
- **纯合成数据不可行**：TOPA/T3 的纯文本合成方案极易饱和甚至降级，合成数据只能作为正则化补充而非替代真实视频。
- **稠密采样帧无助于长视频**：48 帧 vs 24 帧训练无增益（短视频视觉冗余高），长上下文扩展需从 LLM backbone 层面解决（continue pretraining 扩展 context window）。

涉及论文标题：
- T2Vid__Translating_Long_Text_into_Multi-Image_is_the_Catalyst_for_Video-LLMs
