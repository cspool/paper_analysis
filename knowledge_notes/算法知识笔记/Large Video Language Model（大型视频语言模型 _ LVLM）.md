## Large Video Language Model（大型视频语言模型 / LVLM）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Large Video Language Model (LVLM) 是多模态大语言模型 (MLLM) 在视频理解领域的扩展。LVLM 接收视频帧序列和文本作为输入，通过视觉编码器（ViT）将每帧编码为 visual tokens → Projector/MLP 将 visual tokens 映射到 LLM 的嵌入空间 → LLM decoder 进行自回归生成。部分 LVLM（如 VideoChat, Video-LLaMA）使用 Q-Former 模块（来自 BLIP-2）将 visual 和 textual features 对齐，其他（如 MiniGPT4-Video, Video-LLaVA）直接将 frame features 拼接后输入 LLM。训练通常使用 video-instruction tuning：在视频-文本配对数据上进行 SFT（Supervised Fine-Tuning），使 LLM 学会基于视频内容回答问题和推理。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
LVLM 的标准推理 pipeline：
```
# LVLM Inference Pipeline
frames = sample_video(video, fps=1.0, max_frames=F)
visual_tokens = []
for frame in frames:
    patches = ViT.encode(frame)        # ViT: [H, W, 3] -> [N_patches, D_vit]
    projected = Projector(patches)     # MLP: [N_patches, D_vit] -> [N_patches, D_llm]
    visual_tokens.append(projected)

# Interleave visual and text tokens
input_seq = [visual_tokens, text_prompt]
# 30-min video at 1fps = 1800 frames -> 200K+ tokens

output = LLM.generate(input_seq)
```
长视频的挑战：30 分钟视频可超过 200K tokens，超出多数 LVLM 的上下文窗口（如 Qwen2.5-VL-7B 上下文约 128K）。现有解决方案包括：(a) Sparse frame sampling——无论视频多长只采样固定数量帧，但丢失细粒度时序信息；(b) Token compression/pooling——如 LongVU 的时空自适应压缩、LLaMA-VID 的 2-token-per-image 表示；(c) Memory aggregation——如 MA-LMM 的记忆增强；(d) RAG-based——如 Vgent，将长视频索引为可检索知识库。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Vgent 评估的 LVLM 涵盖 2B-7B 参数范围：InternVL2.5-2B、Qwen2.5-VL-3B/7B、Qwen2-VL-2B/7B、LongVU-7B、LLaVA-Video-7B。所有模型均开源（HuggingFace），Vgent 作为 training-free pipeline 包裹任意 LVLM，不修改模型权重。在 MLVU 基准上，Vgent 在 7 种 LVLM 上一致带来 3.0%-5.4% 的绝对准确率提升。关键发现：Qwen2.5-VL-3B + Vgent 达到 70.4% MLVU，超越其 7B base model (68.8%)——说明小模型配合良好 RAG 可以匹敌甚至超越大模型的直接推理能力。LVLM 的性能上限仍是 Vgent 的瓶颈——论文在 Limitations 中指出 pipeline 性能受限于 base LVLM 的能力。

涉及论文标题：
- Vgent__Graph-based_Retrieval-Reasoning-Augmented_Generation_For_Long_Video_Understanding
