## Video Large Multimodal Model (video-LMM) / 视频大型多模态模型

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Video Large Multimodal Model (video-LMM) 是将 LLM 从纯文本域扩展到视频域的模型架构。核心组成：(1) Visual Encoder（如 CLIP-ViT、SigLIP）— 将视频帧编码为视觉特征 token；(2) Multimodal Projector（通常是 MLP 或 cross-attention）— 将视觉特征映射到 LLM 的文本 embedding 空间；(3) LLM Backbone（如 LLaMA、Qwen）— 接收拼接后的视觉+文本 token 序列，自回归生成文本回答。video-LMM 与 image-LMM 的关键区别在于时间维度：需要处理多帧序列并建模帧间时序依赖。代表性开源模型包括 LongVA、LLaVA-Video、Video-LLaVA、NVILA、Apollo、Qwen2-VL 等。典型参数规模在 7B-72B，训练流程通常为两阶段：视觉-语言对齐预训练 → 视频指令微调。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
video-LMM 推理 pipeline（以 TPO 论文中使用的 LongVA-7B 为例）：
```
输入: 视频 V (时长 T 秒), 问题 Q (文本)
输出: 回答 A (文本)

# Step 1: 帧采样
F = uniform_sample(V, num_frames=128)  # 均匀采样 128 帧

# Step 2: 视觉编码 (frozen visual encoder)
for each frame f_i in F:
    v_i = VisualEncoder(f_i)  # CLIP-ViT → [N_patch, d_vis]

# Step 3: 投影到 LLM 空间 (multimodal projector)
h_vis_i = Projector(v_i)  # MLP → [N_proj, d_llm]

# Step 4: Token 拼接
H_vis = concat([h_vis_1, ..., h_vis_128])  # [128*N_proj, d_llm]
H_text = Tokenizer(Q)                       # [L_text, d_llm]
H_input = concat([H_vis, H_text])           # 视觉 token 在前

# Step 5: LLM 自回归生成
for t = 1..max_len:
    logits = LLM(H_input, A[:t-1])
    a_t = argmax / sample(logits[-1])
    if a_t == EOS: break
return A = [a_1, ..., a_t]
```
LongVA 的特点是通过语言到视觉的长上下文迁移技术支持 128 帧输入（通过扩展 LLM 的 RoPE position encoding）。LLaVA-Video 使用 96 帧（Video-MME）或 128 帧（其他 benchmark）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源实现分布在 HuggingFace model hub。主要评估工具：lmms-eval (https://github.com/EvolvingLMMs-Lab/lmms-eval)，支持 Video-MME、LongVideoBench、MLVU 等 benchmark 的统一评测。训练方面：通常使用 DeepSpeed ZeRO 或 FSDP 进行分布式训练；visual encoder 保持冻结以减少显存开销；multimodal projector 和 LLM backbone 进行 full fine-tuning 或 LoRA fine-tuning。TPO 论文中 full fine-tuning（language model + projector），8×A100 80GB，4 小时/模型。

涉及论文标题：
- Temporal Preference Optimization of Large Multimodal Models
