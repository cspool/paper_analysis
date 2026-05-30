## Training-free Context Extension for Video-MLLM (训练无关的视频多模态大模型上下文扩展)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Training-free Context Extension 是 Free-MoRef 提出的无需额外训练的长上下文扩展方法。区别于：(1) LLM Context Expansion —— 通过长序列 post-training 扩大 context limit（如 LongVILA），计算负担大；(2) Token Compression —— 推理前压缩 vision tokens（如 FastV、Video-XL），高压缩率导致信息丢失；(3) Streaming Inference —— 多次调用 LLM 复用 KV Cache（如 RETAKE），延迟与上下文长成正比。Free-MoRef 通过 Partition + MoRef Attention + Reference Fusion，不训练参数、不压缩 token，实现 2x-8x 上下文扩展，FLOPs 仅 ~1/N 增长，first token latency 恒定。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
vision_tokens = model.vision_encoder(sample_frames(video, N_frames))
chunks = multi_reference_partition(vision_tokens, M, N)
for layer in range(L_fusion):
    chunks = MoRef_attention(chunks)
global_ref = reference_fusion(chunks, L_fusion)
for layer in range(L_fusion, num_layers):
    global_ref = standard_layer(global_ref)
answer = decode(global_ref)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在 LLaVA-Video-7B (Qwen2-7B) 上实现，无需修改任何模型权重。兼容 FlashAttention。可与 token compression 或 streaming inference 正交叠加。开源: https://github.com/wkfdb/Free-MoRef。在 VideoMME/MLVU/LongVideoBench 上超越需专门训练的长视频模型（LongVILA, Video-XL, RETAKE）。

涉及论文标题：
- Free-MoRef__Instantly_Multiplexing_Context_Perception_Capabilities_of_Video-MLLMs_within_Single_Inference
