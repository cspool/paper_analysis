## Two-Stage Sharding for VLMs

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
两阶段Sharding是MM-SP系统解决VLM中视觉模态和文本模态处理异构性的负载均衡策略。问题来源：在text-only LLM的序列并行中，所有token由同一tokenizer产生，可直接均分。但在VLM中，视觉数据首先由视觉编码器（ViT+投影器）处理并将placeholder token（如\\<img\\>）扩展为多个真实token（每帧约256 tokens）。如果简单地将placeholder tokens等同text tokens进行切分，会导致视觉编码阶段GPU负载不均。两阶段Sharding的解决方案：(1) Stage1（视觉编码阶段）——将所有帧均分到SP group内的各GPU，每GPU独立执行视觉编码，负载均衡；(2) Stage2（LLM解码阶段）——将所有视觉特征和文本token汇总后，按token数量在sequence维度均分（含dummy token padding确保均匀可分），实现LLM解码的负载均衡。此重分布仅在训练开始时执行一次，开销极小。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Two-Stage Sharding Pseudocode
def two_stage_sharding(frames, text_tokens, sp_rank, sp_size):
    """
    frames: [N_frames, H, W, C] where N_frames varies per sample
    text_tokens: [T]
    """
    # === Stage 1: Per-Image Balanced Distribution ===
    n_local_frames = N_frames // sp_size
    start = sp_rank * n_local_frames
    local_frames = frames[start : start + n_local_frames]
    # Each GPU gets exactly N_frames/sp_size frames
    # Encoding workload is balanced because each frame → ~256 tokens
    
    vis_features = vision_encoder(local_frames)  # [n_local_frames * 256, d]
    
    # === Stage 2: Global Aggregation + Per-Token Balanced Sharding ===
    # All-gather vision features
    all_vis_features = all_gather(vis_features, dim=0)  # [N_frames*256, d]
    
    # Concatenate with text tokens
    full_sequence = concat([all_vis_features, text_embedding(text_tokens)], dim=0)
    total_len = full_sequence.shape[0]  # e.g., 65536 + 2000 = 67536
    
    # Balanced sharding by token count
    tokens_per_rank = ceil(total_len / sp_size)
    # Pad with dummy tokens to make evenly divisible
    padded_len = tokens_per_rank * sp_size
    padded_sequence = pad(full_sequence, (0, padded_len - total_len))
    
    start = sp_rank * tokens_per_rank
    local_tokens = padded_sequence[start : start + tokens_per_rank]
    
    # Adjust labels to ignore padded tokens
    labels = adjust_labels_for_padding(labels, padded_len - total_len)
    
    return local_tokens, labels
```
Ablation结果（Table 5）：在8 GPUs上，long captioning任务中两阶段比一阶段快7%（1.12s vs 1.20s/iter），收益在更长captioning任务中更显著。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
两阶段Sharding在实现上需要注意：(1) Stage1的视觉编码器输出特征维度（每帧token数）需提前知道以做均匀分帧（VILA中每帧固定256 tokens）；(2) dummy token padding需在loss计算中mask掉（修改labels为ignore_index）；(3) 重分布通信仅一次（训练开始时），开销<1% total time。适用于所有包含视觉编码器+LLM解码器的VLM架构的序列并行训练场景。

涉及论文标题：
- LongVILA__Scaling_Long-Context_Visual_Language_Models_for_Long_Videos
