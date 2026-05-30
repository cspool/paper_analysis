## Context Extension for LLMs (via RoPE Scaling + Progressive Training)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Context Extension是扩展LLM上下文窗口的技术。LongVILA在Stage4采用纯文本持续预训练方式扩展LLM上下文：利用SlimPajama数据集共17B tokens，采用渐进式训练调度，逐步将上下文从8K扩展到65K再到262K。同时增大RoPE（Rotary Position Embedding）基频——标准的RoPE基频为10000，通过增加基频值（如扩展到更大的数），RoPE对更长距离的位置编码可以保持区分度。配合LoRA进行参数高效的微调，避免全参数训练的高昂成本。论文实证发现，在进行长视频SFT之前必须先完成上下文扩展，否则模型无法有效利用长上下文信息。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Context Extension via RoPE base frequency scaling
# Standard RoPE: θ_i = base^(-2i/d), base = 10000
# Extended RoPE: θ_i' = (base × scale)^(-2i/d)

def rope_extend(attention_module, scale_factor=8.0):
    """Extend RoPE base frequency for longer context"""
    original_base = attention_module.rope_theta  # e.g., 10000
    attention_module.rope_theta = original_base * scale_factor
    
def progressive_training(model, data_loader, schedule):
    """
    Progressive context extension schedule
    schedule = [(8192, 5B_tokens), (65536, 6B_tokens), (262144, 6B_tokens)]
    """
    for seq_len, num_tokens in schedule:
        model.max_position_embeddings = seq_len
        data_loader.set_sequence_length(seq_len)
        
        tokens_processed = 0
        for batch in data_loader:
            # LoRA forward/backward only
            with lora_enabled(model):
                loss = model(batch.input_ids, attention_mask)
                loss.backward()
                update_lora_params()
            tokens_processed += batch.numel()
            if tokens_processed >= num_tokens:
                break
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
LongVILA使用LongLoRA方法进行上下文扩展：仅训练LoRA参数（adapter层），冻结原始LLM权重，大幅降低显存和计算需求。RoPE基频缩放参考Fu et al. 2024的方法。约需336 H100 GPU hours。适用于需要将预训练LLM从较短上下文（如4K-32K）扩展到长上下文（如128K-256K）的场景，是长视频/长文档VLMs的必要前置步骤。

涉及论文标题：
- LongVILA__Scaling_Long-Context_Visual_Language_Models_for_Long_Videos
