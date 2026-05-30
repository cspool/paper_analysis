## Streaming Causal Attention Masks (SCAM)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SCAM (Streaming Causal Attention Masks) 是 LiveStar 论文提出的流式视频-语言对齐训练策略中的核心注意力掩码机制。在标准 causal attention mask（因果注意力掩码，即每个 token 只能看到自己和之前的 token）基础上，SCAM 通过额外的掩码约束实现流式视频的增量式训练：(1) 对当前语义片段（semantic clip）中已生成的字幕 token 施加 -inf 掩码，防止模型通过"抄写"同一片段中已输出的字幕来 trivial copying；(2) 保留前一语义片段的终端字幕 token 的可见性，使模型感知场景语义边界；(3) 掩蔽之前所有非终端字幕 token，避免信息泄露。SCAM 使得模型能够在交错帧-字幕序列（interleaved frame-caption sequences）上自回归训练，逐步学习从可变长度视频前缀生成时间一致的字幕，同时保持预训练的视觉-语言对齐范式。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
SCAM 位于 LiveStar 训练 pipeline 的 attention 计算阶段，在标准 causal mask 之上叠加语义片段级别的结构约束：

```
# 输入: 交错帧-字幕序列
# C_k = 第k个语义片段, 帧 t_m..t_n 共享字幕 [Cap^k]
# 序列 = [Frm^{t_1}, Cap^1_1, ..., Frm^{t_i}, Cap^k_i, ...]

def build_scam_mask(seq_len, semantic_clips, caption_positions):
    # 标准因果mask：M[i,j] = 0 if j <= i else -inf
    mask = causal_mask(seq_len)
    
    for clip_k in semantic_clips:
        for pos in caption_positions_in_clip(clip_k):
            for other_pos in same_clip_earlier_captions(pos):
                # 掩蔽同一clip中已生成的字幕token（防止copying）
                mask[pos, other_pos] = -inf
            for other_clip in prev_clips_before(clip_k):
                for other_pos in non_terminal_captions(other_clip):
                    # 掩蔽之前clips的非终端字幕
                    mask[pos, other_pos] = -inf
            # 终端字幕不掩蔽 → 传递场景边界信息
    
    return mask

# 训练时在 attention 中使用 SCAM
Attention(Q, K, V, mask=scam_mask)
```

训练目标：max P([Cap_i^k] | [Ctx^{<t_i} {Mask^{≤t_i}}], [Frm^{t_i}])

关键设计：Mask 的稀疏模式确保 (a) 当前字幕的生成不被同一clip的已有字幕污染，(b) 可引用前一clip的终端字幕获知场景转换，(c) 所有视觉帧始终可见。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
SCAM 在 HuggingFace Transformers 框架中实现：在模型 forward 时传入自定义 4D attention mask [batch, 1, seq_len, seq_len]，与标准 causal mask 叠加后送入 scaled_dot_product_attention。LiveStar 训练时使用 InternVideo2.5 的 InternViT + InternLM2.5-7B 架构，SCAM mask 在训练 loop 的每个 step 根据语义片段边界动态构建。训练配置：每序列最多 8192 tokens，仅对 assistant response tokens 计算 cross-entropy loss。M 个 paraphrase captions 池中随机采样以防止重复字幕导致的过拟合。

涉及论文标题：
- LiveStar__Live_Streaming_Assistant_for_Real-World_Online_Video_Understanding
