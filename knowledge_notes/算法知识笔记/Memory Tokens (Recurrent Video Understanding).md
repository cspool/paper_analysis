## Memory Tokens (Recurrent Video Understanding)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Memory Tokens 在 VideoLLaMB 中指在 Recurrent Memory Bridge Layers 中使用的可学习 token 向量（32 个，dim=1024），它们被 prepend 到每个视频语义段的视觉特征前，通过 self-attention 与视觉特征交互，逐步累积全视频信息。Memory tokens 的设计受 RMT (Recurrent Memory Transformer, Bulatov et al. NeurIPS 2022) 启发：RMT 将特殊 memory tokens 添加到 Transformer 输入/输出中以实现 segment-level recurrence，每个 segment 的 memory token 输出作为下一 segment 的输入。VideoLLaMB 将这一思想从纯 NLP 迁移到 video-language 设置，关键创新在于：memory tokens 在 vision-LLM bridge 位置运作（而非 LLM 内部），因此不会干扰 LLM 的语言理解能力；memory tokens 通过 self-attention 同时吸收视觉信息和历史记忆；最终 32 个 memory tokens 代表整个视频的压缩表示，与当前段视觉表示一起送入 LLM。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Memory tokens 在 VideoLLaMB 中的生命周期：

```
# 1. 初始化
m_0 = nn.Parameter(torch.randn(32, 1024))  # 32个可学习token，随机初始化

# 2. 逐步更新 (遍历 K 个语义段)
for i = 1 to K:
    # Prepend memory to segment (32+C tokens total)
    input_seq = torch.cat([m_{i-1}, s_i], dim=0)  # [32+C, 1024]
    
    # Self-attention: memory 与 visual 交互
    attn_output = SelfAttn(Q=input_seq, K=input_seq, V=input_seq)
    # memory tokens attend to: (a) 其他 memory tokens, (b) 段内 visual tokens
    # visual tokens attend to: (a) memory tokens, (b) 段内其他 visual tokens
    
    # 分离 memory 和 visual
    m_i' = attn_output[:32]      # 更新后的 memory (含当前段信息)
    o_i = attn_output[32:]       # 段视觉表示 (含 memory 上下文)
    
    # Retrieval 增强
    m_i = CrossAttn(query=m_i', key=cache, value=cache)

# 3. 最终表示
video_representation = Concatenate(
    m_K,                          # 最终 memory: 全局视频压缩表示 [32, 1024]
    Projector(o_1, ..., o_K)      # 所有段视觉输出投影 [N_proj, LLM_dim]
)
# 送入 LLM: ~32+N_proj tokens (vs 原始 n×256 tokens)

# 内存分析:
# 原始方法: n frames × 256 patches × 1024 dim = n×256 tokens
# VideoLLaMB: 32 memory tokens + K×4 projected visual tokens
# n=320 frames, K=80 → 32 + 320 ≈ 352 tokens (vs 81920 tokens)
```

训练时 memory tokens m_0 作为可学习参数随 Bridge Layer 一起训练。每个 training step 中 16 帧分 4 段，memory tokens 跨 4 步递归更新。推理时可扩展到 320 帧（80 段），memory tokens 在跨段递归中持续累积信息。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
VideoLLaMB 使用 32 个 memory tokens，每个 1024 维（与 ViT-L/14 输出维度一致）。初始化方式：随机初始化（nn.Parameter），随训练学习。参数量：32×1024 ≈ 33K 参数，可忽略。消融实验显示（Table 8）：(1) 使用 memory tokens only（移除 visual output o_i）→ 50.4%（-3.4 vs VideoLLaMB 53.8%），说明当前段视觉信息对 LLM 理解仍然关键；(2) 增加 memory tokens 数从 32→64（单层 Bridge）→ 53.0%（-0.8），可能因训练数据不足导致的过拟合；但 64 tokens + 3 层 Bridge → 54.6%（+0.8），说明更大容量需配合更深 Bridge 层。Memory tokens 的概念与 Perceiver (Jaegle et al. 2021) 的 latent array 和 BLIP-2 (Li et al. 2023) 的 Q-Former queries 有共通之处：都是通过少量可学习 token 压缩大量输入信息。VideoLLaMB 的差异在于 memory tokens 是递归更新的（而非一次性处理所有输入），使其能处理理论上无限长的视频流。

涉及论文标题：
- VideoLLaMB__Long-context_Video_Understanding_with_Recurrent_Memory_Bridges
