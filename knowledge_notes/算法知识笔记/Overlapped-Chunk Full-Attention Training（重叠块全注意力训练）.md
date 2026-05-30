## Overlapped-Chunk Full-Attention Training（重叠块全注意力训练）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Overlapped-Chunk Full-Attention Training 是 StreamingVLM 提出的训练策略，解决"train short, test long"的分布偏移问题——训练无法使用超长视频（计算复杂度 O(T²) 和硬件限制），但推理需要在无限流式输入上稳定运行。核心思想：(1) 将长视频切分为 W=24s 的连续 chunk（重叠 O=12s）；(2) 每个 chunk 作为独立训练样本，chunk 内做 full attention（所有 token attend 到同 chunk 内所有 token）；(3) Vision 和 text tokens 在 chunk 内以 1s 间隔交错排列（非传统 VLM 的 vision-then-text 布局）；(4) 仅在 text position 计算 loss。这种设计的巧妙之处在于：chunk 内 overlapped full attention 的 effective attention pattern 天然近似推理时的 "attention sink + 近期 text 窗口 + 近期 vision 窗口" 模式（Figure 4 右侧），使模型不经特殊训练就习得 recency bias，且训练不增加额外的 attention mask 复杂性。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
参数: W=24s, O=12s, T_sink=512, T_window=512

# 训练数据准备
For each training video:
    # Step 1: 切分为 overlapped chunks
    chunks = []
    for start in range(0, video_duration - W, W - O):
        chunk = video[start:start+W]  # 24s segment

    For each chunk_i:
        # Step 2: 以 1s 间隔交错采样 vision/text tokens
        V_chunk, T_chunk = [], []
        for sec in 0..W-1:
            V_chunk.append(vision_encoder(chunk_i[sec]))
            if has_commentary(chunk_i[sec]):
                T_chunk.append(tokenize(commentary[sec]))
            else:
                T_chunk.append(tokenize("..."))  # silence placeholder

        # Step 3: 取前序 previous text 的 sink + window
        prev_text = commentary_before_chunk
        prev_sink = prev_text[:T_sink]
        prev_window = prev_text[-T_window:]

        # Step 4: Full attention within chunk
        input_seq = interleave(V_chunk, T_chunk)
        # 布局: V[0],T[0], V[1],T[1], ..., V[W-1],T[W-1]
        mask = causal_full_attention  # within-chunk full attention
        loss = CE(logits[text_positions], labels)
```

Annotations: 训练不复制推理的 sink+sliding window mask，而是用 overlapped full attention 近似。W=24s, O=12s 保证每个 chunk 至少 2*W words 的 commentary label。无解说秒插入 "..." placeholder，"..." token 的 loss 也被计算（训练模型学会沉默）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
StreamingVLM 的两阶段训练：(1) SFT：Inf-Streams-Train (525K) + Live-WhisperX-526K (526K)，overlapped-chunk full-attention 格式；(2) Annealing：14K 实时解说样本（16-64s clips，GPT-5 筛选实时解说占比 >80%）。总计算量 128 H100-days。此策略不需要极长视频训练数据或 attention mask 修改。关键 insight：训练时不需要显式地让模型"学习" sliding window——只要 chunk 间有重叠、chunk 内 full attention，模型自动学到 recency bias。验证：Table 6 显示 overlapped SFT 策略相比仅用 Live-WhisperX-526K 在 Inf-Streams-Eval 上提升 +31.29（win rate vs GPT-4o mini）。

涉及论文标题：
- StreamingVLM__Real-Time_Understanding_for_Infinite_Video_Streams
