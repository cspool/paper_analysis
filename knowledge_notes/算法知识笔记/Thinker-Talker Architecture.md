## Thinker-Talker Architecture

术语是什么？

Thinker-Talker Architecture 是 Qwen-Omni 系列（Qwen2.5-Omni、Qwen3-Omni）和 Ming-Omni 系列采用的**双 AR LLM decoder**架构，专为同时生成 text 和 audio 输出设计。它包含两个 sequential AR LLM stages 加一个 Vocoder：

- **Thinker (LLM)**：较大的 AR LLM（Qwen3-Omni: 30B，Qwen2.5-Omni: 7B），接收 multimodal input 并生成 text tokens + per-step hidden states
- **Talker (LLM)**：较小的 AR LLM，每 decoding step concat Thinker hidden states + Talker input embeddings + original multimodal embeddings，自回归生成 audio codec tokens
- **Vocoder**：将 discrete codec tokens 转换为 continuous audio waveform（Qwen2.5-Omni 使用 DiT Vocoder，Qwen3-Omni 使用 lightweight CNN Vocoder）

从算法pipeline角度拆解术语：

Thinker-Talker 的伪代码计算流程：
```
Algorithm: Thinker-Talker Inference

Input: multimodal_input
Output: text_response, audio_waveform

// Phase 1: Multimodal Encoding
mm_input_emb = Concat(
    TokenEmbed(text_prompt),
    AudioEncoder(audio),
    VisionEncoder(image),
    VisionEncoder(video_frames))

// Phase 2: Thinker AR Decode
hidden_states_list = []
text_tokens = []
for step in 1..max_text:
    logits, hidden = Thinker(mm_input_emb)
    token = Argmax(logits)
    text_tokens.append(token)
    hidden_states_list.append(hidden)
    if token == EOS: break
    mm_input_emb = Embed(token)

// Phase 3: Talker AR Decode (cross-stage dependency)
talker_emb = Embed(BOS_audio)
codec_tokens = []
for step in 1..max_audio:
    // KEY: concatenate Thinker hidden states every step
    talker_input = Concat(hidden_states_list, mm_embeddings, talker_emb)
    codec, _ = Talker(talker_input)
    codec_tokens.append(codec)
    if codec == EOS: break
    talker_emb = Embed(codec)

// Phase 4: Vocoder
waveform = Vocoder(codec_tokens)
Return (text_tokens, waveform)
```

关键计算特征：
- Talker 每 step 需 access 完整 Thinker hidden states（跨 stage 数据依赖，非仅单 token）
- Thinker output text tokens (~150.9 avg) ≪ Talker audio codec tokens (~545.4 avg) → Talker 占总延迟大部分
- Thinker model 更大但 text token 输出少，Talker 较小但 audio token 多 → 两者 compute profile 截然不同
- 两 AR stage 都需要独立的 KV cache management（各自的生成序列不同）

术语一般如何实现？如何使用？

Baseline 实现中，开发者需在 HuggingFace Transformers 上手写 custom generate loop for each stage，手动管理 Thinker→Talker hidden state transfer（无 framework-level batching）。vLLM-Omni 将 Thinker 和 Talker 定义为独立 stages，各由 vLLM engine 服务（含 continuous batching + PagedAttention + chunked prefill），Thinker hidden states 通过 preprocess 函数每 iteration 注入 Talker，cross-stage transfer 由 Unified Connector 处理。这种解耦使 Thinker 获 12.97× TPS speedup、Talker 获 7.98× TPS speedup（vs Transformers baseline）。

涉及论文标题：
- vLLM-Omni: Fully Disaggregated Serving for Any-to-Any Multimodal Models
