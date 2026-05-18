## Iterative Prefill for Streaming Video LLM（流式视频LLM的迭代预填充）

术语是什么？通过联网搜索让回答具体和精准。

Iterative Prefill是streaming video LLM特有的推理阶段，区别于传统LLM的one-shot prefill和offline video LLM的batch processing。在streaming video LLM中，视频帧实时到达且无法batch（帧按时序到达），每个frame依次经过Vision Tower→MLP Projector→所有LLM decoder layers，每层生成新KV cache并逐层累积。KV cache以O(N²T)复杂度增长（N²为spatial resolution，T为temporal duration），prefill在每个新frame到达时重复执行，成为端到端延迟主要贡献者（80K cache length时占83% end-to-end latency）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Iterative Prefill的pipeline（以VideoLLM-Online + Llama-3 8B为例）：

```
while video_stream_active:
    frame_t = capture_frame()                    // 实时捕获帧
    vision_embed = VisionTower(frame_t)          // SigLIP-ViT-L-384
    projected = MLP_Projector(vision_embed)

    // Iterative Prefill: 逐层处理当前帧
    for layer L = 0 to N-1:
        Q, K_new, V_new = QKV_Gen(projected)
        // 对完整历史KV cache (所有之前帧) self-attention
        attention_out = Attention(Q, [K_history | K_new],
                                    [V_history | V_new])
        K_history.append(K_new); V_history.append(V_new)
        hidden = FFN(attention_out)
        projected = hidden
    // KV cache增长: |K_history| += tokens_per_frame × layers
    // 10 FPS × Llama-3 8B → 数分钟内超过32GB edge GPU capacity

// 用户query到达时:
question_tokens = Tokenize(user_query)
for layer L = 0 to N-1:
    output = DecoderLayer(question_tokens, K_history, V_history)
generation = autoregressive_decode(output)       // Generation Stage
```

与标准LLM prefill的关键区别：(a) Prefill非一次性——每个新视频帧触发完整prefill pass。(b) KV cache增长无界——随视频时长线性增长。(c) Frame间不能batch——帧按时序到达无法批量处理。(d) 后续query依赖历史——用户可能询问早期视频内容，不能简单丢弃旧KV cache。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

VideoLLM-Online (https://github.com/OpenGVLab/VideoLLM-Online) 是streaming video LLM的开源实现，支持实时视频流输入和多轮对话。实现使用asyncio管理帧到达与模型推理的异步流水线。V-Rex在此基础上增加KV cache retrieval pipeline（ReSV + DRE），offload完整KV cache并按需retrieve。使用流程：部署streaming video LLM→配置vision tower和LLM backbone→设置frame sampling rate→启动视频流→每帧触发iterative prefill→KV cache管理策略决定retention/offloading/retrieval。

涉及论文标题：
- V-Rex: Real-Time Streaming Video LLM Acceleration via Dynamic KV Cache Retrieval

---

