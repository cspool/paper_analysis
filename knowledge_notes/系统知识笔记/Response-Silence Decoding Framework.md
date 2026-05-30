## Response-Silence Decoding Framework

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Response-Silence Decoding Framework（响应-沉默解码框架）是 LiveStar 中 SVeD 从系统架构角度定义的概念。在线视频理解中，模型不需要对每一帧都输出文本响应——大多数帧为"沉默帧"（内容未发生显著变化）。Response-Silence 框架将推理过程分为两个阶段：(1) **响应阶段 (Response)**：当 visual content 发生显著语义变化时，触发 decoding gate 生成更新字幕；(2) **沉默阶段 (Silence)**：当 visual content 与当前字幕一致时，仅执行轻量 verification（单次 forward pass 计算 PPL），不进行 token generation。这与 EOS-based 方法（VideoLLM-online 等每帧都需要完整 decoding 输出 EOS token 表示沉默）形成对比——SVeD 的沉默阶段比 EOS-based 的沉默阶段快得多（forward pass vs full decoding），且不消耗 vocabulary token。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
在流式视频 Serving 系统中的运转：

```
            ┌──────────────────────────────────┐
            │     Video Frame Stream            │
            │  Frm^1 → Frm^2 → ... → Frm^T     │
            └──────────────┬───────────────────┘
                           │ 逐帧进入
                           ▼
            ┌──────────────────────────────────┐
            │   Vision Encoder (InternViT)      │
            │   Frame → 16 Visual Tokens        │
            └──────────────┬───────────────────┘
                           │
                           ▼
            ┌──────────────────────────────────┐
            │   SVeD Verification (Silence)      │
            │   Single Forward Pass              │
            │   Compute PPL(Dec | Ctx, Frm)      │
            └──────────────┬───────────────────┘
                           │
                    PPL > α·PPL_ref?
                    /              \
                  YES                NO
                  /                    \
                 ▼                      ▼
    ┌─────────────────┐    ┌─────────────────┐
    │ Response Phase   │    │ Silence Phase    │
    │ Full Decoding    │    │ Swap Ctx          │
    │ Generate [Dec]   │    │ No Token Output   │
    │ Update KV Cache  │    │ KV Cache Maintain │
    └────────┬────────┘    └────────┬────────┘
             │                      │
             └──────────┬───────────┘
                        ▼
            ┌──────────────────────────────────┐
            │   Memory Management               │
            │   Peak-End Pruning (every W帧)    │
            │   Dual-Level KV Cache             │
            └──────────────────────────────────┘
```

在 1 分钟 @3fps 视频含 5 个语义片段场景中，180 帧输入 → 5 次 Response（完整 decoding）+ 175 次 Silence（verification only）。每个 Silence pass 约 1ms，每个 Response pass 约 50ms（取决于生成 token 数），总耗时 ≈ 175×1 + 5×50 = 425ms。对比 EOS-based（每帧 decoding，约 10ms/frame）→ 180×10 = 1800ms。加速约 4.2×。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现时需要：(1) 维护 Ctx（多模态上下文）和 Dec（当前活跃字幕）两个状态变量；(2) 在 Silence 阶段执行 swap_last_two(Ctx) 将 Dec 从响应位置移到上下文末尾，使下一帧的 verification 能正确关联 Dec 与新帧；(3) Decoding gate 的超参数 α 需要 per-task 调优（论文在 OmniStar-RNG 上选 1.03）；(4) 耦合 Streaming KV Cache 实现 silence pass 的极低延迟（仅需计算新帧 K/V 并 concat 到 cache，无需重算历史）。框架设计允许用户通过 Gradio web demo 实时调整 α 以适应不同应用场景（如实时解说 vs 监控分析）。

涉及论文标题：
- LiveStar__Live_Streaming_Assistant_for_Real-World_Online_Video_Understanding
