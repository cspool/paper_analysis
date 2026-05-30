## Interleaved Textual Timestamp Encoding（交错文本时间戳编码）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Interleaved Textual Timestamp Encoding 是一种将视频帧的时间信息注入 MLLM 的编码策略。核心方法：将每帧的绝对时间戳（如 "10.2s"）通过 LLM 的 text tokenizer 转换为文本 token，然后交错插入到对应帧的 visual tokens 之前，形成 "timestamp → frame" 的交替序列。这与两种替代方案形成对比：(1) Non-interleaved textual encoding：将所有时间信息放在 prompt 开头一次性声明（如 "This video samples N frames at t1, t2, ... seconds"）；(2) Visual overlay：将时间戳作为 OCR-able 文本直接渲染到帧图像上。TimeLens 的实验证明，interleaved textual encoding + raw timestamps 在所有方案中效果最优。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Interleaved Textual Timestamp Encoding 的 token 序列构建流程：
```
# 输入: T frames, fps, query q
# 每帧作为独立 image 处理（绕过 frame merge），以便插入 text tokens

tokens = [system_prompt_tokens, query_tokens]
for i in range(T):
    t = i / fps  # 绝对秒数
    # 保留一位小数，如 "10.2s"
    timestamp_text = f"{t:.1f}s"
    # 通过 LLM text tokenizer 获取文本 token embeddings
    text_tokens = tokenizer.encode(timestamp_text)
    # Vision encoder 提取 frame visual tokens (frozen)
    # frame 被复制两份绕过 Qwen2.5-VL 的相邻帧 merge
    frame_copy1 = frame
    frame_copy2 = frame
    visual_tokens = vision_encoder(concat(frame_copy1, frame_copy2))
    # 交错: timestamp 在 visual 之前
    tokens.extend(text_tokens)
    tokens.extend(visual_tokens)

# 最终序列: [prompt, query, "0.0s", frame_0, "0.5s", frame_1, ...]
output = LLM(tokens)
```

TimeLens 消融结果（Charades-TimeLens mIoU）：
| Timestamp Encoding | Frame Index | Raw Timestamp |
|---|---|---|
| Position Embed. (MRoPE) | - | 36.6 |
| Visual Overlay | 44.0 | 46.3 |
| Non-Interleaved Textual | - | 45.8 |
| Interleaved Textual | 45.6 | **48.3** |

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Interleaved Textual Timestamp Encoding 的实现需处理两个关键细节：(1) 对于 Qwen2.5-VL 等默认将相邻两帧 merge 的架构，需将每帧复制为两份作为独立 image 处理，以在帧间插入 text token（同时计算量等同 2 FPS 的原始实现）；(2) 时间戳格式选择：raw timestamp（如 "10.2s"）优于 frame index（如 "1, 2, 3"），因为后者忽略了帧间时间间隔（非均匀采样时尤其重要）。该方法无需修改 LLM 的 RoPE 机制，完全依赖 MLLM 已有的文本理解能力来感知时间，实现简单且有效。在推理时，只需在帧输入前拼接时间文本 token 即可，无额外计算开销。

涉及论文标题：
- TimeLens__Rethinking_Video_Temporal_Grounding_with_Multimodal_LLMs
