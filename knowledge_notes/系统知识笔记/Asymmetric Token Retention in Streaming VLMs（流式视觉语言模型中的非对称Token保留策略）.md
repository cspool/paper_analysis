## Asymmetric Token Retention in Streaming VLMs（流式视觉语言模型中的非对称Token保留策略）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Asymmetric Token Retention（非对称 Token 保留）是 StreamingVLM 提出的流式推理 KV Cache 管理策略，用于在无限视频流中以稳定、有界的延迟实时生成响应。核心思想：三种 token 类型遵循不同的驱逐优先级——(1) Attention Sink Tokens（Tsink=512）：system prompt + 早期 text tokens，最高保留优先级，从不驱逐，作为 attention 分布稳定性的锚点；(2) Recent Text Tokens（Twindow=512）：最近的文本 token 长窗口，中等保留优先级，保留长期语言记忆和对话连贯性；(3) Recent Vision Tokens（Vwindow=16s）：最近的视觉 token 短窗口，最低保留优先级，仅跟踪当前连续动作。驱逐顺序：旧 vision tokens 优先驱逐（视觉冗余高、时间衰减快），旧 text tokens 仅在超出总 budget 时驱逐（语义信息密度高、长期重要性大）。与均匀 sliding window 的关键区别：非对称设计反映 vision 和 text 在时间重要性的天然不对称——visual appearance 快速变化而语义上下文需要更长记忆。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
```
Function streaming_inference_step(t, KV_cache, V_new, text_new):
    # Config: Tsink=512, Twindow=512, Vwindow_sec=16

    # 1. 新视觉帧编码 + interleave text
    V_tokens = vision_encoder(V_new)

    # 2. 复用历史 KV cache（不重算）
    # Query 对 KV_cache 中保留的 tokens 做 attention
    # KV_cache = [sink_tokens | text_window_tokens | vision_window_tokens]

    # 3. Asymmetric Eviction（非对称驱逐）
    # 驱逐旧 vision tokens（时间超过 Vwindow_sec 秒前的）
    vision_oldest_time = min_vision_timestamp(KV_cache)
    evict_vision_before(KV_cache, t - Vwindow_sec)

    # 驱逐旧 text tokens（超出总 budget 时）
    total_budget_tokens = Tsink + Twindow + vision_token_count
    if len(KV_cache) > total_budget_tokens:
        surplus = len(KV_cache) - total_budget_tokens
        # 驱逐 sink 区间后的最旧 text tokens
        evict_oldest_non_sink_text(KV_cache, surplus)
```

Annotations: 512 text tokens ≈ 一两个段落的上下文。16s vision window @1fps = 16 frames。总 cache size 由三类 tokens 加和决定。Ablation（Table 5）：纯 sink（V_window=0s）导致 win rate 降至 52.90（vs GPT-4o），纯 window（Tsink=0）降至 66.76。V_window=16s 在所有配置中最优——足够覆盖连续动作又高效。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现基于标准 KV Cache API：(1) 逐层维护 key/value 张量及其对应的 token 类型标记（sink / text / vision）；(2) 驱逐即沿序列维度切片：`K_new = K[:, retain_mask, :]`；(3) 驱逐后同步更新 contiguous RoPE 的 position offset；(4) SFT 训练通过截取 previous text 的 Tsink+Twindow 使模型适配。与纯文本 KV cache eviction（H2O、SnapKV）的区别：引入 vision/text 不对称性；与 LiveStar 的 Streaming KV Cache (Dual-Level) 的区别：LiveStar 分 intra/inter-dialogue 两级缓存，StreamingVLM 分 sink/text/vision 三类且驱逐优先级由 token 类型决定。适用范围：无限 visual+text 流式输入场景（视频解说、自动驾驶感知、机器人实时理解）。

涉及论文标题：
- StreamingVLM__Real-Time_Understanding_for_Infinite_Video_Streams
