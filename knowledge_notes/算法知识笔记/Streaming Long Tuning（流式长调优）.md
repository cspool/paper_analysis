## Streaming Long Tuning（流式长调优）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Streaming Long Tuning 是 LongLive 提出的用于 AR 视频生成的长序列训练策略。传统 AR 视频模型采用 train-short-test-long 策略（仅在 5s 短视频上训练，推理时通过滚动 KV cache rollout 长视频），导致误差累积。Streaming Long Tuning 实现 train-long-test-long 对齐：每 iteration 基于前一 iteration 存储的 KV cache 滚动生成下一个 5s clip（而非重新采样），仅对当前 clip 计算 DMD loss（teacher=Wan2.1-T2V-14B 监督），detach 历史帧梯度（仅当前 clip 产生梯度）。显存仅由当前 clip 控制 O(clip)，非全长序列 O(full_video)，避免 naive long tuning 的 OOM。DMD loss 在完整 rollout 上提供全局监督。滚动直至预设最大长度（60s 或 240s）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Streaming Long Tuning (LongLive Algorithm 1)
# l_video=60s, l_clip=5s, teacher=Wan2.1-T2V-14B
C = []; l = 0
(p, p_next) = sample_prompt_pair()
s = sample_switch_time(5s, 55s)
while l < l_video:
    p_active = p if l < s else p_next
    if l == s: C = kv_recache(G_theta, x, C, p_active)
    x_clip = generate_next_5s(G_theta, C, p_active)  # rollout
    loss = DMD_loss(x_clip, teacher_14B, p_active)
    loss.backward()  # gradient only through x_clip (history detached)
    optimizer.step(); l += l_clip
```

Annotations: 关键设计：(a) detach 历史帧 → 显存恒定；(b) teacher 对每个 5s clip 独立监督 → 始终在 teacher 能力范围内；(c) streaming rolling 模拟推理 → train-test alignment。64 H100 × 12h ≈ 32 GPU-days for 60s。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Streaming Long Tuning 在 LongLive 中以 Wan2.1-T2V-1.3B + LoRA (rank=256, ~27% params trainable) + AdamW 实现。不引入额外视频数据，teacher-student DMD self-supervision。Qwen2-72B-Instruct 生成 follow-up prompt 构造 switch 训练数据。适用于：(a) AR 生成模型 train-test mismatch；(b) 自监督蒸馏场景；(c) 主流 KV cache rollout 的因果模型。开源在 https://github.com/NVlabs/LongLive。

涉及论文标题：
- LongLive__Real-time_Interactive_Long_Video_Generation
