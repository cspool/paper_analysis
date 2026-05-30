## Pseudo-streaming（伪流式评估）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Pseudo-streaming（伪流式）是现有流式视频理解基准中常见但存在缺陷的评估设置：在评估时将视频在 query 时间戳处截断（仅使用 query timestamp 之前的帧），但仍以离线批量方式处理——模型可一次性加载所有截断帧到 GPU、一次性编码、完整访问上下文。这与真实流式场景的根本区别在于：(1) 模型在实际推理时不需要逐帧增量编码，(2) 不需要维护跨帧 memory state，(3) 不存在帧积压和编码吞吐瓶颈。因此 pseudo-streaming 评估的 accuracy 无法反映真实流式部署中的实际表现。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Pseudo-streaming Pipeline
video = load_full_video(path)          # 完整加载
t_query = query_timestamp               # query 时间戳
frames_before = video[0:t_query]        # 截断到 query 时刻
visual_tokens = vision_encoder(frames_before)  # 一次性批量编码
input_ids = concat(visual_tokens, text_tokens)
answer = model.generate(input_ids)      # 标准离线推理

# True Streaming Pipeline (StreamingEval)
for frame in stream(fps=1):            # 逐帧增量到达
    z_i = vision_encoder(frame)         # 逐帧编码
    memory_bank.update(z_i)             # 增量更新 memory
    if memory_bank.full:                # FIFO 淘汰
        memory_bank.evict_oldest()
# query 到达时仅能访问 memory_bank snapshot（不含未来帧）
answer = model.generate(memory_bank.snapshot(), query)
```

Pseudo-streaming 的核心缺陷：(a) 模型可 "提前看到" 所有截断帧，获得全局 context；(b) 不存在逐帧编码的 timing 约束；(c) 无法暴露 visual encoding 吞吐瓶颈。StreamingEval 的严格因果约束（Frame Player → Encoder-Memory Updater → Responder 三进程异步 pipeline）和 MaxFPS/TTFT 等指标正是为了修正这些缺陷。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
早期流式基准（VStream-QA、StreamingBench 的部分设置、部分 OVO-Bench 评估）采用 pseudo-streaming。StreamingEval 明确区分了 pseudo-streaming 和真实流式评估：前者只需修改数据加载方式（截断帧子集），后者需实现完整的三进程异步 pipeline 并测量系统级指标。在 StreamingEval 的统一协议下，所有模型都在严格 streaming 条件下评估，使得 pseudo-streaming 的优势（如离线模型的高 accuracy）在真实约束下体现出 latency/resource trade-off。

涉及论文标题：
- StreamingEval__A_Unified_Evaluation_Framework_for_Streaming_Video_Understanding
