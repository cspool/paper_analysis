## Train-Short-Test-Long（短训练长测试问题）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Train-Short-Test-Long 是 AR 视频生成模型中普遍存在的训练-推理范式不匹配问题：训练仅在短视频片段（如 5s）上进行（硬件约束 + 长视频数据稀缺），推理时通过滚动 KV cache rollout 生成长视频。模型从未在训练中见过自生长序列上下文，推理时模型自发误差通过自循环反馈累积，导致内容漂移。LongLive 提出 train-long-test-long 解决方案：通过 streaming long tuning 在训练中模拟推理 rollout，使模型暴露于自生长序列和退化上下文。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Train-Short-Test-Long: training only 5s clips, inference 60s
# → error accumulation because model never trained on self-generated history

# Train-Long-Test-Long (LongLive): training simulates inference rollout
# → model conditioned on imperfect self-generated history with DMD supervision
# → streaming long tuning = train as test
```

Annotations: 显存对比：Naive long tuning → OOM; Streaming long tuning (detach history) → 显存恒定。LongLive VBench-Long 30s: 83.52 vs Self-Forcing train-short-test-long: 81.59。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Train-long-test-long 实现：(a) LoRA (rank=256) 微调而非全模型 finetune；(b) DMD 自监督蒸馏无需 ground truth 长视频；(c) teacher 对每个 5s clip 独立监督；(d) 训练中集成 KV-recache + short window + frame sink（完全对齐推理配置）。适用于任何 AR 生成模型的长序列扩展训练。

涉及论文标题：
- LongLive__Real-time_Interactive_Long_Video_Generation
