## Active Evidence Seeking（主动证据搜寻）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Active Evidence Seeking 是 VideoSeek 提出的视频探索策略，与密集贪婪解析（dense greedy parsing）对立。核心思想：agent 不应预先处理所有帧（exhaustive preprocessing），而应在推理过程中，基于当前已累积的 observation 推断下一步最 informative 的探索方向，仅按需检查少量帧。动机来自论文观察：LVBench 中超过 80% 的问题仅需检查不到 5% 的视频帧即可回答，因此 exhaustive parsing（如 DVD 的 8,074 帧）极其低效。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

各 agent 的证据收集策略对比：

```
// Greedy Parsing（DVD, MR. Video）
frames = sample(V, fps=2)
descriptions = describe_all(frames)    // 无论 query
answer = LLM(descriptions, query)

// Active Evidence Seeking（VideoSeek）
while insufficient(τ, query):
    region = predict_region(τ, query)  // 基于 trajectory + 逻辑流
    if need_global: obs = overview(V)
    elif region_too_long: obs = skim(V, region, query)
    else: obs = focus(V, region, query)
    τ.append(obs)
```

关键差异：(1) Query-aware——工具选择和目标区间由 query 和 trajectory 决定；(2) Incremental——每步观察后重新评估充足性；(3) Logic-guided——利用时间顺序和因果缩小搜索空间。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现需要：(1) System prompt 明确定义评估 evidence 充足性的 thinking policy（"Before answering, list supporting evidence + timestamps... If insufficient, collect more"）；(2) 多粒度工具支持（overview/skim/focus）；(3) 完整 trajectory context（不压缩）。局限性：对需要检测意外事件（anomaly detection）的任务效果可能不佳，因关键 evidence 无法通过逻辑流预测位置。

涉及论文标题：
- VideoSeek__Long-Horizon_Video_Agent_with_Tool-Guided_Seeking
