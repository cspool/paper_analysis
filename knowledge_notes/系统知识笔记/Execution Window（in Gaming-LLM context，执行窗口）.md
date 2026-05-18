## Execution Window（in Gaming-LLM context，执行窗口）

术语是什么？通过联网搜索让回答具体和精准。

Execution Window 在 LEGO 的 Gaming-LLM 共置场景中指 LLM 完成一次动作生成（一个 action）所允许的最大时间窗口。该窗口由 APM（Actions Per Minute）目标定义：Execution Window(ms) = 60000 / APM。例如 100 APM → 600ms window、200 APM → 300ms window、300 APM → 200ms window。在 window 内，LLM 必须完成 prompt prefill + token generation 全过程，并需要利用 window 内多个游戏渲染帧的 rendering headroom 来执行推理。LEGO 将 execution window 作为调度和 headroom prediction 的基本时间单位。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

Execution Window 在 LEGO 调度中的三个关键作用：

**1. Headroom Prediction 的时间单位：**
- 不使用 per-frame 或 per-rendering-task 作为预测粒度（误差 >3%）
- 改用 execution window 作为预测单位：过去 3 个 windows 的总 headroom → 预测下 1 个 window 的总 headroom
- Window 的帧数取决于 APM：100 APM = 36 帧 (600/16.6)，200 APM = 18 帧，300 APM = 12 帧
- Window 跨度大 → 单帧波动影响被平滑 → LR 预测误差降至 0.6%-1.3%

**2. Layer-skipping 决策的约束边界：**
- 预测的 window 总 headroom = LLM 推理的资源预算
- 如果预算不足以运行完整模型 → 需跳过若干层
- 跳层数 = 使推理计算量 ≤ 资源预算的最少跳层数

**3. APM SLO 的直接体现：**
- 99th-percentile APM 是主要实时指标
- 如果所有 LLM action 都在其 execution window 内完成 → APM 目标满足
- 论文实验：LEGO 在所有 18 个 Game-LLM-APM 场景中同时满足 FPS 和 APM

**与 Rendering Frame 的关系：**
```
100 APM: Execution Window = 600ms = 36 frames
  LLM action 每 600ms 到达一次
  需要在 36 个 rendering frames 的 headroom 中完成推理

200 APM: Execution Window = 300ms = 18 frames
  LLM action 每 300ms 到达一次

300 APM: Execution Window = 200ms = 12 frames
  LLM action 每 200ms 到达一次
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现要点：
- Window 大小由游戏开发者根据目标难度等级设定（如 easy=100 APM, hard=200 APM, professional=300 APM）
- Window 内 LLM subtask 调度不要求均匀分布，只需在 window 结束前完成所有 token 生成
- Variable-length prompt 场景：input length 在 [256, 1024] 均匀采样 → 执行时间增长 → LEGO 通过 duration predictor + 动态调整跳层策略适配
- Multi-frame workload spike：由于 window 跨 12-36 帧，单帧 spike 不影响整体

涉及论文标题：
- LEGO: Supporting LLM-enhanced Games with One Gaming GPU
