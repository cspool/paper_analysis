## Multi-Granular Video Toolkit（多粒度视频工具集）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Multi-Granular Video Toolkit 是 VideoSeek 设计的三个分层视频分析工具，分别在不同时间粒度上操作，实现从全局到局部的渐进式视频探索：(1) overview — 粗粒度全局扫描（均匀采样 16α 帧，每帧约 50 words 描述），构建视频 storyline 并标识潜在关键区间；(2) skim — 中粒度区间扫描（在候选长区间内均匀采样 4α 帧，每帧约 25 words 描述 + 约 50 words 高亮相关帧），快速确认 query-relevant 内容的位置，约束为区间长度至少 4α 秒；(3) focus — 细粒度密集检查（以 1 FPS 对短片段最多 4α 秒采样，直接回答 query 或返回 "No relevant content found"）。三个工具不是固定的 coarse-to-fine 流程，而是 agent 根据当前 trajectory 动态选择调用。α 为帧预算缩放因子（LVBench α=4，其余 benchmark α=2）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

工具调用流程（LVBench case uid:860）：

```
Turn 1: agent 无先验 → overview()
  → 64 帧描述（α=4）："后期场景显示人物在车辆上离开村庄"

Turn 2: 推断离开在末尾 → skim(2800s-3148s, "Find departure moment")
  → 16 帧："3048.6s: 人物乘坐车辆离开；3098.3s: 标题卡片确认段落结束"

Turn 3: 需确认车辆类型 → focus(3044s-3056s, "Identify vehicle type")
  → 12 帧（1 FPS）："Pickup truck"
```

消融实验：移除 overview 导致最大性能下降（-13.3 pp），移除 skim（-6.0 pp），移除 focus（-4.7 pp），证明三者互补——overview 提供全局定位、skim 快速缩小范围、focus 精确验证细节。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

每个工具对应一个 Python 函数，接受时间区间、query 和视频路径作为输入。帧采样用均匀采样（np.linspace），视觉解释调用 GPT-5 vision API，按工具特定 prompt 生成文本描述（如 overview 要求 JSON 格式 {"frames": [{"timestamp":..., "description":...}]}）。工具约束：每轮仅一个工具；overview 仅用于冷启动/全局问题；skim 区间至少 4α 秒；focus 区间最多 4α 秒。开源代码：github.com/jylins/videoseek。

涉及论文标题：
- VideoSeek__Long-Horizon_Video_Agent_with_Tool-Guided_Seeking
