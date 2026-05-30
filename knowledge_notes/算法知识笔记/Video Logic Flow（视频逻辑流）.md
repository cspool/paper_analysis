## Video Logic Flow（视频逻辑流）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Video Logic Flow 是 VideoSeek 论文提出的核心概念，指视频中固有的时间顺序（temporal order）和因果结构（causal structure），包括场景转换、事件序列、叙事线索等。与逐帧密集解析依赖纯视觉信号不同，Video Logic Flow 利用视频的故事线逻辑来推断答案关键证据可能出现在哪里。例如，关于"记者离开部落时乘坐什么交通工具"的问题，逻辑流暗示出发事件应出现在视频接近结尾的部分而非开头。这一概念将视频本身的逻辑结构视为免费导航图，使 agent 可以按逻辑引导 exploration 而非盲目扫描。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Video Logic Flow 在 VideoSeek 算法 pipeline 中体现为三方面：

1. **工具选择层面**：overview 建立全局 storyline 后，agent 根据逻辑流推断答案可能出现在视频的哪个区间来指定 skim/focus 的搜索范围。例如 overview 输出显示 "1480s 两人在 mall 入口交谈"，agent 推断后续镜头转向高楼应在 1480s 之后，故指定 skim(1465s–1510s)。

2. **轨迹推理层面**：每轮 thought 显式使用 temporal order 和 causality 推理。系统 prompt 要求："Prefer internal video logic (temporal order/causality) over visual-only cues; use it to target relevant segments when frames are uninformative."

3. **实证证据**：字幕（subtitles）作为显式 textual storyline 揭示逻辑流。LVBench 加字幕后，VideoSeek 从 68.4%→76.7% 且帧数从 92.3→27.2，因为有了逻辑流后 agent 可极精确定位关键区间。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现 Video Logic Flow 不需额外模型或特征提取——它是 prompt engineering 层面的设计。在 system prompt 的 Operational Rules 中明确指示 agent 使用时间/因果逻辑引导探索，并在每轮 thought 中显式推理基于已有 observation 下一步应关注哪个区间及其逻辑原因。局限性：论文承认对非结构化视频（如 anomaly detection）效果可能下降，因关键证据无法通过逻辑流预测。

涉及论文标题：
- VideoSeek__Long-Horizon_Video_Agent_with_Tool-Guided_Seeking
