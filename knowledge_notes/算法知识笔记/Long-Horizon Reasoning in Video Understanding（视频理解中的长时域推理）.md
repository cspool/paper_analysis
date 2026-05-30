## Long-Horizon Reasoning in Video Understanding（视频理解中的长时域推理）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Long-Horizon Reasoning 指需要跨越较长推理步骤序列（multi-step reasoning trajectory）才能完成的复杂任务推理。在 video understanding 语境中，长时域推理要求模型：(1) 多次从视频中收集证据（而非单 pass 输入固定帧集）；(2) 每次新 observation 后重新评估已有 evidence 是否充足；(3) 基于累积的完整 trajectory（而非单一的 intermediate summary）进行推理。VideoSeek 将视频问答形式化为概率模型 p(τ, Y | X, Q) = p(τ | X, Q) × p(Y | X, Q, τ)，即先通过长时域探索构建 trajectory τ，再基于 τ 生成最终答案。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

VideoSeek 的长时域推理 pipeline（LVBench uid:3105）：

```
初始 τ₀: [I, Q="镜头转向高楼后写了什么?"]

Turn 1: overview() → τ₁ 含全局 storyline
Turn 2: skim(1465-1510) → τ₂ 确认"1465-1497s: B1标识旁交谈; 1503s: 转向高楼"
Turn 3: focus(1499-1507) → τ₃ 确认"新年祝福语"但不够精确
Turn 4: focus(1502-1510) → τ₄ 精确读取"祝全市人民新春快乐"
Turn 5: answer → "D"
```

关键特性：(1) 完整 trajectory 作为 context（不 truncate 旧 observation），使 agent 可回溯之前发现；(2) 推理与探索交织——每步推理基于完整历史。论文分析指出：intermediate reasoning 贡献了 4.5 pp 增益（vs 将相同帧直接用于单次推理）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现要点：(1) 维护完整对话 history 作为 trajectory，不做中间 summary；(2) 设置最大轮次限制（N=20）；(3) 若 N 轮内未触发 answer，使用 direct-answer 强制指令基于已有 evidence 回答；(4) trajectory 文本 token 数随轮次线性增长——LVBench 无字幕 49K tokens（4.42 turns）。与预建数据库方法的关键差异：tracking 状态存在于 trajectory text 本身而非预计算索引，因此状态可随新 evidence 动态调整。

涉及论文标题：
- VideoSeek__Long-Horizon_Video_Agent_with_Tool-Guided_Seeking
