## Fast-path/Slow-path Design Pattern in Agent Serving（Agent Serving中的快径/慢径设计模式）

术语是什么？通过联网搜索让回答具体和精准。

Fast-path/Slow-path是AIMS采用的系统设计模式：fast path在本地SLM上执行，当predictor指示下一行为安全时走此路径；slow path回退到云端LLM，用于精度关键的决策。Fast path包含：URC request-level过滤（整请求直接走SLM）和SSE subtask-level过滤（当前subtask预测与LLM路径相似则走SLM）。Slow path包含三个recovery路径：SLE（S-L distance预测匹配）、CD（future收敛点搜索）、SD（subtask分解后重试SSE）。这种设计使AIMS在保持83.58% SLM usage的同时达到77.62% accuracy（macro avg），仅比All-LLM低几个百分点。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

AIMS的fast-path/slow-path路由层级：

```
// Fast-path/Slow-path分层
Level 0 (Fastest): URC判断整请求similarity > τ_req → 全请求SLM，0次LLM call
Level 1 (Fast): SSE判断单subtask similarity > κ → SLM执行该subtask
Level 2 (Slow-recovery 1): SLE预测S-L distance → 若future match则SLM处理d+1个subtask
Level 3 (Slow-recovery 2): CD迭代搜索future convergence point → SLM执行至收敛点
Level 4 (Slow-recovery 3): SD分解 → 全部子subtask通过SSE → SLM整组执行
Level 5 (Slowest): 所有recovery失败 → LLM执行
```

实验消融显示每层贡献：移除SD后accuracy -1.58%/SLM -5.54%；再移除CD后accuracy -3.72%/SLM -8.06%；再移除SLE后accuracy -5.14%/SLM -11.22%；仅URR后accuracy -7.52%/SLM -11.60%。多层recovery路径共同贡献了accuracy保持和SLM最大化。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Fast-path/Slow-path模式在系统设计中通用（类似Viola-Jones cascade detector的early rejector理念），AIMS将其应用于AI agent subtask路由。实现要点：(1) fast path的predictor必须轻量（ModernBERT/Qwen3-0.6B+LoRA，~2GB VRAM total），使决策延迟可忽略；(2) slow path recovery需要有递增的exploration范围（SLE看d步、CD看lookahead步、SD递归回SSE）；(3) 需要offline profiling为所有predictor提供训练数据（subtask binary tree收集SLM/LLM执行trace）。AIMS的fast-path/slow-path设计使scheduler decision overhead仅占总时间3-7%，网络hop latency平均0.58s可忽略。

涉及论文标题：
- AIMS: A Cost-Efficient Framework for LLM-based Agent Deployment in Cloud-Edge Hybrid Environments

---
