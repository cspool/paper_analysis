## Video Agentic Model（视频代理模型）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Video Agentic Model 是一类将视频理解任务形式化为 agent 任务的模型范式：不通过单次前向传播直接输出答案，而是通过多次迭代的 think-act-observe 循环，逐步收集证据、推理并生成最终答案。与传统的 standalone Video-LMM（如 GPT-4o、Gemini 1.5 Pro，固定帧数输入→单次推理→输出答案）不同，video agent 维护一个动态的 trajectory，每步根据此前累积的所有 observation 决定下一步工具调用。代表方法包括 VideoAgent (ECCV 2024, CLIP-based frame retrieval)、VideoTree (CVPR 2025, tree-structured search)、DrVideo (CVPR 2025, document-based retrieval)、DVD (NeurIPS 2025, multi-granular database)、VCA (ICCV 2025, curiosity-driven) 和 VideoSeek (CVPR 2026, logic-flow-guided seeking)。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

两类 video agent 的算法 pipeline 对比：

**预建数据库型**（DrVideo, DVD, MR. Video）：
```
// 离线阶段（计算量正比于视频长度）
V_desc ← dense_parse(V, fps=0.2~2)  // 转为文本描述
DB ← index(V_desc)                    // 可检索索引

// 在线推理
relevant ← retrieve(DB, query)
answer ← LLM(relevant, query)
```

**主动探索型**（VideoSeek）：
```
// 无离线阶段，在线推理
τ ← [I, Q]
for t = 1 to N:
    (thought, action) ← LLM(τ)
    if action == answer: return
    observation ← execute(action, V)  // 按需获取
    τ.append(thought, action, observation)
```

关键差异：预建数据库型将感知与推理分离（感知固定在前处理阶段），VideoSeek 将二者交织——每步观察后更新对答案在哪的信念并动态调整探索方向，使 VideoSeek 可以用 1-5% 的帧数达到相当或更好的准确率。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现 video agent 需要：(1) reasoning LLM（VideoSeek 用 GPT-5，也测试了 o4-mini 和 GPT-4.1）；(2) LMM/vision API 解释视觉内容；(3) 视频分析工具集；(4) trajectory manager 维护对话历史。VideoSeek 证明 reasoning model 选择至关重要——GPT-4.1（non-thinking）仅 53.0%（vs GPT-5 68.4%），因过早终止探索；框架是 model-agnostic 的——可替换任意 reasoning backbone。代码开源：github.com/jylins/videoseek。

涉及论文标题：
- VideoSeek__Long-Horizon_Video_Agent_with_Tool-Guided_Seeking
