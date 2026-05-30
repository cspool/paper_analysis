## Cold-Start SFT for Tool-Aided Reasoning

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Cold-Start SFT for Tool-Aided Reasoning 是指在 RL 训练之前，先用包含工具调用示范的监督数据对基础模型进行微调，使其获得基本的工具使用能力。LongVT 发现，若直接对 Qwen2.5-VL-7B 进行 RL 训练（跳过 SFT），模型会崩溃：无法正确定位时间窗口、无法整合工具输出、tool-call 频率趋近于零（Figure 3b）。因此 Cold-Start SFT 是"先教范式，再优化决策"的必要前提。SFT 阶段教会模型三种基本能力：(1) 提出精确的时间窗口；(2) 基于窗口内细粒度帧进行推理；(3) 窗口不理想时自我纠正。

从算法pipeline角度拆解术语。通过联网搜索让回答具体和精准。
Cold-Start SFT 训练设置：
```
# 模型：Qwen2.5-VL-7B-Instruct
# 框架：LMMs-Engine, 32 GPU
# 数据：228.8K non-tool samples + 19.2K tool-augmented samples
# tool-augmented 数据包括：
#   - Gemini 2.5 Flash 蒸馏的 iMCoTT traces (12.8K) for open-ended QA
#   - Qwen2.5-VL-72B 蒸馏的 temporal grounding traces (6.4K)
# 训练技术：stream packing (buffer=51200 tokens), dynamic batching
# 优化器：AdamW, lr=5e-5, cosine schedule, 300 warmup steps, 3000 total steps
```
SFT 训练目标为标准 next-token prediction loss。关键数据特征：(1) 多轮 tool calling traces 根据视频长度自适应生成（长视频 P_multi 更高）；(2) 混合 image/video reasoning 数据保持通用感知能力。SFT 阶段不需要 tool reward——仅靠模仿 tool-augmented traces 就能教会模型工具调用语义。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Cold-Start SFT 的一般适用条件：当目标 RL 行为涉及基础模型完全不具备的能力（如工具调用、特定输出格式、多步交互范式）时，冷启动 SFT 是必要的。实现方式：(1) 收集/生成包含目标行为的示范数据（可通过更大模型蒸馏、人工标注、或半自动 pipeline）；(2) 用标准 SFT loss 训练，通常 1-3K steps；(3) 使用 stream packing 等训练优化以提高 GPU 利用率。LongVT 的消融实验（Table 3）证实：移除 Cold-Start SFT 后 RL-only 模型在所有 benchmark 上的表现均为最低。

涉及论文标题：
- LongVT__Incentivizing__Thinking_with_Long_Videos__via_Native_Tool_Calling
