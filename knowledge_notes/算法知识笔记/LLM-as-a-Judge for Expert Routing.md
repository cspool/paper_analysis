## LLM-as-a-Judge for Expert Routing

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
LLM-as-a-Judge 是一种利用大语言模型的推理能力进行决策/评估而非直接生成内容的使用范式。在 LEGO 中，LLM 不作为预测器（不直接输出动态系统的未来状态），而是作为"裁判"（routing function）：接收系统环境的三层文本化描述和 K 个 GNN expert 的候选预测，经过逐步推理（观察初始条件 → 分析各 expert 预测的物理合理性 → 选择最一致的结果）选出最合适的 expert。这一设计与 LLM-as-Predictor（直接生成预测）形成对比：LEGO 实验（Table 5）显示 LLM Forecasting 的 MSE 为 6.42 而 LEGO 为 0.0072（~890× 差距），且 LLM Forecasting 推理时间更长（1.27s vs 0.44s per sample）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
// LLM-as-Predictor (baseline, 不可靠)
prompt = "Predict future positions of 5 balls in 3D space given initial state X⁽⁰⁾..."
output = LLM.generate(prompt)  // 可能输出错误格式/幻觉/不合理数值

// LLM-as-a-Judge (LEGO 方法)
prompt = """
  System: 5 balls connected by springs with k=1.0.
  Object: Ball 0 pos=(0.1,0.2,0.3) vel=(0.01,-0.02,0.01)
  ...
  Edge: ball 2 connects ball 0, ball 1, ball 3.
  
  Expert A prediction: (positions at t=10) ...
  Expert B prediction: ...
  
  Question: Which expert's prediction is most physically plausible?
"""
decision = LLM.reason(prompt)  // step-by-step 推理 → "Expert B because..."
```

LLM Judge 的推理过程（Case Study, Figure 5）：
1. 分析初始条件：各物体的位置、速度、受力方向
2. 检查物理一致性："Are the objects moving in the expected directions?"
3. 评估预测范围："Are the predictions within a reasonable range?"
4. 综合判断：选择与物理规律最一致的 expert 预测

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现方式：(a) LLM 通过 API（如 OpenAI API 或本地 Llama 3.1 推理）调用；(b) prompt 包含环境描述 + 各 expert 预测（数值以 digit token 编码）；(c) LLM 输出选择结果被解析为 one-hot index，经 label smoothing 转为 soft weight
- LLM Judge 的优势：(a) zero-shot 能力——无需在特定环境上微调 LLM；(b) 常识推理——可利用预训练中的物理世界知识；(c) 可解释性——LLM 可输出逐步推理过程（Case Study）
- LLM Judge 的局限：(a) 推理成本（交替优化降低调用频率）；(b) 复杂科学场景的理解深度有限；(c) 大规模 expert 选择退化（K>15 时性能下降）；(d) 对 LLM temperature 敏感（低 temperature 更稳定）
- 其他 LLM-as-a-Judge 应用：代码评审、文本质量评估、RLHF 中的 reward model、多 agent 辩论等
- 与 learnable gate 的对比：learnable gate（如 MoE Transformer）仅依赖输入数据分布 → LLM Judge 利用外部语义知识（世界模型）理解环境变化，泛化能力更强

涉及论文标题：
- Marrying LLMs with Dynamic Forecasting A Graph Mixture-of-expert Perspective

---
