## Overthinking in Multimodal Reasoning

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Overthinking（过思考）指推理模型对无需复杂推理的输入生成长链式思维，导致：(1) 准确率不增反降（推理链中的单步幻觉或错误推理覆写正确的初始直觉）；(2) 推理成本和延迟显著增加（生成额外数百 tokens）。VideoAuto-R1 在视频域首次系统性揭示了过思考现象：Table 1 显示 Video-R1 的 CoT 推理（386 tokens avg）在 VideoMME 上准确率 64.3% 甚至低于 direct answering 的 64.6%；Time-R1 的 CoT（138 tokens avg）在 VideoMME 上 63.8% vs direct 65.9%（-2.1%）。图 7 提供了过思考导致错误的定性示例：VideoChat-R1 的 CoT 推理链中幻觉了不存在的舞蹈动作描述，将正确的 direct answer D 覆写为错误的 E。过思考的根本原因是视频感知任务主要依赖视觉识别而非符号推理，冗长的语言推理链引入噪声而非价值。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
过思考的表现形式（来自 VideoAuto-R1 的分析）：

```
# Overthinking 导致的"退化"模式
# Case 1: CoT 幻觉覆写正确直接答案（图 7）
Direct Answer: D (correct) 对 dance video 的最后一个动作
CoT Answer: D→E (incorrect) 
  推理链: 描述了不存在的舞步 → 错误推论 → 覆写正确初始判断

# Case 2: CoT 冗余验证但最终答案不变（MVBench 上 ~75% 案例）
Direct Answer: C (correct)
CoT Answer: C (correct)
  推理链: 150 tokens 逐步描述视频和对比选项 → 与 direct answer 相同结论
  → 浪费推理计算但无精度增益
```

量化证据（Table 8）：
- MVBench: confidence 0.948, think ratio 25%, CoT gain +0.1%（几乎无增益）
- MMVU: confidence 0.933, think ratio 39%, CoT gain +0.4%（边际增益）
- VideoMMMU: confidence 0.874, think ratio 51%, CoT gain +4.0%（确实需要推理的任务）

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
缓解过思考的策略：(1) Auto-Thinking（VideoAuto-R1 的核心方案）：仅在置信度低时触发 CoT，避免强行推理；(2) 难度感知数据过滤（VideoAuto-R1 的 filtering pipeline）：丢弃"所有 8 个 response 均正确"的过于简单样本，避免模型学习对琐碎问题展开复杂推理；(3) 训练时使用 Dual-Answer Reward 引导模型产出简洁初始答案。类似现象也在文本和图像域被观察到：Sui et al. (2025) 的 "Stop Overthinking" 综述、Kumar et al. (2025) 的 Overthink 攻击、Chen et al. (2024) 对 o1-like LLMs 过思考的分析。VideoAuto-R1 的启示：视频感知任务中 explicit language-based reasoning 并非普遍必要，Auto-Thinking 是更匹配视频特性的推理范式。

涉及论文标题：
- VideoAuto-R1__Video_Auto_Reasoning_via_Thinking_Once__Answering_Twice
