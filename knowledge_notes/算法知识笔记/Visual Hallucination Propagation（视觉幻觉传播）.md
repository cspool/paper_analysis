## Visual Hallucination Propagation（视觉幻觉传播）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Visual Hallucination Propagation 是多模态 CoT 推理中的一种级联失败现象：LVLM 在长链推理的某个中间步骤生成一个与图像视觉证据不一致的 token（幻觉），后续所有推理步骤——即使逻辑形式正确——都基于这个错误的中间结论，最终导致错误答案。这是"thinking more"与"seeing less"的矛盾表现：随着推理链增长，文本上下文逐渐主导 attention，视觉 token 被稀释（attention 分析表明长链中视觉 token 的注意力权重显著下降），语言先验覆盖了细粒度视觉线索，模型在关键视觉判别步产生幻觉。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
幻觉传播示例（以 TreeBench 问题为例）：
```
Step i:  候选 {"blue","red"}，模型自信度中等
         base 分布 p("red")=0.52, p("blue")=0.48
         Greedy 选 "red"（错误！实际上是 blue dress）
Step i+1: 基于 "red" 定位红色衣物，描述 red garment
Step i+2: 基于错误定位判断颜色 → 答案错误
```

关键链路：
```
单步幻觉 token → 后续 token 条件于错误 prefix → 
attention 和 logits 全部基于错误前提 → 
级联放大 → 最终答案错误
```
RH-Bench 论文（Liu et al., 2025）将此量化为 RH-AUC 指标：随推理链长度 T 增加，Perception 准确率 H_T 下降，与 Reasoning 准确率 R_T 形成 trade-off。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
应对策略分为两类：(1) RL-based 训练（DeepEyes、Pixel-Reasoner）——训练模型学会在推理中调用 visual tools（zoom/crop），但需要策划数据、设计 reward、消耗大量计算、且与特定 backbone 耦合；(2) Training-free 解码干预（ECRD、VDGD）——在推理时注入视觉证据监督 token 选择，不修改模型权重，可跨 backbone 泛化。ECRD 通过 uncertainty 检测（k*>1 且 margin≤δ）在关键步触发 visual decider 注入微观察，将级联失败打断在第一步。

涉及论文标题：
- See It, Say It, Sorted: An Iterative Training-Free Framework for Visually-Grounded Multimodal Reasoning in LVLMs
