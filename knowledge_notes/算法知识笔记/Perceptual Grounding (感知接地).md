## Perceptual Grounding (感知接地)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Perceptual Grounding（感知接地）在多模态大语言模型（MLLM）中指模型将推理过程锚定在可验证的视觉和文本感知证据上的能力。不同于隐式感知（模型自由决定关注图像的哪些部分且不对外暴露），显式感知接地要求模型明确输出其感知到的关键视觉元素（物体、属性、空间关系）、文本元素（OCR 文本、数值、约束条件），并将后续推理与这些感知证据绑定。

VTPerception-R1 通过系统实验定义了三种感知接地策略：(1) **Explicit Perception**：将预先标注的感知注释直接附加到输入中，模型利用这些外部提供的感知信息进行推理；(2) **Structured Grounding**：通过 prompt 要求模型在推理前先输出自身的感知分析，但感知能力取决于模型自身；(3) **Implicit Grounding**：仅通过轻量 prompt（如 "carefully observe the image"）隐含地引导模型注意视觉内容，不要求显式输出。实验证明：Explicit Perception 在 7B 和 32B 模型上均带来最大收益；Structured Grounding 在小模型上反而有害（模型自身感知能力不足时产生幻觉性观察）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
VTPerception-R1 中感知接地的实现流程：
```
# ===== 训练阶段：建立感知接地能力 =====
# Stage I SFT: 训练模型将感知显式化
Input: (image x, question q)
Target: "<description> 
         Key objects: △AOB, points A,O,B. 
         Property: ∠AOB = 90° (right angle, diameter subtends). 
         Relevant text: AB=BC=25, AC=30.
         </description>
         <think> Since BC is diameter, ∠BXC = 90° ... </think>
         <answer> XY = 10 </answer>"

# Stage II RL: 奖励感知接地质量
# R_vkey: description 覆盖了多少关键视觉线索
K_v = {"△AOB", "right angle", "diameter BC", "intersection X,Y"}  # 标注的关键视觉线索
D_desc = extract_facts(description)  # 从模型生成的 description 提取事实
cov_v = |K_v ∩ D_desc| / |K_v|  # 视觉线索覆盖率
R_vkey = discretize(cov_v, τ_hi, τ_lo)

# R_tkey: think 覆盖了多少关键文本线索
K_t = {"AB=BC=25", "AC=30", "diameter→right angle property"}
D_think = extract_facts(think)
cov_t = |K_t ∩ D_think| / |K_t|
R_tkey = discretize(cov_t, τ_hi, τ_lo)

# R_cons: 推理是否忠于感知证据
F_ans = extract_entities(think) ∪ extract_entities(answer)
E = extract_entities(description) ∪ extract_entities(question)
if has_conflict(F_ans, E):  # 推理引用了感知中不存在的实体
    R_cons = 0
else:
    R_cons = |F_ans ∩ E| / max(1, |F_ans|)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
感知接地通过两个阶段建立：(1) SFT 阶段的感知增强训练——使用经过自动化清洗（VLM dense caption → Grounding DINO 目标检测 → EasyOCR → 结构化描述 → LLM 重建 CoT → 多维度质量评分）的 12K 样本训练模型输出 `<description>` 字段；(2) RL 阶段的感知感知奖励——通过教师模型集成（72B 级模型）生成多样化推理路径，预算验证（top-B by log-probability → correctness + coherence filtering）筛选高质量轨迹，最后从轨迹中提取视觉和文本关键信息作为奖励计算依据。推理时，模型的 `<description>` 字段可以被外部检查者审计，验证感知是否正确。

涉及论文标题：
- VTPerception-R1__Enhancing_Multimodal_Reasoning_via_Explicit_Visual_and_Textual_Perceptual_Grounding
