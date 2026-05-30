## Description-Think-Answer Pipeline (感知-推理-答案分离流水线)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Description-Think-Answer Pipeline（DTA 流水线）是 VTPerception-R1 提出的多模态推理结构化输出格式，通过三个明确的 XML 标签将模型输出组织为 `<description> → <think> → <answer>` 的线性流水线。与传统的自由格式 CoT（感知和推理混杂在一起）相比，DTA 流水线强制显式分离：(1) 感知阶段（description）：提取任务相关的视觉和文本证据；(2) 推理阶段（think）：基于感知证据进行逻辑推导；(3) 答案阶段（answer）：给出最终答案。这种设计使感知过程可被外部审计（检查是否"看到了正确的信息"），推理可被验证是否忠于感知证据。

DTA 流水线在 Stage I SFT 中通过 token 级交叉熵损失训练建立，在 Stage II RL 中通过 R_fmt（格式合规）和 R_cons（一致性）奖励强化。与 Visionary-R1 的 "caption → reason → answer" 结构类似但更聚焦：description 只包含任务相关的感知证据而非通用描述，且额外引入了 R_cons 确保推理不偏离感知。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# DTA 流水线的训练/推理流程

# === 训练阶段 ===
# 输入: 几何题图像 + "In triangle ABC, AB=BC=25, AC=30..."

# 模型自回归生成:
token_1..k = generate("<description>")
# 输出: "Isosceles triangle ABC with AB=BC=25, base AC=30. 
#        Circle with diameter BC intersects AB at X, AC at Y.
#        Key property: diameter → right angle at BXC and BYC."

token_{k+1}..m = generate("<think>")
# 输出: "Since BC is diameter, ∠BXC = ∠BYC = 90°.
#        CX is altitude from C to AB. 
#        Area = 1/2 * AC * BM = 1/2 * 30 * 20 = 300.
#        Also Area = 1/2 * AB * CX → CX = 24.
#        In right △BXC: BX = √(25²-24²) = 7 → AX = 25-7 = 18.
#        △AXY ~ △ABC → XY/25 = 18/30 → XY = 15."

token_{m+1}..n = generate("<answer>")
# 输出: "15"

# 损失: L_SFT = -Σ_{t=1..n} log π_θ(y_t | image, question, y_<t)

# === 推理阶段 ===
# 外部审计可以独立检查:
# 1. description 是否准确描述了图像中的关键几何元素?
# 2. think 中的推理步骤是否基于 description 中的证据?
# 3. answer 是否正确且一致?
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
DTA 流水线通过两个阶段建立：(1) SFT 阶段：使用结构化数据训练模型生成带标签的输出格式，格式违规（缺少标签、标签重复）在 RL 阶段通过 R_fmt = 0 惩罚；(2) RL 阶段：R_fmt 强制格式合规，R_cons 确保推理内容忠于描述证据。在推理时，用户或外部系统可以解析 `<description>` 标签内容并独立验证感知准确性，这在需要可解释性和可审计性的应用场景（如医疗、自动驾驶安全分析）中尤为重要。代码实现基于 Qwen2.5-VL-7B-Instruct 的全参数微调，训练配置同 Perception-Augmented SFT。

涉及论文标题：
- VTPerception-R1__Enhancing_Multimodal_Reasoning_via_Explicit_Visual_and_Textual_Perceptual_Grounding
