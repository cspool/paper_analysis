## Perception-Augmented SFT (感知增强监督微调)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Perception-Augmented SFT（感知增强监督微调）是 VTPerception-R1 Stage I 的训练方法，将原始多模态 CoT 数据转换为结构化格式 `<description>...<think>...<answer>`，通过 token 级交叉熵损失训练模型在推理前先显式提取和表达与任务相关的视觉/文本证据。

核心设计原则：(1) `<description>` 不是通用图像描述，而是仅总结"与问题相关且对推理有用"的视觉/文本证据；(2) `<think>` 保留原始 CoT 推理链，但推理应基于 `<description>` 中的感知证据；(3) `<answer>` 为最终解答。这种 "先看、再想、后答" 的结构显式解耦了感知与推理，使得感知过程可审计、可干预。

数据准备：从 LLaVA-CoT (4K) 和 Vision-SR1 (8K) 采样 ~12K 样本，经过自动化清洗流水线处理——VLM dense caption（GPT-4o 级模型生成密集描述）→ Grounding DINO 目标检测 → EasyOCR 文本提取 → 合并为结构化规范描述 → LLM 基于规范描述重建 CoT → 多维度质量评分（formal_score: 描述准确性, cot_score: 推理逻辑清晰度, answer_score: 答案一致性, 幻觉检测分数）→ 阈值过滤（overall_score ≥ τ 保留）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
SFT 训练流程：
```
# 数据转换: 原始 CoT → 结构化格式
# 输入: (image, question, original_cot, original_answer)
# 输出: "<description> relevant visual/textual facts </description>
#         <think> step-by-step reasoning </think>
#         <answer> final answer </answer>"

# 训练目标: token-level cross-entropy over full target sequence
L_SFT = -Σ_t log π_θ(y_t | x_image, x_question, y_<t)
# 梯度流经所有 token: <description> + <think> + <answer>

# 训练配置:
model = Qwen2.5-VL-7B-Instruct  # 全参数微调
optimizer = AdamW(lr=1e-5, weight_decay=0.1)
batch_size = 1  # per-device
gradient_accumulation = 8  # effective batch = 8
epochs = 3
precision = bf16
# DeepSpeed ZeRO-3 + gradient checkpointing
```

效果：SFT 后模型能够 (i) 高亮关键物体和属性，(ii) 捕获空间/语义关系，(iii) 将感知证据链接为推理步骤。SFT 为 Stage II RL 提供了稳定的感知-描述接口。Table 2 显示 Before RL (SFT-only) 的 VTPerception-R1-7B 在多个 benchmark 上已超过 Qwen2.5-VL-7B-Instruct baseline（如 AI2D: 80.4 vs 77.2, C-MMBench: 46.7 vs 43.1）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现依赖于自动化数据清洗流水线（论文 Appendix A.2）：(1) 图像分析阶段：VLM 生成 dense caption + Grounding DINO 目标检测 + EasyOCR → 合并为结构化规范描述（formal_description），这是后续所有 CoT 重建的唯一图像信息来源（single-source-of-truth 原则）；(2) CoT 重建阶段：LLM 基于问题 + formal_description 重新生成推理链，明确禁止参考原始 CoT 或外部知识；(3) 质量评估阶段：多维度评分（formal_score, cot_score, answer_score, 幻觉分数），加权求和 overall_score，阈值过滤。代码开源在 https://github.com/yizhuoDi/VTPerceprion-R1。

涉及论文标题：
- VTPerception-R1__Enhancing_Multimodal_Reasoning_via_Explicit_Visual_and_Textual_Perceptual_Grounding
