## LLM-Match (LLM 匹配评分)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
LLM-Match 是一种基于 LLM Judge 的文本相似度评估指标，用于自动评价生成式输出（如视频 caption）与 ground truth 的语义匹配程度。不同于 n-gram 重叠指标（ROUGE、BLEU）或嵌入相似度，LLM-Match 使用一个独立的 LLM（Judge）根据精心设计的评分指令对生成结果进行 0.0-1.0 分的语义相似度评分：0.0-0.3（差，关键细节缺失）、0.4-0.6（中等，部分细节匹配）、0.7-0.9（好，大部分关键细节匹配）、1.0（完美，所有关键细节准确）。在 SPIKE-RL 中，LLM-Match 的评分被直接用作 GRPO 训练中的 **reward signal**。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# LLM-Match 评分流程
# Judge LLM: Olmo-7B-hf (或类似 LLM)
# 输入: 模型生成的 caption c, ground truth caption c_gt
# 输出: scalar reward R ∈ [0.0, 1.0]

def llm_match(c_pred, c_gt, judge_model):
    prompt = f"""
    Rate how closely the content of the prediction matches the content of 
    the reference description in terms of meaning and how well it captures 
    important details regarding events in the video. Ignore the difference 
    in length. Score 0.0-1.0 where:

    0.0-0.3: Poor match (key details in the reference are missing)
    0.4-0.6: Moderate match (a few key details are captured)
    0.7-0.9: Good match (most key details are present)
    1.0: Perfect match (all key details accurately captured)

    Output only the numerical score.

    Reference: {c_gt}
    Response: {c_pred}
    Score:
    """
    score = judge_model.generate(prompt, max_tokens=10)
    return float(score)

# SPIKE-RL 训练中: R = llm_match(caption_rollout, caption_ground_truth)
# 在 GRPO group 内 Z-score 归一化: A = (R - μ_R) / σ_R
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
SPIKE-RL 使用 Olmo-7B-hf 作为 LLM-Match Judge，而非使用更大的模型（如 GPT-4o）以降低训练成本。LLM-Match 在 2024-2025 年的 Video-LLM 评估中广泛使用（OpenEQA, FunQA 等），逐渐替代 ROUGE/BLEU 等 n-gram 指标。核心优势：(1) 关注语义匹配而非表面形式，不受句式、长度影响；(2) 可作为 RL 训练的密集 reward signal（连续值 0-1，而非 binary 正确/错误）；(3) 与人类判断的高相关性使其适合作为自动评估的代理。局限性：(1) Judge LLM 本身的 bias（可能偏好特定措辞风格）；(2) 表面上的"高匹配"不保证事实准确性；(3) 在某些类型误差中与人类判断存在系统性偏差。

涉及论文标题：
- SPIKE-RL__Video-LLMs_meet_Bayesian_Surprise
