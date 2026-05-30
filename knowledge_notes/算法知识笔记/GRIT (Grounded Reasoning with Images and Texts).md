## GRIT (Grounded Reasoning with Images and Texts)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
GRIT 是一种通过轻量级 RL（GRPO-GR）训练 MLLM 在推理链中交错生成自然语言和边界框坐标的方法，由 UC Santa Cruz 提出（NeurIPS 2025, arXiv:2505.15879）。GRIT 的核心创新：(a) 训练模型在 `<think>` `</think>` 标签内进行带有边界框坐标 `[x1,y1,x2,y2]` 的视觉推理；(b) 仅需 20 个训练样本即可实现，数据效率极高；(c) 通过三种 reward（格式奖励、计数奖励、答案准确率奖励）引导 GRPO 优化。GRIT 基于 Qwen2.5-VL-3B 和 InternVL3-2B 构建，在 VSR（空间推理）、TallyQA（计数）、GQA（组合推理）上超越 Direct Query 和 CoT baselines。在 ECRD 中，GRIT-3B 被实例化为 Visual Decider——利用其"看图生成接地文本"的能力，在解码歧义步输出 token 选择 + 含坐标的微观察证据句。代码：github.com/eric-ai-lab/GRIT。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
GRIT 在 ECRD 中的使用（作为 Visual Decider）：
```
# GRIT 推理接口
w*, evidence_sentence = GRIT.forward(
    image=original_image,
    context=prefix_tail,          # 当前解码 prefix 尾部
    candidates=C_i                # 候选 token 集
)

# GRIT 内部: 视觉编码 → 定位相关区域 → 
#   <think> grounding reasoning </think>
#   <answer> token_choice + evidence_sentence </answer>
# 解析输出: 提取 w* 和 evidence_sentence
```

GRIT 的训练（GRPO-GR）使用三种 reward：
```
Reward = Format_Reward + Counting_Reward + Answer_Accuracy_Reward
# Format: 检查标签和有效坐标语法
# Counting: 生成的 bbox 数量匹配预期
# Answer: GPT-4o 评判 + BLEU 相似度
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
GRIT 开源在 HuggingFace：GRIT-20-Qwen2.5-VL-3B、GRIT-20-InternVL-2B。作为 Visual Decider 使用时，GRIT 接收图像、当前推理前缀、候选 token 集，输出确定的 token 选择和自然语言证据句。论文中 GRIT alone 的 TreeBench 准确率仅 30.1%（低于 Qwen2.5-VL-7B 的 37.0%），但在 ECRD 框架中作为 decider 时，通过精确触发和证据累积，将 base 7B 提升至 47.9%。

涉及论文标题：
- See It, Say It, Sorted: An Iterative Training-Free Framework for Visually-Grounded Multimodal Reasoning in LVLMs
