## Auto-Thinking (Adaptive Reasoning / Auto Reasoning)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Auto-Thinking（也称 Adaptive Reasoning 或 Auto Reasoning）是一种推理控制策略，允许模型在推理时**动态决定是否调用链式思维（CoT）推理**，而非对所有输入都强制执行 CoT。核心理念是"必要时才推理"（reason-when-necessary）：简单/感知导向的输入直接输出答案（direct answering），复杂/推理导向的输入触发完整 CoT 推理链。相比"始终推理"（always-thinking），Auto-Thinking 旨在保持准确率的同时显著减少推理 token 消耗和延迟。VideoAuto-R1 将 Auto-Thinking 首次系统性地引入视频理解领域，揭示了视频 CoT 并非普遍有效——在感知导向 benchmark（VideoMME, MVBench）上 direct answering 通常匹配甚至超过 CoT 性能。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Auto-Thinking 的实现通常分为两类途径：

**(a) Training-based Auto-Thinking（训练时学习切换策略）**：在训练阶段为每个样本标注 think/no-think 标签（例如 AdaptThink），通过 SFT + RL 让模型学会对 easy/hard 样本分别输出不同格式（有/无 CoT）。VideoAuto-R1 的消融（Table 7）证明此方法在视频域易发生 mode collapse（始终 think 或始终 no-think），因为视频中"must-think"样本稀缺（VideoMMMU 上 CoT-Direct gap 仅 +1~3.4%）。

**(b) Inference-based Auto-Thinking（推理时自动决策）**：VideoAuto-R1 采用的策略。训练时不区分 think/no-think 模式，统一使用 answer→think→answer 格式。推理时通过 confidence score 决定早停或继续：

```
# Inference-Based Auto-Thinking (VideoAuto-R1)
Require: model p_θ, input (v,q), confidence threshold τ=0.97

# Step 1: Generate first answer a_1 until <think> tag detected
a_1_tokens, logprobs = p_θ.greedy_decode(v, q, stop_token="<think>")

# Step 2: Compute confidence score
L = len(a_1_tokens)
if a_1 == "Let's analyze...":   # fallback string
    s = -∞  # force continue
else:
    s = (1/L) * Σ_{ℓ=1}^{L} logprobs[ℓ]   # length-normalized mean log-prob

# Step 3: Decision
if s >= log(τ):     # e.g., τ=0.97 → log(0.97) ≈ -0.0305
    return a_1       # EARLY EXIT: direct answer (~10 tokens)
else:
    a_2 = p_θ.continue_decode()  # Continue: CoT + reviewed answer (~91 tokens)
    return a_2
```

VideoAuto-R1 中 Auto-Thinking 的行为特征（Table 8）：
- 感知导向 benchmark (MVBench): 平均 confidence 0.948, think ratio 25%, CoT gain +0.1%
- 推理导向 benchmark (VideoMMMU): 平均 confidence 0.874, think ratio 51%, CoT gain +4.0%
- Recall of think-needed samples（a_1 错误 a_2 正确的样本被路由到 CoT mode）: 94-100%

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
当前文献中 Auto-Thinking 的实现方式：AdaptThink（Zhang et al. 2025b）在训练时标注 think/no-think 标签，通过 on-policy 数据平衡维持 mode 比例约 1:1；R-4B（Yang et al. 2025b）采用 Bi-Mode Annealing，先用 SFT 二模式预热再 RL 精炼；TON（Think-or-Not, 2025）使用 "thought dropout" SFT + GRPO 自由探索。VideoAuto-R1 区别于这些方法：(1) 训练时不区分模式（统一 answer-think-answer 格式），消除 per-sample 标签和 mode collapse 问题；(2) 推理时通过 token-level confidence（length-normalized log-probability）自动化决策，τ=0.97 跨数据集泛化无需调参。局限性：(1) confidence score 仅基于自回归 log-probability，未显式校准；(2) 对视频 grounding 任务 CoT 几乎无增益（初始与审查答案 mIoU 相同），说明纯语言 CoT 无法细化精确的时间边界。

涉及论文标题：
- VideoAuto-R1__Video_Auto_Reasoning_via_Thinking_Once__Answering_Twice
