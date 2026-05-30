## Thinking-free RLVR（无思考过程的强化学习训练）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Thinking-free RLVR 是 TimeLens 提出的针对感知主导型任务的强化学习训练范式。与传统 "think-then-answer" RLVR（如 DeepSeek-R1 式：模型先生成思考过程 y_thinking，再生成答案 y_answer，奖励 = accuracy + format）不同，thinking-free RLVR 让模型直接输出答案，跳过显式思考过程。奖励函数简化为仅包含任务准确度：`r(y) = r_acc(y) = IoU(Ŝ, S*)`，无需 format reward。TimeLens 证明在 VTG 这类感知主导型任务上：(1) thinking-free RLVR 性能 > thinking-based RLVR（Charades-TimeLens mIoU: 48.3 vs 42.7）；(2) 训练效率更高（1.0× vs 1.9× 训练时间）；(3) 推理更快（无需生成 thinking tokens）。论文观察到 thinking-based RLVR 在训练过程中 thinking 长度逐渐收敛至简单内容，表明模型学会了 bypass 无益的显式推理。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Thinking-free vs Thinking-based RLVR 对比：
```
# Thinking-based RLVR (DeepSeek-R1 style)
y = [y_thinking, y_answer]  # 两部分
r(y) = r_acc(y_answer) + r_format(y)  # 格式奖励必需
# y_thinking: "Let me analyze the video frames. At 5.2s, 
#  I observe the person reaching for the light switch..."
# y_answer: "The event happens in 5.2 - 12.7 seconds."

# Thinking-free RLVR (TimeLens)
y = y_answer  # 直接输出答案
r(y) = IoU(Ŝ, S*)  # 仅用 IoU，无需格式奖励
# y: "The event happens in 5.2 - 12.7 seconds."
```

TimeLens 中 thinking-free RLVR 的关键实践：
- Vision encoder frozen（节省显存）
- 其余参数可训练（LLM backbone + projector）
- 8×H20 GPU，batch size=8，每 prompt 8 rollouts
- lr=1×10⁻⁶，KL coefficient β=0
- Early stopping 当 reward plateau 时（~310 steps / ~2.5K samples）
- 搭配 difficulty-aware Gaussian sampling 选择训练数据

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Thinking-free RLVR 的实现简化了训练和推理流程：(1) 训练时：prompt 和 system instruction 不包含任何 "think step-by-step"或 "reason first" 指令，模型直接从输入生成最终答案；(2) 奖励计算：仅需一个确定性的、可编程验证的 accuracy reward（如 IoU、exact match 等）；(3) 推理时：生成的 token 数量大幅减少（无 thinking tokens），降低 latency 和 serving cost。适用场景：任务以感知/定位为核心（而非复杂推理），如 video temporal grounding、object counting、spatial localization 等。不适用场景：需要多步推理的复杂 QA、数学证明、代码生成等。

涉及论文标题：
- TimeLens__Rethinking_Video_Temporal_Grounding_with_Multimodal_LLMs
