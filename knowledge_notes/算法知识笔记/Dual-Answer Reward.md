## Dual-Answer Reward

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Dual-Answer Reward（双答案奖励）是 VideoAuto-R1 中训练 "answer-think-answer" 模板的核心奖励设计。与标准 GRPO 仅奖励最终答案不同，Dual-Answer Reward 同时监督初始答案 $a_1$ 和审查答案 $a_2$，通过不对称权重 $w_2 > w_1$ 鼓励模型通过推理改进答案。总奖励公式为：

$$R = w_1 R_{task}^{(1)}(a_1) + w_2 R_{task}^{(2)}(a_2) + \lambda R_{fmt} + \alpha R_{fallback}$$

其中 $w_1=0.9, w_2=1.1, \lambda=1, \alpha=0.3$。权重不对称的关键推理（Table 12）： 若 $w_1=w_2=1$，"correct→wrong" 和 "wrong→correct" 两种模式获得相同总奖励（1），无法区分；当 $w_1=0.9, w_2=1.1$，"correct→wrong" 得 0.9，"wrong→correct" 得 1.1，明确鼓励模型通过推理纠正初始错误。训练曲线（Figure 6）显示 $R_{task}^{(2)}$ 始终高于 $R_{task}^{(1)}$，验证推理阶段确实改善答案质量。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Dual-Answer Reward Computation (per rollout)
# 解析输出: \boxed{a_1} <think> r </think> \boxed{a_2}

# Task reward computation (depends on task type)
if task == "QA":
    R_task^{(1)} = exact_match_or_math_verify(a_1, GT)   # {0,1}
    R_task^{(2)} = exact_match_or_math_verify(a_2, GT)   # {0,1}
elif task == "Temporal_Grounding":
    R_task^{(k)} = max_tIoU(pred_segments, GT_segments)  # [0,1]
elif task == "Grounding_QA":
    R_task^{(k)} = R_QA + R_TG                           # [0,2]

# Format reward
R_fmt = 1 if regex_match(template) else 0  # strict: exactly 2 boxes + 1 think block

# Fallback bonus
R_fallback = 1 if (a_1 == "Let's analyze...") and (R_task^{(2)} > 0) else 0

# Total reward
R = 0.9 * R_task^{(1)} + 1.1 * R_task^{(2)} + 1.0 * R_fmt + 0.3 * R_fallback
```

奖励分配矩阵（Table 12）：
| $a_1$ | $a_2$ | $w_1$:$w_2$=1:1 | 0.9:1.1 | 0.9:1.1+α=0.3 |
|-------|-------|-----------------|---------|-----------------|
| ✗ | ✗ | 0 | 0 | 0 |
| Let's analyze | ✗ | 0 | 0 | 0 |
| ✓ | ✗ | 1 | 0.9 | 0.9 |
| ✗ | ✓ | 1 | 1.1 | 1.1 |
| Let's analyze | ✓ | 1 | 1.1 | 1.4 |
| ✓ | ✓ | 2 | 2 | 2 |

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
与标准 GRPO 的差异：标准 GRPO 仅有一个 task reward（最终答案），Dual-Answer Reward 需要解析两轮 boxed 答案并分别评估。格式 reward 通过 strict regex 强制执行 `\boxed{...}<think>...</think>\boxed{...}` 格式。Fallback reward α 仅在 $a_1$ 为 fallback 字符串且 $a_2$ 正确时激活，鼓励模型在无法立刻回答时诚实 defer 而非猜测。消融实验（Table 9）证实：(1) 不对称权重（0.9:1.1）优于均匀权重（1:1），VideoMMMU 从 56.1→56.4；(2) 添加 fallback reward α=0.3 进一步提升至 58.6。权重过度不对称（0.8:1.2）可能导致模型过度依赖推理而退化初始答案能力（VideoMME 65.8 vs 67.3）。

涉及论文标题：
- VideoAuto-R1__Video_Auto_Reasoning_via_Thinking_Once__Answering_Twice
