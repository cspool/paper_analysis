## Fallback Tolerance / Fallback Reward

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Fallback Tolerance（回退容忍）是 VideoAuto-R1 中处理"模型无法不经推理就直接回答"场景的机制。当问题过于复杂（如需要多步数学推导），模型允许在 $a_1$ 位置输出 fallback 字符串 "Let's analyze the problem step by step."（而非强制猜测），然后通过 CoT 推理产出 $a_2$。Fallback Reward $\alpha R_{fallback}$（$\alpha=0.3, R_{fallback} \in \{0,1\}$）在 $a_1$ 为 fallback 字符串且 $a_2$ 正确时提供额外奖励（总奖励从 1.1 增至 1.4），激励模型在无法立刻回答时诚实 defer 而非低置信度猜测。推理时若 $a_1$ 为 fallback 字符串，confidence score 被强制设为 $-\infty$，保证必须进入 CoT 阶段。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Fallback Reward 的效果（Table 12，最后一列）：

```
if a_1 == "Let's analyze the problem step by step.":
    if a_2 is correct:
        total_reward = w_2 * 1 + α * 1 = 1.1 + 0.3 = 1.4
    else:
        total_reward = 0  # 错误推理后仍错误
elif a_1 is a wrong guess (not fallback):
    if a_2 is correct:
        total_reward = w_2 * 1 = 1.1  # 没有 fallback bonus
    else:
        total_reward = 0
```

消融实验（Table 9）显示：(1) w1:w2=0.9:1.1 + fallback α=0.3 在所有 benchmark 上最优（VideoMME 67.3, VideoMMMU 58.6, MVP 39.4, Charades-STA 60.0）；(2) 仅使用不对称权重无 fallback 时 VideoMMMU 56.4（vs 58.6 with fallback），说明 fallback 对推理密集任务提升显著。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：fallback 字符串作为 system prompt 中定义的保留 token sequence（"Let's analyze the problem step by step."），通过 exact match 检测。训练时 fallback reward 是二元值（匹配=1, else=0），与 task reward 独立计算后加权相加。权重 $\alpha=0.3$ 的选择需平衡：过大可能导致模型过度使用 fallback（逃避初始答案），过小则激励不足。VideoAuto-R1 的经验设置（$\alpha=0.3, w_1=0.9, w_2=1.1$）通过消融确定。与训练时 confidence calibration 的差异：fallback 是**显式文本信号**（模型自己决定 defer），而非隐式低概率值——这使得决策可解释、可审计。

涉及论文标题：
- VideoAuto-R1__Video_Auto_Reasoning_via_Thinking_Once__Answering_Twice
