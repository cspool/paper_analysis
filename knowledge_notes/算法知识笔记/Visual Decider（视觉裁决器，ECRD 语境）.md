## Visual Decider（视觉裁决器，ECRD 语境）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Visual Decider 是 ECRD 框架的第二个核心组件，在 Distribution Supervisor 检测到不确定性时（k*>1 且 margin≤δ）被触发。它是一个轻量级的视觉定位模型（论文中用 GRIT-3B，基于 Qwen2.5-VL-3B），接收图像、当前文本前缀尾部、和候选 token 集 C_i，输出：(a) 一个确定的 token w*∈C_i（模型认为正确的选择）；(b) 一句人类可读的微观察证据句 E_i（包含可选的坐标标注用于可解释性，但不参与 scoring）。ECRD 强制提交 w* 并将 E_i 追加到证据池。关键设计特点：(a) decider 仅接收当前步骤的文本前缀尾部而非完整问题——因为其目标是解决当前步的潜在幻觉，而非回答整个问题；(b) 证据仅以文本形式参与后续 scoring（Eq. 5-7），坐标仅用于可解释性；(c) 稀疏触发——仅在 margin≤δ 时调用，确保计算开销按需发生。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Visual Decider 触发与执行
margin = max(p_mix) - second_max(p_mix)  # 协商分布 margin
if k* > 1 and margin <= delta:           # delta = 0.08
    # Trigger: 候选集多 token + 协商后仍不自信
    w*, evidence_sentence = GRIT.forward(
        image,            # 原始图像
        prefix_tail,      # 当前解码前缀的尾部
        C_i               # 候选 token 集 {"5", "3"}
    )
    # GRIT 内部: 视觉编码 → 定位相关区域 → 生成证据 + 选择 token
    commit(w*)                           # 强制采用 decider 选择
    evidence_pool.append(evidence_sentence)  # 追加文本证据

# 示例输出:
# w* = "3"
# evidence_sentence = "The number behind the cardboard box 
#     with the 'favorita' brand and banana illustration is '300'."
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Visual Decider 使用 GRIT-3B（基于 Qwen2.5-VL-3B + GRPO-GR 视觉定位优化）实例化，部署在独立 backend（FP16 on CPU），与 base LVLM 解耦。每次调用延迟 l_0≈1.12-1.46s（H20 GPU）。在 δ=0.08 时，每问题平均调用次数 r 在低个位数。统计表明：decider 直接输出最终答案的案例占总增益的 11.4%，decider 在中链注入视觉接地的案例占 18.2%，其余增益来自 supervisor 的重新加权和证据池的间接稳定性提升。

涉及论文标题：
- See It, Say It, Sorted: An Iterative Training-Free Framework for Visually-Grounded Multimodal Reasoning in LVLMs
