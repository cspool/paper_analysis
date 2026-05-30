## Negotiated Reweighting（协商重加权）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Negotiated Reweighting 是 ECRD 中将 base 模型分布 p_i 与证据诱导分布 r_i 融合的核心机制。其"协商"体现在：(a) 两个分布来源不同——base 来自 LVLM 的完整上下文推理，evidence 来自视觉证据池的纯视觉接地——在 token 选择上可能存在分歧；(b) 自适应权重 α_i = p_{(1)} 决定哪一方更有话语权——base 自信时 base 主导，base 犹豫时 evidence 主导；(c) mass-matching 确保 r_i 在候选集 C_i 内的总概率质量与 p_i 一致，仅重分配 C_i 内部的相对概率而非改变 C_i 的总 mass。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Negotiated Reweighting 具体计算
# 输入: p_i (base分布), r_i (证据诱导分布), C_i (候选集)

# Mass-matching
mass_p = sum_{w in C_i} p_i(w)     # e.g., 0.981
mass_r = sum_{w in C_i} r_i(w)     # e.g., 1.0
r_tilde_i(w) = r_i(w) * mass_p / mass_r

# 自适应协商
alpha_i = max(p_i)  # e.g., 0.498 (base 不自信)
p_mix_i(w) = alpha_i * p_i(w) + (1-alpha_i) * r_tilde_i(w)

# 数值示例:
# C_i = {"5", "3"}
# p_i: "5":0.498, "3":0.483; r_tilde_i: "5":0.503, "3":0.478
# alpha = 0.498
# p_mix("5") = 0.498*0.498 + 0.502*0.503 = 0.501
# p_mix("3") = 0.498*0.483 + 0.502*0.478 = 0.480
# margin = 0.021 ≤ 0.08 → 触发 decider
```

α_i 的设计体现了"最小干预原则"：仅在必要时介入，常规步骤保持模型自身的行为。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Negotiated Reweighting 是纯数学计算，无需额外模型调用。mass-matching 步骤（Eq. 8）确保了合理性——不改变 C_i 的总概率质量，只在内部重新分配。与 VDGD 的直接 logit 替换的本质区别：VDGD 完全丢弃 base 分布的信息，而 ECRD 保留 base 模型的校准置信度作为混合权重。

涉及论文标题：
- See It, Say It, Sorted: An Iterative Training-Free Framework for Visually-Grounded Multimodal Reasoning in LVLMs
