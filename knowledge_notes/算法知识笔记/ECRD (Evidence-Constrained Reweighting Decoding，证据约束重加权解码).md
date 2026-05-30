## ECRD (Evidence-Constrained Reweighting Decoding，证据约束重加权解码)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
ECRD 是一种 training-free、plug-and-play 的解码框架，在 LVLM 推理时监督每一步 token 选择，确保生成的 token 有对应的视觉证据支持。其核心思想是：不依赖 RL 训练让模型"学会何时看图"，而是在测试时用视觉证据监督每一步。ECRD 由两个组件构成：(a) Distribution Supervisor（分布监督器）——维护一个文本证据池，计算证据诱导的 token 分布 r_i(w)，并与 base 模型的分布 p_i(w) 通过自适应权重 α_i = p_{(1)}（base 模型 top-1 概率）协商混合；(b) Visual Decider（视觉裁决器）——当混合分布 margin 不足且候选集包含多个 token 时触发，读图并生成微观察证据句，强制提交正确 token 并扩充证据池。ECRD 的命名体现了其三步流程：Evidence（积累证据）→ Constrain（约束候选）→ Reweight（重分配概率）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# ECRD 解码循环（每步）
for step i in decoding:
    # 1. Base LVLM 输出 next-token 分布
    p_i = softmax(LVLM(prefix))  # [vocab_size]

    # 2. Knee Truncation: 动态选择候选集大小
    p_sorted = sort(p_i, descending=True)
    k* = argmax_k(p_sorted[k] - p_sorted[k+1])
    C_i = top_k(p_i, k*)  # 候选 token 集

    # 3. 证据评分: 对每条证据计算 mean-over-prefix 概率
    for each E_j in evidence_pool:
        q_Ej(w) = mean_{t=1..L} p_VLM(w | e_{<t})  # 式(5)
    S_i(w) = -log(mean_{j} q_Ej(w))  # 式(6)

    # 4. 证据诱导分布（仅在 C_i 内归一化）
    r_i(w) = softmax_{w in C_i}(-S_i(w))

    # 5. Mass-matching: 让 r_i 在 C_i 内的总 mass 等于 p_i
    r_tilde_i(w) = r_i(w) * sum_{C_i} p_i / sum_{C_i} r_i

    # 6. 协商混合: α_i = top-1 概率控制证据权重
    alpha = max(p_i)
    p_mix = alpha * p_i + (1-alpha) * r_tilde_i

    # 7. 不确定性检测
    margin = max(p_mix) - second_max(p_mix)
    if k* > 1 and margin <= delta:  # delta=0.08
        # 触发 Visual Decider
        w*, evidence = GRIT(image, prefix_tail, C_i)
        commit(w*)
        evidence_pool.append(evidence)
    else:
        commit(argmax(p_mix))
```

典型性能：Qwen2.5-VL-7B + ECRD 在 TreeBench 上 37.0%→47.9%（+10.9 点），超过 GPT-4o 和 Gemini-2.5-Flash；在 RH-Bench 上 RH-AUC 从 0.51→0.58。跨 LLaVA-OneVision、Qwen2.5-VL、InternVL3 三个 backbone 系列和多种 scale 一致有效。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
ECRD 作为 decoding wrapper 包裹 frozen LVLM，不修改任何模型权重。Visual Decider（GRIT-3B）单独部署在另一 backend（FP16 on CPU），仅在触发时调用。证据评分 O(k*|E_i|) 在 CPU 上计算（k* 为个位数，|E_i| 增长缓慢）。每问题平均 decider 调用次数 r(δ) 在 δ=0.08 时处于低个位数，总延迟 T ≈ t_0 + l_0·r（l_0≈1.1-1.5s/call），overhead 控制在 20-30%。开源：github.com/uuuuZYC/See-It-Say-It-Sorted。

涉及论文标题：
- See It, Say It, Sorted: An Iterative Training-Free Framework for Visually-Grounded Multimodal Reasoning in LVLMs
