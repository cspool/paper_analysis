## Perception-Aware Reinforcement Learning (感知感知强化学习)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Perception-Aware Reinforcement Learning（感知感知强化学习）是 VTPerception-R1 提出的 RL 训练范式，在标准 DAPO/GRPO 的答案正确性奖励（R_acc）和格式奖励（R_fmt）之上，引入三个专门衡量感知质量的奖励项，将"看到了什么"和"是否基于看到的信息推理"纳入强化学习目标。

三组感知专用奖励：
- **R_vkey (Visual Key-Info Reward)**：衡量 `<description>` 覆盖预标注关键视觉元素（物体属性、几何约束、空间关系）的比例。计算 recall = |K_v ∩ D_desc| / |K_v|，离散化为三档（≥τ_hi → 1.0, τ_lo~τ_hi → 0.5, <τ_lo → 0.0）。
- **R_tkey (Textual Key-Info Reward)**：衡量 `<think>` 覆盖预标注关键文本信息（OCR 文本、数值、单位约束、常识）的比例。计算方式同 R_vkey。
- **R_cons (Description-Reasoning Consistency Reward)**：检查 `<think> + <answer>` 中引用的实体/属性/数值是否在 `<description> + question` 中有据可查。cons = |F_ans ∩ E| / max(1, |F_ans|)；存在明确冲突（如推理引用感知中不存在的数值）时 R_cons = 0。

采用 **Perception-First 加权调度**：训练早期增大 R_vkey 和 R_tkey 的权重，优先建立稳健的感知基础；后期逐步切换到以 R_acc 为主，追求最终答案正确性。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
总奖励函数 R = R_acc + R_fmt + R_vkey + R_tkey + R_rep + R_cons，计算流程：
```
# 采样阶段
for prompt x = (image, question) in batch:
    o[1..G] = π_θ_old.generate(x)  # G 个 rollout
    # 每个 o_i 解析为:
    d_i = extract_tag(o_i, "description")  # 感知描述
    t_i = extract_tag(o_i, "think")         # 推理链
    a_i = extract_tag(o_i, "answer")        # 最终答案

# 奖励计算（6 组件）
for i in 1..G:
    # 1. 答案正确性（序列级）
    R_acc[i] = 1.0 if a_i == ground_truth else 0.0
    
    # 2. 格式合规（结构检查）
    R_fmt[i] = 1.0 if has_all_tags(o_i, ["description","think","answer"]) 
                    and not has_duplicate_tags(o_i) else 0.0
    
    # 3. 视觉关键信息（基于 description）
    D_desc = extract_atomic_facts(d_i)
    cov_v = |K_v ∩ D_desc| / |K_v|  # K_v 来自 RL 数据构建流水线
    R_vkey[i] = 1.0 if cov_v ≥ 0.8 else (0.5 if cov_v ≥ 0.5 else 0.0)
    
    # 4. 文本关键信息（基于 think）
    D_think = extract_atomic_facts(t_i)
    cov_t = |K_t ∩ D_think| / |K_t|
    R_tkey[i] = 1.0 if cov_t ≥ 0.8 else (0.5 if cov_t ≥ 0.5 else 0.0)
    
    # 5. 重复惩罚
    R_rep[i] = -count_repeated_ngrams(o_i, n=3) / len(o_i)
    
    # 6. 描述-推理一致性
    F_ans = extract_entities(t_i + " " + a_i)
    E = extract_entities(d_i + " " + question)
    if has_explicit_conflict(F_ans, E):  # 如: 推理说 AB=30 但 question 说 AB=25
        R_cons[i] = 0.0
    else:
        R_cons[i] = |F_ans ∩ E| / max(1, |F_ans|)
    
    # 总奖励（带 perception-first 调度权重）
    w_acc = schedule_weight(step, "acc")     # 早期小, 后期大
    w_perc = schedule_weight(step, "perc")   # 早期大, 后期小
    R[i] = w_acc * (R_acc[i] + R_fmt[i] + R_rep[i]) 
         + w_perc * (R_vkey[i] + R_tkey[i] + R_cons[i])

# DAPO 策略更新（同 DAPO 条目）
```

消融实验结果（Table 3）验证了各组件的互补性：移除 R_cons → C-MMBench 下降 3.26，C-MMBench-TO 下降 1.70；移除 R_tkey → C-MMBench 下降 2.64，C-MMBench-TO 下降 3.31；移除 R_vkey → AI2D 下降 2.01，MMMU 下降 1.21。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
感知感知 RL 的实现需要配套的数据构建流水线（Algorithm 2 in 论文 Appendix A.3）：(1) 教师模型集成（多个 72B 级模型，随机解码生成多样化推理路径）；(2) 预算验证（按 log-probability 排序 → top-B 候选 → correctness scoring + coherence scoring → 阈值过滤）；(3) 关键信息提取（从验证通过的轨迹中提取视觉关键信息 V 和文本关键信息 Z，Z 包含事实到推理步骤的映射）。最终每个 RL 训练样本表示为 (x, q, verified_answer, verified_trajectory, {V, Z})。代码开源在 https://github.com/yizhuoDi/VTPerceprion-R1，基于 EasyR1-perc 框架实现。

涉及论文标题：
- VTPerception-R1__Enhancing_Multimodal_Reasoning_via_Explicit_Visual_and_Textual_Perceptual_Grounding
