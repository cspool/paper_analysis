## DAPO (Decoupled Clip and Dynamic sAmpling Policy Optimization)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
DAPO (Decoupled Clip and Dynamic sAmpling Policy Optimization) 是 ByteDance Seed 与清华大学提出的大规模 LLM 强化学习算法（NeurIPS 2025, arXiv:2503.14476），专为激发长链推理能力而设计。DAPO 在 GRPO (Group Relative Policy Optimization) 基础上引入四项关键技术以解决其在大规模推理训练中的失效模式：

(1) **Clip-Higher（非对称裁剪）**：将 PPO 的对称裁剪边界分离为独立的低裁剪 ε_low 和高裁剪 ε_high（典型值 ε_low=0.2, ε_high=0.28）。高概率 token 的正常裁剪保持策略稳定性，低概率 token 的高裁剪上界允许"探索性"token 的概率增长，防止熵崩溃（所有 rollout 收敛到相同输出导致梯度为零）。

(2) **Dynamic Sampling（动态采样）**：过滤掉模型已完美掌握（100% 正确率）或完全无法处理（0% 正确率）的 prompt，仅对"有信息量"的 prompt（部分正确部分错误）进行过采样直至填满 batch。这避免了梯度消失问题——当所有 rollout 奖励相同时，advantage 为零，无学习信号。

(3) **Token-Level Policy Gradient Loss**：使用 `loss = (1/Σ|o_i|) Σ_i Σ_t min(r_i,t * A_i,t, ...)` 进行逐 token 损失求和，而非先按样本平均再按 batch 平均。这确保长推理链（通常质量更高）对梯度更新的贡献与序列长度成比例，而非被 token 平均稀释。

(4) **Overlong Reward Shaping（软长度惩罚）**：对被截断的超长响应施加长度感知的渐进惩罚（而非硬 -1），避免正确推理但因超出长度限制被截断的响应受到不合理的严重惩罚，减少奖励噪声。

DAPO 使用 Qwen2.5-32B 基座模型在 AIME 2024 上达到 50 分，超越 DeepSeek-R1-Zero-Qwen-32B（47 分），且训练步数减少 50%。完全开源（代码基于 veRL 框架、数据集 DAPO-Math-17K）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
DAPO 训练迭代伪代码（结合 VTPerception-R1 的 perception-aware 扩展）：
```
# 超参数: G rollouts, ε_low=0.2, ε_high=0.28, overlong_threshold
for each training step:
    prompts = sample_batch(D_train)
    
    # 1. Dynamic Sampling: 过滤无信息量 prompt
    valid_prompts = []
    for prompt in prompts:
        # 快速评估当前策略在该 prompt 上的表现
        quick_rollouts = π_θ.sample(prompt, n=G)
        acc = mean([check_answer(r, gt) for r in quick_rollouts])
        if 0 < acc < 1:  # 部分正确部分错误 → 有信息量
            valid_prompts.append(prompt)
        # 若 valid_prompts 不足 batch_size，继续从 D_train 采样
    
    # 2. 采样 G 个 rollouts per valid prompt
    for prompt in valid_prompts:
        o[1..G] = π_θ_old.generate(prompt)
        # o_i 格式: <description> d_i <think> t_i <answer> a_i
        
        # 3. 计算奖励（VTPerception-R1 扩展了标准 DAPO 的奖励函数）
        for i in 1..G:
            R_acc[i] = exact_match(a_i, ground_truth)
            R_fmt[i] = check_format(o_i)
            R_vkey[i] = compute_visual_key_recall(d_i, K_v)  # perception reward
            R_tkey[i] = compute_textual_key_recall(t_i, K_t)
            R_cons[i] = compute_consistency(d_i, t_i, a_i)
            R_rep[i] = -count_repeated_ngrams(o_i)
            R[i] = R_acc + R_fmt + R_vkey + R_tkey + R_rep + R_cons
        
        # 4. Group-relative Advantage (与 GRPO 相同)
        R_mean = mean(R[1..G])
        for i in 1..G:
            A[i] = R[i] - R_mean  # 组内归一化
    
    # 5. Token-Level Clipped Policy Gradient（DAPO 核心）
    total_tokens = sum([len(o_i) for i in 1..G])
    loss = 0
    for i in 1..G:
        for t in 1..len(o_i):
            r_i_t = π_θ(o_i_t | x, o_i_<t) / π_θ_old(o_i_t | x, o_i_<t)
            # Clip-Higher: 非对称裁剪
            clipped = clip(r_i_t, 1 - ε_low, 1 + ε_high)
            loss += min(r_i_t * A[i], clipped * A[i])
    loss = loss / total_tokens  # token-level 归一化
    
    # 6. 梯度更新
    θ = θ - lr * ∇_θ loss
```

DAPO vs GRPO 关键差异：
| 特性 | GRPO | DAPO |
|------|------|------|
| 裁剪策略 | 对称 ε (如 0.2) | 非对称 ε_low ≠ ε_high |
| 采样策略 | 全量采样 | 动态过滤 + 过采样 |
| 损失归一化 | per-sample 平均 | token-level 求和后全局归一化 |
| 长度控制 | 无或硬截断 | Overlong Reward Shaping |

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
DAPO 通过 veRL (Volcano Engine RL) 框架实现，代码开源在 https://github.com/verl-project/verl（`docs/algo/dapo.md`）。使用方式：(1) 定义奖励函数为 Python callable（可包含多组件如 accuracy + format + perception）；(2) 配置 DAPO trainer（G, ε_low, ε_high, overlong_threshold, KL coefficient）；(3) rollout engine 使用 SGLang 或 vLLM 进行高效采样；(4) 训练 loop 中 auto-regressive 采样 → 计算多组件奖励 → 动态过滤无效 prompt → 计算 group-relative advantage → token-level clipped policy gradient update。在 VTPerception-R1 中，DAPO 作为 Stage II 的基础优化器，论文基于 EasyR1-perc 框架实现（EasyR1 对 DAPO 的封装），使用 Ray 分布式部署（1 主节点 + 1 ORM 节点），TP=4。

涉及论文标题：
- VTPerception-R1__Enhancing_Multimodal_Reasoning_via_Explicit_Visual_and_Textual_Perceptual_Grounding
