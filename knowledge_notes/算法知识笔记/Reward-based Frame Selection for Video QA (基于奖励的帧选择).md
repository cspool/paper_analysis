## Reward-based Frame Selection for Video QA (基于奖励的帧选择)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Reward-based Frame Selection 是指使用 downstream VLM 的 QA 质量作为 reward signal 来优化帧选择策略的方法范式。与 supervised frame selection（需 frame-level 标注或 pseudo-labels）和 heuristic frame selection（uniform/CLIP similarity 等固定规则）不同，reward-based 方法直接优化最终目标（QA accuracy），不需要 frame-level ground-truth。HORNet 是首次将 GRPO 用于 reward-based frame selection 的工作，其关键创新在于将 reward 信号从 VLM output 端反馈到 VLM input 端（帧选择策略），实现了"优化 VLM 看到什么"而非"优化 VLM 生成什么"。核心流程：policy 生成候选帧子集 → frozen VLM 回答问题 → reward 计算（F1-Lev = 0.1·F1_token + 0.9·EditSim）→ GRPO group-relative advantage + policy gradient 更新 policy 参数。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# === Reward-based Frame Selection via GRPO ===
for each (V, q, a):
    p = MLP(TimeSFormer(V))                # per-frame keep prob
    masks = [TopK_sweep(p, k) + Bernoulli(p)]  # K=8 candidates
    rewards = []
    for b in masks:
        a_hat = frozen_VLM(V[b==1], q)
        r = 0.1 * F1(a_hat, a) + 0.9 * EditSim(a_hat, a)
        rewards.append(r)
    r_bar, σ_r = mean(rewards), std(rewards)
    for i in 1..K:
        A_i = (rewards[i] - r_bar) / (σ_r + ε)
        loss -= A_i * log_π(b_i) / K
    update(θ_policy, θ_encoder)  # VLM never updated
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
HORNet 训练配置：223,646 QA pairs, 两阶段 (MSVD+MSRVTT→NExT-QA), K=8, Adam lr=1e-4, batch_size=8。单卡 A100 40GB。VLM (Qwen3-VL-2B) 全程冻结。Reward 设计关键：lemmatized F1 + EditSim 比 exact match 对 minor lexical variations 更鲁棒。核心优势：(1) 直接优化 QA 质量；(2) 不需 frame-level annotation；(3) GRPO critic-free advantage estimation；(4) trained policy 可跨 VLM transfer。HORNet Table 4 证明 GRPO OOD generalization 优于 PPO 和 SFT。开源: https://github.com/ostadabbas/HORNet。

涉及论文标题：
- HORNet__Task-Guided_Frame_Selection_for_Video_Question_Answering_with_Vision-Language_Models
