## GRPO-CSV (GRPO with Completeness Self-Verification)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
GRPO-CSV 是 TimeSearch-R 论文提出的一种改进 GRPO 强化学习算法，专门为长视频时序搜索任务设计。在标准 GRPO 的 outcome-only reward 基础上增加 **Completeness Self-Verification (CSV, 完备性自验证)** 阶段：在 GRPO rollout 中 policy model π_θ 生成 text-video 交错 CoT C 和最终答案 A 后，CSV 提取 C 中所有搜索到的视频帧构成动态帧集 V_c，用同一模型仅基于 V_c 重新回答问题（禁止新搜索），得到 CSV 答案 A_c。Completeness Reward: R_c = 1[Acc(A, A*) > 0.5] · Acc(A_c, A*)，仅当原始答案 A 正确时才施加 CSV reward。总奖励：R = R_c + R_fmt + R_acc。GRPO-CSV 解决标准 GRPO 的两个失败模式：(1) 搜索不充分 —— outcome-only reward 无中间搜索监督，模型可能凭部分证据或语言偏置答对而缺乏视觉 grounding；(2) 推理不一致 —— 中间推理过程可能与最终答案脱节。Ablation 显示移除 CSV 使 completeness 从 60.5% 降至 57.2%，temporal F1 从 7.8 降至 7.4，且训练约 300 step 崩塌（模型停止搜索）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

GRPO-CSV 训练流程伪代码：
```
# 超参数: K=8 rollouts, β=0.005 KL coeff, lr=1e-6 AdamW
# 平台: TRL + DeepSpeed ZeRO-3 + vLLM colocate

for step in range(num_rl_steps):
    for (V, Q, A*) in batch:  # batch_size=4, grad_accum=2
        # ===== GRPO Rollout (vLLM colocate) =====
        for k in range(K):  # K=8
            C_k, A_k = π_old.interleaved_reasoning(V, Q)
        # ===== Reward Computation =====
        for k in range(K):
            R_acc[k] = 1 if A_k == A* else 0
            R_fmt[k] = 1 if valid_format(C_k, A_k) else 0
            # ===== CSV Rollout =====
            V_c = extract_all_frames(C_k)  # 收集搜索到的帧
            A_c = π_old.answer_no_search(Q, V_c)  # 禁止工具
            if Acc(A_k, A*) > 0.5:  # 仅正确轨迹
                R_c[k] = Acc(A_c, A*)
            else:
                R_c[k] = 0
            R[k] = R_c[k] + R_fmt[k] + R_acc[k]
        # ===== GRPO Update =====
        baseline = mean(R[1..K])
        for k in range(K):
            A_adv[k] = R[k] - baseline
        loss = -Σ min(r_t·A_adv, clip(r_t,1-ε,1+ε)·A_adv)
               + β·KL(π_θ||π_ref)
        optimizer.step(loss)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：(1) 基于 TRL (Transformer Reinforcement Learning) 库构建，利用 vLLM colocate 模式在训练 GPU 上同时做 rollout 推理；(2) CSV prompt 与主推理 prompt 不同：要求简短回答且可输出 "I don't know"，tools 被移除防止新搜索；(3) 仅对正确轨迹施加 CSV reward 是关键 —— 避免模型学习低质量搜索策略；(4) SFT cold-start 阶段用 GPT-4o 生成交错 CoT 数据，mask 视频 token 梯度。适用场景：需要中间步骤监督但缺乏 process annotation 的 multi-turn tool-calling RL 训练（视频搜索、网页搜索、代码搜索等）。

涉及论文标题：
- TimeSearch-R__Adaptive_Temporal_Search_for_Long-Form_Video_Understanding_via_Self-Verification_Reinforcement_Learning
