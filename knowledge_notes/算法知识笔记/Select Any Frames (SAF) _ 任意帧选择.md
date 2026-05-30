## Select Any Frames (SAF) / 任意帧选择

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Select Any Frames (SAF) 是 HORNet 提出的帧选择问题形式化框架，将视频帧选择解耦为独立于 VLM 推理的 RL 问题。给定视频 V={v_1,...,v_T}（T 帧均匀采样）、问题 q 和 ground-truth 答案 a，SAF 的目标是学习参数化策略 π_θ 选择一个子集 V'=π_θ(V,q) ⊆ V，最大化 frozen VLM M 的回答质量：θ* = argmax_θ E[R(M(π_θ(V,q), q), a)]。策略输出 binary mask b ∈ {0,1}^T，无时序顺序或连续性约束——策略可自由选择时间上稀疏的关键事件、短关键片段或密集运动段。策略分布分解为独立 Bernoulli：π_θ(b|V,q) = Π_t Bernoulli(b_t|p_t)，p_t 为 frame t 的选择概率。SAF 的核心贡献是将帧选择从 VLM 推理中解耦（modular policy + frozen VLM），使 frame selection policy 可独立训练、可 transfer 到不同 VLM answerer。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
SAF 在 HORNet 中的完整 pipeline：
```
# === SAF Pipeline (HORNet) ===
# 输入: V (T frames), q (question), a (ground-truth answer)
# M: frozen VLM (Qwen3-VL), E: trainable video encoder, π_θ: trainable MLP

# Step 1: Video Encoding
F = E(V)                    # TimeSFormer-Tiny → R^{T × D}, D=768

# Step 2: Frame Selection Policy (per-frame independent)
for t in 1..T:
    p_t = sigmoid(W_2 · GELU(W_1 · GELU(W_0 · F[t])))
# p_t ∈ (0,1), θ = {W_0, W_1, W_2}: <1M params

# Step 3: Sampling (train) / Top-K (inference)
# 训练: b_t ~ Bernoulli(p_t)
# 推理: b = TopK(p, k)

# Step 4: VLM QA
V' = V[b == 1]
a_hat = M(V', q)

# Step 5: Reward
R = 0.1 * F1_token(a_hat, a) + 0.9 * EditSim(a_hat, a)
```

SAF 与现有 frame selection 方法的关键区别：Fully learned selection (vs uniform/clip-similarity heuristics), Reward-based optimization (vs pseudo-label SFT), Frozen VLM (vs fine-tuning), Parameter efficient (<1M vs ~1B+)。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在 HORNet 中，SAF 通过 GRPO 训练：K=8 candidates (7 top-k sweep + 1 Bernoulli), reward = F1-Lev, group-relative advantage, Adam lr=1e-4。两阶段训练 (Stage 1: short videos + F1-Lev, Stage 2: long videos + MCQ accuracy)。推理 deterministic top-k (4-8 frames)。训练硬件：单卡 A100 40GB。开源：https://github.com/ostadabbas/HORNet。Policy 可跨 VLM answerer transfer（+8.5% relative gain with Qwen2.5-VL-3B）。

涉及论文标题：
- HORNet__Task-Guided_Frame_Selection_for_Video_Question_Answering_with_Vision-Language_Models
