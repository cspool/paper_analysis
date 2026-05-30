## HORNet__Task-Guided_Frame_Selection_for_Video_Question_Answering_with_Vision-Language_Models

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：HORNet（Hindsight Optimization Reasoning）是一个基于GRPO（Group Relative Policy Optimization）训练的轻量级帧选择策略，核心设计是Select Any Frames（SAF）——将帧选择形式化为强化学习问题，通过下游VLM的QA准确率提供直接奖励信号。三阶段pipeline：(1) **Video Encoder**（TimeSFormer-Tiny, factorized spatial-temporal attention）从T=32帧均匀采样中提取每帧特征F ∈ R^{T×D}；(2) **MLP Policy π^θ**（<1M trainable params, D→512→256→1 + GELU + sigmoid）输出每帧独立的keep probability p_t ∈ (0,1)，定义Bernoulli分布生成二进制选择mask b ∈ {0,1}^T；(3) **Frozen VLM Answerer**（Qwen3-VL-2B）接收选中帧回答问题，计算reward。GRPO训练：每步生成K=8个候选mask（top-k sweep + 1 stochastic Bernoulli），reward = 0.1·F1_token + 0.9·EditSim（lemmatized），group-normalized advantage A^(i) = (r^(i) - r̄)/(σ_r + ε)，Adam lr=10^-4，两阶段训练（先MSVD+MSRVTT用F1-Lev reward，再NExT-QA用MCQ selection-accuracy reward）。Policy仅MLP和Video Encoder可训练，VLM始终冻结。

  实验比较：(a) Open-Ended QA —— MSVD-QA/MSRVTT-QA/NExT-QA open-ended 上对比 Qwen3-VL-2B baseline（uniform sampling），指标 F1-Lev + 效率（Frame Sel. time / Qwen Proc. time / Avg. Frames）；(b) Multiple-Choice QA —— VideoMME/ActivityNet-QA/NExT-QA MCQ 上对比baseline，指标 Accuracy + 效率；(c) 训练目标消融 —— 仅训练MSVD-QA，对比 No training / SFT (weighted BCE) / PPO (clipped surrogate) / GRPO 在 MSVD (in-dist) 和 MSRVTT (out-of-dist) 上的F1-Lev；(d) 帧选择策略消融 —— 均选4帧，对比 Random / Uniform / HORNet 在 MSVD/MSRVTT (F1-Lev) 和 NExT-QA (Acc.) 上；(e) VLM Answerer 消融 —— 同一HORNet policy 换 frozen answerer (Qwen3-VL-2B → Qwen2.5-VL-3B) 在 MSVD-QA 上F1-Lev对比。

- 硬件平台是什么，配置是什么。
  所有实验在单张 NVIDIA A100 40GB GPU 上执行。视频解码后均匀采样 T=32 帧，每帧 resize 至 288×288。训练：Adam lr=10^-4, batch_size=8, 两阶段训练总计约 223,646 training QA pairs。Qwen3-VL-2B 全程冻结，仅 TimeSFormer-Tiny encoder + MLP policy 可训练（<1M params）。

- 模型是什么。数据集和bench分别是什么。
  模型：Video Encoder = TimeSFormer-Tiny（patch_size=16, D=768, spatial feature maps 16×16×768 → spatial avg pool → per-frame F ∈ R^{T×768}）；MLP Policy: 768→512→1024→256→1 (<1M params)；Frozen VLM Answerer: Qwen3-VL-2B（主实验），Qwen2.5-VL-3B（transfer 消融）。
  训练数据集（223,646 QA pairs, 15,031 videos, 114.2h）：MSRVTT-QA (158,581 QA pairs, mean 15.5s), MSVD-QA (30,933 QA pairs, mean 9.6s), NExT-QA (34,132 QA pairs, mean 43.7s)，覆盖 descriptive/temporal/causal 问题类型。
  Benchmarks：Open-Ended QA —— MSVD-QA, MSRVTT-QA, NExT-QA open-ended（指标 F1-Lev=0.1·F1_token+0.9·EditSim）；MCQ —— VideoMME, ActivityNet-QA, NExT-QA MCQ（指标 selection-accuracy），各随机采样 1,000 QA pairs。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址：https://github.com/ostadabbas/HORNet

  算法 pipeline 伪代码（HORNet 训练与推理全流程）：
  ```
  # ==========================================
  # HORNet Training with GRPO
  # ==========================================
  # 输入: Video V (T frames), Question q, Answer a
  # 超参数: K=8 (candidates), lr=1e-4

  # 1. Video Encoding
  # V: (T, H, W, C), T=32 frames, H=W=288, C=3
  # TimeSFormer-Tiny: patchify (P=16) → spatial self-attention per frame
  #   → temporal self-attention across frames at each patch position
  # E(V): (T, 16, 16, 768) → spatial avg pool → F ∈ R^{T×D}, D=768

  def encode_video(V):
      patches = patchify(V, P=16)  # (T, 16, 16, 16*16*3)
      # Spatial self-attention (per frame independently)
      for each frame t:
          x_t = SpatialTransformer(patches[t])  # (256, 768)
      # Temporal self-attention (across frames per patch position)
      for each patch position (i,j):
          x_ij = TemporalTransformer(all_frames[:, i, j, :])  # (T, 768)
      F = avg_pool_2d(x)  # (T, 768)
      return F

  # 2. MLP Policy: per-frame independent scoring
  def policy(F):
      # θ = {W_0, W_1, W_2}, trainable < 1M params
      for each frame t:
          h_0 = GELU(W_0 @ F[t])        # 768 → 512
          h_1 = GELU(W_1 @ h_0)         # 512 → 256  (paper says 1024→256 in implementation)
          p_t = sigmoid(W_2 @ h_1)       # 256 → 1
      return p = [p_1, ..., p_T]         # ∈ (0,1)^T

  # 3. GRPO Training Step
  for each (V, q, a) in batch:
      F = encode_video(V)
      p = policy(F)

      # Candidate generation: K=8 masks
      masks = []
      # top-k sweep: progressively reduce selected frames
      for k from T down to 1 step size floor(T/K):
          b = top_k(p, k)  # deterministic: keep top-k probability frames
          masks.append(b)
      masks.append(bernoulli_sample(p))  # stochastic exploration

      # Reward computation for each candidate
      rewards = []
      for b in masks:
          V_selected = V[b == 1]
          a_hat = frozen_Qwen3VL(V_selected, q)
          r = 0.1 * F1_token(lemmatize(a_hat), lemmatize(a))
            + 0.9 * EditSim(a_hat, a)
          rewards.append(r)

      # GRPO loss
      r_bar = mean(rewards)
      sigma_r = std(rewards)
      for i in range(K):
          # Log probability of mask b^(i)
          log_pi = sum(b_i[t] * log(p_t) + (1-b_i[t]) * log(1-p_t) for t in 1..T)
          A_i = (rewards[i] - r_bar) / (sigma_r + epsilon)
          L_grpo -= A_i * log_pi / K

      # Update only θ (MLP policy) + encoder; VLM frozen
      θ = Adam(θ, grad(L_grpo), lr=1e-4)

  # 4. Inference (deterministic top-k selection)
  def horNet_inference(V, q, n_frames=4):
      F = encode_video(V)
      p = policy(F)
      top_indices = argsort(p, descending=True)[:n_frames]
      V_selected = V[top_indices]
      answer = frozen_Qwen3VL(V_selected, q)
      return answer
  ```

  关键张量维度：
  - Video input: T=32 frames × 288×288×3 → patchify (P=16) → (32, 18, 18, 768) per-frame patch tokens
  - After spatial avg pool: F ∈ R^{T×D}, T=32 (or variable), D=768
  - MLP Policy: 768 → 512 → 1024 → 256 → 1, output p ∈ [0,1]^T
  - Selected frames: typically 4 (MSVD/MSRVTT) or 8 (NExT-QA long), compressed from 32 input frames and from 11.65~1157.88 baseline avg frames
  - GRPO: K=8 candidate masks per training step, rewards normalized within group
  - Trainable params: <1M (MLP policy ~790K + TimeSFormer-Tiny fine-tuning); VLM ~2B params frozen

  两阶段训练策略：
  - Stage 1 (MSVD + MSRVTT): 短视频 (<100 frames), one-word answers, reward = F1-Lev (0.1·F1 + 0.9·EditSim)
  - Stage 2 (NExT-QA): 长视频 (~1000 frames), MCQ-style, reward = selection-accuracy
  - 意图：短视频学"哪些帧对回答有用"，长视频学"哪些帧包含因果/时序推理所需信息"
