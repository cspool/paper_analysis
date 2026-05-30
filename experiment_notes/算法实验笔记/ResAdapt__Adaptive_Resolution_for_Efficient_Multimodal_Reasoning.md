## ResAdapt__Adaptive_Resolution_for_Efficient_Multimodal_Reasoning

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：ResAdapt —— 一个输入侧自适应（Input-side Adaptation）框架，通过在视觉编码之前动态分配每帧分辨率预算来减少MLLM的视觉token开销。核心创新：
  (1) **Allocator（分配器）**：基于 SmolVLM 架构的轻量级前端预测器。每帧 ft 先通过 frozen lightweight visual encoder 提取粗粒度特征，query 也被单独编码，两者投影至共享维度 D。一个浅层 Transformer decoder 交替进行时序 self-attention over {ft} 和 gated cross-attention to query，输出每帧隐状态 {ht}。每帧通过 Beta 分布参数化潜在动作：
    - at ~ Beta(αt, βt)，st = smin + at · (smax − smin)
    - smin=0.2, smax=1.8，支持降采样和选择性升采样
    - 对数密度跨帧因子化：log qθ(a|x) = Σ log Beta(at; αt, βt)
  (2) **Visual Budget Operator（视觉预算算子）**：在 resize 实例化中，算子 O 对每帧执行双线性缩放 ft = R(ft, st)，st 决定了每帧的 token 数 n(st) ∝ ⌈st·H/P⌉·⌈st·W/P⌉。token retention ratio ρ(s) = Σ n(st) / Σ n(1) ≈ Σ st²·Ht·Wt / Σ Ht·Wt。
  (3) **CAPO（Cost-Aware Policy Optimization，成本感知策略优化）**：解决 naive accuracy−cost penalty 导致策略向最小预算崩溃的问题。核心机制：
    - **Proxy cost**：c(s) = (s̄ − smin)/(smax − smin)，s̄ = mean(st)，避免二次方依赖放大方差
    - **Dynamic cost pivot**：τdyn = κmix · c̄group + (1−κmix) · τfix，在组内平均和全局固定目标之间插值
    - **Asymmetric reward shaping**：若正确且低于 pivot 成本→中等奖励 λ+；若不正确且高于 pivot→强惩罚 λ−（λ− > λ+ > 0）；使用 sigmoid 温度 τs 平滑决策边界
    - **CAPO advantage**：Ãm,n = A_base_m,n + λcapo·Sm,n − γ·cm；对正确 rollout 施加正下限 ε+ > 0
  (4) **正则化**：
    - **Temporal Similarity Loss**（Lsim）：惩罚相似相邻帧上的联合高预算分配，仅在余弦相似度超过阈值 τsim 时激活
    - **Concentration Loss**（Lcon）：软限制 Beta 分布的总 concentration αt+βt ≤ κmax，防止分布坍缩为确定性
  (5) **训练过程**：GRPO 风格循环。每 prompt 采样 M=16 个 allocation 轨迹，每个 transformed input 产生 N=1 个 rollout。CAPO 计算 per-rollout advantages，聚合成 per-allocation advantage。Allocator 和 Backbone 交替优化。

  实验比较：
  (a) **Video QA（Table 1）** —— 两个 backbone（Qwen2.5-VL-7B, Qwen3-VL-8B），两个时间范围（32/128 frames）。6 个 benchmark：VideoMME, LongVideoBench, MMVU, MLVU, VideoMMMU, LVBench。对比 heuristic baselines（Random Drop, FixedScale）、model-side compression（ToMe, VisionZip, FlashVid）、reasoning-time inference augmentation（VideoAuto-R1）。在 ~10% retention 下 ResAdapt 大幅领先，VideoMMMU 上 45.7 vs ToMe 39.2 vs VisionZip 39.1。
  (b) **Temporal Grounding（Table 2）** —— Charades-STA, ActivityNet, NExT-GQA。评估 Recall@{0.3,0.5,0.7} 和 mIoU。在 16.2% retention 下 ResAdapt 保留 Charades-STA mIoU 35.6，远超 Random Drop 25.7 和 ToMe 26.0。
  (c) **Operator Generalization（Table 5）** —— Zero-shot transfer 到 frame selection：用 Allocator 预测的 scale 作为重要性分数对 128 候选帧排序，Top-K 选择 + resize。
  (d) **Ablation Studies** —— CAPO reward design（Table 4, Figures 6/15/16）：对比 β-CAPO vs N-CAPO、direct cost vs cost-free vs CAPO；Temporal regularization（Figure 7/13/14）：Lsim 消融对比。
  (e) **Image Transfer（Table 7）** —— MathVista, MMMU, OCRBench, ChartQA, AI2D, TextVQA 上的零样本迁移边界测试。
  (f) **Latency Breakdown（Table 3）** —— vLLM 4-GPU engine + 单 GPU Allocator，16/32/64/128 frames，三个 retention level（~74%, ~51%, ~28%）。

- 硬件平台是什么，配置是什么。
  训练：32×NVIDIA H100 GPUs，使用 VeRL + DeepSpeed ZeRO + vLLM 进行分布式训练。Global batch size=128。AdamW optimizer, Allocator lr=2×10⁻⁵, Backbone lr=1×10⁻⁶, weight decay=0.01, gradient clipping=1.0。最大 video token budget=8192，训练时采样 T=128 frames。
  推理延迟测试：单 GPU Allocator + 4-GPU vLLM engine（Qwen2.5-VL-7B）。

- 模型是什么。数据集和bench分别是什么。
  模型：
  - Backbone: Qwen2.5-VL-7B-Instruct 和 Qwen3-VL-8B-Instruct（插件式迁移测试）
  - Allocator: SmolVLM 架构（轻量级），Lpred=4 layers, Dpred=1024, Pc=14
  数据集：
  - 训练数据：VideoAuto-R1 数据集的 difficulty-filtered 子集（仅保留图像+视频，丢弃纯文本）+ 16500 Video-R1 高复杂度视频实例（OCR, free-form QA, regression tasks）。总计约 93.4K 训练样本。严格剔除评测集样本防止数据泄露。
  Benchmarks：
  - Video QA: VideoMME, LongVideoBench, MMVU, MLVU, VideoMMMU, LVBench
  - Temporal Grounding: Charades-STA, ActivityNet, NExT-GQA
  - Image Understanding: MathVista, MMMU, OCRBench, ChartQA, AI2D, TextVQA
  评估框架：lmms-eval

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源：Code https://github.com/Xnhyacinth/ResAdapt

  算法 pipeline 伪代码（ResAdapt 推理 + CAPO 训练核心流程）：

  ```
  # ========== 推理：ResAdapt 输入侧自适应 ==========
  # 输入: video V of T frames, query q, Allocator πθ, Backbone πφ
  # 输出: answer y

  def resadapt_inference(V, q, πθ, πφ):
      # Step 1: 轻量级前端特征提取
      f_coarse = lightweight_encoder(V)  # frozen, SmolVLM, [T, D]
      q_enc = query_encoder(q)           # [1, D]

      # Step 2: Transformer decoder 交叉融合
      h = transformer_decoder(
          temporal_self_attn(f_coarse),   # [T, D]
          gated_cross_attn(f_coarse, q_enc)  # query-aware
      )  # [T, D]

      # Step 3: Beta 分布参数化 + 采样（推理时取均值）
      for t in 1..T:
          α_t, β_t = linear_head(h_t)  # softplus activated
          a_t = α_t / (α_t + β_t)      # 推理时取期望
          s_t = s_min + a_t * (s_max - s_min)  # [0.2, 1.8]

      # Step 4: 对每帧执行 budget operator（resize）
      for t in 1..T:
          Ṽ_t = bilinear_resize(V_t, scale=s_t)
          # token count: n_t ≈ s_t² * H*W / P²

      # Step 5: 标准 backbone 单次推理
      y = πφ(q, Ṽ)  # compatible with FlashAttention/vLLM/SGLang
      return y

  # ========== 训练：CAPO (Cost-Aware Policy Optimization) ==========
  def capo_training(prompt_batch, πθ, πφ):
      for x in prompt_batch:
          # Step 1: 采样 M 个 allocation 轨迹
          for m in 1..M:  # M=16
              a_m ~ Beta(α, β)  # 从 Allocator 采样
              s_m = s_min + a_m * (s_max - s_min)
              x̃_m = (q, {resize(V_t, s_m[t])}_{t=1..T})

              # Step 2: 每 allocation 采样 N 个 rollout
              for n in 1..N:  # N=1
                  y_{m,n} ~ πφ(· | x̃_m)
                  R_task_{m,n} = compute_reward(y_{m,n}, ground_truth)
                  u_{m,n} = correctness_indicator(R_task_{m,n})

          # Step 3: 计算 CAPO advantages
          c_m = proxy_cost(s_m)  # c(s) = (s̄ - s_min)/(s_max - s_min)
          τ_dyn = κ_mix * mean(c_m) + (1-κ_mix) * τ_fix

          for m in 1..M:
              for n in 1..N:
                  if u_{m,n} == 1:  # 正确
                      S_{m,n} = λ_+ * σ((τ_dyn - c_m) / τ_s)
                  else:  # 错误
                      S_{m,n} = -λ_- * σ((c_m - τ_dyn) / τ_s)

                  A_base_{m,n} = GRPO_normalize(R_task_{m,n})
                  Ã_{m,n} = A_base_{m,n} + λ_capo * S_{m,n} - γ * c_m
                  A_{m,n} = max(Ã_{m,n}, ε_+) if u_{m,n} == 1 else Ã_{m,n}

          # Step 4: 聚合 per-allocation advantage
          for m in 1..M:
              A_m_CAPO = mean_n(A_{m,n})

      # Step 5: 更新 Allocator（PPO clip + 正则化）
      r_θ = q_θ(a) / q_θold(a)  # per-frame importance ratio
      L_θ = -mean(r_θ * A_CAPO, clip(r_θ, 1-ε, 1+ε) * A_CAPO)
      L_sim = temporal_similarity_loss(s, f_coarse)  # Eq.18-19
      L_con = concentration_loss(α, β)                # Eq.20
      L_alloc = L_θ + λ_sim * L_sim + λ_con * L_con
      θ = θ - lr * ∇L_alloc

      # Step 6: 更新 Backbone（可选，token-level PPO）
      if update_backbone:  # ResAdapt-RL 模式
          r_φ = πφ(y_j | ...) / πφ_old(y_j | ...)
          L_φ = -mean(min(r_φ * A, clip(r_φ, 1-ε, 1+ε) * A))
          φ = φ - lr * ∇L_φ
  ```

  张量计算示例（forward pass，T=32 frames, H=W=448, P=14）：
  - Vanilla: N0 = 32 × 32 × 32 = 32768 visual tokens
  - ResAdapt (ρ=0.11, s̄≈0.33): N_adapt = 32768 × 0.11 ≈ 3604 tokens
  - Backbone attention FLOPs reduction: 0.11² ≈ 0.012, 即约 83× 加速
  - Allocator overhead: <3% of total FLOPs (Lpred=4, Dpred=1024 vs Lmllm=28, Dmllm=3584)
