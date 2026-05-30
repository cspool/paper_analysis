## LongLive__Real-time_Interactive_Long_Video_Generation

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：LongLive 是一个面向实时交互式长视频生成的 frame-level 自回归（AR）框架，基于 Wan2.1-T2V-1.3B 微调而来。核心算法创新有三：(1) **KV-recache** —— 在用户 prompt 切换时，利用已生成帧作为视觉上下文、配对新 prompt 重新计算 KV cache，在交叉注意力层中清除旧 prompt 的残留言语义，同时保留运动与视觉线索以维持时间连续性，支持训练时监督（teacher 同步接收新 prompt 输出监督信号）和推理时多 prompt 切换；(2) **Streaming Long Tuning** —— 将训练-推理对齐为"train-long-test-long"，通过滚动式长序列 rollout：每次迭代基于已存储的 KV cache 编码历史帧、生成下一个 5s 片段，仅对当前片段计算 DMD loss（Wan2.1-T2V-14B 作 teacher），detach 历史帧梯度以防止 OOM，滚动扩展至预设最大长度（60s/240s）；(3) **Short Window Attention + Frame Sink** —— 推理时限制注意力窗口为短窗口（如 9 latent frames），将首帧 chunk 固定为全局 sink token 永久保留在 KV cache 中不被驱逐，恢复长程一致性同时降低计算：注意力复杂度与窗口大小成正比而非总序列长度，训练时同步集成以对齐 train-test 行为。

  实验比较：(a) **短视频生成** —— VBench 官方 prompt 评估 5s 生成，对比 LTX-Video (1.9B)、Wan2.1 (1.3B)、SkyReels-V2 (1.3B)、MAGI-1 (4.5B)、CausVid (1.3B)、NOVA (0.6B)、Pyramid Flow (2B)、Self-Forcing (chunk-wise/frame-wise)；(b) **长视频生成（单 prompt）** —— VBench-Long 官方 prompt 评估 30s 生成，对比 SkyReels-V2、FramePack、Self-Forcing；(c) **交互式长视频生成（多 prompt）** —— 自建 160 个 60s 交互式测试视频，每视频 6 个连续 10s prompt，评估 VBench-Long 长程质量维度 + CLIP Score 语义对齐，对比 SkyReels-V2、Self-Forcing；(d) **KV-recache 消融** —— 10s 单 switch 视频上对比 No KV cache / KV cache / KV recache 三种策略的视觉一致性与 CLIP Score；(e) **短窗口注意力 + Frame Sink 消融** —— 10s 视频上 window sizes (3/6/9/12/15/18/21/24/27 latent frames) + 9-local+3-sink 的 VBench-Long 一致性对比；(f) **INT8 量化** —— PTQ 量化压缩 2.7GB→1.4GB，VBench 质量对比；(g) **LoRA 消融** —— rank 32/64/128/256/512 vs Full Model 的 VBench-Long 分数；(h) **用户调研** —— 26 人 48 题四维度（总体/运动质量/指令跟随/视觉质量）LongLive vs Self-Forcing；(i) 在 SANA-Video（线性注意力 AR 模型）上实现 LongLive 以验证加速能力。

- 硬件平台是什么，配置是什么。
  训练：64× NVIDIA H100 GPU，每 GPU 1 sample（global batch size=64），约 12 小时。基于 Wan2.1-T2V-1.3B 预训练权重，使用 DMD pipeline 初始化（ODE initialization + short-window attention + frame-sink tokens），再进行 streaming long tuning（Algorithm 1），LoRA rank=256 微调（约 350M trainable params, ~27% of 1.3B）。优化器 AdamW：actor lr=1.0×10⁻⁵ (β₁=0.0, β₂=0.999)，critic lr=2.0×10⁻⁶ (β₁=0.0, β₂=0.999)。EMA decay=0.99，自 200 步开始。最大序列长度对应于目标推理时域（60s 或 240s），60s 设定训练 3000 iterations（32 GPU-days）。推理：单张 NVIDIA H100 GPU，支持 20.7 FPS, 最大 240 秒视频在单卡上。INT8 量化推理在单张 NVIDIA 5090 GPU 上测试。
  推理效率：dense causal attention（长期）复杂度 O(L²)；short-window attention 复杂度 O(W·L)，W=9 为窗口大小；窗口大小 9 + 3 sink frames 的组合下，端到端计算时间降低 28%，峰值显存降低 17%（vs 长窗口 baseline on single H100）。

- 模型是什么。数据集和bench分别是什么。
  模型：基础模型 Wan2.1-T2V-1.3B（DiT 架构，5s clips @ 16 FPS, 832×480 resolution），通过 self-forcing DMD pipeline 适配为 chunk-wise 因果注意力 AR 模型（chunk size=3 latent frames，local attention window=9 frames，首个 chunk 为 sink token），再经 streaming long tuning (LoRA rank=256, ~350M trainable params)。Teacher model: Wan2.1-T2V-14B。额外在 SANA-Video（线性注意力 AR 模型）上实现 LongLive 验证通用性。
  训练数据：使用 VidProM 的 text prompts，通过 Qwen2-72B-Instruct 为每个原始 prompt 生成 follow-up prompt 构造 prompt-switch 数据集。不引入额外视频数据，采用 self-supervised DMD 蒸馏方案（teacher supervision 替代 ground truth）。
  Benchmarks：(1) VBench —— 标准短视频评估（5s clips），指标 Total/Quality/Semantic scores；(2) VBench-Long —— 长视频评估（30s），指标 Total/Quality/Semantic + Subject Consistency/Background Consistency/Motion Smoothness/Aesthetic Quality/Imaging Quality；(3) 自建 interactive 60s validation set —— 160 videos × 6 sequential 10s prompts，长程质量（VBench-Long dimensions）+ 逐片段 CLIP Score；(4) 吞吐量 —— Throughput (FPS) on single H100 GPU；(5) 用户调研 —— 26 participants × 48 questions × 4 dimensions（Overall/Motion Quality/Instruction Following/Visual Quality）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址（代码+模型+Demo）：https://github.com/NVlabs/LongLive，https://huggingface.co/Efficient-Large-Model/LongLive-1.3B

  算法 pipeline 伪代码（LongLive 训练与推理全流程）：
  ```
  # ==========================================
  # Algorithm 1: Streaming Long Tuning
  # ==========================================
  # 输入: Causal video generator G_θ, prompt set P
  # 超参数: l_video (max length, e.g., 60s), l_clip=5s, switch s
  while not converged:
      C ← []  # KV cache
      l ← 0   # current video length
      Sample (p, p_next) from P  # pair of original + follow-up prompt
      Sample switch index s ∈ {1,...,⌊l_video/l_clip⌋-1}
      s ← s × l_clip

      while l < l_video:
          p_active ← p if l < s else p_next

          if l等于s:  # prompt switch boundary
              C ← recache(G_θ, x, C, p_active)  # KV-recache
          
          x ← generate_next_clip(G_θ, C, p_active)  # rollout 5s clip
          L ← DMD_Loss(G_θ, x, p_active)  # distillation loss
          L.backward()
          update θ
          l ← l + l_clip

          if l >= l_video:
              C ← []; l ← 0
              Resample (p, p_next) and s

  # ==========================================
  # KV-recache 过程
  # ==========================================
  def recache(G_θ, video_prefix x, old_cache C, prompt p_new):
      # 对已生成的视频前缀 x，用新 prompt p_new 重新计算 KV cache
      # 清除旧 prompt 语义，保留视觉运动连续性
      C_new = []
      for each frame f_i in x:
          # cross-attn: q from frame, k/v from p_new
          # self-attn: 所有帧间的 causal/masked attention
          kv = G_θ.forward_cache(f_i, C_new, p_new)
          C_new.append(kv)
      return C_new  # refreshed cache; subsequent steps use this

  # ==========================================
  # Algorithm 2: Interactive Inference (多 prompt 切换)
  # ==========================================
  # 输入: Prompt sequence P=[p0,...,pn], switch indices S=[s1,...,sn]
  # 输出: 长视频 x (N frames, T diffusion steps per frame)
  x ← [];  C ← []
  p_active ← P.pop(0)

  for i = 1 to N:
      if i ∈ S:
          p_active ← P.pop(0)
          C ← recache(G_θ, x, C, p_active)  # prompt switch: recache

      # 每帧的扩散去噪 (T 步，causal AR)
      x_T^i ~ N(0, I)
      for j = T down to 1:
          \hat{x}_0^i ← G_θ(x_{tj}^i; tj, C, p_active)  # single-step denoiser
          if j == 1:
              x.append(\hat{x}_0^i)           # 输出帧
              C ← G_θ^{KV}(\hat{x}_0^i, 0, C, p_active)  # 更新 KV cache
          else:
              ϵ ~ N(0, I)
              x_{t_{j-1}}^i ← Ψ(\hat{x}_0^i, ϵ, t_{j-1})  # 加噪回退
  return x

  # ==========================================
  # Short Window Attention + Frame Sink
  # ==========================================
  # 训练/推理配置:
  # W = 9 (local attention window in latent frames)
  # T = 5s clip = supervised clip length
  # S = 3 (sink tokens: first chunk of video)
  # Per step KV cache: O(W+T+S) NOT growing with total video length

  # Attention mechanism (per self-attention layer):
  # K_local = K[last_W_frames]     # sliding window keys
  # V_local = V[last_W_frames]     # sliding window values
  # K_sink = K[first_S_frames]     # permanent global sink keys
  # V_sink = V[first_S_frames]     # permanent global sink values
  # K_effective = concat([K_sink, K_local])
  # V_effective = concat([V_sink, V_local])
  # output = softmax(Q @ K_effective^T / sqrt(d)) @ V_effective
  ```

  关键张量计算与设计要点：
  - **DiT 架构**：Wan2.1-T2V-1.3B，cross-attention + self-attention 交替。Cross-attention: Q 来自 visual tokens，K/V 来自 text embeddings (prompt)。Self-attention: causal mask，视觉帧间注意力。
  - **KV cache 结构**：per layer, K ∈ R^{seq×d_head×n_heads}, V ∈ R^{seq×d_head×n_heads}。Short window 限制 seq ≤ W+S+T，而非全长。
  - **KV-recache 量化分析**：对 10s video (single switch), recaching 相比 no-recache 仅增加约 6% 时间（因 recache 只处理前缀帧一次，非逐帧重算）。
  - **Streaming Long Tuning**：每 iteration 仅当前 5s clip 计算梯度（detach 历史帧），显存上限 = clip duration 而非 full sequence，解决 naive long tuning 的 OOM 问题。
  - **DMD (Distribution Matching Distillation)**：学生 G_θ 用噪声帧和 teacher (Wan2.1-T2V-14B) 的分布指导蒸馏，critic network 辅助区分 real/fake 分布。
  - **LoRA 配置**：rank=256, 350M trainable / 1.3B total = ~27%, 相比全微调节省 73% 参数/优化器状态。
  - **INT8 PTQ**：Wan2.1 量化至 INT8, 模型大小 2.7GB → 1.4GB (1.9× 压缩), 吞吐提升 1.3× (12.6→16.4 FPS on 5090)。
