## LongLive__Real-time_Interactive_Long_Video_Generation

- baseline方法是什么？
  Baseline 分为两类：(1) **扩散模型**（Wan2.1, SkyReels-V2, LTX-Video）—— 基于 DiT 架构的双向注意力（bidirectional attention）视频生成模型。双向注意力使所有帧之间的注意力关系为非因果（non-causal），导致 KV cache 机制无法使用，每生成一个视频片段必须重新计算全部帧的注意力。例如 SkyReels-V2 生成 60s 视频在单 H100 上需要约 50 分钟。(2) **自回归（AR）模型**（Self-Forcing, CausVid）—— 基于因果注意力（causal attention）的 frame-level 或 chunk-wise AR 视频生成模型。因果注意力天然支持 KV cache 加速推理，但由于训练长视频成本高，普遍采用 train-short-test-long 策略（仅在短视频上训练，在 rollout 长视频时用模型自己的输出做上下文），导致误差累积、内容漂移和一致性下降。

  Baseline（以 Self-Forcing 为例）全栈执行例子：
  - **算法层**：Wan2.1-T2V-1.3B (DiT) → 适配为 chunk-wise causal AR 模型（ODE initialization + DMD distillation）→ 训练仅 5s clips（train-short）→ 推理时通过滚动式 KV cache rollout 长视频（test-long）。Cross-attention: visual tokens Q attend to text prompt K/V。Self-attention: causal mask，生成帧的 KV 缓存。（缺陷：train-short-test-long 导致随视频变长质量下降；prompt 切换时若丢弃 KV cache 导致视觉断裂，保留 KV cache 导致 prompt 不跟随）。
  - **系统框架层**：PyTorch + HuggingFace Diffusers。无多请求 Serving 框架修改。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：标准 FlashAttention for causal self-attention。
  - **硬件架构层**：NVIDIA H100 GPU。推理时 dense causal attention 复杂度 O(L²)（L=总帧数），生成 180s 视频需处理超百万 token（以 Wan2.1 参考）。

  Baseline 的核心缺陷：
  1. **扩散模型中双向注意力禁止 KV cache**：每步推理需重算全部帧注意力，延迟随视频长度平方增长（O(L²)），导致长视频生成极慢（SkyReels-V2 ~50min/60s）。
  2. **AR 模型 train-short-test-long 不匹配**：训练仅在短视频上进行，推理时长 rollout 中模型输出不断作为自身输入，误差累积使上下文逐渐劣化，内容一致性随时间下降。
  3. **Prompt 切换时 KV cache 困境**：丢弃 KV cache → 视觉断裂、时间不连续；保留 KV cache → 旧 prompt 语义残留在 cache 中导致新 prompt 延迟响应或不跟随。
  4. **无有效的长视频高效推理策略**：dense causal attention 的 O(L²) 计算不可持续。此前 attention sink 在视频模型中被报告为无效（Self-Forcing 验证），因长期 rollout collapse 使得 sink 失去锚定作用。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  LongLive 通过三个核心设计系统性地解决上述缺陷：

  (1) **KV-recache** → 解决 Prompt 切换困境。在 prompt 切换边界，重新用已生成视频前缀当视觉上下文配对新 prompt 计算 KV cache（单次 forward pass 过交叉注意力层），清除旧 prompt 残留语义（cross-attention 中旧 prompt embedding 被替换）但保留自注意力中的运动与视觉连续性信号。训练时同步集成 recache（teacher 也接收新 prompt 做 DMD 监督），消除 train-inference mismatch。多 switch 泛化：推理时 n+1 个 prompt 有 n 个 switch 边界，每个边界执行一次 recache 即可。

  (2) **Streaming Long Tuning** → 解决 train-short-test-long 不匹配。每次 iteration 基于上一 iteration 的 KV cache 滚动生成下一个 5s clip（而非重新采样），仅对当前 clip 计算 DMD loss（teacher=Wan2.1-T2V-14B 对每个 5s clip 独立监督，确保 teacher 在自身能力范围内），gradient 只流经当前 clip（detach 历史帧梯度）。这直接将模型暴露于长 rollout 中自己生成的退化帧，训练即推理条件，使模型学会在长序列中自我纠错、抑制误差累积。同时显存仅按 clip 时长控制（O(clip) 而非 O(full_video)），避免 naive long tuning 的 OOM。

  (3) **Short Window Attention + Frame Sink** → 解决长视频推理效率。推理时注意力仅作用于最近 W 帧（如 W=9 latent frames）+ 永久保留的首帧 chunk（S=3 sink tokens），注意力复杂度从 O(L²) 降至 O(W+S+T)。Frame sink 仅在 streaming long tuning 解决了长期 rollout collapse 后才生效——作为全局语义锚点，将场景身份、色调、风格等持久信息缓存于 sink token 中。Train-test 对齐：在 streaming long tuning 中同样使用 short window + frame sink，resident KV size = O(W+T+S) 不随视频长度增长，避免 OOM。

  对比 baseline 的全栈执行例子（LongLive, 832×480, 1.3B, single H100）：

  - **算法层**：Wan2.1-T2V-1.3B (DiT) → chunk-wise causal AR 适配（ODE init + DMD, short window W=9, frame sink S=3）→ Streaming Long Tuning (60s, LoRA rank=256, 350M trainable)：每次 rollout 5s clip → [KV cache] extend next 5s → DMD loss only on current clip → repeat 至 60s → prompt switch: KV-recache (对已生成前缀用新 prompt 重算 KV) → 继续 rollout。（训练 3000 iters, 64 GPU × 12h = 32 GPU-days）
  - **系统框架层**：PyTorch + DMD pipeline。无多请求 Serving 框架修改。LoRA 微调 27% 参数。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：标准 FlashAttention with causal mask。Short window attention 通过 attention mask 实现（仅 mask 掉 window 外的 KV 位置），与 FlashAttention 兼容。Sink tokens 实现为 KV 序列的前缀拼接。
  - **硬件架构层**：NVIDIA H100 GPU。推理吞吐 20.7 FPS（vs Self-Forcing 17.0 / chunk-wise，8.9 / frame-wise）。Short window + sink 降低端到端计算时间 28%，峰值显存 17%（on H100）。支持 240s 视频生成 on single H100。INT8 PTQ: 2.7GB → 1.4GB (1.9×), 5090 GPU 上 16.4 FPS。

  LongLive 与 baseline 的核心差异：
  | 维度 | Baseline (Self-Forcing et al.) | LongLive |
  |------|-------------------------------|----------|
  | 训练策略 | train-short-test-long（5s） | train-long-test-long（streaming 60s/240s） |
  | Prompt 切换 | KV cache 全弃或全留 | KV-recache（单次重算，清除旧语义保留视觉） |
  | 长视频推理 | dense causal attn O(L²) | short window + frame sink O(W+T+S) |
  | Frame sink | 无效（因长期 collapse） | 有效（streaming tuning 解决 collapse） |
  | Train-test 对齐 | 不匹配（仅短期监督） | 完全对齐（training 模拟 rollout + same window） |
