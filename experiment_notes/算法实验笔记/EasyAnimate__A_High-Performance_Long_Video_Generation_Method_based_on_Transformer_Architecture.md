## EasyAnimate__A_High-Performance_Long_Video_Generation_Method_based_on_Transformer_Architecture

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：EasyAnimate 是一个基于 Diffusion Transformer 的高性能视频生成框架，核心算法创新包括四个层面：(1) **Hybrid Windows Attention** —— 提出多方向滑动窗口注意力（Multidirectional Sliding Window Attention），将注意力头分为 6 组，分别在 fhw、fwh、hfw、hwf、wfh、whf 六种方向上执行滑动窗口注意力，仅需一次注意力计算（相较于 spatial-temporal decoupled attention 的多次计算），然后与 3D full attention 层交替排布（window attention 放在中间层 12-36），在保持生成质量的同时降低计算复杂度。基于 FlashAttention 实现高效计算。(2) **Reward Backpropagation 后训练** —— 使用可微分 reward model（HPSv2.1 + MPS 组合最优）通过 LoRA 微调 DiT 参数，优化目标为最大化采样视频的经验 reward，backprop 步数 K=10（不再是 DDPM 的 K=1，因 rectified flow 下梯度更小），解码帧数 F=1（避免多帧 reward 导致 dynamics 损失和 reward hacking）。(3) **Training with Token Length (TTL)** —— 将相似 token length 的视频分组到同一训练 step，解决不同分辨率和帧数视频训练时 GPU 利用率不均的问题，每次迭代训练的 token 数提升 120.91%。(4) **MLLM 文本编码器** —— 使用 Qwen2-VL-7B（取倒数第二层 hidden state）替代 CLIP/T5，通过 RMSNorm 归一化文本特征并过 FC 层减少与视频特征的 L2 norm 差异，支持多语言输入。

  实验比较：(a) VBench benchmark —— 对比 AnimateDiff-V2、VideoCrafter-2.0、OpenSora V1.2、OpenSoraPlan V1.3、CogVideoX1.5-5B、CogVideoX-5B、HunyuanVideo、Jimeng、Vidu、Gen-3、MiniMax-01、Sora，EasyAnimate Total Score 83.42，Aesthetic Quality 69.48（所有模型中最高）；(b) 人类评估 —— 100 prompts 来自 T2V-CompBench，对比 CogVideoX 和 HunyuanVideo，EasyAnimate 在 Quality（50.31%）、Semantic（44.09%）、Physics（45.03%）三个维度均获最高偏好；(c) 消融 —— 文本编码器对比（T5+CLIP vs Qwen2-VL）、Window Attention 位置/窗口大小/方向数消融（FVD score）、Reward Model 消融（Aesthetic/MPS/HPSv2/组合）、Backprop 步数 K 消融、解码帧数 F 消融；(d) 效率对比 —— Hybrid vs Full Attention 训练/推理延迟对比（1024 分辨率下训练加速 22.39%，推理加速 25.53%）。

- 硬件平台是什么，配置是什么。
  训练：多 GPU 集群（A100 GPU，如单卡 A100 上 12B 模型生成 1024×1024×49 frames 约 30 分钟）。benchmark 测试：A100 GPU（Table 1 的 speed test on A100 GPUs）。具体 GPU 数量论文未明确说明。

- 模型是什么。数据集和bench分别是什么。
  模型：Diffusion Transformer (DiT) based on MMDiT 架构，含 text encoder (Qwen2-VL-7B)、denoising DiT、3D Causal VAE。支持 text-to-video、image-to-video、video-to-video、inpaint、control（Canny/Pose/Depth/MLSD/trajectory/camera）等多种模式。模型规模有 7B 和 12B 版本。使用 rectified flow 采样（非 DDPM），3D RoPE 位置编码（各维度分配 3/8、3/8、2/8 的 hidden channels）。
  数据集：(1) 视频源 —— Panda-70M、InternVid、MiraData、Pexels 及内部数据；(2) 图像源 —— JourneyDB（美学过滤）+ ALLaVa caption 标注；(3) 数据规模 —— ~34M video-text pairs + ~3M image-text pairs（Pretrain: 33.72M videos + 2.87M images; Pretrain-HR: 25.10M + 2.87M; Finetune: 0.47M + 0.04M）。数据处理 pipeline：PySceneDetect 切分 → 三段式过滤（Aesthetic Score via SigLIP-based predictor、Text Score via CRAFT、Motion Score via Farneback optical flow + camera shake classifier）→ InternVL2-40B 生成 dense captions → LLama-3-70B 优化 + 生成 short captions → VideoCLIP-XL-v2 验证 caption-video 相似度。
  Benchmarks: VBench（主要，含 Total Score/Quality Score/Semantic Score/Aesthetic Quality/Subject Consistency/Spatial Relationship/Object Class/Scene 维度）、T2V-CompBench（人类评估 prompt 来源）、WebVid validation set（消融实验 FVD 计算，1000 videos）、FVD 指标。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址：https://github.com/aigc-apps/EasyAnimate（Apache 2.0 License），模型权重发布于 HuggingFace 和 ModelScope。最新版本 V5.1（2025.01.21）支持 diffusers 集成。

  算法 pipeline 伪代码（核心训练/推理流程）：

  ```
  # ==========================================
  # 1. Hybrid Windows Attention (核心算子)
  # ==========================================
  # 输入: video tokens v ∈ R^{F×H×W×D}, F=frames, H×W=spatial, D=hidden_dim
  # 将 token 序列展平为 seq_len = F*H*W

  def multidirectional_sliding_window_attention(Q, K, V, num_heads, window_size):
      """
      头分组 + 多方向滑动窗口注意力
      sliding_dirs = [fhw, fwh, hfw, hwf, wfh, whf]
      每个方向对应一种维度重排:
        fhw: 先 frames 后 height 后 width (默认)
        fwh: 先 frames 后 width 后 height
        hfw: 先 height 后 frames 后 width
        ...
      """
      # 步骤1: 将 Q/K/V 按 heads 分为 6 组
      Qs = split(Q into 6 head_groups)  # 每组 heads//6 个注意力头
      Ks = split(K into 6 head_groups)
      Vs = split(V into 6 head_groups)

      # 步骤2: 各组按各自方向重排 token 顺序
      for i in 1..6:
          Qs[i] = rearrange(Qs[i], sliding_dirs[i])  # e.g. fhw -> fwh
          Ks[i] = rearrange(Ks[i], sliding_dirs[i])
          Vs[i] = rearrange(Vs[i], sliding_dirs[i])

      # 步骤3: 合并后调用标准 FlashAttention (带 sliding_window 参数)
      Q = concat(Qs); K = concat(Ks); V = concat(Vs)
      output = FlashAttention(Q, K, V, window_size=(window, window))

      # 步骤4: 将各组恢复原始 token 顺序
      Os = split(output into 6 head_groups)
      for i in 1..6:
          Os[i] = rearrange(Os[i], inverse_dirs[i])  # e.g. fwh -> fhw
      return concat(Os)

  # Hybrid Windows Attention 层排列
  class EasyAnimateTransformerBlock(layer_idx):
      if layer_idx in [12, 36):  # 中间层使用 window attention
          attn = MultidirectionalSlidingWindowAttention(window_size=H*W)
      else:  # 浅层和深层使用 full attention
          attn = Full3DAttention()

  # ==========================================
  # 2. Training with Token Length (训练策略)
  # ==========================================
  # 每次迭代选择 token 数相近的样本组成 batch
  def training_with_token_length(video_pool):
      # 按 token_length = (H*W*F) / (patch_size^2 * temporal_compression) 分组
      # 例如: 512^2 × 49 frames 和 768^2 × 21 frames 有相似 token 数 -> 同组训练
      batches = group_by_token_length(video_pool)
      for batch in batches:
          loss = rectified_flow_loss(model(batch))
          loss.backward()

  # ==========================================
  # 3. Reward Backpropagation (后训练)
  # ==========================================
  def reward_backpropagation(dit_model, vae, reward_models, prompts, K=10, F=1):
      """
      dit_model: LoRA 微调的 DiT 参数
      reward_models: [HPSv2.1, MPS] 可微分 reward 模型
      K: backprop 的 denoising 步数 (K=10 for rectified flow)
      F: 用于 reward 计算的解码帧数 (F=1, 仅第一帧)
      """
      for prompt in prompts:
          # 1. 从 T 到 K 步: 采样但不计算梯度 (detach)
          z_T = randn()  # 纯噪声
          c = qwen2vl_encode(prompt)  # Qwen2-VL-7B 文本编码
          for t in range(T, K, -1):
              z_{t-1} = flow_matching_step(z_t, c).detach()

          # 2. 从 K 到 0 步: 计算梯度
          for t in range(K, 0, -1):
              z_{t-1} = flow_matching_step(z_t, c)  # 保留计算图

          # 3. 3D Causal VAE 解码: 取前 F 帧计算 reward
          video_frames = vae.decode(z_0)[:F]  # F=1, 因果 VAE 首帧可解码全部

          # 4. 多 reward model 加权
          reward = HPSv2(video_frames, prompt) + MPS(video_frames, prompt)
          loss = -reward  # 最大化 reward -> 最小化负 reward

          # 5. 更新 LoRA 参数
          loss.backward()
          optimizer.step()

  # ==========================================
  # 4. 完整训练 pipeline
  # ==========================================
  # Stage I:   VAE-adapt (SAM 图像数据对齐 VAE 和 DiT)
  # Stage II:  Pretrain (256^2×49 tokens -> 512^2×49 tokens)
  # Stage III: Finetune (512^2×49 -> 1024^2×49, image-to-video)
  # Stage IV:  Post-training (Reward Backpropagation + LoRA)
  ```

  张量计算示例（Hybrid vs Full Attention @ 1024×1024×49 frames, 12B model）:

  ```
  # Full Attention 复杂度:
  # 假设 VAE 8x 空间压缩, patch_size=2, temporal_compression=4
  # seq_len = F * H * W / (patch_size^2 * temporal_compression)
  #         = 49 * 64 * 64 / (4 * 4) = 49 * 256 = 12544 tokens
  # FLOPs_full = 2 * seq_len^2 * d_model ≈ 2 * 12544^2 * 5120 ≈ 1.6e12 FLOPs/layer

  # Hybrid Window Attention 复杂度 (window_size = H*W spatial tokens ≈ 256):
  # FLOPs_window = 2 * seq_len * window * d_model ≈ 2 * 12544 * 256 * 5120 ≈ 3.3e10 FLOPs/layer
  # 约 48x 减少 per window-attention layer
  ```
