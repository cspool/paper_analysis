## VideoRoPE: What Makes for Good Video Rotary Position Embedding

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：**VideoRoPE**，一种为 Video LLM 设计的 3D Rotary Position Embedding 策略，在传统 RoPE 基础上引入三个关键组件：
  (1) **Low-frequency Temporal Allocation (LTA)**：将高维度（对应低频率 β^{-2n/d}、更长单调区间）分配给时间维度 t，将低维度（对应高频率、捕捉局部关系）分配给空间维度 x 和 y。具体而言，d=128 维中，x 和 y 各占 48 维并交叉排列，t 占最后 32 维（高维、低频）。这与 M-RoPE（t 占前 32 维即高频）相反。低频时间分配使时间维度的 position embedding 避免周期性振荡（periodic oscillation），从而抑制 V-NIAH-D 中 distractor 的误导。
  (2) **Diagonal Layout (DL)**：将整个 multimodal 输入沿对角线排列。第 τ 帧中心 patch 的 3D 坐标为 (τ,τ,τ)，其他 patch 按 w-W/2、h-H/2 偏移。这保证了 spatial symmetry：preceding text end 到 visual start 的距离 ≈ visual end 到 subsequent text start 的距离，简化学习过程并减少输入顺序偏置。
  (3) **Adjustable Temporal Spacing (ATS)**：引入可调缩放因子 δ 对 temporal index 进行缩放。视频帧间 temporal spacing = δ，而文本 token 间 spacing = 1。公式：对于第 τ 个 visual token（τ∈[T_s, T_s+T_v)），temporal index = T_s + δ(τ-T_s)，spatial index = T_s + δ(τ-T_s) + w - W/2（x 维度），h-H/2（y 维度）。δ=2 效果最优。

  实验比较：对比以下 RoPE 变体在相同 fine-tuning 条件下（统一用 Qwen2-VL-7B ViT + Qwen2-7B LLM 初始化）：
  - **Vanilla RoPE** (Su et al., 2024)：1D RoPE，直接 flatten 所有 token
  - **TAD-RoPE** (Gao et al., 2024)：1D RoPE 适配，引入 image/text token 不同 step offset
  - **M-RoPE** (Wang et al., 2024a)：3D RoPE，t/x/y 维度各占 32/48/48，t 占低维（高频）

- 硬件平台是什么，配置是什么。
  **NVIDIA A100 GPU**：fine-tuning 共使用 704 A100 GPU hours。Batch size=128，cosine scheduler，learning rate=1e-5，warmup ratio=1e-2。训练 context window=8192 tokens。推理时使用 **vLLM** 框架支持超过 32K token 长序列，128K tokens 推理亦通过 vLLM Server-API 模式实现。

- 模型是什么。数据集和bench分别是什么。
  模型：基于 **Qwen2-VL-7B** 的 Vision Transformer + **Qwen2-7B** (Yang et al., 2024a) 的 LLM（含 Vanilla RoPE）。Fine-tuning 时替换为 VideoRoPE 处理视频时空信息。视频采样 2 FPS，最多 128 帧，动态分辨率调整保持恒定 token 数。head dimension d=128（64 个旋转角对）。

  训练数据集：**LLaVA-Video-178K** (Zhang et al., 2024e) 子集。从~178K 视频、~5M QA pairs 中随机选择 136K 个 <2 分钟的视频 + 18K 个 2-3 分钟的视频，共约 1.3M QA pairs。

  Benchmarks：
  - **长视频理解**：LongVideoBench（8秒-1小时，跨帧推理), MLVU（3分钟-2小时，7 个多选题任务：Topic Reasoning, Anomaly Recognition, Needle QA, Ego Reasoning, Plot QA, Action Order, Action Count), Video-MME（11秒-60分钟，6 个视觉领域 30 个子领域）
  - **长视频检索**：V-NIAH（随机位置插入 needle 图片于 3000 帧 haystack，每 200 帧检查至 3000 帧）, V-NIAH-D（在 V-NIAH 基础上，距 needle 200 帧处周期性插入 semantic distractor）
  - **视频幻觉**：VideoHallucer（包括内在幻觉：Object-Relation, Temporal, Semantic Detail；外在幻觉：Factual, Non-factual）
  - 各 benchmark 在所有 4 个 context length（8K/16K/32K/64K）下评估

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源：代码在 https://github.com/Wiselnn570/VideoRoPE。

  算法 pipeline（完整前向过程）：
  ```
  输入: 文本 T_pre (Ts tokens) + 视频 V (Tv frames, 每帧 W×H patches) + 文本 T_post (Te tokens)
  d = 128 (head dimension), δ = 2 (ATS scaling factor)

  Step 1: 计算 3D position indices (t,x,y) for each token
  For τ = 0 to (Ts + Tv + Te - 1):
    if τ < Ts:                    # preceding text
      (t,x,y) = (τ, τ, τ)
    elif τ < Ts + Tv:             # video frames
      f_idx = τ - Ts              # frame index within video
      center_t = Ts + δ * f_idx
      (t,x,y) = (center_t, center_t + w - W/2, center_t + h - H/2)
    else:                         # subsequent text
      (t,x,y) = (τ + (δ-1)Tv, τ + (δ-1)Tv, τ + (δ-1)Tv)

  Step 2: 构建 RoPE rotation matrix
  θ_n = β^{-2n/d}, n = 0,1,...,d/2-1 = 63  (β base frequency, default 10000)
  Dimension allocation:
    - dims [0,47]:   spatial x, y interleaved (x0,y0,x1,y1,...)
    - dims [48,63]:  temporal t (last 16 angle pairs = highest dimensions)
  M-RoPE 对比: dims [0,15] for t, [16,39] for x, [40,63] for y

  Step 3: 计算 attention scores
  For query vector q at position (t1,x1,y1) and key vector k at (t2,x2,y2):
    A = q^T R_{Δt,Δx,Δy} k
  where R_{Δt,Δx,Δy} is a block-diagonal matrix:
    - Rows 0..47:  rotation by θ_n·Δspatial (x or y depending on interleave)
    - Rows 48..63: rotation by θ_n·Δt (temporal, low-frequency)
  Δt = t1 - t2, Δx = x1 - x2, Δy = y1 - y2

  Step 4: Standard attention + FFN (no other changes to Transformer)
  ```

  **关键差异对比 M-RoPE**：
  | 方面 | M-RoPE | VideoRoPE |
  |------|--------|-----------|
  | 时间维度频率 | 高频 (低维 dims 0-31) | 低频 (高维 dims 96-127) |
  | 振荡行为 | 短单调区间，远距离位置可产生相同 embedding（hash collision） | 宽单调区间，无 hash collision |
  | 空间对称性 | 无（visual token 固定停在 (W-1,H-1) 角） | 有（Diagonal Layout 保证） |
  | 时间缩放 | 无 | ATS δ=2 |
  | x/y 排列 | 顺序 x...y... | 交叉 x,y,x,y,... |
  | V-NIAH Acc | 78.67 | 91.11 (+12.44) |
  | V-NIAH-D Acc | 74.67 | 87.11 (+12.44) |
  | LongVideoBench 64K | 54.35 | 57.26 (+2.91) |
  | MLVU 64K | 61.10 | 65.56 (+4.46) |
  | Video-MME 64K | 59.67 | 61.33 (+1.66) |
  | LongVideoBench 128K | 51.45 | 55.64 (+4.19) |

  **Ablation 结果**（Table 5, 64K context）：
  Baseline (M-RoPE) → +DL → +DL&LTA → +DL&LTA&ATS
  LongVideoBench: 54.35 → 53.63 → 55.60 → 57.26
  MLVU:           61.10 → 62.75 → 63.26 → 65.56
