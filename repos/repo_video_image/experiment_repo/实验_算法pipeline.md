## VisionSelector: End-to-End Learnable Visual Token Compression for Efficient Multimodal LLMs

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：**VisionSelector**，一个轻量级、可端到端学习的视觉 token 压缩框架，由三个核心组件构成：
  (1) **Learnable Importance Scorer (LIS)**：通过两层线性投影（W_q, W_k）将输入视觉 token V ∈ R^{N×D} 投影为 Q 和 K（维度 d），计算简化自注意力矩阵 A = QK^T/√d，每个 token 的重要性得分 s_i = (1/N)·Σ_j A_{ij}。利用全局 token 间交互信息评估相对重要性，而非依赖 MLLM 内部的预训练 attention map。仅 12.85M 可训练参数（Qwen2.5-VL-7B 上）。
  (2) **Differentiable Top-K Selection (DTS)**：训练时通过 sigmoid 连续松弛和二分搜索阈值 t，使 Σ σ(s_i+t) ≈ k 产生 soft mask M ∈ (0,1)^N，通过隐函数微分反向传播梯度 ∂L/∂s = v⊙g − (v^T g/Σ v_i)·v（其中 v_i = M_i(1−M_i)），实现端到端训练。推理时直接使用标准 Top-K 硬选择。
  (3) **Curriculum Annealing Strategy (CAS)**：总损失 L_total = L_CE + λ_t·L_constraint，其中 L_constraint = BCE(M_soft, M_hard) 引导 soft mask 向硬选择逼近。λ_t 从初始值 λ_start 线性增加到 λ_end，确保模型先学习任务再强化选择约束。
  VisionSelector 部署在 modality interface 与 LLM 之间：视觉编码器 → 投影器 → LIS（计算重要性得分）→ DTS（生成 soft/hard mask）→ V_pruned = M⊙V → 与 text embeddings 拼接送入 LLM。训练时仅更新 LIS 参数，冻结 MLLM backbone。在 20% 固定压缩率训练后，可在推理时泛化到任意压缩预算。

  实验比较：对比以下 baseline（统一使用 LMMs-Eval 框架评估）：
  - **FastV** (ECCV 2024)：基于 text→vision attention score 的剪枝
  - **PruMerge+** (ICCV 2025)：视觉编码器阶段的注意力稀疏 + KNN 聚类合并
  - **VisionZip** (CVPR 2025)：text-agnostic，基于末层 attention map 选 dominant tokens + 语义相似度合并
  - **DART** (EMNLP 2025)：基于余弦相似度识别 near-duplicate 组，每组仅保留一个代表 token
  - **DivPrune** (CVPR 2025)：建模为 Max-Min Diversity Problem，最大化保留子集的多样性
  - **Dynamic-LLaVA** (ICLR 2025)：基于 Gumbel-Softmax 的可训练图像预测器（额外对比实验在附录 A.1）

  实验在 10%/20%/30% 三种 token retention budgets 下评估，覆盖 9 个图像理解 + 4 个视频理解 benchmark，以及效率指标（GPU 内存、prefill time、E2E latency）。

- 硬件平台是什么，配置是什么。
  **8 × NVIDIA A800 GPUs (80GB)**，使用 Distributed Data Parallel + DeepSpeed ZeRO Stage 3 训练部署。训练约需 40 分钟（Qwen2.5-VL-7B）。推理时在视频任务 (MVBench, avg 6828 tokens) 评估效率：VisionSelector 内存降至 17.57 GB（baseline 25.97 GB 的 67.7%），prefill time 760.82 ms（1.86× speedup vs baseline 1413.34 ms），E2E latency 924.57 ms（1.74× speedup）。

- 模型是什么。数据集和bench分别是什么。
  模型：**Qwen2.5-VL-7B** (Bai et al., 2025)，额外验证 **Qwen2.5-VL-3B**（附录 A.5，4.00M 可训练参数）和 **LLaVA-OneVision-1.5-8B**（附录 A.6，16.87M 可训练参数）。LIS 投影维度 d=1792（Qwen2.5-VL-7B 的一半 hidden dim）、1024（3B）、2048（LLaVA-OV-1.5-8B）。

  训练数据：混合数据集来自 Cambrian-737K，包含 ChartQA（图表理解）、OCRVQA（文档 OCR）和 COCO 的 10% 随机采样（自然图像），共约 144K 样本。固定随机种子 42 确保复现。

  训练超参数：1 epoch，AdamW + cosine annealing LR scheduler，初始学习率 5e-5，0.03 epochs linear warmup，per-device batch size=16，gradient accumulation steps=4（effective global batch size=256），retention budget=20%，λ_start=0.1 → λ_end=2.0。

  Benchmarks（13 个）：
  - 图像理解（9）：TextVQA (Singh et al., 2019)、DocVQA (Mathew et al., 2021)、OCRBench (Liu et al., 2024b)、ChartQA (Masry et al., 2022)、AI2D (Kembhavi et al., 2016)、ScienceQA (Lu et al., 2022)、MME (Fu et al., 2024)、MMMU (Yue et al., 2024)、POPE (Li et al., 2023b)
  - 视频理解（4）：MVBench (Li et al., 2024)、SEEDBench (Li et al., 2023a)、VideoMME (Fu et al., 2025)、NeXT-QA (Xiao et al., 2021)

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  GitHub: https://github.com/JulietChoo/VisionSelector
  
  **完整算法 Pipeline（配伪代码）**：

  ```
  # === 阶段 1: 视觉编码与投影（冻结） ===
  # 给定输入图像/视频帧，视觉编码器输出 patch features
  # 经 PatchMerger + Projection → V ∈ R^{N×D}

  # === 阶段 2: Learnable Importance Scorer（仅训练此模块）===
  # 层归一化 + 两层线性投影
  V_norm = LayerNorm(V)            # R^{N×D}
  Q = V_norm @ W_q                 # R^{N×d}, W_q ∈ R^{D×d}, d=1792
  K = V_norm @ W_k                 # R^{N×d}, W_k ∈ R^{D×d}
  A = Q @ K.T / sqrt(d)            # R^{N×N}
  s = mean(A, dim=1)               # R^{N}, s_i = (1/N) * Σ_j A_{ij}

  # === 阶段 3: DiffTopK（训练）===
  def DiffTopK_forward(s, k):
      # 二分搜索阈值 t, 使 sum(sigmoid(s+t)) ≈ k
      lower = -max(s) - 10
      upper = -min(s) + 10
      for _ in range(64):
          mid = (lower + upper) / 2
          mask = (sum(sigmoid(s + mid)) < k)
          lower[mask] = mid[mask]
          upper[~mask] = mid[~mask]
      t = (lower + upper) / 2
      M_soft = sigmoid(s + t)       # ∈ (0,1)^{N}
      return M_soft

  def DiffTopK_backward(grad, s, t):
      v = sigmoid(s+t) * (1 - sigmoid(s+t))   # σ'(s+t)
      v_sum = sum(v)
      uv = grad * v
      uv_sum = sum(uv)
      grad_s = uv - (uv_sum / v_sum) * v       # 见论文公式(8)
      return grad_s

  # === 阶段 4: 训练目标 ===
  M_soft = DiffTopK_forward(s, k)
  V_pruned = M_soft ⊙ V             # element-wise, 抑制低分 token
  # V_pruned 与 text embeddings 拼接 → LLM forward
  L_CE = CrossEntropy(outputs, labels)        # 下游任务损失
  M_hard = standard_TopK(s, k)                # 硬 mask（one-hot）
  L_constraint = BCE(M_soft, M_hard)           # 引导极化
  λ_t = λ_start + (λ_end - λ_start) * min(t/t_total, 1.0)
  L_total = L_CE + λ_t * L_constraint
  L_total.backward()                           # 梯度通过 DiffTopK 传递至 s→LIS

  # === 阶段 5: 推理 ===
  # 训练完成后，移除 DiffTopK，使用标准 Top-K
  M_hard = TopK(s, k)               # 硬二元 mask
  V_pruned = M_hard ⊙ V             # 仅保留 top-k tokens
  # 送入 LLM 完成推理
  ```

  训练参数：仅 12.85M（Qwen2.5-VL-7B），占模型总参数的 0.18%。训练耗时约 40 分钟（8 A800 GPU）。推理时计算开销极低（LIS 仅两层线性投影 + QK^T → mean score → TopK），与 FlashAttention 兼容，无额外调度或 kernel 修改。

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

## VideoNSA: Native Sparse Attention Scales Video Understanding

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：**VideoNSA**，基于 Qwen2.5-VL-7B 的 learnable、hardware-aware 混合稀疏注意力机制，用于长视频理解。核心设计：
  (1) **Hybrid Attention**：在 LLM decoder 每层将输入 token 按 position ID 分为 vision tokens X_V 和 text tokens X_T。对 vision tokens 应用 NSA（Native Sparse Attention），对 text tokens 保持标准 GQA（Grouped Query Attention），以保留指令跟随能力。
  (2) **NSA 三支路结构**：
  - **Compression Branch (CMP)**：将连续 block 内的 keys 通过 learnable MLP φ 聚合为粗粒度 block-level 表示：K̃_t^cmp = {φ(K_{[id+1:id+m]}) | 0≤i<⌊(t-m)/d⌋}，m 为 block size，d 为 stride。
  - **Selection Branch (SLC)**：按 importance score p_t^slc 选择 top-n 重要 KV blocks：I_t = {i | rank(p_t^slc[i]) ≤ n}，K̃_t^slc = Cat({K_{[im'+1:(i+1)m']} | i∈I_t})。
  - **Sliding Window Branch (SWA)**：保留最近 w 个 KV pairs：K̃_t^swa = K_{t-w+1:t}。
  (3) **Dynamic Gating**：三支路输出通过 learnable gate（两层 MLP + sigmoid）加权求和：o_V = Σ_{c∈{cmp,slc,win}} g_t^c · Attn(q_t, K̃_t^c, Ṽ_t^c)。最终输出 o = [o_V; o_T]。
  (4) **Frame-aligned block partition**：block size s 设为每帧 token 数（64），block 内 token 取平均得到 block-level 表示，使得每个压缩 block 对应完整一帧的语义。

  实验比较：
  - **Baseline models**：Qwen2.5-VL-7B 用 dense FlashAttention（主 baseline）；量化模型 AWQ；训练无关 token compression（FastV, VisionZip, VScan）；训练无关 sparse attention（MInference, FlexPrefill, XAttention, Tri-Shape）；以及用相同数据 fine-tune 的 Qwen2.5-VL-7B-SFT。
  - **Benchmarks**：长视频理解（LongVideoBench, MLVU, TimeScope, LongTimeScope, LSDBench, VideoEvalPro），时序推理（Tomato），空间理解（VSIBench）。
  - **Ablation study**：三支路组合消融（单支路/两两组合/完整三支路）；稀疏权重迁移至 dense attention（Dense-NSA vs Dense-SFT vs VideoNSA）；context length scaling（36K→64K→128K tokens）；attention budget allocation 策略；gate 分布与 inter-head similarity 分析；attention sink 消融。

- 硬件平台是什么，配置是什么。
  **NVIDIA H100 GPU**：完整训练需要 4600 H100 GPU hours。使用 `torch.compile` 编译，PyTorch 框架。推理延迟测量覆盖 1K 到 128K context length。训练时 max context length 36K tokens，推理扩展到 128K tokens（模型上下文窗口上限）。

- 模型是什么。数据集和bench分别是什么。
  模型：**Qwen2.5-VL-7B** (Bai et al., 2025) 作为 backbone，包含 ViT vision encoder + MLP Projector + Qwen2.5-7B LLM decoder。LLM decoder 使用 GQA（28 query heads，4 shared KV heads）。使用 RoPE 位置编码。

  训练数据集：**LLaVA-Video-178K**（Zhang et al., 2024d）的过滤子集。过滤条件：4 FPS 均匀采样，仅保留 350–550 帧的视频，共 216K video QA pairs（原数据集 961K pairs）。每帧最大像素 50,176。训练 epoch=1。

  Benchmarks：
  - **LongVideoBench (LVB)**：跨帧 long-context referential reasoning，512 TPF × 256 frames
  - **MLVU_test**：多任务长视频理解（PlotQA, Needle, Ego, Count, Order, Anomaly, Topic, SportsQA, TutorialQA），128 TPF × 512 frames
  - **TimeScope**：task-oriented temporal grounding in long videos，64 TPF × 2048 frames
  - **LongTimeScope (LTS)**：10小时超长视频理解，128 TPF × 512 frames
  - **Tomato**：时序推理 benchmark，6种推理类型，3种视频场景，4 FPS × 256 frames
  - **VSIBench**：空间理解 benchmark（物体相对方向、路径规划、尺寸估计等），256 TPF × 128 frames
  - **LSDBench**：长视频采样 dilemma benchmark
  - **VideoEvalPro**：长视频 comprehensive evaluation (Holistic/Local Perception & Reasoning)

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  代码：https://github.com/Espere-1119-Song/VideoNSA
  模型：https://huggingface.co/Enxin/VideoNSA

  训练框架：**SWIFT**（Zhao et al., 2024），NSA 实现基于 **FLA (Flash Linear Attention)**（Yang & Zhang, 2024），参考了 Tilde Research 的 sparse attention 实现（Pai et al., 2025b）。

  算法 pipeline 核心流程：
  ```
  # === Input Encoding ===
  video_frames = sample(video, fps=4, max_frames=550)  # 最大 550 帧
  vision_tokens = ViT(video_frames)                     # ViT 编码每帧
  vision_tokens = MLP_Projector(vision_tokens)          # 映射到 LLM 空间
  text_tokens = tokenizer(instruction)                  # 文本 tokenize
  X = concat([vision_tokens, text_tokens])              # 视觉+文本 token 序列

  # === Per-layer Hybrid Attention ===
  for layer l in range(28):  # Qwen2.5-7B 共 28 层
      X_V, X_T = split_by_position(X)  # 按 position ID 分离

      # Vision tokens → NSA with 3 branches
      for head h in range(28):  # 每个 head 独立执行
          q = X_V @ W_q[h]       # [L_V, d_head]
          k = X_V @ W_k[h//7]    # GQA: 7 query heads share 1 KV head
          v = X_V @ W_v[h//7]

          # Branch 1: Compression (block-level token averaging)
          k_blocks = reshape(k, [-1, block_size, d_head])  # block_size = 64
          k_cmp = φ(mean(k_blocks, dim=1))                  # MLP φ 聚合
          v_cmp = φ_v(mean(reshape(v, [-1, block_size, d_head]), dim=1))
          o_cmp = softmax(q @ k_cmp^T / sqrt(d)) @ v_cmp

          # Branch 2: Selection (top-n important blocks)
          p_slc = importance_score(q, k_blocks)             # 计算 importance
          top_n_indices = topk(p_slc, n=32)                 # 选 top-n blocks
          k_slc = gather(k_blocks, top_n_indices)
          v_slc = gather(v_blocks, top_n_indices)
          o_slc = softmax(q @ k_slc^T / sqrt(d)) @ v_slc

          # Branch 3: Sliding Window (local context)
          k_swa = k[t-w+1:t]  # w = 256
          v_swa = v[t-w+1:t]
          o_swa = softmax(q @ k_swa^T / sqrt(d)) @ v_swa

          # Dynamic gating (2-layer MLP + sigmoid per head)
          g_cmp, g_slc, g_swa = sigmoid(MLP_gate(q))
          o_V_h = g_cmp * o_cmp + g_slc * o_slc + g_swa * o_swa

      # Text tokens → standard GQA
      o_T = GQA(X_T)  # 标准 FlashAttention

      o = concat([o_V, o_T])  # 拼接输出
      X = o + MLP(LayerNorm(o))  # residual + FFN
  ```

  关键超参数：block size s=64，global blocks b=32，sliding window w=256。Attention budget K_attn = b×s + w = 2048+256 = 2304。在 128K context 下仅使用 3.6% 的 attention edges（相比 dense 的 L(L-1)/2）。Optimizer: AdamW (β1=0.9, β2=0.999)，weight decay=0.01，peak LR=1e-6，cosine schedule，warmup ratio=0.1，batch size=32。

## Vgent: Graph-based Retrieval-Reasoning-Augmented Generation For Long Video Understanding

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：**Vgent**，一个训练无关（training-free）的 graph-based RAG 框架，用于长视频理解。包含四个核心阶段：
  (1) **Offline Video Graph Construction**：将视频以 1.0 FPS 采样，每 K=64 帧分割为一个 clip。对每个 clip 调用 LVLM 提取 JSON 格式的 entities（主体、物体、场景）、actions（交互/动作描述）和 scenes（地点/环境描述）。使用 BAAI/bge-large-en-v1.5 计算 entity 描述的 text embedding 相似度，以阈值 τ=0.7 合并跨 clip 的语义等价 entity。将每个 clip 作为图的 node，通过共享 entity 建立 edge，形成视频知识图谱 G=(V, E)，附带全局 entity 集合 U 及描述 t^u。
  (2) **Graph-based Retrieval**：从 query 中用 LVLM 提取 keywords K。对每个 keyword k 和 entity u，计算 BGE embedding 相似度 sim(k, t^u)，若 >θ=0.5 则将该 entity 关联的所有 nodes 纳入候选集。按 similarity rank 后取 Top-N=20 个 clips。
  (3) **Structured Reasoning**：LVLM 基于 query 和 keywords 生成结构化 subqueries Q（binary yes/no 或数值型），对 Top-N clips 逐一验证。仅保留有 subquery 正向匹配的 clip，最多 r=5 个。然后 LVLM 跨 refined clips 汇总信息。
  (4) **Multimodal Augmented Generation**：将 refined clips（视频帧 + 字幕）和 intermediate reasoning results 作为多模态上下文输入 LVLM 生成最终回答。

  实验比较：(1) **与 LVLM base models 对比**：在 7 种 open-source LVLM（InternVL2.5-2B, Qwen2.5-VL-3B/7B, Qwen2-VL-7B, LongVU-7B, LLaVA-Video-7B）上对比 MLVU、VideoMME、LongVideoBench 三个 benchmark；(2) **与 RAG methods 对比**：NaïveRAG（GoldFish 风格）、Video-RAG（CLIP keyframe + object detection + OCR）、以及 proprietary LLM-based methods（VideoAgent, LLoVi, DrVideo, VideoTree）；(3) **Ablation studies**：NaïveRAG vs GraphRAG vs GraphRAG+Structured Reasoning 的组件消融；confidence-based refinement 对比；retrieval 数量 N 和 r 的消融；retrieval embedding 类型（CLIP/BERT/BGE）、retrieval threshold τ 的影响；(4) **Inference time analysis**：per-minute video 的 offline/online 时间对比。

- 硬件平台是什么，配置是什么。
  **NVIDIA A100 80GB GPU**。所有实验在 A100 80G 上完成。推理时间统计（Table 5）：offline graph construction 20.13 sec/min-video，online retrieval+reasoning+generation 3.93 sec/min-video。VideoAgent 的 proprietary LLM 对比数据来自论文原文引用。

- 模型是什么。数据集和bench分别是什么。
  模型（Base LVLMs）：
  - **InternVL2.5-2B**：InternVL2.5 系列，2B 参数
  - **Qwen2.5-VL-3B**：Qwen2.5-VL，3B 参数
  - **Qwen2.5-VL-7B**：Qwen2.5-VL，7B 参数
  - **Qwen2-VL-7B**：Qwen2-VL，7B 参数
  - **LongVU-7B**：spatiotemporal adaptive compression for long video，7B 参数
  - **LLaVA-Video-7B**：LLaVA 视频版本，7B 参数
  - 另有 Qwen2-VL-2B 在附录 MLVU category-level 结果中评估

  Embedding 模型：**BAAI/bge-large-en-v1.5**（默认，用于 entity 合并和 keyword-entity 相似度计算）。对比实验也测试了 CLIP 和 BERT 作为 retrieval embedding。

  语音转文字：**openai/whisper-large**（用于 MLVU benchmark 无字幕视频的 spoken content 提取）。

  Benchmarks：
  - **MLVU**：多任务长视频理解 benchmark，视频长度 3 min ~ 2 hours，平均 12 min，含 7 类子任务：Count, Ego, Needle, Order, PlotQA, Anomaly, Topic
  - **VideoMME**：含 w/o subtitles 和 w/ subtitles 两个子集，按视频长度分 Short/Medium/Long 三档（11 sec ~ 1 hour）
  - **LongVideoBench (LVB)**：侧重需要跨帧长时间上下文推理的 referential reasoning 任务

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  代码仓库：https://xiaoqian-shen.github.io/Vgent（论文中给出的项目主页）。

  算法 pipeline 核心伪代码：
  ```
  # === Phase 1: Offline Video Graph Construction ===
  G = Graph()  # G = (V, E)
  U = set()   # global unique entity set
  for each clip V_i in split(video, fps=1.0, K=64):
      entities, actions, scenes = LVLM.extract_entities(clip=V_i, subtitle=C_i)
      # entities: JSON {entity_name: str, description: str}
      for each entity e_j in entities:
          t_j = BGE.encode(e_j.description)  # text embedding
          sim_scores = {u: cosine_sim(t_j, BGE.encode(u.description)) for u in U}
          u_star = argmax(sim_scores)
          if sim_scores[u_star] > 0.7:  # τ = 0.7
              merge(e_j, u_star)
              add_edges(v_i, {v for v in V if u_star in v.entities})
          else:
              U.add(e_j)
      V.add(v_i)  # node for clip i

  # === Phase 2: Graph-based Retrieval ===
  K = LVLM.extract_keywords(query)  # keywords from query
  R = set()
  for each keyword k in K:
      for each entity u in U:
          if cosine_sim(BGE.encode(k), BGE.encode(u.description)) > 0.5:
              R.update(get_nodes_with_entity(u))
  # Re-rank by avg similarity with query keywords across entities, descriptions, subtitles
  R_sorted = rank(R, query=K, fields=[entities, descriptions, subtitles])
  R_top = R_sorted[:20]  # Top-N=20

  # === Phase 3: Structured Reasoning ===
  Q_struct = LVLM.generate_subqueries(query, K)
  # Q_struct: list of {type: "binary"|"numeric", text: str}
  R_prime = []
  for each clip v_i in R_top:
      responses = [LVLM.answer(q, v_i) for q in Q_struct]
      if any(response > 0 for response in responses):
          R_prime.append(v_i)
  R_prime = R_prime[:5]  # max r=5
  reasoning_summary = LVLM.aggregate(R_prime, Q_struct)

  # === Phase 4: Multimodal Augmented Generation ===
  answer = LVLM.generate(query, context={
      "video_clips": R_prime,
      "reasoning": reasoning_summary
  })
  ```

  核心张量计算：entity 合并与检索均基于 BGE text embedding 的 cosine similarity。每个 entity description 经 BGE 编码为 1024-d 向量，entity 合并时对新 entity 与 U 中所有已有 entity 的 embedding 进行 argmax cosine similarity 匹配。检索时对 keyword 和 entity 的 BGE embedding 进行相同的 cosine similarity 计算。Video-RAG baseline 中对比的 CLIP-based 方法则计算 frame 的 CLIP visual feature 和 query text embedding 之间的 cosine similarity。

## VFlowOpt: A Token Pruning Framework for LMMs with Visual Information Flow-Guided Optimization

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：**VFlowOpt**，一个训练无关（training-free）的视觉 token 剪枝框架，包含三个核心模块：(1) **Visual Token Importance Estimation**：基于 attention calibration 和图像块信息熵计算重要性得分——先通过全局 attention 阈值筛选"相对重要"token 集合 K，再用 K 中 token 的 attention 权重与对应图像块的熵（256 灰度级）加权求和得到重要性得分 I_i = Σ_{k∈K} A_{ki} + α·softmax(H(V_i))；(2) **Progressive Pruning with Token Recycling**：将 LMM 均分为 3 个阶段，按阶段保留率 R=[R1, R2, R3] 逐步剪枝。初始剪枝后，将各空间网格（grid size=a）内的被剪枝 token 按重要性加权平均融合为一个 token，替换该网格内最高重要性 token 的位置，纳入保留集合；(3) **Visual Information Flow-Guided Optimization**：将剪枝策略超参数优化建模为最大化 cosine similarity 问题——minimize LMM 剪枝前后最后一层最后 token 表示的差异，使用 Bayesian Optimization（Gaussian Process + Expected Improvement）搜索最优 (R1, R2, R3, t, α, a)，30 个无标签样本 + 50 次迭代约 30 分钟。

  实验比较：(1) **Image understanding**：与 FastV、SparseVLM、VisionZip 在 LLaVA-OneVision-7B 上的 10 个 benchmark 对比，token 保留率 50%/25%/10%；同类实验在 LLaVA-NeXT-7B 和 Qwen2-VL-7B 上重复验证；(2) **Video understanding**：LLaVA-OneVision-7B 在 SeedBench (video) 和 VideoMME (Short/Medium/Long) 上对比；(3) **Efficiency analysis**：单卡 A100 上测量 FLOPs、KV-Cache 内存、推理延迟随剪枝比例变化；(4) **Ablation study**：移除 importance calibration / token recycling / progressive pruning 的消融实验；优化数据选择（随机 vs MathV360K-GEOS）影响；优化目标选择（last token vs mean pooling vs first token vs top-3 tokens）；样本数和迭代数的影响。

- 硬件平台是什么，配置是什么。
  单张 **NVIDIA A100-SXM4-80GB** GPU。优化阶段约 30 分钟。推理效率评估同样在 A100 上，测量 FLOPs (T)、Latency (ms)、KV-Cache Memory (MB)。

- 模型是什么。数据集和bench分别是什么。
  模型：
  - **LLaVA-OneVision-7B**（主力模型）：SigLIP ViT 视觉编码器 + Qwen2-7B LLM backbone，处理 1152×1152 图像产生 7290 tokens，剪枝点在 LLM 前、第 9 层后、第 18 层后
  - **LLaVA-NeXT-7B**：CLIP ViT + Vicuna-7B，剪枝点在 LLM 前、第 10 层后、第 20 层后
  - **Qwen2-VL-7B**：Qwen2-VL 专用 ViT + Qwen2-7B，max_pixels=3000000，剪枝点在 LLM 前、第 9 层后、第 18 层后

  优化数据：从各模型训练集随机采样 30 个无标签实例（无公开训练集则使用 LLaVA-OneVision 训练集）。评估 Benchmark（图像）：GQA、VizWiz、ScienceQA-IMG、TextVQA、ChartQA、POPE、MME、MMBench、MMStar、DocVQA。评估 Benchmark（视频）：SeedBench (video)、VideoMME（按视频长度分为 Short/Medium/Long 子集）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  代码仓库：https://github.com/sihany077/VFlowOpt（CC BY-NC 4.0，ICCV 2025 接收）。基于 iLLaVA、LMMs-Eval v0.2.4 和 LLaVA-OneVision 构建。

  **VFlowOpt 算法 Pipeline 伪代码：**

  ```
  # === 阶段 1: 重要性估计（在 ViT 最后一层执行） ===
  输入: 视觉 tokens V ∈ R^{N×D}, ViT attention matrix A ∈ R^{N×N}
  
  # Step 1: Attention Calibration —— 筛选相对重要 token
  τ = t * mean(Σ_i Σ_j A_{ij})                            # 阈值，t 为敏感度超参数
  K = {j | Σ_i A_{ij} > τ}                                # 相对重要 token 索引集合
  
  # Step 2: 图像块信息熵
  for i in 1..N:
      将 token i 对应图像块转为灰度: gray = mean(R,G,B)
      计算 256 级灰度直方图: p_k = count(gray==k) / num_pixels
      H(V_i) = -Σ_{k=0}^{255} p_k * log(p_k)              # 熵，越大信息越丰富
  
  # Step 3: 融合重要性得分
  for i in 1..N:
      I_i = Σ_{k∈K} A_{ki} + α * softmax(H(V_i))          # attention + 熵

  # === 阶段 2: Progressive Pruning（3个阶段各执行一次） ===
  输入: tokens V，token features F, 重要性 I, 保留率 R1/R2/R3
  
  N_keep = floor(N * R_current)
  idx_keep = topk(I, N_keep)                                # 保留高分 token
  idx_prune = setdiff(1..N, idx_keep)
  
  # Token Recycling —— 避免信息丢失
  定义 a×a 空间网格覆盖图像平面
  for each grid cell G_{p,q}:
      pruned_in_cell = {t_i in G_{p,q} | i in idx_prune}
      if len(pruned_in_cell) > 0:
          t_merged = Σ I_i * t_i / Σ I_i                    # 加权平均融合
          i_max = argmax_i(I_i for i in pruned_in_cell)    # 最高重要性位置
          F[i_max] = t_merged                               # 替换到保留集合
          idx_keep = idx_keep ∪ {i_max}
  
  V = V[idx_keep]                                          # 下一阶段输入
  N = len(idx_keep)

  # === 阶段 3: Bayesian Optimization 搜索最优超参数 ===
  # 优化目标
  f(R1,R2,R3,t,α,a) = CosineSim(
      h_f,                                                  # 无剪枝时最后 token 表示
      g_s(h_f)                                              # 剪枝后最后 token 表示
  )
  
  约束: R = (R1*L1 + R1*R2*L2 + R1*R2*R3*L3) / L         # 目标平均保留率
  
  # Bayesian Optimization loop (T=50 iterations)
  GP.fit(X0, f(X0))                                        # 初始随机采样拟合 GP
  for n in 1..T:
      x_next = argmax ExpectedImprovement(x; GP)            # 采集函数选点
      R3 = solve_constraint(R, R1, R2, L1, L2, L3)
      y = f(x_next)                                         # 评估目标
      GP.update(x_next, y)                                  # 更新 surrogate
  return argmax f(x)
  ```

  计算复杂度：重要性估计 O(N²) 来自 ViT attention 读取（已由 ViT 前向计算完成），Token Recycling O(N)（各 token 恰好归属一个 grid cell），Bayesian Optimization 每次迭代 O(1)（仅评估 cosine similarity）。与 Flash Attention 兼容。

## TimeViper: A Hybrid Mamba-Transformer Vision-Language Model for Efficient Long Video Understanding

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：**TimeViper**，一个混合 Mamba-Transformer 视觉语言模型，包含两个关键设计：(1) **Hybrid Mamba-Transformer LLM Backbone**（基于 Nanov2-9B），包含 27 层 Mamba-2、4 层 self-attention、25 层 MLP，结合 SSM 的 O(n) 计算效率和 attention 的表达能力；(2) **TransV（Token Information Transfer Module）**，LLM 内部的视觉 token 压缩模块，通过 Gated Cross-Attention 将冗余视觉 token 信息转移到指令 token 中，再丢弃原始视觉 token。TransV 在浅层（第 7 层）使用 uniform dropping（p=50%），在深层（第 39 层）使用 attention-guided dropping（p=90%）。TransV 增加约 100M 参数。

  实验比较：(1) **Ablation Study**：TransV 位置选择（第 2 层 vs 第 7 层）、浅层压缩率（50% vs 90%）、深层压缩策略（uniform vs attention-guided）、与纯 token dropping 对比；(2) **Memory & Prefilling Time**：Vanilla 模型 vs +ToMe vs +ToMe+TransV 的 GPU 内存和 prefilling 时间随帧数 (64→4096) 的变化；(3) **Main Results**：与 GPT-4V/o、Gemini-1.5-Pro 等 API 模型对比，与 LLaMA-VID、LongVA、LongVU、LLaVA-OneVision、LLaVA-Video、Qwen2-VL、Qwen2.5-VL、LongVILA、Kangaroo、Video-XL、Vamba、VideoChat-Flash、VTimeLLM、AuroraCap 等 Transformer-based MLLMs 对比，与 LongLLaVA、AuroraLong、Nanov2-VL 等 Hybrid/Linearized MLLMs 对比；(4) **Fair Comparison**：用相同训练 recipe 训练 Qwen2.5-7B 作为 Transformer baseline 与 TimeViper 对比；(5) **Frame Scalability**：输入帧数从 256→512→768→1024 的 benchmark 性能变化；(6) **Qualitative Analysis**：Mamba 层 attention 模式多样性（稀疏、局部、全局）、self-attention 层的 attention sink 现象、vision token attention 随深度递减。

- 硬件平台是什么，配置是什么。
  论文未明确说明训练 GPU 型号和数量。推断使用多卡 NVIDIA GPU（基于模型规模 9B + 7.8M 训练数据量级，且提到使用 data packing 技术加速训练）。GPU 内存分析测试未说明具体 GPU 型号，仅报告"Out of Memory at 128 frames"（vanilla 模型）。训练配置：learning rate=1e-5（TransV 模块 lr=5e-5），AdamW optimizer，weight decay=0.01，warmup ratio=0.03，cosine annealing scheduler。数据打包（data packing）支持 TransV 导致的可变序列长度。

- 模型是什么。数据集和bench分别是什么。
  模型：Visual Encoder = **SigLIP** (ViT, 768 vision tokens per frame @ 384×384)。Projector = Token Merging (ToMe)，每帧压缩至 16 tokens。LLM Backbone = **Nanov2-9B** (27 Mamba-2 + 4 Self-Attention + 25 MLP layers from NVIDIA Nemotron-Nano 2 hybrid architecture)。TransV 模块在 LLM 内部，约 100M 额外参数。

  训练数据（两阶段）：
  - Stage 1 (Image-Text Alignment)：3M image-text pairs，采样自 CC12M + PixelProse captions
  - Stage 2 (Video Instruction Tuning)：总计约 4.8M 样本：1.8M video instruction data（LLaVA-Video 1.3M + Kinetics400/WebVid recaptioned 253K by ShareGemini/ShareGPT-4 + VideoGPT-Plus 112K + ET-Instruct 100K + LongVid/MovieChat 11K），2.8M image instruction data（LLaVA-OneVision），26K dense video captioning（ActivityNet/COIN/HiREST/ViTT/YouCook2），250K temporal video grounding（YT-Temporal/DiDeMo/QuerYD/InternVid/HowTo100M, annotated by VTG-IT/TimeIT/TimePro/HTStep/LongVid）

  评测 Benchmark：
  - **Multi-choice QA**：VideoMME (2.7K QA, 11s-1h videos)、LVBench (2094 MCQ, hour-long)、MLVU (2174 QA, M-Avg)、LongVideoBench (retrieval-based QA, avg 473s)、MVBench (4K QA, 20 task categories)
  - **Temporal Video Grounding**：Charades-STA (6672 videos, mIoU)
  - **Video Detailed Captioning**：VDC (1027 videos, LLaMA3-8B judged score)

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  代码仓库：https://github.com/xiaomi-research/timeviper（Code TBD，尚未正式公开）。
  Project Page：https://xuboshen.github.io/TimeViper/

  **TimeViper 算法 Pipeline 伪代码：**

  ```
  输入: 视频 V (T 帧), 文本指令 I
  输出: 生成的回答 Y

  # === 1. Visual Encoding & Token Merging (ToMe) ===
  For each frame f_t in V:
      v_t = SigLIP_ViT(f_t)                  # shape: [768, D]
      v_t_compressed = ToMe(v_t)              # reduce 768 → 16 tokens
  X_0 = concat([v_1_compressed, ..., v_T_compressed])  # shape: [16T, D]

  # === 2. Text Tokenization ===
  X_1 = Tokenizer(I)                          # shape: [T_1, D]

  # === 3. Hybrid LLM Forward with TransV ===
  X = [X_0, X_1]                              # multimodal input, shape: [16T+T_1, D]

  For layer l = 0 to L (total 56 layers, including Mamba-2, Self-Attn, MLP):

      # --- Mamba-2 Layer (27/56 layers, O(n) complexity, O(1) KV-cache) ---
      # Equation: h_t = A_t * h_{t-1} + B_t * x_t
      #           y_t = C_t^T * h_t
      # SSM maintains a fixed-size hidden state h_t capturing historical info
      X = Mamba2_SSM(X)                       # recurrent state-space update
      X = X + MLP(X)                          # residual MLP

      # --- Self-Attention Layer (4/56 layers) ---
      # y = Softmax(L ⊙ QK^T/sqrt(D)) * V
      Q, K, V = W_Q(X), W_K(X), W_V(X)
      X = X + CausalSelfAttention(Q, K, V)
      X = X + MLP(X)

      # --- TransV (Token Information Transfer) ---
      If l == 7:  # shallow layer
          # Uniform dropping: drop 50% of vision tokens
          X_0_ids = UniformDrop(X_0, rate=0.5)
          X_0_kept, X_0_dropped = split(X_0, X_0_ids)

          # Gated Cross-Attention: transfer vision info to instruction tokens
          Q_inst = W_Q(X_1)
          KV_vis = W_KV(X_0_dropped)
          X_1_tilde = CrossAttn(Q_inst, KV_vis)
          alpha = tanh(learnable_alpha_7)
          X_1 = X_1 + alpha * X_1_tilde

          X_0 = X_0_kept  # discard dropped vision tokens

      If l == 39:  # deep layer
          # Attention-guided dropping: keep top-k vision tokens
          # most attended by the last instruction token
          attn_scores = Attention(X_1[-1], X_0)  # last inst token as query
          X_0_ids = TopK(X_0, score=-attn_scores, k=T_0*0.1)  # keep 10%
          X_0_kept, X_0_dropped = split(X_0, X_0_ids)

          Q_inst = W_Q(X_1)
          KV_vis = W_KV(X_0_dropped)
          X_1_tilde = CrossAttn(Q_inst, KV_vis)
          alpha = tanh(learnable_alpha_39)
          X_1 = X_1 + alpha * X_1_tilde

          X_0 = X_0_kept

  # === 4. Autoregressive Generation ===
  Y = []
  While not EOS:
      logits = LM_Head(X)
      y_next = Sample(logits)
      Y.append(y_next)
      X = [X, Embed(y_next)]
  ```

  **关键张量计算细节：**
  - ToMe 压缩：768 → 16 tokens/frame，基于 ViT 内部 token 相似性合并
  - Mamba-2 SSM 更新：h_t ∈ R^{N×D}（隐状态维度 N=128），递推式状态刷新，O(1) 推理内存
  - TransV Cross-Attention：Q ∈ R^{T_1×D}，KV ∈ R^{T_d×D}（T_d 为被丢弃的 vision token 数），输出加入 instruction tokens
  - α_l 初始化为 0（保证初始时 instruction 理解不受影响），训练后学习到合适的转移比例
  - 最终 vision token 压缩比：(1-0.5)×(1-0.9) = 5% 保留，即 95% 的 vision tokens 被丢弃

## Temporal Preference Optimization of Large Multimodal Models

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：**Temporal Preference Optimization (TPO)**，一种 video-LMM 的后训练（post-training）框架。核心 pipeline 三步：(1) **Temporal Preference Modeling** — 通过操纵视频输入生成对比响应对：给定从视频中采样的帧集合 F，先用 CogVLM2 生成每帧 caption，再由 GPT-4o-mini 生成相关问题 Q；preferred response 使用 Q + 相关帧 F 作为输入由 video-LMM 生成；dis-preferred response 使用 Q + 不相关帧（从剩余帧采样，Irrelevant）或部分相关帧（随机子采样，Incomplete）作为输入生成。(2) **LLM-based Post-Filtering** — 用 GPT-4o-mini 对偏好数据对进行规则化过滤，剔除 dis-preferred 优于 preferred、preferred 事实错误、或问题模糊的样本。(3) **DPO Training** — 使用 Direct Preference Optimization + SFT loss 联合训练，损失函数 L = L_DPO + α·L_SFT。

  实验比较：(1) 与 SFTSelf（自生成数据做 SFT）、SFTLLM（GPT-4o-mini 生成数据做 SFT）、Hound-DPO（DPO on video-LMM，使用 ChatGPT 评分生成偏好数据）三种训练策略对比；(2) 与 GPT-4o、Video-LLaVA、LLaVA-1.5、Qwen-VL-Max、ShareGPT4Video、InternVL-Chat-V1.5、VideoChat2、LongLLaVA、Video-CCAM、NVILA、Qwen2-VL、Apollo 等 SOTA 模型对比；(3) 消融实验：不同输入帧数、数据集规模（2k/5k/10k）、有无 post-filtering、不同 dis-preferred 数据混合比例（Incomplete:Irrelevant）、Needle-in-a-Haystack 任务。

- 硬件平台是什么，配置是什么。
  训练：8 × NVIDIA A100 80GB GPU，batch size 64，full fine-tuning（language model + multimodal projector），visual encoder frozen。LongVA-TPO 训练约 4 小时（lr=4×10⁻⁶），LLaVA-Video-TPO 训练约 4 小时（lr=3×10⁻⁷），cosine lr scheduler with warmup ratio 0.1，1 epoch。数据准备阶段使用 GPT-4o-mini（text-only input）进行 question curation 和 post-filtering。

- 模型是什么。数据集和bench分别是什么。
  模型：LongVA-7B（长上下文 video-LMM）和 LLaVA-Video-7B（SOTA 7B video-LMM），均基于此做 TPO 微调得到 LongVA-TPO 和 LLaVA-Video-TPO。
  训练数据：手动 curator 200 关键词，爬取 8000 个互联网视频，生成 10K 偏好数据对（LongVA-TPO）；对 LLaVA-Video-TPO，从 LLaVA-Video-178K 数据集中采样 10K QA 对，dis-preferred 仅用 Incomplete 方式生成。数据任务分布：Temporal Reasoning 8.7%、Action Reasoning 12.4%、Causal Reasoning 11.1%、Information Extraction 18.0%、Descriptive 12.8%、Summarization 7.5%、Object Reasoning 14.9%、Spatial Reasoning 13.5%。
  评测 benchmark：LongVideoBench（长视频上下文推理）、MLVU（多任务长视频理解）、Video-MME（多模态视频评测，含 Short/Medium/Long 三个子集）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  代码开源：https://github.com/ruili33/TPO；数据集和 checkpoint：https://huggingface.co/collections/ruili0/temporal-preference-optimization-67874b451f65db189fa35e10。

  TPO 算法 pipeline 伪代码：
  ```
  输入: 视频 V, video-LMM π_θ, 参考模型 π_ref
  输出: 优化后的模型 π_θ*

  # === Phase 1: Temporal Preference Data Generation ===
  For each video V in crawled_videos:
      F = sample_frames(V)                    # 采样帧集合
      captions = CogVLM2.caption(F)           # 逐帧 caption
      Q = GPT4o-mini.generate_questions(captions)  # 问题生成

      # Preferred response: 使用相关帧
      r⁺ = π_θ(V[F], Q)

      # Dis-preferred response (两种策略):
      F_irrelevant = sample(V \ F)            # 策略(a): 不相关帧
      F_incomplete = random_subset(F)         # 策略(b): 不完整帧
      r⁻_irrelevant = π_θ(V[F_irrelevant], Q)
      r⁻_incomplete = π_θ(V[F_incomplete], Q)

      # Post-Filtering with GPT-4o-mini (3 条规则)
      keep = filter(captions, Q, r⁺, r⁻)
      if keep:
          D.add((V, Q, r⁺, r⁻))

  # === Phase 2: DPO Training ===
  For each (V, Q, r⁺, r⁻) in D:
      # DPO loss (公式 2)
      log_ratio⁺ = log π_θ(r⁺|V,Q) - log π_ref(r⁺|V,Q)
      log_ratio⁻ = log π_θ(r⁻|V,Q) - log π_ref(r⁻|V,Q)
      L_DPO = -log σ(β · (log_ratio⁺ - log_ratio⁻))

      # SFT loss (公式 3)
      L_SFT = -log π_θ(r⁺|V,Q)

      # Combined loss (公式 4)
      L = L_DPO + α · L_SFT

      θ ← θ - η · ∇_θ L
  ```
  关键超参：LongVA-TPO: β=0.3, α=0.5, lr=4×10⁻⁶；LLaVA-Video-TPO: β=0.2, α=1, lr=3×10⁻⁷。两者均 full fine-tuning（language model + multimodal projector），visual encoder frozen，1 epoch。

## SPIKE-RL__Video-LLMs_meet_Bayesian_Surprise

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：(1) **SPIKE**：推理时框架，将 Video-LLM 的信念表示为显式概率分布（文字化信念假设），通过 KL 散度计算 Bayesian Surprise 得分，引导 surprise-weighted frame sampling。(2) **SPIKE-RL**：基于 GRPO 强化学习优化信念假设生成，LLM-Match 作为 reward signal，通过最终视频 caption 质量反向传播 credit 到中间信念序列。实验比较 surprise localization（三个 benchmark：Oops!、FunQA、Mr. Bean）和下游任务（BlackSwan、FunQA Task 2、ExFunTube、VideoMME-S、NextQA）上与 uniform sampling、RGB Histogram、ECR、Katna、Optical Flow 等 query-free 采样方法的性能。

- 硬件平台是什么，配置是什么。
  训练：4 × H100，单节点，DeepSpeed ZeRO-3 offload。推理：论文未明确单独给出推理平台，但使用 Qwen2.5-VL-7B-Instruct 作为 backbone，FlashAttention-2、bfloat16、PEFT。

- 模型是什么。数据集和bench分别是什么。
  模型：Backbone 为 Qwen2.5-VL-7B-Instruct（主要）和 Qwen2.5-VL-32B（扩展）；LLM-Match reward model 为 Olmo-7B-hf；历史摘要压缩使用 BART-Large-CNN。训练集：2000 个视频，30% surprising（Oops! 训练集）+ 70% unsurprising（ActivityNet Captions）。评测 benchmark：
  - Surprise Localization: Oops!（4,791 视频，精确时间戳）、FunQA（424 视频，标注最 surprising 片段）、Mr. Bean（48 视频，自定义，笑声轨道为标注）
  - 下游任务: BlackSwan Suite（Reporter-MCQ）、FunQA Task 2（解释生成）、ExFunTube（解释生成）、VideoMME-S（多模态推理，短视频无字幕）、NextQA（时间/常识/因果推理）

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  代码开源：https://github.com/sahithyaravi/SPIKE-RL。基于 Qwen2.5-VL（开源）训练 SPIKE-RL。

  算法 pipeline 核心流程：
  ```
  输入: 视频帧序列 X_{1:T}，帧预算 F
  输出: surprise-guided 采样的 F 帧

  # Step 1: 均匀采样 K ≤ F 个时间锚点
  timesteps = uniform_sample(T, K)

  For each timestep t = t_1, ..., t_K:
      # Step 2: 构建上下文
      W_t = X_{t-W:t-1}           # 前序帧窗口（W=4）
      H_t = summarize(X_{t-C:t-W-1})  # 历史文本摘要（BART-Large-CNN 压缩）

      # Step 3: 生成信念假设（N=3 个假设）
      B_t = {b_{t,1}, ..., b_{t,N}}
          = VideoLLM.generate(H_t, W_t, nucleus_sampling)

      # Step 4: 计算先验分布 P_prior
      For each hypothesis b_{t,i}:
          NLL_prior_i = -log P_M(b_{t,i} | H_t, W_t)
      P_prior = softmax(-NLL_prior / τ)   # τ 为温度参数

      # Step 5: 计算后验分布 P_post（加入当前帧 O_t = X_t）
      For each hypothesis b_{t,i}:
          NLL_post_i = -log P_M(b_{t,i} | H_t, W_t, O_t)
      P_post = softmax(-NLL_post / τ)

      # Step 6: Bayesian Surprise = KL(P_post || P_prior)
      S_t = D_KL(P_post || P_prior)
          = Σ_i P_post(b_{t,i}) * log(P_post(b_{t,i}) / P_prior(b_{t,i}))
      # 实际使用 JSD 替代 KL：S_t = JSD(P_post, P_prior) ∈ [0, 1]

  # Step 7: Surprise-weighted 采样
  p_i = softmax(S_i / τ_s), τ_s = 0.7  # 将 surprise 转为采样概率
  selected_frames = multinomial_sample(F, p)  # 高 surprise 段可被多次采样
  ```

  SPIKE-RL 训练流程（GRPO）：
  1. 对每个视频，执行 M=3 条 rollout 轨迹
  2. 每条轨迹：SPIKE 产生信念假设 + surprise 得分 → surprise-weighted 采样帧 → VideoLLM 生成 caption c
  3. LLM-Match 评估 caption c 与 ground truth 相似度，得 reward R
  4. 组内 Z-score 归一化：A^{(r)} = (R^{(r)} - μ_R) / σ_R
  5. 策略梯度优化：L = -1/M Σ_r A^{(r)} Σ_t Σ_k log p_θ(b_{t,k}^{(r)} | H_t^{(r)}, W_t^{(r)})
  6. 训练超参：LR=1e-6，GRPO β=0.1，N_hypotheses=3，max_prompt_length=8192，batch_size=4，epochs=1

## StreamingVLM__Real-Time_Understanding_for_Infinite_Video_Streams

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：(1) **Streaming-aware KV Cache**：推理时维持紧凑 KV cache，复用 attention sink tokens（Tsink=512）、近期 text 长窗口（Twindow=512）、近期 vision 短窗口（Vwindow=16s），旧 vision tokens 优先驱逐，旧 text 仅在超出 budget 时驱逐。(2) **Contiguous RoPE**：当旧 token 被驱逐后，后续 token 的 RoPE 位置索引左移保持连续，超出总窗口后位置索引不再增长，保持在训练长度范围内。(3) **Overlapped-Chunk Full-Attention Training**：训练时不复制推理的滑动窗口，而是将长视频切分为 W=24s 的 chunk（O=12s 重叠），每个 chunk 内做 full attention，让每个 chunk 内部的 attention pattern 近似推理时的 attention sink + sliding window。Vision/text tokens 以 1s 为间隔交错排列，仅在 text 位置计算 loss。
  
  实验比较：
  - Captioning：Inf-Streams-Eval（20 场完整比赛，平均 2.12 小时）上对比 GPT-4o mini（chunk 模式）、LiveCC-7B-Instruct（chunk/infinite 模式）、ReKV（训练无关 KV cache 驱逐方法），以 GPT-5 投票 win rate 为指标；LiveCC-Sports-3K CC 上对比 LLaVA-Video、GPT-4o、Gemini、LiveCC。
  - VQA：MVBench、VideoMME (w/o subs.)、LongVideoBench、OVOBench Realtime 上对比 Qwen2.5-VL-7B-Instruct（SFT 前的 base model），验证 SFT pipeline 提升通用视觉能力。
  - Efficiency：单卡 H100 上测试 per-token latency vs. 视频长度，对比 Full Attention、Sliding Window w/o Overlap、Sliding Window w/ Overlap。
  - Ablation：Contiguous RoPE vs. Native RoPE（infinite/chunk 模式）；Tsink/Twindow/Vwindow 大小；SFT strategy 和数据集消融（Live-WhisperX-526K → +Inf-Streams-Train → +High-Quality Annealing）。

- 硬件平台是什么，配置是什么。
  训练：NVIDIA H100 GPU，总计算量约 128 H100-days（两阶段：SFT 525K+526K 样本 → 高质量 annealing 14K 样本）。推理：单张 NVIDIA H100，bfloat16，维持 8 FPS 实时视频理解。

- 模型是什么。数据集和bench分别是什么。
  模型：Backbone 为 Qwen2.5-VL-7B-Instruct；对比模型包括 GPT-4o mini、LiveCC-7B-Instruct、LiveCC-7B-Instruct (infinite mode)。
  训练集：
  - Inf-Streams-Train：自行构建，5 种体育项目（篮球 712 场、足球 544 场、冰球 402 场、棒球 399 场、美式足球 392 场），总计 2,449 场比赛，使用 WhisperX 提取 ASR 实时解说 → GPT-5 清洗（keep 46.32%/edit 37.89%/delete 15.79%）→ overlapped chunking（W=24s, O=12s），最终 525K streaming samples。
  - LiveCC 的 Live-WhisperX-526K：526K streaming samples。
  - High-quality annealing data：14K samples（16-64s clips，GPT-5 筛选实时解说比例 > 80%）。
  评测 benchmark：
  - Captioning: Inf-Streams-Eval（20 场完整比赛，平均 2.12h，per-second 对齐）、LiveCC-Sports-3K CC（49 运动，416 clips, ≥10s）
  - VQA: MVBench（细粒度动作/物体/计数/时序）、VideoMME（多任务 QA/caption/grounding）、LongVideoBench（长视频 QA，需长期记忆和跨段推理）、OVOBench Realtime（流式感知理解）

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  代码开源：https://github.com/mit-han-lab/streaming-vlm。基于 Qwen2.5-VL-7B-Instruct（开源）训练。

  算法 pipeline 核心流程（推理时）：
  ```
  输入: 无限视频帧流，基础模型 Qwen2.5-VL
  参数: T_sink=512, T_window=512, V_window=16s

  初始化 KV cache = {}，position_offset = 0

  Loop over time steps:
      # Step 1: 新帧 tokenize
      V_new = vision_encoder(frames_new)   # 如 1s=24 frames → N_v tokens

      # Step 2: interleave vision/text tokens（1s 间隔）
      # 无解说词的秒插入占位符 "..."

      # Step 3: Contiguous RoPE 位置编码
      current_pos = KV_cache.length
      RoPE_indices = [position_offset, ..., position_offset + len(V_new) + len(T_new) - 1]
      # 但不超过训练最大长度 L_max
      RoPE_indices = RoPE_indices % L_max   # 简化，实际以 bounded range 保持连续

      # Step 4: Full attention 计算（仅对当前窗口内 tokens）
      Q, K, V = project(V_new + T_new)
      attend_to = KV_cache  # 复用历史 KV，不 recompute
      attention = softmax(Q @ K_attend.T / sqrt(d)) @ V_attend

      # Step 5: 更新 KV cache（eviction policy）
      KV_cache = KV_cache + new_KV
      # 保留: T_sink 个 attention sink tokens（system prompt + 早期 text）
      # 保留: T_window 个最近 text tokens  
      # 保留: V_window 秒的最近 vision tokens
      # 驱逐: 旧 vision tokens 优先，旧 text 仅在超 budget 时驱逐

      # Step 6: 自回归生成
      output = model.generate(..., past_key_values=KV_cache, position_ids=RoPE_indices)
  ```

  训练时（overlapped-chunk, full-attention）：
  ```
  输入: 完整体育比赛视频
  参数: W=24s, O=12s

  # 切分为 overlapped chunks
  For chunk_i in sliding_window(video, window=W, overlap=O):
      # chunk_i 包含 W 秒内以 1s 间隔 interleaved 的 vision+text tokens
      V_chunk = vision_encoder(chunk_i_frames)   # 24 frames@1fps
      T_chunk = tokenize(chunk_i_commentary)     # 对应解说词

      # Full attention within chunk（每个 token attend 到同 chunk 所有 token）
      # 不复制推理时的 sink/sliding window mask
      # 仅在 text position 计算 cross-entropy loss
      loss = CE(model(V_chunk ⊕ T_chunk).logits[text_positions], labels)

      # 前一段的 previous text 取 T_sink 开头 + T_window 结尾 tokens
  ```

  训练-推理一致性关键点：overlapped chunk 内 full attention 的 attention pattern 近似推理时 "sink tokens → 全可见 + 近期 text 窗口 + 近期 vision 窗口" 的有效注意力模式，teaching the model recency bias without training on prohibitively long contexts。

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：SAGE (Smart Any-horizon aGEnt) —— 一个面向长视频推理的Any-Horizon Agent系统。核心包含三部分：
  (1) **SAGE Agent System（System Design）**：两阶段多轮推理架构。Stage-1（Context VLM）接受128帧采样视频(F)、视频元数据(M)、工具定义(T)和用户查询(Q)，SAGE-MM输出 video-context + query-intent + final-answer 或 tool-call。Stage-2（Iterative Reasoner）迭代决定是否可回答或继续调用工具，最多11步。支持6种工具：web-search（Google Search API via Serper）、parse-website、transcribe-speech（Whisper-large-v3）、ground-event（Qwen3-VL-30B-A3B-Instruct）、extract-video-parts、analyze（Qwen3-VL-30B-A3B-Instruct）。关键创新：通过知识驱动（web search + speech transcription）而非纯时序定位来实现高效事件搜索；segment-level grounding（每次最多10分钟）而非whole-video grounding。
  (2) **Synthetic Data Generation Pipeline**：使用Gemini-2.5-Flash一次性处理完整长视频生成10-20个QnA pairs（覆盖全时间跨度，通过percent_video_parsed字段强制覆盖）。再使用SAGE系统（Gemini-2.5-Flash作为SAGE-MM）生成tool call trajectories用于cold-start SFT。相比人工标注节省~100倍成本、相比subclip pipeline节省~10倍时间。99.1k训练问题来自6659个YouTube视频，417.7k state-action pairs用于SFT。
  (3) **Multi-Reward RL Post-Training（GRPO）**：使用GRPO进行trajectory-level优化。Rollout 8条轨迹/sample。Reward由step-level rewards + accuracy reward组成：
    - s_format: JSON格式奖励（+0.05或-0.10）
    - s_reasonable-tool: GPT-4o判断当前tool call是否合理（+0.10或-0.10）
    - s_args-repeat: 惩罚重复tool call参数（-0.05 × sqrt(num-repetitions)）
    - s_args-valid: 惩罚无效参数（-0.1或0）
    - a_N (Accuracy Reward): LLM-as-Judge (GPT-4o) 判断最终答案语义正确性。错误回答:-0.5, 正确+visual tools:+1.25, 正确无tools:+1.0, JSON无效:-2.0
  KL-divergence loss coefficient=0.005。前100步Nmax=6稳定训练，之后Nmax=11。

  实验比较：
  (a) **SAGE-Bench主结果（Table 4）**：1744样本（802 MCQ + 942 open-ended），平均时长727秒。DIRECT baselines（Gemini-2.5-Flash, GPT-4o, Video-Thinker-7B, LongVILA-R1-7B, VideoRFT-7B, Video-R1-7B, Qwen3-VL系列）和AGENT baselines（VideoAgent, LVAgent, LongVT, VideoMind, VideoExplorer, VideoChat-R1.5）。SAGE-MM基于Qwen3-VL-8B-Instruct SFT+RL在SAGE-Bench达68.0 overall，SAGE-Flash（Gemini-2.5-Flash作为tool backend）达71.8。
  (b) **MINERVA（Table 5）**：SAGE在>600秒视频上improvement 2.6%。
  (c) **Video-MMMU & Video-MME（Table 6）**：SAGE-Flash在Video-MMMU达68.1，超Video-R1（61.5）。
  (d) **Duration-wise分析（Table 8）**：600-1200秒bucket改善8.2%（SAGE）、14.6%（SAGE-Flash）。
  (e) **Training Mode消融（Table 7）**：AGENT模式优于DIRECT模式训练。
  (f) **Any-Horizon Reasoning（Table 9）**：RL改善SFT模型的tool overcalling，使单轮/多轮分布更接近expert Gemini-2.5-Flash。
  (g) **Tool消融（Table 10）**：transcribe-speech和extract-video-parts最重要。
  (h) **Runtime（Table 11）**：SAGE 8.6s/sample，vs VideoMind 24.7s，VideoAgent 1445.0s。
  (i) **Appendices中的额外消融**：Video input重要性（Table 12）、Eval mode（Table 13）、SFT必要性（Table 14）、#Turns vs Duration（Table 15）、Nmax影响（Table 16）、Variance（Table 17）、Per-tool accuracy（Table 18）。

- 硬件平台是什么，配置是什么。
  训练：16×NVIDIA H100 GPUs，SFT和RL阶段均使用此配置。SFT: batch size=64, lr=1e-5（linear decay），1 epoch。RL: batch size=16, rollout 8条trajectories/sample, lr=1e-6（cosine decay），KL coeff=0.005，训练480 steps。RL前100步Nmax=6，之后Nmax=11。
  推理评估：使用vLLM serving所有模型，温度0.0。非确定性输出时temperature=0.7，最多4次重试。

- 模型是什么。数据集和bench分别是什么。
  模型：
  - SAGE-MM variants: Qwen3-VL-8B-Instruct（默认）、Qwen3-VL-4B-Instruct、Qwen2.5-VL-7B-Instruct、Molmo2-8B
  - Tool backends: Qwen3-VL-30B-A3B-Instruct（ground-event + analyze tools）、Whisper-large-v3（transcribe-speech）
  - Expert orchestrators（无training）: Gemini-2.5-Flash、GPT-4o
  - LLM-as-Judge: GPT-4o（accuracy reward + evaluation）
  数据集：
  - 训练：13个YouTube频道（Formula1, ZachChoi, TheDailyShow, MrBean, TheOffice, Friends, fluffyguy, trevornoah, Vox, kurzgesagt, veritasium, QuantaScienceChannel, WalkingAlice），6659视频 → 99.1k 训练问题 + 417.7k SFT state-action pairs。RL数据：7.68k样本（一半需tool calls，一半single-turn）。
  Benchmarks：
  - SAGE-Bench：1744样本（802 MCQ + 942 open-ended），平均727秒，来自娱乐YouTube视频
  - MINERVA：复杂视频推理benchmark（体育、短片、烹饪）
  - Video-MMMU：多学科专业视频知识获取
  - Video-MME：视频分析评估bechmark

- 开源情况。基于开源文档和论文，使用例子解释算法pipeline，至少具体到伪代码或张量计算。
  开源: https://github.com/allenai/SAGE

  算法pipeline伪代码（SAGE推理 + GRPO训练核心流程）：

  ```
  # ========== SAGE推理 ==========
  # Input: 视频文件video_path, 用户查询Q
  # Stage-1: Context VLM
  F = sample_frames(video_path, n=128, fps=2)  # 128帧, 2 FPS采样
  M = {"path": video_path, "duration": duration_seconds}
  T = [web_search, parse_website, transcribe_speech, 
       ground_event, extract_video_parts, analyze]
  
  # SAGE-MM推理第一步
  prompt_1 = f"T|F|Q|M: {T}\n{M}\n{Q}"
  action_1 = SAGE-MM(prompt_1)  # JSON: video_context, query_intent, 
                                 #        recommended_tool | final_answer
  
  if action_1.final_answer is not None:
      return action_1.final_answer  # 单轮推理（any-horizon: 短视频直接回答）
  
  # Stage-2: Iterative Reasoner
  tool_results_1 = execute_tool(action_1.recommended_tool)
  C = action_1.video_context  # visual context cache
  
  for step in range(2, N_max+1):  # N_max=11
      # 累积历史中的所有action和tool结果
      history = [A_1, R_1, ..., A_{step-1}, R_{step-1}]
      prompt_k = f"T|Q|M|C|{history}"
      action_k = SAGE-MM(prompt_k)  # JSON: answerable, recommended_tool | final_answer
      
      if action_k.final_answer is not None:
          return action_k.final_answer
      
      tool_results_k = execute_tool(action_k.recommended_tool)
  
  return None  # 超时未回答

  # ========== GRPO RL训练 ==========
  # Input: S_i = {T, F, M, Q} for sample i
  # Rollout generation (batch_size=16, 8 trajectories each)
  for i in range(batch_size):
      for k in range(8):  # rollout 8 trajectories
          tau_{i,k} = []
          S_1 = {T, F, M, Q}
          for j in range(1, N_max+1):  # N_max=6 (前100步) or 11
              A_j = SAGE-MM(S_j)
              tau_{i,k}.append((S_j, A_j))
              if A_j.final_answer is not None:
                  break
              R_j = execute_tool(A_j.recommended_tool)
              S_{j+1} = {T, Q, M, C, A_1, R_1, ..., A_j, R_j}
  
  # Reward computation
  for each tau_i in rollout_trajectories:
      # Step-level rewards
      s_format    = +0.05 if valid_json else -0.10
      s_reasonable = +0.10 if GPT4o_judge_reasonable(tau_i, Q) else -0.10
      s_args_repeat = -0.05 * sqrt(count_repetitions(tau_i))
      s_args_valid  = -0.10 if invalid_args else 0
      
      # Accuracy reward (LLM-as-Judge: GPT-4o)
      if final_action_is_invalid_json:
          a_N = -2.0
      elif GPT4o_judge_correct(final_answer, ground_truth):
          a_N = +1.25 if used_visual_tools_in_tau_i else +1.0
      else:
          a_N = -0.5 if N >= 1 else ...
      
      # Uniform reward for all actions in trajectory
      R_i = sum(step_rewards) + a_N
      r(A_1) = r(A_2) = ... = r(A_N) = R_i
  
  # GRPO advantage computation and policy update
  for each sample i:
      advantages = compute_group_advantages([R_{i,1}, ..., R_{i,8}])
      # Update SAGE-MM using GRPO loss:
      # L = -E[min(r_t * A, clip(r_t, 1-eps, 1+eps) * A)] + beta * KL(pi||pi_ref)
      # KL coeff beta = 0.005
  ```

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

## OneVision-Encoder__Codec-Aligned_Sparsity_as_a_Foundational_Principle_for_Multimodal_Intelligence

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：OneVision-Encoder (OV-Encoder) —— 一个HEVC风格的自监督Vision Transformer，将视觉表示学习与视频信号的预测性结构对齐。核心创新：
  (1) **Codec Patchification（编解码器分块）**：利用HEVC/H.265编解码器暴露的运动矢量（motion vectors）和预测残差（prediction residuals）作为时空显著性信号，仅在密集视频中编码3.1%-25%的高信息熵patch，其余静态背景区域不参与计算。具体包括三种输入形式：
    - Dense Video-Codec Patchification：64帧密集视频输入，按GOP（Group of Pictures, 每32帧一个I-frame）结构组织，I-frame全量编码（建立空间上下文），P-frame仅选择运动+残差显著patch，clip-level token budget固定为2048（512 for 2 I-frames + 1536 for 62 P-frames），对比密集处理（16384 patches）减少87.5%。
    - Chunk-wise Patchification：视频均匀分块，每块随机采样1帧，实现稀疏时序采样。
    - Single-Image Spatial Patchification：静态图像按行主序patchify。
  (2) **3D-RoPE位置编码**：统一三种输入形式的位置编码。Dense Video-Codec使用完整时空偏移(Δt, Δx, Δy)，Chunk-wise使用(Δc, Δx, Δy)（chunk间偏移），单图退化为(0, Δx, Δy)纯空间编码。
  (3) **百万级聚类判别（Cluster Discrimination）训练目标**：冻结预训练metaCLIP编码器提取图像嵌入（cluster为2M类中心）和视频嵌入（cluster为400K类中心），将聚类结果作为伪标签进行多标签语义判别训练，联合捕捉物体级（object-level）和动作级（motion-level）语义。损失函数为sigmoid binary cross-entropy over multi-label assignments。
  (4) **两阶段预训练pipeline**：Stage 1（13B samples, 仅图像, resolution 224, 2M classes）→ Stage 2（4B samples, 图像+视频+OCR, 图像res 448/视频res 224, 64帧固定clip, GOP=32, video:image ratio=1:1）。
  (5) **Attentive Pooling Head**：多注意力头池化（multi-head attention pooling），从SigLIP适配，用于聚合spatiotemporal tokens到compact class embeddings。
  (6) **ViT-L架构**：24层transformer, hidden dim 1024, 16 attention heads, patch 14×14, GELU+LayerNorm, Flash Attention 2。

  实验比较：
  (a) **LMM Probing（Table 2）** —— Qwen3-4B-Instruct2507作为语言backbone，固定LLM比较不同vision encoder。16个benchmarks：7个视频（MVBench, MLVU-dev, NExT-QA, VideoMME, PerceptionTest, TOMATO, LongVideoBench） + 9个图像/文档（AI2D, ChartQA, DocVQA, InfoVQA, MMBench-EN, OCRBench, OCRBench v2, MMStar, RealWorldQA）。对比Qwen3-ViT（from Qwen3-VL-4B）、SigLIP2、OV-Encoder-Frame（dense frame variant）。所有模型用同一1.5M LLaVA-Next/LLaVA-Next-Videos instruction-tuning corpus和native-resolution evaluation策略。
  (b) **Stage-wise Analysis（Table 3）** —— 对比Stage1（image-only）vs Stage2（+OCR+video+Codec），在8 image benchmarks上评估。
  (c) **Attentive Probing（Table 4）** —— 冻结backbone + 轻量attention classifier head。7个视频benchmarks（SSV2, Diving48, PerceptionTest, CharadesEgo, Epic-Kitchens Verb/Noun, Kinetics-400, HMDB51）。对比CLIP, SigLIP, MetaCLIP/2, AIMv2, SigLIP2, DINOv3。两种设置：8 frames/2048 patches 和 16 frames/4096 patches。
  (d) **Patch Budget Scaling（Table 5）** —— token budget 512/1024/2048/4096对应dense 2/4/8/16帧，对比SigLIP2 dense全帧 vs OV-Encoder Codec稀疏选择。patch reduction 75%-96.9%。
  (e) **Codec-guided Ablation（Table 6）** —— 三种干预实验：Non-motion Patch Replacement (50%)、Counterfactual Motion Replacement (50%)、Patch-Position Shuffle。证明codec-selected motion patches的语义必要性和位置重要性。

- 硬件平台是什么，配置是什么。
  预训练：128×NVIDIA A800 GPUs（16 nodes × 8 GPUs）。Stage 1训练13B samples，Stage 2训练4B samples。Attentive probing实验：8×NVIDIA A800 GPUs。框架：PyTorch + Flash Attention 2。训练策略：AdamW optimizer, Stage1 LR=0.001, Stage2 LR=0.0001, weight decay=0.2。

- 模型是什么。数据集和bench分别是什么。
  模型：OneVision-Encoder Large —— ViT-L/14, 24 transformer layers, hidden dim 1024, 16 attention heads, patch 14×14, MLP expansion ratio 4×, 3D RoPE (T:H:W=4:6:6)。GELU activations + LayerNorm。Attentive pooling head (multi-head attention based)。
  数据集：
  - Image: LAION-400M (250M), COYO-700M (400M), OBELICS (15M documents), Zero250M (15M curated), ImageNet-21K (14M)。总计约694M图像。
  - Video: HowTo100M (50M exovideo), Panda-70M (50M exovideo), Kinetics-710 (658K action), SSV2 (221K action)。总计约100M视频。
  - 标注pipeline：用frozen metaCLIP-H14提取特征，Union-Find去重，k-means聚类为2M图像类中心 + 400K视频类中心，每样本分配Top-10最近中心作为multi-label。OCR数据用PaddleOCR识别+分词，生成100 fine-grained tags。
  Benchmarks：
  - LMM probing: MVBench, MLVU-dev, NExT-QA, VideoMME, PerceptionTest, TOMATO, LongVideoBench-Val-Video, AI2D, ChartQA, DocVQA, InfoVQA, MMBench-EN, OCRBench, OCRBench v2, MMStar, RealWorldQA。
  - Attentive probing: SSV2, Diving48, PerceptionTest, CharadesEgo, Epic-Kitchens-100 (Verb/Noun), Kinetics-400, HMDB51。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  完全开源：Code https://github.com/EvolvingLMMs-Lab/OneVision-Encoder，Model https://huggingface.co/collections/lmms-lab-encoder/onevision-encoder，Data card https://github.com/EvolvingLMMs-Lab/OneVision-Encoder/blob/main/docs/data_card.md。

  算法pipeline伪代码（Codec Patchification核心流程）：

  ```
  # 输入: raw video V of T frames, GOP size K=32, sparsity ratio r, patch budget B=2048
  # 输出: selected visual tokens

  def codec_patchification(V, K=32, B=2048):
      # Step 1: HEVC编解码
      for each GOP in partition(V, K):
          I_frame, P_frames = HEVC_encode_decode(GOP)
          # I-frame: 完整RGB帧，P-frame: 含motion vectors + residuals

      # Step 2: 提取运动矢量和残差信号
      for each P-frame p in P_frames:
          # motion_vectors: shape [H, W, 2] (x, y 位移)
          # residuals: shape [H, W] (亮度残差能量)
          motion_magnitude[p] = L2_norm(motion_vectors[p], dim=-1)  # [H, W]
          residual_energy[p] = abs(residuals[p])                     # [H, W]

      # Step 3: patch-level saliency scoring
      patch_size = 14
      P0 = (H // patch_size) * (W // patch_size)  # = 256 per frame
      for each P-frame p:
          for each patch (i,j) in grid:
              # 在patch内聚合motion magnitude和residual energy
              saliency[p][i,j] = sum(motion_magnitude[p][patch_region])
                               + sum(residual_energy[p][patch_region])

      # Step 4: 全局Top-K选择（跨所有P-frames）
      I_frame_patches = patchify(I_frame)  # 所有P0个patches保留
      all_P_patches = concat([patchify(P_frames[p]) for p in P_frames])
      all_saliency = concat([saliency[p].flatten() for p in P_frames])

      # 全局排序，选top salient patches
      B_I = 512  # 2 I-frames × 256 patches
      B_P = B - B_I  # = 1536
      top_k_indices = argsort(all_saliency, descending=True)[:B_P]

      selected_P_patches = all_P_patches[top_k_indices]  # [B_P, patch_dim]

      # Step 5: 组装token序列
      tokens = concat([I_frame_patches, selected_P_patches])  # [B, patch_dim]
      # 每个保留patch记录原始时空坐标用于3D-RoPE
      positions = concat([I_frame_positions, P_frame_positions[top_k_indices]])
      # 部分未选中patch使用visible_indices机制

      # Step 6: ViT编码
      tokens = tokens + 3D_RoPE(positions)         # 位置编码
      features = ViT(tokens)                        # [B, D], 24 layers
      embeddings = attentive_pooling(features)      # [1, D] video-level

      # Step 7: 聚类判别损失
      # Image branch: contrast against 2M object centroids
      # Video branch: contrast against 400K motion centroids
      for m in {obj, vid}:
          similarity = embeddings @ centroids_m.T   # [1, K_m]
          loss_m = sigmoid_BCE(similarity, multi_labels_m)
      loss = loss_obj + loss_vid
  ```

  训练配置：
  - Stage 1（仅图像）: image resolution 224, AdamW LR=0.001, wd=0.2, k=2M classes, negative ratio r=0.1, positive labels l=10, 13B samples
  - Stage 2（图像+视频+OCR）: image res 448, video res 224, LR=0.0001, video:image ratio=1:1, 64帧clip, GOP=32, B=2048, 4B samples
  - 三种video processing modes混合batch: Codec 50%, Frame sampling 37.5%, Tiling 12.5%

  Token压缩比：64帧 × 256 patches/frame = 16,384 patches dense → 2048 tokens = 87.5% reduction。实际保留 3.1%-25% patches（对应budget 512-4096）。

## ReVisionLLM__Recursive_Vision-Language_Model_for_Temporal_Grounding_in_Hour-Long_Videos

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：ReVisionLLM —— 第一个能够对小时级长视频进行精确时序定位的VLM。核心创新包括：
  (1) **Hierarchical Adapter（层次化适配器）**：从CLIP ViT-L/14提取的视频帧特征（仅使用CLS token, 768维）通过滑动窗口分割为段（segment），每个段生成两种时间特征：
    - **稀疏时间特征（Sparse Temporal Features）**：通过Cross-Attention（视频段为query, 文本query为key）对齐跨模态语义，再经Self-Attention（学习型稀疏token + 文本对齐段特征concatenated）压缩为单个紧凑嵌入（768维），段级压缩比为L_w:1（如2分钟视频→1个768维向量）。
    - **密集时间特征（Dense Temporal Features）**：通过线性投影层 h_d: R^{L_w × 768} → R^{L_w × 4096}，将CLS token投影到LLM embedding空间，保留原始时间分辨率。
    公式：$\tilde{C}^i = \text{Cross-Attention}(C^i, Q)$, $\mathcal{S}^i = \text{Self-Attention}([S^i; \tilde{C}^i])_0$
  (2) **递归层次化处理（Recursive Hierarchical Processing）**：3个层次（L=3），顶层用稀疏特征扫描全视频（如150分钟→100个段），中间层聚焦感兴趣区域（约50分钟→33个段），底层用密集特征在选定段内精确定位起止时间（250帧密集特征）。每层LLM输出形式为 "From s to e" 或 "Not Present."。
  (3) **渐进式训练策略（Progressive Training）**：
    - Stage 1（短片段训练）：先用密集特征微调LLM（LoRA, r=64, α=128）进行精确边界预测；再冻结LLM，微调Hierarchical Adapter生成稀疏特征，引入Contrastive Segments（不含目标事件的负样本段），训练目标简化为判断事件存在性（"Yes"/"No"），改善LLM置信度校准。
    - Stage 2（长视频训练）：冻结Hierarchical Adapter，使用稀疏特征识别小时级视频中的相关段，仅微调新的LoRA模块。
  (4) **LLM置信度校准排序（Calibrated Confidence）**：替代传统CLIP相似度排序，使用LLM生成每个词的概率分布计算平均熵的倒数作为置信度分数：$R^i = 1 / (\frac{1}{K} \sum_{k=1}^K H_k^i)$，其中 $H_k^{(i)} = -\sum_w p(w|T_{<k}, \mathcal{D}^{(i)}) \log p(w|T_{<k}, \mathcal{D}^{(i)})$。按置信度排序选Top-K预测。

  实验比较：
  (a) **Main Results（Table 1）** —— MAD数据集（R1@.1, R5@.1, R1@.3, R5@.3）和 VidChapters-7M 数据集（R1@.5, R5@.5, R1@.3, R1@.5, R1@.7, R1@.9）。对比非LLM方法：M-Guide, CONE, SOONet, SnAG, RGNet, M-DETR, CLIP, BERT；以及VLM baseline：VTimeLLM + CONE。ReVisionLLM在MAD上R1@.1=15.0%（+2.6% vs RGNet），ReVisionLLM-I达R1@.1=17.3%（+4.9%）。
  (b) **Ablation on Modules（Table 2）** —— 累积消融：VTimeLLM baseline → +CONE → +Contrastive Segments → +Calibration → +Recursive Process。递归处理贡献最大增益（R1@.1: 8.4%→15.0%）。
  (c) **Ablation on Model Variants（Table 3）** —— 对比4个变体：ReVisionLLM（默认, Top-to-Bottom, 57% frames），ReVisionLLM-U（统一权重共享, 159M vs 363M params），ReVisionLLM-I（Bottom-to-Top, 100% frames, 最高精度R1@.1=17.4%），ReVisionLLM-(U+I)（统一+逆序, 100% frames）。
  (d) **Ablation on Video Length（Figure 5）** —— 视频长度从2h扩展到10h，递归方法保持稳定性能，非递归方法在10h完全失败。
  (e) **Ablation on Number of Hierarchies（Table 4）** —— 0/1/2/3层对比，0层失败（R1=0），3层最佳。
  (f) **Calibration ECE（Table S1）** —— 对比VTimeLLM+CONE和ReVisionLLM的Expected Calibration Error（ECE@IoU=0.1/0.3/0.5），ReVisionLLM持续低ECE（0.46 vs 0.62）。
  (g) **Generalization: Text-to-Video Retrieval（Table 5）** —— MSRVTT数据集上R@1=49.1, R@5=77.5, R@10=85.7，与SOTA方法X-Pool, DiffusionRet, UATVR, TEFAL, CLIP-ViP, T-MASS 对比。

- 硬件平台是什么，配置是什么。
  训练：8×NVIDIA A100 GPUs，总batch size=128。Stage 1（短片段）：5 epochs for MAD / 1 epoch for VidChapters-7M。Stage 1 adapter训练：1 epoch, batch size=32, LR=1×10⁻³。Stage 2（长视频）：2 epochs, batch size=8, LR=1×10⁻⁴。优化器：AdamW, cosine LR decay, warmup ratio=0.03。LoRA配置：r=64, α=128。预训练linear projector使用LCS-558K数据集（来自LLaVA），1 epoch, batch size=128, LR=1×10⁻³。

- 模型是什么。数据集和bench分别是什么。
  模型：ReVisionLLM —— 三组件架构：(1) Frozen CLIP ViT-L/14 vision encoder (24-layer transformer, 224×224 input, CLS token only per frame) + Frozen CLIP text encoder (12-layer transformer)。(2) Hierarchical Adapter —— Cross-Attention (2 layers) + Self-Attention (2 layers) + Feed-Forward Network + Linear Projection (768→4096 to match LLM embedding space)。(3) Vicuna-7B v1.5 LLM (32 transformer layers, based on LLaMA), LoRA fine-tuned。
  变体：ReVisionLLM-U (unified shared weights, 159M trainable), ReVisionLLM-I (inverse bottom-to-top), ReVisionLLM-(U+I) (unified + inverse)。
  数据集：
  - MAD：约1,200小时完整电影，384K自然语言query，平均视频110分钟，平均moment仅4.1秒（极低moment-to-video比）。
  - VidChapters-7M：817K视频，7M+ user-annotated chapters，最长12小时，每视频2-30个chapters（1秒至10分钟不等）。
  - MSRVTT（泛化实验）：10K视频，每视频20 captions，10-32秒长度。
  Benchmarks/metrics：
  - 主要：Rk@θ (Recall@k at IoU=θ)，MAD上使用R1@.1, R5@.1, R1@.3, R5@.3，VidChapters-7M上使用R1@.5, R5@.5, R1@.3, R1@.5, R1@.7, R1@.9。
  - 校准：ECE (Expected Calibration Error) at IoU thresholds。
  - 泛化：R@k (Recall at Rank k) on MSRVTT。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源：代码 https://github.com/Tanveer81/ReVisionLLM（论文声称开源，需验证最新状态）。

  算法pipeline伪代码：

  ```
  # ===== Stage 1: Short Segment Training =====
  # 输入: video V of T frames, text query Q, ground truth τ=(s,e)
  # 输出: trained LoRA module + Hierarchical Adapter

  # Step 1: Multimodal Encoding
  F = CLIP_ViT_extract_CLS(V)  # shape: (T, 768), 每帧仅取CLS token
  Q_feat = CLIP_Text_encode(Q)  # shape: (N_s, 768)

  # Step 2: Segment Partition
  C = sliding_window(F, L_w=125s, stride=25s for MAD)  # (|C|, L_w, 768)
  # 每段均匀采样250 frames

  # Step 3: Dense Feature Generation (for bottom hierarchy)
  D_i = LinearProjection(C_i)  # (L_w, 768) → (L_w, 4096)
  # 训练目标: LLM预测 From s to e, 使用dense特征

  # Step 4: Sparse Feature Generation (for upper hierarchies)
  for C_i in C:
      # Cross-Attention: 视频段对齐文本query
      C_tilde_i = CrossAttention(query=C_i, key=Q_feat, value=Q_feat)
      # Self-Attention: 压缩段信息到学习型稀疏token
      S_i_learnable = learnable_token  # shape: (1, 768)
      A = SelfAttention(concat([S_i_learnable, C_tilde_i]))
      S_i = A[0]  # shape: (768,), 紧凑段表示

  # Step 5: Contrastive Training (improve calibration)
  # 正样本: 包含目标事件的段
  # 负样本: 随机采样不包含目标事件的段
  # 训练目标: "Does <event> happen? Answer yes or no."
  # 正样本→Yes, 负样本→No

  # ===== Stage 2: Long Video Training =====
  # 冻结Hierarchical Adapter, 训练新LoRA

  # Step 6: Hierarchical Processing (Inference)
  # Hierarchy ℓ=3 (top): 扫描全视频
  segments_coarse = sliding_window(V, L_w=125s)
  S_coarse = HierarchicalAdapter.get_sparse(segments_coarse)
  # LLM input: [S_coarse, prompt]
  τ_3 = LLM_predict(S_coarse, "From s to e")  # 粗粒度边界

  # Hierarchy ℓ=2 (mid): 聚焦τ_3附近区域
  C_focused = get_segments_around(τ_3)  # 约33个段
  S_focused = HierarchicalAdapter.get_sparse(C_focused)
  τ_2 = LLM_predict(S_focused, "From s to e")  # 中等粒度

  # Hierarchy ℓ=1 (bottom): 精确边界定位
  C_precise = get_segments_around(τ_2)
  D_precise = HierarchicalAdapter.get_dense(C_precise)  # 250帧密集特征
  τ_1 = LLM_predict(D_precise, "From s to e")  # 精确边界: 秒级精度

  # Step 7: Confidence Calibration & Ranking
  for each prediction i:
      for each generated word k:
          H_k_i = -sum_w p(w|T_<k, D_i) * log(p(w|T_<k, D_i))
      R_i = 1 / mean(H_i)  # 置信度 = 熵的倒数
  top_k = argsort([R_i], descending=True)[:k]
  ```

  张量计算示例（稀疏特征生成）：
  ```
  # 输入
  C_i ∈ R^{250 × 768}   # 视频段特征 (250帧, 768维)
  Q ∈ R^{N_s × 768}     # 文本query特征
  S ∈ R^{1 × 768}       # 可学习稀疏token

  # Cross-Attention
  # Q_proj: C_i × W_q → (250, 768)
  # K_proj: Q × W_k → (N_s, 768)
  # V_proj: Q × W_v → (N_s, 768)
  Attn = softmax(Q_proj @ K_proj^T / sqrt(768))  # (250, N_s)
  C_tilde_i = Attn @ V_proj  # (250, 768)

  # Self-Attention with learnable sparse token
  X = concat([S, C_tilde_i])  # (251, 768)
  # Multi-head Self-Attention (2 heads, 2 layers)
  output = SelfAttention_2layers(X)  # (251, 768)
  S_i = output[0]  # (768,) -- 稀疏特征，压缩比250:1
  ```

## Owl-1__Omni_World_Model_for_Consistent_Long_Video_Generation

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：Owl-1 (Omni World ModeL) —— 一个基于世界模型的长视频生成框架，通过建模底层世界演化来产生长期一致的长视频。核心创新包括：
  (1) **Omni World Model（全向世界模型）**：构建闭环的 state-observation-dynamics 三元组 (Eq. 1-3)，模拟世界的演化过程。latent state variable s_t 编码当前时刻和历史信息，通过 state decoder D（视频扩散模型）解码为显式视频观测 o_t；从观测和状态预测未来世界动态 d_t；动态再反过来更新状态变量 s_{t+1} = g(s_t, d_t)，形成自回归闭环。
  (2) **Comprehensive Condition from Latent State**：latent state s_t 由所有历史观测推导而来（Eq. 4），作为下一轮生成的综合条件，相比传统方法仅用 last-frame 条件，具有更长的时序感受野和长期一致性。
  (3) **Anticipation of Future Dynamics**：显式预测未来世界动态 d_t（文本形式），嵌入到状态演化中，解决传统方法重复生成同质内容的问题，提升内容的多样性和可控性。
  (4) **Multi-Stage Training（三阶段训练）**：
    - Stage 1 (Alignment)：冻结视频扩散模型，仅训练LMM，用MSE loss对齐 latent state s_t 与视频扩散模型 text encoder 的文本特征（Eq. 7）。
    - Stage 2 (Generative Pretraining)：联合微调LMM和视频扩散模型，用latent state s_t 替代原始text condition输入视频扩散模型进行denoising训练（Eq. 8）。
    - Stage 3 (World Model Training)：在带dense caption的长视频数据上微调，加入world dynamics prediction，用next-token prediction teacher-forcing监督（Eq. 9）。

  实验比较：
  (a) **VBench-I2V（Table 1）** —— 2s短视频，8个维度评估。对比 VideoCrafter-I2V、ConsistI2V、SEINE-512x512、I2VGen-XL、Animate-Anything、SVD-XT-1.0、DynamiCrafter-1024。Owl-1 Total Score=89.15，在 Motion Smoothness (98.92) 和 Temporal Flickering (98.69) 上表现优异。
  (b) **VBench-Long（Table 2）** —— 7s长视频，16个维度评估。对比 Mira、OpenSoraPlan、OpenSora、Mochi-1、CogVideoX、Kling、Vchitect-2.0、Gen-3、MiniMax。Owl-1 在 Subject Consistency (98.29) 和 Background Consistency (98.61) 上为 best/open-source best。
  (c) **定性可视化（Figure 4, 5, 6）** —— 展示 8s 通用视频生成和 24s (3 scenes × 8s) 世界模型驱动的长视频生成效果，验证 state variable 在跨场景 transitions 时的 consistency。

- 硬件平台是什么，配置是什么。
  训练平台：8 × NVIDIA A800 GPUs（每张80G显存）。Stage 1 (Alignment) 训练1天，Stage 2 (Generative Pretraining) 训练5天，Stage 3 (World Model Training) 训练1天，共约7天。框架：PyTorch。

- 模型是什么。数据集和bench分别是什么。
  模型：LMM = Chameleon model [31]，Video Diffusion Model = DynamiCrafter-1024 [35]。LoRA fine-tuning LMM（rank=8），全参数微调视频扩散模型。总可训练参数约2B（LMM LoRA ~798M + DynamiCrafter全参 ~1.2B）。Learnable state queries s_t 长度128。每视频分割为4s clips作为observation o_t，每clip采样2帧输入LMM。
  数据集：
  - General video generation: WebVid（400K random samples, 10M+ videos, 52K hours）+ Panda70m（2M random samples, 70M videos, 平均8s）
  - Dense video captioning: ActivityNet Captions（20K videos, 100K captions, 平均120s）+ Vript（12K high-res videos, 400K segments, 密集script标注）
  Benchmarks：
  - VBench-I2V: 8 dimensions（Video-Image Subject/Background Consistency, Subject/Background Consistency, Motion Smoothness, Dynamic Degree, Aesthetic Quality, Imaging Quality, Temporal Flickering）
  - VBench-Long: 16 dimensions（含Subject Cons., Background Cons., Temporal Flickering, Motion Smoothness, Dynamic Degree, Aesthetic Quality, Imaging Quality, Object Class, Multiple Objects, Human Action, Color, Spatial Relationship, Scene, Appearance Style, Overall Consistency）

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源：Code https://github.com/huang-yh/Owl（论文明确提供GitHub链接）。Chameleon LMM和DynamiCrafter-1024均为公开模型。

  算法pipeline伪代码（Owl-1 推理生成流程）：

  ```
  # 输入: starting image I, text description d_0, num_clips N
  # 输出: long video consisting of N clips

  def owl1_long_video_generation(I, d_0, N):
      # Step 1: 用图像扩散模型生成首帧
      first_frame = SD2.1_v(I, d_0)

      # Step 2: 初始化latent state
      s_0 = LMM.encode_state(I, d_0, learnable_queries)  # shape: [128, dim]

      # Step 3: 生成首段视频clip
      o_0 = VideoDiffusion.decode(state=s_0, image_cond=first_frame, prev_obs=None)
      # VideoDiffusion: 以 s_0 替代原始text condition
      # noise_pred = epsilon_theta(o_{t,m}, m, s_0, None)
      # 通过L2 denoising loss优化后采样得到o_0

      # 构建初始LMM输入序列
      seq = [I_tokens, d_0_tokens, s_0_queries, o_0_visual_tokens, d_0_tokens]

      video_clips = [o_0]

      for t in range(1, N):
          # Step 4: 预测世界动态 (Eq. 2)
          d_t = LMM.predict_dynamics(s_{t-1}, o_{t-1})
          # d_t = f(s_{t-1}, o_{t-1}), teacher-forcing style next-token pred

          # Step 5: 更新latent state (Eq. 3)
          s_t = LMM.update_state(s_{t-1}, d_t)
          # s_t = g(s_{t-1}, d_t), 聚合历史信息

          # Step 6: 解码为视频观测 (Eq. 1)
          # 若跨场景切换则丢弃image_cond，仅用s_t作为条件
          if scene_transition:
              o_t = VideoDiffusion.decode(state=s_t, image_cond=None,
                                          prev_obs=o_{t-1})
          else:
              last_frame = o_{t-1}.last_frame()
              o_t = VideoDiffusion.decode(state=s_t, image_cond=last_frame,
                                          prev_obs=o_{t-1})
              # noise_pred = epsilon_theta(o_{t,m}, m, s_t, o_{t-1})

          # Step 7: 追加到序列
          o_t_tokens = VQVAE.encode(uniform_sample_key_frames(o_t, K=2))
          seq.extend([s_t_queries, o_t_tokens, d_t_tokens])
          video_clips.append(o_t)

      return concat(video_clips)
  ```

  张量计算流程（前向pass）：
  - LMM输入序列 Seq = [..., s_t_queries (128×dim), o_t_vq_tokens (2帧×N_tokens/帧), d_t_text_tokens, ...]
  - LMM对序列做causal self-attention，输出updated s_{t+1} queries 和 predicted d_t tokens
  - Video Diffusion Model接收 s_t (128×dim) 作为cross-attention condition，替代原始text embedding
  - Denoising: 从随机噪声 z_T 开始，每一步 z_{m-1} = denoise(z_m, m, cross_attn(s_t), concat_image_cond(o_{t-1})), 最终 z_0 = o_t

## Molmo2__Open_Weights_and_Data_for_Vision-Language_Models_with_Video_Understanding_and_Grounding

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：Molmo2 —— 一个全开源（权重、数据、代码）的视觉语言模型家族（4B/8B基于Qwen3，7B基于OLMo3），支持单图、多图、视频输入，具备像素级视觉grounding能力（点标注、计数、目标跟踪）。核心设计：
  (1) 架构：SigLIP 2 So400m/14 384px ViT → Connector (MLP + multi-head attention pooling, 2×2 for image / 3×3 for video) → LLM (Qwen3或OLMo3)，vision tokens之间启用双向注意力（bi-directional attention），视频帧以2 fps采样（max 128帧SFT / 384帧long-context），帧间插入文本时间戳标记，多图使用"Image N"标记。
  (2) 三阶段训练pipeline：① Pre-training（仅图像，32k steps, batch 128, seq 2560）：PixMo-Cap captioning (60%) + PixMo image pointing (30%) + NLP (10%)；② SFT（联合视频/图像/多图，30k steps, batch 128, seq 16384）：7大类数据混合（Table 1），人工定义采样率；③ Long-context SFT（同数据mix, seq 36864, 384 frames, 2k steps, context parallelism Ulysses attention 8 GPUs/example）。
  (3) 训练技术创新：token weighting（video caption weight=0.1, pointing weight=0.2, 其他 √(4/n)策略平衡长短输出）；序列packing（动态规划solver pool=48, 平均3.8 examples/packed sequence, ~15x训练效率）；message-tree编码（多annotations用custom attention mask防止跨分支attention）；pointing预训练（pre-training阶段引入pointing数据稳定效果）；point坐标用压缩HTML-like纯文本格式（<points coords="timestamp obj_id x y;...">），比JSON大幅减少token数。
  (4) SlowFast encoding（推理时）：default pooling 3×3 slow frames + 9×9 fast frames + query-based frame selection，在~43% fewer visual tokens下匹配224 frames性能。
  (5) 9个新数据集（7视频+2多图）：Molmo2-Cap (104k dense video captions, avg 924 words/video)、Molmo2-AskModelAnything (140k human QA)、Molmo2-CapQA (1M synthetic QA)、Molmo2-SubtitleQA (300k subtitle QA)、Molmo2-VideoPoint (650k space-time points)、Molmo2-VideoTrack (15k complex queries)、AcademicVideoPoint/AcademicVideoTrack (repurposed)、Molmo2-MultiImageQA (72k QA)、Molmo2-MultiImagePoint (470k points)。

  实验比较：
  (a) 视频理解（Table 2）—— 12视频benchmarks (NextQA, PerceptionTest, MVBench, Tomato, MotionBench, TempCompass, Video-MME, Video-MME-Sub, LongVideoBench, MLVU, LVBench, VideoEvalPro) + 自建Molmo2-CapTest caption F1 + Molmo2-VideoCount accuracy + human ELO。对比GPT-5/GPT-5 mini/Gemini 2.5/3 Pro/Claude Sonnet 4.5 (API)、InternVL3.5/Qwen3-VL/Keye-VL-1.5/GLM-4.1V/MiniCPM-V-4.5/Eagle2.5 (open-weight)、PLM/LLaVA-Video/VideoChat-Flash (open models)。
  (b) 视频Grounding（Table 3-5）—— 视频counting (BURST-VC, Molmo2-VideoCount)、视频pointing (Molmo2-VideoPointVal F1)、视频tracking (MeViS/Ref-YT-VOS/Ref-Davis/ReasonVOS/Molmo2-Track, J&F/F1/HOTA)。对比GPT-5/Gemini/Qwen3-VL/VideoLISA/VideoGLaMM/Sa2VA/VideoMolmo/SAM 3。
  (c) 图像Benchmark（Table 6）—— 11图像benchmarks (AI2D/ChartQA/DocVQA/InfoQA/TextVQA/VQA v2/RWQA/MMMU/MathVista/CountBench/PixMoCount) + MuirBench + MMIU + Blink。
  (d) 图像Pointing（Table 7）—— Point-Bench (Affordance/Spatial/Reasoning/Steerability/Counting)。
  (e) Ablations（Table 8-11, 18）—— 视频消融、counting/pointing消融、tracking消融、long-context SFT消融、pre-training pointing消融。

- 硬件平台是什么，配置是什么。
  训练：Nvidia H100 GPU。4B: pre-train 32 GPUs/15.2h, SFT 128 GPUs/58.8h (7.5k GPU hr), long-context 128 GPUs/25.3h。8B: pre-train 64 GPUs/12.1h, SFT 128 GPUs/63.0h (8.1k GPU hr), long-context 128 GPUs/26.0h。总计约11k GPU hours for 8B。
  框架：PyTorch + FSDP2 + SDPA (非FlashAttention, 因custom attention mask) + torch.compile (静态shape) + AMP bfloat16。
  数据加载：torchcodec抽帧。On-the-fly packing算法集成入PyTorch DataLoader。

- 模型是什么。数据集和bench分别是什么。
  模型：Vision Encoder —— SigLIP 2 So400m/14 384px (380M params, 27 layers, dim 1152, 16 heads)。Connector —— MH attentional pooling (image 2×2, video 3×3) + MLP SwiGLU projection → LLM dim。LLM —— Qwen3-4B (36 layers, dim 2560, 8 KV heads), Qwen3-8B (36 layers, dim 4096, 8 KV heads), OLMo3-7B (32 layers, dim 4096, 32 KV heads)。Optimizer: AdamW β=(0.9,0.95), separate LR: pre-train (ViT 6e-6/Connector 2e-4/LLM 2e-4), SFT (ViT 5e-6/Connector 5e-6/LLM 1e-5)。Max crops: train K=8 / inference K=24。
  训练数据：PixMo + Molmo2自建9数据集 + 开源image/video/NLP数据集(Table 13, 100+子数据集)，总training examples约8M+（图像QA 2.4M + 视频QA 2.4M + 图像Pointing 1.1M + 视频Pointing 0.37M + 视频Tracking 0.80M + Captions/LongQA 1.2M + NLP 0.99M）。
  Benchmarks：视频理解12项 + 视频grounding 6项 + 图像理解11项 + 图像pointing Point-Bench + 多图3项 + NLP 4项 + 人类偏好ELO。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  完全开源：Code https://github.com/allenai/molmo2，模型权重、训练数据、训练代码全部开源（不含closed VLMs蒸馏）。HuggingFace + vLLM集成。Demo: playground.allenai.org。

  算法pipeline伪代码（SFT训练核心流程）：
  ```
  # === 视频处理 ===
  frames = torchcodec.extract(video, fps=2, max_frames=128)
  # 每帧384×384 → ViT SigLIP 2 → 27 layers
  # → 取第3层和第9层hidden states concat
  # → MH pooling: 3×3 window, mean as query
  # → 每帧约81 visual tokens

  # === 模型前向 ===
  # Vision token序列: [video_start] frame1_tokens <t0.0> frame2_tokens <t0.5> ... [video_end]
  # Bidir attention: vision tokens互相attend (cross-frame/image)
  # Text tokens: causal attention + attend to all vision tokens

  # === Packing & Message Trees ===
  # DP solver选择最优packing组合, pool_size=48
  # Message tree: 同一visual input多个annotations → 分支
  # Custom attention mask防止cross-branch attention
  # 平均3.8 examples/packed sequence, 15x efficiency

  # === Token Weighted Loss ===
  if is_video_caption:
      weight = 0.1
  elif is_pointing:
      weight = 0.2
  else:
      weight = 4.0 / sqrt(n_answer_tokens)
  loss = weight * cross_entropy(logits, labels)

  # === Point Format (压缩版) ===
  # <points coords="ts obj_id x y;...">inline text</points>
  # <tracks coords="ts obj_id x y;...">inline text</tracks>
  # ts: seconds to 1 decimal, (x,y): 0-1000 normalized
  # obj_id: sequential starting at 1, used for tracking/counting
  ```

  张量计算示例（128 frame video, Molmo2-8B）：
  ```
  ViT per frame: 384×384 → 27×27=729 patches
  Pooling: 3×3 × 16 MH heads → 81 tokens/frame
  Total vision tokens = 128 × 81 ≈ 10,368
  LLM (Qwen3-8B): 36 layers, dim 4096, 8 KV heads
  Bidir vision attention block: [10368, 10368] → FLOPs ≈ O(10368²×4096)
  Causal text block: [1000, 11368] → standard causal attn
  Total FLOPs dominated by bidir vision block
  ```

## LLaVA-Mini__Efficient_Image_and_Video_Large_Multimodal_Models_with_One_Vision_Token

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：LLaVA-Mini —— 引入 query-based compression 和 modality pre-fusion 两个模块，将输入 LLM backbone 的 vision token 从 576 压缩到 1 个（压缩率 0.17%），同时保持与 LLaVA-v1.5 可比的多模态理解性能。
  核心设计：(1) Query-based Compression —— 引入 C×C 个可学习压缩 query Q^v，通过 cross-attention 与全部 N^2 个 vision token 交互，使用 2D sinusoidal positional encoding 保留空间信息，产生 C^2 个压缩后 vision token Ĥ^v。C 默认设为 1（标准分辨率）或 8（高分辨率 HD 模式）。(2) Modality Pre-fusion —— 在 LLM backbone 之前，用 N_fusion=4 个与 LLM 同构的 Transformer decoder 块，将全部 vision token 和 text token 拼接后输入，提取文本位置的输出作为 fusion token Ĥ^q，提前将视觉信息融入文本表示。
  训练两阶段：(1) Stage 1 Vision-Language Pretraining：仅训练 projection layer，冻结 vision encoder 和 LLM，使用 558K caption 数据。(2) Stage 2 Instruction Tuning：引入 compression + pre-fusion 模块，除 vision encoder 外全可训练，使用 665K instruction 数据。增强变体额外加入 100K Video-ChatGPT 视频数据及开源数据共 3M samples。

  实验比较：
  (a) 图像理解 —— 11 benchmarks (VQA-v2, GQA, VisWiz, SciQA-IMG, TextVQA, POPE, MME, MMBench, SEED-Bench, LLaVA-Bench-in-the-Wild, MM-Vet)，对比 LLaVA-v1.5 以及 BLIP-2, InstructBLIP, IDEFICS, Qwen-VL, SPHINX, mPLUG-Owl2 等，还有 token 压缩方法 MQT-LLaVA, PruMerge/PruMerge++, LLaMA-VID, VoCo-LLaMA, TokenPacker。
  (b) 视频理解 —— 5 video QA benchmarks (MSVD-QA, MSRVTT-QA, ActivityNet-QA) + video-based generative performance benchmark + MVBench 20 子任务，对比 Video-ChatGPT, Video-LLaVA, Video-LLaMA, LLaMA-VID 等。
  (c) 长视频 —— MLVU 和 EgoSchema，对比 MovieChat, MA-LMM, TimeChat 等。
  (d) 效率分析 —— FLOPs (calflops) + latency (A100, no engineering acceleration) + VRAM usage，对比 LLaVA-v1.5。
  (e) 消融实验 —— Modality pre-fusion 层数 (0/1/2/3/4)、vision token 数量 (1/4/16/64/144/576)、query-based compression vs average pooling、compression 与 pre-fusion 在 LLM 内外执行对比、纯 pre-fusion 无 compression 效果。
  (f) 跨硬件效率 —— RTX 3090 (24G), A100 (40G), A800 (80G) 延迟测试。
  (g) 各组件 FLOPs 分解 —— Vision Encoder / Projection / Compression / Pre-fusion / LLM。

- 硬件平台是什么，配置是什么。
  训练：8 NVIDIA A800 GPU。Batch size 256，Stage 1 1 epoch，Stage 2 2 epochs。Optimizer AdamW，learning rate 1e-3 (Stage 1 projection) / 1e-4 (Stage 2 LLM)，cosine decay schedule，warmup ratio 0.03。
  推理延迟测试：NVIDIA A100 (40G)，无工程加速技术。跨硬件延迟测试额外含 RTX 3090 (24G) 和 A800 (80G)。
  VRAM 测试：RTX 3090 (24G) 处理 3 小时视频。

## Machine_Mental_Imagery__Empower_Multimodal_Reasoning_with_Latent_Visual_Tokens

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：Mirage —— 在两阶段微调范式下，让 VLM 在文本 token 之间插入紧凑的 latent visual tokens（压缩后的隐空间视觉特征向量），替代显式图像生成，实现多模态推理链。核心设计：
  (1) 数据合成：对每个任务用 task-specific tool 生成 helper image I（如 VSP 用 OpenAI Gym 渲染 annotated map，SAT 用 CogVideoX-5B 生成场景帧），再用大推理 VLM (Qwen2.5-VL 32B) 基于 x + I + y 生成 interleaved reasoning chain o = o_pre ⊕ I ⊕ o_post，构成训练集 D = {(x, I, o, y)}。
  (2) Stage 1 — Joint Supervision for Latent Grounding：将 helper image I 输入 VLM 得到 patch-level features {e_1,...,e_n}，通过 average pooling 压缩为 k 个 salient vectors {ê_1,...,ê_k}。训练目标 L_1 = L_visual + γ·L_text，其中 L_visual = ℓ_cos(ê_j, g_θ(o_pre, ê_{1:j-1})) 用 cosine similarity 强制隐层状态逼近压缩后的 visual embedding，L_text 为左右两段文本的标准 cross-entropy。γ=0.1 控制 visual alignment loss 权重。
  (3) Stage 2 — Text-Only Supervision with Latent Relaxation：移除 L_visual，仅保留文本 CE loss。模型自回归生成自己的 latent tokens {e_i} = f_θ(x, o_pre, e_{<i})，梯度通过 o_post 的文本 CE loss 反向传播到这 k 个连续 latent embeddings，使其在 visual subspace 内自适应优化，不再强制匹配固定的 compressed image embeddings。
  (4) Stage 3 — Reinforcement Learning (GRPO)：使用 VERL 框架 + GRPO，优化文本 token 概率同时允许梯度流经 latent tokens。奖励 = σ_f·r_format + σ_c·r_correct（σ_f=0.1, σ_c=0.9），KL 正则系数 λ_kl=0.01。rollout num=5, mini batch=8。

  实验比较：
  (a) VSP（Spatial Reasoning + Spatial Planning, Level 3-6）—— 对比 Zero-Shot, Direct SFT, CoT SFT, GRPO, CoT SFT+GRPO, Anole, MVoT。Mirage (Direct) 在 Spatial Reasoning 86%、Spatial Planning 76%；+GRPO 后提升至 89% 和 60%。
  (b) COMT（Math Geometry, 200 test）、BLINK-Jigsaw（150 test）、SAT（Synthetic GoalAim/ObjM + Real, 500 test）—— 对比 Zero-Shot, Direct SFT, CoT SFT, GRPO, SFT+GRPO。Mirage 在 COMT 77%, Jigsaw 88%, SAT Avg 98% (Synthetic) / 72% (Real)。
  (c) 模型规模泛化 —— Qwen2.5-VL 3B 上重复 COMT/Jigsaw/SAT 实验，Mirage 相比 text-only baseline 在 Jigsaw 上 +5%, SAT Real 上 +10%。
  (d) 消融实验 —— (i) 训练阶段：w/o Stage 1：VSP Spatial Planning 降至 52% Avg；w/o Stage 2 (仅 Stage 1)：降至 21% Avg。(ii) latent token size k ∈ {2,4,6,8}：k=4 和 k=6 最优（87-88%），k=8 骤降至 75%。(iii) loss coefficient γ ∈ {0.1,0.5,1}：γ=0.1 最优 (87%)，增大 γ 相当于弱化 visual alignment。(iv) helper image as prior 数据质量验证：配合 helper image 直接输入 fine-tuned 模型可接近 100% 准确率。
  (e) 潜变量行为分析 —— t-SNE 可视化 100 samples 的 latent tokens 与 text/image embeddings：latent tokens 聚集在 visual cluster 外侧，与 text distribution 明显分离，验证 Stage 2 在 visual submanifold 内的灵活偏移。

- 硬件平台是什么，配置是什么。
  训练：单个 NVIDIA H100 GPU。Stage 1 约 3.5 hours，Stage 2 约 7.2 hours（VSP Spatial Reasoning 为例）。text-only CoT SFT 约 5.5 hours 作为参考。
  推理：论文未明确说明推理平台，但基于 Qwen2.5-VL 7B 规模，可在单 H100/A100 GPU 推理。

- 模型是什么。数据集和bench分别是什么。
  模型：Base VLM — Qwen2.5-VL-7B-Instruct（默认），小模型验证使用 Qwen2.5-VL-3B-Instruct。Vision encoder 部分冻结不训练。Loss weight γ=0.1，latent token size k=4 (default)。
  数据生成模型：外部推理 VLM — Qwen2.5-VL-32B 生成 textual thoughts。Helper image 生成 — OpenAI Gym (VSP map rendering), CogVideoX-5B (SAT video generation, sampling 9 frames + VLM selects most informative frame)。
  数据集配置 (Tab 8)：
  - VSP Spatial Reasoning: #SFT 3000, #RL 2000, #Test 400
  - VSP Spatial Planning: #SFT 3000, #RL 2000, #Test 400
  - BLINK Jigsaw: #SFT 1000, #RL 2000, #Test 150
  - SAT (GoalAim + ObjM): #SFT 1000, #RL 2000, #Test 500
  - COMT Math Geometry: #SFT 820, #RL -, #Test 200
  Benchmarks: VSP (spatial planning + spatial reasoning sub-tasks, binary→three-way extended), BLINK-Jigsaw (visual extrapolation), SAT (static + dynamic spatial relations, GoalAim/ObjM subtasks, Synthetic + Real split), COMT (Mathematical Geometry subset)。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址：https://github.com/UMass-Embodied-AGI/Mirage
  代码已开源，论文给出了 Method 章节的完整数学公式推导。

  算法 pipeline 执行例子（以 VSP Spatial Reasoning 为例，k=4 latent tokens, Qwen2.5-VL 7B）：
  ```
  # === Stage 1: Joint Supervision ===
  Input: x = (map_image, text_query), helper image I
  # 1. Extract & compress helper image features
  patch_features = VLM.vision_encoder(I)  # {e_1, ..., e_n}, n patches
  latent_target = avg_pool(patch_features, k=4)  # {ê_1, ê_2, ê_3, ê_4}

  # 2. Forward pass with interleaved training
  # o = o_pre ⊕ <latent_slot> ⊕ o_post
  h_pre = VLM.forward(x, o_pre)                    # text tokens before latent
  h_1 = VLM.forward(x, o_pre, <bos_latent>)         # hidden state → latent token 1
  h_2 = VLM.forward(x, o_pre, h_1, <bos_latent>)    # hidden state → latent token 2
  h_3 = VLM.forward(x, ..., h_2, <bos_latent>)      # hidden state → latent token 3
  h_4 = VLM.forward(x, ..., h_3, <bos_latent>)      # hidden state → latent token 4
  logits_post = VLM.forward(x, o_pre, h_{1:4}, o_post)

  # 3. Loss computation
  L_visual = avg([cos_sim(ê_j, h_j) for j in 1..4])  # Eq.1, cosine similarity
  L_text   = CE(o_pre) + CE(o_post)                   # Eq.2, cross-entropy
  L_1      = L_visual + 0.1 * L_text                  # γ = 0.1

  # === Stage 2: Text-Only Supervision ===
  # Latent tokens are self-generated (no external ê_j)
  e_1 = VLM.hidden_state(x, o_pre)                    # Eq.3
  e_2 = VLM.hidden_state(x, o_pre, e_1)               # autoregressive latent gen
  e_3 = VLM.hidden_state(x, o_pre, e_1, e_2)
  e_4 = VLM.hidden_state(x, o_pre, e_1, e_2, e_3)
  logits_post = VLM.lm_head(VLM.forward(x, o_pre, e_{1:4}))
  L_2 = CE(o_pre) + CE(o_post | e_{1:4})             # Eq.4
  # Gradients flow back to e_{1:4} through o_post CE loss

  # === Stage 3: GRPO RL (VERL framework) ===
  for each query in RL_train:
      samples = [VLM.generate(x) for _ in 1..5]  # rollout_n=5
      for each sample:
          r = 0.9 * (1 if answer_correct else 0) + 0.1 * (format_correct)
      # Group Relative Policy Optimization on text tokens
      # Latent tokens receive gradient but are excluded from KL penalty
  ```

  Adam optimizer, β1=0.9, β2=0.95, weight_decay=0.01, lr=1e-5, batch_size=8, grad_accum=2, warmup=10 steps, epochs=10. SFT LR 1e-5, RL LR 1e-6. Trainable: all except vision encoder.

- 模型是什么。数据集和bench分别是什么。
  模型：Vision Encoder — CLIP ViT-L/336px (Radford et al., 2021)。LLM Backbone — Vicuna-v1.5-7B (默认)，增强变体使用 LLaMA-3.1-8B-Instruct。Vision token 压缩参数 C=1 (标准分辨率 336×336) 或 C=8 (高分辨率 672×672, LLaVA-Mini-HD)，pre-fusion 层数 N_fusion=4。
  训练数据：Stage 1 — 558K caption data (LLaVA pretraining data)。Stage 2 — 665K instruction data (LLaVA instruction data)。增强变体额外使用 100K Video-ChatGPT instruction data + 部分开源数据，共 3M training samples。
  Benchmarks 图像：VQA-v2, GQA, VisWiz, ScienceQA-IMG, TextVQA, POPE, MME, MMBench, SEED-Bench, LLaVA-Bench-in-the-Wild, MM-Vet (共 11 个)。
  Benchmarks 视频：MSVD-QA, MSRVTT-QA, ActivityNet-QA, Video-based Generative Performance (Correctness/Detail/Contextual/Temporal/Consistency), MVBench (20 子任务), MLVU (7 子任务), EgoSchema (共 7 个)。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址：https://github.com/ictnlp/LLaVA-Mini。模型权重：https://huggingface.co/ICTNLP/llava-mini-llama-3.1-8b

  算法 pipeline 伪代码：

  ```
  # ===== 输入 =====
  # X_v: 输入图像 (H×W×3), X_q: 语言指令文本 (l_q 个 token)

  # ===== Vision Encoding =====
  # CLIP ViT-L/336px: 图像 → 576 个 vision token (N^2=24×24)
  H_v = CLIP_ViT(X_v)              # [576, d_h]
  H_v = Projection(H_v)            # [576, d_h], 映射到 LLM embedding 空间
  H_q = LLM_Embedding(X_q)         # [l_q, d_h], 文本 token 嵌入

  # ===== Query-based Compression =====
  # C 个压缩 query (默认 C=1), 使用 2D sinusoidal position encoding
  Q_v = learnable_queries           # [C^2, d_h], C=1→[1, d_h]
  pos = 2D_Sinusoidal_PE()
  A = Softmax((Q_v + pos(Q_v)) @ (H_v + pos(H_v)).T)  # [C^2, 576]
  H_v_compressed = A @ H_v         # [C^2, d_h], 压缩后 vision token

  # ===== Modality Pre-fusion =====
  # N_fusion=4 层与 LLM 同构的 Transformer decoder blocks
  # 将全部 vision token 与 text token 拼接后输入
  concat = Concat(H_v, H_q)        # [576 + l_q, d_h]
  output = PreFusion(causal_mask=concat)  # [576 + l_q, d_h]
  H_q_fused = output[-l_q:]        # [l_q, d_h], 仅取文本位置的输出

  # ===== LLM Backbone =====
  # 只输入 1 个压缩 vision token + 融合后的 text token
  llm_input = Concat(H_v_compressed, H_q_fused)  # [1 + l_q, d_h]
  response = LLM(llm_input)        # 自回归生成回复

  # ===== 高分辨率图像 (HD) =====
  # 图像分割为 4 个子图 (2×2), 每子图独立编码
  H_v_sub = [ViT(sub) for sub in split(X_v, 2×2)]  # 4 × [576, d_h]
  H_v_full = ViT(X_v)              # 原图 [576, d_h]
  # 5 组 vision tokens 全部送入 pre-fusion
  # compression 压缩 4×576 子图 token 为 C^2 (C=8, 即 64) 个
  # LLM 输入: 64 vision tokens + l_q text tokens

  # ===== 视频处理 =====
  # M 帧视频, 每帧独立处理, C=1
  # Per frame: 1 compressed vision token + l_q fused text tokens
  # 视频总输入: M×1 vision tokens + l_q fused text tokens (pooled from M frames)
  # 对比 LLaVA-v1.5: M×576 vision tokens + l_q text tokens
  ```

  张量计算流程（标准分辨率 336px, C=1）：
  - Vision Encoder: 336×336×3 → ViT → [576, 1024] → Projection (Linear) → [576, 4096] (Vicuna-7B hidden dim)
  - Compression: Q_v [1,4096] + PE, H_v [576,4096] + PE → Cross-Attention → A [1,576] → Ĥ_v [1,4096]
  - Pre-fusion: Concat([576,4096], [l_q,4096]) → 4×Transformer → output[-l_q:, :] → Ĥ_q [l_q, 4096]
  - LLM: Concat([1,4096], [l_q,4096]) → Vicuna-7B 32 layers → autoregressive response
  - FLOPs: Vision 0.35T + Projection 0.02T + Compression 0.001T + Pre-fusion 0.13T + LLM 1.46T = **1.96T** (vs LLaVA-v1.5 8.55T, 减少 77%)

## LLM2CLIP__Powerful_Language_Model_Unlock_Richer_Visual_Representation

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：LLM2CLIP —— 两阶段高效微调框架，将 LLM（Llama 3.1 8B）注入预训练 CLIP，以极低训练成本增强跨模态表示能力。
  **Stage 1 (LLM Caption Contrastive Fine-tuning)**：对 LLM 进行"embedding化"改造，使其输出特征对图像caption具有足够的可分离性。具体设计：(1) 模型架构 — 移除 causal attention mask 启用双向注意力，使用 average pooling（而非[EOS] token）聚合所有输出token获得句子嵌入，通过 LoRA (r=16, α=32) 进行参数高效微调；(2) 训练方法 — 使用监督 SimCSE 对比损失，正样本对为同一图像的两个不同 caption（由系统prompt "Given a caption, retrieve a similar relevant caption" 构建），不使用 MNTP；(3) 训练数据 — 30M DreamLIP caption 数据 + 1.5M Echo Embeddings 纯文本对混合对比训练。使用 AdamW (lr=2e-4, 300-step linear warmup)，sequence length 512，有效 batch size 2048，1 epoch，32 NVIDIA A100 GPU。
  **Stage 2 (LLM2CLIP Post Fine-tuning)**：将 CC fine-tuned LLM 作为文本编码器替换原始 CLIP text encoder，冻结 LLM 梯度，在其输出后附加一个 4 层 Linear Adaptor（inverted bottleneck MLP，来自 FuseMix，~67.1M 参数）作为可学习模块，与 CLIP Vision Encoder 进行跨模态对比学习。ViT 梯度全开。使用 AdamW (lr=1e-5, cosine decay rate 0.05)，总 batch size 4096（offline-loading 模式下可达 16384），4 epochs，每步从 DreamLIP caption 数据中随机采样一个 caption。嵌入维度设为 1280。默认使用 15M DreamLIP 标注子集（CC3M + CC12M）；60M 设置额外使用 YFCC15M 和 30M LAION 子集。Offline-loading 策略：预计算文本嵌入，将 LLM 推理开销从多 epoch 降低到单次 pass，训练时无需加载 LLM 到 GPU 显存。
  训练数据配比：真实短 caption 与 MLLM 生成 dense caption 按 50% 比例混合。

  实验比较：
  (a) 系统对比 —— 在 ViT-B/16、ViT-L/14、ViT-L/14-336、EVA02-L/14、SigLIP-SO/14、SigLIP2-SO/14 上应用 LLM2CLIP，对比 CLIP/EVA02/SigLIP/SigLIP2/MetaCLIP/Long-CLIP/ALIGN/BLIP/jina-clip-v2/InternVL/VLM2Vec；(b) 多语言检索 —— Flickr-CN、COCO-CN、XM3600 (36语言)，对比 CN-CLIP/EVA-L-224/SigLIP2；(c) Zero-shot 分类与 Linear Probe —— ImageNet；(d) Zero-shot/Supervised 分割与检测 —— COCO-S/ADE/VOC/Cityscapes (zero-shot seg mIoU) + OV-COCO (open-vocab detection) + COCO val2017 (supervised)；(e) MLLM 性能 —— LLaVA-1.5-7B 替换视觉编码器后评估 VQA-v2/GQA/VizWiz/SQA-I/TextVQA/POPE/MME/MMBench/MMBench-CN/LLaVA-Bench/SEED；(f) Stage-1 消融 —— 训练方法(LoRA/SimCSE/MNTP/attention type/pooling)、不同 LLM backbone(Qwen2.5/LLaMA-variants/DeepSeek-R1)、adaptor 设计；(g) Stage-2 消融 —— adaptor 结构(Linear vs Transformer, 层数)、训练方法(单 encoder/双 encoder/拼接)、dense caption 配比(0%-100%)；(h) 效率分析 —— LoRA vs Frozen + Offline-loading。

- 硬件平台是什么，配置是什么。
  Stage 1: 32 NVIDIA A100 GPU。
  Stage 2: 2 nodes，每节点 8 NVIDIA A100 40GB GPU。Offline-loading 模式下 batch size 可达 16384，训练时间从 17h (LLM LoRA) 降至 1.3h。

- 模型是什么。数据集和bench分别是什么。
  模型：Vision Encoders — ViT-B/16 (86M)、ViT-L/14 (307M)、ViT-L/14-336、EVA02-L/14-224、EVA02-L/14-336、SigLIP-SO/14 (428M)、SigLIP2-SO/14。Text Encoder — Llama 3.1 8B (经 Stage-1 CC fine-tuning)。对比 text encoders — bge-en-icl、LLM2Vec-Llama-3-8B、NV-Embed-v2、VLM2Vec、bge-m3-XLM-R、jina-v3-XLM-R、e5 (XLM-R)、Qwen2.5-0.5B、LLaMA-3.2-1B、LLaMA-3-8B、DeepSeek-R1-Distill-Llama-8B。
  训练数据：Stage 1 — DreamLIP 30M captions + Echo Embeddings 1.5M 纯文本对。Stage 2 — DreamLIP 标注 CC3M/CC12M/YFCC15M/LAION 子集（3M/15M/60M 设置）。
  Benchmarks：(1) 短文本检索 — Flickr30K 1K test、MS COCO 5K test；(2) 长文本检索 — ShareGPT4V 1K subset、Urban1K、DOCCI；(3) 多语言检索 — Flickr-CN、COCO-CN、Crossmodal-3600 (36语言)；(4) 分类 — ImageNet zero-shot (单模板 + 80模板平均) 和 linear probe；(5) 分割 — ADE20K、COCO-Stuff164k、VOC20、Cityscapes zero-shot + COCO val2017 supervised；(6) 检测 — OV-COCO open-vocabulary；(7) MLLM — LLaVA-1.5-7B 在 VQA-v2/GQA/VizWiz/SQA-I/TextVQA/POPE/MME/MMBench/MMBench-CN/LLaVA-Bench/SEED。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址：https://aka.ms/llm2clip。承诺开源所有训练/测试代码、数据集、LLM2CLIP 增强版 SigLip2/EVA02/OpenAI CLIP 模型权重。

  算法 pipeline 伪代码：

  ```
  # ===== Stage 1: LLM Caption Contrastive Fine-tuning =====
  # LLM: Llama 3.1 8B, 移除 causal mask → bidirectional attention
  # LoRA: r=16, α=32, 应用于 Q/K/V/O 投影矩阵
  # 输入: 来自 DreamLIP 的 caption pairs (c_i, c_j) 属于同一图像

  def stage1_cc_finetune(llm, captions_pairs):
      for (c_i, c_j) in captions_pairs:
          # 前向: 移除 causal mask, 使用 bidirectional attention
          h_i = llm(c_i, attention_mask="bidirectional")  # [L_i, d_llm]
          h_j = llm(c_j, attention_mask="bidirectional")  # [L_j, d_llm]

          # Average pooling 获得句子嵌入 (而非 [EOS] token)
          e_i = mean(h_i, dim=0)  # [d_llm]
          e_j = mean(h_j, dim=0)  # [d_llm]

          # 监督 SimCSE 对比损失 (in-batch negatives)
          # e_i, e_j 为正样本对, 同 batch 其他样本为负样本
          sim_matrix = e_i @ e_j.T / temperature  # [B, B]
          labels = arange(B)  # 对角线为正样本
          loss = CrossEntropyLoss(sim_matrix, labels)

          # 仅更新 LoRA 参数, AdamW lr=2e-4
          loss.backward()
          optimizer.step()

      return llm  # 输出 CC-fine-tuned LLM

  # ===== Stage 2: LLM2CLIP Post Fine-tuning =====
  # 预计算: 用 CC-fine-tuned LLM 对所有 caption 做 offline inference
  # 获得文本嵌入存盘 → 训练时直接加载嵌入
  def offline_precompute(llm_cc, all_captions):
      embeddings = []
      for caption in all_captions:
          h = llm_cc(caption, attention_mask="bidirectional")
          embeddings.append(mean(h, dim=0))
      return embeddings  # 存入磁盘

  # Adaptor: 4层 inverted bottleneck MLP (FuseMix 设计)
  # d_llm=4096 -> Linear -> d_hidden -> GeLU -> Linear -> d_llm
  #          -> Linear -> d_hidden -> GeLU -> Linear -> d_llm
  #          -> Linear -> d_hidden -> GeLU -> Linear -> d_llm
  #          -> Linear -> d_out=1280
  # 训练配置:
  #   ViT: 梯度全开 (学习 LLM 知识)
  #   LLM: 梯度冻结 (不加载到 GPU)
  #   Adaptor: 梯度全开 (可学习)
  def stage2_llm2clip(vision_encoder, adaptor, precomputed_text_emb, images):
      for (img, txt_emb_precomputed) in dataloader:
          # 视觉编码
          v_feat = vision_encoder(img)  # [B, 1280]

          # LLM 文本特征 (预计算嵌入 → adaptor)
          # 原始 CLIP text encoder 被完全丢弃
          t_feat = adaptor(txt_emb_precomputed)  # [B, 1280]

          # 跨模态对比损失 (CLIP loss)
          # L2 normalize
          v_feat = v_feat / ||v_feat||_2
          t_feat = t_feat / ||t_feat||_2

          logits = v_feat @ t_feat.T * exp(t)  # temperature t 可学习
          labels = arange(B)
          loss_i2t = CrossEntropyLoss(logits, labels)
          loss_t2i = CrossEntropyLoss(logits.T, labels)
          loss = (loss_i2t + loss_t2i) / 2

          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

      return vision_encoder, adaptor  # LLM2CLIP 模型

  # ===== 推理 =====
  def llm2clip_inference(image, text_query):
      # 视觉编码
      v = vision_encoder(image)  # [1280]

      # 文本编码: LLM → average pooling → adaptor
      h = llm_cc(text_query, bidirectional=True)
      t = adaptor(mean(h, dim=0))  # [1280]

      # 相似度分数
      score = cosine_similarity(v, t)
      return score
  ```

  Adaptor 结构（FuseMix inverted bottleneck MLP）：
  ```
  # 输入: LLM hidden state ∈ R^4096 (Llama 3.1 8B)
  # 每层: Linear(d_in, d_hidden) → GeLU → Linear(d_hidden, d_in) + residual
  # 最终投影: Linear(4096, 1280)

  def FuseMixAdaptor(x):
      for layer in range(4):
          residual = x
          x = Linear_in(x)     # 4096 → d_hidden
          x = GeLU(x)
          x = Linear_out(x)    # d_hidden → 4096
          x = x + residual     # residual connection
      x = FinalProjection(x)   # 4096 → 1280
      return x
  ```

  关键设计选择与实验证据：
  - CC fine-tuning 是必需的：原始 Llama 3.1-8B 在 COCO caption-to-caption retrieval 上 Top-1 仅 5.2%，CC 后提升至 29.5%（Table A1）
  - 替换而非复用 CLIP text encoder：同时保留两个 text encoder 或对齐两个 encoder 带来 marginal 甚至 negative gain（Table 9/Table A8）
  - Supervised SimCSE >> Unsupervised SimCSE >> MNTP alone（Table 6/Table A5）
  - LoRA 优于 Frozen + Adaptor 用于 Stage-1，但 Stage-2 Frozen + Adaptor 提供最佳效率-性能权衡
  - Offline-loading 将训练时间从 17h 降至 1.3h，同时 batch size 从 704 增至 16384（Table A4）

## Growing_a_Twig_to_Accelerate_Large_Vision-Language_Models

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：TwigVLM/TwigVLM++——在冻结的base VLM早期层上"生长"一个轻量twig block，通过两种策略实现推理加速：(1) **Twig-guided Token Pruning (TTP)**：在prefilling阶段，利用twig最后一层（靠近prediction head）的attention map指导视觉token剪枝，替代传统方法中早期层不敏感的attention信号。具体流程：输入tokens X经过base VLM前K层得X^(K)，再经twig block得最后twig层的attention map A^(K+T)，用该attention map选择top-R最重要的视觉tokens保留，其余丢弃。引入**FinalWipe**策略在Kf层后移除所有视觉tokens，使平均保留token数 R̄=[M×K+R×(Kf-K)]/L，在固定R̄下允许更大的R。(2) **Self-Speculative Decoding (SSD)**：在decoding阶段，以浅层子网络Ms（前K层+twig）为draft model自回归生成候选tokens，以深层子网络Mb（完整base VLM）为target model并行验证。draft每步预测5个tokens后触发验证（含early-exit:概率<θ=0.6时停止），接受匹配的tokens并追加一个bonus token。
  TwigVLM++扩展：(i) **Multi-head twig架构**：解耦D-Head（标准next-token prediction）和P-Head（专用于token重要性评分），P-Head通过可学习gating投影Gq/Gk调制自注意力层的Q/K计算重要性分数s=1/H·Σσ((Gq(xq)⊙q̃)(Gk(Xk)⊙K̃)^T/√dh)；(ii) **两阶段训练**：Stage-1用L_NTP+α·L_PredKL+γ·L_AttnKL训练twig，Stage-2用GRPO式RL仅训练P-Head，reward为pruned输入下对参考答案的mean log-probability，配合动态pruning ratio schedule（候选集R={64,...,192}，annealing分布逐渐偏向小R）；(iii) **Tree-based SSD**：draft model构建token tree（expansion width E=10, selection width K=10, depth D=4），target model用tree attention并行验证多条候选路径。

  实验比较：(a) 主实验 —— LLaVA-1.5-7B在6个benchmark(GQA/MMB/MME/VQA^T/SQA^I/VQA^V2)三个pruning ratio(66.7%/77.8%/88.9%)下 vs FastV/SparseVLM/PDrop/MustDrop/VisionZip/FasterVLM；(b) LLaVA-NeXT-7B同样benchmark对比；(c) Qwen2.5-VL-7B在image benchmark(GQA/MME/MMB/SQA^I/VQA^T/VQA^V2/MMStar)和video benchmark(OCRBench/Blink/VideoMME/EgoSchema/MVB)下对比；(d) 生成速度对比 —— TextVQA(短response, S̄≈10)和MM-Vet(长response, S̄≈100)上的RelSpd；(e) 消融 —— 视觉token选择注意力源、加速策略组合(TTP/SSD)、twig block初始化、twig层数T、pruning位置K、FinalWipe位置Kf；(f) TwigVLM++消融 —— Stage-1 head/loss组合、Stage-2 static vs dynamic pruning ratio、RL训练数据量；(g) LLaVA-1.5-13B扩展实验；(h) Token acceptance rate分析、data efficiency训练数据比例实验。

- 硬件平台是什么，配置是什么。
  8×NVIDIA A100 GPU服务器。训练TwigVLM的LLaVA-1.5-7B twig block约10 GPU hours（占base VLM训练时间的~10%），TwigVLM++约20%时间。推理使用相同硬件配置。

- 模型是什么。数据集和bench分别是什么。
  模型：Base VLM包括LLaVA-1.5-7B、LLaVA-NeXT-7B、Qwen2.5-VL-7B、LLaVA-1.5-13B。Twig配置：T=3 twig layers，pruning位置K=2，FinalWipe位置Kf=24。
  训练数据：LLaVA-665K（用于LLaVA-1.5和LLaVA-NeXT的twig训练），MAmmoTH-VL-10M中5M单图样本（用于Qwen2.5-VL的twig训练）。Stage-2 RL仅用50K SFT样本。
  Benchmarks：GQA、MMBench(MMB)、MME、TextVQA(VQA^T)、ScienceQA-IMG(SQA^I)、VQA-v2(VQA^V2)、MM-Vet、MMStar、OCRBench、Blink、VideoMME、EgoSchema、MVBench。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址：https://github.com/MILVLG/twigvlm (Apache 2.0 License)

  算法 pipeline 伪代码（TwigVLM推理流程，论文Algorithm 1）：

  ```
  # bVLM: 完整 base VLM M_b
  # twig: twig block (T transformer layers)
  # K: 共享的低层数
  # K_f: FinalWipe层位置
  # R: 剪枝后保留的visual token数
  # delta: maximum draft token length (default 5)
  # theta: 停止draft的置信度阈值 (default 0.6)
  def sVLM_forward(tokens):
      X_k = bVLM.forward_low_layers(tokens, k=K)
      prob, Attn_last = twig.forward(X_k)
      a_i = argmax(prob)
      return X_k, prob, Attn_last, a_i

  def TwigVLM_inference(img, ques):
      draft_toks = []
      final_resp = []
      # Prefilling阶段：sVLM前向
      X_k, _, Attn, a_i = sVLM_forward((img, ques))
      draft_toks.append(a_i)
      # TTP: 用twig最后层的attention剪枝visual tokens
      # X_k_b: bVLM的共享token latent
      X_k_b = pruning(X_k, Attn, r=R)  # 按Eq.(5): P(X^(K)_Mb, A^(K+T)_Ms, R)
      # SSD循环: draft→verify迭代
      while EOS_TOKEN not in final_resp:
          X_k, prob, _, a_i = sVLM_forward(a_i)
          draft_toks.append(a_i)
          X_k_b = concat(X_k_b, X_k, axis=1)
          # 停止draft并触发验证
          if len(draft_toks) >= delta or prob < theta:
              # FinalWipe: 移除K_f层后的所有visual tokens
              tgt_probs = bVLM.forward_high_layers(
                  X_k_b, k=K, final_wipe=K_f)
              # 验证draft tokens
              right_toks = [a for a, p in zip(draft_toks, tgt_probs[:-1])
                            if argmax(p) == a]
              right_toks.append(argmax(tgt_probs[-1]))
              final_resp.extend(right_toks)
              draft_toks = []
              X_k_b = None
              a_i = final_resp[-1]
      return final_resp
  ```

  TwigVLM++ P-Head评分计算（Eq.7）：
  ```
  # 输入: X^(K+T) — twig最后一层SA层的输入
  # Q, K = X^(K+T)·W_q, X^(K+T)·W_k  (标准自注意力投影)
  # q̃ ∈ R^{H×d_h}: query向量(最后textual token位置)
  # K̃ ∈ R^{H×M×d_h}: key矩阵(visual token位置)
  # G_q, G_k: 可学习gating投影(Linear+nonlinear activation)
  scores_m = []
  for h in range(H):
      gated_q = G_q(x_q)^{(h)} ⊙ q̃^{(h)}     # element-wise gating
      gated_k = G_k(X_k)^{(h)} ⊙ K̃^{(h)}
      scores_h = softmax(gated_q @ gated_k.T / sqrt(d_h))
      scores_m.append(scores_h)
  s = mean(scores_m, dim=0)  # 最终token重要性分数 ∈ R^M
  # 用s替代原attention map进行pruning: Eq.(8)
  ```

  TwigVLM++ Stage-2 RL (GRPO-style, Eq.12-15)：
  ```
  # π_θ: P-Head产生的token重要性分布
  # 对每个样本采样 G=32 个pruning action a_i
  for a_i in range(G):  # 每个action: 无放回采样R个visual token位置
      π_i = π_θ
      for j in range(R):
          a_j ~ Categorical(π_i)          # 按当前分布采样
          π_i[a_j] = 0                     # 移除已选位置
          π_i = π_i / sum(π_i)            # 重归一化
      # reward = pruned输入下生成参考答案的mean log-prob
      r_i = (1/S) * Σ log p_Mb(y*_j | X̂, y*_{<j})  # Eq.(13)
  # Group-level advantage normalization
  Â_i = (r_i - mean({r})) / std({r})                # Eq.(14)
  # 纯on-policy更新 (importance ratio = 1)
  L_stage2 = (1/G) * Σ Â_i * log π_θ(a_i)           # Eq.(15)
  ```

  算法 pipeline 全栈执行流程（以 LLaVA-1.5-7B base + TwigVLM 为例）：
  - 算法层：Image → CLIP Vision Encoder → 576 visual tokens + text prompt tokens → 拼接输入 → base VLM前2层 → twig block (3层, 初始化为VLM第3-5层权重) 得attention map → TTP按R=41剪枝保留top visual tokens → 剩余VLM层(3-32层)处理pruned序列(含FinalWipe在24层移除所有visual tokens), 同时twig作为draft model自回归生成候选tokens → base VLM(target model)并行验证并接受匹配tokens → 生成答案。
  - 系统框架层：基于HuggingFace Transformers，复用base VLM权重初始化twig，仅训练twig block（冻结前K层和剩余层）。训练用LLaVA-665K SFT数据+AR loss。推理时draft和target model共享前K层KV-cache。
  - 编译框架层：论文未明确说明。
  - kernel调度层：标准FlashAttention（v2.3.2），利用GPU并行计算能力通过SSD将decode阶段从逐token串行转为批token并行验证。Tree-based SSD使用tree attention（topology-aware causal mask替代标准causal mask）。
  - 硬件架构层：8×NVIDIA A100 GPU训练，推理可用单卡A100。

## Free-MoRef__Instantly_Multiplexing_Context_Perception_Capabilities_of_Video-MLLMs_within_Single_Inference

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：Free-MoRef 是一个 training-free 的推理时方法，通过修改 Video-MLLM 中 LLM 的 self-attention 层来在单次推理中多路复用上下文感知能力。核心设计包含三个组件：(1) **Multi-Reference Partition** —— 将长 vision token 序列按时间关系划分为 M 个 temporal units × N 个 fragments，最终聚合为 N 个 reference chunks，每个 chunk 作为原始长视频的抽象摘要，且分配相同的 system prompt 和 question。(2) **MoRef Attention（Mixture-of-Reference Attention）** —— 替换 LLM 的标准 self-attention。对 N 个 parallel chunks 分别执行 FlashAttention 得到 O = [O^sys, O^vis, O^ques]，其中 O^sys 在各 chunk 间相同（因果注意力的单向性），O^vis 保持各 chunk 的差异性，O^ques 通过 gating 函数加权聚合：O^fusion = (Σ ω_i · O_i^ques).repeat(N)，其中 ω_i = max(A[i]) / Σ max(A[i])，A = softmax(Q^ques × (K^vis)^T) 为 query-vision 跨模态注意力图。最终输出为 O^MoRef = [O^sys, O^vis, O^fusion]。(3) **Reference Fusion** —— 在 decoder 的中间层 L，基于注意力图 A 计算重要性估计矩阵 E ∈ R^{N × l_vis}（沿 l_ques 维度平均），在每个 chunk 内剪枝 1-1/N 的不重要 vision tokens，然后将各 chunk 保留的 token 按时间关系聚合为 global reference，后续 decoder 层只用 global reference 推理。该方法以约 1/N 的计算复杂度实现对全部 vision tokens 的 completely-aware perception，支持 FlashAttention，且可与 streaming inference 或 token compression 结合。

  实验比较：(a) Multiplexed Context Understanding —— LLaVA-Video-7B 在 64/128/256/512 frames 下对比 Free-MoRef vs 直接扩展帧数（full attention），指标为 FLOPs 和 VideoMME/MLVU/LongVideoBench 准确率；(b) SOTA 对比 —— 在 VideoMME、MLVU、LongVideoBench 上对比 InternVL2、InternVL2.5、Qwen2-VL、LLaVA-OneVision、LLaVA-Video、Kangaroo、LongVILA、LongVA、Video-XL、RETAKE（均为 7B~8B 规模）；(c) 消融 —— 各组件贡献（Multi-Reference Partition、MoRef Attention、Reference Fusion）、Chunk 数 N（1/2/4/8）、Temporal Units M（1/4/32/64）、Reference Fusion 层 L（1/3/6）对性能的影响；(d) 任务类型分析 —— VideoMME 的 12 个子任务（TP/SP/AP/ARec/ORec/OCR/CP/TR/SR/AR/OR/IS）在不同 context 长度下的表现。

- 硬件平台是什么，配置是什么。
  单张 NVIDIA A100 GPU（论文实验部分明确说明 "all experiments are executed on a single A100 GPU"）。对超过 context length 限制的场景，使用 accelerate toolkit 辅助管理显存。256 frames baseline 在单卡 A100 上因超过 Qwen2 的 32768 token 限制直接 OOM，Free-MoRef 可在不使用 accelerate 的情况下直接推理 512 frames。

- 模型是什么。数据集和bench分别是什么。
  模型：LLaVA-Video-7B（基于 Qwen2-7B 作为 LLM backbone，vision encoder 为 SigLIP，FPS=1，默认最大 64 frames，每帧 182 tokens 通过 spatial pooling 得到）。
  Benchmarks：(1) VideoMME —— 900 个视频、256 小时总时长、2700 个多选题 QA pair、分 short/medium/long 三个子集、覆盖 30 个子领域；(2) MLVU —— 多任务长视频理解 benchmark、7 种 QA 任务类型、视频长度 3分钟至 2小时+、平均 12 分钟；(3) LongVideoBench —— 强调细粒度推理问题、17 种问题类别、10 种视频类型、分 8-15s/15-60s/3-10min/15-60min 四组。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址：https://github.com/wkfdb/Free-MoRef
  
  算法 pipeline 伪代码：

  ```
  # ==========================================
  # Free-MoRef 推理流程 (Training-Free)
  # ==========================================
  # 输入: N_frames 个视频帧, question Q_text
  # 超参数: M (temporal units), N (parallel chunks), L (fusion layer)

  # 步骤 1: Vision Encoding
  frames = load_video(video_path, fps=1, max_frames=N_frames)
  # 每帧: Vision Encoder → 182 tokens → spatial pooling → vision_tokens
  vision_tokens ∈ R^{N_frames × 182 × D}

  # 步骤 2: Multi-Reference Partition
  # 将 vision_tokens 按时间顺序划分为 M 个 units
  # 每个 unit 内按时间分解为 N 个 fragments
  # 聚合各 unit 的相同 index 的 fragment → N 个 reference chunks
  for i in range(N):
      chunk_i = []
      for unit_j in range(M):
          # 取 unit_j 的第 i 个 fragment
          chunk_i.append(units[unit_j].get_fragment(i))
      reference_chunks[i] = concat(chunk_i)  # ∈ R^{(total_tokens/N) × D}
  
  # 为每个 chunk 拼接相同的 system prompt + question
  for i in range(N):
      input_seq[i] = concat([system_prompt, vision_chunk[i], question])

  # 步骤 3: MoRef Attention (替换 LLM 的 self-attention, layers 0..L-1)
  for layer in range(L):  # shallow layers
      for each chunk i in range(N):
          Q_i, K_i, V_i = W_Q(input_seq[i]), W_K(input_seq[i]), W_V(input_seq[i])
          O_i = FlashAttention(Q_i, K_i, V_i)  # 标准 causal attention
          # O_i = [O_i^sys, O_i^vis, O_i^ques]
      
      # 计算 gating weights (跨模态 query-vision 注意力)
      for each chunk i:
          A_i = softmax(Q_i^ques @ K_i^vis.T)  # ∈ R^{l_ques × l_vis}
          w_i = max(A_i) / sum(max(A_j) for j in range(N))
      
      # 聚合 question tokens
      O_fusion = sum(w_i * O_i^ques for i in range(N))
      O_fusion = O_fusion.repeat(N)
      
      # 组装 MoRef 输出
      for each chunk i:
          O_i^MoRef = [O_i^sys, O_i^vis, O_fusion]
      
      # 残差连接 + FFN (标准 transformer)
      for each chunk i:
          input_seq[i] = input_seq[i] + O_i^MoRef
          input_seq[i] = input_seq[i] + FFN(LayerNorm(input_seq[i]))

  # 步骤 4: Reference Fusion (在 layer L)
  # 计算重要性估计矩阵
  for each chunk i:
      E_i = mean(A_i, dim=l_ques)  # ∈ R^{l_vis}
  # 剪枝: 每个 chunk 保留 top 1/N 的 vision tokens
  for each chunk i:
      topk_indices = topk(E_i, k=l_vis // N)
      kept_tokens[i] = vision_chunk[i][topk_indices]
  
  # 按时间关系聚合为 global reference
  global_vision = temporal_merge([kept_tokens[i] for i in range(N)])
  global_seq = concat([system_prompt, global_vision, question])
  
  # 步骤 5: Deep layers default inference (layers L..end)
  for layer in range(L, num_layers):
      Q, K, V = W_Q(global_seq), W_K(global_seq), W_V(global_seq)
      O = FlashAttention(Q, K, V)  # 标准 causal attention
      global_seq = global_seq + O
      global_seq = global_seq + FFN(LayerNorm(global_seq))
  
  # 步骤 6: 从 global_seq 的 question position 解码输出
  answer = decode(global_seq[ques_positions])
  ```

  关键张量形状：
  - vision_tokens: [N_frames, 182, D] → 展平后 [N_frames × 182, D]
  - 经 Multi-Reference Partition 后每个 chunk: [l_vis/N, D] + [l_sys, D] + [l_ques, D]
  - A_i ∈ [l_ques, l_vis/N], E_i ∈ [l_vis/N]
  - 复杂度对比：full attention O((N·l_vis + l_sys + l_ques)²) → MoRef O(N·(l_vis/N + l_sys + l_ques)²) ≈ O(1/N · full)，即约为 1/N
  - 128 frames@Free-MoRef: FLOPs = 110.4% of 64-frame baseline (vs 400% for full attention)
  - 256 frames@Free-MoRef: FLOPs = 163.2% of baseline (vs 1600% for full attention)
  - 512 frames@Free-MoRef: FLOPs = 400% of baseline (vs 6400% for full attention)

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

## EVA__Efficient_Reinforcement_Learning_for_End-to-End_Video_Agent

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：EVA 是一个基于强化学习的端到端视频 Agent 框架，核心是 planning-before-perception 范式，通过迭代 summary–plan–action–reflection 循环实现自主视频理解。技术栈：(1) MDP 建模 —— s_t = {q, h_t, F_t}（query、历史、视觉证据），策略 π_θ(a_t | s_t) 参数化为 MLLM；(2) 灵活帧选择工具 —— 支持 start_time、end_time、nframes、resize 四个参数，同时控制时间和空间粒度；(3) 三阶段训练 pipeline —— SFT Cold-Start（10k 样本，训练 tool-call 格式、交错图文推理、帧级理解和基本帧选择策略，lr=2e-6, bs=8, 2 epochs）→ KTO 纠错（11k 标注策略，63% correct + 37% rejected，纠正猜测/欠采样/过采样等失败模式，lr=2e-6, β=0.1）→ Data-Enhanced GRPO 在线强化学习（9.6k 开放 QA + 1.1k MCQ，batch size=64, rollouts=8 per sample, lr=1e-6, 1 epoch，收集失败案例让 teacher MLLM 为 HD-VILA 新视频生成新 QA pairs）；(4) Reward —— Accuracy（MCQ: CSV self-verification r_csv；open-ended: ROUGE r_rouge = (R1+R2+RL)/3）+ Format Reward（tool call 但答案错误给 0.05 补偿抑制猜测）。训练用 32 H100 GPU。

  实验比较：(a) LSDBench —— 对比 Gemini-2.0-Flash (2700 frames/696.6K tokens)、LongVA、Qwen2-VL、Qwen2.5-VL、LongVila 等，EVA 用 76.9 frames/10.3K tokens 达到 51.0%（vs baseline Qwen2.5-VL* 49.2% with 32 frames/21K tokens）；(b) 长视频理解 —— LongVideoBench、MLVU、VideoMME、LVBench 上对比 GPT-4o、Gemini-1.5-Pro、Video-R1、VideoChat-R1、Qwen2.5-VL（static）和 VideoAgent、FrameThinker、VideoMTR（adaptive agent），EVA-GRPO 各 benchmark 分别达 55.0%、68.3%、60.2%、43.3%；(c) Video-Holmes 零样本视频推理 —— 7 子任务 vs GPT-4o、Gemini-2.0-Flash、InternVL2.5/3、Video-R1 等，EVA-GRPO 37.2%；(d) 消融 —— SFT→KTO→GRPO 阶段逐步增益、GRPO 数据组成（MC vs OE vs mixed）、ELV-Halluc（SAH Ratio 8.8%→5.0%）；(e) 效率 —— 总 token 可比或低于 uniform sampling，推理时间由自适应 compact visual tokens 决定。

- 硬件平台是什么，配置是什么。
  训练：32 × NVIDIA H100 GPU（GRPO 阶段）。推理评估：vLLM 框架（temperature=0），原始视频分辨率 720p。Base model：Qwen2.5-VL-7B-Instruct。Teacher MLLM：Qwen2.5-VL-72B（数据构造阶段）。

- 模型是什么。数据集和bench分别是什么。
  模型：Qwen2.5-VL-7B-Instruct（主模型，支持多分辨率视觉输入）。Teacher MLLM: Qwen2.5-VL-72B。
  训练数据集（自建）：EVA-SFT（10k 样本，源 QA pairs 来自 llava-video 和 cgbench）、EVA-KTO（11k 标注帧选择策略）、EVA-RL（9.6k open-ended + 1.1k MCQ）。外部数据：HD-VILA（GRPO 数据增强阶段的新视频源）。
  Benchmarks（7 个）：(1) LSDBench —— 长视频采样困境 benchmark；(2) LongVideoBench —— 3763 videos/6678 QA pairs，最长 1h；(3) MLVU —— ~2600 QA pairs，平均 636s，9 任务；(4) VideoMME —— 900 videos/2700 QA pairs，30 子领域；(5) LVBench —— 1549 QA pairs，平均 4101s；(6) Video-Holmes —— 视频推理，7 子任务 SR/IMC/TCI/TA/MHR/PAR/CTI；(7) ELV-Halluc —— 语义聚合幻觉 benchmark。
  评价指标：Accuracy（全部 benchmark），Visual Token 数量（效率），SAH Ratio（ELV-Halluc）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址：https://github.com/wangruohui/EfficientVideoAgent
  
  算法 pipeline 伪代码：
  ```
  # === EVA 推理流程（planning-before-perception） ===
  # s_0 = {q, h_0=[], F_0=[]}  # 初始仅 query

  for t in 1..max_rounds:
      # 1. Summary: 对当前帧生成详细描述
      summary_t = MLLM.summarize(F_{t-1})
      
      # 2. Planning: 基于 query + 历史 + summary 提出潜在 actions
      plan_t = MLLM.plan(q, h_{t-1}, summary_t)
      # 估算每个 action 的 token cost 和 expected outcome
      
      # 3. Action: 生成 frame_select tool call
      action_t = {
        "tool": "frame_select",
        "arguments": {"start_time": t0, "end_time": t1,
                      "nframes": N, "resize": r}
      }
      # resize=0.1 → 低分辨率全局浏览; resize=0.4~0.5 → 高分辨率聚焦
      F_t = F_{t-1} ∪ extract_frames(V, action_t.arguments)
      
      # 4. Reflection: 评估视觉信息是否充足
      if MLLM.reflect(q, F_t).is_sufficient: break

  answer = MLLM.answer(q, h_T, F_T)
  ```

  训练 pipeline 伪代码：
  ```
  # Stage 1: SFT Cold-Start (lr=2e-6, bs=8, 2 epochs)
  # 数据格式: Summary → Planning → Action → Reflection
  for batch in EVA-SFT:
      loss = CrossEntropy(MLLM(batch.input), batch.target)
      MLLM.backward(loss)
  
  # Stage 2: KTO Correction (lr=2e-6, β=0.1)
  # 单样本偏好: {trajectory, label ∈ {chosen, rejected}}
  for batch in EVA-KTO:  # 63% chosen + 37% rejected
      loss = KTO_loss(MLLM, batch, β=0.1, ref=MLLM_SFT)
      MLLM.backward(loss)
  
  # Stage 3: Data-Enhanced GRPO (lr=1e-6, bs=64, 8 rollouts/sample)
  for batch in EVA-RL:  # 90% OE + 10% MCQ
      # 采样 G=8 个候选响应
      responses = [MLLM.sample(q, v) for _ in range(8)]
      # 计算 reward
      for τ in responses:
          r_acc = r_csv(τ) if MCQ else r_rouge(τ)  # ROUGE = (R1+R2+RL)/3
          r_fmt = 0.05 if has_tool_call(τ) and not correct(τ) else 0
          R(τ) = w_acc * r_acc + w_fmt * r_fmt
      # GRPO advantage + policy update
      A_i = (R_i - mean(R)) / std(R)
      loss = GRPO_clip_loss(π_θ, π_ref, A)
      MLLM.backward(loss)
      # 每 N 步: 收集 failures, teacher 生成新 QA
      if step % N == 0:
          new_QA = teacher_MLLM(HD-VILA_new_video, 
                                in_context=failures)
          EVA_RL = EVA_RL ∪ new_QA
  ```
  
  使用例子：
  ```bash
  # vLLM serving + inference
  python serve_eva.py \
      --model_path /path/to/EVA-GRPO \
      --base_model Qwen2.5-VL-7B-Instruct
  
  # 评估 LSDBench
  python eval_lsdbench.py --model eva --checkpoint EVA-GRPO
  ```

## DyCoke__Dynamic_Compression_of_Tokens_for_Fast_Video_Large_Language_Models

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：DyCoke 是一个 training-free 的两阶段视觉 token 动态压缩方法，专为 Video LLM 推理加速设计。核心流程：(1) Visual Token Temporal Merging (TTM) —— 在 prefilling 阶段，以滑动窗口（window length=4）对视频帧进行连续均匀采样，分为 O(Odd) 组和 E(Even) 组，计算相邻组对应位置 token 的余弦相似度 S = h_i·h_j / (||h_i|| ||h_j||)，剪枝 E 组中与 O 组高相似的 token，再计算 O 组内帧间相似度，保留窗口第一帧的完整 token，剪枝其余，使视觉 token 减少 k%（第一阶段约压缩 50-60%）；(2) KV Cache Dynamic Pruning —— 在 decoding 阶段，在第 L 层计算预测 token 与视觉 token 的 cross-attention 权重 A^(L) = Softmax(Q^(L) K^(L)^T / sqrt(D))，提取视觉 token 注意力分数子集 A_v^(L)，按 top p% 阈值 τ 保留高注意力 token 的 KV cache，将低注意力 token 存入 DP cache（Dynamic Pruning cache）；每隔 N 次迭代（当相邻迭代注意力分布余弦相似度低时），重新计算 cross-attention 矩阵，将 DP cache 中注意力回升的 token 动态加回 KV cache，同时将注意力下降的 token 移除至 DP cache。此阶段在 TTM 基础上进一步压缩 70-90% visual token。平均每帧保留约 15 tokens 参与注意力计算。该方法为 plug-and-play，三个超参数：K（第一阶剪枝率）、L（注意力评估层）、P（第二阶剪枝率），TTM 阶段处理时间 < 10^{-3} 秒（32 帧输入）。
  实验比较：(a) 在 ActivityNet-QA、NextQA、PerceptionTest、VideoDetailCaption、VideoMME、MVBench 六个 benchmark 上，与 FastV（基于 prefilling 阶段注意力分数的 one-shot 剪枝）和 LLaVA-PruMerge（基于 CLIP 视觉编码器注意力分数的 one-shot 剪枝）对比，覆盖 LLaVA-OV-0.5B/7B/72B 三种模型规模，使用 FLOPs 统一公平比较剪枝力度；(b) 效率分析 —— MVBench 上 16/32 帧的实际端到端推理 latency 和 GPU memory 对比，VideoDC 上 per-token decoding latency 对比；(c) Cost-Effectiveness —— 相同计算预算下增加输入帧数的性能收益（VideoMME: 16→32 frames）；(d) 消融实验 —— w/o DP cache（动态剪枝替换为 one-shot 剪枝）、Random Pruning（TTM 随机选择 token 替换相似度选择）、不同 L/P 超参数组合、TTM 过剪枝（K=0.9）的影响；(e) 附录 MVBench 32 帧输入下各子任务细粒度 accuracy；(f) 附录 K 值与输入帧数（8/16/32）的 joint ablation。

- 硬件平台是什么，配置是什么。
  LLaVA-OV-0.5B: NVIDIA RTX 4090 (24GB)；LLaVA-OV-7B: NVIDIA A6000 (48GB)；LLaVA-OV-72B: NVIDIA A100 (80GB)。框架：PyTorch。评估框架：LMMs-Eval（ActivityNet-QA, NextQA, PerceptionTest, VideoDetailCaption, VideoMME），MVBench 使用官方代码评估。

- 模型是什么。数据集和bench分别是什么。
  模型：LLaVA-OneVision-0.5B (d=896, m=4864, T=24), LLaVA-OneVision-7B (d=3584, m=18944, T=28), LLaVA-OneVision-72B (d=8192, m=29568, T=80)。每帧 token 数 N_v=196。默认 32 帧输入（MVBench 额外测 16 帧）。所有模型参数冻结，不做训练。
  Benchmarks：(1) ActivityNet-QA —— 人工标注的动作相关问答，用 GPT-4o-mini 评分 Accuracy 和 Score (0-5)；(2) PerceptionTest —— 感知能力诊断；(3) VideoMME —— 涵盖短/中/长视频多领域，含 w/o subs 和 w-subs 两个子集；(4) NextQA —— 时序动作解释问答；(5) VideoDetailCaption —— 视频细节描述，GPT-4o-mini 评分；(6) MVBench —— 20 个视频理解子任务（Action Sequence, Action Prediction, Action Antonymy, Fine-grained Action, Unexpected Action, Object Existence, Object Interaction, Object Shuffle, Movement Direction, Action Localization, Scene Transition, Action Counting, Movement Counting, Movement Attributes, State Change, Fine-grained Pose, Character Order, Egocentric Navigation, Episodic Reasoning, Counterfactual Inference），每个子任务 200 样本，多选题格式。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址：https://github.com/KD-TAO/DyCoke （Apache-2.0, CVPR 2025 accepted）。基于 LLaVA-NeXT 代码库，安装后通过 lmms-eval 调用，传入 dycoke=True, dycoke_k, dycoke_l, dycoke_p 参数启用。
  
  算法 pipeline 张量计算伪代码：
  ```
  # === DyCoke 两阶段压缩流程 ===
  # 输入: 视频 M_v 帧, 文本 prompt tokens
  # 超参数: K (TTM保留率), L (注意力评估层), P (DP保留率p%)

  # ---- Prefilling Stage ----
  # 视觉编码器: 每帧 → N_v 个 embedding tokens
  # H_v' shape: (M_v * N_v, D)  where D is hidden dim
  
  # Stage 1: Token Temporal Merging (TTM)
  # window_size = 4
  # 分组: O = frames[0::2] (odd indices), E = frames[1::2] (even indices)
  for i in range(0, M_v, window_size):
      window_tokens = H_v'[i * N_v : (i + window_size) * N_v]
      # 计算相邻组对应位置 token 余弦相似度
      S = cosine_similarity(O_tokens, E_tokens)  # (N_v,)
      # 剪枝 E 组中高相似 token
      mask_E = S < threshold  # top-K% by similarity pruned
      E_pruned = E_tokens[mask_E]
      # 组 O 内帧间相似度计算，保留第一帧完整
      O_first = O_tokens[0 * N_v : 1 * N_v]  # keep all
      for o_idx in range(1, len(O_tokens)//N_v):
          S_o = cosine_similarity(O_first, O_tokens[o_idx * N_v:...])
          mask_o = S_o < threshold_o
          O_pruned.append(O_tokens[o_idx][mask_o])
      merged = concat[O_first, O_pruned, E_pruned]

  H = concat[merged_tokens, text_tokens]  # LLM 输入

  # Standard prefilling: 计算 Q, K, V 并填充 KV cache
  # for each transformer layer l:
  #   Q^l = H W_Q^l, K^l = H W_K^l, V^l = H W_V^l
  #   KV_cache[l] = (K^l, V^l)

  # ---- Decoding Stage ----
  # Stage 2: KV Cache Dynamic Pruning
  # DP_cache = {}   (pruned tokens backup)
  
  for decoding_step t in range(1, max_new_tokens):
      # 每隔 N 次迭代 (当相邻迭代注意力分布余弦相似度低时)
      if t % N == 0 or similarity_low:
          # 在第 L 层计算 cross-attention
          A^(L) = Softmax(Q^(L)_pred K^(L)_visual^T / sqrt(D))
          # Q^(L)_pred shape: (1, D), K^(L)_visual shape: (N_v_remaining, D)
          # A^(L)_v shape: (N_v_remaining,)  — 预测 token 对各视觉 token 的 attention
          
          # 按 top P% 保留
          threshold = percentile(A^(L)_v, 100 - P)
          keep_idx = where(A^(L)_v >= threshold)
          prune_idx = where(A^(L)_v < threshold)
          
          # 更新 KV cache: 保留高注意力 token, 低注意力 → DP cache
          KV_cache_visual[L] = KV_cache_visual[L][keep_idx]
          DP_cache[L] = DP_cache[L] ∪ KV_cache_full[L][prune_idx]
          
          # 将 DP cache 中注意力回升的 token 加回
          A_dp = recompute_attention(Q^(L)_pred, DP_cache[L])
          recall_idx = where(A_dp >= threshold_new)
          KV_cache_visual[L] = KV_cache_visual[L] ∪ DP_cache[L][recall_idx]
          DP_cache[L] = DP_cache[L] - DP_cache[L][recall_idx]

      # 用压缩后的 KV cache 计算 attention 并自回归生成
      h_t = LLM_decode(KV_cache)  # 仅需计算当前 token 的 K/V
      KV_cache = concat[KV_cache, (h_t W_K, h_t W_V)]
  ```

  使用例子（基于开源仓库）：
  ```bash
  # 评估 DyCoke on MVBench (LLaVA-OV-7B)
  accelerate launch --num_processes=1 \
    -m lmms_eval \
    --model llava_onevision \
    --model_args pretrained="lmms-lab/llava-onevision-qwen2-7b-ov",conv_template=qwen_1_5,model_name=llava_qwen,device_map=auto,dycoke=True,dycoke_k=0.5,dycoke_l=3,dycoke_p=0.7 \
    --tasks mvbench \
    --batch_size 1 \
    --log_samples \
    --output_path ./logs
  ```

## Divide__then_Ground__Adapting_Frame_Selection_to_Query_Types_for_Long-Form_Video_Understanding

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：DIG 是一个 training-free 的帧选择框架，按查询类型自适应选择采样策略。核心流程：(1) Query Identification —— 用 Qwen3-Next-80B-A3B 通过 Chain-of-Thought prompting 将查询分类为 global query（需要整体视频理解）或 localized query（针对特定时间片段）；(2) 对 global query 直接用 uniform sampling；对 localized query 启动三阶段 pipeline：(a) Content-Adaptive Frame Selection (CAFS) —— 以 2 fps 采样 M 帧，用 DINOv2 提取特征向量，计算相邻帧余弦距离 d_i = 1 - sim(V_I_i, V_I_{i+1})，检测距离峰值（prominence > 0.1）作为分割点，取相邻分割点间中间帧作为 r-frames（代表帧）；(b) Reward Assignment —— 用 LMM 对每个 r-frame 进行二维评分（0-100）：(1) 帧对问题的直接有用性，(2) 帧是否暗示相邻帧包含补充信息；(c) Video Refinement —— 迭代式 reward-guided selection：计算均值 R̄，每轮更新 R'_j = max(R_j - R̄, 0)，迭代至奖励集稳定，以最终正值 r-frames 为关键帧；对每个关键帧取其周围窗口 [K_{j-wlen}, K_{j+wlen+1}]（wlen=2）的视频段做 union，得到 refined video，最后从 refined video 中 uniform sampling 作为 LMM 输入。每帧用 56 tokens 表示。
  实验比较：(a) 与 uniform sampling (UNI)、AKS [18]、Q-Frame [64] 在 MLVU、LongVideoBench (LVB)、VideoMME (仅 medium+long splits) 上的 accuracy 对比，覆盖 Qwen2.5-VL-7B 和 Qwen2.5-VL-32B 两种 LMM，帧数从 8 到 256；(b) 模块分析 —— 6.1 帧选择策略 vs 查询类型：global vs localized 上的 uniform vs pipeline 性能对比；(c) 6.2 CAFS 有效性 —— LoC（Localized Coverage）和 GIC（Global Coverage）指标，与 UNI 和 FPS 对比，CAFS 替换为 uniform sampling 的消融；(d) 6.3 Reward Assignment —— LMM-based reward vs CLIPScore 对比（Qwen2.5-VL-7B/32B 作为 reward assigner）；(e) 6.4 窗口长度 wlen ∈ {0,2,4,8} 消融；(f) 6.5 效率分析 —— FLOPs vs accuracy 散点图；(g) 扩展实验 —— Qwen3-VL-8B 上 8 至 768 帧的 scalability 测试，对比 UNI 和 AKS；(h) 附录 F.3 逐任务 breakdown（MLVU: PQA/NQA/AC/AO/ER/AR/TR, VideoMME: ORA/ORC/ARA/INS/COP/TER/TEP/SPP/SPR/OCR/ATP/ACR, LVB: L1-Perception/L2-Relation 子任务）；(i) 附录 G 效率分析 —— runtime profiling（QI/CAFS/RA/VR 各阶段耗时）和 Query Identification 带来的效率增益。

- 硬件平台是什么，配置是什么。
  所有实验在 8 × NVIDIA A100 GPU 节点上执行。推理加速使用 vLLM backend（用于 query identification 和 reward assignment 阶段）。评估框架为 LMMs-Eval。

- 模型是什么。数据集和bench分别是什么。
  模型：(1) 主 LMM —— Qwen2.5-VL-7B 和 Qwen2.5-VL-32B（推理 backbone）；(2) Query Identification LLM —— Qwen3-Next-80B-A3B-Instruct；(3) 特征提取 —— DINOv2（CAFS 阶段）；(4) 扩展实验 —— Qwen3-VL-8B。所有模型参数冻结，不做训练。
  Benchmarks（3 个，均不使用字幕/音频，仅视频+问题）：
  (1) MLVU [54] —— 多任务长视频理解，~2600 QA pairs（dev set MC questions），平均时长 636.2s，含 462 global + 1708 localized queries，任务分 PlotQA、NeedleQA、Action Count、Action Order、Ego Reasoning、Anomaly Recognition、Topic Reasoning；
  (2) LongVideoBench [55] —— 3763 videos / 6678 QA pairs（val split 1337），平均时长 732.2s，全为 localized queries（referring reasoning），任务分 L1-Perception（S2E/S2A/O2E/T2O/S2O/T2E/E2O/T2A）和 L2-Relation（TOS/E3E/SAA/O3O/T3O/T3E/TAA/SSS/SOS）；
  (3) VideoMME [56] —— 900 videos / 2700 QA pairs，仅用 medium（516.8s）和 long（2466.3s）splits，含 479 global + 2221 localized queries，任务分 Object Reasoning、Object Recognition、Action Reasoning、Information Synopsis、Counting Problem、Temporal Reasoning、Temporal Perception、Spatial Perception、Spatial Reasoning、OCR、Attribute Perception、Action Recognition。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址：https://github.com/Jialuo-Li/DIG
  
  算法 pipeline 伪代码：
  ```
  # === DIG 推理流程 ===
  # 输入: 视频 V (T frames, 原始帧率), 查询 Q
  # 参数: wlen=2, 每帧 56 tokens
  
  # Step 1: Query Identification
  is_global = LLM_classify(Q)  # Qwen3-Next-80B-A3B, CoT prompt
  if is_global:
      frames = uniform_sample(V, N)
      return LMM_inference(frames, Q)   # 直接推理
  
  # Step 2: Content-Adaptive Frame Selection (CAFS)
  # 以 2 fps 采样 M 帧
  F = sample_2fps(V)  # {f_{I_i}}_{i=1}^M
  # DINOv2 特征提取
  for i in 1..M:
      V_i = DINOv2(f_{I_i})   # shape: (d_dino,)
  # 逐帧距离计算
  for i in 1..M-1:
      d_i = 1 - cosine_sim(V_i, V_{i+1})  # scalar
  # 峰值检测 + prominence 过滤
  peaks = {i | d_{i-1} < d_i and d_{i+1} < d_i}
  valid_peaks = {j in peaks | prominence(d_j) > 0.1}
  # 选 r-frames (相邻峰值中间帧)
  R_idx = {(I_{p1} + I_{p2})/2 for consecutive p1, p2 in valid_peaks}
  r_frames = {f_I | I in R_idx}  # N_r 个代表帧
  
  # Step 3: Reward Assignment (LMM-based)
  for each r_frame in r_frames:
      score = LMM_reward(r_frame, Q)  # 0-100, 二维评分
      # prompt: 描述 + 打分 (直接有用性 + 相邻帧补充信息)
  rewards = [R_1, R_2, ..., R_{N_r}]
  
  # Step 4: Video Refinement (迭代式 reward-guided selection)
  while not converged:
      R_mean = mean(rewards)
      rewards = [max(R_i - R_mean, 0) for R_i in rewards]
      positives = {j | rewards[j] > 0}
      if positives == prev_positives: break
  
  # Step 5: Segment Combination
  refined_segments = []
  for each j where rewards[j] > 0:
      # 窗口合并: [K_{j-wlen}, K_{j+wlen+1}]
      segment = V[K_{j-wlen} : K_{j+wlen+1}]
      refined_segments.append(segment)
  refined_video = union(refined_segments)
  
  # Step 6: Final Sampling & Inference
  final_frames = uniform_sample(refined_video, N)
  answer = LMM_inference(final_frames, Q)
  ```
  
  张量计算细节：
  - DINOv2 特征：`V_i ∈ R^d, d=768`（ViT-B 或 ViT-L），为 frame-level global feature
  - 余弦距离：`d_i = 1 - (V_i · V_{i+1}) / (||V_i|| · ||V_{i+1}||)`，标量
  - Prominence 计算：左基线 l_min = min(d_k)，其中 k 从 j-1 向左搜索到 d_k > d_j 为止；右基线同理；prominence = d_j - max(l_min, r_min)
  - Reward 二维评分：prompt 驱动 LMM 输出 {"description": str, "reward": int}
  - 迭代收敛：reward 集不再变化时终止，无需预设 Top-K

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：D-CoDe 是一个 training-free 的视频理解框架，将图像预训练的 VLM（LLaVA-NeXT 7B）扩展到视频领域，由两个组件组成：(1) Dynamic Compression —— 时间维度：先从视频中均匀采样 ⌊α·N⌋ 帧（α=0.85），再从剩余帧中迭代选择与已选帧语义最不相似的 supplementary frame（基于 CLIP global feature 的余弦相似度），共选 N 帧；空间维度：对每帧的 M 个 visual token 按 ℓ2 norm 计算 salience，保留 top-⌊β·M⌋ 高激活 token（β=0.625），然后按余弦相似度（阈值 τ=0.9）贪婪合并冗余 token（anchor + cluster 取平均），输出压缩后的 token 集；(2) Question Decomposition —— 用 GPT-3.5-turbo-0125（temperature t=0.5）将原始问题分解为多个子问题，每个子问题独立用压缩后的 visual tokens 推理得到子答案，将子答案拼接后与原始问题、压缩 visual tokens 一起送入 LLM 生成最终答案。
  实验比较：(a) 与 training-required 方法（Video-LLaVA, Video-LLaMA2, MovieChat+, Vista-LLaMA）和 training-free 方法（DeepStack-L, M3, IG-VLM, SF-LLaVA, TS-LLaVA）在 Multiple Choice VideoQA（NExT-QA, EgoSchema, IntentQA）和 Open-Ended VideoQA（MSVD-QA, MSRVTT-QA, TGIF-QA, ANet-QA）上的 accuracy 对比；(b) 模块消融（EgoSchema）—— Baseline → +dynamic spatial token compression → +dynamic temporal frame selection → +question decomposition 的逐级增益；(c) 采样策略消融 —— Uniform vs Question-aware vs Supplementary Frame Selection；(d) 压缩范围消融 —— 空间 token 合并的距离约束的影响；(e) Decomposition Prompt 消融 —— 不同 prompt 变体的影响；(f) Decomposed Content 消融 —— 子问题 vs 子答案的有效性对比；(g) 超参数消融 —— α（uniform ratio）、β（retention ratio）、τ（similarity threshold）、t（temperature）在 EgoSchema 上的 sensitivity；(h) 错误分析 —— 频繁场景切换视频上的性能退化。

- 硬件平台是什么，配置是什么。
  所有实验在单卡 NVIDIA RTX A6000 GPU 上执行。基础模型 LLaVA-NeXT 7B（Vicuna-7B LLM），使用 RoPE 缩放因子 2 扩展 context length 到 8192 tokens。Question Decomposition 使用 OpenAI API 调用 gpt-3.5-turbo-0125。

- 模型是什么。数据集和bench分别是什么。
  模型：LLaVA-NeXT 7B（基于 Vicuna-7B LLM + CLIP 视觉编码器），所有参数冻结不做微调。
  Benchmarks（7 个）：(1) Multiple Choice VideoQA —— NExT-QA（因果和时间理解）、EgoSchema（自我中心长视频模式级理解）、IntentQA（意图识别），metric 为 Accuracy；(2) Open-Ended VideoQA —— MSVD-QA（短视频文本描述）、MSRVTT-QA（多样化网络视频）、TGIF-QA（GIF 中的重复计数和状态转换）、ActivityNet-QA（长视频丰富活动语义），metric 为 GPT-Accuracy（事实正确性）和 GPT-Score 0-5（完整性和流畅性），统一使用 gpt-3.5-turbo-0125 评估。
  帧采样：每视频采样 N 帧，N 根据数据集平均视频长度经验性确定。所有帧统一 resize 到 336×336。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址：https://github.com/hukcc/D-CoDe
  代码结构：`Dcode.py`（核心实现，含 `generate_subquestions()`、`supp_frame_selection()`、`token_select_and_merge()`）、`dataset.py`、`prompt.py`、`utils.py`、`run_inference_multiple_choice_qa.py`、`run_inference_video_qa.py`、`scripts/`（各 benchmark 评估脚本）、`eval/`（评估代码）。
  
  算法 pipeline 伪代码：
  ```
  # === D-CoDe 推理流程 ===
  # 输入: 视频 V (T frames), 问题 Q
  # 参数: N=15 (selected frames), α=0.85, β=0.625, τ=0.9, t=0.5

  # Step 1: 视频帧预编码
  for t in 1..T:
      g_t = CLIP_visual(I_t)         # 全局 CLIP 特征, shape: (d_clip,)
      F_t = VisualEnc(I_t)           # 逐帧 visual tokens, shape: (M, d)

  # Step 2: Dynamic Temporal Frame Selection
  # Stage 1: 均匀采样
  N_uniform = floor(α * N)           # e.g. 0.85 * 15 = 12
  V_uniform = uniform_sample({I_t}, N_uniform)
  V_selected = V_uniform

  # Stage 2: 补充帧选择 (基于语义多样性)
  for k in 1..(N - N_uniform):
      for each I_m in V \ V_selected:
          s_m = mean(cosine_sim(g_m, g_n) for I_n in V_selected)
      I* = argmin(s_m)               # 选与已选帧最不相似的帧
      V_selected = V_selected ∪ {I*}

  # Step 3: Dynamic Spatial Token Compression (每帧独立)
  F_compressed = []
  for each I_k in V_selected:
      F = VisualEnc(I_k)             # shape: (M, d)
      # 3a: Token Pruning (按 ℓ2 norm)
      a_i = ||f_i||_2 for i in 1..M
      K = floor(β * M)
      F_pruned = TopK(F, key=a_i, k=K)  # (K, d)
      
      # 3b: Greedy Token Merging (按余弦相似度)
      π = argsort({a_i}, descending)  # 按 salience 降序排列
      merged_set = []
      for i in π:
          if f_i is not merged:
              N_i = {j | cosine_sim(f_i, f_j) >= τ, f_j is unmerged}
              f_i_rep = (f_i + Σ_{j∈N_i} f_j) / (1 + |N_i|)
              mark all j in N_i as merged
              merged_set.append(f_i_rep)
      F_compressed.append(merged_set)

  # Step 4: 拼接压缩后 visual tokens
  F_final = concat(F_compressed)     # shape: (Σ|merged_set_k|, d)

  # Step 5: Question Decomposition
  prompt = "I am working on a video understanding task. ..."
  Q_1..Q_n = GPT3.5(Q, prompt, temperature=t)  # n 不限制

  # Step 6: 逐子问题推理
  A_sub = []
  for Q_i in Q_1..Q_n:
      A_i = LLaVA_NeXT(F_final, Q_i)
      A_sub.append(A_i)

  # Step 7: 最终答案生成
  A_final = LLaVA_NeXT(F_final, concat(A_sub), Q)
  ```

  关键张量维度：
  - 视频帧: T 帧(取决于视频长度), 采样 N 帧(N 经验性确定, EgoSchema 用 15)
  - 每帧 visual tokens M: LLaVA-NeXT 编码后 ~576 tokens (336×336, patch_size=14 或类似)
  - CLIP global feature d_clip: ~768 或 ~1024 (取决于 CLIP 变体)
  - Visual token hidden dim d: LLaVA-NeXT 7B 的 hidden size (~4096)
  - α=0.85: 85% 帧来自 uniform sampling, 15% 来自 supplementary selection
  - β=0.625: 保留 62.5% 高 ℓ2 norm tokens
  - τ=0.9: 余弦相似度 >= 0.9 的 token 被合并
  - RoPE 缩放因子 2, context length 8192

## A_Glimpse_to_Compress__Dynamic_Visual_Token_Pruning_for_Large_Vision-Language_Models

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：GlimpsePrune 是一个基于数据驱动的动态视觉 token 剪枝框架，受人类视觉认知的"一瞥"机制启发。核心组件：(1) Glimpse Token —— 在 LLM decoder 的 prefill 阶段，将可学习的 glimpse token 嵌入追加到指令 token 之后，利用因果注意力机制与所有 visual token 交互，在中间层 K 提取 glimpse token 对所有 visual token 的 cross-attention 分数；(2) Visual Importance Predictor (VIP) —— 由 M=4 层 Self-Attention Block 组成的轻量选择模块，以 glimpse attention A 和多层 visual features V 为输入，其中 visual features 通过 conditional self-attention（2D RoPE）注入到 query/key 中作为额外的视觉先验条件，输出每个 visual token 的重要性 map P；(3) One-shot Pruning —— 在 LLM decoder 第 K = ⌈2/3 × L⌉ 层进行一次性剪枝，移除不重要的 visual token 及其在所有前序层的 KV cache 条目，glimpse token 随即丢弃，后续 L-K 层 prefill 和全部 decoding 阶段均在缩减后的序列上运行。
  实验比较：(a) 与 SOTA 方法（PDrop[7], VisionZip[4], DivPrune[23], CDPruner[20], VScan[5]）在 free-form VQA（12个数据集）和 short-form VQA（10个 benchmark）上的 accuracy 对比，在三种 retention rate 约束下（≤11.1%, ≤22.2%, ≤33.3%）；(b) 不同架构上的泛化（Qwen2.5-VL-7B, LLaVA-1.5-7B, Qwen2.5-VL-3B, LLaVA-1.5-13B）；(c) 消融实验 —— glimpse token 的训练、语言 loss/定位 loss 的贡献、visual condition 的必要性、pruning layer K 的选择（Qwen2.5-VL-3B）；(d) 效率分析 —— prefill FLOPs、decoding 阶段初始 KV cache 长度、峰值内存、时间开销；(e) RL fine-tuning（GlimpsePrune+）—— 结合 GRPO 的强化学习微调在 free-form VQA 上达到 110% baseline 性能。

- 硬件平台是什么，配置是什么。
  训练：单卡 NVIDIA A100 GPU（训练 GlimpsePrune 约 0.5 小时）。RL fine-tuning：4× NVIDIA A100 GPU。
  推理评估：单卡 NVIDIA A100 GPU，100 个 DocVQA 样本测量 prefill/decoding 性能。

- 模型是什么。数据集和bench分别是什么。
  模型：Qwen2.5-VL-7B（主实验，28层 LLM decoder，支持 4~16384 visual tokens 动态高分辨率输入），Qwen2.5-VL-3B（消融实验，36层 LLM decoder），LLaVA-1.5-7B/13B（固定 576 visual tokens），评估模型用 Qwen2.5-32B-Instruct-GPTQ-Int8 作为 evaluator LLM。
  Free-form VQA 数据集（12个）：Flickr30k, V7W, GQA, OpenImages, VSR, CUB, DocVQA, TextCaps, TextVQA, DUDE, SROIE, InfoVQA —— 全部来自 VisCoT benchmark [36]。
  Short-form VQA 数据集（10个）：VQAv2, GQA, VizWiz, ScienceQA, POPE, MME, MMBench(en/cn), SEEDBench, V* bench —— 使用 LMMs-Eval 框架 [39] 评估。
  训练数据：GQA 数据集随机 20K 样本（训练 glimpse token + VIP，1 epoch）。RL fine-tuning：VisCoT 数据集随机 240K 样本（12个数据集各取 20K，10K 短回答 + 10K 自由回答）。
  评价指标：Free-form VQA 用 Qwen2.5-32B-Instruct-GPTQ-Int8 评分（0-1 分）；Short-form VQA 用 exact-match/rule-based string comparison。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址：https://github.com/HVision-NKU/GlimpsePrune
  
  算法 pipeline 伪代码：
  ```
  # === Prefill 阶段前 K 层（含 Glimpse Token） ===
  # 输入: visual tokens V (shape: N_v × D)，text tokens T (shape: N_t × D)
  # 插入 glimpse token g (可学习嵌入矩阵 G ∈ R^{L × D})
  X_0 = concat([V, T, g[0]])  # shape: (N_v + N_t + 1) × D

  for l in 1..K:
      # 标准 causal self-attention + FFN
      X_l = DecoderLayer_l(X_{l-1})
      # 在 glimpse token 位置加上对应层的可学习嵌入
      X_l[-1] += g[l]

  # === 提取 Glimpse Attention（第 K 层） ===
  # 从第 K 层的 attention 计算中提取 glimpse token 对 visual tokens 的 cross-attention
  Q_g = W_Q @ X_K[-1]   # shape: H × D_head (multi-head)
  K_v = W_K @ X_K[:N_v] # shape: N_v × H × D_head
  A = Softmax(Q_g @ K_v^T / sqrt(D_head))  # glimpse attention, shape: N_v × H

  # === VIP: 视觉重要性预测 ===
  # 输入: A (glimpse attention, N_v × H), V_hier = {V_m | m ∈ selected_layers} (M=4层多尺度视觉特征)
  A' = Linear(A)  # projection: (N_v × H) → (N_v × E), E=256
  for m in 1..M:
      V_m' = Linear_v(V_m)  # projection: (N_v × C) → (N_v × F), F=512
      # Conditional Self-Attention with 2D RoPE
      Q_m = W_Q_vip @ A'     # shape: N_v × E
      K_m = W_K_vip @ A'
      # 将 visual feature 拼接到 Q/K 中作为额外条件
      Q_m' = Q_m + Linear_q_cond(V_m')  # visual conditioning
      K_m' = K_m + Linear_k_cond(V_m')
      V_m_val = W_V_vip @ A'
      A' = A' + SelfAttn_2DRoPE(Q_m', K_m', V_m_val)
  P = Sigmoid(Linear_out(A'))  # importance map, shape: N_v, per-token prob ∈ [0,1]

  # === One-shot Pruning（第 K 层） ===
  # 选 Top-N_v' 个最高 P 值的 visual token（N_v' = min(retention_rate × N_v, max_retain)）
  mask = TopK(P, N_v')
  X_K_pruned = X_K[mask]     # 移除不重要 visual token 的 hidden state
  # KV cache 中所有层（1..K）对应的不重要 visual token 条目同步移除
  for l in 1..K:
      KV_cache[l] = KV_cache[l][mask]  # 对应 visual token 位置的 K/V
  # 丢弃 glimpse token
  X_K_pruned = X_K_pruned[:-1]

  # === 剩余 Prefill 层（K+1..L） ===
  for l in K+1..L:
      X_l = DecoderLayer_l(X_{l-1})  # 序列长度 = N_v' + N_t

  # === Decoding 阶段 ===
  # 从 X_L 生成第一个 token，随后 autoregressively 生成，所有 token 在缩减后的 KV cache 上运行
  # 每个 decoding step 的 attention cost: O((N_v' + N_t + n_generated) × D)，而非 O((N_v + N_t + n_generated) × D)
  ```
  
  关键张量维度：
  - Visual tokens N_v: Qwen2.5-VL 支持 4~16384, LLaVA-1.5 固定 576
  - LLM hidden size D: 2048(Qwen2.5-VL-3B) / 3584(Qwen2.5-VL-7B) / 4096(LLaVA-1.5-7B) / 5120(LLaVA-1.5-13B)
  - VIP hidden size E=256, visual condition size F=512
  - M=4 个 self-attention block, 4 个 attention head
  - Pruning layer K: Qwen2.5-VL-7B: K=19, Qwen2.5-VL-3B: K=24, LLaVA-1.5-7B: K=22, LLaVA-1.5-13B: K=27
  - 训练目标: L_total = L_language + L_dice + 0.1 × L_bce，其中 L_dice 和 L_bce 基于 GQA 的 bounding box ground truth
  
  训练流程（GlimpsePrune+ RL fine-tuning）：
  ```
  # Phase 1: 训练 glimpse token + VIP（LVLM 参数冻结）
  # Phase 2: RL fine-tuning 循环
  for iteration in 1..N:
      # Step 1: 用当前 pruner 进行 token 剪枝
      P = VIP(glimpse_attn, visual_features)
      X_pruned = prune(X, P, retention_rate)
      
      # Step 2: GRPO 策略优化（group=4）
      for group in 1..4:
          response = PolicyModel.generate(X_pruned)  # LoRA rank=64
      reward = RewardModel(question, response, standard_answer)
      L_policy = GRPO_loss(responses, rewards)  # group-wise ranking
      L_KL = 0.04 × KL(PolicyModel || ReferenceModel)
      Update(PolicyModel, L_policy + L_KL)
      
      # Step 3: 更新 glimpse token + VIP
      Update(glimpse_token, VIP, L_language + L_loc)
  ```

## Atlas__Multi-Scale_Attention_Improves_Long_Context_Image_Modeling

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：Multi-Scale Attention (MSA) 是一种面向高分辨率(long-context)图像建模的新型注意力原语，核心设计包含两个组件：(1) Hierarchical Representation —— 使用固定大小的 S-token max-pooling 核（stride=s，S=s²，如 4×4 strided max-pool），从输入 feature map F^(1) 开始，迭代生成 O(log_S N) 个空间尺度的粗粒度摘要表示 F^(l) = S(F^(l-1), S)；(2) Bi-directional Cross-Scale Communication —— Top-Down（Global Context Aggregation）：每个窗口内的 token 通过 dense cross-attention 与同一窗口内所有 token 及所有更粗尺度(L+1..L)的对应 child window tokens 交互；Bottom-Up（Fine-to-Coarse Refinement）：每个粗尺度 token 通过 cross-attention 从直接 parent window 的细粒度 token 恢复局部细节。每个 token 到任意其他 token 的通信复杂度为 O(log N)（通过中间粗尺度 token），总 runtime 复杂度 O(N·K·log_S N)（K=window size=256 tokens, S=16）。基于 MSA 构建 Atlas 架构：采用 progressive scale-dropping 策略，以 L 个 macro-stage（等于 scale 数）逐步放弃最精细尺度，将计算资源集中于高层特征。例如 4-scale MSA 采用 D={2,2,2,6} 配置：前 2 个 block 处理所有 4 个 scale，之后逐步丢弃 scale-1/2/3，最后一个 block 仅处理 scale-4。附录 C 引入 QKV Caching 机制避免跨尺度 cross-attention 的重复 QKV 重计算。
  实验比较：(a) 架构比较（Table 1）—— Atlas-B/16 vs ViT-B, Swin-B, FasterViT-4, LongViT-B, ConvNext-B, MambaVision-B 在 1024×1024 HR-IN100 上训练 320 epoch 的 runtime vs accuracy；(b) Long-Context 扩展实验（Table 2）—— Atlas-S/16 vs MambaVision-S/16 在 1024/2048/4096px（最高 64K tokens）训练 100 epoch；(c) Block-level 消融（Table 3）—— MSA block vs Window ViT, ShiftedWindow ViT (Swin), ViT, Hierarchical Attention (FasterViT), Dilated Attention (LongViT), MambaVisionMixer，384×384 输入、9216 tokens、4-block Base 架构、100 epoch；(d) Communication Mechanism 消融（Table 4）—— single-scale only / multi-scale only / +bottom-up / +top-down / +both (MSA)，256×256 输入、4096 tokens；(e) Composition Strategies 消融（Table 5）—— Stack vs Conv Downsampling vs Atlas (D2D10)，512×512 输入、4096 tokens；(f) 50-epoch 多分辨率实验（Table 6）—— Atlas-B/16 vs ViT/WViT/ConvNext/FasterViT/LongViT/MambaVision 在 256/512/1024/2048px。

- 硬件平台是什么，配置是什么。
  训练与评估：单节点 8× NVIDIA H100 GPU。所有 runtime 计时在该硬件上 wall-clock 测量。训练使用 linearly decaying learning rate proportional to batch size (Goyal, 2017)。

- 模型是什么。数据集和bench分别是什么。
  模型：Atlas-B/16（12 head, 768 embed-dim, ~86M params）、Atlas-S/16（6 head, 384 dim, ~25M params）。Baseline 模型：ViT-B（standard Transformer）、Swin-B、FasterViT-4（Hierarchical Attention）、LongViT-B（Dilated Attention）、ConvNext-B（纯卷积）、MambaVision-B/S（Hybrid SSM+Attention）。所有模型 patch_size=16，Base 模型使用 12 head / 768 dim，Small 模型使用 6 head / 384 dim。
  数据集：High-Resolution ImageNet-100 (HR-IN100)，从 ImageNet-1K 上采样到目标分辨率（1024px~4096px），~126K 训练样本、5000 验证样本、100 类。评估分辨率：1024×1024（4096 tokens）、2048×2048（16384 tokens）、4096×4096（65536 tokens）。
  Metric：Top-1 Accuracy (%), Runtime (hours/minutes), Relative Speedup。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址：https://github.com/yalalab/atlas

  算法 pipeline 伪代码（MSA Block，对应 Algorithm 1）：
  ```
  # === MSA Block: input 为多尺度特征列表 X = [X^(1), ..., X^(L)] ===
  # 参数: k×k window size, S downsampling rate (stride s, S=s²)
  # X^(l) shape: (B, N_l, C), N_l = 每个 scale 的序列长度

  # 1. Iterative Summarization（fine→coarse 构建多尺度）
  for l in 2..L:
      X^(l) += Summarize(X^(l-1), S)   # strided max-pool, stride=s
  # X^(l) shape: (B, N_l, C), 其中 N_l = N_{l-1} / S

  # 2. Top-Down Communication: Global Context Aggregation (coarse→fine)
  for l in L..1:  # 从最粗到最细
      # 对 scale l 每个 window W^(l):
      #   Q_l = Linear_Q(W^(l))         # (B, K, C_head)
      #   K_{l:L} = concat([Linear_K(W^(l)), Linear_K(W^(l+1)), ..., Linear_K(W^(L))])
      #   V_{l:L} = concat([Linear_V(W^(l)), Linear_V(W^(l+1)), ..., Linear_V(W^(L))])
      #   W^(l) = Softmax(Q_l @ K_{l:L}^T / sqrt(d)) @ V_{l:L}
      X^(l) = CrossAttention(Q=X^(l), KV=concat([X^(l), X^(l+1), ..., X^(L)]))

  # 3. Bottom-Up Communication: Fine-to-Coarse Refinement
  for l in 2..L:  # 从第二细到最粗
      # 每个粗尺度 token 仅 cross-attend 其直接 parent window:
      #   Q_l = Linear_Q(W^(l))
      #   K_{l-1} = Linear_K(W_parent^(l-1))
      #   V_{l-1} = Linear_V(W_parent^(l-1))
      #   W^(l) = Softmax(Q_l @ K_{l-1}^T / sqrt(d)) @ V_{l-1}
      X^(l) = CrossAttention(Q=X^(l), KV=X_parent^(l-1))

  return [X^(1), ..., X^(L)]
  ```

  伪代码（Atlas Architecture，对应 Algorithm 2）：
  ```
  # === Atlas Architecture ===
  # 参数: k×k window size, P patch size, S downsample rate, D={d_1,...,d_L}
  # 输入: Img (B, H_in, W_in, C_in)

  # 0. Conv Stem（与 FasterViT 相同）
  X^(1) = ConvStem(Img, P)  # 两层 residual conv → (B, H/16, W/16, C)

  # 1. 初始化多尺度特征
  for l in 2..L:
      X^(l) = StridedMaxPool(X^(l-1), S)
  # X = [X^(1), X^(2), ..., X^(L)]

  # 2. Progressive Downsampling Stages
  for s in 1..L:  # stage s 仅处理 scale s..L
      for blk in 1..d_s:
          [X^(s), ..., X^(L)] = MSABlock([X^(s), ..., X^(L)], k, S)
      # 完成 d_s 个 block 后，scale s 被丢弃

  # 3. Readout
  predictions = readout(X^(L))  # 使用最粗尺度的特征
  return predictions
  ```

  QKV Caching 优化（Appendix C）：
  ```
  # 维护每个 scale l 的 QKV 缓存
  cache = {l: None for l in 1..L}

  def get_qkv(X^(l), cache):
      if cache[l] is None or feature_changed(X^(l)):
          cache[l] = (Q_proj(X^(l)), K_proj(X^(l)), V_proj(X^(l)))
      return cache[l]

  # Cache 更新时机: self-attention at scale L 后，以及每次 cross-attention 后
  ```

  关键张量维度：
  - Input: 1024×1024×3 → 4096 tokens (patch_size=16), 2048×2048 → 16384 tokens, 4096×4096 → 65536 tokens
  - Window size K = 256 (16×16), Downsampling rate S = 16 (4×4 strided max-pool)
  - Scale 数 L = log_S N: 1024px → L=3, 2048px → L=4, 4096px → L=4
  - Atlas config: {d1,...,dL}, 如 4-scale → D={2,2,2,6}
  - Base model: 12 heads, 768 embed-dim; Small model: 6 heads, 384 dim
  - 每个 token 到任意其他 token 通信步数 ≤ log_S N（通过粗尺度中间 token）
  - Runtime complexity: O(N · K · log_S N)，K=256, S=16

## AdaptVision__Efficient_Vision-Language_Models_via_Adaptive_Visual_Acquisition

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：AdaptVision 是一个基于强化学习的自适应视觉 token 获取框架，借鉴人类 active vision 的 coarse-to-fine 机制：(1) 首先用 1/4 低分辨率图像（25% visual tokens）处理，模型自主决定是直接回答还是调用 bounding box tool 裁剪高分辨率关键区域；(2) 若 tool call，从原始高分辨率图像中裁剪 bbox 区域 Icrop，合并到序列中继续推理后生成最终回答；(3) 训练使用 Decoupled Turn Policy Optimization (DTPO) —— 将策略损失按 turn 解耦为 Tool Token 和 Answer Token 两部分分别归一化，并分别计算 tool advantage A_tool 和 outcome advantage A_oc 进行精确 credit assignment；(4) 奖励函数包含 Outcome Reward（准确度 R_acc + 格式 R_form + 平衡 R_bal）和 Tool Reward（裁剪正确性 R_crop - α×面积惩罚 R_area）。
  实验比较：(a) 与静态 token 压缩方法（FastV 50%, SparseVLM 50%/70%, VisionZip 50%/70%）和动态方法（VisionThink, VisionThink†）在 9 个 VQA benchmark 上的性能和 token 消耗对比（Table 1, Table 4）；(b) 与 Down-Sample（25% token）对比验证 coarse-to-fine 有效弥补低分辨率的信息损失（+5.8% avg 仅 +7% token）；(c) 推理延迟对比（Fig. 4，vs Vanilla/VisionThink†，1.67× speedup）；(d) 消融实验：reward 设计（balance reward / tool reward 的单独移除对训练稳定性的影响，Fig. 5a）；(e) GRPO vs DTPO 训练动力学对比（Fig. 5b, Fig. 6a）；(f) 自适应 tool-use 分析（各 benchmark 的 tool call ratio，Fig. 6b）；(g) 超参数 sensitivity（λ 和 α，Table 2）；(h) 不同 reward model 对比（GPT-4o vs Qwen3-VL-4B 作为 judge，Table 3）。

- 硬件平台是什么，配置是什么。
  训练：4 节点 × 每节点 8× NVIDIA H20 GPU（共 32× H20 GPU），FP16 mixed-precision 训练。
  推理评估：使用 vLLM 框架，temperature=0。

- 模型是什么。数据集和bench分别是什么。
  模型：Qwen2.5-VL-7B-Instruct（视觉编码器 CLIP-ViT + modality projector + 7B LLM decoder）。
  RL 训练框架：veRL（https://github.com/volcengine/verl）。
  训练数据：来自 Yang et al. VisionThink-Smart-Train 数据集（https://huggingface.co/datasets/Senqiao/VisionThink-Smart-Train），包含可用低分辨率直接回答的 VQA 样本和需要高分辨率才能准确回答的样本。
  Benchmark（9 个）：ChartQA (test)、OCRBench (test)、DocVQA (val)、MME (test)、MMVet (test)、RealWorldQA (test)、POPE (test)、MathVista (testmini)、MathVerse (testmini)。
  评价指标：LLM-as-judge (GPT-4o) 进行 binary correctness 判断（1=正确，0=错误），format reward 检查推理/回答/工具调用格式合规性。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址：https://github.com/AdaptVision/AdaptVision
  
  算法 pipeline 伪代码：
  ```
  # === AdaptVision 推理流程（两阶段 coarse-to-fine） ===
  # 输入: 高分辨率图像 I_high (H×W), 问题 q
  # 系统提示 x_sys (含 bounding box tool 定义)

  # Phase 1: 低分辨率编码与首轮推理
  I_low = resize(I_high, scale=1/4)           # 1/4 分辨率
  V_low = VisualEncoder(I_low)                 # ViT encode → n_low 个 visual tokens, n_low ≈ 0.25×n_full
  V_low = Projector(V_low)                     # 对齐到 LLM embedding 维度 d
  x = concat([x_sys, V_low, q])               # 拼接输入序列

  # 首轮生成 (policy π_θ 自回归采样)
  o_1:N = autoregressive_sample(LLM, x, temperature=1.0 during training / 0 during inference)

  # 判断响应类型
  if "<tool call>" in o_1:N:
      # Phase 2: Tool call → 裁剪高分辨率区域
      bbox = parse_bbox(o_1:N)                 # 提取 [x1, y1, x2, y2] in 绝对像素坐标
      I_crop = crop(I_high, bbox)              # 从原始高分辨率图像裁剪
      V_crop = VisualEncoder(I_crop)            # ViT encode → n_crop 个 visual tokens
      V_crop = Projector(V_crop)
      # 拼接裁剪区域 tokens 继续推理
      x_ext = concat([x, o_1:T, V_crop])
      o_T+1:N = autoregressive_sample(LLM, x_ext)
      # 总 visual tokens = n_low + n_crop
  else:
      # 直接回答: 总 visual tokens = n_low (仅低分辨率)
      pass

  # 最终答案: o_N 中的 <answer> 标签内容

  # === DTPO 训练流程 ===
  # 每步训练:
  # 1. 从 policy π_θ_old 采样 G=16 个候选响应 {o_i}
  # 2. 计算每个响应的 reward:
  #    R_acc = LLM_judge(pred_answer, ground_truth) ∈ {0,1}
  #    R_form = 0.5 if format合规 else 0
  #    R_bal = -0.1 if (tool_call ∧ correct) or (direct ∧ low_confidence ∧ correct)
  #    R_crop = GPT4o_judge(cropped_region, question) ∈ {0,1}
  #    R_area = clip(r_a/μ_a - 1, 0, 1) if (R_acc=1 ∧ R_crop=1)
  #    R_oc = R_acc + R_form + R_bal
  #    R_tool = R_crop - α·R_area  (α=2)
  #    R = R_oc + R_tool
  # 3. 分别计算 advantage:
  #    A_oc^(i) = (R_oc^(i) - mean({R_oc}))/std({R_oc})
  #    A_tool^(i) = (R_tool^(i) - mean({R_tool}))/std({R_tool})
  # 4. Token-level advantage (Eq.13):
  #    if direct_answer: A_i,t = A_oc^(i) + λ·A_tool^(i)
  #    if tool_call:     A_i,t = A_oc^(i) + λ·A_tool^(i)·I(1≤t≤T_i)
  # 5. Decoupled loss (Eq.12):
  #    L_tool = (1/ΣT_i)· Σ_i Σ_{t=1..T_i} clip_ratio_loss(π_θ, π_θ_old, A_i,t)
  #    L_answer = (1/Σ(N_i-T_i))· Σ_i Σ_{t=T_i+1..N_i} clip_ratio_loss(π_θ, π_θ_old, A_i,t)
  #    L = L_tool + L_answer
  # 6. 用 AdamW (lr=1e-6) 更新 π_θ，80 steps
  ```

  DTPO 相比 GRPO 的核心改进：
  - GRPO: 整个序列用同一个 advantage 归一化 → tool token 被序列长度 N_i 和组数 G 压制，梯度信号不平衡
  - DTPO: tool token 和 answer token 分别在各自组内按 token 数归一化 + 分别计算独立的 advantage → 精确 credit assignment + 平衡优化

## RETAKE: Reducing Temporal and Knowledge Redundancy for Long Video Understanding

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：RETAKE 是一个 training-free 的即插即用方法，包含两个核心模块联合减少长视频中的时间冗余和知识冗余：
  (1) **DPSelect (Dist Peak Select)**：基于帧间 token-averaged cosine distance 计算相邻帧差异，通过 max pooling (window=3) 识别距离峰值帧作为 pivot frames，再按 top-k 补充剩余关键帧。该策略模仿人类视觉对运动峰值的感知机制，压缩比为 α_dp。
  (2) **PivotKV**：在 chunked prefilling 过程中，对每个 chunk 的 KV cache 进行压缩。对非 pivot 帧，基于层内 self-attention 权重（对 key 维度求和取 head-mean）计算 token 重要性分数。Pivot frames 的 token 被强制保留（通过将重要性分数加 ∞）。最终按 α_kv 比例选择 top-k token 保留到历史 KV cache 中。
  实验比较：(a) 与现有 VideoLLMs 和长视频理解方法（LongVA, LongVILA, Video-XL 等）的性能对比；(b) DPSelect 与 M2SM, A2Summ, MA-LLM 等 keyframe selection 方法的对比；(c) PivotKV 与 FastV, FitPrune, LOOK-M, SparseVLM, PyramidDrop, VL-Cache 等 token compression 方法的对比；(d) DPSelect 与 PivotKV 的消融实验及 trade-off 分析；(e) 细粒度时间感知能力（Needle QA, Action Order, Key Information Retrieval, Temporal Grounding）分析。

- 硬件平台是什么，配置是什么。
  硬件平台：NVIDIA A100 GPU（论文 Table 3 ablation 中提及直接增加帧数超过 A100 显存限制导致 OOM）。论文未明确说明具体 GPU 显存容量和 CPU 配置。

- 模型是什么。数据集和bench分别是什么。
  模型：QWen2VL-7B 和 LLaVA-Video-7B 作为基础 VideoLLM，RETAKE 以即插即用方式扩展。视觉编码器使用原始模型的 VFM（vision foundational model），输入帧较长边 resize 到 448px（QWen2VL）或较短边 336px（LLaVA-Video）。采样策略：2FPS 密集采样，最大 2048 帧（QWen2VL）或 1024 帧（LLaVA-Video）；通过调整压缩比确保 context length 不超过 32K。
  Benchmarks：VideoMME（900 视频/2700 MCQA，短/中/长三类）、MLVU（3 分钟-2 小时视频，9 任务）、LongVideoBench（3763 视频/6678 MCQA，最长 1 小时）、LVBench（平均 4101 秒/视频，1549 MCQA，6 任务）。所有数据集均为人工标注。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源代码：https://github.com/SCZwangxiao/video-ReTaKe
  完整算法 pipeline（对应论文 Algorithm 1）：
  ```
  Input: 视频帧 F ∈ R^{T×3}（T 帧），prompt embeddings P ∈ R^{L×d}，
         chunk size τ，DPSelect 压缩比 α_dp ∈ (0,1]，PivotKV 压缩比 α_kv ∈ (0,1]
  Output: 生成文本 O

  Step 1 — DPSelect 帧压缩:
    M = VE(F)                    # M ∈ R^{T×N×d}，视觉编码
    M_hat, S = DPSelect(M)       # M_hat ∈ R^{α_dp·T·N×d}，S ∈ {0,1}^{α_dp·T·N}
    H = VTC(P, M_hat)            # 视觉-文本拼接，H ∈ R^{(α_dp·T·N+L)×d}

    DPSelect 内部:
      d_i = (1/N) Σ_j (1 - cos(M[i,:,j], M[i+1,:,j]))    # i ∈ [0,T-2]
      P = {i | d_i 是局部极大值 (max pooling window=3)}     # pivot frames
      K = P ∪ ArgTopK(d[非P], k=α_dp·T - |P|)            # 关键帧时间戳
      M_hat = Flatten(M[K, :, :])
      S = token 是否来自 P 中 pivot frame 的 binary mask

  Step 2 — Chunked Prefilling:
    将 H 划分为 chunks: H_1, H_2, ..., H_{α_dp·T/τ+1}
    KV = []  # 初始化空 KV cache

  Step 3 — 逐 chunk 处理 & PivotKV 压缩:
    for each chunk H_i:
      KV_i = LLM(H_i, KV)       # chunk 的 KV cache
      if H_i 是视频 chunk:
        # PivotKV 压缩
        A = Softmax(Q·K_i^T / √d_h)              # A ∈ R^{h×l_q×l_q}
        a_bar = Σ_{j=1}^{l_q} (1/h) Σ_{k=1}^h A_{k,:,j}   # ∈ R^{l_q}
        s = S[iτN : (i+1)τN]                      # pivot mask for chunk
        a_bar = a_bar + s · ∞                      # 强制保留 pivot tokens
        I = ArgTopK(a_bar, k=α_kv·l_q)
        K_hat_i = K_i[:, I, :], V_hat_i = V_i[:, I, :]
        KV = Concat(KV, K_hat_i, V_hat_i)
      else:  # 文本 chunk 不压缩
        KV = Concat(KV, KV_i)

  Step 4 — Decoding:
    O = LLM(Q, KV)              # 标准自回归解码
  ```

  效率优化：使用额外的 CUDA stream 将第 l 层的 PivotKV 压缩与第 l+1 层的 prefilling 重叠执行（见图 4），将 TTFT 开销从 +28%/62% 降低到 +8%/11%。TPOT 降低约 20%，FLOPs 降低 9-18%。

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：AdaRETAKE 是一个 training-free 的自适应视频冗余减少方法，通过动态分配视频帧（时间维度）和 LLM 层（模型维度）上的压缩比例来实现高效视频 token 压缩。包含三个模块：(1) Temporal-adaptive Allocation —— 将长视频等分为 chunk（chunk 内包含 τ 帧），计算每个 chunk 内相邻帧之间的余弦相似度距离 d_i，根据各 chunk 的平均距离按比例分配每个 chunk 的压缩比 α_i；(2) Layer-adaptive Allocation —— 在每个 chunk prefill 时，利用当前层的 video-prompt attention scores 计算每个视频 token 的重要性分数 a_i^(l)，基于全局 Top-K 阈值确定每层的显著性 token 数量 s_i^(l)，然后按比例分配层间压缩比 α_i^(l) = w_i^(l) × α_i（含最小权重 ε=0.01 的数值稳定化重归一化）；(3) Token Compression —— 在前两个模块确定的压缩比下，对每个 chunk 的 KV cache 进行压缩，仅保留 attention 分数最高的 Top-K 个 visual token，逐层更新 KV cache。理论分析证明该层间分配策略能近似最小化 L1 压缩损失的上界（基于 submodular function 的 greedy (1-1/e) 近似保证）。
  实验比较：(a) 与 SOTA MLLM（GLM-4V-Plus, GPT-4o, Gemini-1.5-Pro, VITA-1.5, mPLUG-Owl3, NVILA, ByteVideoLLM, TPO, VideoLLaMA3, LLaVA-Video, Qwen2-VL, Qwen2.5-VL, LLaVA-OneVision, Oryx-1.5, Aria, InternVL2.5）在 VideoMME (Long/Overall)、MLVU (dev)、LongVideoBench (val)、LVBench (val) 上的 accuracy 对比；(b) 与 token 压缩方法（FastV, FitPrune, LOOK-M, SparseVLM, PyramidDrop, VL-Cache）在 VideoMME Long、MLVU val、LVBench val 上的 accuracy 对比；(c) 消融实验（QWen2VL-7B）—— token compression 有无、scaling up frames、layer-wise allocation 有无、temporal allocation 有无、scaling up context length、scaling up max frames；(d) 细粒度感知能力消融——MLVU（Needle QA, Action Order, Action Count）和 LVBench（Key Information Retrieval, Temporal Grounding）；(e) 泛化性——集成到 LLaVA-Video-7B、QWen2VL-7B、QWen2.5VL-7B、QWen2.5VL-72B 多种 MLLM。

- 硬件平台是什么，配置是什么。
  论文未明确说明 GPU 型号。论文提到以 2 fps 采样视频，7B 模型最多 2048 帧、72B 模型最多 1024 帧，最大 context length C_max 为 16K（主实验）/ 1K（消融实验）。推理评估使用 LMMs-Eval 框架。论文引用 KV cache 占用 GPU 内存最多（Hooper et al., 2024）作为设计动机。

- 模型是什么。数据集和bench分别是什么。
  模型：LLaVA-Video-7B (Zhang et al., 2024e)、QWen2-VL-7B (Wang et al., 2024a)、QWen2.5-VL-7B、QWen2.5-VL-72B。MLLM 视觉编码器 + LLM decoder 架构，LLM 层数 L 在模型间不同。
  数据集与 Benchmark：VideoMME（900 个视频，256 小时，2700 个多项选择题，30 个子领域，含 Short/Medium/Long 三个时长子集）；MLVU（视频时长 3 分钟到 2 小时，9 个评估任务包括 topic reasoning, anomaly recognition, video summarization, plot QA）；LongVideoBench（3763 个视频，最长 1 小时，6678 个多项选择题，17 个类别，聚焦 referring reasoning）；LVBench（平均视频时长 4101 秒，1549 个多项选择题，覆盖实体识别、事件理解、关键信息检索、时间定位和推理任务）。
  评价指标：Accuracy（多项选择题正确率）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址：https://github.com/SCZwangxiao/video-FlexReduc.git
  
  算法 pipeline 伪代码：
  ```
  # === 参数定义 ===
  # T: 总帧数, N: 每帧 token 数, τ: 每 chunk 帧数, S: prompt 长度
  # L: LLM 层数, C_max: 最大 context length, d: hidden dimension
  # h: attention heads 数量

  # Step 1: 视频编码
  M = VisualEncoder(video_frames)        # shape: (T, N, d)
  P = WordEmbedding(text_prompt)         # shape: (S, d)
  M_chunks = split(M, chunk_size=τ)     # [M_1, M_2, ..., M_{T/τ}], each (τ, N, d)

  # Step 2: Temporal-adaptive Allocation
  for i in 1..T/τ:
      # 计算 chunk i 内相邻帧的余弦距离
      d_i = zeros(τ-1)
      for t in 1..τ-1:
          sim_sum = 0
          for j in 1..N:
              sim_sum += CosineSim(M_i[t,j], M_i[t+1,j])
          d_i[t] = 1 - sim_sum / N
      d_bar_i = mean(d_i)  # chunk i 的平均帧间距离

  # 按距离比例分配压缩比
  total_budget = (C_max - S) / (T * N)
  for i in 1..T/τ:
      α_i = total_budget * d_bar_i / sum(d_bar_1 ... d_bar_{T/τ})
  # α_i: chunk i 的压缩比（压缩后/压缩前 token 数）

  # Step 3: Layer-adaptive Allocation + Token Compression（逐 chunk 处理）
  for i in 1..T/τ:
      # Chunked Prefilling（等价于标准 prefill 的 autoregressive 行为）
      for l in 1..L:
          # 常规 attention 计算
          Q_i^(l) = X_i^(l) @ W_Q^(l)      # 含 prompt tokens
          K_i^(l) = X_i^(l) @ W_K^(l)
          V_i^(l) = X_i^(l) @ W_V^(l)
          A_i^(l) = Softmax(Q_i^(l) @ K_i^(l)^T / sqrt(d_head))  # (h, S+τN, S+τN)

          # Layer-adaptive: 计算 video-prompt 显著性
          # 提取 video token 对 prompt token 的 attention
          A_video_prompt = A_i^(l)[:, :S, S:]  # (h, S, τN)
          a_i^(l) = sum_{j=1..S} (1/h) * sum_{heads} A_video_prompt[:, j, :]  # (τN,)

      # 全局阈值 K
      a_concat = concat([a_i^(1), a_i^(2), ..., a_i^(L)])  # (τN × L,)
      a_hat = Kth_largest(a_concat, K=α_i × τN × L)

      # 计算每层显著性 token 数
      for l in 1..L:
          s_i^(l) = sum_{j=1..τN} 1(a_i^(l)[j] > a_hat)

      # 计算每层压缩比权重
      for l in 1..L:
          w_i^(l) = s_i^(l) / sum_{k=1..L} s_i^(k)
          # 数值稳定化（ε=0.01）
          w_hat_i^(l) = max(w_i^(l) - ε, 0) / sum_k(max(w_i^(k) - ε, 0)) × (1 - Lε) + ε

      # 最终层压缩比
      for l in 1..L:
          α_i^(l) = w_hat_i^(l) × α_i

      # Token Compression（每层独立）
      for l in 1..L:
          K_keep = α_i^(l) × τN
          I = ArgTopK(a_i^(l), K_keep)   # Top-K 重要性 token 索引
          K^(l) = concat([K^(l), K_i^(l)[:, I]])  # 仅保留重要 token 的 KV cache
          V^(l) = concat([V^(l), V_i^(l)[:, I]])

      # 最后一个 chunk 保留 prompt tokens，其余丢弃
      if i < T/τ:
          drop prompt KV cache entries

  # Step 4: Decoding
  # 在所有 chunk 处理完后，使用压缩后的 KV cache 进行 autoregressive 生成
  answer = LLM.generate(X_L, KV_cache_compressed)
  ```

  关键张量维度：
  - 输入帧: T ∈ [128, 2048]（7B: max 2048, 72B: max 1024），fps=2
  - 每帧 token 数 N: 论文未明确说明每帧 token 数，取决于 MLLM 的 visual encoder/projector
  - Chunk 大小 τ: 10 秒（帧数 = τ × fps）
  - Context length C_max: 16K（主实验）/ 1K（消融）
  - LLM hidden dim d: ~3584 (7B) / ~8192 (72B)，取决于具体 MLLM
  - KV cache 内存: 对长序列推理 KV cache 占 GPU 内存最多（如 16K context 下 ~GB 级别）
  - 压缩比 α_i 全局满足 Σ α_i = (C_max - S) / (TN)，将 T×N 个 video token 压缩到 C_max - S 长度
  - ε=0.01: 最小层压缩权重，防止某层完全不保留 token 导致数值不稳定

  理论保证（Theorem 4.1 / A.4-A.5）：
  - Token 压缩损失的 L1 上界: ε^L = 2C^(L) - 2C^(L) ∏_{l=1}^{L} Σ_i I_i^(l) A_i^(l)
  - 通过选择全局 Top-K attention score 的 token 保留，可达到 (1-1/e) 近似的 submodular 最优
  - 贪婪算法（逐 token 选最大边际增益）在 cardinality constraint K 下达到最优

## CurveStream__Boosting_Streaming_Video_Understanding_in_MLLMs_via_Curvature-Aware_Hierarchical_Visual_Memory_Management

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：CurveStream 是一个 training-free 的 curvature-aware 层级化视觉记忆管理框架，通过特征流形几何曲率评估语义转换强度，动态管理 MLLM 的视觉记忆队列。核心包含两个模块：(1) Curvature-Aware Scorer (CAS) —— 使用冻结的 DINOv2-small 视觉编码器提取每帧的全局特征表示 F_t ∈ R^D 并 L2 归一化，计算一阶 Motion Variation M_t = 1 - cos(F_t, F_{t-1}) 和二阶 Geometric Curvature C_t = 1 - cos(d1, d2)，其中 d1 = F_{t-1} - F_{t-2}，d2 = F_t - F_{t-1}，最终 Curvature Score CS_t = M_t + λC_t；(2) Hierarchical Visual Memory Management (HVMM) —— 使用 EMA 在线更新曲率分数的运行均值 μ_t 和方差 σ_t²，构建 K-Sigma 动态双阈值 g1 = μ_t + k1σ_t, g2 = μ_t + k2σ_t (k1 < k2)，将每帧根据 CS_t 自适应分为 Clear Memory（CS_t ≥ g2，保留原始高分辨率）、Blurred Memory（g1 ≤ CS_t < g2，降采样到 224×224）或 Discard（CS_t < g1，丢弃），当 |M_t| > N_max=20 时执行 FIFO 驱逐最旧 token。
  实验比较：(a) 与 Open-source Online MLLMs（Flash-VStream-7B, VideoLLM-online-8B, Dispider-7B, TimeChat-Online-7B, StreamForest-7B）和 Training-free Offline-to-Online Methods（ReKV, HERMES, FreshMem）在 StreamingBench（10 个实时视觉理解子任务）和 OVOBench（6 个实时视觉感知子任务）上的 accuracy 对比；(b) 跨 Base MLLM 的泛化实验（LLaVA-OneVision-7B, Qwen2-VL-7B, Qwen2.5-VL-7B, Qwen3-VL-8B）；(c) 离线 benchmark 实验 —— MVBench（20 子任务）, EgoSchema, VideoMME, FAVOR-Bench；(d) 跨参数规模 scalability 实验 —— Qwen3-VL 系列 4B/8B/32B；(e) 消融实验 —— curvature metric 有效性（vs Uniform Sampling, Cosine Similarity, Optical Flow, Pyramid Optical Flow）、各组件独立/联合贡献（CAS only, HVMM only, CurveStream full）、curvature score weight λ 的 robustness、K-Sigma 双阈值 (k1, k2) 的 sensitivity、Clear Memory 保留比例对 accuracy 和 token 成本的影响。

- 硬件平台是什么，配置是什么。
  所有 benchmark 评估在单张推理 GPU 上独立执行，以充分验证严格受限内存条件下的框架鲁棒性。论文未明确说明 GPU 型号。特征提取前端使用 DINOv2-small 模型获取时序特征的局部几何表示。所有方法统一建立 memory bank 容量上限 N_max=20 frame tokens，严格模拟流视频处理的物理 GPU 内存约束。

- 模型是什么。数据集和bench分别是什么。
  模型（Base MLLMs）：LLaVA-OneVision-7B, Qwen2-VL-7B, Qwen2.5-VL-7B, Qwen3-VL（4B/8B/32B）。
  视觉编码器（CAS 前端）：DINOv2-small（冻结）。
  在线 benchmark：StreamingBench（10 个实时视觉理解子任务：OP/CR/CS/ATP/EU/TR/PR/SU/ACP/CT），OVOBench（6 个实时视觉感知子任务：OCR/ACR/ATR/STU/FPD/OJR）。
  离线 benchmark：MVBench（20 个细粒度子任务，短视频），EgoSchema（自我中心长视频），VideoMME（含 Short/Medium/Long 子集，最长数小时），FAVOR-Bench（微动动力学感知）。
  对比 baseline：Base MLLMs（均匀采样 1fps 或 64 frames），Streaming 方法（Flash-VStream, FreshMem, HERMES, ReKV, StreamForest, Dispider, TimeChat-Online），封闭源 MLLMs（GPT-4o, Gemini 1.5 Pro, Claude 3.5 Sonnet）。
  评价指标：Accuracy (%)，各子任务细分准确率。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址：https://github.com/streamingvideos/CurveStream（论文声明的代码仓库）

  算法 pipeline 伪代码（基于 Algorithm 1）：
  ```
  # === 初始化 ===
  M_0 = []                           # 空视觉记忆队列
  μ_0 = 0, σ_0 = 0                   # 曲率分数瞬态分布参数
  N_max = 20                         # 最大记忆容量
  λ = 0.2                            # 几何曲率惩罚权重
  k1 = 0.0, k2 = 1.0                 # K-Sigma 阈值乘数
  visual_encoder = DINOv2-small      # 冻结的视觉编码器

  # === 在线处理每个输入帧 I_t ===
  for each frame I_t from infinite video stream:
      # Step 1: 提取特征
      F_t = visual_encoder(I_t)      # shape: (D,)，DINOv2-small 输出
      F_t = F_t / ||F_t||_2          # L2 normalization

      if t > 3:
          # Step 2: CAS — 计算曲率分数
          # 一阶 Motion Variation: M_t = 1 - cos(F_t, F_{t-1})
          M_t = 1 - dot(F_t, F_{t-1}) / (||F_t|| * ||F_{t-1}||)

          # 二阶 Geometric Curvature: 特征位移向量的角度偏差
          d1 = F_{t-1} - F_{t-2}     # shape: (D,)
          d2 = F_t - F_{t-1}         # shape: (D,)
          C_t = 1 - dot(d1, d2) / (||d1|| * ||d2||)

          # 最终曲率分数
          CS_t = M_t + λ * C_t       # scalar

          # Step 3: HVMM — 在线更新分布参数 (EMA)
          μ_t = γ * μ_{t-1} + (1 - γ) * CS_t          # γ ∈ (0,1) 历史窗口控制
          σ_t² = γ * σ_{t-1}² + (1 - γ) * (CS_t - μ_t)²

          # Step 4: 计算 K-Sigma 动态双阈值
          g1 = μ_t + k1 * σ_t         # 模糊记忆下界
          g2 = μ_t + k2 * σ_t         # 清晰记忆下界 (k1 < k2)

          # Step 5: 层级状态路由
          if CS_t >= g2 or t == t_q:   # t_q: 查询时刻
              s_t = Clear Memory
              r_t = High               # 保留原始高分辨率（base model native resolution）
          elif g1 <= CS_t < g2:
              s_t = Blurred Memory
              r_t = Low                # 降采样到 224×224
          else:  # CS_t < g1
              s_t = Discard            # 丢弃低信息冗余帧
              # 不存入 memory bank

          # Step 6: 更新记忆队列
          M_t = Update(M_{t-1}, I_t, s_t, r_t)

          # Step 7: FIFO 驱逐（确保常值内存占用）
          if |M_t| > N_max:
              evict oldest tokens from M_t (FIFO)
      else:
          # t ≤ 3: 前 3 帧积累期，直接以 Clear Memory 存入
          M_t = Update(M_{t-1}, I_t, s_t=Clear, r_t=High)

  # === 查询时刻 t_q ===
  # M_tq 中的视觉 tokens 与自然语言查询 Q 拼接，送入 MLLM 生成答案 A
  answer = MLLM(concat([visual_tokens_from_M_tq, text_tokens_of_Q]))
  ```

  关键张量维度：
  - 输入帧 I_t: H×W×3，分辨率取决于 base MLLM 的动态高分辨率策略
  - Feature dim D: DINOv2-small 输出维度（~384 或 ~768 取决于变体）
  - 记忆队列 M_t: 最大容量 N_max = 20 frames
  - Clear Memory 帧保留原始分辨率（base MLLM native）
  - Blurred Memory 帧统一降采样到 224×224
  - λ = 0.2, k1 = 0.0, k2 = 1.0, γ 论文未明确给出（EMA momentum）
  - 曲率分数 CS_t: scalar ∈ [0, 2]（M_t ∈ [0,1], C_t ∈ [0,1], λ=0.2）

  几何理论解释（Appendix C）：
  - C_t 严格等价于单位切向量变化平方的一半：C_t = 1/2 ||T2 - T1||²
  - 恒速运动（如平滑相机平移）: T1 ≈ T2 → C_t ≈ 0 → 曲率惩罚自然抑制物理运动噪声
  - 语义突变（如镜头切换/新实体进入）: T2 投射到近乎正交子空间 → C_t 急剧增大 → 明确的曲率尖峰

## Accelerating_Vision_Transformers_with_Adaptive_Patch_Sizes

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：Adaptive Patch Transformer (APT) 是一种基于内容感知的自适应 patch 划分方法，在上游数据预处理阶段而非模型前向过程中减少 token 数量。核心设计：(1) 多尺度熵计算 —— 以四叉树层级方式计算图像粗粒度 patch（如 64×64、32×32）的像素熵 H(P) = -∑p_i log₂ p_i，熵值低于阈值的区域分配大 patch，高于阈值的继续细分直至最小 patch（p×p）；(2) Patch Aggregation —— 大 patch 同时被 resize 到 p×p 和拆分为 p×p 子 patch 经原始 patch embedding E 编码后用 Conv2d 级联降采样聚合，两路通过 zero-initialized MLP（受 ControlNet 启发）融合为最终 embedding，保证初始化时性能不退化；(3) Sequence Packing —— 不同图像产生不同数量的 token，通过拼接为单一序列 + block-diagonal attention mask 批量处理，兼容 FlashAttention/xFormers；(4) 位置编码插值 —— 从原始 (H/p)×(W/p) 网格插值到大 patch 对应的 (H/sp)×(W/sp) 网格。
  实验比较：(a) 与输入级 baseline（Random masking / Resizing-only）在全微调和 1-epoch 微调下的 ImageNet Top-1 Accuracy 对比（ViT-B/L/H，多分辨率）；(b) 与层级 token 合并方法（EViT, ToMe, PPT, DTEM，含 FlashAttention 改进版）的 Accuracy vs Throughput trade-off（ViT-L/14@224, ViT-H/14@336）；(c) 下游任务 —— Visual QA（LLaVA-1.5-7B/13B，VQA-v2/GQA/SQA^I/VQA^T/POPE/MME/MMB/MMB^C/MMV）、Object Detection（EVA-02-B/L @1536×1536，COCO mAP/AP50）、Semantic Segmentation（EVA-02-L @512/640，ADE20K aAcc/mIoU）；(d) 消融 —— APT overhead（τ=-1 无压缩时约 10% 开销）、Zero-initialization vs Residual/NonZero/Resizing、熵阈值 τ 的 speed-accuracy trade-off、不同 scorer（Entropy/Laplacian/Upsampling）对比。

- 硬件平台是什么，配置是什么。
  ImageNet 实验：8× NVIDIA A100（单节点），推理吞吐在单 GPU 测量。Object Detection / Segmentation / VQA 实验：8× NVIDIA RTX A6000（单节点）。数据加载时在 CPU 多核上并行计算熵，与 GPU 模型计算重叠，无额外开销。

- 模型是什么。数据集和bench分别是什么。
  模型：ViT-B/16 (86M), ViT-L/14 (304M), ViT-H/14 (632M) —— 均使用 MAE 预训练权重（timm 库）；LLaVA-1.5-7B/13B（VQA，ViT-L/14 视觉编码器）；EVA-02-B/L（Object Detection + Semantic Segmentation，window attention 架构）。
  数据集：ImageNet-1K（分类）、COCO（检测）、ADE20K（分割）、VQA-v2/GQA/ScienceQA-IMG/TextVQA/POPE/MME/MMBench/MMBench-CN/MM-Vet（VQA）。
  Metric：Top-1 Accuracy（分类）、Img/s（吞吐）、GFLOPS、Wall-clock Time、Speedup；mAP/AP50（检测）；aAcc/mIoU（分割）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  项目页面：https://rccchoudhury.github.io/apt/（论文未明确给出 GitHub 代码仓库链接，项目页面可能包含代码）。实现基于 timm (PyTorch Image Models) 和 FlashAttention/xFormers。

  算法 pipeline 伪代码：
  ```
  # === 输入图像: H × W × C ===
  # 参数: p=16 (base patch size), S=3 (scale数), τ₁=5.75, τ₂=4.0 (熵阈值)

  # Step 1: 多尺度熵计算与 patch 分配
  def assign_patches(image, scales=[64, 32, 16], thresholds=[τ₆₄, τ₃₂], min_size=16):
      patches = []  # list of (bbox, scale_idx)
      queue = [(0, 0, H, W, 0)]  # (y, x, h, w, scale_idx)

      while queue:
          y, x, h, w, scale_idx = queue.pop(0)
          crop = image[y:y+h, x:x+w]

          if scale_idx == len(scales) - 1:  # 最小 scale，强制保留
              patches.append((y, x, h, w, min_size))
              continue

          # 计算该区域像素熵
          hist = histogram(crop, bins=256)
          p = hist / hist.sum()
          H = -sum(p_i * log2(p_i) for p_i in p if p_i > 0)

          current_size = scales[scale_idx]
          if H < thresholds[scale_idx]:  # 低熵 → 使用当前大 patch
              patches.append((y, x, h, w, current_size))
          else:  # 高熵 → 拆分为 4 个子 patch
              hh, hw = h // 2, w // 2
              for dy, dx in [(0,0), (0,hw), (hh,0), (hh,hw)]:
                  queue.append((y+dy, x+dx, hh, hw, scale_idx+1))

      return patches  # list of (y, x, h, w, assigned_size)

  # Step 2: Patch Aggregation（对每个 patch P_i, size = s_i × s_i = 2^k p × 2^k p）
  def embed_patch(patch_Pi, k):  # k: scale index (0=16×16, 1=32×32, 2=64×64)
      # 路径 a: 子 patch 嵌入 + 卷积聚合
      sub_patches = split_into(patch_Pi, p, p)  # 拆为 2^k × 2^k 个子 patch
      sub_embeddings = [E(sub_patch) for sub_patch in sub_patches]
      # reshape: (2^k, 2^k, d_embed) → Conv2d^(k) 降采样 k 次 → (1, 1, d_embed)
      feat_map = stack(sub_embeddings).reshape(2^k, 2^k, d_embed)
      for _ in range(k):
          feat_map = Conv2d_3x3_stride2(feat_map)
      emb_sub = feat_map.flatten()  # (d_embed,)

      # 路径 b: resize + 嵌入
      resized = Resize(patch_Pi, (p, p))  # resize 到 p×p
      emb_resized = E(resized)  # (d_embed,)

      # 融合（ZeroMLP 初始化为零权重矩阵）
      emb_final = ZeroMLP(emb_sub) + emb_resized  # ZeroMLP 初始输出为 0
      return emb_final  # (d_embed,)

  # Step 3: Sequence Packing（batch 内变长序列）
  # 对一个 batch 的 B 张图像，各自产生 N_i 个 token
  def pack_sequences(token_seqs, pos_encodings):
      # token_seqs: [tokens_1, tokens_2, ..., tokens_B], tokens_i shape (N_i, d_embed)
      # 拼接所有 token
      packed = concat(token_seqs, dim=0)  # (ΣN_i, d_embed)

      # 构建 block-diagonal attention mask
      # mask[i,j] = 0 if token i, j 属于同一图像，否则 = -inf
      total_len = sum(N_i for N_i in seq_lengths)
      mask = zeros(total_len, total_len)
      offset = 0
      for N_i in seq_lengths:
          mask[offset:offset+N_i, offset:offset+N_i] = 0  # 可互相 attend
          offset += N_i
      mask[mask == 0以外的位置] = -inf

      return packed, mask

  # 位置编码插值（大 patch 的位置编码从细粒度网格插值）
  def interpolate_pos_embed(pos_embed_HW, target_grid_h, target_grid_w):
      # pos_embed_HW: (H/p, W/p, d_embed)
      # target: (H/(sp), W/(sp), d_embed)
      return bilinear_interpolate(pos_embed_HW, (target_grid_h, target_grid_w))
  ```

  关键张量维度：
  - 输入图像: H×W×3, 如 224×224, 336×336, 384×384, 1536×1536
  - Base patch size p=16 (ViT-B/EVA-02) 或 p=14 (ViT-L/H)
  - Token embedding d_embed: 768 (ViT-B), 1024 (ViT-L), 1280 (ViT-H)
  - 最大 scale S=3, patch 大小: 16/32/64 (ViT-B) 或 14/28/56 (ViT-L/H)
  - τ 典型值: τ_32=5.75, τ_64=4.0 (分类/VQA/分割), τ 更低 (检测: τ_128=0.3, τ_64=2.0, τ_32=2.0)
  - Token 压缩比: 典型 ~14% (224分辨率) ~ ~30% (336/高分辨率)

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：ADMIRE 是一个training-free的即插即用方法，基于预训练LVLM的第一层attention权重对多图像进行重要性评分，动态调整各图像分辨率。包含三个模块：(1) TIE (Text-guided Image Scorer) —— 利用LLM第一层attention map的文本token注意力计算每张图像的重要性分数，将图像分为very important/important/less important/not important四类；(2) KIE (Key Image Resolution Enhancer) —— 对Top-k张very important图像进行分辨率上采样（最大pixel数放大N倍）；(3) DVD (Dynamic Visual token Dropper) —— 对less important图像按attention score保留50% visual tokens，对not important图像直接丢弃。
  实验比较：(a) 与OCR-free SOTA模型（LayoutLMv3, DocFormerv2, GPT4(v), LongVA-7B, Idefics3-8B, LLaVA-next-interleave-7B, DocOwl2-8B）对比MP-DocVQA/DUDE/NewsVideoVQA/SlideVQA四个benchmark上的ANLS；(b) 性能vs效率trade-off —— 比较ADMIRE vs All-XN vs Random-Top5在不同上采样倍数(2,4,6)下的ANLS/Total Tokens/FTL/s；(c) 消融实验 —— TIE vs Random选择，KIE+DVD vs Vanilla/All/Random；(d) KIE中enhanced images数量的影响(k=1,3,5,7,10)；(e) 不同图像数量区间的泛化性；(f) SFT后的兼容性；(g) 真实工业场景PRQA数据集上的case study。

- 硬件平台是什么，配置是什么。
  论文未明确说明硬件平台和GPU型号。论文提到使用InternVL2-8B和Qwen2VL-7B模型，推理时测量FTL/s（first token latency per second）和visual token数量。

- 模型是什么。数据集和bench分别是什么。
  模型：InternVL2-8B [5]、Qwen2VL-7B [24]（基础分辨率448×448）
  数据集：MP-DocVQA（工业文档，36k训练/5k验证，1-40页），DUDE（多领域文档，24k训练/5k验证，1-50页），NewsVideoVQA（新闻视频，8k训练/0.7k验证，3-41帧），SlideVQA（幻灯片，10k训练/1.6k验证，15-20页），PRQA（中国体检报告多页QA，1303对，未公开）
  评价指标：ANLS (Average Normalized Levenshtein Similarity)、平均Visual Tokens数量、平均FTL/s

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  代码将开源于 https://github.com/Alipay-Med/admire.git（论文发表时尚未公开，noncommercial use）。
  
  算法pipeline伪代码：
  ```
  # Input: M images {I_i}, text query T
  # Step 1: Encode images with ViT → visual tokens {V_i}, tokenize T → {t_i}
  # Step 2: Concatenate into sequence X, pass through LLM first layer
  X = concat([V_0, ..., V_M, t_0, ..., t_Lt])
  Q = W_Q * (X + PE(X))   # shape: (L, D) → (L, D_head)
  K = W_K * (X + PE(X))
  A = Softmax(Q @ K^T / sqrt(D))   # attention map, shape: (L, L)
  
  # Step 3: TIE - Score each image using text-guided attention
  A_hat = Pool_t(A[p_t])     # pool text-token attention rows → (L_v_total,)
  S_i = Pool_v(A_hat[p_v^i]) # pool per-image visual tokens → scalar
  S = Softmax([S_0, ..., S_M])  # normalized importance scores
  
  # Step 4: Classify images by importance
  p_kie = TopK(S, k)                    # very important (k=3 or 5)
  gamma = mean({S_j | j not in p_kie})  # expected score of remaining
  p_Idvd = {j | S_j <= 0.5 * gamma}     # not important
  p_Vdvd = {j | 0.5*gamma < S_j <= 1.5*gamma}  # less important
  
  # Step 5: KIE - Upscale very important images (max pixels × N)
  for j in p_kie:
      V_j' = ViT(resize(I_j, factor=N))  # N=2,4,6
  
  # Step 6: DVD - Compress/drop less/not important images
  for j in p_Idvd:
      V_j' = []                           # drop entirely
  for j in p_Vdvd:
      idx = argsort(A_hat[p_v^j])[:L_j/2]  # keep top half by attention
      V_j' = V_j[idx]
  
  # Step 7: Feed processed tokens into LLM
  X' = concat([V_0', ..., V_M', t_0, ..., t_Lt])
  answer = LLM(X')
  ```
  
  复杂度：TIE的额外计算开销为O((M*L_v)^2 * D)，仅使用一层attention，开销可控。采用Top5选择时，最大visual token数为5*n*L_v + (M-5)*L_v，远小于全图增强的M*n*L_v。

## Beyond_Accuracy__Evaluating_Grounded_Visual_Evidence_in_Thinking_with_Images__ViEBench

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：ViEBench 是一个 process-verifiable 的 VLM 评估 benchmark，核心创新包含两个组件：(1) Benchmark 数据集 —— 200 张高分辨率多场景图像（retail, urban, industry, daily life），每个样本有 expert-annotated 黄金 BBox 标注视觉证据的精确空间位置，关键视觉证据平均仅占图像面积的 < 0.7%，迫使模型必须执行精确的 zooming/cropping 操作（"Thinking-with-Images"）；任务分为 perception（50%）和 reasoning（50%）两类，其中 reasoning 任务要求模型整合局部视觉线索与先验知识进行多步逻辑推理；(2) Dual-Axis Capability Matrix —— 基于 Intersection-over-Area (IoA) 指标构建二维诊断矩阵，将 Grounding 轴（IoA > 0.5 为 G⁺，否则 G⁻）与 Answer 轴（答案对/错为 A⁺/A⁻）交叉形成四个诊断象限：Valid Grounded Reasoning (G⁺·A⁺)、Ground-Success Answer-Failure (G⁺·A⁻)、Ungrounded Correct Answer (G⁻·A⁺)、Dual Ground-Answer Failure (G⁻·A⁻)，实现过程级细粒度诊断。
  实验比较：(a) Agentic Models (7个) vs End-to-end VLMs (9个) 在 ViEBench perception 和 reasoning 任务上的 accuracy 对比；(b) Agentic Models 的七项细粒度指标（Acc, GS, G⁺·A⁺, G⁺·A⁻, G⁻·A⁺, G⁻·A⁻, TR）在 perception 和 reasoning 子集上的全面过程级审计；(c) 双向 IoA 分析（IoA(B_pred, B_gt) vs IoA(B_gt, B_pred)）揭示不同模型的 crop 策略（expansive coverage vs tight focus）；(d) 对比 ViEBench 与现有 benchmark（V* Bench, HRBench, InfoVQA, VisualProbe）的功能覆盖差异。

- 硬件平台是什么，配置是什么。
  论文未明确说明评估所用 GPU 型号和硬件配置。Agentic models 严格按其官方仓库的评估设置和环境配置执行；End-to-end VLMs 使用 VLMEvalKit 框架进行统一评估。

- 模型是什么。数据集和bench分别是什么。
  模型（Agentic Models）：Pixel Reasoner, Thyme, DeepEyes, Mini-o3, Qwen3-VL-8B-Instruct, Qwen3-VL-235B-A22B-Instruct, Qwen3-VL-32B-Instruct。这些模型具备 tool-use 能力（自主 zooming/cropping），评估时遵循各模型官方仓库配置。
  模型（End-to-end VLMs）：GPT-4o, o3, Qwen2.5-VL-7B-Instruct, InternVL3-8B, LLaVA-CoT, LLaVA-OneVision (standard + SI variant), Keye-VL-1.5-8B, MiMo-VL-7B-RL。这些模型不具备显式 cropping 机制，仅报告 perception 和 reasoning accuracy。
  数据集（ViEBench）：200 个高分辨率多选 QA pairs，来自 Web search + VisualProbe。场景分布：urban (32%), daily life (32%), industrial (19%), retail (17%)。任务分布：perception (50%), reasoning (50%)。关键证据空间稀疏度：perception 任务 gold BBox 平均占图像面积 0.32%，reasoning 任务 0.63%。
  评价指标：Accuracy (Acc.)、Grounded Score (GS)、Valid Grounded Reasoning (G⁺·A⁺)、Ground-Success Answer-Failure (G⁺·A⁻)、Ungrounded Correct Answer (G⁻·A⁺)、Dual Ground-Answer Failure (G⁻·A⁻)、Tool Ratio (TR)。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址：https://github.com/Xuchen-Li/ViEBench（代码将在论文发表后发布）
  标注平台：基于 Label Studio (https://github.com/HumanSignal/label-studio) 定制的 web-based 标注界面

  ViEBench 评估协议伪代码：
  ```
  # === ViEBench 评估流程 ===
  # 输入: 模型 M, ViEBench benchmark 样本集 D = {(I_i, Q_i, A_gt_i, B_gt_i, task_type_i)}

  for each sample (I, Q, A_gt, B_gt, task_type) in D:
      # Step 1: 模型推理
      if M is agentic (with tools):
          # 模型自主决策是否调用 zooming/cropping tool
          response, crop_history = M(I, Q)  # crop_history 包含 B_pred 列表
      else:
          # End-to-end 模型直接推理（无显式 crop 输出）
          response = M(I, Q)
          crop_history = None

      # Step 2: Answer 评估
      A_pred = extract_answer(response)
      answer_correct = (A_pred == A_gt)

      # Step 3: Grounding 评估（仅对 agentic models）
      if crop_history is not None:
          B_pred = crop_history  # 模型生成的 crop 区域
          # 双向 IoA 计算
          IoA_pred_gt = Area(B_pred ∩ B_gt) / Area(B_gt)  # coverage: crop 覆盖了多少证据
          IoA_gt_pred = Area(B_pred ∩ B_gt) / Area(B_pred)  # concentration: crop 中证据占比
          IoA = max(IoA_pred_gt, IoA_gt_pred)
          grounding_success = (IoA > 0.5)

      # Step 4: 象限分配
      if grounding_success and answer_correct:
          quadrant = "G⁺·A⁺ (Valid Grounded Reasoning)"
      elif grounding_success and not answer_correct:
          quadrant = "G⁺·A⁻ (Ground-Success Answer-Failure)"
      elif not grounding_success and answer_correct:
          quadrant = "G⁻·A⁺ (Ungrounded Correct Answer)"
      else:
          quadrant = "G⁻·A⁻ (Dual Ground-Answer Failure)"

  # === 汇总指标 ===
  Acc = count(answer_correct) / |D|
  GS = count(grounding_success) / |D|  # 仅 agentic models
  G⁺A⁺ = count(G⁺·A⁺) / |D|
  G⁺A⁻ = count(G⁺·A⁻) / |D|
  G⁻A⁺ = count(G⁻·A⁺) / |D|
  G⁻A⁻ = count(G⁻·A⁻) / |D|
  TR = count(tool_invoked) / |D|  # Tool Ratio
  ```

  IoA 计算公式：
  ```
  IoA(B_pred, B_gt) = (Area(B_pred) ∩ Area(B_gt)) / Area(B_gt)
  IoA(B_gt, B_pred) = (Area(B_pred) ∩ Area(B_gt)) / Area(B_pred)
  IoA_final = max(IoA(B_pred, B_gt), IoA(B_gt, B_pred))
  # IoA > 0.5 → G⁺ (successful grounding)
  # IoA ≤ 0.5 → G⁻ (failed grounding)
  ```

  关键设计要点：
  - ViEBench 的 IoA 使用 max 而非标准 IoU，同时容忍 precise tight crop（高 IoA(B_gt, B_pred)）和 conservative expansive crop（高 IoA(B_pred, B_gt)）
  - Perception 任务要求模型识别细粒度视觉属性，reasoning 任务要求多步逻辑推理整合视觉线索与先验知识
  - Gold BBox 平均仅占 0.32%-0.63% 图像面积，确保在全局视图下 sub-perceptual，强制 tool-use
  - 标注流程：专业标注员标注 → 资深审查员验证 → 模糊样本精炼或丢弃

## Bridging_Episodes_and_Semantics__A_Novel_Framework_for_Long-Form_Video_Understanding

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：HERMES 是一个受人类认知启发（episodic memory + semantic memory）的长视频理解框架，由两个可插拔模块组成：(1) Episodic COmpressor (ECO) —— 在线滑动处理视频 window-by-window，维护最大容量为 E 的 memory buffer；当 buffer 满时，迭代合并（通过 cosine similarity）最相似的 frame 对，将冗余帧压缩为 episode prototypes，直到 buffer 大小回到 E 以内；(2) Semantics reTRiever (SeTR) —— 以 stride=k 将视频帧分为两组 K（保留帧，N/k 个）和 K̄（被压缩帧，N-N/k 个），计算两组间的 dot-product similarity，将每个 K̄ 帧合并到最相似的 K 帧中，减少 80% 帧数同时保留高语义信息；(3) Episodic Q-Former —— 在 Q-Former 中插入 ECO，对跨窗口 query 也进行 episode-level 聚合；(4) Hierarchical Q-Former —— frame Q-Former（独立增强每帧语义）→ frame-to-sequence adapter → video Q-Former（全局聚合）。整体 pipeline：Video → Window Encoder (ViT-G/14) → ECO + Episodic Q-Former（episodic stream）+ SeTR + Hierarchical Q-Former（semantic stream）→ concat → linear projection → frozen Vicuna-7B LLM → 生成文本答案。
  实验比较：(a) 与 SOTA 在 MovieChat-1k (zero-shot/full-supervised VQA)、LVU/Breakfast/COIN (long-form video classification) 上的 accuracy/score 对比（vs MovieChat, Video-ChatGPT, Video-LLaMA, VideoChat, S5, MA-LMM, VIS4mer, TranS4mer, Movies2Scenes, FACT）；(b) Pilot Study: ECO 和 SeTR 作为 plugin 集成到 MA-LMM、LongVA、LLaVA-OneVision 三种 SOTA 模型，在 MovieChat-1k 和 VideoMME 上的 accuracy/latency/memory 对比；(c) 消融实验: ECO memory update 策略（w/o, Random, FIFO, ECO）、SeTR 压缩方法（w/o, MaxPool, AvgPool, K-Means, SeTR）、Hierarchical Q-Former vs flat variants (fQFormer, vQFormer)；(d) 效率分析: ECO vs ToMe vs MA-LMM 的 accuracy vs inference time trade-off；(e) 超参数 sensitivity: ECO episode 数 (4~32) 和 SeTR keep ratio (0.05~0.5) 对 accuracy 的影响（MovieChat-1k zero-shot + Breakfast full-supervised）。

- 硬件平台是什么，配置是什么。
  训练：8× NVIDIA V100 GPU (32GB VRAM each)。推理：单卡 V100 GPU (32GB)。MovieChat-1k 测试集 (170 videos, 100 frames each) 推理耗时 13 分钟（~22 FPS）。MovieChat-1k 训练耗时 < 12 分钟 (8× 32GB V100)。

- 模型是什么。数据集和bench分别是什么。
  模型：Visual Encoder 使用 ViT-G/14 (EVA-CLIP, 冻结)；Episodic Q-Former 初始化自 InstructBLIP 权重；LLM 使用 Vicuna-7B (冻结)；线性投影层和 Q-Former 的 ECO 插入部分可训练。
  数据集：LVU（电影内容，7 个子任务: Relation/Speak/Scene/Director/Genre/Writer/Year，Top-1 Accuracy）、Breakfast（教学视频，过程理解，Top-1 Accuracy）、COIN（教学视频，更广泛的程序性活动，Top-1 Accuracy）、MovieChat-1k（长视频 QA，Global + Breakpoint 模式，GPT-3.5-turbo 评估 accuracy + score）、VideoMME（多模态 LLM 视频评估，w/o subtitles）。
  Baseline 模型：MovieChat[32]、Video-ChatGPT[22]、Video-LLaMA[49]、VideoChat[20]、FACT[21]、Obj. Transformer[43]、VIS4mer[15]、TranS4mer[16]、S5[41]、Movies2Scenes[5]、MA-LMM[14]、LongVA[51]、LLaVA-OneVision[19]。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址：https://joslefaure.github.io/assets/html/hermes.html（project page + code）
  
  算法 pipeline 伪代码（ECO + SeTR 完整流程）：
  ```
  # === 参数 ===
  # N: 采样帧数（默认100）, w: window size（默认10）
  # E: 最大 episode 数（默认20）, k: SeTR stride, keep_ratio: 0.2
  # B: batch size, T: visual tokens per frame, C: channels

  # === Step 1: 视频帧采样与 Window Encoding ===
  V = sample_frames(video, N)           # 从原始视频采样 N 帧
  windows = split(V, w)                 # 分为 N/w 个 window
  for k in 1..N/w:
      W_k = ViT_G_14(windows[k])        # ViT-G/14 encode, shape: (B, w, T, C)

  # === Step 2: ECO 在线压缩（每个 window 到达时） ===
  M = []                                # episode memory buffer
  for k in 1..N/w:
      if len(M) + w <= E:
          M = concat(M, W_k)            # buffer 有空间，直接追加
      else:
          A = concat(M, W_k)            # 临时拼接, shape: (B, ||A||, T, C)
          while ||A|| > E:
              # 找最相似的 frame 对
              (i*, j*) = argmax_{i≠j} cosine_sim(A_i, A_j)
              A_i* = (A_i* + A_j*) / 2  # 合并最相似的两帧
              A = remove(A, A_j*)       # 删除被合并的帧
          M = A                         # 更新 buffer

  # === Step 3: Episodic Q-Former ===
  Q_0 = learned_queries                 # shape: (B, q, C')
  Q_0 = SelfAttention(Q_0)
  Q = CrossAttention(Q_0, M)            # cross-attend to visual episodes
  Q = ECO_q(Q)                          # 对 queries 也进行 episode-level 聚合
  # 输出: Q ∈ R^{B × q × C'}

  # === Step 4: SeTR 语义检索 ===
  F = concat(W_1, ..., W_{N/w})         # 全部 window features, shape: (B, N, T, C)
  F = normalize(F)                      # 归一化
  # 以 stride k 分组
  K_indices = [0, k, 2k, ...]           # 保留组
  K_bar_indices = rest                  # 被压缩组
  F_K = F[:, K_indices, :, :]           # N/k 帧
  F_Kbar = F[:, K_bar_indices, :, :]    # N - N/k 帧
  # 计算相似度并合并
  for each frame f in F_Kbar:
      sim = dot_product(f, F_K)         # similarity scores
      j* = argmax(sim)
      F_K[j*] = (F_K[j*] + f) / 2       # 合并到最相似的保留帧
  F_prime = F_K                         # shape: (B, N/k, T, C)

  # === Step 5: Hierarchical Q-Former ===
  # Frame Q-Former: 独立增强每帧语义
  F_fq = fQFormer(F_prime)              # (B, N/k, q_f, C')
  # Frame-to-Sequence Adapter
  F_seq = Linear(F_fq)                  # (B, N/k * q_f, C')
  # Video Q-Former: 全局聚合
  Q_sem = vQFormer(F_seq)               # (B, q, C')

  # === Step 6: 合并表示 → LLM 生成 ===
  U = Linear(concat([Q, Q_sem]))        # (B, 2q, C') → project to LLM dim
  answer = Vicuna_7B.generate(U, instruction)
  ```
  
  关键张量维度：
  - 输入帧: N=100 (MovieChat-1k/LVU/COIN/Breakfast), N=40 (MovieChat-1k Breakpoint)
  - Window size w: 10 (default)
  - ViT-G/14 输出: T visual tokens, C channels per frame
  - ECO memory buffer max size E: 20 episodes (MovieChat-1k/LVU/COIN/Breakfast), 10 (Breakpoint)
  - SeTR stride k = 1/keep_ratio: keep_ratio=0.2 → 保留 20% frames = 20 帧
  - LLM: Vicuna-7B, hidden dim ~4096
  
  ECO 核心操作（Algorithm 1 精确伪代码）：
  ```
  A = M ⊕ W_k                           # 拼接 buffer 和新 window
  while ||A|| > E:
      (i*, j*) = argmax_{i≠j} (A_i · A_j) / (||A_i|| · ||A_j||)  # cosine similarity
      A_i* = (A_i* + A_j*) / 2          # 元素级平均
      A = A \ A_j*                      # 删除 A_j*
  M = A
  ```

  训练说明：HERMES 利用预训练的 ViT-G/14（冻结）和 Vicuna-7B（冻结），Q-Former 初始化为 InstructBLIP 权重。对于 zero-shot 设置，Q-Former 和 adapter 不训练（training-free 的 ECO 和 SeTR 已经有效）；对于 fully-supervised 设置，仅微调 Q-Former 和 adapter，MovieChat-1k 仅需 1 epoch 训练（< 12 分钟 on 8× V100）。Plugin 实验：ECO 和 SeTR 在推理时插入 target 模型（MA-LMM, LongVA, LLaVA-OneVision），零额外训练。

## Adaptive_Keyframe_Sampling_for_Long_Video_Understanding

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：Adaptive Keyframe Sampling (AKS) 是一个 plug-and-play 的 keyframe 选择模块，放在 MLLM visual encoder 之前，目标是在固定数量的视频 token 约束下最大化 keyframe 的信息量。核心设计：(1) Relevance 计算 —— 使用 VL 模型（默认 BLIP ITM）计算每个候选帧 $\mathbf{F}_t$ 与 prompt $\mathbf{Q}$ 的匹配分数 $s(\mathbf{Q}, \mathbf{F}_t)$；(2) Coverage 估计 —— 基于 Ripley's K-function 的递归分 bin 机制，将时间轴 [0, T) 递归二分为 bin，在每个 bin 内统计 keyframe 数量，通过不均分布惩罚项 $|m_1 - m_2|$ 量化 coverage；(3) ADA（Adaptive Sampling）算法 —— 综合 relevance 和 coverage 的分层优化：在每层递归中，若 $s_{\text{top}} - s_{\text{all}}$ 超过阈值 $s_{\text{thr}}$，则倾向于保留高相关性帧（TOP 模式）；否则将当前 bin 二分为子 bin 并均匀分配 keyframe 数（BIN 模式）。ADA 是 TOP（纯相关性最大化）和 BIN（纯 coverage 保证）的自适应折中。
  实验比较：(a) 将 AKS 应用于三种 baseline MLLMs（Qwen2VL-7B、LLaVA-OV-7B、LLaVA-Video-7B），对比 uniform sampling 基线在 LongVideoBench val 和 VideoMME 上的 QA accuracy；(b) 与 SOTA MLLMs（GPT-4V, GPT-4o, Gemini-1.5-Flash/Pro, VideoLLaVA, MiniCPM-V 2.6, PLLaVA, VILA 等）对比；(c) 诊断实验 —— 不同 sampling 策略（UNI/TOP/BIN/ADA）在 LongVideoBench val 和 VideoMME 上的 accuracy 对比；(d) 消融实验 —— sampling frequency（1/0.5/0.25/0.125/0.1 fps）、VL scorer 选择（BLIP/Sevila/CLIP）、ADA 超参数 L 和 $s_{\text{thr}}$ 的影响；(e) 泛化实验 —— AKS 扩展到 video referring 和 video captioning 任务。

- 硬件平台是什么，配置是什么。
  论文未明确说明 GPU 型号。论文提到以 1 fps 采样候选帧，使用 BLIP ITM 计算 prompt-frame relevance 分数（预计算并存储），MLLM 推理基于 Qwen2VL/LLaVA-OV/LLaVA-Video 的标准推理流程。使用 LMMs-Eval 框架进行评估。

- 模型是什么。数据集和bench分别是什么。
  模型：作为 baseline MLLM 的有 Qwen2-VL-7B（使用 Qwen2-7B LLM）、LLaVA-OneVision-7B（使用 Qwen2-7B LLM）、LLaVA-Video-7B（使用 SigLIP 视觉编码器 + Qwen2-7B LLM，支持最多 64 帧输入）。VL scorer 默认使用 BLIP ITM，可选 Sevila 和 CLIP。
  数据集与 Benchmark：LongVideoBench（3763 个视频，最长 1 小时，6678 个多项选择题，17 个类别）的 val 子集；VideoMME（900 个视频，256 小时，2700 个多项选择题，30 个子领域，含 Short/Medium/Long 三个时长子集）。均不使用视频字幕辅助回答。
  评价指标：Accuracy（多项选择题正确率百分比）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址：https://github.com/ncTimTang/AKS
  
  算法 pipeline 伪代码：
  ```
  # ==== AKS: Adaptive Keyframe Sampling ====
  # 输入: 视频 V ∈ R^{T×W×H×C}（T 帧）, prompt Q, 目标 keyframe 数 M
  # VL scorer: BLIP ITM 或其他 VL 模型

  # Step 1: 候选帧采样与预计算 relevance score
  candidates = sample_frames(V, fps=1)  # 1 fps → 约 T 个候选帧
  matching_scores = []  # 预计算并存储
  for each candidate frame F_t in candidates:
      # 将 prompt Q（text）和帧 F_t（image）送入 VL 模型
      q_emb = VL_text_encoder(Q)       # text embedding
      f_emb = VL_image_encoder(F_t)    # image embedding
      s_t = ITM(q_emb, f_emb)          # image-text matching score, scalar
      matching_scores.append(s_t)
  # matching_scores: list of shape (T,), 与 question 对应

  # Step 2: ADA 递归分层优化 keyframe 选择
  def ADA(matching_scores, level, max_level, s_thr, M):
      # 计算当前 bin 内所有帧的平均分和 Top-M 帧的平均分
      s_all = mean(matching_scores)
      s_top = mean(topk(matching_scores, M))
      margin = s_top - s_all

      if margin >= s_thr or level >= max_level:
          # TOP 模式: 帧间区分度足够 → 直接选 Top-M 高分帧
          return argtopk(matching_scores, M)
      else:
          # BIN 模式: 拆分 bin → 均分配
          mid = len(matching_scores) // 2
          left_scores = matching_scores[:mid]
          right_scores = matching_scores[mid:]
          M_left = M // 2
          M_right = M - M_left
          left_indices = ADA(left_scores, level+1, max_level, s_thr, M_left)
          right_indices = ADA(right_scores, level+1, max_level, s_thr, M_right)
          # 将右半部分的索引偏移回全局索引
          right_indices_global = [idx + mid for idx in right_indices]
          return left_indices + right_indices_global

  # Step 3: 选择 keyframe 并送入 MLLM
  selected_indices = ADA(matching_scores, level=0, max_level=L, s_thr, M)
  # selected_indices: 长度为 M 的整数列表，指示选中帧的全局索引
  keyframes = [candidates[i] for i in selected_indices]

  # Step 4: MLLM 推理
  visual_tokens = [VisualEncoder(frame) for frame in keyframes]
  # visual_tokens: M 组 token，经 Projector 对齐后与 text prompt 拼接
  answer = MLLM(visual_tokens, Q)
  ```

  关键设计细节：
  - Coverage 度量：递归 level-0: 2 个 bin（[0,T/2), [T/2,T)），bin width T/2；level-1: 4 个 bin，bin width T/4；... level-L: 2^L 个 bin。在每层，penalty term = |m_1 - m_2| + |m_3 - m_4| + ...，其中 m_i 是第 i 个 bin 内 keyframe 数量。最大递归深度 L ≤ ⌈log₂ M⌉。
  - ADA 的两个超参数：L（最大递归深度）和 s_thr（区分度阈值）。L 控制 coverage 粒度，s_thr 控制何时放弃 coverage 约束转向 TOP 模式。LongVideoBench 偏好较小的 L 和 s_thr（问题多聚焦于单个时刻），VideoMME 偏好较大的值（问题需要多个时刻的信息）。
  - VL scorer：默认 BLIP ITM（基于 object-level 预训练，对 object 相关问题更敏感），可选 CLIP（基于 generic image-text 预训练，对全局感知问题更好）。预计算所有候选帧的 relevance 分数存储在 matching_scores 列表中。
  - 候选帧采样频率：默认 1 fps；可降低至 0.1 fps 仍保持高于 uniform baseline 的 accuracy（LongVideoBench: 60.1 @ 64 frames, 0.1 fps vs 58.9 uniform）。

  关键张量维度：
  - 视频帧数 T: 可变化，主流实验 32 或 64 个 keyframe 输入 MLLM
  - 候选帧数（1 fps）：约为视频时长（秒），如 600s 视频 ≈ 600 候选帧
  - VL 模型 embedding 维度: BLIP text/image embedding 维度取决于具体 BLIP 变体
  - L (max recursion level): 典型值 3-5（Table 5），覆盖 2³ 到 2⁵ 个 bin
  - s_thr: 典型值 0.2-1.0（Table 5），控制 TOP vs BIN 的切换倾向

  复杂度分析：
  - 预计算 relevance 分数：O(T × (text_enc + img_enc + ITM))，使用 BLIP ITM（轻量模型），远低于 MLLM forward cost
  - ADA 递归：O(T × L)，L 最大为 ⌈log₂ M⌉ ≤ 6（M=64 时），可忽略
  - 总 overhead：主要来自 VL scorer 预计算，相比 MLLM 推理开销小
  - 与 MLLM 无关：AKS 仅改变输入帧，MLLM 本身（Qwen2VL/LLaVA-OV/LLaVA-Video）不做任何修改

## FOCUS__Efficient_Keyframe_Selection_for_Long_Video_Understanding

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是 FOCUS（Frame-Optimistic Confidence Upper-bound Selection），一个训练无关（training-free）、模型无关（model-agnostic）的 keyframe selection 模块，将 query-aware keyframe selection 建模为 multi-armed bandit 中的 Combinatorial Pure-Exploration (CPE) 问题。核心流程：将视频划分为固定长度 clip 作为 bandit arms→ Stage I 并行采样探索所有 arm → Stage II 对 optimistic UCB 最高的 α*m 个 arm 做细化采样 → 基于经验均值选择 top-m arm → 在选中 arm 内通过 nearest-neighbor 插值采样 keyframes。实验比较：(1) 与 uniform sampling 的 QA 准确率对比（GPT-4o、Qwen2-VL-7B、LLaVA-OV-7B、LLaVA-Video-7B）；(2) 与 SOTA keyframe selection（Top-K、AKS、Q-Frame）对比；(3) 效率对比：GPU hours 和 frames seen 占比；(4) α 超参数消融；(5) 消融实验：two-stage vs single-stage、Bernstein confidence radius vs empirical mean、clip length、vision-language encoder 选择；(6) 额外 benchmark：MLVU、VSI-Bench。

- 硬件平台是什么，配置是什么。
  单卡 NVIDIA H100 (80GB) GPU。所有 keyframe selection 方法的 GPU hours 均在该 GPU 上测量。MLLM 推理所用 GPU 论文未明确说明具体型号。

- 模型是什么。数据集和bench分别是什么。
  模型：FOCUS 作为前置 keyframe selection 模块，接在四个 MLLM 之前：GPT-4o (0513)、Qwen2-VL-7B、LLaVA-OV-7B、LLaVA-Video-7B。Frame scoring 使用 BLIP ITM (Li et al., 2022) 计算 cosine similarity 作为 frame-query relevance。也可替换为 CLIP (Radford et al., 2021) 或 SigLIP (Zhai et al., 2023)。
  数据集/Benchmark：LongVideoBench (Wu et al., 2024)、Video-MME (Fu et al., 2025)、MLVU (Zhou et al., 2025)、VSI-Bench (Yang et al., 2025)。评价框架使用 LMMs-Eval (Zhang et al., 2024a)，禁用字幕，zero-shot 评估。Video-MME 按 Short(<2min)/Medium(4-15min)/Long(30-60min) 分类；LongVideoBench 按 Short(<3min)/Medium(3-20min)/Long(>20min) 分类。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源：代码在 https://github.com/NUS-HPC-AI-Lab/FOCUS。
  算法 Pipeline（对应 Algorithm 2——两阶段 Optimistic Confidence Upper-bound Arm Selection）：

  ```
  输入: 视频 V = (x_1, ..., x_T), query q, clip 长度 l, arm 数 M, 目标 arm 数 m,
        探索因子 α, 初始化 pulls q, 细化 pulls z, 目标 frame 数 k

  Stage I: Coarse Exploration
  1. 将 V 划分为 M 个固定长度 l 的 clip，每个 clip 作为一个 bandit arm A_a
  2. 对每个 arm a ∈ {1..M}，并行均匀采样 q 帧，通过 BLIP ITM 计算 reward:
     r_t = cosine_similarity(BLIP.encode_image(x_t), BLIP.encode_text(q))
  3. 更新每个 arm a 的经验均值 μ̂_a 和经验方差 σ̂_a²
  4. 计算 Bernstein confidence radius:
     β_a(n) = sqrt(2 * σ̂_a² * ln(n) / max(1, N_a(n))) + 3 * ln(n) / max(1, N_a(n))
     其中 n = M*q 为总 pulls 数，N_a(n) = q
  5. 计算 optimistic mean: μ̃_a(n) = μ̂_a(n) + β_a(n)

  Stage II: Fine-grained Exploitation
  6. A_coarse = TopM(μ̃, α*m)  // 取 optimistic mean 最高的 α*m 个 arm
  7. 对 a ∈ A_coarse，并行采样 z 帧，更新 μ̂_a(n), σ̂_a², N_a(n)
  8. A_fine = TopM(μ̂, m)  // 取经验均值最高的 m 个 arm（无偏估计）

  Frame Selection within Selected Arms
  9. 每个 arm a ∈ A_fine 分配 k_a = round(k/m) 个 frame slot（均匀分配，调整至总和=k）
  10. 在 arm 内通过 nearest-neighbor 插值所有 frame 的 reward r̂_{a,t}
  11. 构建 per-arm 采样分布 p_a ∝ r̂_{a,t}，不放回采样 k_a 帧
  12. 合并为最终 keyframe 集 K = ∪_{a∈A_fine} K_a，|K| = k

  输出: K
  ```

  关键张量计算：BLIP 视觉编码器对每帧输出 visual embedding e_t ∈ R^d，文本编码器对 query 输出 text embedding e_q ∈ R^d，reward r_t = (e_t · e_q) / (||e_t|| · ||e_q||) ∈ [0,1]。所有 BLIP forward 批处理执行（并行 arm-pull），避免串行 GPU 利用率浪费。

  超参数默认值：clip length l = 16 秒，α = 0.25，q = 论文未明确说明，z = 论文未明确说明。效率：处理 LongVideoBench 上仅需 1.6% 帧的 BLIP forward（vs AKS w/ pre-filtering 的 3.7%），5.5 GPU hours（vs AKS 的 9.3 GPU hours）。

## GroundVTS__Visual_Token_Sampling_in_Multimodal_Large_Language_Models_for_Video_Temporal_Grounding

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：GroundVTS 是一个面向 Video Temporal Grounding (VTG) 的 Vid-LLM 框架，核心创新是 **Visual Token Sampling (VTS)** 模块——在 visual encoder 和 multimodal projector 之后、LLM 输入之前，对 visual token 进行 query-guided 细粒度采样。VTS 包含两个步骤：(1) **Query-Guided Token Scoring** —— 将 visual embeddings V 和 text query embeddings Q 通过可学习投影矩阵 W_v, W_q 映射到低维子空间 D_r，计算温度缩放的点积相似度 w = softmax(V'q'^T / τ)，得到每个 visual token 的 query 相关性权重；(2) **Differentiable Top-K Selection** —— 基于 Gumbel-Softmax 松弛 + Straight-Through Estimator (STE) 实现可微分的 top-K 选择，z_i = softmax((log w_i + g_i)/τ_g)，forward 使用 hard top-K mask，backward 通过连续松弛 z_i 传播梯度。最终 selected tokens 通过 MLP + 重归一化权重获得。采样比例 ρ ∈ (0,1] 控制保留的 visual token 数量 K = ⌈ρ·N_v⌉。保留原始位置编码（仅 mask 掉未选中 token 的位置编码）以维持时间一致性。训练采用三阶段渐进式优化：(Stage 1) VTS Warm-up —— 冻结 LLM，仅训练 VTS 模块；(Stage 2) Joint LoRA Adaptation —— LoRA (rank=8, α=16) 微调 LLM + VTS + Projector，使用 LLaVA-Video-178K；(Stage 3) Grounding Fine-tuning —— 继续 LoRA 微调，使用自建 Grounding-FT 70K 样本。两个模型变体：GroundVTS-Q（基于 Qwen2.5VL-7B, 2 FPS, ρ=0.5, D_r=512）和 GroundVTS-I（基于 InternVL3.5-8B, 16 frames/video, ρ=0.5, D_r=128）。

  实验比较：(a) Moment Retrieval —— Charades-STA 和 ActivityNet-Captions 上对比 LLaVA-OV、TimeChat、VTimeLLM、Momentor、HawkEye、ChatVTG、NumPro、LLaVA-ST 等 SOTA 方法，以及 Qwen2.5VL-7B 和 InternVL3.5-8B 的微调 baseline；(b) QVHighlights —— MR + HD 对比 SeViLA、UniVTG、VTG-LLM、TimeChat、NumPro 等；(c) Out-of-Distribution —— NExT-GQA（零样本 grounded VQA）、DiDeMo（OOD moment retrieval）、LongVideoBench（长视频理解迁移）；(d) Visual Token Density —— ρ 从 0.1 到 1.0 的稳健性分析；(e) 消融实验 —— 训练阶段组合、采样策略（Token-Level vs Frame-Level vs Uniform vs Random）、位置编码有无；(f) 额外消融 —— 参数自由投影 vs 可学习投影、数据集组成；(g) MVBench 通用 VQA 能力保持验证。

- 硬件平台是什么，配置是什么。
  训练使用 GPU，batch_size=2 per GPU, gradient_accumulation=4，优化器 AdamW (β1=0.9, β2=0.999)。具体 GPU 型号论文未明确说明。推理评估使用标准 PyTorch + HuggingFace Transformers。

- 模型是什么。数据集和bench分别是什么。
  模型：GroundVTS-Q（基于 Qwen2.5VL-7B, total 8.32B params, trainable 153.0M）和 GroundVTS-I（基于 InternVL3.5-8B, total 8.56B params, trainable 145.2M）。VTS 模块参数 ~29-35M，Projector ~34-45M，LoRA (rank=8, α=16, dropout=0.05) ~77-79M。
  训练数据集：(1) LLaVA-Video-178K —— 大规模视频多模态数据集（Stage 1 和 Stage 2）；(2) Grounding-FT —— 自建 VTG 指令微调数据集，聚合 Charades-STA、QVHighlights、ActivityNet-Captions 训练集，70K 标注视频-查询对，统一为 ShareGPT instruction-response 格式，含 MR 和 HD 两种任务。
  Benchmarks：(1) Charades-STA (R1@0.3/0.5/0.7 + mIoU)；(2) ActivityNet-Captions (R1@0.3/0.5/0.7 + mIoU)；(3) QVHighlights (MR: R1@0.5/0.7; HD: mAP + Hit@1)；(4) NExT-GQA (mIoU, mIoP, IoU@0.5, IoP@0.5, Acc@GQA)；(5) DiDeMo (R1@0.3/0.5, mIoU)；(6) LongVideoBench (Acc by duration)；(7) MVBench (20 子任务通用 VQA)。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址：https://github.com/Florence365/GroundVTS

  算法 pipeline 伪代码（VTS 核心 + 三阶段训练）：

  ```
  # ==========================================
  # GroundVTS 推理流程 (VTS 核心)
  # ==========================================
  # 输入: 视频 V = {F_t}_{t=1}^T, 文本查询 text_query
  # 超参数: ρ (sampling ratio), τ (temperature), τ_g (Gumbel temp), D_r (hidden dim)

  # 步骤 1: Vision Encoding + Projection
  H_v = VisionEncoder({F_t})  # T frames → N_v visual tokens ∈ R^{N_v × D_v}
  V = Projector(H_v)          # MLP → ∈ R^{N_v × D}, D 对齐 LLM embedding 维度

  # 步骤 2: Text Tokenization
  Q = TextTokenizer(text_query)  # ∈ R^{N_t × D}

  # 步骤 3: VTS - Query-Guided Token Scoring (Eq.2-3)
  V' = W_v @ V              # W_v ∈ R^{D × D_r}, V' ∈ R^{N_v × D_r}
  q' = W_q @ mean(Q, dim=0) # W_q ∈ R^{D × D_r}, q' ∈ R^{D_r}
  w = softmax(V' @ q' / τ)  # ∈ R^{N_v}, query-token relevance weights

  # 步骤 4: VTS - Differentiable Top-K Selection (Eq.4-6)
  K = ceil(ρ * N_v)
  g_i ~ Gumbel(0, 1)
  z_i = exp((log w_i + g_i) / τ_g) / Σ_j exp((log w_j + g_j) / τ_g)  # soft
  I_K = TopK_indices(w, K)
  z_i^hard = 1 if i ∈ I_K else 0                                    # hard (forward)
  \tilde{z}_i = z_i^hard + z_i - stopgrad(z_i)                       # STE

  # 步骤 5: Weighted Token Re-encoding (Eq.7)
  \hat{w}_i = exp(w_i/τ') · \tilde{z}_i / Σ_j exp(w_j/τ') · \tilde{z}_j
  \tilde{v}_i = \hat{w}_i · MLP(v_i)

  # 步骤 6: Position Encoding + LLM Input
  PE_selected = PE_original[I_K]  # 保留 dense sampling 原始位置编码
  input_seq = concat([\tilde{V} + PE_selected, Q])
  answer = LLM.generate(input_seq)

  # ==========================================
  # 三阶段训练流程
  # ==========================================
  # Stage 1: VTS Warm-up (lr=1e-5, 1 epoch, LLaVA-Video-178K)
  freeze(LLM, VisionEncoder, Projector)
  trainable = [W_v, W_q, MLP_vts]  # 仅 VTS

  # Stage 2: Joint LoRA Adaptation (lr=2e-4, 2 epochs, LLaVA-Video-178K)
  unfreeze(Projector)  # VTS + Projector + LoRA(LLM)
  # LoRA: rank=8, α=16, dropout=0.05

  # Stage 3: Grounding Fine-tuning (lr=1e-4, 3 epochs, Grounding-FT 70K)
  # 同 Stage 2 冻结配置，使用 MR + HD instruction-style QA pairs
  ```

  关键张量维度：
  - Visual tokens N_v: QwenVL @ 2 FPS → 动态; InternVL @ 16 frames → 固定
  - Token 投影: V' ∈ R^{N_v × D_r}, q' ∈ R^{D_r}; D_r=512(GroundVTS-Q), D_r=128(GroundVTS-I)
  - VTS 采样: K = ⌈ρ·N_v⌉, ρ=0.5 即保留 50% visual tokens
  - 非均匀 token 分布: 高 query 相关性区域 dense sampling, 低相关性区域 sparse/zero
  - Trainable params: VTS ~29-35M + Projector ~34-45M + LoRA ~77-79M = 145-153M total
  - 训练配置: batch_size=2/GPU, grad_acc=4, AdamW (β1=0.9, β2=0.999)

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

## HiPrune__Training-Free_Visual_Token_Pruning_via_Hierarchical_Attention_in_Vision-Language_Models

- 属于算法pipeline的实现是什么？实验比较什么？
  论文提出了 HiPrune，一种 training-free、model-agnostic 的视觉 token 剪枝方法。核心发现：视觉编码器（ViT）不同层有分层注意力模式——中间层关注图像中的 main object（object-centric），深层关注全局上下文信息（global representation）。基于此，HiPrune 从中间层选 Anchor Tokens（高注意力）+ Buffer Tokens（空间相邻），从输出层选 Register Tokens（高注意力），三者组合保留图像局部细节和全局信息。HiPrune++ 在此基础上额外加入与 text embedding 余弦相似度高的 token，增强指令跟随能力。
  实验比较 HiPrune 与 9 种 SOTA token 剪枝方法（ToMe、FastV、SparseVLM、HiRED、TRIM、VisionZip、PyramidDrop、DivPrune、VisPruner）在 4 个 VLM 上的准确率和效率。
  关键指标：在 LLaVA-1.5-7B 上，保留 1/3 token 时保持 99.3% 准确率，FLOPs 减少 58.7%；保留 1/9 token 时 HiPrune++ 仍保持 96.1%。在 LLaVA-NeXT-7B（高分辨率）上，保留 2/9 token 保持 99.7%。Qwen2.5-VL 上也达到 SOTA。

- 硬件平台是什么，配置是什么。
  NVIDIA A100-PCIE 40GB，部分 overhead 分析在 RTX 5090 上测量 wall-clock latency。

- 模型是什么。数据集和bench分别是什么。
  模型：LLaVA-1.5-7B（CLIP-ViT-L/14, 576 tokens/image）、LLaVA-NeXT-7B（CLIP-ViT-L/14, 2880 tokens/image）、Qwen2.5-VL-3B/7B/32B-Instruct（动态分辨率 ViT）、Video-LLaVA-7B（视频）、LLaVA-1.5-13B、LLaVA-NeXT-13B。
  数据集/Benchmark：GQA、MMB、MMB-CN、MME、POPE（幻觉）、SQA-IMG、VQA-V2、VQA-Text（TextVQA）、VizWiz、DocVQA（文本为主任务）、MVBench（视频）、Vinoground（视频密集时序推理）。COCO val2017 用于 IoU 分析。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源链接：https://github.com/Danielement321/HiPrune
  工具：LMMs-Eval 工具包评估，calflops 计算 FLOPs。

  算法 pipeline（以 LLaVA-1.5 为例，l=9, α=0.1）：
  ```
  # Step 1: ViT 编码图像，获取各层注意力
  image_tokens, all_attns = encoder(image)  
  # image_tokens: (N=576, d), all_attns: list of (H, N+1, N+1)

  # Step 2: 从 object layer l 提取注意力分数
  mid_attn = all_attns[l].squeeze(0)         # (H, N+1, N+1)
  mid_attn = mid_attn.mean(0)                # 平均 multi-head -> (N+1, N+1)
  mid_attn = mid_attn.sum(0)[1:]             # 每个 token 收到总注意力 -> (N,)
  # a_i^{[l]} = (1/H) * sum_h sum_n A^{[l]}[h,n,i]

  # Step 3: 选择 Anchor Tokens（中间层 top-k 注意力）
  N_a = round(N' * α / 5)                    # α=0.1, 5 tokens per cluster
  anchor_idx = topk(mid_attn, k=N_a)

  # Step 4: 选择 Buffer Tokens（空间相邻，Cross scheme）
  # 每个 anchor 的上下左右 4 个邻居
  buffer_idx = cat([anchor_idx-1, anchor_idx+1, 
                    anchor_idx-p, anchor_idx+p])
  anchor_buffer_idx = unique(cat([anchor_idx, buffer_idx]))

  # Step 5: 从输出层选 Register Tokens
  deep_attn = all_attns[-1].squeeze(0).mean(0).sum(0)[1:]  # (N,)
  mask = zeros(N); mask[anchor_buffer_idx] = 1
  deep_attn -= mask
  r_sum = N' - len(anchor_buffer_idx)
  register_idx = topk(deep_attn, k=r_sum)

  # Step 6: HiPrune++ 可选 - 文本引导补充
  avg_text = text_encoder(text).mean(-2)     # 平均 text embedding
  avg_text /= avg_text.norm(-1)
  image_tokens_norm = image_tokens / image_tokens.norm(-1)
  similarity = avg_text @ image_tokens_norm.T  # (N,)
  t_sum = round(N' * β / 5)                 # β=0.1
  text_idx = topk(similarity, k=t_sum)

  # Step 7: 保留选中 token，丢弃其余
  retained_idx = cat([anchor_idx, buffer_idx, register_idx, text_idx])
  retained_tokens = image_tokens[retained_idx]  # (N', d)
  # retained_tokens 送入 projector → LLM
  ```

  张量计算关键点：
  - N=576 (LLaVA-1.5), N=2880 (LLaVA-NeXT 5 crops)，N 在 Qwen 动态分辨率下可变
  - d=1024 (CLIP-ViT-L/14 hidden dim)
  - 注意力维度：(H=16, N+1, N+1)，+1 for CLS token
  - HiPrune 在 ViT 输出后、projector 之前执行，与 FlashAttention 兼容
  - 排序开销 <1% prefill latency，HiPrune++ text encoder 开销 <10%

## Investigating_Video_Reasoning_Capability_of_Large_Language_Models_with_Tropes_in_Movies

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：(1) **FEVoRI (Face-Enhanced Viper of Role Interactions)** —— 在 ViperGPT 的 Visual Programming API 中集成 DeepFace 人脸识别工具（`face_identify`），通过 ICL example 引导 LLM 生成的代码逐帧识别人物并跟踪角色交互，将抽象概念（如 "Big Bad" 反派角色）的推理映射为具体的人物识别、行为分析和负面事件归因的链式推理。通过角色感知（role awareness）增强对 Abstract Perception 挑战的处理能力。(2) **ConQueR (Context Query Reduction)** —— 系统性地将电影叙事上下文（context）和 trope 查询（query）解耦并渐进分解：先将 trope 定义分解为多个维度（如 "Big Bad" → evil characteristics + negative judgments + causation of terrible events），再对每一帧识别的人物、动作、事件与 trope 各维度逐一匹配对齐，提升 Long-range Compositional Reasoning 能力。(3) **ABCD (AST Based Code Dignosis)** —— 利用 VP 生成代码的 AST 量化数据集的 Abstract Perception（VLM Calls / VLM Tokens）和 Long-range Compositional Reasoning（AST Nodes / AST Edges）水平。

  实验比较：(a) 主实验 —— Fullset(V)和 Mainset(V+D)上 LLoVi(C-R)、SeViLA(LMM-IF)、LLaMA-VID(LMM-IF)、ViperGPT(VP)、FEVoRI、FEVoRI+ConQueR、Gemini 1.5 的 F1/Precision/Recall 及四类别 F1（CT/RI/ST/SL）；(b) FEVoRI 消融 —— modality(V vs V+D)、frames(120 vs everyshot vs 16)、VLM(BLIP-2 vs Gemini)、Coder(GPT-4 vs GPT-3.5)的 F1 变化；(c) ABCD 分析 —— TiM vs NExT-QA/GQA/OKVQA 的 AST 复杂度对比；(d) 与 Human Performance (65 F1, TiMoS) 的对比。

- 硬件平台是什么，配置是什么。
  论文未明确说明硬件平台和 GPU 配置。方法均为 training-free（FEVoRI、ConQueR 为 prompt 工程 + tool API 扩展，不需训练），推理调用 GPT-3.5/GPT-4 API（OpenAI）和 Gemini API。DeepFace 人脸识别在 CPU 上运行。SeViLA 微调使用五折交叉验证，硬件平台论文未明确说明。

- 模型是什么。数据集和bench分别是什么。
  模型：(1) Captioner-Reasoner: LLoVi (BLIP-2 VLM + LLM summarizer)；(2) LMM-IF: SeViLA (zero-shot + fine-tuned localizer)、LLaMA-VID (long-video-tuning, 240 frames)；(3) Visual Programming: ViperGPT (GPT-4 code generator + BLIP-2/Gemini VLM)；(4) Proposed: FEVoRI (ViperGPT + DeepFace face_identify tool)、ConQueR (progressive context-query decomposition)；(5) Upper bound: Gemini 1.5 (trillion-scale)。
  数据集：TiM (Tropes in Movies) —— 684 movies (MovieNet 数据源) × 95 tropes (TVTropes 数据源)，分 Fullset/VDeset/Mainset。Mainset: 50 movies, 平均 1699.6 frames, 1822.2 subtitle lines, 65k characters, 6.08 tropes per movie。Trope 分类: Character Traits(CT)、Role Interaction(RI)、Situation(ST)、Storyline(SL)。任务: 二分类 (trope present or not), metric: Micro F1。
  对比数据集: NExT-QA、GQA、OKVQA（用于 ABCD 分析）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址：https://ander1119.github.io/TiM（数据集+代码）

  算法 pipeline 伪代码（FEVoRI + ConQueR 推理流程）：

  ```
  # === FEVoRI (Face-Enhanced Viper of Role Interactions) ===
  # 基于 ViperGPT 框架，扩展 face_identify API (DeepFace backend)
  # 输入: video (frames), annotation (shot boundaries), trope query, definition
  def execute_command_FEVoRI(video, annotation, possible_answers, query):
      # Trope: Big Bad
      # Definition: The character who is the direct cause of all 
      #             bad happenings in a story.
      video_segment = VideoSegment(video, annotation)
      info = {"character_actions": {}, "negative_impacts": {}}
      
      for i, frame in enumerate(video_segment.frame_iterator()):
          # Step 1: 逐帧识别人物 (face_identify利用DeepFace)
          for character in frame.find("person"):
              character_id = video_segment.face_identify(character)
              if character_id is None: continue
              
              # Step 2: 查询角色行为
              action_query = frame.simple_query(
                  "What is this person doing?")
              
              # Step 3: 判断行为是否有负面影响
              negative_query = f"Does the action '{action_query}' " \
                               "have a negative impact?"
              has_negative_impact = frame.llm_query(
                  negative_query, to_yesno=True)
              
              # Step 4: 累积角色行为与负面事件
              info["character_actions"][character_id].append(
                  action_query)
              if "yes" in has_negative_impact.lower():
                  info["negative_impacts"][character_id] += 1
      
      # Step 5: 汇聚信息后判断 trope 是否存在
      answer, reason = video_segment.select_answer(
          info, query, possible_answers)
      return answer, reason, info

  # === ConQueR (Context Query Reduction) ===
  # 渐进分解电影context和trope query
  def execute_command_ConQueR(video, annotation, 
                               possible_answers, query):
      # Trope: Big Bad
      # Definition: direct cause of all bad happenings
      video_segment = VideoSegment(video, annotation)
      info = {"happened_bad_events": {}, "character_infos": {}}
      
      for i, frame in enumerate(video_segment.frame_iterator()):
          for person in frame.find("person"):
              person_id = video_segment.face_identify(person)
              if person_id is None: continue
              
              # 渐进描述人物外观
              if person_id not in info["character_infos"]:
                  desc_query = "Describe appearance in 10 words"
                  character_desc = person.simple_query(desc_query)
                  info["character_infos"][person_id] = {
                      "description": character_desc,
                      "actions": {}}
              
              # 查询人物动作
              action = person.simple_query(
                  "Describe action in the scene")
              info["character_infos"][person_id]["actions"][
                  f"{i} frame"] = action
          
          # Step: 检查是否有负面事件
          check_neg = "Is there any negative event in the scene?"
          any_neg = frame.simple_query(check_neg, to_yesno=True)
          
          if "yes" in any_neg.lower():
              event = frame.simple_query(
                  "What's happening in the scene")
              info["happened_bad_events"][f"{i} frame"] = {
                  "event": event, "potential_cause": []}
              
              # Step: 逐一匹配角色是否为负面事件的潜在原因
              for pid, cinfos in info["character_infos"].items():
                  desc = cinfos["description"]
                  for prev_i in range(i, max(i-5, 0), -1):
                      prev_action = cinfos["actions"].get(
                          f"{prev_i} frame", None)
                      if prev_action is not None:
                          # 匹配人物描述与事件
                          pq = f"Is person '{desc}' a " \
                               f"potential cause of '{event}'?"
                          is_person = frame.simple_query(
                              pq, to_yesno=True)
                          # 匹配动作与事件
                          aq = f"Is action '{prev_action}' " \
                               f"a potential cause of '{event}'?"
                          is_action = frame.simple_query(
                              aq, to_yesno=True)
                          if "yes" in is_person.lower() or \
                             "yes" in is_action.lower():
                              info["happened_bad_events"][
                                  f"{i} frame"]["potential_cause"
                                  ].append(pid)
                          break
      
      # 汇聚信息后判断 trope
      answer, reason = video_segment.select_answer(
          info, query, possible_answers)
      return answer, reason, info
  ```

  算法 pipeline 全栈执行流程（FEVoRI+ConQueR on ViperGPT, GPT-4 code generator + BLIP-2/Gemini VLM）：
  - 算法层：Trope query + definition → GPT-4 生成 Python 程序（含 ICL example 引导的 FEVoRI/ConQueR 推理模式）→ 程序调用 ViperGPT API（frame.find("person") 检测人物 → frame.simple_query(prompt) 调用 VLM(BLIP-2/Gemini) 提取视觉语义 → video_segment.face_identify() 调用 DeepFace 人脸识别分配角色 ID → frame.llm_query(prompt, to_yesno=True) 调用 LLM 做 Yes/No 判断）→ 逐帧积累角色交互信息 → video_segment.select_answer(info, query) 汇聚全局推理 → 输出 {True/False} + reasoning。
  - 系统框架层：ViperGPT 框架（Python 代码执行引擎 + VLM/LLM API 集成），不涉及 Serving 框架修改。
  - 编译框架层：论文未明确说明。
  - kernel调度层：论文未明确说明。
  - 硬件架构层：论文未明确说明。

## LiveStar__Live_Streaming_Assistant_for_Real-World_Online_Video_Understanding

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：LiveStar —— 面向在线视频理解的直播流助手，核心包含三个算法创新：(1) **SCAM (Streaming Causal Attention Masks)** 训练策略 —— 构建交错帧-字幕序列（interleaved frame-caption sequences），通过 causal masked attention 约束进行增量式视频-语言对齐。训练目标为 `max P([Cap_i^k] | [Ctx^{< t_i} {Mask^{≤t_i}}], [Frm^{t_i}])`，Mask 矩阵阻止对当前语义片段中已生成字幕 token 的注意力，但保留前一语义片段的终端字幕以传递场景语义边界。从 M 个释义字幕池中随机采样字幕防止过拟合。(2) **SVeD (Streaming Verification Decoding)** 推理框架 —— 在每个 incoming frame 时刻 t_j，通过单次前向传播验证最新字幕 [Dec] 的 perplexity：`PPL^{t_j}([Dec]) = sqrt[N]{1/P([Dec] | [Ctx^{≤t_j}], [Frm^{t_j}])}`。若 `PPL^{t_j}([Dec]) > α · PPL^{t_i}([Dec])`（α=1.03 默认），激活解码 gate 生成新字幕；否则保持沉默并将 [Dec] 移至上下文末尾。(3) **Peak-End Memory Compression** —— 受认知科学 Peak-End 规则启发，对 10+ 分钟视频（3 fps）进行记忆压缩。利用预计算 PPL 检测关键帧，结合语义片段终端字幕，以概率剪枝超出窗口 W（默认 40 帧）的旧帧。

  实验比较：
  (a) OmniStar 在线评估 —— 5 tasks (RNG/OTG/FDQ/COQ/MIQ)，对比 VideoLLM-online, VideoLLM-MoD, MMDuet，指标含 SemCor, TimDiff, TimRedun, TimCover, SumFluen, FPS；(b) OmniStar 离线评估 —— 固定解码时间点，对比 GPT-4V/4o, LLaVA-Video, InternVideo2.5, InternVL2.5, MiniCPM-V 2.6, Qwen2.5-VL, VideoLLM-online, VideoLLM-MoD, MMDuet；(c) Ego4D Narration Stream —— 对比 VideoLLM-online, VideoLLM-MoD, LION-FS, MMDuet；(d) SVBench —— 对话和流式评估对比；(e) 消融 —— 响应-沉默阈值 α（1.0-1.1）、记忆压缩策略（Uniform/FIFO/Peak-End vs KV Cache）、释义字幕池大小 M（1/2/3）。

- 硬件平台是什么，配置是什么。
  训练：8× NVIDIA A800 GPU。Full fine-tuning（Vision Encoder 冻结，MLP projector + LLM 全更新）。AdamW optimizer (β1=0.9, β2=0.999, weight decay=0.05)，learning rate 4×10⁻⁵，per-device batch size=1，gradient accumulation 4 steps (effective bs=32)，cosine LR scheduling with warmup ratio=0.03，训练 1 epoch。每序列最多 8192 tokens（8K context window）。静态分辨率策略：输入帧 resize 至 448×448，patch downsampling ratio=0.5。

- 模型是什么。数据集和bench分别是什么。
  模型：Vision Encoder — InternViT（冻结），每帧提取 16 tokens，帧率 1-4 FPS。MLP Projector — 将帧嵌入映射到 LLM embedding 空间。LLM Backbone — InternLM2.5-7B（全微调）。序列长度最多 8192 tokens。
  训练数据：Phase I — 63K 视频片段（ActivityNet Captions 9K + Shot2Story 33K + Ego4D Narration Stream 20K + MVBench 1K）。Phase II — 20K OmniStar 训练样本。总计 83K。
  Benchmarks：(1) OmniStar — 20,137 视频，15 种真实场景（Travel & Events, Sports, Pets & Animals, Music, Autos & Vehicles, Film & Animation, Nonprofits & Activism, Science & Technology, Education, Howto & Style, News & Politics, Entertainment, Comedy, People & Blogs, Gaming），5 任务（RNG/OTG/FDQ/COQ/MIQ）；(2) Ego4D Narration Stream；(3) SVBench。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址：https://github.com/yzy-bupt/LiveStar。

  算法 pipeline 伪代码：

  ```
  # ===== SCAM 训练: Streaming Causal Attention Masks =====
  # 输入: 视频帧序列 + 语义片段 
  # 对语义片段 C_k = {t_m,...,t_n}, 每帧 Frm^{t_i} 伴随字幕 Cap^k
  # 从 M 个 paraphrased captions 中随机采样

  def build_scam_mask(seq_len, clip_boundaries, caption_positions):
      mask = causal_mask(seq_len)  # 标准 causal mask
      for t_i in current_clip:
          # 掩蔽当前clip中已生成的字幕token
          for cap_pos in same_clip_captions_before(t_i):
              mask[t_i, cap_pos] = -inf
          # 掩蔽之前clips中非终端字幕
          for pos in non_terminal_captions(prev_clips):
              mask[t_i, pos] = -inf
      return mask

  # 训练目标: max P([Cap_i^k] | [Ctx^{<t_i} {Mask^{≤t_i}}], [Frm^{t_i}])
  # 仅对 assistant response tokens 计算 cross-entropy loss

  # ===== SVeD 推理: Streaming Verification Decoding =====
  def sved_inference(frame_stream, alpha=1.03):
      Dec = None; Ctx = []; t_i = 0
      for t_j, Frm in enumerate(frame_stream):
          Ctx.append(Frm)
          if Dec is not None:
              # 单次 forward pass 验证 perplexity
              PPL_new = forward_pass(Dec, Ctx).PPL
              if PPL_new > alpha * PPL_cache[t_i]:
                  Dec = generate(Ctx)       # 激活解码
                  Ctx.append(Dec); t_i = t_j
                  PPL_cache[t_j] = forward_pass(Dec, Ctx).PPL
              else:
                  swap_last_two(Ctx)        # 沉默: Dec移到末尾
          else:
              Dec = generate(Ctx)
              Ctx.append(Dec); t_i = t_j
              PPL_cache[t_j] = forward_pass(Dec, Ctx).PPL
      return all_captions

  # ===== Peak-End Memory Compression =====
  def peak_end_compression(frames, captions, window_W=40):
      # 关键帧: 低PPL=高重要性, PPL from SVeD
      for f in frames:
          f.score = 1.0 / f.precomputed_PPL
      terminal_caps = [clip.last_caption for clip in clips]
      # 概率剪枝: P(delete) ∝ relative_PPL × elapsed_time
      for f in frames_older_than(window_W):
          p = (f.PPL / max_PPL_in_clip) * \
              (f.time / total_duration)
          if random() < p: drop(f)
      return frames_after_prune, terminal_caps
  ```

  关键张量计算与设计要点：
  - Vision: 448×448 → InternViT → [16, D] per frame (16 visual tokens)
  - Perplexity: PPL = exp(-1/N Σ log P(token_i | context, past_tokens))
  - SCAM Mask: [seq_len, seq_len] 稀疏 causal mask，跨clip non-terminal captions 被masked
  - SVeD: 仅需单次 forward pass 算 PPL（非完整 decoding），比 EOS-based 方法更快
  - Peak-End: W=40 frames ≈ 13.3s @3fps, 剪枝概率正比于 PPL 和 elapsed time
  - KV Cache: 双级缓存 (intra-dialogue frame-level + inter-dialogue cross-conversation)

## LongLLaVA__Scaling_Multi-modal_LLMs_to_1000_Images_Efficiently_via_Hybrid_Architecture

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：LongLLaVA —— 提出一种 Hybrid Mamba-Transformer 多模态大语言模型架构，通过三个层面扩展 MLLM 的长上下文多图理解能力：
  (1) **Hybrid LLM Architecture**：在 LLM backbone 中使用 4 组 hybrid layer stack，每组以 Attention:Mamba = 1:7 的比例交替集成 Transformer 层和 Mamba 层。同时采用 Mixture of Experts (MoE) 每隔一层集成，共 16 个 experts，每 token 激活 top-2。使用 RMSNorm 层归一化、Grouped Query Attention (GQA)、SwiGLU 激活函数。总参数 53B，推理时激活参数 13B (LongLLaVA-A13B)。LongLLaVA-9B 变体通过保留仅 Expert-0 构建，激活参数 9B。
  (2) **2D Bilinear Token Compression**：使用 CLIP (openai/clip-vit-base-patch32) 作为视觉编码器，两层 MLP 作为 projector。在 projector 前应用 2×2 bilinear pooling，将每张图像 token 数从 576 压缩到 144，同时保持 patch 间空间关系。
  (3) **Data Processing Protocol**：设计特殊 token 区分不同模态信息 —— `<img>` / `</img>` 包围图像 token 序列；`<vid>` / `</vid>` 包围视频帧序列；`<t>` 插入连续帧之间表示时间依赖；`\n` 用于高分辨率子图空间布局（分隔全局图与子图块、子图行）。
  (4) **Three-Stage Progressive Training**：Stage I (Single-image Alignment) —— 使用 600K image-caption pairs (ALLaVA-Caption + ShareGPT4V)，仅训练 projector，冻结 vision encoder 和 LLM；Stage II (Single-image Instruction Tuning) —— 使用 932K QA pairs (LLaVA-1.5 + Mantis-Single)，训练 projector + LLM，冻结 vision encoder；Stage III (Multi-image Instruction Tuning) —— 使用 200K Mantis + 200K VideoChat2 + 50K ShareGPT4Video + 200K Replay (single-image) + 50K Replay (pure-text) + 50K Sub-Image，全面多图能力训练。

  实验比较：
  (a) **Multi-image Evaluation** —— MileBench (Temporal/Semantic/IR)、Video-MME (128 frames w/o subs, Short/Medium/Long)、MVBench (20 video tasks)、LongVideoBench，对比 GPT-4V/GPT-4o/Gemini-1.5-Pro/Claude3-Opus (商业) 和 LongVA/InternVL2/InternVL2.5/OmChat/LongVILA/Qwen2-VL/Qwen2.5-VL/VideoLLaMA2/mPLUG-Owl3/Phi-3-Vision/Cobra/VideoChat2 (开源)。
  (b) **Diagnostic Long-Context Evaluation** —— VNBench (retrieval/ordering/counting atomic capabilities) 和 Video-NIAH (1200 images, needle-in-a-haystack)，对比 GPT-4o/GPT-4V/Qwen2-VL/VideoLLaMA2。
  (c) **Ablation Studies** —— MLLM Backbone 对比 (Vicuna-13B vs Jamba-9B)、Token Compression (no pooling / 1D pooling / 2D pooling)、Dataset Construction (single-image only / multi-image addition)、Training Strategy (mixed vs progressive)、Replay Data ablation。
  (d) **Scaling Analysis** —— Frames scaling (Video-MME 1→256 frames)、Shots scaling (VL-ICL few-shot vs fine-tuning)。
  (e) **Token Compression Impact** —— 5 general VL benchmarks + V* Bench (small object localization) 不同 token count 下的性能与推理开销，以及 Sub-Image Partitioning 缓解策略。
  (f) **Efficiency Analysis** —— 处理 100K tokens 的 Prefill time、Throughput、Memory usage、Max Throughput，对比 Falcon-mamba (Mamba only) 和 LLaVA-1.6 (Transformer only)。
  (g) **Applications** —— Healthcare (Pathology: VQA-RAD + PathVQA; 3D CT: CT-RATE) + Remote Sensing (FIT-RSFG-VQA + STAR dataset)。
  (h) **Single-image Evaluation** —— GQA/MME/MM-Vet/ScienceQA/SEED-Bench-v1/MMBench/MMMU/BLINK/ChartQA/DocVQA，对比商业和开源模型。

- 硬件平台是什么，配置是什么。
  训练：3 × 8 NVIDIA A800 GPU（共 24 卡）。数据序列随机采样拼接至 176K tokens，`<eos>` 分隔。单 epoch 训练，cosine learning rate schedule，warmup ratio 0.03，peak learning rate 1e-5。
  推理效率测试：单张 A100 80GB GPU (或 A800 80GB)。Efficiency metrics 使用 vLLM 框架 + Int8 quantization 评估 100K tokens 输入的 Prefill/Throughput/Memory。所有评估默认使用 Int8 quantization 降低计算开销，FP16 precision。

- 模型是什么。数据集和bench分别是什么。
  模型：Vision Encoder — CLIP ViT (openai/clip-vit-base-patch32)。Projector — 2-layer MLP。LLM Backbone — Hybrid Mamba-Transformer (Attention:Mamba=1:7, MoE 16 experts top-2)。LongLLaVA-9B (9B active params, Expert-0 only) 和 LongLLaVA-A13B (53B total, 13B active)。每张图像 token 数 144 (经 2×2 bilinear pooling)。
  训练数据 — Stage I: ALLaVA-Caption + ShareGPT4V (~600K captions)；Stage II: LLaVA-1.5 + Mantis-Single (~932K QA pairs)；Stage III: Mantis 200K + VideoChat2 200K + ShareGPT4Video 50K + Replay (single-image 200K + pure-text 50K) + Sub-Image 50K。Pure-text Instruction Tuning 使用 Evol-instruct-GPT4 + WildChat + SmolTalk + Tulu3 (DEITA) + LongAlign (~813K entries)。
  Benchmarks Multi-image: MileBench, Video-MME (30 sub-fields, 128 frames), MVBench (20 tasks), LongVideoBench (QA up to 1h), VNBench (synthetic video, retrieval/ordering/counting), VL-ICL (Matching Image task), Video-NIAH (1200 images)。
  Benchmarks Single-image: GQA, MME (perception), MM-Vet (6 VL capabilities), ScienceQA, SEED-Bench-v1 (image), MMBench (20 dimensions), MMMU (183 subfields, 30 image types), BLINK, ChartQA, DocVQA。
  Applications: Healthcare — VQA-RAD, PathVQA, CT-RATE (1304 samples, 512-1024 px, 100-984 slices)。Remote Sensing — FIT-RSFG-VQA, STAR dataset (1024×768 和 3327×4083 px)。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址：https://github.com/FreedomIntelligence/LongLLaVA

  算法 pipeline 伪代码：

  ```
  # ===== LongLLaVA 完整推理流程 =====

  # -- 视觉编码与压缩 --
  # 输入: N 张图像 (336×336) 或视频帧
  vision_encoder = CLIP_ViT()  # openai/clip-vit-base-patch32
  projector = 2-layer_MLP()

  for image_i in images:
      # CLIP 编码: 336×336 → 576 tokens (24×24 patch grid)
      H_v_raw = vision_encoder(image_i)  # [576, d_clip]
      # 2D 双线性池化: 2×2 pooling → 144 tokens (12×12 layout)
      # Reshape: [576, d] → [24, 24, d] → pool → [12, 12, d] → [144, d]
      H_v = bilinear_pool_2d(H_v_raw)      # [144, d_llm]
      H_v = projector(H_v)                 # [144, d_llm]

      # -- 数据格式包装 --
      if is_video:
          # 视频: <vid>\n<image><t><image>...\n</vid>
          token_seq += [<vid>, <image>, H_v_i, <t>, ..., </vid>]
      elif is_multi_image:
          token_seq += [<img>, H_v_i, </img>]
      else:  # single image
          token_seq += [<img>, H_v_i, </img>]

      if is_high_res:
          # 高分辨率: 全局图 + 子图分块 (pad to multiple of 168, 168×168 blocks)
          # 子图按 raster scan 排列, \n 分行
          sub_images = partition_and_pad(image_i, block_size=168)
          for row_blocks in sub_images:
              for block in row_blocks:
                  token_seq += [<img>, encode_and_pool(block), </img>]
              token_seq += [\n]

  # -- LLM Backbone (Hybrid Mamba-Transformer) --
  # 4 个 stack，每个 stack 内 Attention:Mamba = 1:7
  # 每隔一层有 MoE (16 experts, top-2 gating per token)
  for each stack in range(4):
      for layer in range(stack_size):
          if layer % 8 == 0:  # Attention layer
              # GQA: Q [1, d_head×n_query], K,V [seq, d_head×n_kv]
              H = RMSNorm(H)
              Q, K, V = W_Q(H), W_K(H), W_V(H)
              attn_out = FlashAttention(Q, K, V, causal=True, num_kv_heads=n_kv)
              H = H + attn_out
              if is_moe_layer:
                  # MoE FFN: 16 experts, top-2 routing
                  gate_logits = router(H)
                  top2_experts, top2_weights = topk(softmax(gate_logits), k=2)
                  H_moe = sum(w * expert_i(H) for i, w in zip(top2_experts, top2_weights))
                  H = H + H_moe
              else:
                  H = H + SwiGLU_FFN(H)
          else:  # Mamba layer
              H = RMSNorm(H)
              H = H + MambaBlock(H)  # SSM: Δ, A, B, C 参数扫描
              if is_moe_layer:
                  H = H + MoE_FFN(H)
              else:
                  H = H + SwiGLU_FFN(H)

  # 自回归生成
  response = autoregressive_decode(H, max_new_tokens)
  ```

  关键张量计算流程（LongLLaVA-A13B, single image 336×336）：
  - Vision Encoder: 336×336×3 → CLIP ViT-B/32 → [576, d_clip] → bilinear 2×2 pool: reshape to [24,24,d] → avg pool 2×2 → [12,12,d] → flatten → [144, d_clip] → 2-layer MLP projector → [144, d_llm]
  - Hybrid LLM: [144 + l_text, d_llm] 输入。每 8 层为 1 组 hybrid stack (1 Attention + 7 Mamba)。Attention layers 中 QKV 计算使用 GQA (n_kv < n_heads)。Mamba layers 中 SSM 输入 → 1D Conv + SiLU → Δ/B/C/A projection → selective scan → output gate。MoE layers 中 router 输出 16-dim logits → top-2 gating → expert FFNs 加权求和。
  - FlashAttention 兼容：Attention layers 支持标准 FlashAttention (causal mask)。Mamba layers 使用 Mamba SSM kernel (selective scan)。
  - Int8 Quantization: 评估时使用 Int8 量化降低计算开销，LLM backbone 权重 + 激活量化。

  关键设计：
  - 1:7 Attention:Mamba 比例：在 1.3B 模型上训练实验验证 1:3 与 1:7 性能差距极小但 1:7 计算效率显著更高
  - Expert 仅保留 Expert-0 (LongLLaVA-9B)：MMLU 和 BBH 上不同专家选择方法差异极小
  - 2D Pooling 优于 1D Pooling：12×12 layout 保持空间关系，GQA/Mile 优于 1D
  - Progressive Training 优于 Mixed Training：Multi-image 任务上有明显提升，Single-image 持平
  - Replay Data 关键：防止 single-image 和 text 能力退化。Text replay 50K 已饱和，Single-image replay 随数据量持续改善

## LongVT__Incentivizing__Thinking_with_Long_Videos__via_Native_Tool_Calling

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：三阶段训练pipeline（1）Cold-Start SFT：在Qwen2.5-VL-7B基础上用247.9K SFT样本（含19,161条tool-augmented iMCoTT traces）进行监督微调，教会模型提出时间窗口、调用crop_video工具、基于retrieved frames推理；（2）Agentic RL (GRPO)：用1.6K RL样本，以联合奖励函数（answer accuracy + format compliance + temporal IoU）优化策略，让模型学会何时检查视频、裁剪多长、如何整合证据；（3）Agentic RFT：用15.4K自蒸馏高质量RL rollout traces进一步微调，稳定agentic行为。
  实验比较：LongVT-7B-SFT/RFT vs Qwen2.5-VL-7B、Video-R1-7B、VideoRFT-7B、Video-Thinker-7B，以及专有模型GPT-4o、Gemini 1.5 Pro。在VideoMME、VideoMMMU、LVBench、VideoSIAH-Eval四个长视频理解benchmarks和Charades-STA temporal grounding benchmark上评估。

- 硬件平台是什么，配置是什么。
  NVIDIA A800-SXM4-80GB GPUs：SFT用32卡，RL和RFT用64卡。推理评估用8卡A800。

- 模型是什么。数据集和bench分别是什么。
  模型：Qwen2.5-VL-7B-Instruct作为基座模型。
  数据集：自建VideoSIAH数据套件（247.9K SFT样本 + 1.6K RL样本 + 15.4K RFT样本 + 652条VideoSIAH-Eval QA对），辅以LLaVA-CoT(54.6K)、OpenVLThinker(2.8K)、We-Math 2.0(602)、LongVideo-Reason CoT(5.2K)、Video-R1 CoT(165.6K)作为SFT数据。
  Benchmarks：VideoMME（平均1018s/视频）、VideoMMMU（平均506s/视频）、LVBench（平均4101s/视频）、VideoSIAH-Eval（平均1688s/视频）、Charades-STA（temporal grounding）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源：代码、数据、模型权重公开在 https://github.com/EvolvingLMMs-Lab/LongVT。
  
  算法pipeline伪代码：
  ```
  # Stage 1: Cold-Start SFT (32 GPU, LMMs-Engine, stream packing)
  model = Qwen2.5-VL-7B-Instruct
  train_data = concat(SFT_w_tool(19.2K), SFT_wo_tool(228.8K))
  for step in 1..3000:
      batch = dynamic_stream_packing(train_data, buffer_size=51200)
      loss = -Σ log P(x_t | x_<t)   # next-token prediction
      update(model, AdamW(lr=5e-5, cosine, warmup=300))

  # Stage 2: Agentic RL via GRPO (64 GPU, verl + SGLang)
  model = load(SFT_checkpoint)  
  for step in 1..160:
      for prompt in batch(16):
          # 采样K=16个rollouts，每个rollout包含多轮tool calling
          for k in 1..16:
              y_k ~ π_θ_old(·|x, max_new_tokens=16384
                             , max_prompt_len=36000)
              # y_k = <think>... hypothesize window ...
              #       <tool_call>{"name":"crop_video",
              #         "arguments":{"start_time":t_s,"end_time":t_e}}
              #       </tool_call>
              #       <tool_response> cropped frames </tool_response>
              #       <think>... verify evidence ...</think>
              #       <answer> final answer </answer>
          
          # 计算联合奖励
          R_acc[k] = Judge_LLM(answer_k, answer_gt) ∈ {0, 0.5, 1}
          R_fmt[k] = 1 if format == <think>..<tool_call>..<answer> else 0
          R_time[k] = IoU([t_s,t_e], [t_s',t_e'])
          R[k] = R_acc[k] + R_fmt[k] + R_time[k]
          
          # GRPO advantages
          b = mean(R[1..K])
          A[k] = R[k] - b
          
          # 更新policy (KL-constrained)
          J = 1/K Σ A[k] Σ log π_θ(y_t|x,y_<t) - β*KL(π_θ||π_ref)
      update(model, AdamW(lr=1e-6, constant))

  # Stage 3: Agentic RFT (64 GPU)
  # 筛选高质量rollouts: answer正确 AND temporal IoU ≥ 0.3
  rft_data = filter(RL_rollouts, 
                    answer_correct & IoU(span_pred, span_gt) >= 0.3)
  model = load(best_RL_checkpoint)
  for step in 1..1600:
      batch = dynamic_stream_packing(rft_data, buffer_size=51200)
      loss = -Σ log P(x_t | x_<t)
      update(model, AdamW(lr=5e-5, cosine, warmup=160))
  ```

  张量计算示例（RL阶段，单rollout，B=1）：
  - 输入：prompt x + 512 frames video → 视觉编码器 → vision tokens (~512×256=131K tokens) + text tokens
  - 模型生成：max 16384 new tokens（含think + tool_call JSON + tool_response + think + answer）
  - crop_video执行：根据start_time/end_time从原始视频中重采样64帧 → 再次编码为vision tokens
  - KL divergence计算：每token位置计算 π_θ(·|x,y_<t) 与 π_ref(·|x,y_<t) 的KL散度
  - IoU计算：对predicted [t_s, t_e] 和 GT [t_s', t_e'] 计算 |intersection|/|union|
  - 多轮：up to 5 turns（T1到T5），每轮可再次调用crop_video精炼窗口

## TimeSearch-R__Adaptive_Temporal_Search_for_Long-Form_Video_Understanding_via_Self-Verification_Reinforcement_Learning

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：**TimeSearch-R** 框架，包含两大核心创新：(1) **Interleaved Text-Video Thinking** —— 将时序搜索重新定义为文本-视频交错的思维过程。在每个推理步 k，policy model π_θ 生成文本推理 T_k；若 T_k 包含搜索指令，video environment 根据时间边界 [t_s^k, t_e^k] 和文本 query q_k 执行 search() 函数检索视频片段 V_k，将其追加到 CoT 中供后续步骤使用。搜索函数使用 SigLIP-400M 视觉编码器计算候选帧与 query 的相关性，再通过 DPP (Determinantal Point Process) 同时优化相关性和多样性，选出 F 帧最有信息量的帧。该过程循环直到模型输出最终答案或达到预算上限（最多 8 轮搜索，每轮最多 8 帧）。(2) **GRPO-CSV (Completeness Self-Verification)** —— 针对原始 GRPO 仅对最终答案给予奖励而忽略中间搜索决策导致的"搜索不充分"和"推理不一致"问题。CSV 在 GRPO rollout 阶段生成 text-video 交错 CoT C 和最终答案 A 后，提取 C 中的视频帧构成动态帧集 V_c，用同一 policy model 仅基于 V_c 重新回答（禁止新搜索）得到 CSV 答案 A_c。Completeness Reward: R_c = 1[Acc(A,A*)>0.5] · Acc(A_c,A*)，确保仅对正确轨迹施加 CSV 奖励。总奖励: R = R_c + R_fmt + R_acc。

  实验比较：(1) Temporal Search: 对比 Uniform (Qwen2.5-VL/GPT-4o)、VideoAgent (GPT-4)、Retrieval-based (GPT-4o)、T* (GPT-4o)、VideoTree (GPT-4) 等方法；指标包括 Temporal F1/Precision/Recall、Visual F1、QA Accuracy (Haystack-LVBench, Haystack-Ego4D)。(2) Long-Form Video Understanding: 对比 Qwen2.5-VL-7B、GPT-4o、Gemini-1.5-Pro、VideoAgent、VideoTree、T*、Video-R1-7B；指标包括 VideoMME (short/medium/long/overall)、MLVU (m-avg)、LongVideoBench。(3) Ablation: 训练阶段 (zero-shot CoT → SFT → GRPO → GRPO-CSV)、GRPO-CSV 组件消融 (w/o CSV vs w/ CSV vs w/ CSV+Acc)、数据组成消融 (有无 filtering、ego/exo domain diversity)。(4) Efficiency: end-to-end latency vs VideoAgent/T*/Retrieval-based on Haystack-Ego4D on A100。

- 硬件平台是什么，配置是什么。
  RL 训练：32 × NVIDIA A100 GPU。使用 DeepSpeed ZeRO-3 Offload 做内存优化，vLLM colocate mode 做 rollout 推理加速。Batch size per GPU = 1，Gradient Accumulation Steps = 2。Mixed precision bfloat16 + Flash Attention 2.0。推理效率评估：A100 GPU 上测 end-to-end latency。SFT 冷启动阶段使用 GPT-4o 生成 text-video 交错推理数据。

- 模型是什么。数据集和bench分别是什么。
  模型：基于 Qwen2.5-VL-7B-Instruct 做两阶段训练 (SFT → RL with GRPO-CSV)。基座模型使用 SigLIP-400M 作为搜索函数的视觉编码器（独立于 policy model）。训练数据经过两阶段过滤：(1) Stage 1: Visual Dependency Filtering —— 用 4 帧均匀采样过滤可被纯语言偏置解答的样本；(2) Stage 2: Search Usefulness Filtering —— 用最多 64 帧 + dynamic temporal search 过滤即使充分搜索也无法回答的样本。数据来源：Haystack-Ego4D (49.5%)、VideoMarathon/Panda-70M (35.6%)、CinePile (9.5%)、其他 (5.4%)；open-ended QA 占 60.3%，multiple-choice 占 39.7%；平均视频时长 1659s。

  评测 benchmark：(1) Temporal search: Haystack-LVBench (needle-in-a-haystack, 含 temporal/visual similarity + QA accuracy)、Haystack-Ego4D (test-tiny subset)。(2) Long-form video understanding: VideoMME (w/o sub, 分 short/medium/long)、MLVU (m-avg)、LongVideoBench (LVB)。另外自定义两个评估指标：completeness（搜索帧集是否足以回答正确）和 consistency（中间推理与最终答案是否一致）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  代码开源：https://github.com/Time-Search/TimeSearch-R

  TimeSearch-R 算法 pipeline 伪代码：
  ```
  # ===== 初始化 =====
  # policy model π_θ: Qwen2.5-VL-7B-Instruct (Qwen2.5-VL backbone)
  # search function: SigLIP-400M encoder + DPP frame selection
  # 视频 V, 问题 Q, 初始预览 frames Ṽ (uniform sampling)

  # ===== Phase 0: 推理-搜索交错过程 =====
  def time_search_reasoning(V, Q):
      Ṽ = uniform_sample(V, fps=2, max_frames=768)  # 初始预览
      C = []  # text-video interleaved CoT
      for k in range(1, 9):  # max 8 search turns
          # Step 1: 文本推理 - policy model 生成 reasoning
          T_k = π_θ.think(Ṽ, Q, C)  # <think>...</think>
          
          # Step 2: 检查是否搜索或回答
          if contains_<tool_call>(T_k):
              t_s, t_e, q, F = parse_search_params(T_k)
              # Step 3: 视频环境执行搜索
              # DPP-based frame selection
              F_cand = subsample(V, t_s, t_e, N_cand)
              # 计算视觉嵌入
              h_i = SigLIP.encode(f_i) for f_i in F_cand  # [N, d]
              q_emb = SigLIP.encode_text(q)              # [d]
              # 相关性分数
              r_i = (q_emb^T · h_i)  # unnormalized relevance
              r_i = (r_i - min(r)) / (max(r) - min(r) + ε)  # [0,1]
              # DPP kernel: L̃ = diag(r) · S · diag(r)
              S_ij = h_i^T · h_j  # pairwise cosine similarity
              L̃_ij = r_i · r_j · S_ij
              # Fast greedy MAP inference 选 top-F 帧
              V_k = greedy_map_inference(L̃, F)  # |V_k| = F ≤ 8
              C.append((T_k, V_k))  # 追加到 CoT
          elif contains_<answer>(T_k):
              A = parse_answer(T_k)
              return C, A
      return C, None  # budget exhausted

  # ===== Phase 1: GRPO-CSV Training =====
  # 每步训练:
  for prompt (V, Q, A*) in batch:
      # --- GRPO Rollout ---
      # 生成 8 个 rollout trajectories
      for i in range(8):
          C_i, A_i = time_search_reasoning(V, Q)
          # 提取动态帧集 V_c
          V_c_i = extract_all_frames(C_i)
      
      # --- CSV Rollout ---
      for i in range(8):
          # 用同一 π_θ 基于 V_c_i 重新回答（禁止搜索）
          A_c_i = π_θ.answer_no_search(Q, V_c_i)
      
      # --- 奖励计算 ---
      for i in range(8):
          R_acc_i = 1 if A_i == A* else 0  # Accuracy reward
          R_fmt_i = 1 if format_valid(C_i) else 0  # Format reward
          if Acc(A_i, A*) > 0.5:  # 仅正确轨迹
              R_c_i = Acc(A_c_i, A*)  # Completeness reward
          else:
              R_c_i = 0
          R_i = R_c_i + R_fmt_i + R_acc_i
      
      # --- GRPO Policy Update ---
      # advantage = (R_i - mean(R)) / std(R)
      # L_GRPO = -E[min(r_i(θ)·A_i, clip(r_i(θ),1-ε,1+ε)·A_i)]
      # r_i(θ) = π_θ(C_i,A_i|Ṽ,Q) / π_old(C_i,A_i|Ṽ,Q)
      # plus KL penalty β · KL(π_θ || π_ref) with β=0.005
      θ = optimizer(AdamW, lr=1e-6).step(L_GRPO + β·KL)

  # ===== Phase 2: SFT Cold-Start (两阶段训练的前置步骤) =====
  # GPT-4o 生成 text-video interleaved CoT 训练数据
  # 训练时 mask 视频 token，只计算 reasoning token 的 cross-entropy loss
  # L_SFT = -Σ log π_θ(token_t | context, Ṽ, Q)  over reasoning tokens
  ```

  关键张量流：
  ```
  输入: 初始预览 Ṽ (uniform sampling, max 768 frames @ 2fps)
  每帧: Qwen2.5-VL 原生 dynamic-FPS + absolute time encoding
        frame → [12-256 visual tokens] concatenated with timestamps

  Step k 搜索:
    query q → SigLIP-400M text encoder → q_emb ∈ R^d (d=768)
    frames in [t_s,t_e] → SigLIP-400M vision encoder → h_i ∈ R^d
    DPP kernel: L̃ ∈ R^(N×N), 选 F=8 帧最大化 det(L̃_S)
    selected frames → V_k = {f_k^1, ..., f_k^F}

  Policy model forward:
    input: [Ṽ_tokens, T_1, V_1_tokens, T_2, V_2_tokens, ..., T_k, Q]
    model: Qwen2.5-VL-7B backbone (32 layers, GQA, RoPE)
    output: next reasoning text / search instruction / answer

  CSV phase:
    input: [V_c_tokens, Q]  (仅搜索到的帧 + 问题，禁止 tool_call)
    output: A_c (bare answer, "I don't know" if insufficient)
  ```

  训练两个阶段的作用：
  - **SFT (Cold-Start)**: 教模型正确的 reasoning format 和 <tool_call> 格式，从 zero-shot 无法搜索（Temporal F1=0.0）提升到 F1=7.8
  - **RL (GRPO-CSV)**: 进一步提升 reasoning consistency (+2.6%) 和 QA accuracy (59.2%→66.6%)，GRPO-CSV 防止训练崩塌（w/o CSV 约 300 步后模型停止搜索）


## LongVILA__Scaling_Long-Context_Visual_Language_Models_for_Long_Videos

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：(1) 五阶段训练pipeline：Stage1 多模态对齐（冻结LLM和视觉编码器，仅训练多模态投影器）→ Stage2 大规模预训练（冻结视觉编码器，训练LLM+投影器，使用VILA-1.5-40B重标注的COYO-25M）→ Stage3 短视频监督微调（全参数微调，混合图像+短视频数据如YouCook2/ShareGPTVideo）→ Stage4 LLM上下文扩展（文本only持续预训练，渐进式从8K→65K→262K上下文，使用SlimPajama 17B tokens，RoPE基频增大+LoRA微调，约336 H100 GPU hours）→ Stage5 长视频监督微调（全参数训练，使用LongVILA_SFT数据集15,292个长视频，每个视频paired caption+QA问答对，配合MM-SP系统）。
  (2) 长视频指令数据生成pipeline：长视频→切分为10秒短片段→VILA-1.5模型逐片段生成caption→LLM基于所有片段caption生成QA对（涵盖总结、空间、属性、动作、对象、OCR、时序等7类问题）。

  实验比较：
  (a) 9个视频benchmark（ActivityNet-QA/EgoSchema/EventBench/LongVideoBench/PerceptionTest/MVBench/NExT-QA/VNBench/VideoMME），对比GPT-4V/GPT-4o/Gemini-1.5-Pro和Video-LLaVA/Flash-VStream/ShareGPT4Video/VideoLLaMA2/VideoLLaMA2.1/Kangaroo/PLLaVA/LLaVA-OV。LongVILA-7B在VideoMME取得65.1% (w/ subtitle)。
  (b) VideoMME按时长细分(Short/Medium/Long)，256 frames，LongVILA-7B取得72.9/64.9/57.4 (w/ sub)，证明frame scaling在长视频上的收益。
  (c) 训练阶段顺序ablation (Table 1)：1-2-3-4-5 vs 1-2-4-(3&5) vs 4-1-2-3-5 vs 4-1-2-(3&5)，最优为原始顺序57.5 avg VideoMME w/o sub。
  (d) Needle-in-a-Haystack：LongVILA在2048 frames训练后，6000 frame (1M+ tokens)测试达99.8%准确率，远超32-frame baseline和LongVA（3000 frame 87.6%）。
  (e) LongVILA-Caption benchmark（100长视频）：8→128→256 frames，Correctness/Detailed/Contextual分数从1.87/1.85/2.27提升至3.23/3.11/3.43。
  (f) 10个图像VLM benchmark (Table 9)：证明长视频训练不损害图像理解能力，LongVILA-7B S3模型在多个图像benchmark上领先。

- 硬件平台是什么，配置是什么。
  训练系统：H100节点（每节点8×H100 80GB，NVLink 900 GB/s intra-node，InfiniBand 50 GB/s inter-node single path）。最大序列长度实验使用32 A100节点（每节点8×A100 80GB）。推理系统：单节点8×H100 80GB。模型复杂度Profiling：单张A100 GPU，FP16，Flash-Attention2。Stage4上下文扩展：约336 GPU hours on 80GB A100。

- 模型是什么。数据集和bench分别是什么。
  模型：基于VILA-1.5（Encoder-Decoder VLM架构），视觉编码器(ViT) → 多模态投影器(linear/MLP) → LLM解码器。LLM backbone为Qwen2-1.5B和Qwen2-7B，使用GQA（8 KV heads, 32 Q heads）。每帧产生约256个tokens，1400帧视频约274K tokens。训练后支持8→2048帧视频输入。
  数据集：COYO-25M（VILA-1.5-40B重标注）、YouCook2、ShareGPTVideo、Shot2Story20k→LongVILA_SFT（15,292长视频，12类别：Travel/Sports/Education/Pets/People/News/Music/Science/Comedy/Entertainment/Film/Gaming，每视频1 caption + 1 QA对）、SlimPajama（17B text tokens，Stage4使用）、LongVILA-Caption（100长视频人工校验）。
  Benchmarks：ActivityNet-QA, EgoSchema, EventBench, LongVideoBench, PerceptionTest, MVBench, NExT-QA, VNBench, VideoMME；图像benchmarks: VQAv2, GQA, VizWiz, SQA-I, VQA-T, MMB, MMB-CN, SEED, LLaVAW, MM-Vet。

- 开源情况。基于开源文档和论文，使用例子解释算法pipeline，至少具体到伪代码或张量计算。
  开源：github.com/NVlabs/VILA/tree/main/longvila，基于HuggingFace Transformers通过monkey-patching集成。
  
  训练pipeline伪代码：
  ```
  # Stage 1: Multi-modal Alignment
  freeze(vision_encoder); freeze(llm)
  for (img, text) in alignment_data:
      v = projector(vision_encoder(img))
      loss = CrossEntropy(llm([v; text_tokens]), labels)
      update(projector)

  # Stage 2: Large-scale Pre-training  
  freeze(vision_encoder)
  for (img, text) in coyo_relabeled:
      v = projector(vision_encoder(img))
      loss = CrossEntropy(llm([v; text]), labels)
      update(llm, projector)

  # Stage 3: Short Supervised Fine-Tuning
  for (img_or_frames, text) in mixed_short_data:
      v = projector(vision_encoder(img_or_frames))
      loss = CrossEntropy(llm([v; text]), labels)
      update(all_params)

  # Stage 4: Context Extension (text-only, LoRA)
  rope_base *= scale  # 增大RoPE基频
  for text in slimpajama_8k_65k_262k:  # progressive schedule
      loss = llm_lora(llm(text))
      update(lora_params)
  
  # Stage 5: Long Video SFT (full params, MM-SP)
  for (frames, text) in long_video_sft:
      # Two-stage sharding in MM-SP
      local_frames = distribute_evenly(frames, sp_ranks)  # Stage1: per-image balance
      vis_feats = vision_encoder(local_frames)  # balanced encoding
      all_tokens = gather_concat(vis_feats, text_tokens)  # global aggregation
      local_tokens = shard_by_token_count(all_tokens, sp_ranks)  # Stage2: per-token balance
      loss = llm_2d_attention(local_tokens)  # 2D-Attention: A2A intra + P2P inter
      update(all_params)
  ```
  张量计算示例（256 frames, 32 GPUs）：
  - 输入：B=1, 256 frames × ~256 tokens/frame = 65536 vision tokens + T text tokens
  - Stage1 sharding：32 GPUs各处理8 frames，视觉编码负载均衡
  - Stage2 sharding：全局tokens按sequence dim均匀分配，每GPU持有 (65536+T)/32 tokens
  - 2D-Attention (4×8 mesh)：intra-node 4 GPUs A2A交换head-dim分片 → inter-node 8 groups P2P传输KV → SDPA本地计算

## MMTok__Multimodal_Coverage_Maximization_for_Efficient_Inference_of_VLMs

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：MMTok —— 基于多模态覆盖最大化（Multimodal Coverage Maximization）的 training-free 视觉 token 选择算法。核心思路是将 token 选择建模为最大覆盖问题（Maximum Coverage Problem），从 n 个 vision token 中选择 k 个（k ≪ n）以最大化覆盖文本 token（T-V coverage）和全体视觉 token（V-V coverage）的信息量。覆盖函数定义为子模函数（submodular function）f(S; M) = (1/m) Σᵢ maxⱼ M_{i,j}，通过贪心算法获得 (1-1/e) 近似最优解。
  具体计算流程：(1) 视觉编码器提取 vision token {v₁',…,vₙ'}（投影前）和 {v₁,…,vₙ}（投影后）；(2) 计算 text-vision 相似度矩阵 M^{tv} = tᵢᵀvⱼ（使用投影后的 vision token 以对齐文本语义）；(3) 计算 vision-vision 相似度矩阵 M^{vv} = vᵢ'ᵀvⱼ'（使用投影前的 vision token 以保留纯视觉信息）；(4) 对两个相似度矩阵分别做 softmax 归一化校准（温度 τ_t=0.02, τ_v=0.2）：M^{tv'}_{i,j} = exp(M^{tv}_{i,j}/τ_t) / Σⱼ exp(M^{tv}_{i,j}/τ_t)；M^{vv'} 同理；(5) 合并目标 f(S; M^{tv'}, M^{vv'}) = f(S; M^{tv'}) + α·f(S; M^{vv'})，α=0.5；(6) 贪心算法 O(kn) 选 k 个 vision token，每次选使得增量覆盖最大的 token。方法为 training-free，无需任何微调或额外训练参数。

  实验比较：(a) 性能对比 —— 在 LLaVA-1.5-7B (576 tokens → 192/128/64)、LLaVA-1.5-13B (576 → 192/128/64)、LLaVA-NeXT-7B (max 2880 → 640/320/160)、LLaVA-NeXT-13B (max 2880 → 640/320/160)、Qwen-2.5-VL-7B (dynamic tokens → 20%/10%/5%) 上，对比 FastV (vision-only)、SparseVLM (language-only)、VisionZip (CLS-attention-based)、DivPrune (diversity-based)、VisionZip fine-tuned。(b) 高 IC 任务极端压缩 —— 在 5 个高 Image Contribution 任务 (POPE, MME, MMB, SEED, GQA + TextVQA for NeXT) 上，压缩至 64→32→16→8→4→2 tokens，对比 VisionZip 和 DivPrune。(c) 效率分析 —— H100 上 LLaVA-NeXT-13B 和 A6000 上 Qwen2.5-VL-7B 的推理时间、GPU 利用率、显存对比。(d) 消融实验 —— T-V only vs V-V only vs Softmax 变体 vs MMTok full；token selection vs 图像 resize 策略对比；温度参数 τ_v 自适应搜索；word pooling 策略 (Mean/Max/First, Pre/Post) 对比；decoder 内 token 二次选择。(e) 推理任务 —— MMStar benchmark。(f) MMTok++ 改进 —— 排除 padding patches + 修复 overflow bug，进一步测试 32/16/8/4/2 tokens 极限性能。

- 硬件平台是什么，配置是什么。
  推理效率测试：NVIDIA H100 (80GB) 单卡 —— LLaVA-NeXT-13B loading 25.42GB，POPE dataset total infer time 测量。NVIDIA A6000 (48GB) 单卡 —— Qwen2.5-VL-7B loading 15.87GB，MME task infer time 测量；A6000 还用于 MMTok 贪心选择算法运行时间 profiling（100 runs 平均）。主要实验（各 benchmark 评估）论文未明确说明硬件平台，推测使用与效率实验相同的 GPU。

- 模型是什么。数据集和bench分别是什么。
  模型：(1) LLaVA-1.5-7B —— Vision Encoder: CLIP-ViT-L-336px (576 fixed vision tokens), LLM: Vicuna-7B。(2) LLaVA-1.5-13B —— 同上架构，LLM: Vicuna-13B。(3) LLaVA-NeXT-7B —— 动态多分辨率，最高 5 图像 × 576 = 2880 tokens。(4) LLaVA-NeXT-13B —— 同上，最大 2880 tokens。(5) Qwen-2.5-VL-7B-Instruct —— 动态分辨率 + token merging layer，平均 token 数 276.9~976.5（随数据集而异）。

  数据集和 Benchmark：(1) GQA (Acc.) —— 真实世界视觉推理与组合问答。(2) MMBench / MMB (Acc.) —— 多模态全方位评估。(3) MME (Perception+Cognition, P+C) —— 多模态大模型综合评测。(4) POPE (F1) —— 物体幻觉检测。(5) ScienceQA-IMG / SQA (Acc.) —— 科学多模态推理（低 IC 数据集）。(6) VQA-v2 Test-Dev (Acc.)。(7) TextVQA (Acc.) —— 图中文字问答。(8) MMMU (Acc.) —— 多学科多模态理解。(9) SeedBench-Image / SEED-I (Acc.)。(10) OCRBench (Acc.) —— Qwen 实验专用。(11) MMStar (Coarse/Fine-Grained/Instance/Logical/Math/Sci&Tech Acc.) —— 推理任务 benchmark。
  评估框架：Lmms-eval (https://github.com/EvolvingLMMs-Lab/lmms-eval)。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址：https://github.com/Ironieser/mmtok
  
  算法 pipeline 详细解释（张量计算级别）：

  **输入**：图像 I ∈ R^(3×H×W)，文本查询 T = {t₁,…,tₘ}，目标 token 数 k。
  
  **Step 1 — 视觉编码**：图像通过 CLIP-ViT 视觉编码器 → vision tokens V' = {v₁',…,vₙ'} ∈ R^(n×d)（投影前）。Vision tokens 通过 MLP projection layer → V = {v₁,…,vₙ} ∈ R^(n×d')（投影后，与 LLM embedding 空间对齐）。对于 LLaVA-1.5，n=576 (336×336 image)。
  
  **Step 2 — Text-Vision 相似度**：文本 T 通过 LLM tokenizer + embedding → text hidden states {t₁,…,tₘ}（使用 projection 后的 vision token 计算）。归一化为单位向量：∥tᵢ∥₂ = ∥vⱼ∥₂ = 1。计算 T-V 相似度矩阵 M^{tv} = T · Vᵀ ∈ R^(m×n)，其中 M^{tv}_{i,j} = tᵢᵀ vⱼ。
  
  **Step 3 — Vision-Vision 相似度**：使用投影前的 vision token V'。归一化后计算 V-V 相似度矩阵 M^{vv} = V' · V'ᵀ ∈ R^(n×n)，其中 M^{vv}_{i,j} = vᵢ'ᵀ vⱼ'。
  
  **Step 4 — Softmax 校准**：对两个相似度矩阵分别按行做 temperature-scaled softmax。M^{tv'}_{i,j} = exp(M^{tv}_{i,j}/τ_t) / Σⱼ exp(M^{tv}_{i,j}/τ_t)，τ_t=0.02。M^{vv'}_{i,j} = exp(M^{vv}_{i,j}/τ_v) / Σⱼ exp(M^{vv}_{i,j}/τ_v)，τ_v=0.2。
  
  **Step 5 — 多模态覆盖贪心选择 (Alg. 2)**：
  ```
  输入: M^{tv'} ∈ R^(m×n), M^{vv'} ∈ R^(n×n), k, α=0.5
  S = ∅
  for i = 1 to k:
    for each s ∈ N \ S:
      // 计算增量覆盖 f(S ∪ {s}; M^{tv'}, M^{vv'})
      // f(X; M^{tv'}) = (1/m) Σᵢ₌₁ᵐ max_{j∈X} M^{tv'}_{i,j}
      // f(X; M^{vv'}) = (1/n) Σᵢ₌₁ⁿ max_{j∈X} M^{vv'}_{i,j}
      g(s) = f(S ∪ {s}; M^{tv'}) + α · f(S ∪ {s}; M^{vv'})
    s* = argmax_s g(s)
    S = S ∪ {s*}
  返回 S  // 选中的 k 个 vision token 索引
  ```
  复杂度: O(kn)，对 m ≪ n 和 d 固定。对 2880 tokens 选 160 tokens 仅需 6.4ms on A6000，13.93 GFLOPs。
  
  **Step 6 — LLM 推理**：仅将选中的 k 个 vision token {v_s}_{s∈S} 与 text tokens 拼接后送入 LLM decoder。
  
  **与 baseline 对比示例**（以 LLaVA-1.5-7B, "Describe the cat in the image" 为例）：
  - Baseline: CLIP-ViT → 576 vision tokens → concat with ~10 text tokens → LLM (attention cost O(586²))
  - MMTok: CLIP-ViT → 576 vision tokens → greedy coverage selection (64 tokens) → concat with ~10 text tokens → LLM (attention cost O(74²))。保留 96.6% 原始性能（Avg. over 8 benchmarks），推理 token 数减少 88.9%。

## Multimodal_Long_Video_Modeling_Based_on_Temporal_Dynamic_Context

- 属于算法pipeline的实现是什么？实验比较什么？
  - 提出 Temporal Dynamic Context (TDC) 视频编码方法：将视频分割为语义一致的场景，保留首帧静态特征，用 Q-Former 将后续帧的视觉+音频 token 压缩为动态上下文 token。同时提出 Long Video Chain-of-Thought (LVCoT) 训练无关策略，将超长视频分段处理，逐段推理后汇总得到最终答案。
  - 实验比较：在 MVBench、PerceptionTest、EgoSchema、MLVU、Video-MME 上与 vision-focused MLLMs（LLaVA-OneVision, InternVL2, LongVU, VideoChat2 等）和 audio-visual MLLMs（VideoLLaMA2, PandaGPT, NExT-GPT 等）对比；在 Music-AVQA、AVSD 上做 audio-visual 联合理解评测。Ablation 研究 segment 数量、query 类型（AvgPool vs Learned Query）、context token 数量、text instruction 作用、LVCoT 效果。

- 硬件平台是什么，配置是什么。
  - 论文未明确说明训练所用 GPU 型号和数量。

- 模型是什么。数据集和bench分别是什么。
  - Backbone LLM: Qwen2-7B 和 LLaMA3.2-3B。
  - Visual Encoder: DINOv2 + SigLIP，每帧 144 tokens。
  - Audio Encoder: BEATs，16kHz 重采样，约 50 tokens/s。
  - Q-Former: 由预训练 BERT 初始化，默认 16 个 query tokens。
  - 训练数据 Stage 1（视觉-语言对齐）: LLaVA-OneVision (3.2M samples)。
  - 训练数据 Stage 2（视频指令微调）: LLaVA-Video, TextVR, YouCook2, EgoQA, Kinetics-710, NExTQA, CLEVRER, TGIF, WebVidQA, DiDeMo, ShareGPT4Video, MovieChat（Qwen2-7B: 2M, LLaMA3.2-3B: 540K samples）。
  - 训练数据 Stage 3（音频-视频指令微调）: AVQA, Music-AVQA, AVSD, LongVALE, AVInstruct + Stage 2 subset（Qwen2-7B: 300K, LLaMA3.2-3B: 120K）。
  - Benchmarks: MVBench (avg 16s), PerceptionTest (avg 23s), EgoSchema (avg 180s), MLVU (avg 651s), Video-MME (avg 1010s), Music-AVQA, AVSD。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：https://github.com/Hoar012/TDC-Video
  - 算法 Pipeline 伪代码：

```
# === Video Scene Segmentation ===
video_frames = sample_frames(video, fps=1)  # 每秒1帧
embeddings = DINOv2(video_frames)           # 高维特征
similarities = cosine_sim(embeddings[i], embeddings[i+1])  # 帧间相似度
split_points = top_k_lowest(similarities, S-1)  # S-1 个低相似度分割点
scenes = segment(video_frames, split_points)     # 分割为 S 个场景(S≤24)

# === Per-Scene Encoding (sliding window length N) ===
for scene in scenes:
    # 首帧静态特征（完整保留，144 visual + 50 audio tokens）
    F_x1 = SigLIP(scene[0]);  F_a1 = BEATs(audio[0])

    # AvgPool 首帧 visual tokens 得 K=16 个 query tokens
    Q = AvgPool(F_x1)  # shape: (16, D)

    # 后续帧动态压缩
    for i in 2..N:
        F_xi = SigLIP(scene[i]);  F_ai = BEATs(audio[i])
        # Q-Former cross-attention + instruction text F_s
        F_Q_i = QFormer(Q, [F_xi · F_ai], F_s)  # 压缩为 16 tokens

    # 场景最终表示: 静态 tokens + 动态上下文
    F_TDC = [F_x1 · F_a1 · <Sep> · F_Q_2 · F_Q_3 · ... · F_Q_N]

# === LVCoT for Extremely Long Videos ===
segments = divide_equally(video, M)   # M=3 by default
thoughts = []
for seg in segments:
    ans = LLM(F_TDC(seg), question)
    thoughts.append(f"From {seg.t_start}s to {seg.t_end}s: {ans}")

final = LLM(F_TDC(full_video), question, prev_thoughts=concat(thoughts))
```

  - 训练策略：三阶段训练，每阶段 1 epoch。Stage 1/2 全参数训练，Stage 3 用 LoRA 减少显存。优化器 AdamW，LR 1e-5 (stage 1/2) / 2e-5 (stage 3)，cosine decay，warmup ratio 0.03，max sequence length 8192。Visual/audio encoder 全程冻结，仅训练 temporal compressor 和 LLM。Q-Former 由 pre-trained BERT 初始化。

## PEARL__Personalized_Streaming_Video_Understanding_Model

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：PEARL —— 一个**training-free、plug-and-play**的个性化流式视频理解框架，由两大核心组件构成：
  (1) **Dual-grained Memory System（双粒度记忆系统）**：
    - **Streaming Memory（流式记忆）**：对连续视频流使用 PySceneDetect 检测场景边界（基于 HSV 色彩空间像素变化的快速剪辑检测，阈值 27.0，最小片段 1s，最大片段 8s），分段为有序 clip 序列 V = [X1, X2, ...]。对每个新增 clip Xi，使用 Qwen3-VL-Embedding-2B 多模态嵌入模型计算嵌入 e^i = femb(Xi)，存储 (Xi, ei) 对用于后续检索。视频以 1 FPS 采样。
    - **Concept Memory（概念记忆）**：当用户在时间戳 tc 发出 Concept-Definition QA 时，从当前 clip X^tc 提取视觉证据（frame-level 取最后一帧，video-level 取整个 clip），用基础 VLM 生成紧凑的概念描述（frame-level 聚焦永久/稳定特征如性别、面部、发型、体型；video-level 聚焦核心运动学如身体运动、动作序列、涉及的身体部位），将概念名、视觉证据和文本描述三者结构化存储。
  (2) **Concept-aware Retrieval Algorithm（概念感知检索算法）**：
    - 查询重写（Query Rewriting）：识别用户查询 Q 中出现的概念名，从 Concept Memory 检索对应的概念描述，将 Q 中的概念名替换为描述文本得到重写查询 Q̃
    - 嵌入检索：计算 e^Q = femb(Q̃)，与 Streaming Memory 中所有 clip 嵌入 {ei}i≤tq 计算余弦相似度
    - Top-K 选择 + 邻接扩展：选择相似度最高的 K=4 个 clip，每个 clip 扩展其相邻 N=1 个 clip（frame-level, video-level 用 N=0）以捕获局部时序上下文
    - 最终将检索到的概念条目、历史 clip、当前 clip X^tq 和原始查询 Q 送入 VLM 生成响应
  PEARL 无需任何参数更新，无缝适配多种 VLM 架构（LLaVA-OV-7B、Qwen2-VL-7B、Qwen3-VL-8B）。

  实验比较：
  (a) **PEARL-Bench 主实验（Table 3）** —— Frame-level（Real-Time + Past-Time + Avg）和 Video-level（Real-Time）指标。对比 8 个方法：
    - Proprietary Offline: Gemini3-pro-preview (64 frames)
    - Open-source Offline: LLaVA-OV-7B, Qwen2-VL-7B, InternVL3.5-8B, Qwen3-VL-8B (all 64 frames)
    - Open-source Online: ReKV(LLaVA-OV-7B, 0.5fps), StreamForest-7B(1fps), TimeChat-Online-7B(1fps)
    - PEARL variants: LLaVA-OV-7B+PEARL, Qwen2-VL-7B+PEARL, Qwen3-VL-8B+PEARL (all 1fps)
    - Human Score 上界: 97.61% Real-Time, 96.45% Past-Time, 97.49% Video-level Real-Time
    - Text-only 下界 (Qwen3-VL-8B): 11.06% Real-Time, 17.45% Past-Time
    - PEARL 将 Qwen3-VL-8B 的平均帧级准确率从 28.77% 提升至 52.24% (+23.47%)，超过 Gemini3-pro-preview (48.19%)。
  (b) **消融实验（Table 4）** —— 在 Qwen3-VL-8B 上逐组件启用：Text-only (14.26%) → +Current Clip (18.07%) → +Concept Memory (38.42%, Real-Time 猛增 35.57%) → +Streaming Memory (47.96%, Past-Time 猛增 20.26%) → +Query Rewriting = Full PEARL (52.24%)。
  (c) **效率对比（Table 5）** —— 端到端推理延迟：
    - LLaVA-OV-7B: 670ms (64f, 29.48% Avg)
    - LLaVA-OV-7B+PEARL: 775ms (1fps, 38.03% Avg) —— 仅增加 105ms 延迟换取 8.55% 精度提升
    - Qwen3-VL-8B+PEARL: 2111ms (1fps, 52.24% Avg)
    - 延迟分解（Fig.5）：PEARL 核心模块（Concept Retrieval + Query Rewriting + Streaming Memory Retrieval）延迟极低且跨模型恒定，VLM 推理占主导。
  (d) **超参数分析（Fig.4）** —— Past-Time QA 的 Top-K 和邻接扩展 N：K=0 时无法访问历史证据性能极低；K≥3 后趋于饱和。N=1 与 N=2 差距很小。默认 K=4, N=1。
  (e) **模型规模实验（Table 7）** —— Qwen2-VL 系列 (2B, 7B) 和 Qwen3-VL 系列 (4B, 8B) 加/不加 PEARL。PEARL 在所有规模上稳定提升：Qwen3-VL-4B+18.00%, Qwen3-VL-8B+23.47%, Qwen2-VL-7B+9.36%, Qwen2-VL-2B+4.17%。离线模型增大规模无显著收益（范式错配），加 PEARL 后大规模模型优势才得以释放。

- 硬件平台是什么，配置是什么。
  所有实验在 **NVIDIA H200 GPU** 上进行。基础 VLM：LLaVA-OV-7B、Qwen2-VL-7B、Qwen3-VL-8B。嵌入模型：Qwen3-VL-Embedding-2B。场景检测：PySceneDetect（检测阈值 27.0，最小 1s/最大 8s clip）。视频流以 1 FPS 采样。评估策略：循环选项旋转（每个多选题评估 4 次旋转正确选项位置 A/B/C/D，4/4 正确才算正确），消除选项位置偏差。

- 模型是什么。数据集和bench分别是什么。
  模型：PEARL 是 training-free 框架，可适配三种 VLM 架构：
  - LLaVA-OV-7B + PEARL
  - Qwen2-VL-7B + PEARL
  - Qwen3-VL-8B + PEARL
  嵌入模型：Qwen3-VL-Embedding-2B（编码视觉描述和视频 clip 到统一特征空间，用于余弦相似度检索）。
  数据集与 benchmark：**PEARL-Bench**（论文自建，首个 PSVU benchmark）：
  - 总计 132 个视频（Frame-level 112 + Video-level 20），平均时长 1458 秒
  - 2173 条精细标注，均带精确时间戳
  - Frame-level 划分：Concept-Definition QA 418 + Real-Time QA 922 + Past-Time QA 394 = 1734
  - Video-level 划分：Concept-Definition QA 80 + Real-Time QA 359 = 439
  - 视频来源：动漫、电影、真人秀（frame-level）+ Mixamo 数字人合成（video-level，8 角色 × 20 动作 × 20 背景随机组合）
  - 概念名：从 U.S. SSA 数据库随机选取 10k 常用名替换原名，防止先验知识泄露
  - 质量控制：自动过滤（消融法检测 trivial 问题）+ 10 位研究者人工审查
  - Real-Time QA 包含 6 个子任务：Presence, Behavior, Appearance, Location, Relation, Action
  - Past-Time QA 包含 2 个子任务：Event-based, Time-based

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  代码开源：https://github.com/Yuanhong-Zheng/PEARL，CC-BY 4.0 许可证。
  仓库包含：`clip_memory.py`（片段/记忆管理）、`concept_database.py`（概念存储）、`concept_desc.py`（概念描述生成）、`video_scene_splitter.py`（视频场景分割）、`video_qa_inference.py`（视频QA推理）、`eval.py`（评估指标聚合）、`server/`（VLM和嵌入服务器启动脚本，支持 Qwen3-VL-8B、Qwen3-VL-Embedding-2B、LLaVA-OneVision）、`scripts/`（多GPU评估pipeline shell脚本）、`third_party/`（qwen-vl-utils, Qwen3-VL-Embedding 等）。

  算法 pipeline 伪代码：

```
# === 初始化 ===
StreamingMemory = {}          # {clip_id: (clip_Xi, embedding_ei)}
ConceptMemory = {}            # {concept_name: (visual_evidence, description)}
embedding_model = Qwen3-VL-Embedding-2B
vlm = Qwen3-VL-8B (or LLaVA-OV-7B, Qwen2-VL-7B)

# === 流式视频处理循环 ===
for each arriving video clip X^t at timestamp t:
    # 场景检测与分段（PySceneDetect, HSV delta threshold=27.0）
    if scene_boundary_detected(X^t):
        e^t = embedding_model.encode(X^t)   # 多模态嵌入, 1fps采样
        StreamingMemory.append((X^t, e^t))

    # 解析用户指令
    if instruction is ConceptDefQA(concept_name, concept_type):
        # 概念注册
        if concept_type == "frame-level":
            visual_evidence = X^t.last_frame   # 取最后一帧
        else:  # video-level
            visual_evidence = X^t               # 取整个clip
        description = vlm.generate_concept_desc(visual_evidence, concept_name)
        # 生成描述：frame-level聚焦永久特征(性别/面部/发型/体型)
        #           video-level聚焦核心运动学(身体运动/动作序列/涉及部位)
        ConceptMemory[concept_name] = (visual_evidence, description)

    elif instruction is Query(Q):
        # === Concept-aware Retrieval ===
        # Step 1: Concept Retrieval —— 识别Q中的概念名并检索描述
        mentioned_concepts = extract_concept_names(Q, ConceptMemory.keys())
        C_sub = {c: ConceptMemory[c] for c in mentioned_concepts}
        replacement_rules = {c: desc for c, (_, desc) in C_sub.items()}

        # Step 2: Query Rewriting —— 将概念名替换为视觉描述文本
        Q_tilde = vlm.rewrite_query(Q, replacement_rules)
        # e.g., "What is Adaliz doing?" →
        #       "What is a young female with long black hair doing?"

        # Step 3: Streaming Memory Retrieval —— 余弦相似度匹配
        e_Q = embedding_model.encode(Q_tilde)  # shape: [d_embed]
        similarities = {i: cosine_sim(e_Q, e_i)
                        for i, (_, e_i) in StreamingMemory.items() if i <= t}
        top_K_clips = top_k(similarities, K=4)  # K=4 frame, 4 video

        # Step 4: Adjacent Expansion —— 扩展邻接clips捕获时序上下文
        V_context = top_K_clips ∪ {adjacent_clips(c, N=1) for c in top_K_clips}
        # N=1 for frame-level, N=0 for video-level

        # Step 5: VLM Response —— 组装上下文并推理
        response = vlm.generate(
            concepts=C_sub,
            historical_clips=V_context,
            current_clip=X^t,
            query=Q
        )
        return response
```

  张量计算流程（以 Qwen3-VL-8B+PEARL 为例）：
  - 视频 clip Xi → Qwen3-VL-Embedding-2B → embedding ei ∈ R^d_embed
  - 重写查询 Q̃ → embedding e^Q ∈ R^d_embed
  - cos(e^Q, ei) = e^Q · ei / (||e^Q|| · ||ei||)，排序取 Top-4
  - 检索到的历史 clips Vcontext + 当前 clip X^tq + 概念描述 Csub + 原始查询 Q → VLM tokenizer → [visual_tokens; concept_text_tokens; query_tokens] → VLM decoder → 生成答案 A

  关键设计要点：
  - 概念描述与 clip 嵌入在同一特征空间（Qwen3-VL-Embedding-2B），保证检索一致性
  - Query Rewriting 将个性化名称转换为嵌入模型可理解的描述性语义，是检索质量的关键（消融显示 +4.28% Avg）
  - Streaming Memory 持续的增量归档 + 概念级精确检索，区别于传统在线模型的固定大小状态压缩

## Representation_Shift__Unifying_Token_Compression_with_FlashAttention

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：提出 Representation Shift（表示漂移）作为训练无关、模型无关的 token 重要性度量。核心公式为 s = ||MLP(LN(x')) - x'||₂，即对 token 经过 MLP 层的表示变化量（L2距离）进行量化，变化大的 token 被认为更重要，变化小的 token 被剪枝。此方法不需要 attention map，可与 FlashAttention 无缝集成。剪枝策略：视频任务中在前3层每层逐步减少 token 20%/10%；图像任务中在 [1,4,7] 层剪枝 20% token。扩展到 CNN（ResNet）时通过行/列级剪枝实现，扩展到 SSM（Vision Mamba）时替换激活值基重要性分数。
  
  实验比较：
  (a) Video-text retrieval（Table 2）：UMT-B/L + Attn（基于 attention map 的剪枝）vs UMT-B/L + Ours（representation shift + FlashAttention），7个 benchmark（MSRVTT, MSVD, ActivityNet, DiDeMo, LSMDC, SSV2-Label/Template）。Ours 实现 5.47×/5.50× speedup（UMT-B/L），Attn 仅 1.78×/1.91×。R@1 指标上 Ours 平均高于 Attn +7.2%（UMT-L）。
  (b) 与 vid-TLDR 结合（Table 3）：将 vid-TLDR 的 attention-based importance 替换为 representation shift + FlashAttention，平均 3.74×/3.67× speedup（UMT-B/L），性能几乎不变。
  (c) Video QA（Table 4）：MSRVTT-QA, MSVD-QA 上 UMT-B/L + Ours 实现 4.00×/3.83× throughput 提升。
  (d) Image classification（Table 5）：DeiT-T/S/B + ImageNet1K，Ours 比 Attn-based 准确率高 +2.8%/+5.7%/+2.7%，同时吞吐量更高。
  (e) CNN（Table 6）：ResNet-34/50 + ImageNet1K，Line-wise/Token-wise 两种剪枝，ResNet-50 Line-wise 准确率 76.4% vs Base 76.1%，吞吐量 3553 vs 2927 img/s。
  (f) SSM（Table 7）：ViM-T + ImageNet1K，Ours 准确率 75.5% vs Top-ViM 75.1%。
  (g) 消融实验（Figure 5）：操作选择——MLP vs Attention vs Entire Block，MLP 最优；距离度量——L2 vs L1 vs Cosine，L2 最优。
  (h) 可靠性分析（Table 8）：top 50% vs bottom 50% token，平均准确率差 26.3%，验证 representation shift 的有效性。

- 硬件平台是什么，配置是什么。
  GPU：单张 NVIDIA RTX A6000（用于 throughput 测量和评估）。所有实验在单 GPU 上完成。

- 模型是什么。数据集和bench分别是什么。
  模型：UMT-B, UMT-L（Video Transformer，vanilla attention）；DeiT-T, DeiT-S, DeiT-B（Vision Transformer）；ResNet-34, ResNet-50（CNN）；ViM-T（Vision Mamba / SSM）。
  数据集/Benchmark：视频——MSRVTT, MSVD, ActivityNet, DiDeMo, LSMDC, SSV2-Label, SSV2-Template（video-text retrieval，报告 R@1/R@5/R@10 + harmonic mean of V2T/T2V）；MSRVTT-QA, MSVD-QA（video question-answering，报告 accuracy）；图像——ImageNet-1K（image classification，报告 Top-1 accuracy + throughput + GFLOPs）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源：https://github.com/mlvlab/Representation-Shift（MIT License, ICCV 2025）。

  算法pipeline伪代码：
  ```
  # 输入: 多帧视频 tokens x ∈ R^(T×H×W, C)
  # 超参: drop_layers = [0,1,2] (video) 或 [1,4,7] (image)
  #       drop_ratio = 0.2 或 0.1
  
  for layer_idx in range(num_layers):
      # Step 1: LayerNorm + MLP
      x_norm = LayerNorm(x)                    # [N, C]
      x_mlp = MLP(x_norm)                      # [N, C]
      
      if layer_idx in drop_layers:
          # Step 2: 计算 representation shift
          delta_x = L2_norm(x_mlp - x, dim=-1)  # [N]，每个 token 的 L2 距离
          
          # Step 3: 选择保留的 token
          num_keep = N * (1 - drop_ratio)
          keep_indices = topk(delta_x, k=num_keep)  # 保留 rep shift 最大的 token
          
          # Step 4: 剪枝
          x = x[keep_indices]                    # [N*(1-r), C]
      
      # Step 5: 残差连接（MLP）
      x = x + x_mlp[keep_indices]  # 或 x + x_mlp（非剪枝层）
      
      # Step 6: Self-Attention with FlashAttention
      x = x + FlashAttention(LayerNorm(x))       # [N', C]
  ```

  张量计算流程（以 UMT-B, 12 frames × 224² 为例）：
  - 输入视频 tokens x ∈ R^(L=12×14×14=2352, C=embed_dim)
  - 第0层（pruning layer, 20%）：
    - x' = x + FlashAttention(LN(x)) → x' ∈ R^(2352, C)
    - Δ = ||MLP(LN(x')) - x'||₂ → Δ ∈ R^2352
    - Top-K indices (K=1881) ← 保留前80%
    - x = x[indices] + MLP(LN(x'))[indices] → x ∈ R^(1881, C)
  - 第1层（pruning layer, 20%）：
    - 同样流程 → x ∈ R^(1505, C)
  - 第2层（pruning layer, 20%）：
    - 同样流程 → x ∈ R^(1204, C)
  - 后续层：token 数保持 1204 不变，正常 Transformer block
  - 最终 1204 tokens → task-specific head

  关键设计要点：
  - 在 MLP 层计算 representation shift（而非 Attention），因 MLP 逐 token 独立操作，产生的 representation shift 更具判别性
  - 使用 L2 距离（而非 L1 或 cosine），在所有层级上最鲁棒
  - Token 数在早期层逐步减半，后续层保持不变，保留核心特征
  - 与 FlashAttention 兼容：剪枝决策不依赖 attention map，仅依赖 token 本身的表示变化
  - 训练无关（training-free）：直接加载预训练模型，无需额外训练

## SCOPE: Saliency-Coverage Oriented Token Pruning for Efficient Multimodal LLMs

- 属于算法pipeline的实现是什么？实验比较什么？
  提出一种免训练的视觉 token 剪枝算法 SCOPE（Saliency-Coverage Oriented token Pruning），在 MLLM 的 vision encoder 之后、LLM 之前插入剪枝模块，联合建模 token 的显著性（saliency）和语义覆盖度（coverage），通过迭代贪心选择最大化 SCOPE score 的 token 子集来替代原始全量 visual token。实验比较不同 token 保留数量（192/128/64 for LLaVA-1.5，640/320/160 for LLaVA-Next）下各方法相对原始完整模型（Upper Bound）的性能保持率，baseline 包括 FastV、SparseVLM、VisionZip、PDrop、DivPrune。

- 硬件平台是什么，配置是什么。
  4 × NVIDIA A100 GPU。推理 batch size 设置为 1。

- 模型是什么。数据集和bench分别是什么。
  模型：LLaVA-1.5 7B / 13B、LLaVA-Next 7B / 13B（图像理解），Video-LLaVA（视频理解），Qwen2-VL 7B。
  图像 benchmark：GQA (testdev_balanced_instructions, 12578 samples)、MMBench (~3000 MCQs)、MME (dev split, 4377 samples)、POPE (test split, 9000 samples, F1 score)、ScienceQA、TextVQA (test split, 5000 samples, EM)、SEEDBench (19000 MCQs)、MMVet (test split, 218 samples, GPT evaluator)、DocVQA、ChartQA、OCRBench。
  视频 benchmark：TGIF、MSVD、MSRVTT、ActivityNet。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  代码开源：https://github.com/kinredon/SCOPE。基于 lmms-evals 评估框架实现。

  算法 pipeline：
  1. Vision Encoder（如 CLIP ViT-L/14）将输入图像编码为 N 个 visual token V = {v_1, ..., v_N} ∈ R^d（LLaVA-1.5: N=576, LLaVA-Next: N=2880/5图）。
  2. 从 vision encoder 倒数第二层（layer -2）提取 CLS token 对各 visual token 的 attention score A_v 作为 saliency。
  3. 预计算所有 token 对之间的 cosine similarity 矩阵 S_{uv} = sim(u, v) = u^T v / (||u||·||v||)。
  4. 初始化空集 S = ∅，coverage scores c_u = 0 ∀u ∈ V。
  5. 迭代 K 次（K 为保留 token 数）：
     a. 对每个候选 token v ∈ V\S，计算 marginal gain: Δ(v; S) = Σ_{u∈V} max(S_{uv}, c_u) - c_u
     b. 计算 SCOPE score: Δ(v, A_v^α; S) = Δ(v; S) · A_v^α
     c. 选择 SCOPE score 最高的 token v*，S = S ∪ {v*}，更新 c_u = max(c_u, S_{uv*})
  6. 将选出的 K 个 visual token S 与 text token 拼接后送入 LLM 进行 autoregressive 生成。
  7. 缩放因子 α 默认为 1.0。

  与 saliency-only（仅 Top-K attention）对比：SCOPE 额外计算 token 间相似度矩阵（O(N^2) 存储），并通过迭代贪心选择（O(K·N^2) 时间）替代单次排序。在 LLaVA-1.5 7B 上 K=64（↓88.9%）时仍保持 96.0% 平均性能，saliency-only baselines 最强者 VisionZip 仅 93.5%。在 LLaVA-Next 7B K=160（↓94.4%）时保持 95.1% 性能，VisionZip 仅 92.5%。

## Scaling the Long Video Understanding of Multimodal Large Language Models via Visual Memory Mechanism (FlexMem)

- 属于算法pipeline的实现是什么？实验比较什么？
  FlexMem 是一种训练无关（training-free）的视觉记忆机制，通过双路径压缩（Dual-Pathway Compression, DPC）对 MLLM 的视觉 KV cache 进行迭代式压缩和记忆管理，将压缩后的 local memory 写入 memory bank，并在问答时通过记忆召回（memory recall）选出最相关片段用于解码解答。实验比较了：(1) 与 VideoRAG 方法（AKS）和视觉压缩方法（AdaRETAKE、Video-RAG、BOLT、Panels、DToMA）在相同 MLLM backbone 下的性能；(2) 与 SOTA Video-MLLM（GPT-4o、Gemini-1.5-Pro、Qwen2.5-VL、Video-XL、TSPO、LongVU 等）的跨模型性能；(3) 在 OVOBench streaming QA 场景下使用 MemIndex 的在线 vs 离线性能对比；(4) 消融实验验证双路径压缩策略、context memory 作用、memory reading 策略以及每 clip 帧数的影响；(5) 在单张 3090 24GB 显存受限条件下与 AKS 和 AdaRETAKE 的可扩展性对比。

- 硬件平台：单张 NVIDIA RTX 3090 GPU (24GB) 为主实验平台；部分对比实验在一张 NVIDIA A800 上运行（Table 1 中 FlexMem* 标记）。所有受限实验均在 24GB 显存上限下完成。

- 模型：LLaVA-Video 7B（13k input tokens）和 LLaVA-OneVision 7B（7k input tokens）作为 base MLLM。数据集和 benchmark：MLVU（多任务长视频理解，含 single-detail/multi-detail/holistic 子类）、LongVideoBench（长上下文多模态推理，视频最长 1 小时）、LVBench（极端长视频理解，平均时长 68.4 分钟）、Video-MME（短视频/中视频/长视频混合）、TimeScope（1 分钟到 8 小时超长时间跨度）、OVOBench（在线流式视频问答，含 EPM/ASI/HLD backward tracing 子任务）。

- 开源情况：已开源，代码地址 https://github.com/city1517/FlexMem 。FlexMem 为 training-free 即插即用方法，直接应用于任意 Video-MLLM 无需额外训练。

- 算法 pipeline 详细解释（基于论文 Section 3 Method）：

  **核心思想**：模拟人类看视频行为——持续观看、形成记忆、问答时召回相关记忆片段。迭代处理视频分片，理论上支持无限长视频。

  **步骤 1 — 视频分片与帧采样**：
  将长视频 V 均匀分片为 N 个 clips {V1, ..., VN}。每 clip 固定 8 帧（消融实验确定），总采样帧数 512（TimeScope/LVBench/MLVU）或 1024（Video-MME/LongVideoBench）。对比 baseline 仅 32 或 64 帧均匀采样。

  **步骤 2 — 首次编码（Clip V1）**：
  ```
  Input: V1 (8 frames → visual encoder → KV caches), optional Tq
  Output: M1 (local memory → M_bank), C1 (context memory → next iter)
  MLLM(V1, <Tq>) → M1, C1
  ```

  **步骤 3 — 迭代编码（Clip Vi, i ≥ 2）**：
  ```
  MLLM(<Ml>, C_{i-ns}, ..., C_{i-1}, Vi, <Tq>) → Mi, Ci
  ```
  输入包括：长期记忆 `<Ml>`（从 M_bank 召回，可选）+ ns 个前序 context memory（确保时序连续性）+ 当前 clip Vi + 可选 Tq。输出 Mi（写入 M_bank）和 Ci（传给下一轮）。

  **步骤 4 — Dual-Pathway Compression（每步核心张量计算）**：

  在 MLLM 每层 l，计算当前 clip Vi 对历史 context C 的 cross-clip attention：
  ```
  A_v^l = softmax(Q_{Vi} · K̂_C^T / √d)  ∈ R^{|Vi| × |C|}
  ```

  **(a) Context Memory 路径** — 选择最能聚合历史信息并传播给后续 token 的 KV（服务于 prefill 阶段）：
  ```
  s_j^l = Σ_{k∈C} a_{jk}^l + Σ_{h∈Vi} a_{hj}^l
  ```
  第一项：token j 从历史 context 聚合的信息量。第二项：token j 对后续 token 的因果传播量。
  ```
  c_i^l = {k_j^l, v_j^l | j ∈ argtopK(s_j^l, α_c·|Vi|)}
  ```
  Ci = {c_i^1, ..., c_i^L}——所有 L 层的 context features，用于下一轮迭代的信息传递。

  **(b) Local Memory 路径** — 选择 clip 内最具显著性的 token（服务于 decoding 阶段）：
  ```
  ŝ_j^l = Σ_{k∈Vi} a_{kj}^l
  ```
  仅考虑 clip 内部影响力。
  ```
  m_i^l = {k_j^l, v_j^l | j ∈ argtopK(ŝ_j^l, α_s·|Vi|)}
  ```
  Mi = {m_i^1, ..., m_i^L}——存储到 M_bank 供最终召回。

  **DPC 设计动机**：prefill 阶段目标是将当前 clip 编码进丰富历史上下文（需要 context aggregation），decoding 阶段目标是基于最显著的视觉证据回答问题（需要 local saliency）。两个阶段对 KV cache 的需求不同，因此用两种不同的重要性度量分别压缩。

  **步骤 5 — Memory Recall（记忆召回）**：

  **方式 A: Encoding-based Reading**（精度高，需 encoding 时传入 Tq）：
  ```
  g_i = Σ_{l=3→L} Σ_{j∈Tq} Σ_{k∈Vi} a_{jk}^l
  Recall(M_bank, Tq) = {Mi | g_i ∈ argmax, 取 top na 连续 clip}
  ```
  仅取第 3 层及以后深层 attention（浅层 attention 分布均匀无区分力）。

  **方式 B: MemIndex（快速索引，独立于 memory encoding）**：
  目标——线性回归拟合 encoding-based relevance：
  ```
  min_σ Σ ||σ(r̂_i) - g_i||², σ(r̂_i) = Σ_{l=3→L} α^l·r̂_i^l
  ```
  - 问题编码：q = Q_{Tq}[-1]（最后一个 token 的 query embedding）
  - 视觉索引：从 Mi 取 top k 个 attention-selected key vectors K_{Vi}^*
  - 在选定 H 层上计算点积 attention：r̂_i = Σ_{l∈H} Σ_j Attention(q, k_j^l)
  - 学习层权重 α^l，选 top K=3 层计算最终 relevance

  MemIndex 独立于 memory encoding，适合多问题或 streaming 场景。

  **步骤 6 — 答案解码**：
  ```
  MLLM(M_i, ..., M_{i+na-1}, Tq) → Y
  ```
  仅用召回的 na 个 memory 片段 + Tq 解码。LLaVA-Video 用 13k tokens，LLaVA-OV 用 7k tokens，远小于 AdaRETAKE 的 40k。

  典型性能：单 3090 上 FlexMem+LLaVA-Video 在 TimeScope 超 baseline 32.2%，LVBench 超 19.7%。512/1024 帧采样 vs baseline 64 帧，五个 benchmark 全面领先 AKS 和 AdaRETAKE。24GB 受限下仅损失 0.5% 性能。

## See It, Say It, Sorted: An Iterative Training-Free Framework for Visually-Grounded Multimodal Reasoning in LVLMs

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：ECRD（Evidence-Constrained Reweighting Decoding），一种 training-free、plug-and-play 的推理时解码框架，不修改模型权重，仅在 test time 介入 token selection 过程。包含两个核心组件：(1) **Distribution Supervisor（分布监督器）**：维护一个文本证据池（textual evidence pool），对 base LVLM 的 top-k 候选 token 计算证据诱导分布 r_i(w)，然后与 base 分布 p_i(w) 通过自适应权重 α_i = p_{(1)}（base 模型 top-1 概率）进行协商混合，得到 p_i^{mix}(w)。当 base 模型置信度高时（p_{(1)} 大），保持 base 分布主导；当 base 分布分散时（p_{(1)} 小，hallucination-prone），证据权重增大。(2) **Visual Decider（视觉裁决器）**：由 GRIT（基于 Qwen2.5-VL-3B 的视觉定位模型）实例化，当协商分布的 margin Δ_i = p_{(1)}^{mix} - p_{(2)}^{mix} ≤ δ 且 k* > 1 时触发，读取图像和当前推理前缀，输出一个候选 token w* 和一条人类可读的微观察证据句 E_i，强制提交 w* 并将 E_i 追加到证据池。证据池仅包含文本（坐标仅用于可解释性，不参与 scoring），避免反复编码图像裁剪。

  实验比较：
  (a) TreeBench 上 vs base LVLMs（Qwen2.5-VL 7B/32B/72B、LLaVA-OneVision 7B/72B、InternVL3 8B/38B/78B），ECRD 在各 backbone 和 scale 上一致提升 +4.5~+10.9 个点 overall accuracy；(b) vs RL-based 视觉定位推理模型 DeepEyes-7B、Pixel-Reasoner-7B、TreeVGR-7B；(c) vs training-free baselines Woodpecker、ViperGPT、ControlMLLM、beam search、self-consistency、diverse sampling；(d) vs VDGD（Visual Description Grounded Decoding，ECRD 的前身方法）；(e) RH-Bench 上 Reasoning/Perception/RH-AUC；(f) V*Bench、MathVista、ChartQA、OCRBench、HallusionBench 五个通用多模态 benchmark；(g) Ablation: base only vs +VDGD vs +supervisor vs +supervisor+Qwen2.5-VL-3B decider vs +supervisor+GRIT-3B decider (full ECRD)；(h) 不确定性阈值 δ 的 cost-accuracy trade-off 分析（Fig. 3）；(i) 定性分析：supervisor 重新加权解决歧义 vs visual decider 中链注入视觉证据 vs visual decider 直接输出最终答案。

- 硬件平台是什么，配置是什么。
  单张 NVIDIA H20-NVLink GPU。所有测试在 H20 上进行：t_0（δ=0 时每问题平均时间）在 V*Bench 8.98s、MathVista 12.92s、ChartQA 9.76s、OCRBench 3.24s、HallusionBench 11.67s。l_0（单次 visual decider 调用全局平均延迟）在 1.12-1.46s 之间。Latency model: T(δ) ≈ t_0 + l_0 · r(δ)，其中 r(δ) 为每问题平均 decider 调用次数。

- 模型是什么。数据集和bench分别是什么。
  模型：Base LVLMs 包括 Qwen2.5-VL 系列（7B/32B/72B）、LLaVA-OneVision 系列（7B/72B）、InternVL3 系列（8B/38B/78B）。Visual Decider 为 GRIT-3B（基于 Qwen2.5-VL-3B，针对视觉定位优化）。Ablation 中还测试了 Qwen2.5-VL-3B 作为 decider 的变体。Base 模型和 decider 均冻结，不做任何 fine-tuning。Private model 参考线：GPT-4o、o3、Gemini-2.5-Flash、Gemini-2.5-Pro。

  Benchmarks：(i) TreeBench — 评估"thinking with images"，分为 Perception（Attr./Mater./Phys./ObjRet./OCR）和 Reasoning（Persp./Order./Cont.&Oc./Contain./Compar.），metric 为 answer accuracy；(ii) RH-Bench — 评估 Reasoning/Perception 和 RH-AUC（平衡推理长度和幻觉的指标）；(iii) V*Bench — 视觉搜索引导的多模态能力（Attr./Spatial/Overall）；(iv) MathVista — 视觉上下文中的数学推理；(v) ChartQA — 图表问答；(vi) OCRBench — OCR 能力；(vii) HallusionBench — 语言幻觉与视觉错觉诊断。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址：https://github.com/uuuuZYC/See-It-Say-It-Sorted

  ECRD 算法 pipeline 伪代码（以单步解码为例）：

  ```
  # ===== 初始化 =====
  E_0 = {d_global}  # 证据池初始化为全局图像描述
  prefix = [instruction_tokens, image_tokens]

  # ===== 逐步解码循环 =====
  for step i in 0..max_len:
      # 1. Base LVLM forward: 获取 next-token 分布
      logits_i = frozen_LVLM(prefix)  # shape: [vocab_size]
      p_i = softmax(logits_i)         # base 分布

      # 2. Knee truncation: 选择 top-k* 候选集
      p_sorted = sort(p_i, descending=True)
      k* = argmax_k (p_sorted[k] - p_sorted[k+1])  # 式(2)
      C_i = {w_1, ..., w_k*}  # 候选 token 集，式(3)

      # 3. 证据评分: 对每条证据计算 mean-over-prefix 概率
      for each evidence sentence E_j in E_i:  # E_j = (e_1,...,e_L)
          q_Ej(w) = (1/L) * sum_{t=1..L} p_VLM(w | e_{<t})  # 式(5)
          # 即 evidence sentence 每个 prefix 下 token w 的平均条件概率
      S_i(w) = -log( (1/N) * sum_{j} q_Ej(w) )  # 式(6)
      # 证据池中 N 条证据的平均支持度（log 空间）

      # 4. 证据诱导分布: 仅在 C_i 内归一化
      r_i(w) = softmax_{w in C_i}(-S_i(w))  # 式(7)

      # 5. 质量匹配缩放: r_i 的总 mass 匹配 p_i 在 C_i 内的 mass
      mass_p = sum_{w in C_i} p_i(w)
      mass_r = sum_{w in C_i} r_i(w)
      r_tilde_i(w) = r_i(w) * (mass_p / mass_r)  # for w in C_i，式(8)

      # 6. 协商混合: base + evidence 的自适应融合
      alpha_i = max(p_i)  # top-1 概率作为自适应权重，式(11)
      p_mix_i(w) = alpha_i * p_i(w) + (1-alpha_i) * r_tilde_i(w)  # w in C_i, 式(10)
      p_mix_i(w) = alpha_i * p_i(w)                          # w not in C_i

      # 7. 不确定性检测: 决定是否调用 visual decider
      margin = max(p_mix_i) - second_max(p_mix_i)  # 式(12)
      if k* > 1 and margin <= delta:  # delta=0.08
          # 触发 visual decider
          w_star, evidence_sentence = GRIT(image, prefix_tail, C_i)
          # GRIT 读图 + 当前文本前缀 + 候选集，输出最佳 token 和证据句
          x_i = w_star              # 强制提交 decider 选择的 token
          E_{i+1} = E_i ∪ {evidence_sentence}  # 追加到证据池，式(13)
      else:
          x_i = argmax(p_mix_i)     # 直接取混合分布 top-1
          E_{i+1} = E_i             # 证据池不变

      prefix = concat(prefix, x_i)  # 更新解码前缀

  # ===== 终止 =====
  # 当生成 EOS token 或达到 max_len 时停止
  # 最终 answer 从完整 prefix 中提取
  ```

  张量计算关键维度：
  - logits_i: [1, vocab_size]（如 Qwen2.5-VL-7B 的 vocab 约 152k）
  - C_i: [1, k*]，k* 由 knee truncation 动态决定，通常为个位数
  - q_Ej(w): 对每条证据句子 E_j 的每个 prefix 位置计算 p_VLM(w|e_{<t})，对 |E_i| 条证据取平均。证据评分 O(k*|E_i|)，由于 k* 为个位数且 |E_i| 增长缓慢，overhead 很小
  - alpha_i: 标量，p_{(1)} ∈ [0,1]
  - p_mix_i: [1, vocab_size]，仅在 C_i 内的 mass 被重新分配
  - visual decider 调用延迟: l_0 ≈ 1.12-1.46s/call（H20 GPU）
  - GRIT 输出: w* ∈ C_i（单个 token），E_i = 一句自然语言证据（约 20-50 tokens）+ 可选坐标标注

  关键设计理念：
  - **Training-free**: LVLM 和 GRIT decider 完全冻结，无需任何 fine-tuning/RL/偏好优化
  - **Cost-aware**: visual decider 仅在 k*>1 且 margin≤δ 时触发，δ=0.08 时 r(δ) 在低个位数，达到 accuracy-cost elbow
  - **Textual evidence**: 证据池仅存文本，无需反复编码图像裁剪，后续 token 可直接引用之前的微观察，形成跨步长 evidence reuse
  - **自适应权重 alpha_i = p_{(1)}**: 当 base 模型自信时（分布尖锐），alpha→1 保持 base 主导；当 base 模型犹豫时（分布平坦），alpha 小，证据权重增大，精准在 hallucination-prone 步干预

## SlowFast-VGen: Slow-Fast Learning for Action-Driven Long Video Generation

- 属于算法pipeline的实现是什么？实验比较什么？
  提出 SLOWFAST-VGEN，一种双速学习系统，包含：(1) Slow Learning——基于 ModelScopeT2V 的 masked conditional video diffusion model，对前序视频 chunk 和语言 action 做条件生成后续 chunk；(2) Fast Learning——推理阶段的 TEMP-LORA 模块，通过将输入输出 latent 拼接后加噪去噪训练 LoRA 参数，在参数中存储情节记忆；(3) Slow-Fast Learning Loop——内层 fast learning 循环适配每个 episode 并积蓄 TEMP-LORA 参数，外层 slow learning 循环利用多 episode 数据更新核心权重。

  实验比较：
  - 视频生成质量：FVD、PSNR、SSIM、LPIPS vs AVDC/Streaming-T2V/Runway Gen-3 Turbo/AnimateDiff/SEINE/iVideoGPT
  - 长视频一致性：SCuts (PySceneDetect 场景切换数)、SRC (Scene Revisit Consistency，回访场景余弦相似度)
  - 长时规划：RLBench 机器人操作（物体归位距离）和 Minecraft 游戏导航（到预定义路径点距离）的 Dist 与 FVD

- 硬件平台是什么，配置是什么。
  慢学习训练：约 64 张 V100 GPU，batch size 128。推理和快学习：单张 V100 GPU。训练时冻结 VAE 和 CLIP Encoder，仅训练 UNet。

- 模型是什么。数据集和bench分别是什么。
  模型：基于预训练 ModelScopeT2V（latent video diffusion + 3D UNet + 时空 attention blocks）修改。UNet 参数记为 Φ（slow learning weights），TEMP-LORA 低秩矩阵 Θ 作为 fast learning weights，最终权重 W' = Φ + Θ。Slow learning rate 5e-6，fast learning rate 1e-4，LoRA rank 32，Adam 优化器，context window 32 frames。
  数据集：自采集 200k 视频配语言 action 标注，涵盖 5 个领域：Unreal Engine（Google 3D Tiles/Unreal City Sample/购买素材，Python 脚本自动化 agent 控制）、Minecraft（手动游戏录制键盘鼠标）、EPIC-KITCHENS（第一人称厨房日常）、Robot（OpenX-Embodiment + Metaworld + RLBench）、Driving（HDD + Unreal Engine 生成）。测试集从自采数据集中预留。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源情况：论文未明确说明代码开源（项目网站 slowfast-vgen.github.io 有 Code 链接但截至检索时未公开）。

  算法 pipeline（三层结构）：

  **第1层 — Slow Learning (Masked Conditional Video Diffusion)**：
  给定 fp 帧过去帧和 fg 帧待生成帧：
  ```
  z_{t,:fp} = z_{0,:fp}                                   # 条件帧不加噪
  z_{t,fp:(fp+fg)} = sqrt(ᾱ_t)·z_{0,fp:(fp+fg)} + sqrt(1-ᾱ_t)·ε  # 生成帧加噪
  z_t = concat(z_{t,:fp}, z_{t,fp:(fp+fg)})               # 拼接送入 UNet
  loss = ||ε - ε_Φ(z_t[fp:(fp+fg)], t, c)||²              # 仅在后 fg 帧计算 loss
  ```
  条件 c 为 CLIP 编码的语言 action text。VAE 编码视频到 latent，VAE 冻结。

  **第2层 — Fast Learning (TEMP-LORA)**：
  推理时逐 chunk 生成，每轮迭代 i：
  ```
  // 生成当前 chunk
  Y_i = (Φ + Θ_i)(X_i, C_i)              # X_i 是上一轮输出，C_i 是当前 action
  
  // 训练 TEMP-LORA 存储情节记忆
  X_i' = concat(X_i, Y_i)                 # 拼接输入和输出 latent
  z_t^{i'} = sqrt(ᾱ_t)·X_i' + sqrt(1-ᾱ_t)·ε  # 全序列加噪（不保留 clean 条件帧）
  loss_Θ = ||ε - ε_{Φ+Θ_i}(z_t^{i'}, t)||²  # 全序列计算 loss，无文本条件
  Θ_{i+1} = Θ_i - α·∇_Θ loss_Θ           # 更新 LoRA 参数
  ```
  核心设计：抛弃原始 TEMP-LORA 的 input→output 格式，对拼接的全序列加噪去噪训练，强调记忆整个轨迹而非关注即时转换。

  **第3层 — Slow-Fast Learning Loop**：
  ```
  while not converged:
      D_s = ∅
      for each (x, episode) in D:
          初始化 Θ_0^e
          for i in 0..I-1:
              D_s = D_s ∪ {X_i^e, X_{i+1}^e, Θ_i^e}  # 收集 input/output/TEMP-LORA
              固定 Φ，更新 Θ_i^e（fast learning）
      for {X_i^e, X_{i+1}^e, Θ_i^e} in D_s:
          Φ_i^e = Φ + Θ_i^e
          基于 Φ_i^e(X_i^e) 和 ground-truth X_{i+1}^e 计算 loss
          固定 Θ_i^e，更新 Φ（slow learning）
  ```

  视频规划（Video Planning）：采用 UPDP 框架，将任务规划形式化为 text-conditioned video generation。ChatGPT 分解任务为子目标 → 每个子目标生成 video chunk → 逆动力学模型将连续帧转为可执行 action。

  关键张量维度（推断自论文描述）：
  - latent z_t: fp+fg 帧，每帧 latent 维度取决于 VAE 压缩率
  - LoRA ΔW = AB^T: A∈R^{m×r}, B∈R^{n×r}, r=32
  - 推理开销：wo TEMP-LORA 约 12.93s/sample, w TEMP-LORA 约 13.81s/sample（+6.8%）; 显存 9579MB vs 9931MB（+3.7%）
  - 可生成最长 1000 帧无明显失真和退化

## T2Vid__Translating_Long_Text_into_Multi-Image_is_the_Catalyst_for_Video-LLMs

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：提出 Sparrow 数据增强方法，将纯文本 instruction 数据转化为类视频的多图像序列（text-to-image synthesis），然后与真实视频数据混合训练视频-LLM。核心流程：(1) 从现有文本 instruction 数据集中取 (long-context, instruction, answer) 三元组；(2) 按词数（~115词/段）用 NLTK 将 long-context 分割为多段；(3) 每段用 Pillow ImageFont 渲染为 448×448 白底黑字图像（20pt Arial, 20px margin）；(4) 将合成图像序列与真实视频帧混合，统一格式后 fine-tune image-LLM。该方法旨在解决视频训练数据 instruction diversity 不足导致的数据效率低下问题。
  - 实验比较：
    - 数据缩放实验：不同数据量（0/10K/30K/60K/100K）下，纯视频数据 vs Sparrow 混合数据（视频:合成=2:1）的性能对比
    - 消融实验：30K 规模下，纯 ShareGemini、纯 Video-ChatGPT、各半混合、Sparrow（Video+合成）、Video+纯文本 五种数据组合的性能对比
    - 主流方法对比：在 Video-MME 上与 proprietary models (GPT-4V, GPT-4o, Gemini 1.5 Pro) 和 open-source video-LLMs (VideoChat2, VideoLLaMA 2, VITA, Kangaroo 等) 对比
    - 长视频理解：LongVideoBench、MLVU-M、Video-MME-Long 上评估长视频理解能力
    - 帧数泛化：推理时从 24 帧扩展到 48/128/256 帧的性能变化
    - 细粒度任务分解：Video-MME 上按 Perception (OCR/计数/物体识别等)、Cognition (时序推理/空间推理等)、Information Synopsis 三类任务性能分解

- 硬件平台是什么，配置是什么。
  - 论文未明确说明训练使用的具体 GPU 型号。给出 GPU hours 作为效率度量：全量 200K 视频数据训练需 276.8 GPU hours，Sparrow 30K 混合数据仅需 33.6 GPU hours，效率提升 8.2×。推测使用多卡训练（batch 含多视频帧）。环境依赖：conda Python 3.9 + PyTorch + Flash-Attention 2。

- 模型是什么。数据集和bench分别是什么。
  - 模型：Mini-InternVL-Chat-4B-V1.5（简称 InternVL，3.8B，最多 13 子图 patch，每子图 256 visual tokens）和 MiniCPM-Llama3-8B-V2.5（简称 MiniCPM-8B，最多 10 patch，每 patch 96 visual tokens）。均为预训练 image-LLM，通过 fine-tuning 扩展到视频理解。InternVL 训练时冻结 vision encoder，其余参数全量训练（lr=5e-6）。MiniCPM-8B 全量训练。训练时关闭动态分辨率 patchifying 选项。
  - 数据集：
    - ShareGemini-Webvid-core100k：100K 视频-描述对，源自 WebVid（开源短视频 <30s），caption 由 Gemini-1.5-Pro API 标注，经聚类去重
    - Video-ChatGPT：100K 视频-instruction 对，源自 ActivityNet，含视频摘要/内容问答/创造性生成三类 instruction，半自动标注（人工精炼 + GPT-3.5 辅助）
    - 文本数据源：LongAlpaca（5K）和 LongQLora（5K），各取 5K 样本合成
  - Benchmarks：
    - Video-MME：综合视频-LLM 评估，含短视频 (<2min)、中等 (4-15min)、长视频 (30-60min)，手动收集标注
    - MVBench：20 个视频任务，覆盖感知和认知（场景转换、情节推理等）
    - TempCompass：时序理解评估（动作、速度、属性变化），MCQ 格式
    - LongVideoBench：长上下文交错视频-语言理解
    - MLVU-M：多任务长视频理解

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源仓库：https://github.com/VITA-MLLM/Sparrow，arXiv: 2411.19951
  - 合成数据集：https://huggingface.co/datasets/xjtupanda/Sparrow-Synthetic
  - 环境搭建：`conda create -n sparrow python=3.9` → `pip install -r requirements.txt` → `pip install -U flash-attn --no-build-isolation`

  **算法 Pipeline 详解**：

  **Step 0 — 模型架构（标准 MLLM）**：
  ```
  输入视频 V 下采样到 T 帧: F = {f_i}_{i=1}^T
  逐帧特征提取: E = {E_i}_{i=1}^T = ViT(F),  E_i ∈ R^{(H×W)×C}
  投影到 LLM 空间: Ê = Proj(E),  Proj 为 MLP
  与文本 token 拼接: [w_V; w_T] → LLM → 自回归生成答案
  ```

  **Step 1 — Sparrow 数据合成**：
  ```
  输入: (long_context, instruction, answer) 三元组

  # 1. 按词数分割 long_context
  segments = nltk.word_tokenize(long_context)
  chunks = [segments[i:i+115] for i in range(0, len(segments), 115)]

  # 2. 每个 chunk 渲染为图像
  for chunk in chunks:
      text = " ".join(chunk)
      img = Image.new('RGB', (448, 448), 'white')
      draw = ImageDraw.Draw(img)
      font = ImageFont.truetype('arial.ttf', 20)
      # 逐行绘制，左右 margin 20px
      y = 20
      for line in wrap_text(text, width=408):  # 408 = 448 - 2*20
          draw.text((20, y), line, fill='black', font=font)
          y += font_height
      synthetic_frames.append(img)

  # 3. 输出: (synthetic_frames, instruction, answer) → 类 video 格式
  ```

  **Step 2 — 混合训练**：
  ```
  # 每个 batch 以 2:1 比例采样
  real_sample  = (video_frames, instruction, answer)  # 来自 ShareGemini/Video-ChatGPT
  syn_sample   = (synthetic_frames, instruction, answer)  # 来自 Sparrow

  # 视频帧处理（InternVL, max 64 frames, FPS=1）
  if len(real_frames) > max_frames:
      real_frames = uniform_downsample(real_frames, max_frames)

  # Vision encoder 编码（InternVL: 冻住, MiniCPM: 训练）
  real_visual_tokens = ViT(real_frames)  # shape: [T, H*W, C]
  syn_visual_tokens  = ViT(syn_frames)
  # Projector 投影
  real_embeds = Proj(real_visual_tokens)
  syn_embeds  = Proj(syn_visual_tokens)

  # 与文本拼接送入 LLM
  input_ids = concat([vision_embeds, text_embeds])
  logits = LLM(input_ids)
  loss = -log P(answer | video_frames, instruction)
  ```

  **关键张量维度**：
  - InternVL-4B: 每子图 → 256 visual tokens × 13 tiles max; 训练时关闭 patchify，video max 64 frames; 每帧 token 数 = 256（单 tile 模式）
  - MiniCPM-8B: 每子图 → 96 visual tokens × 10 tiles max; video max 24 frames; 每帧 token 数 = 96（单 tile 模式）
  - 合成图像: vision encoder 同样处理，448×448 白底黑字图像 → 经 ViT patch embedding → visual tokens
  - LLM backbone: InternVL 用 InternLM2, MiniCPM 用 LLaMA3-8B

  **训练配置**：
  - 学习率：5e-6，全量端到端训练（InternVL 冻 vision encoder）
  - 数据混合：视频数据 (ShareGemini:Video-ChatGPT = 1:1) + 合成数据 (LongAlpaca:LongQLora = 1:1)，视频:合成 = 2:1
  - 长视频推理：24 帧训练，48/128/256 帧推理可扩展（但超 LLM context 则性能崩溃）

  **效率对比**：
  | 配置 | 样本数 | GPU hours | Video-MME Overall |
  |------|--------|-----------|-------------------|
  | Full video data | 200K | 276.8 | 56.3 |
  | Sparrow 30K | 30K (20K video + 10K syn) | 33.6 | 56.7 |
  | 效率提升 | 15% 样本 | 8.2× faster | +0.4 pts |

## Speak_While_Watching__Unleashing_TRUE_Real-Time_Video_Understanding_Capability_of_Multimodal_Large_Language_Models

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：在 MLLM（Qwen2.5-VL）上修改位置编码策略，提出三种打破全局位置连续性的方法——Overlapped Streaming Position Encoding (OSPE)、Group-Decoupled Position Encoding (GDPE)、Gap-Isolated Position Encoding (GIPE)，使视觉感知（prefill）和文本生成（decode）可以并行执行，而非串行交替。核心是重新分配 token 的位置 ID 和对应的 causal mask，以在保持模型架构不变的前提下实现真正的实时并行流式推理。
  - 实验比较：在 Offline 和 Streaming 两种范式下，对比 Origin（原始 Qwen2.5-VL 连续位置编码）、Interleave（交替感知-生成的流式 baseline）、OSPE、GDPE、GIPE 在 Video Description（PE-Video 数据集）和 Video QA（FunQA 数据集）上的表现。还包含 scheduling disturbance 鲁棒性测试（训练 wait-K=3，测试 wait-K=Random）和 3B vs 7B 模型规模的扩展性分析，以及理论加速比分析。

- 硬件平台是什么，配置是什么。
  - 论文未明确说明训练/推理的具体 GPU 型号和数量。代码仓库（EIT-NLP/Speak-While-Watching）使用 Python 3.10 + PyTorch，可选 Flash-Attention 加速。理论加速分析中提及可使用两张 GPU 分别执行 prefill 和 decode 的并行流水线，但实际代码实现的是位置编码层面的并行设计，未实现真正的多 GPU 并行执行。

- 模型是什么。数据集和bench分别是什么。
  - 模型：Qwen2.5-VL（基于 Qwen2.5-VL-3B-Instruct 和 Qwen2.5-VL-7B-Instruct 两个变体），其采用显式三维位置编码（x, y, t）表示视觉 token 的空间结构和时间动态，文本 token 三维坐标保持一致。
  - 数据集与 Benchmark：
    - PE-Video Dataset（facebook/PE-Video）：高质量视频描述数据集，含丰富运动动态和人工精炼字幕，用于流式 Video Description 任务。过滤规则：视频 5-30 秒、2fps 采样、字幕 token 数/视频时长比值在 3-5 之间，随机选 20K 样本训练。
    - FunQA Dataset：流式 Video QA 任务，含 HumorQA、CreativeQA、MagicQA 三个子集，每个子集各有两个子任务类型（Description Q&A 和 Counterintuitive Reasoning Q&A），共 6 个子任务。要求模型基于连续到达的视频帧进行开放式描述性回答。
  - 评估指标：CIDEr、BLEU-1/4、METEOR、ROUGE-L、BLEURT（语义质量），以及基于 GPT-5 的 LLM-as-Judge 语言流利度评分（1-5 分）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源仓库：https://github.com/EIT-NLP/Speak-While-Watching（16 stars, 公开，无明确 License 声明）
  - 训练入口：`code/qwen-vl-finetune/scripts/sft.sh`，通过环境变量 `QWEN2_5_VL_VARIANT` 选择 variant（origin/batch/group/gap/overlap/interleave）
  - 评估入口：`eval.sh`，输出 JSON 结果到 `code/evaluation/output/${VARIANT}_infer.json`
  - 数据预处理：`mani_data.py` 从 HuggingFace 加载 PE-Video，按时长和 token 密度过滤

  **算法 Pipeline 伪代码（以 GDPE 为例）**：

  假设流式推理共 N 轮，第 i 轮输入 $m_i$ 个视觉 token $V_i$，生成 $k_i$ 个文本 token $A_i$。

  **常规 Interleave（baseline）—— 全局连续位置编码**：
  ```
  pos = 0
  for i in 1..N:
      # 编码视觉 token V_i
      for j in 1..m_i:
          PE(V_i[j], pos)   # 位置 pos, pos+1, ..., pos+m_i-1
          pos += 1
      # 生成文本 token A_i
      for j in 1..k_i:
          PE(A_i[j], pos)   # 位置 pos, pos+1, ..., pos+k_i-1
          pos += 1
  ```
  问题：A_i 生成完成前，V_{i+1} 的起始位置无法确定 → 必须串行交替。

  **GDPE（Group-Decoupled）—— 分组独立位置编码**：
  ```
  pos_v = 0  # 视觉组独立位置计数器
  pos_a = 0  # 文本组独立位置计数器
  for i in 1..N (视觉和文本可并行):
      # 视觉流（独立索引）
      PE(V_i, start=pos_v)
      pos_v += m_i
      # 文本流（独立索引）
      PE(A_i, start=pos_a)
      pos_a += k_i
  ```
  Causal Mask 配置（训练时完整序列已知）：
  - V_{i+1} 只能 attend 到 V_1 到 V_i（组内因果）
  - A_i 只能 attend 到 V_1 到 V_i 和 A_1 到 A_i（跨模态因果+组内因果）
  - 训练时通过自定义 causal mask 实现，推理时视觉 prefill 和文本 decode 可并行

  **关键张量维度**：
  - 视觉 token：每帧经 vision encoder (ViT) 输出 token 序列，经 projector 映射到 LLM 嵌入空间。Qwen2.5-VL 中每帧产生可变数量视觉 token（取决于图像分辨率和动态分辨率策略）。
  - 位置编码：3D RoPE，(x, y, t) 三维，文本 token 的三维坐标相同（t=0 或固定值）。GDPE 修改了位置 ID 分配规则——视觉组 t 维度按帧序号递增，文本组 t 维度按 token 序号递增，两组独立。
  - wait-K=3：每接收 1 帧生成 3 个 token（由 PE-Video 和 FunQA 的平均帧-文本比例 ≈3 确定），N 轮。
  - 2fps 采样，max 30s 视频，理论最大 60 帧/视频。

  **训练流程**：
  - 基于 Qwen2.5-VL 预训练权重进行 SFT fine-tuning
  - variant="group"(GDPE) 时修改位置 ID 分配逻辑和 causal mask
  - 输出 checkpoint 到 `code/qwen-vl-finetune/output/qwen2_5vl-pe-${VARIANT}/`
  - 仅需少量微调数据（20K 样本）

  **理论加速**：
  - 并行流式延迟：$T_{\text{parallel},i} = \max(m_i/R_v, k_i/R_t)$
  - 串行交替延迟：$T_{\text{interleave},i} = m_i/R_v + k_i/R_t$
  - 加速比上限 2×（当感知和生成负载均衡时，即 workload ratio $r \approx 1$）
  - Video Description 场景（$r \gg 1$，视觉主导）加速有限；Video-CoT 场景（$r \approx 1$）加速最大

## TSPO__Temporal_Sampling_Policy_Optimization_for_Long-form_Video_Language_Understanding

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：**TSPO (Temporal Sampling Policy Optimization)**，一个基于强化学习的可训练稀疏帧采样框架，用于长视频语言理解。核心包括三部分：(1) **Event-aware Temporal Agent**：基于 CLIP-Large (400M参数，冻结) + 3.5M 可学习参数，通过 local window attention 捕获 event-query 关联，使用 Gumbel-Softmax 进行概率化关键帧选择；(2) **TSPO RL 优化算法**：将关键帧选择和语言生成建模为联合决策过程，通过 GRPO (Group Relative Policy Optimization) 进行端到端策略优化，Video-MLLM 骨干保持冻结；(3) **双风格训练数据构建管道**：Comprehensive Temporal Data（从 LLaVA-Video-178K 过滤出的多关键帧依赖 QA）+ Video Needle-in-a-Haystack Data（拼接目标视频和无关视频段合成的 10∼60 分钟长视频）。奖励机制包含 Answering Accuracy Reward（R_A = 多选题答案是否正确）和 Temporal Localization Reward（R_T = 采样帧中目标视频帧占比）。
  实验比较：与 uniform sampling、training-free keyframe search（LongVU/DINOv2、CoS/MLLM-13B）、training-based 方法（FrameVOYAGER、MLLM-VFS）、以及 SFT-based keyframe 学习在 LongVideoBench / MLVU / Video-MME / LVBench 四个长视频 benchmark 上的准确率。同时进行跨 Video-MLLM 迁移实验（LLaVA-Video→Qwen2VL/Qwen2.5VL/LLaVA-Video-72B）、推理效率对比（frame extraction time、token 数、LLM time）。

- 硬件平台是什么，配置是什么。
  训练：8 × NVIDIA A800 80GB GPU。推理效率实验：单 GPU（论文未明确说明推理 GPU 型号，但 toy_example 要求 ≥28GB 显存）。

- 模型是什么。数据集和bench分别是什么。
  模型：Temporal Agent 基于 CLIP-Large (400M 参数，冻结) + 3.5M 可训练参数（local window attention + MLP projector）。Video-MLLM 骨干：LLaVA-Video-7B（训练时冻结），迁移实验使用 Qwen2VL-7B、Qwen2.5VL-7B、LLaVA-Video-72B。
  训练数据集：TSPO-10K（自建，包含 Comprehensive Temporal Data 和 Video Needle-in-a-Haystack Data，共 10,000 条样本，源自 LLaVA-Video-178K）。
  Benchmarks：LongVideoBench（验证集 1,337 视频，平均 12min）、MLVU（Dev 集 M-Avg 部分，3min∼2h）、Video-MME w/o sub（900 视频，短/中/长三档，2700 QA）、LVBench（平均 4101 秒，超长视频 benchmark）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源：https://github.com/Hui-design/TSPO（Public，AAAI 2026）。包含训练脚本（train_deepspeed.sh）、demo 脚本（llava_video_tspo.py / qwen25vl_tspo.py）、评估脚本（evaluation/）、模型实现（src/open_tspo/）、特征提取工具（mp_tools/）。

  算法 pipeline（张量计算级别）：

  ```
  输入: 长视频 V (T 帧), 文本查询 q, Video-MLLM π_l (冻结)

  # Step 1: 候选帧采样
  V_c = UniformSample(V, fps=1)  # T_c 帧候选

  # Step 2: CLIP 特征提取 (CLIP-Large, 冻结)
  F_f = CLIP_visual(V_c)         # [T_c, D], D 为特征维度
  F_t = CLIP_text(q)             # [1, D]

  # Step 3: Event-aware 特征增强 (3.5M 可训练参数)
  # Local window attention with sinusoidal positional encoding
  F_e = LocalWindowAttention(F_f + SinusoidalPE, window_size=w)  # [T_c, D]
  F_e = MLP_project(F_e)                                          # [T_c, D]

  # Step 4: 跨模态相似度计算
  Sim_event = CosineSimilarity(F_e, F_t)   # [T_c]
  Sim_frame = CosineSimilarity(F_f, F_t)   # [T_c]
  S = Sim_event + Sim_frame                # [T_c], 融合分数

  # Step 5: Gumbel-Softmax 概率采样
  γ ~ Gumbel(0, 1)                         # [T_c], 探索噪声
  P = Softmax(S/τ + γ)                     # [T_c], 概率分布
  P_sorted, I = TopK(P, T_s)               # 选 T_s 个关键帧索引+概率

  # Step 6: Video-MLLM 推理 (冻结)
  V_s = V_c[I]                              # 关键帧
  o = π_l(q, V_s, V_c)                     # 自回归生成回答

  # Step 7: 奖励计算
  R_A = 1 if (predicted_option == ground_truth) else 0   # 准确性奖励
  R_T = count(I in target_video_range) / T_s             # 定位奖励
  R = R_A + R_T  (Needle-in-a-Haystack) 或 R = R_A + 1  (Comprehensive)

  # Step 8: GRPO 策略更新 (仅更新 Temporal Agent 参数 θ_ts)
  # 对每组 G 个采样得到优势 A_i = (R_i - mean(R)) / std(R)
  # 目标: max E[π_ts(V_s|q,V_c) / π_ts_old(V_s|q,V_c) * A_i]
  # Video-MLLM 比率恒为 1 (冻结)，仅优化 θ_ts
  θ_ts ← θ_ts + lr * ∇J*_tspo(θ_ts)
  ```

  **关键设计**：
  - Temperature annealing: τ 从 0.025 逐渐退火至 0.01，训练初期鼓励探索，后期收敛到确定性关键段。
  - 推理时去除 Gumbel 噪声，直接 Softmax + TopK 确定性采样。
  - 训练时 T_s=16 帧，推理时 T_s=64 帧（候选帧 T_c=1FPS）。
  - GRPO 的 group size G 由 DeepSpeed 分布式训练管理。
  - Window size w=12, batch size=1, learning rate=5×10⁻⁴, 单 epoch 训练。

## Test-Time Temporal Sampling for Efficient MLLM Video Understanding

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：T3S 是训练无关、即插即用的推理包装器，在推理时对视频进行多试次随机帧采样（multi-trial frame sampling）和 token 子采样（token subsampling），将 m 个短子序列打包到一个前向传播中处理，最后通过 logit 聚合（均值、置信度加权或双试次交叉验证）输出预测。
  - 实验比较：对比 baseline（无采样的单序列 MLLM 推理）和同类训练无关方法 FastV、VTW、AdaReTake，在 VideoMME、LongVideoBench、MLVU 三个长视频理解 benchmark 上评估准确率和加速比。

- 硬件平台是什么，配置是什么。
  - 论文未明确说明具体 GPU 型号。论文 4.5 节提到"on a single GPU"进行实验，多 GPU 作为未来扩展方向提及。

- 模型是什么。数据集和 bench 分别是什么。
  - 模型：Qwen2.5-VL-7B、LLaVA-Video-7B、Oryx-1.5-7B（均为开源 7B 级 MLLM）。
  - 数据集/Benchmark：VideoMME（900 视频/2700 QA）、LongVideoBench（3763 视频/6678 多选题）、MLVU（多域长视频理解，M-Avg 指标）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：https://github.com/kaibinwang3/T3S
  - 算法 pipeline 解释：
  
  **核心思想**：视频存在大量时空冗余，传统 MLLM 对每帧每个 token 均完整编码并送入 self-attention（O(L²)）。T3S 将单长序列替换为 m 个短且多样化的子序列，通过随机采样的统计覆盖性弥补丢弃 token 的信息损失，同时将 attention 复杂度降为 O(∑αᵢ²L²)。
  
  **伪代码**（对应 Algorithm 1）：
  ```
  输入: 视频 V (共 F 帧), 文本tokens t
  参数: N (每试次帧数), m (试次数), αᵢ (每试次token保留率), k (top-k值)
  输出: 下一个输出 token t*
  
  for i = 1 to m:                          // Stage 1&2: 多试次采样
      P_i = RandomSample({1,...,F}, N)      // 从F帧中随机选N帧
      V̂_i = V[P_i]                          // 提取子序列帧
      v^(i) = E_v(V̂_i)                      // 视觉编码器编码, |v^(i)| = L = N×M
      v̂^(i) = C(v^(i), αᵢ)                  // token子采样, |v̂^(i)| = ⌊αᵢL⌋
                                             // C默认为均匀随机patch采样
  end for
  
  // Stage 3: 打包推理 + 聚合
  {o₁,...,oₘ} = MLLM(<v̂^(1),t> || ... || <v̂^(m),t>)  // 单次前向，带块对角线attention mask
  若 m=2 使用交叉验证: K = TopK(o₁, k); t* = argmax_{t∈K} o₂[t]
  否则: o_avg = (1/m) Σ oᵢ; t* = argmax o_avg[t]
  ```

  **张量级计算**：
  - Baseline: 输入 [L, D] 的视觉 token 序列，self-attention 计算 QK^T 矩阵 [L, L]，复杂度 O(L²D)。
  - T3S (m=2, α₁=0.5, α₂=0.3): 输入两个子序列，长度分别为 0.5L 和 0.3L，总长度 0.8L。Packed 序列中 self-attention 使用块对角线 mask，每个子序列仅与自身计算 attention。总复杂度 O((0.5²+0.3²)L²D) = O(0.34 L²D)，理论节省 66%。
  - 实际加速：Qwen2.5-VL-7B 上约 2.0× 加速，因为 packing 的序列长度总和（0.8L）短于原始 L，且每个 attention 块更小。
  
  **Logit 聚合**（第 3.5 节公式）：
  - 均值聚合: o_avg = (1/m) Σ oᵢ, oᵢ ∈ R^D（D=词表大小）
  - 置信度加权: wᵢ ∝ 1/H(π(oᵢ)), o_weighted = Σ wᵢ oᵢ（H 为预测分布熵）
  - 双试次交叉验证 (m=2): 试次 1 提出 top-k 候选 K，试次 2 在 K 上重新排序

  **关键超参**：m=2, N=256 (Qwen/Oryx) 或 128 (LLaVA-Video), α₁=0.5, α₂=0.3, topk k=2。评估工具为 VLMEvalKit。

## TimeLens: Rethinking Video Temporal Grounding with Multimodal LLMs

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：**TimeLens**，一套构建强视频时间定位（VTG）能力的 MLLM 后训练 pipeline，核心包含四大组件：
  (1) **Data Curation** — 手动审查三大 VTG benchmark（Charades-STA、ActivityNet Captions、QVHighlights），按严格标准（query 清晰度、事件存在性、唯一性、标注精度、穷尽性）诊断错误并重新标注，产出 **TimeLens-Bench**（4279 视频、9404 标注）；同时对训练数据进行自动化重标注（使用 Gemini-2.5-Pro 重新描述事件并标注时间戳），产出 **TimeLens-100K**（约 20K 视频、100K 标注）。
  (2) **Interleaved Textual Timestamp Encoding** — 将每帧的原始时间戳（如 "10.2s"）通过 LLM text tokenizer 转为文本 token，交错插入到对应帧的 visual tokens 之前，形成 interleaved visual-text 序列。相比 position-embedding (MRoPE/3D RoPE)、visual overlay、non-interleaved textual 等方案更简单且效果最优。
  (3) **Thinking-free RLVR (GRPO)** — 使用 GRPO 作为 RL 算法，模型直接输出时间片段 `(t_start, t_end)` 而不生成思考过程。奖励函数简化为单一的 `r(y) = IoU(Ŝ, S*)`，无需格式奖励。训练效率比 thinking-based RLVR 更高（1.0× vs 1.9× 训练时间），且性能更优。
  (4) **RLVR Recipes** — (a) Early Stopping：当 temporal IoU reward 和 group 内 reward 标准差 plateau 时（约 310 steps / ~2.5K samples），停止训练以避免性能退化。(b) Difficulty-based Sampling：用待训练模型对训练数据进行离线推理计算 IoU，定义 difficulty d_i = 1 - IoU(Ŝ_i, S*_i)，以高斯分布 g(d; μ, σ²) 进行采样，μ=0.05, σ=0.2 时可获得最优性能。

  实验比较：(1) Timestamp Encoding 消融：Interleaved Textual vs Visual Overlay vs Non-Interleaved Textual vs Position Embedding (MRoPE)，每种用 raw timestamp 和 frame index 两种格式；(2) Training Paradigm 消融：SFT (32K/100K data) vs Thinking-based RLVR vs SFT+Thinking-free RLVR vs Thinking-free RLVR alone；(3) RLVR Recipes：Early stopping 有效性验证（追踪 reward 和 evaluation metrics 曲线）、Difficulty-based sampling 不同难度均值的影响；(4) 主表对比：TimeLens-7B 和 TimeLens-8B vs GPT-4o、GPT-5、Gemini-2.0/2.5-Flash/Pro、VideoChat-Flash-7B、VideoChat-R1-7B、Time-R1-7B、TRACE、TRACE-uni、TimeSuite、Grounded-VideoLLM、MiMo-VL-7B、Qwen2.5-VL-7B、Qwen3-VL-8B/235B；(5) 不同模型规模 (3B/7B) 验证；(6) 原始 noisy 训练数据 vs TimeLens-100K 消融。

- 硬件平台是什么，配置是什么。
  训练：8 × NVIDIA H20 GPU。RLVR 训练时间 1.0× ≈ 4h10m（约 310 steps、~2.5K 训练样本）。SFT 训练：batch size 128，lr=1×10⁻⁵，1 epoch。RLVR 训练：batch size 8，每 prompt 采样 8 roll-outs，lr=1×10⁻⁶，KL coefficient β=0。Vision encoder frozen，其余参数可训练。消融实验使用较低分辨率（min_tokens=16 per frame，total_tokens=3584），最终模型使用较高分辨率（min_tokens=64，total_tokens=14336）。视频采样 2 FPS；interleaved textual encoding 时额外用 1 FPS + frame duplication 绕过 MRoPE 机制做公平对比。

- 模型是什么。数据集和bench分别是什么。
  模型：基于 Qwen2.5-VL-7B 和 Qwen3-VL-8B 分别得到 TimeLens-7B 和 TimeLens-8B。也验证了 Qwen2.5-VL-3B → TimeLens-3B。
  训练数据：**TimeLens-100K**，从 CosMo-Cap、InternVid-VTime、DiDeMo、QuerYD、HiREST 等数据集中采样视频，使用 Gemini-2.5-Pro 自动化重标注，约 100K 条高质量 VTG 标注（~20K 视频，时长主要分布在 0-240s），按视频时长均匀采样。
  评测 benchmark：**TimeLens-Bench**，包含 Charades-TimeLens（Daily Life, 1313 视频）、ActivityNet-TimeLens（Activity, 1455 视频）、QVHighlights-TimeLens（Mixed, 1511 视频），四项指标：R1@0.3, R1@0.5, R1@0.7, mIoU。额外验证：VUE-TR（Vidi benchmark）、Video-MME（通用视频理解 benchmark）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  论文声明：代码、数据、模型将全部开源（项目页面 https://timelens-arc-lab.github.io/）。论文未提供已生效的 GitHub/模型下载链接，但给出了各 baseline 模型的来源引用。

  TimeLens 算法 pipeline 伪代码：
  ```
  # === Phase 0: Data Preparation (offline) ===
  # TimeLens-Bench: manual re-annotation
  For each (video, query, segment) in {Charades-STA, ActivityNet, QVHighlights}:
      # Diagnose-then-Refine
      errors = check_criteria(video, query)  # 检查 5 类错误
      if errors:
          revise_query_or_select_new_event(video)
          annotate_precise_segment(video, new_query)
      cross_validate(batch)  # 不同标注者交叉验证

  # TimeLens-100K: automated re-annotation
  For each video in training_corpus:
      prompt_events = "Identify distinct events distributed across time"
      events = Gemini-2.5-Pro(video, prompt_events)
      For each event in events:
          query = Gemini-2.5-Pro.describe(event)
          timestamp = Gemini-2.5-Pro.localize(event)  # ["MM:SS", "MM:SS"]
          D_train.add((video, query, timestamp))

  # === Phase 1: Timestamp Encoding ===
  def interleaved_textual_encode(video_frames, fps):
      """每个 frame 前插入文本时间戳 token"""
      tokens = []
      for i, frame in enumerate(video_frames):
          t = i / fps  # 当前帧时间（秒）
          timestamp_text = f"{t:.1f}s"  # e.g. "10.2s"
          text_tokens = tokenizer(timestamp_text)  # LLM text tokenizer
          frame_tokens = vision_encoder(frame)
          # frame 作为独立 image 处理（duplicate 为两份绕过 MRoPE）
          tokens.extend(text_tokens)      # 时间戳 token 在前
          tokens.extend(frame_tokens)      # 视觉 token 在后
      return tokens
  # 输入 prompt: "The numbers before each video frame indicate its
  #  sampling timestamp (in seconds). Please find the visual event
  #  described by the sentence '{query}', determining its starting
  #  and ending times. Format: 'The event happens in <start> - <end> seconds'."

  # === Phase 2: Difficulty-aware Sampling (offline, before RL) ===
  For each (v_i, q_i, S*_i) in D_train:
      Ŝ_i = π_θ(v_i, q_i)  # offline inference
      d_i = 1 - IoU(Ŝ_i, S*_i)  # difficulty: higher = harder
      # 按高斯分布采样权重
      w_i = g(d_i; μ=0.05, σ=0.2) / p̂(d_i)  # density-corrected
  D_sampled = weighted_sample(D_train, w_i, size=~12K)

  # === Phase 3: Thinking-free RLVR with GRPO ===
  π_θ = load(Qwen2.5-VL-7B)  # 或 Qwen3-VL-8B
  freeze(vision_encoder)  # vision encoder 冻结
  For step in 1..max_steps:
      For each (v, q) in D_sampled[batch]:
          # 对每个 (v,q) 采样 G=8 个 responses
          For g in 1..G:
              y^(g) = π_θ(v, q)  # 直接输出 "(t_start, t_end)"
              # Thinking-free: 无 thinking 过程
              r^(g) = IoU(Ŝ^(g), S*)  # 仅用 IoU 作为奖励
          # 计算 advantage
          r_mean = (1/G) * Σ r^(j)
          For each g:
              A^(g) = r^(g) - r_mean
          # GRPO loss
          L = -(1/G) * Σ A^(g) * log π_θ(y^(g) | v, q)
          # KL 正则项 β=0 在本工作中不使用
      θ = θ - lr * ∇L  # lr=1e-6

      # Early stopping check
      if reward_plateau(temporal_IoU_reward) and
         reward_plateau(group_stddev):
          break  # 约 310 steps 时触发
  # 输出: π_θ* (TimeLens model)
  ```

  张量计算示例（interleaved textual encoding 的 forward pass）：
  ```
  输入: video V ∈ R^{T×H×W×3}  # T frames
  # Step 1: 每帧复制为两个相同 copy（绕过 MRoPE 的 frame merge）
  V_expanded = duplicate(V)  # T × 2 个 frame copies

  # Step 2: Vision encoder 处理
  For each frame pair (f_i, f_i_copy):
      patch_i = patch_embed(concat(f_i, f_i_copy))  # 每两个 frame 合并
      visual_tokens_i = vision_transformer(patch_i)  # shape: [n_patches, d_model]

  # Step 3: Interleaved timestamp injection
  For each visual_tokens_i:
      t_i = i / fps  # 时间戳 (秒)
      text_token_i = text_embed(f"{t_i:.1f}s")  # [1, d_model]
      # 交错插入
      sequence_i = [text_token_i; visual_tokens_i]  # prefix: timestamp before visual

  # Step 4: Full sequence
  full_seq = concat([sequence_0, sequence_1, ..., sequence_{T-1}])
  # Also prepend system prompt tokens
  full_seq = [prompt_tokens; full_seq]

  # Step 5: LLM decode
  output = LLM(full_seq)  # autoregressive generate "(t_start, t_end)"
  ```

  训练数据质量对比消融（Tab. 5）：
  | Training Data | Charades-TimeLens mIoU | ActivityNet-TimeLens mIoU | QVHighlights-TimeLens mIoU |
  |---|---|---|---|
  | Original Noisy Data | 35.6 | 31.3 | 44.6 |
  | TimeLens-100K | 48.3 | 43.1 | 56.7 |

  核心发现总结：
  - Interleaved textual prefix + raw timestamps 是最优 timestamp encoding
  - Thinking-free RLVR 在效率和性能上均优于 SFT 和 thinking-based RLVR
  - Early stopping (reward plateau 时) 省计算且避免性能退化
  - Difficulty-based sampling (μ=0.05 → difficulty ~0.95) 对 RLVR 性能至关重要

## V2Drop: Variation-aware Vision Token Dropping for Faster Large Vision-Language Models

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：**V2Drop**，一种基于 token 变异性（variation）的视觉 token 渐进式压缩方法。核心思路：(1) 不再依赖 attention weights 等外部信号判断 token 重要性，而是直接测量 visual token 在相邻 LLM 层之间的表示变化（L2 distance）；(2) 在 3 个预选层（如 LLaVA-1.5-7B 上的 layers 3, 17, 22）进行多阶段渐进式剪枝（progressive dropping），每层按 variation score 降序排序后保留 top-K 高变异性 token，丢弃低变异性的"惰性 token"；(3) 消除 attention-based 方法的位置偏见（positional bias），且天然兼容 FlashAttention 等高效算子。
  实验比较：V2Drop vs. ToMe、FastV、HiRED、LLaVA-PruMerge、SparseVLM、PDrop、DART、DyCoke 在多个 image/video benchmark 上的性能保留率、生成延迟、吞吐量、GPU 峰值显存。

- 硬件平台是什么，配置是什么。
  NVIDIA A100-PCIe-80GB GPU。软件：Python 3.10、PyTorch 2.1.2、CUDA 12.1。

- 模型是什么。数据集和bench分别是什么。
  模型：LLaVA-1.5-7B、Qwen2-VL-7B、LLaVA-OV-7B。
  Benchmarks：GQA、MMBench、MME、POPE、ScienceQA、TextVQA、AI2D、MMStar（图像理解）；MVBench、VideoMME（视频理解）。
  指标：benchmark 性能保留率（以原始模型为 100%）、LLM Generation Latency、Model Generation Latency、Total Latency、GPU Peak Memory、Throughput。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  已开源：https://github.com/xuyang-liu16/V2Drop（Apache-2.0，CVPR 2026）。
  算法流程（见论文 Algorithm 1）：
  输入：视觉 token F^v ∈ R^{M×D'}，剪枝层 L={l_1, l_2, l_3}，压缩目标 {K_a, K_b, K_c}
  M_curr ← M
  for l = 1 to L do:
      if l ∈ L:
          // Step 1: Variation Computation
          for i = 1 to M_curr:
              s_i^{(l)} ← ||f_i^{(l)} - f_i^{(l-1)}||_2  // L2 距离
          S^{(l)} ← {s_1, ..., s_{M_curr}}
          // Step 2: Token Ranking & Selection
          indices ← argsort(S^{(l)}, descending)
          F̂_l^v ← {f_indices[j] : j=1,...,K_l}  // 保留 top-K
          F_curr^v ← F̂_l^v, M_curr ← K_l
      // Step 3: 继续前向传播
      F_curr^v ← TransformerLayer(F_curr^v)
  return F_curr^v

  张量计算开销：对 M 个 D' 维 token 计算 L2 距离需 3MD' FLOPs（LLaVA-1.5 中约 7M FLOPs），仅为单层 attention（32B FLOPs）的 0.022%。三层总开销约 21M FLOPs，占完整前向传播的 0.002%。
  渐进式压缩 schedule 示例（LLaVA-1.5-7B, retain 192 tokens）：M=576 → layer 3 保留 top-50% → layer 17 保留 top-30% → layer 22 保留指定数量的所有 token。

## VideoAuto-R1: Video Auto Reasoning via Thinking Once, Answering Twice

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：**VideoAuto-R1**，一个 video auto-reasoning 框架，包含两个核心组件：
  (1) **Thinking Once, Answering Twice 训练范式**：输出格式为 $\boxed{a_1} \rightarrow$ `<think>` $r$ `</think>` $\rightarrow \boxed{a_2}$。模型首先生成初始答案 $a_1$，然后进行显式推理 $r$，最后输出审查后答案 $a_2$。使用 GRPO（Group Relative Policy Optimization）进行 RL 训练，dual-answer reward 设计为 $R = w_1 R_{task}^{(1)}(a_1) + w_2 R_{task}^{(2)}(a_2) + \lambda R_{fmt} + \alpha R_{fallback}$，其中 $w_2 > w_1$（如 0.9:1.1）以鼓励最终答案准确。$R_{fallback}$ 允许模型在 $a_1$ 输出 fallback 字符串 "Let's analyze the problem step by step" 时获得奖励（当 $a_2$ 正确时），避免低置信度猜测。
  (2) **Confidence-based Early-Exit 推理策略**：推理时首先生成 $a_1$，计算其 length-normalized mean log probability $s(a_1) = \frac{1}{L} \sum_{\ell=1}^{L} \log p_{\theta}(t_{\ell} \mid t_{<\ell}, q)$ 作为置信度分数。若 $s(a_1) \geq \log \tau$（默认 $\tau = 0.97$），则早停直接返回 $a_1$；否则继续生成推理过程和 $a_2$。若 $a_1$ 为 fallback 字符串则 $s(a_1) = -\infty$ 强制继续推理。

  实验比较：
  (a) **Training Strategy Comparison（Table 6）**：对比 SFT、RL without thinking（仅 direct answer）、RL with thinking（标准 CoT GRPO）、VideoAuto-R1。VideoAuto-R1 在 VideoMME 67.3 vs RL with thinking 66.1，VideoMMMU 58.6 vs 56.4，同时平均响应长度从 149 tokens 降至 44 tokens。
  (b) **Adaptive Reasoning Strategies（Table 7）**：对比 training-based auto-thinking（AdaptThink 风格，按样本标注 think/no-think 标签）vs inference-based（VideoAuto-R1）。Training-based 方法存在 mode collapse，在 MVBench 上 auto 模式下甚至不如 no-think baseline（70.5 vs 71.1）。VideoAuto-R1 在 auto 模式下 71.0 vs no-think 70.9 vs always-think 71.0，且无需额外标签。
  (c) **Video QA Benchmarks（Table 3）**：与 10+ 个 thinking-only 视频推理模型（Video-R1, Time-R1, VideoChat-R1, Video-RTS, VITAL, LongVILA-R1, LOVE-R1, VideoChat-R1.5 等）在 VideoMME, MVBench, LongVideoBench, MMVU, VideoMMMU, MVP 上对比。VideoAuto-R1 (Qwen2.5-VL-7B) 在 VideoMME 67.3、VideoMMMU 58.6、MVP 39.4 均达到 SOTA。Qwen3-VL-8B 版本进一步提升至 VideoMMMU 65.0。
  (d) **Temporal Grounding Benchmarks（Table 4）**：Charades-STA、ActivityNet、NExT-GQA。VideoAuto-R1 (Qwen2.5-VL-7B) Charades-STA mIoU 60.0 vs Time-R1 58.8 vs VITAL 59.9。Qwen3-VL-8B 进一步提升。
  (e) **Image Benchmarks（Table 5）**：MathVista, MathVision, MathVerse, MMMU, MMMU-Pro, MM-Vet。
  (f) **Reward Design Ablation（Table 9）**：对比 $w^1:w^2$ 不同权重比（1:1, 0.9:1.1, 0.8:1.2）和 fallback reward $\alpha$ 有无。非对称权重 + fallback 最优。
  (g) **Early-Exit Threshold Analysis（Figure 3）**：$\tau$ 从 0.86 到 0.98 对 accuracy 和 think ratio 的影响。推理密集 benchmark 上提高 $\tau$ 持续改善准确率，感知 benchmark 上准确率几乎不变。
  (h) **Confidence-Task Correlation（Table 8）**：MVBench 平均置信度 0.948（think ratio 25%、gain +0.1），VideoMMMU 平均置信度 0.874（think ratio 51%、gain +4.0）。
  (i) **Data Filtering Ablation（Table 11）**：对比 Text/Image/Video 组合的过滤策略。Text+Image+Video filtered 83K 最优。
  (j) **Cold-Start SFT Ablation（Table 17）**：SFT with Video-R1-CoT data 导致性能下降（66.0→60.1）；SFT→RL 仍差于直接 RL（61.7 vs 66.1）。
  (k) **Frame Count Ablation（Table 15）**：64/128/256/2048 frames 下的性能变化。

- 硬件平台是什么，配置是什么。
  **32 NVIDIA H100 GPU**，训练约 35 小时。使用 DeepSpeed + vLLM 加速 GRPO rollout generation。GRPO rollout size G=16，全局 batch size 256，训练 1 epoch。测试使用 greedy decoding（temperature=0），最大 response length 4096 tokens。

- 模型是什么。数据集和bench分别是什么。
  模型：
  - **Qwen2.5-VL-7B-Instruct**（主 backbone）：最多 16K total video tokens，sweep over {64, 128, 256} frames
  - **Qwen3-VL-8B-Instruct**（扩展 backbone）：最多 128K total video tokens，sweep over {64, 256, 2048} frames
  - 训练时 visual encoder 冻结，仅微调 projector 和 LLM
  - 优化器：AdamW，lr=$1 \times 10^{-6}$，weight decay=0.01，max grad norm=1.0，constant lr schedule，KL penalty $\beta$=0.01

  训练数据（83K，从 137K 过滤）：
  - Text 6.4K：DAPO-Math（数学推理）
  - Image 27.5K：ViRL, ThinkLite-Hard（图像推理）
  - Video 49.4K：Video-R1, TVBench, STI-Bench, MMR-VBench, Charades-STA, ActivityNet, Time-R1, NExT-GQA（视频 QA + 时序定位）
  - 过滤策略：对每个样本生成 8 个 responses，用 Qwen3-30B-A3B-Instruct 评估 correctness。全对（too easy）或全错（too hard）的 QA 样本被丢弃。时序定位数据全保留。

  评估框架：lmms-eval（greedy decoding, temperature=0）

  Benchmarks：
  - **Video QA（Perception）**：VideoMME (w/o subtitles), MVBench, LongVideoBench, MMVU (multi-choice)
  - **Video QA（Reasoning）**：VideoMMMU, MVP (Minimal Video Pairs, pairwise accuracy on MVP-mini)
  - **Temporal Grounding**：Charades-STA (Recall@0.3/0.5/0.7, mIoU), ActivityNet (Recall@0.3/0.5/0.7, mIoU), NExT-GQA (Acc, mIoU)
  - **Image Reasoning**：MathVista (testmini), MathVision (testmini), MathVerse (testmini), MMMU (val), MMMU-Pro (overall), MM-Vet (test)

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  项目主页：https://ivul-kaust.github.io/projects/videoauto-r1。代码基于开源 Qwen2.5-VL / Qwen3-VL + DeepSpeed + vLLM + lmms-eval。

  训练算法 pipeline 伪代码：
  ```
  # === VideoAuto-R1 Training (GRPO with Dual-Answer Reward) ===
  # 输入: prompt q (video + question), base model π_{ref}
  # 输出格式要求: \boxed{a_1} <think> r </think> \boxed{a_2}

  for each training batch with prompts {q_i}:
      # Step 1: Rollout generation (vLLM-accelerated, temperature=1.0)
      for each q_i:
          {o_i^1, ..., o_i^G} = π_{θ_old}.sample(q_i, G=16)  # 每个 prompt 采样 16 个候选

      # Step 2: Reward computation
      for each output o_i^j:
          parse \boxed{a_1}, <think> r </think>, \boxed{a_2} from o_i^j

          # Task rewards
          if task_type == QA:
              R_task^{k} = exact_match_or_math_verify(normalize(a_k), GT) ∈ {0,1}  # k=1,2
          elif task_type == Temporal_Grounding:
              R_task^{k} = max_tIoU(pred_segments, GT_segments) ∈ [0,1]
          elif task_type == Grounding_QA:
              R_task^{k} = R_QA + R_TG ∈ [0,2]

          # Format reward: strict regex check for template compliance
          R_fmt = 1 if matches("\boxed{...}<think>...</think>\boxed{...}") else 0

          # Fallback reward
          R_fallback = 1 if (a_1 == "Let's analyze...") and (R_task^{2} > 0) else 0

          # Total reward (w1=0.9, w2=1.1, λ=1, α=0.3)
          R_i^j = 0.9 * R_task^{1} + 1.1 * R_task^{2} + 1 * R_fmt + 0.3 * R_fallback

      # Step 3: GRPO advantage normalization
      for each group i:
          μ_i = mean({R_i^j}_{j=1..G}), σ_i = std({R_i^j}_{j=1..G})
          A_i^j = (R_i^j - μ_i) / (σ_i + ε)

      # Step 4: Policy update
      for each (q_i, o_i^j, A_i^j):
          ρ_i^j = π_θ(o_i^j|q_i) / π_{θ_old}(o_i^j|q_i)
          L = -1/G * Σ min(ρ_i^j * A_i^j, clip(ρ_i^j, 1-ε, 1+ε) * A_i^j) + β * D_KL(π_θ || π_ref)

          θ ← AdamW(L, lr=1e-6, wd=0.01)
  ```

  推理算法伪代码（Algorithm 1）：
  ```
  # === VideoAuto-R1 Inference (Confidence-Based Early Exit) ===
  # 输入: trained model p_θ, video v, question q, threshold τ=0.97
  Require: p_θ, v, q, τ, fallback_string f = "Let's analyze the problem step by step."

  # Step 1: Generate until first <think> tag is detected
  tokens, logprobs = p_θ.greedy_decode(v, q, stop_token="<think>")

  # Step 2: Extract first boxed answer a_1
  a_1_tokens = extract_between(tokens, "\boxed{", "}")
  L = len(a_1_tokens)
  a_1 = detokenize(a_1_tokens)

  # Step 3: Compute confidence score
  if a_1 == f:
      s = -inf  # fallback forces continuation
  else:
      # Length-normalized mean log probability
      s = (1/L) * Σ_{ℓ=1..L} logprobs[ℓ]  # log p_θ(t_ℓ | t_{<ℓ}, v, q)

  # Step 4: Early-exit decision
  if s >= log(τ):  # e.g., τ=0.97 → log(0.97)≈-0.0305
      return a_1  # Early exit: direct answer
  else:
      # Continue generation: reasoning + reviewed answer
      tokens_rest = p_θ.continue_decode(max_tokens=4096)
      r = extract_between(tokens_rest, "<think>", "</think>")
      a_2 = extract_last_boxed(tokens_rest)
      return a_2  # Return reviewed answer
  ```

  张量计算层面：confidence score $s(a_1)$ 的分子是 LLM 自回归解码中已产生的 per-step log-probability 的均值，$a_1$ 通常仅包含不到 10 个 token，因此 confidence 计算开销极小。early-exit 时避免生成额外数百 tokens，显著降低延迟和推理成本。实现上通过检测第一个 `<think>` tag 出现来终止早期生成，无需外部校准器。

## VideoLLaMB: Long-context Video Understanding with Recurrent Memory Bridges

- 属于算法pipeline的实现是什么？实验比较什么？
  VideoLLaMB 提出一种用于长视频理解的递归记忆桥接框架，核心实现包括：(1) SceneTiling 算法——基于 ViT [CLS] token 的帧间余弦相似度计算 depth score，按 μ+α·σ 阈值将视频语义分割为 K 个语义段；(2) Recurrent Memory Bridge Layers——单层 Transformer，在每个语义段前 prepend 固定数量（32）的 memory tokens，通过 self-attention 递归更新 memory tokens 并输出视觉表示；(3) Memory Cache with Retrieval——将历史 memory tokens 存储在 memory cache 中，以当前 memory token 为 query、cache 为 key/value 进行 cross-attention 检索更新，缓解 BPTT 梯度消失问题。实验比较长视频理解（EgoSchema/NExTQA/VideoMME）、综合视频理解（MVBench）、规划任务（EgoPlan）、流式视频字幕生成以及自建 NIAVH benchmark 上的帧检索能力。

- 硬件平台是什么，配置是什么。
  训练使用 4× NVIDIA A800 GPU；推理评估使用单张 NVIDIA A100 (80GB) / A800 GPU。论文声称单张 A100 即可处理 320 帧视频，仅训练于 16 帧。

- 模型是什么。数据集和bench分别是什么。
  模型：LLM 使用 Vicuna-7B-v1.5，视觉编码器使用 ViT-L/14（基于 CLIP），遵循 Video-LLaVA 架构。训练初始化自 LLaVA-1.5 配置。
  数据集：训练数据与 PLLaVA 相同（Video-LLaVA 视频数据 + LLaVA-1.5 微调图像数据），额外使用 VideoChat2 数据集进行扩展训练。
  Benchmark：EgoSchema（零样本子集评估，180s 平均长度）、NExTQA（验证集，45s 平均长度，含 temporal/causal/description 三类问题）、VideoMME（11s~1h 视频，short/medium/long 子集）、MVBench（20 类多选问答）、EgoPlan（3355 题，零样本具身规划）、自建 NIAVH（Needle In A Video Haystack，基于 Ego4D + Sora/DALL-E，320s 上下文）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源：https://github.com/bigai-nlco/VideoLLaMB，基于 LLaVA 代码库构建（Python 99.3%），README 包含安装步骤、CLI 使用、streaming、Gradio demo、训练/评估脚本和模型 zoo。无 license 文件。

  算法 pipeline 伪代码：
  ```
  # === VideoLLaMB 推理 Pipeline ===
  # 输入: video V = {v_1, v_2, ..., v_n} (n frames), query Q
  # 输出: answer A

  # Step 1: Vision Encoding
  for each frame v_i in V:
      f_i = ViT-L/14(v_i)  # 每帧提取特征, f_i ∈ R^{H×W×D}
  F = {f_1, f_2, ..., f_n}

  # Step 2: SceneTiling 语义分割
  for i = 1 to n-1:
      c_i = CosineSim(ViT.cls_token(v_i), ViT.cls_token(v_{i+1}))  # 相邻帧CLS余弦相似度
  for i = 1 to n-1:
      cl_i = max(c_{1..i-1}), cr_i = max(c_{i+1..n-1})
      d_i = (cl_i + cr_i - 2*c_i) / 2  # depth score
  μ = mean({d_1..d_{n-1}}), σ = std({d_1..d_{n-1}})
  threshold = μ + α * σ
  seg_boundaries = {i | d_i > threshold}  # 选取K-1个分割点
  将视频划分为K个语义段 {s_1, s_2, ..., s_K}, s_i = {f_{start_i}..f_{end_i}}

  # Step 3: Recurrent Memory Bridge (逐段处理)
  m_0 = random_init_memory_tokens(N_mem=32, D=1024)  # 初始化memory tokens
  MemoryCache = []  # memory cache用于跨段检索

  for i = 1 to K:
      # 3a: prepend memory tokens to segment features
      input_i = Concat(m_{i-1}, s_i)  # [32+M_i, 1024], M_i为段内帧数
      
      # 3b: Self-attention in Bridge Layer (单层Transformer, 8 heads, hidden=1024)
      [m_i', o_i] = BridgeLayer_SelfAttn(input_i)  
      # m_i': 更新后的临时memory tokens [32, 1024]
      # o_i: 段内视觉表示 [M_i, 1024]
      
      # 3c: Memory Retrieval (cross-attention with cache)
      if MemoryCache is not empty:
          M_cache = Concat(m_0, m_1, ..., m_{i-1})  # 拼接历史memory
          # Cross-attention: query = m_i', key = M_cache, value = M_cache
          m_i = Softmax(W_Q*m_i' * (W_K*M_cache)^T / sqrt(d_k)) * W_V*M_cache
      else:
          m_i = m_i'
      
      MemoryCache.append(m_i)

  # Step 4: LLM输入投影
  visual_summary = Projector(o_1, o_2, ..., o_K)  # 所有段视觉输出投影
  final_input = Concat(m_K, visual_summary)  # 当前memory + 视觉表示送入LLM

  # Step 5: LLM生成
  A = Vicuna-7B(Concat(final_input, Tokenize(Q)))

  # 计算复杂度:
  # Bridge层: O((C+M)^2) per segment, C=段帧数, M=32 memory tokens
  # Memory Retrieval: O(M*K) per step, K=段数
  # 总时间: O(K^2), 空间: O(K)
  # LLM: O(M^2) token输入 (vs 原始 O(n^2))
  ```

  张量计算层面：ViT-L/14 对每帧 224×224 图像提取 patch features（patch_size=14, 256 patches + 1 CLS token, D=1024）。SceneTiling 仅计算 CLS token 间余弦相似度，额外开销极小（O(n) 次余弦计算）。Memory Bridge 的 self-attention 输入为 [32+M, 1024]，M 为段内帧数（如 16 帧分 4 段则 M≈4），计算量极小。Memory Retrieval 的 cross-attention 为 [32] query × [32*K, 1024] key/value，线性增长。最终 LLM 输入约 32+N_proj 个 token（vs 无压缩的 n_patches * n_frames = 256*16=4096 tokens），GPU 显存呈线性缩放（Figure 4），支持 320 帧处理。

  Streaming caption 模式下：SceneTiling 仅用左侧相似度 d_i = (cl_i - c_i)/2 实时检测场景边界，无需预知全视频，在边界处自动生成事件字幕。

## VideoSeek: Long-Horizon Video Agent with Tool-Guided Seeking

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：VideoSeek 是一个 model-agnostic 的长时域视频 agent，遵循 ReAct 风格的 think-act-observe 循环。核心创新是用多粒度工具（overview/skim/focus）按视频逻辑流主动 seek 答案关键帧，而非密集解析全视频。
  - 实验比较：在四个 benchmark 上对比 standalone LMMs（GPT-4o, Gemini 1.5 Pro, Qwen2.5-VL-72B, Gemini 2.0 Flash, GPT-5）和 video agents（VideoAgent, VideoTree, DrVideo, VCA, MR. Video, DVD）。主要指标：accuracy（%）和 #Frames（处理的帧数）。消融研究分析 thinking model 替换（GPT-5 vs o4-mini vs GPT-4.1）和工具配置影响（逐一移除 overview/skim/focus）。附录有 α 参数（帧预算缩放因子）的 sensitivity 分析以及 intermediate reasoning 效果分析。

- 硬件平台是什么，配置是什么。
  - 论文未明确说明推理所用 GPU 或硬件配置。论文指出 runtime 受网络延迟、GPU 类型、API 调度等因素影响，因此不将 runtime 作为主要效率指标。推理通过 API 调用 GPT-5/o4-mini/GPT-4.1 完成，视觉内容也由 GPT-5 解释。论文未说明本地部署配置。

- 模型是什么。数据集和bench分别是什么。
  - 模型：默认 thinking LLM 为 GPT-5（API），ablations 中使用 o4-mini 和 GPT-4.1。视觉解释也由 GPT-5 完成。对比的 LMMs：GPT-4o, GPT-5, Gemini 1.5 Pro, Gemini 2.0 Flash, Gemini 2.0 Flash Thinking, Gemini 2.5 Pro, Qwen2.5-VL-32B/72B, Video-R1, VideoChat-R1, SEED-Bench-R1。对比的 agents：VideoAgent, VideoTree, DrVideo, VCA, MR. Video, DVD。
  - Benchmarks：
    - LVBench：1,549 MC 题，103 个 hour-long 视频，评估 long-term memory 和 extended comprehension
    - Video-MME (long subset)：900 题，300 视频，平均 2,466 秒
    - LongVideoBench (long split)：564 题，188 视频，时长 900–3,600 秒
    - Video-Holmes：1,837 题，270 个 suspense 短片，7 个推理维度（SR, IMC, TCI, TA, MHR, PAR, CTI）

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：github.com/jylins/videoseek（CVPR 2026）。核心 agent 依赖闭源 LLM（GPT-5 API），但开源的 toolkit 设计和 prompt 策略可直接复用。
  - 算法 pipeline（参照论文 Algorithm 1）：

    输入：用户 query Q，视频 X，系统指令 I，thinking model θ_think，工具集 T（overview, skim, focus, answer），最大轮次 N

    ```
    1. 初始化 trajectory τ ← ⟨I, Q⟩
    2. T ← T ∪ {answer}
    3. for t = 1 to N:
    4.   (z_t, a_t) ← θ_think(τ)          // 基于已有 trajectory 推理 + 工具规划
    5.   if a_t 仅含单个 answer:
    6.     Y ← parse_answer(a_t); break
    7.   o_t ← call_tools(a_t, X, T)      // 执行工具，获取观察
    8.   τ ← τ ∪ ⟨z_t, a_t, o_t⟩         // 附加到 trajectory
    9. if Y 为空:
    10.  Y ← θ_think(τ ∪ I_answer)        // 直接回答指令
    11. return Y
    ```

    工具说明：
    - overview: 从全视频均匀采样 16α 帧，生成每帧简要描述（~50 words），构建粗略 storyline
    - skim: 在选定区间 [t1, t2] 上采样 4α 帧，以 ~25 words/帧 描述并高亮与 query 相关的时间戳（~50 words）
    - focus: 在短片段 [t3, t4] 上以 1 FPS 密集采样，直接回答 query 或返回 "No relevant content found"
    - 工具设计约束：每轮仅调用一个工具；α 为帧预算缩放因子（LVBench α=4，其余 α=2）

    张量计算层面：VideoSeek 本身不涉及底层张量计算，它是 prompt-based agent 框架。每轮 think-act-observe 的输入为完整 trajectory τ 的文本表示，由 GPT-5 API 处理。视觉帧图像的视觉 token 化由 GPT-5 的 vision encoder 内部完成（论文未给出 encoder 细节）。每轮工具返回的 observation（帧描述文本 + 时间戳）追加到 τ 中。trajectory 的文本 token 数量随轮次线性增长，论文报告 LVBench 无字幕平均 49K tokens，有字幕 57K tokens。

## VisiPruner: Decoding Discontinuous Cross-Modal Dynamics for Efficient Multimodal LLMs

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：**VisiPruner**，一个无需训练的 MLLM 推理剪枝框架，通过揭示 MLLM 跨模态交互的三阶段规律实现对视觉 token 的分层压缩：
  (1) **浅层（Task Recognition）**：在 LLaVA-v1.5 7B 的 layer 1，将所有视觉 token 的 cross-attention 合并到单个随机视觉 token 上作为 attention sink；layer 2+ 完全跳过视觉相关 attention（cross-attention + visual self-attention），仅保留 FFN 对视觉 token 的处理。
  (2) **中层（Sparse Cross-Modal Fusion）**：提出 **influence-based 动态 token 选择方法**——在每个过滤层，计算每个视觉 token 被 mask 后对最后输入 token 的 attention output 的影响，使用 **cosine similarity**（阈值 < 0.995 定义为过滤层）和 **L2 distance**（阈值 < 0.2 的 token 被丢弃）双指标联合评估。平均将 576 个视觉 token 压缩至 10.3 个关键 token。算法具体步骤：① 在每层计算原始 cross-attention output O_i；② 逐个 mask 视觉 token j（设 W'_{i→j}=0），重算 masked attention output O'_{i masked}；③ 计算 CosineSim(O_i, O'_i) 和 L2Dist(O_i, O'_i)；④ 若 cosine < 0.995，定义该层为 filtering layer，将 L2 < 0.2 的 token 丢弃，仅保留剩余 key tokens 进入后续层。
  (3) **深层（Linguistic Alignment）**：在 middle layer 之后持续追踪保留 token 的影响，若连续两个层均无 measurable impact，则定义后一层为 vision exit layer（ℓ_exit，LLaVA-v1.5 7B 平均在第 23.9 层）。超过 ℓ_exit 后移除所有保留的视觉 token，进一步消除冗余计算。

  实验比较：与现有 training-free token pruning 方法对比——
  - **FastV** (Chen et al., 2024a)：基于 last-to-vision attention 选择最重要视觉 token
  - **FitPrune** (Ye et al., 2024b)：基于 attention-distribution saliency 剪枝
  - **SparseVLM** (Zhang et al., 2025b)：基于 cross-attention importance rank-based 剪枝
  - **PyramidDrop** (Xing et al., 2024)：在多个阶段逐步减少视觉 token
  所有比较在相同 visual attention computation reduction ratio 下进行（如 -98.3% 对应的 retained tokens 可能不同）。

- 硬件平台是什么，配置是什么。
  论文未明确说明具体 GPU 型号。主干模型为 LLaMA 2 7B/13B（Touvron et al., 2023）作为 MLLM 的 LLM backbone，使用标准的 GPU 推理环境。FLOPs 分析基于 LLaMA 2 7B 架构：hidden dim d=4096, FFN intermediate dim m=11008, 32 layers, 32 attention heads。论文提到"due to hardware constraints, our analysis was limited to models with up to 13 billion parameters"。

- 模型是什么。数据集和bench分别是什么。
  模型：LLaVA-v1.5 7B、LLaVA-v1.5 13B、InternVL2.5 8B、Qwen2-VL 7B、MobileVLM-v2 3B。
  数据集/benchmark：
  - GQA (Hudson and Manning, 2019)：视觉问答
  - MME (Fu et al., 2024)：综合多模态评估
  - POPE (Li et al., 2023b)：物体幻觉检测
  - MMBench (Liu et al., 2024)：多模态综合能力
  - MMStar (Chen et al., 2024b)：多模态评估
  - ScienceQA / SQA (Lu et al., 2022)：科学推理
  - TextVQA / VQAT (Singh et al., 2019)：OCR 视觉问答
  - MM-Vet (Yu et al., 2024)：多模态综合能力（需要生成式回答）

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源：代码地址 https://github.com/EIT-NLP/VisiPruner，Apache 2.0 许可证。仓库包含 `llava/cli_pruning.py`（核心剪枝 CLI）、`scripts/`（GQA/MME/TextVQA 评估脚本）、`visualization/`（logits lens、attention 可视化、L1 norm 分析 Notebook）。

  伪代码（VisiPruner 推理前向传播）：
  ```
  Input: visual_embeddings H_v (N_v x d), text_embeddings H_t (N_text x d)
  Hyperparams: shallow_mid_layer S_mid, cosine_threshold=0.995, l2_threshold=0.2
  Output: generated answer tokens

  # Stage 1: Shallow layers (1 to S_mid)
  for l in 1..S_mid:
      if l == 1:
          # Attention Merging: merge all cross-attn to one random visual token k
          A_cross = softmax(Q_t @ K_v^T / sqrt(d))
          A_merged = zeros_like(A_cross)
          A_merged[:, k] = sum(A_cross, dim=1)  # all weights -> token k
          H_cross = A_merged @ V_v
          H_t = TransformerBlock(H_t + H_cross)  # only FFN+self-attn for text
      else:  # l >= 2
          # Skip all visual attention (cross + visual self-attn)
          H_t = TransformerBlock_text_only(H_t)
          H_v = FFN_only(H_v)  # no self-attention among visual tokens
      H = concat(H_v, H_t)

  # Stage 2: Middle layers (S_mid+1 onward)
  filtering_layer_found = False
  for l in S_mid+1..L:
      if not filtering_layer_found:
          # Compute original attention output for last text token
          O_last = Attention(Q_t[-1], K_all, V_all)
          for each visual token j:
              W'_i->j = 0  # mask token j
              O'_masked = Attention_masked(Q_t[-1], K_all, V_all, mask=j)
              cos_sim[j] = dot(O_last, O'_masked) / (||O_last|| * ||O'_masked||)
              l2_dist[j] = ||O_last - O'_masked||_2
          if min(cos_sim) < 0.995:
              filtering_layer_found = True
              H_v = H_v[l2_dist >= 0.2]  # keep only influential tokens
      # Continue with retained visual tokens only
      H = TransformerLayer(concat(H_v, H_t))

  # Stage 3: Deep layers - Vision Exit
      if filtering_layer_found:
          if tokens_have_no_impact_for_2_consecutive_layers:
              H_v = []  # remove all vision tokens, exit at this layer
              # Continue with text-only processing
      H = TransformerLayer(H)

  return generated_tokens
  ```

  复杂度分析（LLaVA-v1.5 7B，576 visual + 74 text tokens）：
  - 视觉相关 attention 计算最大减少 99.0%（-98.3% 配置下）
  - 总 FLOPs：从 3.82T 降至 1.76T（-53.9%）
  - FFN FLOPs 公式：3 × n × d × m；attention FLOPs：2 × n² × d
  - 视觉 FLOPs 总体减少 62.8%（考虑到仍保留 FFN 对视觉 token 的处理）
  - KV cache 大幅缩减：浅层和深层不存视觉 KV，中间仅存 ~10 tokens

## WorldMM: Dynamic Multimodal Memory Agent for Long Video Reasoning

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：**WorldMM**，一个多模态记忆代理框架，用于对小时到周级别的长视频进行推理。核心包含三个阶段和三类互补记忆：
  (1) **多模态记忆构建 (Multimodal Memory Construction)**：
     - **Episodic Memory（情节记忆）**：将长视频按多时间尺度 T = {t₀, t₁, ..., t_N}（如30s, 3min, 10min, 1h）分段，每段用Video LLM生成caption，再转化为(entity, action, entity)三元组，构建多尺度知识图谱 M_e = {G_{t₀}, ..., G_{t_N}}。
     - **Semantic Memory（语义记忆）**：用更大时间粒度t_s分段，提取语义三元组（关注概念性知识而非具体事件），通过Consolidate(G^k_{t_s}, T^{k+1}_{t_s})过程（embedding相似度匹配→LLM合并/去重/纠错）增量更新跨场景的长期关系、习惯图谱 M_s = G^M_{t_s}。
     - **Visual Memory（视觉记忆）**：两种构建策略——(i) 特征检索：将视频以t_v切片，用多模态编码器（VLM2Vec-V2）编码为特征向量 f_v^k，构成 M_v^f = {f_v^1, ..., f_v^L}；(ii) 时间戳检索：每帧配时间戳 M_v^I = {(t_i, I_i)}。
  (2) **自适应记忆检索 (Adaptive Memory Retrieval)**：
     - Retrieval Agent R 是核心调度模块，每轮 i 输入用户query q和历史检索记录 r_{<i}，输出 (m_i, q_i) 或 STOP：
       R(q, r_{<i}) = { (m^i, q^i) if insufficient and i≤N; STOP otherwise }
     - 记忆检索方式各异：Episodic用Personalized PageRank (PPR) + LLM跨尺度重排序（top-m）；Semantic用PPR边级评分（边得分=两端节点PPR之和）；Visual用余弦相似度检索或时间戳直接访问。
     - 最多N=5轮迭代，自适应停止。
  (3) **响应生成 (Response Generation)**：Retrieval Agent决定STOP后，将全部检索历史送入Response Agent生成最终答案。
  实验比较：
  - 主实验：WorldMM-GPT和WorldMM-8B vs. 四类baseline（Base video LLMs、Long video LLMs、RAG-based、Memory-based），在5个长视频QA benchmark上的accuracy对比。WorldMM-GPT平均69.5%，比最强baseline高8.4%。
  - 消融实验：不同记忆组合(E/V/E+S/E+V/E+S+V)的accuracy对比；单模组变体（固定时间尺度单图、embedding检索代替graph、去consolidation、仅feature/仅timestamp检索）。
  - 效率实验：端到端延迟 vs. accuracy 的 trade-off 对比。
  - 泛化实验：不同backbone（Gemini 3 Flash + Qwen3-VL-Emb / VLM2Vec-V2, GPT-5 + Qwen3-VL-Emb / VLM2Vec-V2）对比。
  - 时序检索实验：tIoU指标对比（WorldMM 10.09% vs. baselines 0.58-4.35% on EgoLifeQA）。
  - 多轮检索消融：最大检索步数1→5的性能提升（EgoLifeQA上最大提升9.3%）。

- 硬件平台是什么，配置是什么。
  论文未明确说明具体GPU型号和实例配置。Backbone推理使用GPT-5 API（闭源商业模型，远程调用）、Gemini 2.5 Pro API、以及本地部署的Qwen3-VL-8B-Instruct（8B参数）。记忆构建阶段使用GPT-5-mini提取captions和三元组。视觉编码使用VLM2Vec-V2。语音转录使用Distil-Whisper large-v3.5。框架本质是API-heavy pipeline，对本地硬件要求主要来自开源模型推理（Qwen3-VL-8B约需16GB+ GPU显存）和embedding编码。

- 模型是什么。数据集和bench分别是什么。
  模型：
  - 骨干Video LLM：GPT-5（闭源）、Qwen3-VL-8B-Instruct（开源，8B）、Gemini 3 Flash（用于backbone泛化实验）
  - 记忆构建LLM：GPT-5-mini
  - 多模态编码器：VLM2Vec-V2、Qwen3-VL-Embedding-2B（用于消融）
  - 语音转录：Distil-Whisper large-v3.5
  数据集和Benchmark：
  - EgoLifeQA：500题，第一人称周级别视频（44.3h），5类：EntityLog, EventRecall, HabitInsight, RelationMap, TaskMaster
  - Ego-R1 Bench：300题，同一周级别视频，侧重多步推理
  - HippoVlog：1,000题，vlog风格（0.45h），4类：Auditory, Visual, Auditory+Visual, Summarization
  - LVBench：1,534题，通用长视频（1.14h），3类粒度：Short(<30s), Medium(30s-5min), Long(>5min)
  - Video-MME (long subset)：900题，通用长视频（0.69h），12类细粒度分类
  - 评估指标：Accuracy（多选题）、tIoU（时序交并比，用于检索质量评估）

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源情况：论文提供项目页面 https://worldmm.github.io，但论文明文未确认代码是否已开源。HippoRAG（基于PPR检索）、EgoRAG等baseline有开源实现。论文在Appendix中提供了完整的prompt模板（Fig.9-17）。
  
  算法pipeline伪代码（核心检索循环）：
  ```
  # === 阶段1: 多模态记忆构建 (离线) ===
  输入: 长视频 V, 时间尺度集合 T={t0,...,tN}
  
  # 1a. 情节记忆 (Episodic Memory)
  for ti in T:
      将V按ti等长切分为段 S_i = {s_i^1, s_i^2, ...}
      for s_i^k in S_i:
          采样帧 + 语音转录(Whisper)
          cap_i^k = VideoLLM.generate_caption(frames, transcript)
          trip_i^k = LLM.extract_triplets(cap_i^k)  # → [(e1, action, e2), ...]
      G_ti = construct_KG(trip_i^k from all segments)
  M_e = {G_t0, ..., G_tN}
  
  # 1b. 语义记忆 (Semantic Memory)
  G_s = empty_KG()
  for segment k in coarse_segments(t_s):
      T_s^k = LLM.extract_semantic_triplets(cap)
      matched = embedding_similarity(G_s, T_s^k)  # >0.6阈值
      T_remove, T_update = LLM.consolidate(G_s, T_s^k, matched)
      G_s = (G_s \ T_remove) U T_update  # Consoldiate公式
  M_s = G_s
  
  # 1c. 视觉记忆 (Visual Memory)
  M_v_f = {VLM2Vec.encode(segment): segment for segment in split(V, t_v)}
  M_v_I = {(t, frame): frame for each frame at timestamp t}
  
  # === 阶段2: 自适应记忆检索 (在线) ===
  输入: query q
  history = []
  for i in 1..N:  # N=5
      decision = RetrievalAgent(q, history)  
      if decision == STOP: break
      
      m_type, query_i = decision
      
      if m_type == "episodic":
          candidates = []
          for G_ti in M_e:
              # PPR: Personalized PageRank, seed=query_i中的实体节点
              ppr_scores = PersonalizedPageRank(G_ti, seed=query_i)
              candidates += top_k_by_ppr(ppr_scores, k)
          # LLM跨尺度重排序: 从多尺度候选中选出top-m
          results = LLM.cross_scale_rerank(query, candidates)  # → top-m
      
      elif m_type == "semantic":
          ppr_scores = PersonalizedPageRank(G_s, seed=query_i)
          # 边得分 = 两端节点PPR分数之和
          edge_scores = {e: ppr(u)+ppr(v) for e=(u,v) in G_s}
          results = top_k_triplets(edge_scores, k=10)
      
      elif m_type == "visual":
          if is_timestamp_range(query_i):
              # 格式: "DAY X HH:MM:SS DAY Y HH:MM:SS"
              results = fetch_frames_from_M_v_I(query_i)
          else:
              f_query = VLM2Vec.encode(query_i)
              # 余弦相似度检索
              results = top_k_by_cosine_sim(f_query, M_v_f, k)
      
      history.append((m_type, query_i, results))
  
  # === 阶段3: 响应生成 ===
  answer = ResponseAgent(q, history)
  return answer
  ```
  
  关键张量/数据流与计算：
  - Caption生成: video_frames[N_frames, H, W, 3] + transcript[T_tokens] → VideoLLM → caption[S]
  - 三元组提取: caption[S] → LLM.extract → [(entity:str, action:str, entity:str)]
  - PPR检索（KG迭代传播）: 邻接矩阵 A[N×N], seed向量 s₀[N]（query实体对应位置=1）, 迭代 s = α·A^T·s + (1-α)·s₀, 到收敛后s[i]为节点i的PPR分数
  - 视觉特征编码: video_segment[T, H, W, 3] → VLM2Vec-V2 → f_v[D], D为embedding维度
  - 余弦检索: sim = (f_query·f_v^k) / (||f_query||·||f_v^k||)
  - 语义合并: Consolidate使用embedding cosine similarity >0.6匹配 + LLM决策T_remove/T_update

## XStreamVGGT: Extremely Memory-Efficient Streaming Vision Geometry Grounded Transformer with KV Cache Compression

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：**XStreamVGGT**，一个 tuning-free 的 KV cache 压缩方法，无缝集成 pruning 和 quantization 两种技术，用于 StreamVGGT 的流式 3D 视觉几何推理。由两个核心组件构成：

  **(1) 基于 Query 引导的 KV Cache Pruning**（消除多帧冗余）：
  - 对当前帧的 Query tokens 按固定组大小 g 分组并平均池化：`Q_pooled = concat(Q_special, GroupAvg(Q_normal, g))`，然后跨 attention heads 平均得到 `Q̄_t ∈ R^{N_pooled × C}`。
  - 对中间帧的 Key tokens 也跨 heads 平均：`K̄_prunable ∈ R^{T_prunable × C}`。
  - Token 重要性分数通过 query-key 内积计算：`S_matrix = Q̄_t · (K̄_prunable)^T`，然后沿 Query 维度平均：`S = (1/N_pooled) · Σ_i S_matrix[i,:]`。
  - 基于重要性分数做 top-k 选择，总 cache 长度控制在预算 `L_max` 以内（实验中设为 2K tokens）。始终保留第一帧 KVs（作为几何参考）和当前帧 KVs。第一帧和当前帧 token 之外的历史帧才参与剪枝。
  - 剪枝在每层 temporal global attention 完成后触发。分组池化设计保持与 FlashAttention 等高效 attention kernel 的兼容性。

  **(2) 维度自适应 KV 量化**（基于 KV 分布特性）：
  - 分析发现：Key tensors 存在显著的 channel-wise outliers（少量 channel 的数值远大于其他 channel），而 Value tensors 的分布更均匀。
  - 量化策略：对 Keys 使用 **per-channel 量化**（每个 channel 独立计算 scale s_c 和 zero-point z_c），对 Values 使用 **per-token 量化**（每个 token 独立计算 scale s_t 和 zero-point z_t）。
  - 采用非对称均匀量化：`x̂ = clamp(⌊x/s⌋ + z, 0, 2^b-1)`，使用 KIVI 方案，INT4 精度，group size 64。
  - 量化紧耦合在 pruning 之后：先对 pruned KV cache 进行量化存储，attention 计算时 dequantize 回浮点精度。

  实验比较：对比以下方法：
  - **VGGT** (CVPR 2025)：Offline 推理，使用全局 Alternative-Attention，无 KV cache 限制。
  - **StreamVGGT** (2025)：Online streaming 推理，frame-wise causal attention + unbounded KV cache。

  评估任务：3D 重建（7-Scenes, NRGBD），相机姿态估计（TUM Dynamics, ScanNet），单目/视频深度估计（Sintel, Bonn, KITTI）。
  效率评估：50-1000 帧输入序列，测量 GPU 内存消耗和推理速度（FPS）。
  消融实验：cache length（2K/4K/6K/8K），pruning 和 quantization 的独立效果。

- 硬件平台是什么，配置是什么。
  **单张 NVIDIA A100 GPU (80GB)**。所有效率实验在此配置上完成。StreamVGGT 和 VGGT 随帧数增加出现 FPS 显著下降并快速 OOM，XStreamVGGT 可稳定运行 1000 帧以上不 OOM。

- 模型是什么。数据集和bench分别是什么。
  模型：**StreamVGGT**（基于 VGGT，1.2B 参数，Alternating-Attention 架构，L 层 spatio-temporal transformer encoder，每层含 frame-wise spatial self-attention + temporal causal attention）。输入图像处理遵循 Point3R 协议：变长宽比处理，resize 最大边长 ≤ 518 pixels。token 序列含 camera token (1)、register tokens (R)、patch tokens (N)，每帧总长 1+R+N。

  数据集和 Benchmark：
  - **3D 重建**：7-Scenes [22], NRGBD [2]。指标：Accuracy (Acc↓), Completion (Comp↓), Normal Consistency (NC↑)。
  - **相机姿态估计**：TUM Dynamics [24], ScanNet [8]。指标：ATE↓, RPE_trans↓, RPE_rot↓。
  - **深度估计**（单目+视频）：Sintel [3], Bonn [19], KITTI [13]。指标：Abs Rel↓, δ<1.25↑。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  代码开源：https://github.com/ywh187/XStreamVGGT/

  算法 pipeline 伪代码：

  ```
  # 初始化：XStreamVGGT 推理循环，每帧 t 执行以下步骤

  # 第一步：Token 化
  F_t = PatchEmbed(I_t)           # I_t ∈ R^{3×H×W} → F_t ∈ R^{N×C}
  Input_t = [g_t; r_t; F_t]        # camera token + register tokens + patch tokens
                                   # Input_t ∈ R^{(1+R+N)×C}

  # 第二步：逐层 Transformer 处理（L 层 Alternating-Attention）
  for ℓ = 1 to L:
      # 2a. 帧内空间 self-attention（无 KV cache）
      H_t^(ℓ) = SpatialSelfAttn(Input_t^(ℓ-1))

      # 2b. 时序 causal attention（使用 KV cache）
      K_t^(ℓ), V_t^(ℓ) = Proj_KV(H_t^(ℓ))  # 新计算当前帧 K, V
      Q_t^(ℓ) = Proj_Q(H_t^(ℓ))

      # 2c. 拼接历史 cache 和新 K/V，causal attention
      K_all = concat(Cache.K_{1:t-1}^(ℓ), K_t^(ℓ))
      V_all = concat(Cache.V_{1:t-1}^(ℓ), V_t^(ℓ))
      Out_t^(ℓ) = FlashAttn(Q_t^(ℓ), K_all, V_all, causal_mask=True)

      # 2d. KV Cache 剪枝（当 cache 长度超过 L_max 时触发）
      if len(K_all) > L_max:
          # 分组池化 Query（保留特殊 token，patch token 分组平均）
          Q_pooled = concat(Q_special, GroupAvg(Q_normal, g=16))
          Q̄ = mean(Q_pooled, dim=heads)     # 跨 head 平均

          # 提取中间帧 prunable Key，跨 head 平均
          K̄_prunable = mean(K_{first+1 : t-1}, dim=heads)

          # 计算重要性分数
          S = mean(Q̄ @ K̄_prunable^T, dim=query)  # 沿 query 维度平均

          # Top-k 选择 + 保留首帧和当前帧
          I_middle = TopK(S, k = L_max - T_first - T_current)
          I_keep = {1..T_first} ∪ I_middle ∪ {T-T_current+1..T}

          # 同步剪枝 K 和 V
          Cache.K_{1:t}^(ℓ) = Cache.K_{1:t}^(ℓ)[I_keep]
          Cache.V_{1:t}^(ℓ) = Cache.V_{1:t}^(ℓ)[I_keep]

      else:
          # 未达预算，直接追加
          Cache.K_{1:t}^(ℓ).append(K_t^(ℓ))
          Cache.V_{1:t}^(ℓ).append(V_t^(ℓ))

      # 2e. 维度自适应量化（剪枝后对 cache 量化存储）
      for each channel c in Cache.K:
          s_c = (K_max[c] - K_min[c]) / (2^4 - 1)
          z_c = round(-K_min[c] / s_c)
          K̂_c = clamp(round(K_c / s_c) + z_c, 0, 15)  # INT4

      for each token i in Cache.V:
          s_i = (V_max[i] - V_min[i]) / (2^4 - 1)
          z_i = round(-V_min[i] / s_i)
          V̂_i = clamp(round(V_i / s_i) + z_i, 0, 15)   # INT4

      # 存储量化后的 K̂, V̂ 及 scale/zero-point 参数

  # 第三步：任务头预测
  CameraParams = Head_camera(Out_t^(L))
  PointMap = Head_pointmap(Out_t^(L))
  DepthMap = Head_depth(Out_t^(L))
  ```

  关键超参数：pooling group size g=16，cache budget L_max=2K，KIVI INT4 量化 + group size 64。
  剪枝和量化均为 tuning-free，无需额外训练。

## mPLUG-Owl3__Towards_Long_Image-Sequence_Understanding_in_Multi-Modal_Large_Language_Models

- 属于算法pipeline的实现是什么？实验比较什么？
  论文提出 Hyper Attention Transformer Block (HATB)，在语言模型的 transformer block 中并行执行 cross-attention 和 self-attention，实现图文多模态融合。相比 Flamingo 的串行式 cross-attention 和 LLaVA 的视觉特征直接拼接到文本序列中，HATB 通过跨模态注意力稀疏地替换少量 transformer 层（Qwen2 的 28 层中仅 4 层 [0, 9, 17, 25]），大幅减少额外参数量和推理开销。

  实验比较：
  (a) **单图 VQA**（VQAv2, OK-VQA, GQA, VizWizQA, TextVQA）——对比 CogVLM/EVLM-Chat/Flamingo/Qwen-VL-Chat/InstructBLIP/mPLUG-Owl2/LLaVA-1.5/LLaVA-Next/VILA-1.5/Idefics2/Mantis-SigLIP。
  (b) **通用 MLLM Benchmark**（MMBench-EN/CN, MM-Vet, POPE, AI2D）——对比 OpenFlamingo/Cambrian/MiniCPM-Llama3-V2.5 等。
  (c) **视频理解**（NextQA, MVBench, VideoMME, LongVideoBench）——对比 VideoChat2/Video-LLaMA2/Video-ChatGPT/ShareGPT4Video/PLLaVA/Idefics2/Mantis-SigLIP/LLaVA-Interleave。
  (d) **多图理解**（NLVR2, Mantis-Eval, MathVerse-mv, SciVerse-mv, BLINK, Q-Bench2, MI-Bench）——对比 Qwen-VL-Chat/InstructBLIP/CogVLM/VideoLLaVA/VILA/Idefics2/Mantis-SigLIP/LLaVA-Interleave。
  (e) **消融实验**：cross-attention 集成方式（Concatenate vs Pre-Cross-Attention vs Post-Cross-Attention vs Hyper Attention）、Hyper Attention 层数（2/4/8 层）、Adaptive Gating/Shared LayerNorm/MI-Rope 的组件贡献。
  (f) **Distractor Resistance**（自建）：从 MMBench dev set 采样，随机插入 N-1 张干扰图（N=1,5,10,20,50,100,200,400），采用 CircularEval 评估模型抗干扰能力，对比 LLaVA-Next-Interleave/Mantis-Idefics2/Qwen-VL/mPLUG-Owl2。

- 硬件平台是什么，配置是什么。
  训练：多GPU，Stage 1 TP=1，Stage 2/3 TP=4（模型切分4份），ZeRO-1 优化，Mixed-precision FP16/BF16，Gradient Checkpointing。
  推理对比：使用 V100-32G 做效率对比（mPLUG-Owl3 输入 128 frames 可运行，LLaVA-Interleave 最多 ~20 images / 80GB VRAM）。

- 模型是什么。数据集和bench分别是什么。
  模型架构：Vision Encoder (Siglip-400m) → Linear Projection → Language Model (Qwen2)，含 Hyper Attention Transformer Block (HATB)。总参数量 ~8B。
  
  训练三阶段数据集：
  Stage 1 (Pretraining)：DataComp-1B, LAION-en, COYO-700M, COYO-700M-OCR, LAION-zh, Wukong, CC12M, CC3M, OCR-CC, COCO, SBU，~41M image-text pairs。
  Stage 2 (Multi-Image Training)：ShareGPTVideo, Selective Caption, LLaVA-Interleave, VATEX, Text Reading, Interleaved Caption, MMDU。
  Stage 3 (SFT)：LLaVA-SFT-665K, The Cauldron, Mantis, LLaVA-Interleave, ALLaVA, ShareGPTVideo-QA 240K, Video Instruct 100K, MSR-VTT/MSVD Caption。

  Benchmarks：VQAv2, OK-VQA, GQA, VizWizQA, TextVQA, MMBench-EN/CN, MM-Vet, POPE, AI2D, NextQA, MVBench, VideoMME, LongVideoBench, NLVR2, Mantis-Eval, MathVerse-mv, SciVerse-mv, BLINK, Q-Bench2, MI-Bench, Distractor Resistance（自建）。

- 开源情况。
  开源，代码仓库：https://github.com/X-PLUG/mPLUG-Owl
  
  基于开源文档和论文，Hyper Attention 的计算流程如下：
  ```
  # 输入：文本序列 S_text = [T1, T_img, T2, T_img, T3]
  #      图像特征 H_img = [I1^t, I2^t] ∈ R^{L×D_t}
  #      (经 Siglip-400m 提取 + Linear Projection 对齐维度)
  
  # 文本嵌入
  H_text = WordEmbed(S_text)  # shape: [L, D_t]
  
  # 对每一层 l in [0, 1, ..., N-1]:
  for l in range(N):
      H_text = H_text + SelfAttention(LayerNorm(H_text))  # 标准 self-attention
      
      if l in HATB_layers:  # 稀疏替换，如 [0, 9, 17, 25]
          # Hyper Attention: cross-attention 与 self-attention 并行
          # 1. 共享 LayerNorm
          H_text_norm, H_img_norm = LayerNorm_shared(H_text), LayerNorm_shared(H_img)
          
          # 2. Self-attention (与上述相同，此处简写)
          # 3. Cross-attention —— Query 来自文本, Key/Value 来自视觉
          Q = W_Q(H_text_norm)                  # 复用 self-attention 的 Q 投影
          K_img = W_img_KV(H_img_norm)[:D]      # 视觉专用 KV 投影 (modality-specific)
          V_img = W_img_KV(H_img_norm)[D:]
          
          # MI-Rope: 为视觉特征赋位置编码
          # 每张图 I_n 的所有 patch 共享其占位符 T_img 的 rotary position
          Q = apply_rotary_pos(Q, pos_text)
          K_img = apply_rotary_pos(K_img, pos_images)  # 复用占位符位置
          
          # Causal cross-attention mask: 每个 text token 只能 attend 前面的视觉特征
          H_cross = CrossAttention(Q, K_img, V_img, mask=causal_mask)
          
          # 4. Adaptive Gating —— 基于文本语义的门控
          g = Sigmoid(W_gate^T · H_text)           # g ∈ R^{L×1}
          H_fused = H_self * g + H_cross * (1 - g)  # 逐 token 融合
          
          H_text = H_fused
      
      H_text = H_text + FFN(LayerNorm(H_text))  # 标准 FFN
  ```
  
  关键设计要点：
  - W_img_KV 用 LLM 预训练 KV 权重初始化，仅 2D×D 额外参数
  - Adaptive Gate W_gate 是单层线性+Sigmoid，轻量
  - Shared LayerNorm 复用 transformer block 原有 LN，不复训
  - MI-Rope 为每张图的 all patches 赋共享位置编码（来自占位符 T_img 的位置索引）
  - Causal attention mask 确保文本只能 attend 前置图像，保持自回归特性

## VTPerception-R1: Enhancing Multimodal Reasoning via Explicit Visual and Textual Perceptual Grounding

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：**VTPerception-R1**，一个两阶段训练框架，显式地将感知（perception）与推理（reasoning）解耦，增强多模态大语言模型的感知接地（perceptual grounding）能力。核心包括：
  
  **Stage I — Perception-Augmented SFT**：将原始 CoT 数据转换为结构化目标格式 `<description>...<think>...<answer>`，其中 `<description>` 专门总结与问题相关的视觉/文本证据（非通用 caption），`<think>` 保留原始推理链，`<answer>` 给出最终解。训练目标为 token 级交叉熵损失 L_SFT = -Σ_t log π_θ(y_t | x, y_<t)。SFT 数据从 LLaVA-CoT (4K) 和 Vision-SR1 (8K) 采样共约 12K 样本，经过自动清洗流水线（VLM dense caption + Grounding DINO + EasyOCR → 结构化描述 → LLM 重建 CoT → 多维度质量评分过滤）。

  **Stage II — Perception-Aware RL**：基于 DAPO（Decoupled Clip and Dynamic sAmpling Policy Optimization）目标，引入六个奖励项：
  - R_acc：答案正确性奖励
  - R_fmt：格式合规奖励（强制 `<description> → <think> → <answer>` 模板）
  - R_rep：重复惩罚（惩罚重复 n-gram，鼓励简洁描述）
  - R_vkey：视觉关键信息奖励，衡量 `<description>` 覆盖关键视觉元素的比例，离散化为三档（1.0 / 0.5 / 0.0）
  - R_tkey：文本关键信息奖励，衡量 `<think>` 覆盖关键文本元素（OCR 文本、数值、单位、约束、常识）的比例
  - R_cons：描述-推理一致性奖励，检查 `<think> + <answer>` 中的实体/属性/数值是否被 `<description> + question` 的证据支持，存在冲突时直接为 0
  - 采用 perception-first 加权调度：训练早期侧重感知接地，后期切换到正确性

  实验比较：对比以下基于 Qwen2.5-VL-7B 的 baseline 方法——
  - Vision-SR1-7B（自奖励视觉推理）
  - Vision-R1-7B（cold-start + RLVR）
  - Perception-R1-7B（视觉感知奖励）
  - Visionary-R1（caption→reason→answer 结构）
  - MM-Eureka-Qwen-7B（GRPO + 规则奖励）
  - VL-Rethinker-7B（自反思强化学习）
  - Qwen2.5-VL-7B-Instruct（原始指令微调基座）
  - VTPerception-R1-7B (Before RL)（SFT 后 RL 前的自身消融）
  在 MathVista、MMMU、EMMA、AI2D、Creation-MMBench、C-MMBench-TO 六个 benchmark 上评估。

- 硬件平台是什么，配置是什么。
  SFT 阶段：DeepSpeed ZeRO-3 + bf16 精度 + 梯度检查点。RL 阶段：基于 EasyR1-perc 框架，使用 Ray 分布在 1 个主节点 + 1 个 ORM 节点上，Tensor Parallel Size = 4。论文未明确说明 GPU 型号和数量。

- 模型是什么。数据集和bench分别是什么。
  模型：**Qwen2.5-VL-7B-Instruct** 作为基座模型，全参数微调。
  
  训练数据：
  - SFT 数据：从 LLaVA-CoT（4K）和 Vision-SR1（8K）采样合并 ~12K 样本，经自动化清洗流水线处理
  - RL 数据：从 MMK12（5K）、LLaVA-CoT（5K）、Vision-R1-rl（5K）、Mulberry（5K）聚合 ~22K 多模态推理样本，覆盖数学、科学、图表理解；通过教师模型集成 + 预算验证 + 关键信息提取流水线构建
  
  训练超参数：SFT 阶段 lr=1e-5, weight decay=0.1, batch size=1, gradient accumulation=8, 3 epochs。RL 阶段训练 2 epochs。
  
  Benchmarks（6 个）：MathVista（6141 题，数学推理）、MMMU（11.5K 题，大学多学科）、AI2D（科学图表理解）、EMMA（数理化学科交叉推理）、Creation-MMBench（765 实例，51 个细粒度任务）、C-MMBench-TO（纯文本变体）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源代码：https://github.com/yizhuoDi/VTPerceprion-R1
  
  算法 Pipeline 伪代码：
  ```
  # ===== Stage I: Perception-Augmented SFT =====
  # 输入: 图像 x_img, 问题 q
  # 目标: 模型生成 <description> d, <think> t, <answer> a
  
  d = model.generate(x_img, q, prefix="<description>")
  # d 仅包含与问题相关的视觉/文本证据，非通用 caption
  
  t = model.generate(x_img, q, d, prefix="<think>")
  # t 为基于 d 的推理链
  
  a = model.generate(x_img, q, d, t, prefix="<answer>")
  # a 为最终答案
  
  L_SFT = CrossEntropy(concat(d, t, a), target)
  # 目标序列: <description>...<think>...<answer>
  
  # ===== Stage II: Perception-Aware RL (DAPO) =====
  # 对每个 prompt x，采样 G 个响应 {o_i}
  # o_i 格式: <description> d_i <think> t_i <answer> a_i
  
  # 计算各奖励分量:
  R_acc = exact_match(a_i, ground_truth)         # 答案正确性
  R_fmt = check_format(o_i)                        # 模板合规
  R_rep = -count_repeated_ngrams(o_i)             # 重复惩罚
  
  # 视觉关键信息奖励:
  D = extract_facts(d_i)                          # 从 description 提取事实
  cov_v = |K_v ∩ D| / |K_v|                        # K_v 为标注的关键视觉线索
  R_vkey = 1.0 if cov_v >= τ_hi else (0.5 if cov_v >= τ_lo else 0.0)
  
  # 文本关键信息奖励:
  D_t = extract_facts(t_i)                        # 从 think 提取事实
  cov_t = |K_t ∩ D_t| / |K_t|                      # K_t 为标注的关键文本线索
  R_tkey = 1.0 if cov_t >= τ_hi else (0.5 if cov_t >= τ_lo else 0.0)
  
  # 一致性奖励:
  F_ans = extract_entities(t_i) ∪ extract_entities(a_i)
  E = extract_entities(d_i) ∪ extract_entities(q)
  cons = |F_ans ∩ E| / max(1, |F_ans|)
  R_cons = 0 if has_conflict(F_ans, E) else cons
  
  # 总奖励:
  R_i = R_acc + R_fmt + R_vkey + R_tkey + R_rep + R_cons
  
  # DAPO 目标 (token-level clipped policy gradient):
  A_i_t = normalize(R_i - mean({R_j}))  # group-relative advantage
  r_i_t(θ) = π_θ(o_i_t | ...) / π_old(o_i_t | ...)
  J(θ) = E[ 1/(Σ|o_i|) Σ_i Σ_t min(r_i_t * A_i_t, clip(r_i_t, 1-ε_low, 1+ε_high) * A_i_t) ]
  ```
  
  关键设计：
  - DAPO 四项技术：非对称裁剪（ε_low ≠ ε_high）、动态采样、token 级优化、过长响应惩罚
  - 排除所有 rollout 全对或全错的无效 group，避免退化更新
  - Perception-first 加权调度：早期训练增大 R_vkey 和 R_tkey 权重，后期切换到 R_acc

  - 开源情况：代码已开源在 GitHub，基于 EasyR1-perc 框架（DAPO 实现）和 DeepSpeed ZeRO-3。
