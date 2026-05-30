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
