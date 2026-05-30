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
