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
