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
