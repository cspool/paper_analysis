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
