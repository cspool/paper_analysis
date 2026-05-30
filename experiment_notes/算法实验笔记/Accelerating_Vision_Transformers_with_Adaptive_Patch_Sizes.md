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
