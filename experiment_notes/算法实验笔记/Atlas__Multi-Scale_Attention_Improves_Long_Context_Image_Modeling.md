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
