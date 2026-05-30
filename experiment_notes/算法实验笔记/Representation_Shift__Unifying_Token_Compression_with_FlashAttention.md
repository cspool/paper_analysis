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
