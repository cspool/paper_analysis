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
