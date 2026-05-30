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
