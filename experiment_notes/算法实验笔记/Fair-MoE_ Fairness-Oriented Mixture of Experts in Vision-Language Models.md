## Fair-MoE: Fairness-Oriented Mixture of Experts in Vision-Language Models

- 属于算法pipeline的实现是什么？实验比较什么？
  - Fair-MoE 提出面向医疗 VLM 公平性的 MoE 框架，包含两个核心组件：
    1. **FO-MoE（Fairness-Oriented Mixture of Experts）**：在 CLIP 的图像和文本 encoder 中引入两类 MoE 层——**patch embedding-based MoE**（替换最后一个 attention block 的 MLP 层）和 **feature-based MoE**（放置在 encoder 之后）。Embedding-based MoE 通过专家容量（capacity C）和 top-c 筛选过滤偏置 patch embedding；Feature-based MoE 进一步消除偏置特征，提取公平的任务相关特征。两类 MoE 均采用 sparse gating：`W = softmax(G(I))`，`Ŵ = Top_c(Top_r(W, k), α)`，其中 α = C(N+1)k/M。
    2. **FOL（Fairness-Oriented Loss）**：由五部分组成——F_EI（图像 embedding-based MoE 方差优化）、F_ET（文本 embedding-based MoE 方差优化）、F_FI（图像 feature-based MoE 方差优化）、F_FT（文本 feature-based MoE 方差优化）和 L_distance（Sinkhorn distance loss）。FOL 的核心创新是将 MoE load balance 中使用的 variance 度量同时用于 fairness：`F_EI = Σ_{p∈P} Σ_{j=0}^{M^1-1} (Var(O_{N_j}) - Var(O_{N|p_j}))^2`，同时优化不同属性组分布的距离（Sinkhorn）和离散度（variance difference）。
  - 实验比较 Fair-MoE 与 CLIP（Vanilla，b16/l14）和 FairCLIP（SOTA 公平性 VLM，b16/l14）在青光眼诊断任务上的公平性和准确性。消融研究包括：FO-MoE 组件有效性、FOL 各子损失有效性、embedding-based vs feature-based MoE、Text vs Image MoE 模块。

- 硬件平台是什么，配置是什么。
  - 单张 NVIDIA GeForce RTX 3090 GPU（24GB 显存）。
  - 训练协议与 FairCLIP 保持一致。

- 模型是什么。数据集和bench分别是什么。
  - 模型：基于 CLIP 架构（ViT-B/16 ~200M 参数，ViT-L/14 ~500M 参数）。对比模型包括 CLIP/b16, CLIP/l14, FairCLIP/b16, FairCLIP/l14, FairMoE/b16, FairMoE/l14。
  - 数据集：Harvard-FairVLMed（青光眼多模态数据集），7000 训练 / 1000 验证 / 2000 测试样本，每样本包含 SLO 眼底图像 + 临床笔记 + 标签，4 个受保护属性：Race（种族）、Gender（性别/GEN）、Ethnicity（民族/ETH）、Language（语言/LAN）。
  - Benchmark 指标：
    - AUC（Area Under the Curve）：整体性能
    - DPD（Demographic Parity Difference）：公平性，衡量不同组获得正向结果的概率差
    - EOD（Equal Opportunity Difference）：公平性，同时考虑 TPR 和 FPR
    - ES-AUC（Equity-Scaled AUC）：性能与公平性的权衡，`ES-AUC_s = AUC_s / (1 + Σ_a |AUC_s - AUC_{s,a}|)`

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - **开源情况**：代码已开源在 https://github.com/LinjieT/Fair-MoE-Medical-Fairness-Oriented-Mixture-of-Experts-in-Vision-Language-Models。论文发表于 MICCAI 2025，arXiv: 2502.06094。
  - **算法 pipeline 解释**：

  Fair-MoE 基于 CLIP 的对比学习框架，在图像和文本 encoder 中替换/增加 MoE 层：

  **前向流程（单张图像 + 文本对）**：
  ```
  # 图像侧
  I_image = ViT_patch_embed(image)              # (N+1)×D patch embeddings
  I_enc = attention_blocks[0..K-2](I_image)     # 前 K-1 个 attention block
  I^1 = I_enc                                     # 输入 embedding-based MoE
  W^1 = softmax(G^1(I^1))                         # Gate: R^{(N+1)×D} → R^{(N+1)×M^1}
  Ŵ^1 = Top_c(Top_r(W^1, k^1), α)                 # Sparse + capacity filtering
  I^2_a = Σ_{b=0}^{M^1-1} Ŵ^1_{a,b} · E^1_b(I^1_a)  # Expert 加权聚合
  I_feat = I^2_0                                  # 取 [CLS] token
  W^2 = Top_r(softmax(G^2(I_feat)), k^2)          # Feature-based MoE gate
  I^3 = Σ_{b=0}^{M^2-1} Ŵ^2_b · E^2_b(I_feat)     # Fair image feature

  # 文本侧（对称结构）
  T_text = tokenize + embed(report)              # L×D text embeddings
  # ... 同样经过 embedding-based MoE 和 feature-based MoE ...
  T^3 = fair_text_feature                         # Fair text feature

  # 对比学习 + FOL
  similarity = cosine(I^3, T^3)
  L_CLIP = contrastive_loss(similarity, labels)
  L_FOL = F_EI + F_ET + F_FI + F_FT + L_distance
  L_total = L_CLIP + λ · L_FOL
  ```

  **FOL 方差优化核心逻辑（以图像 embedding-based MoE 为例）**：
  ```
  # 从整个数据集和特定属性组分别采样 N 个 batch
  # 收集 gate weights 矩阵 O_N, O_{N|p}
  for p in ProtectedAttributes:  # race, gender, ethnicity, language
      for j in range(M^1):       # 每个 expert
          loss += (Var(O_N[:, j]) - Var(O_{N|p}[:, j]))^2
  ```

  **Expert 结构**：
  ```
  E^1_b(x) = T̃^1_b · σ(W̃^1_b · x)   # 两层 MLP + 激活函数
  ```
  其中 σ 为激活函数，W̃^1_b 和 T̃^1_b 为可学习参数。

  **对比学习范式**：论文遵循 CLIP 的对比学习训练方式，将匹配的图像-文本对作为正样本，不匹配的作为负样本，通过 InfoNCE loss 进行优化，并在其上叠加 FOL。
