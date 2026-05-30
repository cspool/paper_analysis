## Expert Functional Redundancy in MoE (MoE 中的专家功能冗余)

术语解释
Expert Functional Redundancy 是 BuddyMoE 中提出的核心观察：大型 MoE 模型中多个 expert 存在功能相似性（functional similarity），即不同 expert 学习到相似或重叠的函数映射能力，这一观察构成了 buddy expert substitution 的理论基础。

术语是什么？
MoE 模型的 expert 功能冗余表现在两个层面：(1) **输出相似性**：多个 expert 对同一输入的 hidden states 输出高度相似；(2) **共激活模式**：特定 expert pairs 经常被同一 token 同时选中。实证证据包括 BuddyMoE 论文 Figure 4 的 expert similarity heatmap（大量 bright 区域表示 high similarity）和 Figure 7 的 co-activation heatmap（sparse but high-intensity patterns）。Prior work 确认 MoE 模型可容忍 aggressive pruning to 4 bits with minimal quality loss，间接验证了 expert 冗余。

C-PRUNE 论文进一步揭示了两个结构化的冗余层次：(1) **Intra-layer Expert Homogeneity（层内专家同质性）**：同一 MoE 层内的 expert 因训练动态发展出功能重叠，在参数空间中表现为高 cosine similarity（Figure 1 的 layer-specific heatmaps）；(2) **Inter-layer Similarity Patterns（跨层相似模式）**：深层（deeper layers）的 expert 比浅层更同质——rightmost heatmap 显示全局相似度随层深度递增。C-PRUNE 利用此观察在全局剪枝中施加 depth penalty，对深层 expert 给予更高的剪枝概率。

从算法pipeline角度拆解术语：
BuddyMoE 利用 expert 功能冗余的四阶段 pipeline：(1) 离线 profiling 量化冗余——在 calibration corpus 上记录 pairwise co-activation count M[i][j]；(2) 条件共激活分布计算 q_{j|i} = M[i][j] / Σ M[i][j']；(3) CFT buddy list 构建——对每个 pivot i，按 q_{j|i} 降序排列，选前缀覆盖 α 比例累积激活 mass；(4) 运行时替代——缺失 expert 被 GPU-resident buddy 替代，~0ms vs ~10ms CPU→GPU 传输。Layer-wise heterogeneity：早期层呈现 broader redundancy（更 diffuse），后期层更 specialized（tighter clusters）。

术语一般如何实现？如何使用？
- 冗余度量方法：co-activation frequency、output similarity（cosine/MSE）、或组合
- Profiling dataset 需匹配部署领域以准确反映路由行为
- 冗余是 BuddyMoE、expert pruning、expert merging 等技术的共同前提
- 关键限制：冗余程度因模型架构（expert 数量、gating 策略、训练数据）而异，需 per-model profiling

涉及论文标题：
- BuddyMoE Exploiting Expert Redundancy to Accelerate Memory-Constrained Mixture-of-Experts Inference
- Cluster-Driven Expert Pruning for Mixture-of-Experts Large Language Models

---
