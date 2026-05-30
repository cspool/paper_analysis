## Fair-MoE: Fairness-Oriented Mixture of Experts in Vision-Language Models

- baseline方法是什么？
  - **CLIP (Vanilla)**：标准的 Vision-Language Model，使用 ViT 作为图像 encoder、Transformer 作为文本 encoder，通过对比学习对齐图像和文本 embedding。CLIP 的 encoder 对所有输入不加区分地通过 MLP 层处理，可能无意识地学习偏置信息。
  - **FairCLIP (SOTA fair VLM)**：基于 CLIP 架构，通过最小化不同受保护属性组分布之间的 Sinkhorn distance 来增强公平性。但 FairCLIP 保留了 CLIP 的原始架构，未针对公平性进行特定架构适配。
  - 全栈执行例子（以 FairCLIP/b16 在 Harvard-FairVLMed 青光眼诊断任务为例）：
    - **模型推理算法层**：CLIP 对比学习框架。图像经过 ViT-B/16 encoder（12 层 Transformer blocks），每层包含 multi-head self-attention + MLP。文本经过 Transformer encoder 对称处理。最终通过 cosine similarity 匹配图像-文本对。FairCLIP 在原 CLIP loss 基础上加入 Sinkhorn distance loss，最小化不同属性组（race/gender/ethnicity/language）embeddings 分布之间的距离。
    - **系统框架层**：PyTorch + HuggingFace Transformers，标准训练循环。论文未明确说明具体的训练框架细节。
    - **编译框架层**：论文未明确说明。
    - **kernel调度层**：论文未明确说明。标准 PyTorch CUDA kernel 执行。
    - **硬件架构层**：NVIDIA GeForce RTX 3090 (24GB)。
  - **Baseline 痛点**：
    1. **CLIP 架构无偏置过滤能力**（核心痛点）：CLIP 的 encoder 不加区分地处理所有 patch embedding，模型可能从偏置的 patch 中学习到与受保护属性（如种族、性别）相关的 spurious correlation，而非真正的疾病特征。在 Harvard-FairVLMed 上，CLIP/b16 在 Race 属性上的 DPD=14.57、EOD=18.47（数值越大越不公平）。
    2. **FairCLIP 仅通过 loss 约束公平性**：FairCLIP 仅在损失函数层面通过 Sinkhorn distance 约束来缩小不同组分布之间的距离，但架构层面没有任何机制来过滤或抑制偏置特征的提取。这导致 FairCLIP/l14 在 Race 上 DPD=16.01（甚至比 CLIP 更差），说明单纯的距离最小化不足以保证公平性。
    3. **loss 设计仅关注分布距离**：现有 fairness loss（包括 FairCLIP 的 Sinkhorn distance）仅最小化不同属性组分布之间的距离，忽略了分布离散度（dispersion/variance）的作用。方差既是 MoE load balancing 的关键（影响训练稳定性和模型容量利用），也是 fairness 的重要度量（不同组的方差差异过大意味着某组内个体差异被系统性放大或压缩）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **Fair-MoE 方法**：通过两个关键组件——FO-MoE（架构层面）和 FOL（损失函数层面）——同时解决上述三个痛点：
    1. **FO-MoE（Fairness-Oriented Mixture of Experts）**（解决痛点 1 和 2）：在图像和文本 encoder 中引入两级 MoE——
       - **Embedding-based MoE**：替换最后一个 attention block 的 MLP 层，使用 sparse gating + expert capacity 机制：`Ŵ^1 = Top_c(Top_r(W^1, k^1), α)`。`Top_c` 通过 capacity C 限制每个 expert 处理的 patch 数量，只有权重最高的 α = C(N+1)k/M 个 patch 被各 expert 处理，其余被清零。这使模型能够**主动过滤偏置 patch embedding**——偏置 patch（如包含肤色、性别特征信息的图像区域）分配到不相关的 expert 或权重过低而被过滤，仅保留与疾病诊断任务相关的公平特征。
       - **Feature-based MoE**：放置在 encoder 之后，对 [CLS] feature 做进一步 sparse gating，消除编码后的偏置特征，提取最终公平的特征向量供对比学习使用。
    2. **FOL（Fairness-Oriented Loss）**（解决痛点 3）：在 Sinkhorn distance（L_distance）基础上新增四个方差优化项——
       - `F_EI = Σ_{p∈P} Σ_{j=0}^{M^1-1} (Var(O_{N_j}) - Var(O_{N|p_j}))^2`（图像 embedding-based MoE）
       - 类似地定义 F_ET、F_FI、F_FT
       - 核心思想：对整个数据集的 gate weight 方差 Var(O_N) 和各属性组内 gate weight 方差 Var(O_{N|p}) 之间的差异进行惩罚，使不同组的 gate weight 分布不仅**位置接近**（距离最小化）而且**形状一致**（方差对齐）。这同时服务于 load balancing（MoE 训练稳定性）和 fairness（各组专家使用模式一致）。

  - 全栈执行例子（与 baseline 同配置，FairMoE/l14）：
    - **模型推理算法层**：基于 CLIP/ViT-L/14，在图像和文本 encoder 的最后一个 attention block 中用 embedding-based MoE（M^1 experts）替换 MLP，在 encoder 输出后增加 feature-based MoE（M^2 experts）。Gate 为可学习 MLP，sparse top-k 路由 + capacity filtering。对比学习 loss 加上 FOL = F_EI + F_ET + F_FI + F_FT + L_distance。**架构改变使模型能主动过滤偏置信息而非被动地仅通过 loss 约束**。
    - **系统框架层**：与 baseline 相同（PyTorch + HuggingFace Transformers），标准训练循环。
    - **编译框架层**：论文未明确说明。
    - **kernel调度层**：论文未明确说明。标准 PyTorch CUDA kernel 执行。MoE 的 sparse gating 通过 mask 操作实现，无额外通信开销（单 GPU 训练）。
    - **硬件架构层**：与 baseline 相同（NVIDIA RTX 3090）。
    
    关键性能对比（FairMoE/l14 vs baselines，Harvard-FairVLMed）：
    - **Race**: ES-AUC 72.53（+5.00% vs FairCLIP/l14），AUC 73.93（+2.36%），DPD 2.63（↓83.6% vs FairCLIP/l14），EOD 4.25（↓75.1% vs FairCLIP/l14）
    - **Gender**: ES-AUC 69.97（+2.60% vs FairCLIP/l14），AUC 74.97（+4.17% vs FairCLIP/l14）
    - **Ethnicity**: ES-AUC 67.10（+2.87% vs FairCLIP/l14），DPD 8.79（↓42.8% vs FairCLIP/l14）
    - **Language**: ES-AUC 63.80（+0.23% vs FairCLIP/l14），AUC 71.37
    
    消融关键发现：
    - 移除 FOL → Race AUC 下降 2.56%，Gender ES-AUC 下降 2.34%，验证方差优化的必要性
    - 移除 embedding-based MoE → Race ES-AUC 从 70.9 降至 66.2，验证 patch 级偏置过滤的有效性
    - 移除 Image MoE → Language ES-AUC 从 66.1 降至 54.8（最大降幅），验证图像侧公平性更关键
