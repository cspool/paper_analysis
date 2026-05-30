## Cluster-Driven Expert Pruning for Mixture-of-Experts Large Language Models

- baseline方法是什么？
  **Baseline 为未剪枝的 MoE LLM（DeepSeek-V2-Lite / Qwen1.5-MoE-A2.7B）以及已有的 expert pruning 方法（Random Pruning、Seer Prune、Group&Merge）。**

  **已有剪枝方法的原理**：
  - Random Pruning：随机选择 experts 删除。
  - Seer Prune：基于 gate activation 统计（推理时收集 gate 激活频率），保留最频繁激活的 experts，删除低频 experts。
  - Group&Merge：将专家按某种相似度分组后合并。

  **这些 baseline 方法的共性问题**：
  1. 忽略 **intra-layer expert homogeneity**（层内专家功能冗余）：同一层内多个专家因训练动态发展出功能重叠。
  2. 忽略 **inter-layer similarity patterns**（跨层相似模式）：深层比浅层包含更多同质专家，冗余度随深度递增。
  3. 将专家视为独立单元处理，缺乏跨层全局视角。

  **Baseline 全栈执行例子（以 DeepSeek-V2-Lite, Seer Prune 为例）**：

  - **算法层**: 在 calibration 数据上收集每层每个 expert 的 gate 激活次数 → 按激活频率排序 → 保留 top-k 高频 experts，删除其余 → 路由权重不做调整（直接丢弃被删 expert 的路由条目）。该方法仅依赖 gate 统计信号，不考虑 expert 参数本身的功能相似性。
  - **系统框架层**: 论文未明确说明推理框架。使用 HuggingFace transformers 标准推理流程。Seer Prune 等 baseline 方法在模型层面操作（修改 model.state_dict()），不涉及推理框架修改。
  - **编译框架层**: 论文未明确说明（使用 PyTorch + HuggingFace transformers 标准推理）。
  - **Kernel/运行时调度层**: 论文未明确说明。MoE 层使用标准 sparse MoE kernel（top-k gating + 分组 GEMM）。剪枝后 expert 数量减少，对应 expert FFN 的 GEMM 计算量减少。
  - **硬件架构层**: 32× NVIDIA A100 80GB GPU 集群，无自定义硬件。

  **Baseline 的核心缺陷**：
  1. **Gate-only 信号局限**：Seer Prune 仅依赖 gate 激活频率判断专家重要性，忽略专家参数本身的功能相似性。两个功能几乎相同的专家可能都获得较高激活频率，导致冗余未被识别和消除。
  2. **层隔离假设失效**：现有方法逐层独立决策剪枝，忽略了深层专家更同质的趋势。Group&Merge 虽考虑了层内相似性，但未利用跨层同质模式优化全局剪枝策略。
  3. **丢弃式剪枝损失信息**：Random Pruning 和 Seer Prune 直接丢弃被剪专家及其路由权重，导致的功能损失无法恢复。Group&Merge 虽有合并但无权重自适应调节。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **论文方法**: C-PRUNE = Layerwise Expert Clustering（层内聚类）+ Global Cluster Pruning（全局聚类剪枝）+ Parameterized Expert Merging（参数化专家合并）。

  **Defect → Design 映射**：

  | Baseline 缺陷 | C-PRUNE 设计选择 | 解决机制 |
  |---|---|---|
  | Gate-only 信号忽略参数相似性 | Expert Embedding: 用 calibration 数据计算每个 expert 的实际输出向量 φ(f_i) 作为功能特征，而非 gate 激活次数 | 通过参数空间分析（cosine similarity）直接衡量专家功能的冗余度，而非间接依赖路由统计。Expert activation-based embedding 捕获了专家的实际计算行为 |
  | 层隔离假设，忽略跨层同质趋势 | Global Cluster Pruning: 建立跨所有层的 unified importance score，惩罚深层 expert（depth penalty），全局统一排序决定剪枝 | 利用"深层专家更同质"的先验，在全局优化中对深层施加更大的剪枝压力。Layerwise ratio 0.2 + Global ratio 0.1 的配置达到最佳 trade-off |
  | 丢弃式剪枝损失信息 | Parameterized Expert Merging: 剪枝 cluster 中所有专家通过 weighted averaging 合并为一个专家，ω_i ∝ exp(γ·A_ik)（相似度越高权重越大）；路由权重更新时加入 exploration noise 维持多样性 | 保留被剪专家的功能信息（加权合并），同时通过温度 γ 控制融合锐度，避免信息丢失 |
  | 无自适应剪枝阈值 | Adaptive Clustering Threshold τ^(l) = mean_deviation + δ·σ^(l)，层越深阈值越大 | 根据每层的实际专家嵌入分布自适应调节聚类半径，深层允许更大的聚类（更大的冗余容忍度） |

  **C-PRUNE 全栈执行例子**：

  - **算法层**: 
    1. Expert Embedding: 在 task-specific calibration 数据上前向传播，每层每个 expert 取 K 个样本输出的均值作为嵌入 φ(f_i) ∈ R^d
    2. 亲和矩阵: A_ij = σ(α·cos(φ(f_i), φ(f_j)))
    3. Hierarchical Agglomerative Clustering: 层内自底向上合并最相似的 expert/cluster
    4. 聚类后合并: θ̂_k = Σ softmax(γ·A_ik)·θ_i（保留信息，而非丢弃）
    5. 全局剪枝: 跨层统一评分，深层专家受 depth penalty 更可能被剪
    6. 路由更新: Ŵ_k = mean(W_i) + ε·N(0,I)
    7. 可选 task-specific fine-tuning: 在目标 domain 数据上微调剪枝后模型
    结果：20% pruning rate 下，DeepSeek 15.7B→13.0B，MMLU 仅降 1.4%（vs Random 降 64%）；Qwen 14.3B→11.8B，保留 88% MMLU。GSM8K 上 C-PRUNE 反超 base model（33.56 vs 32.21）。

  - **系统框架层**: 论文未明确说明推理框架。C-PRUNE 在 HuggingFace transformers 上实现，对 model 的 expert parameters 进行修改（合并/删除）。不涉及 Serving 框架的调度修改。

  - **编译框架层**: 论文未明确说明（使用 PyTorch 标准推理流程）。剪枝后的 MoE 模型可无缝加载到标准推理框架中，因为 expert 合并后模型结构不变（仍是 MoE FFN layers，仅 expert 数量减少）。

  - **Kernel/运行时调度层**: 论文未明确说明具体 kernel 实现。剪枝后 routed experts 减少（64→52），每个 token 的 MoE 计算量减少（top-8 from 64 candidates vs from 52 candidates），但 top-k gating + 分组 GEMM 的 kernel 模式不变。论文提到 1.2× 推理加速（来自 expert 数量减少带来的计算量下降）。

  - **硬件架构层**: 32× NVIDIA A100 80GB GPU，无自定义硬件。

  **对比 Baseline 的核心改进路径**：
  ```
  Baseline (Seer Prune):
  Calib Data → Gate Forward → Act Count → Sort by Freq → Discard Low-Freq Experts → Pruned Model
  
  C-PRUNE (Ours):
  Calib Data → Expert Forward → Expert Embedding φ(f_i) 
  → Cosine Affinity Matrix A → Hierarchical Clustering (per-layer)
  → Weighted Expert Merging (within clusters) → Global Importance Scoring (cross-layer)
  → Global Pruning + Routing Update → Pruned Model
  → (Optional) Task-Specific Fine-tuning
  ```
