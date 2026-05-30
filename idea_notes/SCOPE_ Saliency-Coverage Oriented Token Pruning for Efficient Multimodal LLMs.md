## SCOPE: Saliency-Coverage Oriented Token Pruning for Efficient Multimodal LLMs

- baseline方法是什么？
  Baseline 是现有的 **saliency-based visual token pruning** 方法（FastV、SparseVLM、VisionZip、PDrop），其核心策略是：利用 attention scores（如 CLS token 到 visual token 的 attention、或 text-to-vision attention）对 visual token 的重要性进行排序，然后保留 Top-K 个最高 attention 的 token，丢弃其余。

  Baseline（以 FastV/LLaVA-1.5 7B 为例）全栈执行例子：
  - 算法层：图像 → CLIP ViT-L/14 → 576 visual tokens → 与 text tokens 拼接 → LLaVA-1.5 LLM 前几层处理 → 提取早期层 text-to-vision attention scores → Top-K 选择 highest attention visual tokens（如 K=64） → 丢弃其余 512 tokens → 保留的 64 tokens 送入 LLM 剩余层 → 自回归生成答案。问题：(1) saliency-based 方法仅关注高 attention token，导致大量语义信息被丢弃（如问题 "Where is the cat?" 时 attention 集中于 cat 而忽略周围环境 context）；(2) attention 分布高度偏斜——少数 token 获得极高 attention，其余 token attention 接近均匀（flat tail），难以区分 informative vs redundant tokens；(3) 论文图 2 的 θ-coverage 分析显示 saliency-only 的语义覆盖度甚至低于 random selection baseline。
  - 系统框架层：HuggingFace Transformers + lmms-evals 评估框架。剪枝模块插入 vision encoder 之后、LLM 之前。无需额外训练。
  - 编译框架层：论文未明确说明。
  - kernel调度层：标准 FlashAttention。无自定义 kernel 修改。
  - 硬件架构层：NVIDIA A100 GPU（4×A100 用于实验评估）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法：SCOPE 通过 **联合建模 saliency 和 coverage** 解决 baseline 缺陷，将 token 选择从"仅按 attention 排序"改为"迭代贪心最大化 SCOPE score"。

  **(a) 缺陷1：语义完整性缺失（semantic incompleteness）** → 引入 set-coverage 概念
  Baseline 仅基于 saliency 选择 token，丢弃了大量语义 context（如 cat 周围的场景信息）。SCOPE 定义 token 集的 coverage：C(u, S) = max_{s∈S} sim(u, s)（即每个 full-set token u 是否被 S 中至少一个 token 以 cosine similarity 覆盖），定义了 θ-coverage 指标量化语义覆盖度。实验证明 SCOPE 的 θ-coverage 显著高于 saliency-only。

  **(b) 缺陷2：attention 分布偏斜导致 token 区分度低** → Token-coverage Gain 机制
  Baseline 中 tail tokens 的 attention 值几乎相同（flat），导致无法区分真正有用的 tokens。SCOPE 不依赖单一的 attention 排序，而是计算每个候选 token v 的 marginal gain Δ(v; S) = Σ_{u∈V} max(C(u, S), sim(u, v)) - C(u, S)，量化 v 加入后能为全体 token 提供多少额外 coverage。这使具有不同语义特征的 token（即使 attention 较低）也能因其提供的 coverage 增益而被选中。

  **(c) 缺陷3：saliency 和 coverage 未被联合优化** → SCOPE Score = Δ(v; S) · A_v^α
  SCOPE 通过乘积形式整合 saliency 和 coverage：argmax Δ(v; S) · A_v^α，α=1.0 为默认缩放因子。迭代选择过程中，第一步偏好高 attention 的显著 token（如 cat），后续步骤则倾向于提供新 coverage 的 token（如场景背景），逐步实现对图像整体的"显著+覆盖"平衡。消融实验（Table 4）证实：Ours（saliency+coverage）> Coverage-only > Saliency-only > Random。

  对比 baseline 的全栈执行例子（SCOPE, LLaVA-1.5 7B, K=64 tokens）：
  - 算法层：图像 → CLIP ViT-L/14 → 576 visual tokens V → 提取 layer -2 的 CLS-to-visual attention A_v（saliency） + 预计算 576×576 cosine similarity 矩阵 S_{uv} → 初始化 S=∅, c_u=0 → 迭代 64 次：
    1. ∀v ∉ S: Δ(v; S) = Σ_{u} max(S_{uv}, c_u) - c_u
    2. score(v) = Δ(v; S) · A_v^α
    3. v* = argmax score(v); S = S ∪ {v*}; c_u = max(c_u, S_{uv*})
    → 输出 64 个选定 token → 与 text token 拼接送入 LLM → 自回归生成答案。
    额外开销：token 相似度矩阵 O(N²)=576²≈332K 对（可预先计算），每轮选择 O(N²) 迭代 64 次。在 4×A100 上，从 2880→160 tokens 时端到端延迟 188.8s vs full 601.9s（3.2× speedup），POP E 性能 81.3% vs full 86.4%。
  - 系统框架层：lmms-evals + HuggingFace Transformers。剪枝模块插入 vision encoder 之后。Training-free，不修改 LLM weights。
  - 编译框架层：论文未明确说明。
  - kernel调度层：标准 FlashAttention。剪枝后 token 数减少使 attention 计算量按 O(K²/N²) 比例降低。
  - 硬件架构层：4×NVIDIA A100 GPU。推理 batch size=1。效率分析（Table 5）显示从 2880→160 tokens 延迟降为 3.2×（从 601.9s→188.8s），比 PDrop 的 3.3× speedup（184.0s）略慢但以远高准确度完成（SCOPE 81.3% vs PDrop 53.2%）。

  核心贡献总结：
  - 首次揭示 saliency-based token pruning 的语义完整性缺失问题，并通过 θ-coverage 定量分析验证。
  - 将 submodular coverage maximization 引入 visual token pruning，提出 SCOPE score 联合优化 saliency 和 coverage。
  - 在 LLaVA-1.5 7B/13B、LLaVA-Next 7B/13B、Video-LLaVA、Qwen2-VL 等多模型上验证，极端压缩（K=64/576, ↓88.9%）下保持 96.0% 性能。
