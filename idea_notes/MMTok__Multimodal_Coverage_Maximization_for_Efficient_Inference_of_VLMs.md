## MMTok__Multimodal_Coverage_Maximization_for_Efficient_Inference_of_VLMs

- baseline方法是什么？
  Baseline 方法为现有的 training-free 视觉 token 剪枝算法：(1) **FastV** (vision-only) —— 根据深层 LLM 层中 vision token 的 attention 分数排序，丢弃低 attention token。(2) **SparseVLM** (language-only) —— 利用 text-to-vision attention 分数评估 vision token 重要性，但忽略全局图像信息。(3) **VisionZip** (CLS-attention-based) —— 使用 [CLS] token 对各 vision token 的预训练 attention 信号排序重要性。(4) **DivPrune** (diversity-based) —— 最大化所选 vision token 集合的 intra-set diversity，仅使用 vision 信息。(5) **VisionZip fine-tuned** —— 在 token selection 之上增加训练过程。

  Baseline（以 LLaVA-1.5-7B + VisionZip 为例）全栈执行例子：
  - **算法层**：CLIP-ViT-L-336px 编码 336×336 图像 → 576 个 vision token（经 MLP projection 对齐 LLM 空间）→ VisionZip: 计算 [CLS] token 与 vision token 的预训练 attention 分数 → top-k ranking 选 token → 拼接 text tokens 送入 Vicuna-7B LLM decoder → 自注意力推理。缺陷：(1) 仅用 vision 侧 [CLS] attention 做 ranking，无法感知文本查询语义（如 "Describe the image" vs "What color is the car?" 得到相同 vision token selection）；(2) 忽略 vision token 之间的覆盖关系，可能选出高度相似的冗余 token。
  - **系统框架层**：Lmms-eval 评估框架。无额外 Serving 框架修改。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：标准 PyTorch 自注意力算子，无自定义 kernel。
  - **硬件架构层**：NVIDIA H100 / A6000 GPU。LLaVA-1.5-7B 576 tokens 原始推理。

  Baseline 的核心缺陷：
  1. **单模态信息不足**：Vision-only 方法（FastV, VisionZip, DivPrune）无法利用文本查询语义区分重要 token（同一图像不同问题应有不同选择）。Language-only 方法（SparseVLM）仅使用文本引导，忽略图像整体信息。多模态任务天然需要两种模态协同选择 token。
  2. **缺乏通用选择准则**：各方法使用各异的准则（attention ranking、diversity、CLS importance），缺乏统一的数学框架。DivPrune 的 diversity 最大化与 MMTok 的 coverage 最大化是互补视角，但 DivPrune 仅考虑 vision intra-set diversity。
  3. **极端压缩时性能崩溃**：以 VisionZip 为例，在 LLaVA-1.5-7B 上从 576 → 2 tokens，高 IC 任务平均性能仅保留 43%（vs MMTok 62.1%）；LLaVA-NeXT-7B 上 20→10 tokens 时 VisionZip 仅为 38.5%。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  MMTok 通过多模态覆盖最大化（Multimodal Coverage Maximization）系统性解决上述缺陷：

  (1) **多模态覆盖准则** → 解决单模态信息不足。同时优化两个覆盖子问题：Text-Vision Coverage（选出的 vision token 覆盖文本查询的全部语义信息）+ Vision-Vision Coverage（选出的 vision token 覆盖全体 vision token 的信息）。通过 softmax 温度校准（τ_t=0.02, τ_v=0.2）消除跨模态相似度量纲差异，通过 α=0.5 加权融合。这使 token 选择同时感知 "哪些图像区域与问题相关"（T-V）和 "哪些图像区域最能代表全图"（V-V），两个模态互补。

  (2) **子模函数优化框架** → 提供通用准则与理论保证。将覆盖函数 f(S; M) = (1/m) Σᵢ max_{j∈S} M_{i,j} 证明为子模函数，贪心算法获得 (1-1/e) ≈ 63% 近似最优解的理论保证（Nemhauser et al., 1978）。两个覆盖函数的和仍为子模函数（Corollary 1），因此多模态融合不破坏理论性质。复杂度 O(kn)，实际开销极低（2880 tokens 选 160 仅 6.4ms, 13.9 GFLOPs on A6000）。

  (3) **Training-free + 极端压缩鲁棒性** → 解决压缩崩溃。无需任何微调，在 LLaVA-1.5-7B 上仅 4 tokens 仍保留 71.4% 原始性能（高 IC 任务），VisionZip 仅 43.8%。在 POPE 上 2 tokens 即保留约 80% 性能。引入 Image Contribution (IC) 指标指导更精确的评估：IC = (Perf_All - Perf_0)/Perf_0，仅在高 IC 任务上评估 token selection 质量，避免低 IC 任务（如 ScienceQA IC=0.09, MMMU IC=0.09）误导评估。

  对比 baseline 的全栈执行例子（MMTok, LLaVA-1.5-7B, POPE benchmark, k=64 tokens）：

  - **算法层**：CLIP-ViT-L-336px 编码 336×336 图像 → 576 个 vision token V'（投影前）+ V（投影后 MLP aligned）→ 文本 "Is there a cat in the image?" tokenize → text hidden states T → 计算 M^{tv} = T·Vᵀ (cosine similarity, after projection) → 计算 M^{vv} = V'·V'ᵀ (cosine similarity, before projection) → Softmax 校准 (τ_t=0.02, τ_v=0.2) → 合并覆盖目标 f = f(S; M^{tv'}) + 0.5·f(S; M^{vv'}) → 贪心选 k=64 tokens (每次迭代选使 f 增量最大的 token) → 64 个 vision token 与 text tokens 拼接送入 Vicuna-7B LLM → 自注意力 (O((64+m)²) vs baseline O((576+m)²)) → 输出答案。POPE F1: 85.77 vs baseline 85.90 (99.9% 保留率)，推理 token 减少 88.9%。
  - **系统框架层**：Lmms-eval 评估框架。无额外 Serving 框架修改。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：相似度矩阵构建使用 PyTorch 内置库（矩阵乘法 matmul），贪心选择仅涉及 max 操作。无自定义 kernel。
  - **硬件架构层**：NVIDIA H100/A6000 GPU。在 H100 上 LLaVA-NeXT-13B token selection 开销 < 7ms，总推理时间从 1705s 降至 913s（POPE task），1.87× speedup；GPU 利用率从 86.7% 降至 58.0%；运行内存减少 60.7%（从 4.59GB 增量降至 1.78GB 增量）。
