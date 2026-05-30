## VisionSelector: End-to-End Learnable Visual Token Compression for Efficient Multimodal LLMs

- baseline方法是什么？
  Baseline 是 **训练无关（training-free）的启发式 token 压缩方法**，包括三类：attention-based（FastV, PruMerge+, VisionZip）、similarity-based（DART, DivPrune）、和 transformation-based（TokenPacker 等）。这些方法在 MLLM 推理时直接丢弃或合并视觉 token，但不进行任何训练，压缩策略依赖预训练模型内部的特征分布。

  全栈执行例子（以 Qwen2.5-VL-7B + FastV 为例）：
  - 模型推理算法层：给定高分辨率输入图像 → ViT 编码为 patch tokens → PatchMerger 初步压缩 → MLP Projector 映射为 V ∈ R^{N×D} → 前向第 2 层 self-attention → 使用 text→vision 的 attention scores 作为剪枝准则，删除低分 token → 保留的 token 与 text token 拼接，继续 LLM 各层前向 → 生成回答。FastV 根据 attention score ranking 丢弃 token，PruMerge+ 用 attention 稀疏 + KNN 聚类，VisionZip 从末层 attention map 选 dominant tokens + 语义相似度合并，DART 通过余弦相似度去重，DivPrune 用 Max-Min Diversity 选子集。
  - 系统框架层：LMMs-Eval 框架（HuggingFace Transformers + FlashAttention-2），在 token projection 后插入剪枝操作，无额外 serving 框架修改。训练时冻结全部参数。
  - 编译框架层：论文未明确说明。
  - kernel 调度层：FlashAttention-2 kernel 执行 LLM 的 self-attention，无自定义 kernel。
  - 硬件架构层：NVIDIA A800 GPU (80GB)，无硬件定制。

  Baseline 的核心缺陷：
  (1) **启发式偏差（Attention Sink / Dispersion）**：attention-based 方法继承预训练 attention map 中的 attention sink 偏差（前几个 token 获得不成比例的高 attention mass），在极端压缩率下（10%）会保留位置靠前但语义无关的 token，性能急剧下降（VisionZip 从 20%→10% 下降 ~14 个百分点）。
  (2) **细粒度信息丢失**：similarity-based 方法（DART 去重、DivPrune 多样性保留）在压缩过程中丢弃精细的语义细节 token，在 OCR 和文档密集视觉任务中性能显著下降。
  (3) **缺乏灵活性**：固定压缩策略无法根据下游任务动态调整，压缩率固定，需要针对不同压缩率分别调参。
  (4) **跨模型泛化差**：各方法的性能高度依赖底层 MLLM 的内部特征分布（如 VisionZip/PruMerge+ 在 Qwen2.5-VL 的 PatchMerger 位置引起 OOM，FastV 在 LLaVA-OV-1.5 上性能明显低于在 Qwen2.5-VL 上）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **VisionSelector** 将 token 压缩从 "训练无关启发式" 范式转变为 "端到端可学习决策" 范式。训练时下游任务损失直接驱动 LIS 学习 token 重要性，推理时用标准 Top-K 高效硬选择。

  全栈执行例子（对比 baseline）：

  1. **算法 pipeline 层**（核心创新）：
     缺 → VisionSelector 在 encoder→LLM 间插入 LIS + DTS + CAS。LIS 通过 QK^T 全局交互（而非依赖预训练 attention map）计算每个 token 的重要性得分，DTS 通过 sigmoid 连续松弛 + 隐函数微分实现梯度透传，CAS 通过逐步增大 λ_t 消除训练/推理 gap。**直接解决缺陷1（无 attention sink 偏差）和缺陷2（保留细粒度关键信息）**，因为重要性评分由下游任务端到端学习。在 10% 保留率下比 VisionZip 高 12.14 个百分点，30% 保留率下 MME 达到 100.07%（超过 100% 的 baseline，实现增益性压缩）。**解决缺陷3（灵活性）**：训练时固定 20% 压缩率，推理时泛化到任意预算；**解决缺陷4（跨模型泛化）**：因其学习范式独立于模型内在特征分布，在 Qwen2.5-VL-7B/3B/LLaVA-OV-1.5-8B 三个架构上均显著优于 baseline。

  2. **系统框架层**：
     Baseline 插入层固定的 heuristic 操作在特定模型位置可能 OOM（VisionZip 和 PruMerge+ 在 Qwen2.5-VL 的 PatchMerger 输出处因 token 数过多引起）。VisionSelector 位于 modality interface 之后、LLM 之前，与 FlashAttention 完全兼容，参数独立于 backbone，可作为 plug-and-play 模块无缝集成。**解决缺陷4（跨模型兼容性）**。

  3. **编译框架层**：论文未明确说明。

  4. **kernel 调度层**：论文未明确说明。DTS 推理时仅需标准 Top-K（高效 GPU kernel），不引入额外 kernel 修改，与 FlashAttention-2 兼容。

  5. **硬件架构层**：论文未明确说明。

  量化对比：
  - 训练成本：仅 12.85M 参数 / 40 分钟（8×A800）/ 144K 样本
  - 20% 保留率：相对性能 94.83%（vs DivPrune 86.75%, VisionZip 86.43%, DART 78.16%）
  - 10% 保留率：相对性能 87.75%（vs DivPrune 75.61%, VisionZip 72.73%）
  - 视频效率（MVBench, 6828 tokens）：prefill 760.82ms（baseline 1413.34ms, 1.86× speedup），E2E 924.57ms（baseline 1605.31ms, 1.74× speedup），内存 17.57 GB（baseline 25.97 GB, 32.3% reduction）
