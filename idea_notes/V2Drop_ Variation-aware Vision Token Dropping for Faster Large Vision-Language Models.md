## V2Drop: Variation-aware Vision Token Dropping for Faster Large Vision-Language Models

- baseline方法是什么？
  Baseline 是**基于 attention weights 的 inner-LLM token 压缩方法**（以 FastV、SparseVLM、PDrop 为代表）：
  这些方法在 LLM 内部利用 attention weights（如 cross-modal attention 或 self-attention scores）来量化每个 visual token 的重要性，然后剪枝低重要性 token。
  
  **全栈执行例子（以 FastV + LLaVA-1.5-7B 为例）：**
  - **算法pipeline**：图像经过 ViT 编码为 576 个 visual embeddings → Projector（2-layer MLP）映射为 vision tokens → 进入 LLM decoder 第 2 层后，计算 visual token 到 instruction token 的 cross-attention 分数作为重要性 → 一次性丢弃低分 token（one-time dropping at layer 2）→ 剩余 token 继续正常前向传播。
  - **系统框架**：作为 LLaVA 的 plug-and-play 模块运行，不修改模型架构。论文未明确说明 Serving 调度层面的集成。
  - **编译框架**：论文未明确说明。
  - **kernel调度**：FastV 需要显式计算 attention weights（通过 `.attn_weights` 获取），这与 FlashAttention 不兼容。FlashAttention 将 attention 计算融合为单一 kernel，不暴露中间 attention weights，导致 FastV 必须 fallback 到标准 attention 实现。GPU 峰值显存超过未压缩模型（增加 3.7%）。
  - **硬件架构**：论文未明确说明。运行在 NVIDIA A100 GPU 上。

  FastV 的核心缺陷：
  (i) **信息无关的位置偏见（Positional Bias）**：attention 机制天然偏向序列末尾位置的 token（无论内容），赋值高重要性，导致保留不相关 token 同时丢弃语义重要的早期 token，加剧多模态幻觉。
  (ii) **与高效算子不兼容**：依赖显式 attention score 计算，与 FlashAttention 等高效 attention 算子不兼容，导致显存和计算开销增加。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  V2Drop 提出**从 token 变异性（token variation）视角**进行 token 压缩，通过测量 visual token 在相邻 LLM 层之间的表示变化（L2 距离）来直接评估 token 的重要性，而非依赖外部 attention 信号。
  
  核心洞察：参与 LLM 推理的高重要性 token 会在跨层传播时产生显著的表示变化；而"惰性 token"（lazy tokens）保持相对静态，对最终预测贡献有限，可安全丢弃。
  
  **全栈执行例子（V2Drop + LLaVA-1.5-7B）：**
  - **算法pipeline**：图像经过 ViT 编码为 576 个 visual embeddings → Projector 映射为 vision tokens → 进入 LLM decoder → 在 layer 3：计算每个 vision token 从 layer 2 到 layer 3 的 L2 variation（||f_i^(l) - f_i^(l-1)||_2）→ 按 variation 降序排序，保留 top-50%（约 288 token）→ 在 layer 17：再次计算 variation，保留 top-30%→ 在 layer 22：最终保留目标数量（如 192 token）→ 剩余 token 继续前向传播完成生成。
  - **系统框架**：论文未明确说明。作为 LLaVA/Qwen2-VL/LLaVA-OV 的 plug-and-play 模块运行，不修改模型架构。
  - **编译框架**：论文未明确说明。
  - **kernel调度**：V2Drop 仅需计算 token 之间的 L2 距离（简单的张量减法+norm），无需 access attention weights，天然兼容 FlashAttention。实验证明 GPU 峰值显存与 random dropping 几乎相同（15046MB vs 15045MB），无额外显存开销。
  - **硬件架构**：论文未明确说明。运行在 NVIDIA A100 GPU 上。

  理论支撑（Theorem 1）：在 mild smoothness 假设下，$\|\Delta f_j\| \approx \|J_j\|_{\text{op}} \cdot \|\Delta x_j^{(t)}\|$，即 token 的输出影响与其跨层变化量成正比，variation 是 token importance 的计算高效代理。
  
  对比 baseline 的解决效果：
  | 缺陷 | Baseline 表现 | V2Drop 解决方式 |
  |------|-------------|---------------|
  | 位置偏见 | 注意力赋分偏向末尾 token | Variation 信号与位置无关，仅反映 token 语义重要性 |
  | FlashAttention 不兼容 | 需 fallback 到标准 attention，显存增 3.7%+ | 无需 attention weights，显存节省 3.3% |
  | 视频长序列 | SparseVLM 显存增 54.8% | 显存节省 7.8%，吞吐 1.38x |
  | 一次性剪枝 | 一次性丢弃损失大 | 渐进式剪枝，保留 94.0%（图像）/98.6%（视频）原始性能 |
