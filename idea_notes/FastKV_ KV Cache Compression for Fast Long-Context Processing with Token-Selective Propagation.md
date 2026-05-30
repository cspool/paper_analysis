## FastKV: KV Cache Compression for Fast Long-Context Processing with Token-Selective Propagation

- baseline方法是什么？
  Baseline 分为两类：(1) **仅解码加速方法**（StreamingLLM、H2O、SnapKV）：预填充阶段处理完整上下文构建完整 KV cache，解码阶段根据注意力重要性 score 裁剪 KV cache。它们不减少预填充计算量，在长上下文（128K）下预填充时延占总时延主导部分，且 H2O 因需导出完整 attention map 而无法使用 FlashAttention-2、在 8K+ 上下文即 OOM。(2) **预填充感知加速方法**（GemFilter、PyramidInfer）：在预填充阶段减少处理的 token 数量。GemFilter 在单个 filter layer 处从完整上下文中选择关键 token，然后仅用这些 token 重新预填充所有层——导致早期层（注意力分布各异）被迫使用同一 token 子集，信息丢失严重；PyramidInfer 从第一层即开始按 cosine schedule 逐步减少 token，在上下文稳定性建立之前就过早丢弃 token。这两类 baseline 的核心缺陷是**预填充计算量减少与 KV cache 预算刚性耦合**——要减小 KV cache 必须更激进地减少预填充计算，导致准确率大幅下降。

  **Baseline 全栈执行例子（以 GemFilter 为例，LLaMA-3.1-8B，128K 输入）：**
  - **算法层**：Embedding → Layer 0-13 full-context prefill（计算 128K×128K attention map）→ Layer 13 收集所有 head 的 attention scores → 跨 head 平均计算 saliency → TopK 选择 20% token（25600 tokens）→ 丢弃其余 102400 tokens 的信息 → Layer 0 到 Layer 31 仅用 25600 tokens 重新 prefill → 输出 logits。被丢弃 token 的语义信息永远不会被任何层处理。
  - **系统框架层**：HuggingFace Transformers + FlashAttention-2 kernel。GemFilter 需执行两轮 prefill（首次 128K + 二次 25.6K），相当于 1.6 轮完整前向传播。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：FlashAttention-2 on A100 SXM。首次 prefill 计算完整 128K attention；二次 prefill 仅计算 25.6K。FlashAttention-2 的 tiled 计算和 SRAM 优化对两次 prefill 均适用。
  - **硬件架构层**：单张 NVIDIA A100 SXM GPU（80GB）。KV cache 在 GQA 下每层每 token 约占用 2×8 KV heads × 128 head_dim × 2 bytes = 4KB/token/layer（FP16），128K×32 层 ≈ 16.4GB。压缩至 20% 后约 3.3GB。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **FastKV**，核心设计：(1) **Token-Selective Propagation (TSP)**：在模型中部的 TSP 层（LLaMA-3.1-8B 的 layer 15），基于 window tokens 的注意力权重计算每 token 的 saliency score，仅向后续层传播 top-R_TSP 个 token 的 hidden states——但早期层（0 到 TSP）保持完整上下文计算，确保每个早期 layer 可自由关注其偏好的 token 子集。(2) **TSP rate 与 KV retention rate 完全解耦**：TSP rate 控制预填充计算量（等于 1 - Σ_{l>TSP}(1 - R_TSP)，约 60%），KV retention rate 独立控制解码时每层保留的 KV cache 比例（10% 或 20%），二者互不约束——可在保护准确率的同时激进压缩 KV cache。

  **如何解决 Baseline 缺陷：**
  - **vs GemFilter**：GemFilter 的 filter layer 决定所有层使用同一 token 子集 → FastKV 的前 TSP 层保留完整上下文，每层可独立关注不同 token；GemFilter 中丢弃 token 的信息完全丢失 → FastKV 中被 TSP 丢弃的 token 已在早期层的注意力计算中将其语义融合到传播 token 中（Figure 7 可视化）。GemFilter 预填充=KV retention → FastKV 完全解耦两个比率。
  - **vs PyramidInfer**：PyramidInfer 从 layer 0 即开始减少 token → FastKV 仅在 layer 15（上下文稳定后）开始 TSP。PyramidInfer 不压缩 KV cache（KV retention = prefill compute rate）→ FastKV 独立设为 10%，大幅减小解码时延。
  - **vs SnapKV/StreamingLLM/H2O**：这些方法预填充完整上下文后压缩 KV → FastKV 在预填充阶段即减少后续层的计算量，同时独立压缩 KV cache，同时加速预填充和解码。

  **FastKV 全栈执行例子（LLaMA-3.1-8B，128K 输入，R_TSP=0.2，R_KV=0.1）：**
  - **算法层**：Embedding → Layer 0-15 全量 prefill（128K 上下文，构建 K_X, V_X）→ 每层完成 prefill 后立即执行 KV_Compress：对每 KV group 计算 group-wise saliency（head-wise attention scores 在 group 内平均），保留 top-10% 关键 token 的 KV entries → Layer 15 同时执行 TSP：基于最后 N_obs=8 个 window tokens 的注意力权重，MaxPooling(kernel=7)+跨 head 平均计算 saliency score → 选择 top-20% token 的 hidden states + 所有 window token → 仅传播这 25608 个 hidden states 到 layer 16 → Layer 16-31 在 25608 个 token 上计算注意力并各自压缩 KV cache → LMHead 输出 logits。总 prefill FLOPs ≈ 完整上下文 × (15/32 + 17/32×0.2) ≈ 60%。
  - **系统框架层**：HuggingFace Transformers self-attention 层被修改：在 TSP 层增加 HiddenCompress 步骤；每个 decoder layer 增加 KV_Compress 步骤。与 FlashAttention-2 完全兼容（不使用需要导出完整 attention map 的操作）。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：FlashAttention-2 on A100 SXM。TSP 层的 saliency scoring 仅基于 8 个 window token query 的行进行 MaxPooling + averaging，不加载完整 attention map，额外开销仅 0.15s（128K 下占总 prefill 0.88%）。KV compression 后每层仅需存储和访问 10% 的 KV entries，解码时 attention 的 memory-bound 瓶颈显著缓解。
  - **硬件架构层**：单张 NVIDIA A100 SXM（80GB）。128K 输入下：prefill 阶段，layer 0-15 计算完整 128K attention → 每层 KV cache 压缩至 12800 tokens → 总 KV cache ≈ 128K×15×0.1 + 128K×0.2×17×0.1 ≈ 2.4GB（vs 完整 16.4GB）。128K 上下文生成 256 token 时：端到端时延从 18.81s 降至 6.63s（约 2.84× 加速）。

  **FastKV 通过观察 "早期层注意力不稳定、后期层注意力收敛" 的层依赖上下文动态特性，将 TSP 和 KV compression 置于正确的时机——早期保留完整上下文满足每层异构注意力需求，后期识别并传播稳定关键 token 获得预填充加速——同时将两个压缩比例解耦，实现了第一个同时加速预填充和解码且保持高准确率的 KV cache 压缩框架。**
