## RETAKE: Reducing Temporal and Knowledge Redundancy for Long Video Understanding

- baseline方法是什么？
  Baseline 是标准的 concatenation-based VideoLLM（QWen2VL-7B 和 LLaVA-Video-7B），其推理全栈执行过程如下：
  - **算法 pipeline**：视觉编码器（VFM）对均匀采样（2FPS）的所有帧提取特征（每帧数百个 visual tokens）→ 连接器将 visual tokens 与 prompt tokens 拼接 → LLM 对所有 token 做 full self-attention prefilling → 自回归解码生成回答。长视频下 visual token 总数急剧膨胀（>256 帧即可能超过 A100 显存限制）。
  - **系统框架**：所有 visual tokens + text tokens 一次性送入 LLM，KV cache 保存全部 token 的 key/value 状态。KV cache 显存占用量 ∝ context length，限制了可处理的帧数上限约 256-300 帧。
  - **kernel 调度**：标准 PyTorch/HuggingFace Transformers 的 FlashAttention kernel，无特殊 token 调度优化。
  - **硬件架构/芯片设计**：论文未明确说明。

  Baseline 的核心缺陷：(1) 时间冗余——相邻帧之间高度相似，大量 visual tokens 承载冗余信息；(2) 知识冗余——LLM 的注意力机制天然包含 token 级冗余信息，低注意力 token 可被丢弃而不显著影响性能，但 baseline 未利用这一特性；(3) 现有 token compression 方法（FastV, SparseVLM 等）仅基于 prompt tokens 压缩 visual tokens，忽略了 visual token 之间的冗余。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  RETAKE 通过 DPSelect（减少时间冗余）+ PivotKV（减少知识冗余）联合压缩，全栈执行过程如下：
  - **算法 pipeline**：
    (1) DPSelect：在视觉编码后、送入 LLM 前，计算相邻帧的 token-averaged cosine distance，用 max pooling 识别距离峰值帧作为 pivot frames，再按 top-k 补充关键帧。这与人类通过峰值刺激感知运动的机制一致，有效过滤静态冗余帧。
    (2) Chunked Prefilling：将压缩后的视频序列划分为固定大小 chunks，逐 chunk 送入 LLM prefilling，数学上等价于一次性 prefilling。
    (3) PivotKV：每个 chunk prefilling 后，计算当前 chunk 内的 self-attention 权重，求和得到 token 重要性分数。Pivot frames 的 token 被强制保留（分数 +∞），非 pivot 帧中低注意力 token 被剪枝。Pivot frames 保证关键低层细节不丢失，LLM 注意力机制隐含地利用高层多模态知识识别 token 冗余。
  - **系统框架**：RETAKE 以即插即用方式作用于现有 VideoLLM，无需任何训练。通过同时减少视觉编码后的帧数和 LLM 中的 KV cache token 数，在固定显存预算下可处理 8× 更多帧（256→2048），context length 控制在 16K-32K。
  - **kernel 调度**：效率优化使用额外 CUDA stream 实现 PivotKV 压缩（第 l 层）与 prefilling（第 l+1 层）的 overlap，将 TTFT 开销从 +28%/62% 降至 +8%/11%。
  - **硬件架构/芯片设计**：论文未明确说明。

  对比 baseline 的关键改进：
  | 维度 | Baseline | RETAKE |
  |------|----------|--------|
  | 帧选择 | 2FPS 均匀采样 | DPSelect 峰值感知关键帧选择 |
  | Visual tokens | 全部保留 | α_dp 比例保留关键帧 |
  | KV cache | 全部保留 | PivotKV 按注意力分数剪枝非 pivot token |
  | 最大帧数 | ~256 (A100 OOM) | 2048 (8×) |
  | 额外计算开销 | 0 | +8-11% TTFT (优化后) |
  | 解码延迟 | 基准 | -20% TPOT（因 KV cache 更短） |
  | 训练需求 | 需要视频-文本对训练 | 完全 training-free |

- baseline方法是什么？
  Baseline 方法是使用固定压缩比或启发式压缩分配的 visual token 压缩方法，主要包括：(1) FastV / FitPrune —— 使用累积 attention scores 作为 eviction 标准，固定压缩比（不区分时间维度和层维度）；(2) PyramidDrop —— 在层维度使用单调递增的压缩比（深层压缩更多），忽略 Section 3 中观察到的层注意力非单调波动（局部最小值在 layers 2,14,21，最大值在 layers 7,18）；(3) VL-Cache —— 使用启发式动态分配但缺乏理论保证；(4) 基础 MLLM（如 Qwen2-VL-7B, LLaVA-Video-7B）—— 不做 visual token 压缩，固定采样 128-256 帧，帧数受 GPU 内存限制无法增加。

  Baseline（QWen2-VL-7B，无压缩，128 frames）全栈执行例子：
  - 算法层：用户上传一段长视频 + 文本问题 "What happens at 5:30?" → 以 2 fps 采样 128 帧 → Vision Encoder（ViT）逐帧编码 → Projector 映射为 N×128 个 visual tokens → 与 S 个 text tokens 拼接为 (128N + S) 长度序列 → LLM 28 层 decoder autoregressive prefill → KV cache 存储全部 (128N + S) 个 token 的 K/V，内存 O(2 × 28 × (128N+S) × d) → Decoding 生成答案
  - 系统框架层：基于 HuggingFace Transformers 推理，论文未明确说明 Serving 框架
  - 编译框架层：论文未明确说明
  - kernel调度层：使用标准 FlashAttention 进行 attention 计算
  - 硬件架构层：论文未明确说明

  Baseline 的缺陷：
  1. **固定帧数采样限制视频感知时长**：GPU 内存在长视频推理中主要被 KV cache 占据（Hooper et al., 2024），固定 128 帧仅能覆盖约 64 秒视频（2 fps），对于 LVBench 平均 4101 秒的视频丢失大量信息。
  2. **时间维度冗余分布不均但被统一压缩**：Section 3 分析显示，重击者（heavy-hitter）比例在同一视频的不同时间片段上差异达 3 倍——静态场景高度冗余而动态场景信息密集，但 baseline 方法对所有时间段使用相同压缩比。
  3. **层维度冗余分布非单调但被启发式或单调处理**：PyramidDrop 假设深层应压缩更多（单调），但实际重击者比例沿层呈现非单调波动（局部最小值 layers 2/14/21，最大值 layers 7/18），单调分配导致在注意力低谷层保留过多 token、在注意力高峰层过度压缩。
  4. **缺乏理论保证**：现有方法（VL-Cache 等）的启发式压缩比分配缺乏理论上的最优性保证。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法：AdaRETAKE 通过两个 training-free 的自适应分配模块解决上述问题：
  (1) **Temporal-adaptive Allocation**：将视频分为 10 秒 chunk，计算每个 chunk 内相邻帧的余弦相似度作为帧间距离；距离越大的 chunk 信息变化越剧烈（动画/场景切换），分配更高的保留比例；距离越小的 chunk 静态冗余越严重，分配更低的保留比例。压缩比全局满足 Σα_i = (C_max - S)/(TN)。
  (2) **Layer-adaptive Allocation**：在每个 chunk 的 prefill 过程中，计算每一层 video-token-to-prompt 的累积 attention score；在全局 Top-K（K=α_i × τN × L）阈值下，统计每层显著性 token 数量 s_i^(l)，按比例分配各层压缩比 α_i^(l) = w_i^(l) × α_i。引入最小权重 ε=0.01 保证数值稳定（防止某层完全不保留 token）。
  (3) **理论保证**：证明压缩损失的 L1 上界 ε^L = 2C^(L) - 2C^(L) ∏ Σ I_i^(l) A_i^(l)，且基于 submodular 优化理论证明选择全局 Top-K attention score 的 token 能实现 (1-1/e) 近似最优。

  对比 baseline 的全栈执行例子（AdaRETAKE + QWen2-VL-7B，2048 frames, C_max=16K）：
  - 算法层：用户上传同一段长视频 + 问题 "What happens at 5:30?" → 以 2 fps 采样 2048 帧（覆盖 1024 秒 / ~17 分钟） → Vision Encoder (ViT) 逐帧编码 → Projector 映射为 2048N 个 visual tokens → 视频按 τ=10s 等分 chunk M = [M_1, ..., M_k] → Temporal-adaptive: 计算各 chunk 内相邻帧余弦距离 d_bar_i，按比例分配压缩比 [α_1, ..., α_k] → 逐 chunk prefill: Layer-adaptive 计算每层 video-prompt attention a_i^(l)，全局 Top-K 确定每层显著性 → 每层压缩比 α_i^(l) 分配 → Token Compression: 每层保留 Top-K 个 visual token KV cache → 所有 chunk 处理完后，KV cache 总大小为 C_max=16K → Decoding 在 16K 长度序列上 autoregressive 生成 → 虽然从 2048→2048N 压缩到 16K token，但因保留了更多时间跨度的关键帧，对长时间事件理解更好（LVBench +5.9% @7B, +6.0% @72B）
  - 系统框架层：基于 HuggingFace Transformers，chunked prefill 等价于标准 prefill（Zeng et al., 2024b），论文未说明 Serving 框架
  - 编译框架层：论文未明确说明
  - kernel调度层：标准 FlashAttention，压缩操作基于累积 attention scores 的 ArgTopK + KV cache 索引选择，无额外 kernel 开销
  - 硬件架构层：论文未明确说明

  解决对应关系：
  - Baseline 缺陷 1（固定帧数限制）→ 通过 token 压缩在固定 GPU 内存预算内支持最多 2048 帧（vs baseline 128-256 帧），从 256→2048 帧扩展；LVBench（平均视频最长）提升最显著（5.9%-6.0%），因压缩让更多时间跨度的帧信息被保留。
  - Baseline 缺陷 2（时间维度冗余不均）→ Temporal-adaptive Allocation 基于帧间余弦距离动态分配压缩比：高冗余 chunk（静态场景，d_bar 小）压缩更多，低冗余 chunk（动态场景，d_bar 大）保留更多，Section 5.3 消融显示 +1.0% avg 提升。
  - Baseline 缺陷 3（层维度非单调冗余）→ Layer-adaptive Allocation 基于 video-prompt attention 分数按层自适应分配：在 attention 高峰层（如 layer 7,18）保留更多 token，在低谷层（如 layer 2,14,21）压缩更多；消融显示 +0.8% avg 提升；对比 PyramidDrop（单调分配）的 +2.1-3.2% 优势（Table 2）。
  - Baseline 缺陷 4（无理论保证）→ 提供完整的 L1 压缩损失上界分析和 submodular greedy (1-1/e) 近似最优证明（Theorem 4.1, Appendix A）。
