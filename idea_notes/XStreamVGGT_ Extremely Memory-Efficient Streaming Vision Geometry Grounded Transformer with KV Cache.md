## XStreamVGGT: Extremely Memory-Efficient Streaming Vision Geometry Grounded Transformer with KV Cache Compression

- baseline方法是什么？
  Baseline 是 **StreamVGGT**，一个 streaming 4D visual geometry transformer，将 VGGT 的全局 Alternative-Attention 替换为 frame-wise causal attention，实现在线流式 3D 重建。其核心依赖 KV cache 机制：每帧的 Key 和 Value tensors 被缓存并在后续帧的 temporal attention 中重用（与 LLM 的 autoregressive 解码类似）。KV cache 大小随输入帧数线性增长，最终导致无界内存增长。

  全栈执行例子（以 StreamVGGT + 单 A100 为例）：
  - **模型推理算法层**：输入 RGB 帧 I_t → Patch Embedding 编码为 F_t ∈ R^{N×C} → 拼接 camera token g_t、register tokens r_t → L 层 Alternating-Attention（每层先 intra-frame spatial self-attention，再 temporal causal attention 拼接历史 K_{1:t-1}, V_{1:t-1} 和新 K_t, V_t）→ 任务头输出相机参数和点云/深度。K, V 全量保留，cache 无界增长。
  - **系统框架层**：基于 PyTorch + FlashAttention-2 实现，无额外的 serving 框架适配。帧序列逐个处理，KV cache 存储在 GPU 显存中。
  - **编译框架层**：论文未明确说明。
  - **kernel 调度层**：FlashAttention-2 kernel 执行 attention，无自定义 kernel。
  - **硬件架构层**：单张 NVIDIA A100 GPU (80GB)。随帧数增加，StreamVGGT 在 ~200 帧后 FPS 显著下降，约 300 帧时触发 OOM。

  Baseline 的核心缺陷：
  (1) **KV cache 无界增长**：时序 causal attention 中每帧产生的 K, V tensors 全部追加到 cache 中，cache 大小 = O(T × (1+R+N) × L × C)，其中 T 为帧数。视觉 token 数量远大于文本 token（每帧含 N 个 patch tokens），导致 KV cache 膨胀速度远超 LLM 场景。
  (2) **视觉 token 高度冗余**：视频帧之间存在大量 intra-frame 空间相关性和 inter-frame 时序一致性冗余，大量 patch tokens 对应场景中变化极小或不变化的区域，但全量保留在 cache 中浪费内存和计算。attention heatmap 显示只有少量 Query-relevant 区域获得显著注意力权重。
  (3) **无法扩展到长序列应用**：内存消耗和推理延迟随帧数线性增长，对机器人、自动驾驶等长时间运行场景形成关键瓶颈。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **XStreamVGGT** 通过无缝集成 KV cache pruning 和 dimension-adaptive KV quantization，将 KV cache 从"无界增长"转换为"有界内存"的流式推理。

  全栈执行例子（对比 baseline）：

  1. **算法 pipeline 层**（核心创新）：
     缺 → XStreamVGGT 在 StreamVGGT 每层 temporal attention 后插入两步压缩：

     **(a) Query-guided KV Pruning**：
     - 池化当前帧 Query（分组平均 + 跨 head 平均）得到紧凑表示 Q̄_t，与中间帧 Key 的跨 head 平均 K̄_prunable 计算内积作为 token 重要性分数 S = mean(Q̄_t @ K̄_prunable^T, dim=query)。
     - Top-k 选择保留高重要性 token，始终保留第一帧（几何参考）和当前帧（最新视觉证据）。
     - 当 cache 达到预算 L_max=2K 后，cache 大小不再增长，时间复杂度从 O(T) 降至 O(L_max)。
     - Group pooling 设计（group size g=16）使重要性识别与 FlashAttention 完全兼容，无需读取中间 attention scores。
     **直接解决缺陷1（无界增长）**：cache 有界化；**解决缺陷2（视觉冗余）**：利用 query-key 语义匹配识别并保留信息量最大的 token。

     **(b) Dimension-Adaptive KV Quantization**：
     - 发现 StreamVGGT 中 Key tensors 存在显著的 channel-wise outliers，而 Value tensors 分布更均匀。
     - 对 Keys 使用 per-channel 量化（每个 channel 独立 scale，避免 outlier channel 主导量化步长），对 Values 使用 per-token 量化。
     - KIVI INT4 量化：4-bit 存储，attention 计算时 dequantize 回 FP16/FP32。
     - 量化仅应用于最终 pruned cache，不影响剪枝决策本身的精度。
     **进一步压缩存储**：在 pruning 基础上再减少 ~4× 内存（INT4 vs FP16），总计 4.42× 内存减少。

     效果：3D 重建 NC 指标仅下降 1.4%（NRGBD），深度估计 Abs Rel 几乎无退化（Sintel 0.254 vs 0.254, KITTI 0.072 vs 0.072），相机姿态 ATE 仅增加 0.006；5.48× FPS 加速。

  2. **系统框架层**：
     Baseline 的 KV cache 在 GPU 显存中全量保留，随帧数增长导致 OOM。XStreamVGGT 的 pruning + quantization 均在 PyTorch 层实现，作为 StreamVGGT 的 plug-and-play 替换。剪枝的 Q̄K̄^T 计算使用标准 PyTorch 操作，量化使用自定义 scale/zero-point 计算 + clamp/round 操作，无额外框架依赖。代码开源于 https://github.com/ywh187/XStreamVGGT/。

  3. **编译框架层**：论文未明确说明。

  4. **kernel 调度层**：
     Baseline 使用 FlashAttention-2 处理 temporal attention。XStreamVGGT 的剪枝方案通过 group pooling 而非直接读取 attention scores 来识别 token 重要性，保持与 FlashAttention-2 的完全兼容（FlashAttention 不输出中间 attention scores）。剪枝后 attention 的 K/V 长度固定为 L_max，减少了 attention kernel 的计算量。量化/反量化在 attention kernel 外部完成。
     论文未实现自定义 kernel。

  5. **硬件架构层**：
     Baseline 和 XStreamVGGT 均运行在单张 NVIDIA A100 GPU (80GB)。XStreamVGGT 通过减少 KV cache 占用释放了 GPU 显存，使原本在 ~300 帧 OOM 的 StreamVGGT 可扩展到 1000+ 帧。无硬件定制。
