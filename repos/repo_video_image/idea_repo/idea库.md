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

## VideoNSA: Native Sparse Attention Scales Video Understanding

- baseline方法是什么？
  Baseline 是 **Qwen2.5-VL-7B + 标准 dense FlashAttention**（Dao, 2023），以及 **训练无关 token compression 方法**（FastV, VisionZip, VScan）和 **训练无关 sparse attention 方法**（MInference, FlexPrefill, XAttention, Tri-Shape）。

  全栈执行例子（以 Qwen2.5-VL-7B + FlashAttention 为例）：
  - 模型推理算法层：视频帧以 1 FPS（或更低）采样为少量帧 → ViT 编码每帧为 64-256 tokens → MLP Projector 映射 → 与 text tokens 拼接送入 Qwen2.5-7B decoder → 每个 self-attention 层对全部 vision tokens + text tokens 执行 dense causal attention（复杂度 O(L²)），每层 28 个 query heads × 4 个 KV heads 做 GQA → 逐层前向直至生成完整回答。Frame sampling 密度的提升直接改善准确率，但 token 数量随帧数线性增长，Attention 计算量平方增长，很快触及模型 128K context length 上限。
  - 系统框架层：LMMs-Eval / VLMEvalKit 框架标准推理 pipeline（HuggingFace Transformers + FlashAttention-2 kernel），无特殊 Serving 框架修改。推理时 KV-cache 随 sequence length 线性增长。
  - 编译框架层：论文未明确说明。使用 torch.compile 加速。
  - kernel 调度层：FlashAttention-2 kernel（Dao, 2023），在 H100 GPU 上执行，无自定义 sparse attention kernel。
  - 硬件架构层：NVIDIA H100 GPU，无硬件定制。

  Baseline 的核心缺陷：
  (1) **平方复杂度瓶颈**：dense attention 计算 O(L²)，长视频（如 10 小时 LongTimeScope）需要大量帧，token 数量远超模型 128K context 限制，实际只能大幅降采样（如 1 FPS），丢失关键过渡帧。
  (2) **Token compression 不可逆信息丢失**：FastV/VisionZip/VScan 等方法在 attention 前丢弃低分 token，复杂推理任务（如 Tomato 时序推理）中丢失的信息无法恢复，导致准确率显著低于 full-token 方法。
  (3) **训练无关 sparse attention 缺乏适应性**：MInference/XAttention 等方法的稀疏模式在推理时固定（如静态局部窗口/strided pattern），无法根据视频内容动态调整信息流，限制了跨任务泛化能力。
  (4) **Attention sink 问题**：dense attention 中 softmax 归一化导致大量 attention mass 分配给前几个 token（"attention sink"），这些 token 的 value norm 极小，对残差状态贡献微乎其微，浪费计算。且在更深层 sink 比例逐渐增加。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **VideoNSA** 引入三项核心创新：(1) Hybrid attention：vision tokens 用 NSA，text tokens 用 dense GQA；(2) NSA 三支路动态稀疏：Compression + Selection + Sliding Window + learnable gates；(3) End-to-end training：在视频数据上端到端训练稀疏注意力权重。

  对应解决 baseline 的四个缺陷：

  **(1) 近线性复杂度取代平方复杂度：**
  Baseline → dense attention 对所有 L 个 token 间的 L(L-1)/2 条边计算 attention。
  VideoNSA → NSA 将每条 query 可见的 KV 数压缩为固定 attention budget K_attn = b×s + w = 32×64 + 256 = 2304。在 128K context 下仅使用 3.6% 的 attention edges，复杂度近似 O(L · K_attn) = O(L)。这使得 VideoNSA 可在 128K tokens 下运行，超过 training 时的 36K tokens，在多个 benchmark 上性能持续提升。例如在 LongTimeScope（10 小时视频）上，VideoNSA 可在 single GPU 处理 10000+ frames。

  **(2) 保留所有 token，避免不可逆信息丢失：**
  Baseline token compression → 直接丢弃/合并 token，复杂推理任务中信息永久丢失。
  VideoNSA → 保留全部 vision token，但通过压缩支路将 block 内 tokens 聚合为粗粒度表示（减少冗余的同时保留帧级语义），通过选择支路保留最重要的 KV blocks，通过滑动窗口支路保证局部时间覆盖。三支路相互补充：压缩支路处理全局语义，选择支路聚焦关键区域，滑动窗口支路保证局部连续性。ablation 证实单支路/两两组合均显著劣于完整三支路（如仅 CMP 在 TimeScope 上仅 41.5 vs 完整 83.7）。

  **(3) Data-dependent 动态稀疏 vs 静态稀疏模式：**
  Baseline 训练无关 sparse attention → 固定的局部/跨步 pattern，无法适应任务差异。
  VideoNSA → 三支路由 learnable gate（2-layer MLP + sigmoid）动态加权，不同 layer 不同 head 学习到不同的支路偏好：压缩支路在各层保持较高平均权重（全局冗余压缩），选择和滑动窗口支路在浅中层活跃但在深层逐渐衰减（深层聚焦高层特征聚合）。Gate 的 task-dependent 分布使得模型可以针对不同任务自动调整信息流。Table 2 显示在三支路 + dynamic gating 下，LongVideoBench 从 48.x（单/两支路）跃升至 60.0。

  **(4) 动态稀疏抑制 attention sink：**
  Baseline dense attention → 深层 attention sink 逐渐增加，attention mass 集中在 value norm 极低的首 tokens。
  VideoNSA → 三个支路表现出截然不同的 sink 行为：选择支路几乎无 sink（top-k 过滤使 value norm 分布平滑），压缩支路产生最多 sink 但通过动态 gating 被压制，滑动窗口支路 sink 出现在局部邻域边界。整体 sink 比例仅 0.3%，且 sink 在时间轴上更平滑分布，不集中于序列起始位置。这些 sink 的 beneficical 特性——引导 attention 流向有意义的高层语义位置，而非浪费在首 token 上。

  全栈执行例子（VideoNSA + Qwen2.5-VL-7B）：
  - 模型推理算法层：视频以 4 FPS 采样 350-550 帧 → ViT 编码每帧为 64 tokens（max pixels 50,176）→ MLP Projector 映射 → 与 text tokens 拼接。在每个 self-attention 层：vision tokens 按 frame-aligned block（block_size=64 = 1 frame）分为 blocks → 压缩支路：对每个 block 做 token-mean + MLP φ 聚合为粗粒度 KV；选择支路：按 importance score 选 top-32 blocks；滑动窗口支路：保留最近 256 tokens → 三支路分别计算 attention → learnable gate 加权求和 → 与 text tokens 的 GQA 输出拼接 → FFN。最终 decoder 逐 token 生成回答。
  - 系统框架层：基于 SWIFT 训练框架，NSA 实现基于 FLA（Flash Linear Attention）库。推理使用 HuggingFace Transformers + FlashAttention-2 kernel。无自定义 Serving 框架修改。KV-cache 大小 ≈ (vision_tokens + text_tokens) × num_layers × num_kv_heads × d_head，与 dense attention 相同（因为保留所有 tokens），但 prefill 计算量降低约 96.4%。
  - 编译框架层：论文未明确说明。使用 torch.compile 编译加速。
  - kernel 调度层：FlashAttention-2 kernel 在 H100 GPU 上执行。NSA 各支路的 attention 操作复用 FlashAttention 实现的 batch GEMM + online softmax。压缩支路是主要延迟瓶颈（Figure 6），在 128K 下占主导地位，论文指出需要进一步优化其 kernel 设计和 memory efficiency。
  - 硬件架构层：NVIDIA H100 GPU，无硬件定制。

- baseline方法是什么？
  Baseline 是 **NaïveRAG for video understanding**（以 GoldFish 风格为代表），以及 **Video-RAG**（基于 CLIP keyframe + object detection + OCR 的 RAG 方法）。

  全栈执行例子（以 NaïveRAG + Qwen2.5-VL-7B 为例）：
  - 模型推理算法层：将长视频分割为多个 clip，每个 clip 用 LVLM 转换为 text description → 将 text chunks 作为独立的 retrieval document → 用户 query 的 text embedding 与各 clip description 做 cosine similarity → 检索 Top-N 相似 clips → 将 clips 的 visual frames 直接拼接输入 LVLM 生成回答。整个流程无 clip 间关系建模，每个 clip 被视为独立的无关联文档。
  - 系统框架层：开源 LVLM（Qwen2.5-VL/InternVL2.5/LLaVA-Video 等）的标准推理 pipeline，无特殊 Serving 框架修改。推理时按视频 clip 编码为 visual tokens，对每个 frame 用 ViT→Projector 映射后送入 LLM decoder 逐层自回归生成。
  - 编译框架层：论文未明确说明。
  - kernel 调度层：使用标准 PyTorch + HuggingFace Transformers 在 A100 80GB GPU 上推理，无自定义 kernel。
  - 硬件架构层：NVIDIA A100 80GB GPU，无硬件定制。

  核心缺陷：(1) **clip 独立性假设破坏时序依赖**：将长视频 clip 作为独立 document 检索，丢失了 clip 之间的时序连续性和跨 clip 实体关联。视频中同一实体（如"人物A"）可能出现在多个 clip 中，NaïveRAG 无法建模这种跨段依赖关系，导致检索不准确；(2) **检索噪声导致 LVLM 推理失败**：论文发现约 40-44% 的 failure cases 中正确 clip 已被检索到，但无关 clip 的噪声干扰了模型，导致错误输出；(3) **缺少后检索精炼机制**：检索到的 clip 直接送入 LVLM 生成回答，长视频中每帧消耗数百 tokens，无关信息会淹没关键内容，尤其在需要多 clip 时间推理的任务（如 Count、Order）中表现差；(4) **需要 proprietary LLM 的替代方案**：视频 RAG 的现有 SOTA 方法（VideoAgent, DrVideo, VideoTree）依赖 GPT-4 等闭源 API 进行多轮交互规划推理，成本高且不灵活，基于开源 LVLM 的高效长视频 RAG 未被探索。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **Vgent** 引入两大创新：(1) Graph-based Retrieval（离线视频图构建 + 图检索）；(2) Structured Post-Retrieval Reasoning（结构化子查询验证 + 信息聚合）。

  对应解决四个缺陷：

  **(1) 图结构建模跨 clip 实体关联，解决时序依赖断裂：**
  Baseline → 每个 clip 作为独立文档，clip 间无关联。
  Vgent → 构建视频知识图谱 G=(V,E)：对每个 clip 调用 LVLM 提取 entities/actions/scenes 的 JSON 描述 → 用 BGE text embedding 计算 entity 间 cosine similarity → 阈值 τ=0.7 合并语义等价 entity → 通过共享 entity 建立 clip 间 edge 连接。例如，clip_0 提取到"man in dark sweater"、clip_5 提取到"same man"，两者通过合并后的统一 entity 节点相连。图构建完成后对用户 query 是离线且独立的（query-independent），多个问题可复用同一图，无需重新处理视频。实验证实 GraphRAG 在 MLVU 上比 NaïveRAG 提升 4.1%，尤其在 Count/Order 等需要多 clip 推理的任务上效果显著。

  **(2) 结构化推理消除检索噪声：**
  Baseline → 检索到的 clips 直接送 LVLM 生成回答，硬负例（检索到但与问题无关的 clip）干扰模型推理。
  Vgent → 引入中间推理步骤：LVLM 基于 query/keywords 生成结构化 subqueries（binary yes/no 或 numerical value，如 "Is there a laptop open?" "Is someone interacting with the laptop?"） → 对每个 Top-N 检索 clip 逐一回答 subqueries → 过滤掉所有 subquery 均为否定/零的 clip，保留至少一个 subquery 正向匹配的 clip → 最多保留 r=5 个 refined clips → LVLM 跨 refined clips 汇总信息形成推理中间结果 → 将 refined clips 和推理结果作为多模态上下文输入 LVLM 生成最终答案。实验证实 Structured Reasoning 在 GraphRAG 基础上额外提升 MLVU 2.6%、VideoMME 1.6%。消融显示 confidence-based refinement 仅提升 0.2%，而 structured reasoning 提升幅度显著更大，验证了基于结构化验证的方法优于模型自反思路径。

  **(3) 训练无关（training-free）框架兼容任意开源 LVLM：**
  Baseline → proprietary LLM-based 方法依赖 GPT-4 API 的多轮交互推理。
  Vgent → 整个 pipeline 是 training-free 的，通过 prompt engineering 调用开源 LVLM 完成 entity extraction、keyword extraction、subquery generation、subquery answering 和信息聚合。在 7 种开源 LVLM（2B-7B 参数规模）上一致提升 3.0%-5.4%，其中 Qwen2.5-VL-3B + Vgent 达到 70.4%（MLVU），超越其 7B base model（68.8%），证明小模型+良好 RAG 可以超越大模型。

  **(4) 离线图构建降低多查询场景开销：**
  Baseline → Video-RAG 等方法对每个 query 需要重新提取 keyframe 和运行 object detection（query-dependent）。
  Vgent → 离线构建图（query-independent），每 min 视频约 20.13 sec 的一次性开销。完成后，每个新 query 仅需 3.93 sec/min-video 的在线检索+推理+生成。VideoMME 上每视频 3 个问题场景下，比 Video-RAG 加速 1.73×。

  全栈执行例子（Vgent + Qwen2.5-VL-7B）：
  - 模型推理算法层：离线阶段 → LVLM 将每 64 帧 clip 编码为 entities/actions/scenes 文本描述 JSON → BGE 编码 entity 描述为 1024-d embedding → cosine similarity 合并等价 entity → 构建 clip-entity 二分图。在线阶段 → 用户 query "Did I open the laptop?" → LVLM 提取 keywords {laptop, open, interact} → BGE 编码 keywords → 在图 entity 集合中检索相似 entity → 取关联的 Top-20 clips → LVLM 生成 subqueries ["Is there a laptop open?", "Is someone interacting with the laptop?"] → 每个 clip 逐一回答 subquery → 保留匹配的 clip（最多 r=5）→ LVLM 汇总 subquery 答案形成推理链 "Clip 2: closed laptop, Clip 5: person opening laptop" → LVLM 将 refined clips + 推理链作为多模态上下文生成最终回答 "Yes"。
  - 系统框架层：开源 LVLM 的标准推理 pipeline（HuggingFace Transformers + Qwen2.5-VL），无特殊 Serving 修改。BGE embedding 检索使用 FAISS 或直接 cosine similarity 计算（论文未明确说明向量索引库）。
  - 编译框架层：论文未明确说明。
  - kernel 调度层：标准 PyTorch + HuggingFace Transformers 在 A100 80GB GPU 上推理，无自定义 kernel。
  - 硬件架构层：NVIDIA A100 80GB GPU，无硬件定制。

## VFlowOpt: A Token Pruning Framework for LMMs with Visual Information Flow-Guided Optimization

- baseline方法是什么？
  Baseline 是 **基于 attention 的训练无关 token 剪枝方法**，以 FastV (ECCV 2024)、SparseVLM、VisionZip (CVPR 2025) 为代表。

  全栈执行例子（以 VisionZip + LLaVA-OneVision-7B 为例）：
  - 模型推理算法层：SigLIP ViT 将 1152×1152 图像编码为 7290 个视觉 tokens → MLP Projector 映射到 LLM 空间 → LLaVA-OneVision (Qwen2-7B) 逐层处理。VisionZip 在 ViT 编码器内用 [CLS] token 的 attention scores 评估各视觉 token 重要性，无 [CLS] 时退化为各 token 收到的平均 attention。按固定剪枝比率一次性丢弃低分 tokens（单阶段剪枝），丢弃的 token 信息永久丢失。剪枝策略（保留率、剪枝层位置）对所有 LMM 模型统一使用手工设定，无模型特定优化。
  - 系统框架层：基于 LMMs-Eval 框架的标准推理 pipeline（HuggingFace Transformers），无特殊 Serving 修改。推理时 KV-cache = O(L × T_vision × D)，token 剪枝后 KV-cache 等比例缩小。
  - 编译框架层：论文未明确说明。
  - kernel 调度层：使用标准 PyTorch + HuggingFace 推理，GPU (A100) 上运行，无自定义 kernel。
  - 硬件架构层：NVIDIA A100-SXM4-80GB GPU，无硬件定制。

  核心缺陷：(1) **Attention-based importance map 有偏**：冗余 token（如背景区域）对同类冗余 token 分配不恰当的高 attention，用所有 token 的 attention 均值估计重要性会导致背景 token 重要性被高估、关键 token 被低估；(2) **单阶段粗粒度剪枝导致信息丢失**：所有冗余 token 一次性丢弃且不可恢复，尤其在浅层剪枝时，某些看似不重要的 token 在深层可能变得关键；(3) **剪枝策略缺乏模型适配性**：同一手工设定的剪枝超参数（如保留率、剪枝位置）不加区分地应用于所有 LMM，而不同模型（LLaVA vs Qwen2-VL）的 visual information flow 特征不同，统一策略导致性能显著退化；(4) **纯 attention 信号忽略视觉内容丰富度**：仅依赖 attention 分数评估重要性，但 attention 反映的是 token 间的交互强度而非图像块的信息丰富度。信息熵高的区域（纹理复杂、物体边界）即使 attention 不高也可能包含关键视觉信息。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **VFlowOpt** 引入三大创新：(1) Attention Calibration + 熵增强的重要性估计；(2) Progressive Pruning + Token Recycling；(3) Visual Information Flow-Guided Bayesian Optimization。

  对应解决四个缺陷：

  **(1) Attention Calibration 消除冗余 token 偏差：**
  Baseline → 用所有 token 的 attention 均值评估重要性，冗余背景 token 注意力偏向同类，使背景似乎"重要"。
  VFlowOpt → 先通过全局 attention 阈值 τ = t·mean(Σ_i Σ_j A_{ij}) 筛选出"相对重要"的 token 集合 K（被 attention 关注较多的 token），再仅用 K 中 token 的 attention 计算 I_i = Σ_{k∈K} A_{ki}。这样排除了冗余 token 的 noisy attention，重要性估计更可靠。消融实验证实：移除 importance calibration 导致 MMStar 从 57.8→56.2、SQA 从 92.3→91.8。

  **(2) Progressive Pruning + Token Recycling 避免信息丢失：**
  Baseline → 单阶段剪枝，所有目标 token 一次性丢弃，信息永久丢失。
  VFlowOpt → 三阶段逐步剪枝：LLM 处理过程平均保留率 R̄ = (R1·L1 + R1·R2·L2 + R1·R2·R3·L3)/L。初始阶段保留更多 token（浅层视觉信息更关键），深层逐渐激进剪枝（冗余增加）。Token Recycling：将剪枝 token 按 a×a 空间网格分组，组内以重要性为权重做加权平均融合 t_merged = Σ I_i·t_i / Σ I_i，融合 token 替换该网格最高重要性 token 位置归入保留集合。这样在减少 token 数量的同时，压缩保留了低重要性区域的视觉特征。消融实验证实：移除 token recycling 导致 POPE 从 89.1→86.8；移除 progressive pruning 导致 MMStar 从 57.8→56.0。

  **(3) Visual Information Flow Optimization 自动适配不同 LMM：**
  Baseline → 手工设定保留率，所有模型用同一策略。
  VFlowOpt → 基于 LMM 可解释性研究的发现（视觉信息从 vision tokens → query text tokens → 最后位置 last token 聚合），将剪枝策略设计建模为优化问题：max CosineSim(h_f, g_s(h_f))，即最大化无剪枝与剪枝最后 token 表示的余弦相似度。用 Bayesian Optimization（GP + Expected Improvement, 50 迭代, 30 样本, ≈30 分钟）自动搜索超参数 (R1, R2, R3, t, α, a)。核心洞察：不同的 LMM 有不同的内部信息流特征——LLaVA、LLaVA-NeXT、Qwen2-VL 三者的 ViT 架构、LLM backbone、信息聚合模式各不同——统一策略必然 suboptimal，自动优化实现"一模型一策略"的定制化。

  **(4) 熵项捕捉视觉内容丰富度：**
  Baseline → 纯 attention 信号评估重要性，忽略图像内容。
  VFlowOpt → 将图像块信息熵 H(V_i) = -Σ p_k·log(p_k)（256 灰度级）归一化后加权加入重要性得分，α 控制熵贡献权重。高熵区域（纹理复杂、细节丰富）即使 attention 一般也会获得加分，实现"attention 告诉你 token 间谁重要，熵告诉你谁的内容值得看"的双信号评估。两个信号均由 Bayesian Optimization 自动调权，无需手工设定 α。

- baseline方法是什么？
  Baseline 是 **Transformer-based MLLMs for long video understanding**，以 Video-XL (Qwen2-7B + 内部 token 压缩)、VideoChat-Flash (Qwen2-7B + 层级压缩)、LLaVA-Video (Qwen2-7B + ViT fine-tuning) 为代表。

  全栈执行例子（以 Video-XL 为例）：
  - 模型推理算法层：ViT 编码视频帧为 vision tokens → Projector 压缩（可选 ToMe 等）→ 全 Transformer LLM (Qwen2-7B) 逐层计算 self-attention：每层 attention 复杂度 O(T²)，其中 T = T_vision + T_text。Video-XL 在 LLM 内部做 token 压缩，基于 attention score 选择保留的 vision tokens 并生成新的 compact tokens。所有层均为 self-attention，KV-cache 随序列长度线性增长。
  - 系统框架层：基于 HuggingFace Transformers 的标准 MLLM pipeline，无特殊 Serving 框架修改。推理时 KV-cache 内存 = O(L × T × D)，长视频场景下 KV-cache 成为主要瓶颈。
  - 编译框架层：论文未明确说明。
  - kernel 调度层：使用标准 Flash Attention 加速 Transformer attention 计算，无自定义 kernel。长序列下 attention 仍是计算瓶颈。
  - 硬件架构层：NVIDIA GPU（如 A100/H100），无硬件定制。

  核心缺陷：(1) **Attention 计算复杂度高**：全 Transformer backbone 的 self-attention 为 O(T²)，处理小时级视频（如 2.7M tokens）不可行；(2) **KV-cache 内存爆炸**：全 Transformer 的 KV-cache 为 O(L×T)，长序列推理时显存成为瓶颈；(3) **Vision token 冗余未在 hybrid 架构中探索**：现有 token dropping/compression 策略均基于 Transformer attention scores 设计（如基于 attention 分数排序丢弃），在 hybrid Mamba-Transformer 架构中信息存储/传递机制不同，直接迁移不可行；(4) **Compression 信息丢失**：token dropping 策略不可逆，被丢弃的 token 信息永久丢失；token compression 生成新 special tokens 则破坏了原始 token 的身份和位置信息。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **TimeViper** 引入两大创新：(1) Hybrid Mamba-Transformer LLM Backbone（Nanov2-9B），用 SSM 替代大部分 attention 层；(2) TransV token transfer mechanism，通过 gated cross-attention 将被丢弃的视觉 token 信息转移到指令 token 中。

  对应解决四个缺陷：

  **(1) O(n) 替代 O(n²)：**
  Baseline (Video-XL) → 全 Transformer 每层 O(T²) attention。
  TimeViper → 27 层 Mamba-2 每层 O(n) 递推计算（h_t = A_t·h_{t-1} + B_t·x_t），仅 4 层 self-attention 保留 O(n²)。实际效果：输入 32K tokens（约 2K frames×16 tokens）、输出 1K tokens、batch_size=32 时，TimeViper 每秒生成 token 数比 Qwen3 高 40.1%。

  **(2) O(1) KV-cache 替代 O(L×T) KV-cache：**
  Baseline → KV-cache 随序列长度线性增长，长视频推理显存爆炸。Vanilla 模型在 128 frames 即 OOM。
  TimeViper → Mamba-2 层仅需维护固定大小隐状态 h_t ∈ R^{N×D}（N≈128），无 KV-cache 存储需求；4 层 self-attention 的 KV-cache 远小于全 Transformer。ToMe+TransV 联合压缩后，4K frames 时内存节省 54.8%，可处理 10K+ frames。

  **(3) Hybrid 架构专用的 token 压缩策略：**
  Baseline → Attention-based token dropping（如 PDrop、PyramidDrop），依赖 Transformer attention scores 识别冗余 token，在 hybrid 架构中 Mamba 层缺乏显式 attention scores。
  TimeViper → 首先通过信息交换分析实验（blocking V2I/V2R）揭示 **vision-to-text information aggregation 现象**：视觉信息在浅层从 vision tokens 汇聚到 instruction tokens，深层 vision tokens 近乎 100% 冗余。基于此发现设计 TransV：(a) 浅层用 uniform dropping 避免 attention score 不可靠的问题；(b) 深层用 attention-guided dropping（以最后一个 instruction token 为 query 计算 attention scores）此时 attention 已可靠；(c) 在丢弃前通过 gated cross-attention 将被丢弃 token 的信息转移到 instruction tokens，避免信息丢失。α_l 初始化为 0 → 训练后学习到最优转移比例。

  **(4) Token Transfer 替代 Token Dropping/Compression：**
  Baseline → token dropping 不可逆丢失信息；token compression 生成新的 special tokens 破坏原始 token 身份。
  TimeViper → TransV 的 cross-attention 将被丢弃的 vision tokens 作为 KV、instruction tokens 作为 Q，计算得到的信息增量通过 tanh(α)·CrossAttn 加回到 instruction tokens。这意味着视觉信息被"转移"而非"丢弃"，instruction tokens 承载了原属于被丢弃 vision tokens 的关键信息。保留的 5% vision tokens + enriched instruction tokens 足以支撑任务性能。

  全栈执行例子（对比 baseline）：
  - 模型推理算法层：
    Baseline (Video-XL) → ViT encode → Projector → Qwen2 self-attention 逐层计算（每层 O(T²) attention + O(T) KV-cache 增长）→ 基于 attention score 选择保留/丢弃 tokens → 生成回答。
    TimeViper → SigLIP encode frames (768 tokens/frame) → ToMe 压缩至 16 tokens/frame → Concat [vision_tokens, instruction_tokens] → 逐层交替通过 Mamba-2 (SSM 递推，h_t = A_t h_{t-1} + B_t x_t，O(n)，无 KV-cache) 和 Self-Attention（标准 QK^T/sqrt(D)，O(n²)，有 KV-cache）→ 第 7 层 TransV：UniformDrop 50% vision tokens，CrossAttn(Q_inst, KV_dropped_vis)→ instruction tokens 吸收视觉信息 → 丢弃被转移的 vision tokens → 第 39 层 TransV：Attention-guided drop 90% remaining vision tokens（仅保留原 vision tokens 的 5%）→ CrossAttn 转移信息 → 深层继续 processing（512 frames 时 context 从 8192+text 降至 ~435+text tokens）→ 生成回答。关键区别：Mamba-2 层通过遗忘-记忆门控机制隐式建模时序位置（无需 MRoPE），TVG 任务 mIoU 仍达 40.5，超过使用 MRoPE 的 Qwen2.5-VL-7B(43.6) 的 93%。
  - 系统框架层：
    Baseline → HuggingFace Transformers 标准 MLLM，推理时 prefill → decode 两阶段，KV-cache 管理。
    TimeViper → 同标准 MLLM 训练 pipeline：两阶段训练（Image-Text Alignment 3M → Video Instruction Tuning 4.8M），Data Packing 支持 TransV 可变序列长度。推理时 TransV 在 prefill 阶段执行 token 压缩，decode 时 context 已大幅削减 → prefilling time 在 4096 frames 时减少 15.7% vs ToMe-only。兼容 HuggingFace Transformers。
  - 编译框架层：论文未明确说明。使用 PyTorch + 标准训练框架。
  - kernel 调度层：论文未明确说明。Mamba-2 SSM 使用标准 selective scan kernel，self-attention 使用 Flash Attention 加速。
  - 硬件架构层：论文未明确说明。NVIDIA GPU 推理。

## TimeSearch-R__Adaptive_Temporal_Search_for_Long-Form_Video_Understanding_via_Self-Verification_Reinforcement_Learning

- baseline方法是什么？
  Baseline 是**手设计的时序搜索 (Hand-crafted Temporal Search)**，以 VideoAgent、T*、VideoTree 为代表。典型执行流程：
  - 模型推理算法层：VideoAgent 用 LLM (GPT-4) 作为中央 agent，通过 prompt 驱动多轮工具调用——先调用 VLM (GPT-4o) 做帧 captioning，再调用 CLIP 做帧检索，然后在纯文本模态中聚合信息做推理预测答案。T* 先用 VLM 从问题中提取目标物体，再调用目标检测模型 (YOLO-world-110M) 定位包含目标物体的关键帧，最后用检索到的帧集完成问答。VideoTree 引入树结构搜索来提高效率。所有方法均依赖人工设计的搜索工作流，缺乏端到端优化，搜索策略是次优的。
  - 系统框架层：VideoAgent 和 T* 均基于 API 模型 (GPT-4/GPT-4o) 做多轮调用，无训练过程。各 agent 通过 prompt engineering 编排 VLM/CLIP/YOLO 等子模型，非训练型框架。
  - 编译框架层：论文未明确说明。
  - kernel 调度层：论文未明确说明。T* 使用 YOLO-world-110M (轻量检测器) 做帧检索，VideoAgent 使用 CLIP-1B 做相似度搜索，均为标准模型推理无自定义 kernel。
  - 硬件架构层：推理均在 GPU 上 (A100)，无硬件架构定制。

  核心缺陷：(1) **搜索策略次优**：手设计的搜索工作流无法泛化到不同问题/视频类型，搜索决策缺乏数据驱动的优化；(2) **搜索-推理割裂**：帧集在推理开始前固定，而实际视频推理是动态过程，中间推理结果应能驱动进一步搜索；(3) **多模型编排复杂**：VideoAgent 需协调 LLM agent + VLM captioner + CLIP retriever，T* 需 VLM + YOLO 检测，流水线脆弱且难以端到端优化；(4) **端到端延迟高**：VideoAgent 的 end-to-end latency 为 34.9s (Haystack-Ego4D)，多模型多次调用产生大量 overhead。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **TimeSearch-R** 将时序搜索重新定义为 text-video 交错的思维过程 (Interleaved Text-Video Thinking)，通过端到端 RL 从数据中学会最优搜索策略。核心机制 GRPO-CSV (Completeness Self-Verification) 补充原始 GRPO 仅奖励最终答案的缺陷，监督中间搜索步骤。

  对应解决四个缺陷：
  **(1) 端到端优化替代手设计工作流**：TimeSearch-R 将搜索策略学习建模为 policy optimization：policy model π_θ 在每个推理步 k 自主决定是否搜索、搜索哪个时间区间 [t_s^k, t_e^k]、用什么文本 query q^k、要多少帧 F。通过 GRPO 的 8 个 rollout 比较和 advantage-based 更新，模型从数据中学会最优搜索策略，无需人工规定搜索 pipeline。Temporal F1 从 baseline T* 的 2.5 提升到 8.1（3 倍以上）。
  **(2) 搜索-推理深度交错**：将搜索和推理融为一体——每轮 <think>...</think> 推理后可以跟 <tool_call>...</tool_call> 搜索请求，搜索结果直接追加到 CoT 供下一轮推理使用，实现了搜索和推理的循环迭代。这模拟了人类"假设驱动搜索"的认知模式：根据中间推理结果驱动进一步搜索。
  **(3) 单一的端到端模型**：仅需一个 Qwen2.5-VL-7B 模型完成所有推理和搜索决策，搜索函数 (SigLIP-400M + DPP) 仅作为轻量环境接口执行帧检索。不需要像 VideoAgent 一样协调多个异构模型。
  **(4) 更低延迟**：TimeSearch-R end-to-end latency 为 13.4s，比 VideoAgent 的 34.9s 降低 61.6%。因为搜索决策直接由 policy model 生成（无需额外 LLM agent 调度），且 DPP 帧选择比 CLIP retrieval 更高效。

  全栈执行例子（对比 baseline）：
  - 模型推理算法层：
    Baseline (VideoAgent) → LLM agent 分析问题 → 调用 VLM captioning 获取帧描述 → 调用 CLIP 检索相似帧 → 在文本空间聚合描述做推理 → 输出答案。搜索和推理在模态间转换，搜索策略由 prompt 固定。
    TimeSearch-R → 初始预览 Ṽ (768 frames @ 2fps) → π_θ 在 <think> 中推理 "I need to find when the person starts cooking" → <tool_call>{"name":"seek_video_frames","arguments":{"query":"person cooking in kitchen","start_time":120,"end_time":300,"num_frames":8}}</tool_call> → SigLIP-400M 计算候选帧嵌入 → DPP 选出 8 帧（兼顾相关性和多样性）→ 帧 + 时间戳返回追加到 CoT → π_θ 继续推理 "The cooking started at 180s-240s, now I need to check..." → 继续搜索或输出 <answer>。关键区别：搜索参数 (query, start_time, end_time) 由模型端到端学会，而非 hand-crafted；CSV 奖励确保搜索到的帧确实足以支撑正确答案。
  - 系统框架层：
    Baseline → 多模型编排 (GPT-4 agent + VLM + CLIP/YOLO)，API 调用频繁，无模型训练。
    TimeSearch-R → SFT (GPT-4o 生成交错 CoT 数据) → RL with GRPO-CSV on TRL library。训练：32 × A100 GPU，DeepSpeed ZeRO-3 Offload，vLLM colocate mode，Flash Attention 2.0，bfloat16。KL penalty β=0.005，batch size per GPU=1，gradient accumulation=2，AdamW lr=1e-6。推理：单模型 end-to-end，通过 <tool_call> 接口调用 SigLIP + DPP 搜索函数。
  - 编译框架层：论文未明确说明。使用 PyTorch native DDP + DeepSpeed ZeRO-3。
  - kernel 调度层：论文未明确说明。使用 Flash Attention 2.0 加速 attention 计算，DPP 搜索使用 standard matrix operations。
  - 硬件架构层：训练在 32 × NVIDIA A100 GPU 上，推理在 A100 GPU 上。无硬件架构定制。


## Temporal Preference Optimization of Large Multimodal Models

- baseline方法是什么？
  Baseline 是 video-LMM 的 **标准 Supervised Fine-tuning (SFT)** 范式：video-LMM（如 LongVA-7B、LLaVA-Video-7B）通过弱监督学习隐式获取时序定位能力——训练时依赖视频帧与文本回答之间弱对应关系的 next-token prediction loss，缺乏显式的时序对齐信号。全栈执行例子：
  - 模型推理算法层：video-LMM 将视频帧均匀采样 F=64 或 128 帧（LongVA）或 96 帧（LLaVA-Video）拼接为视觉 token 序列，与文本问题 token concatenate 后送入 LLM backbone 自回归生成回答。没有任何偏好信号告诉模型"什么样的回答是时序上更准确/更相关/更完整的"。训练数据来自视频 caption 合成的 Q&A 对（如 ShareGPT4Video），缺乏对模型时序定位能力的直接监督。
  - 系统框架层：标准 PyTorch + Transformers 训练框架，8 × A100 80GB，DeepSpeed/FSDP（论文未明确说明具体分布式框架，但 standard full fine-tuning）。
  - 编译框架层：论文未明确说明。
  - kernel 调度层：论文未明确说明。标准 attention 推理，无特殊 kernel 优化。
  - 硬件架构层：论文未明确说明。

  核心缺陷：(1) **缺乏显式时序对齐**：SFT 的 next-token prediction 只优化文本 token 匹配概率，不区分回答是否实际参考了正确的视频时间片段，模型可能"蒙对"文字但并未真正定位到正确帧；(2) **无法区分时序答案质量**：当模型给出两个不同的回答时，SFT 无法区分哪个在时序上更准确——训练信号仅来自 ground truth 文本而非视频-回答的时序对齐程度；(3) **数据标注成本高**：若要提供显式时序标注（如 temporal grounding 标签），成本极高且难以扩展到大规模训练集。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **Temporal Preference Optimization (TPO)** 通过操纵视频输入自动生成对比偏好数据，在 post-training 阶段使用 DPO 注入时序偏好先验。核心洞察：不需要人工标注时序标签——只需改变视频输入（完整/不完整/不相关帧）的"可见证据量"，就能让模型自己产生质量有差异的回答，从而自动构建时序偏好对。

  对应解决三个缺陷：
  **(1) 操纵视频输入替代人工时序标注**：TPO 不依赖人工标注时序标签，而是通过操纵视频帧本身来制造时序信息差——preferred response 使用完整相关帧（充分时序证据），dis-preferred response 使用不完整帧（部分证据）或不相关帧（无证据）。这自动确保 preferred > dis-preferred 的时序质量，无需人工判断"哪个回答时序上更好"。
  **(2) DPO 注入时序偏好信号**：将自动生成的偏好对 (V, Q, r⁺, r⁻) 送入 DPO 训练，lose 函数 L_DPO = -log σ(β(log π_θ(r⁺)/π_ref(r⁺) - log π_θ(r⁻)/π_ref(r⁻))) 直接教模型区分"好回答"和"差回答"（从时序定位角度），同时配合 SFT loss L_SFT = -log π_θ(r⁺) 保持基础生成能力。
  **(3) LLM-based Post-Filtering 保证数据质量**：GPT-4o-mini 自动过滤三类噪声——dis-preferred 偶然优于 preferred、preferred 事实错误、问题模糊——确保偏好信号的可靠性，提供可扩展的数据清洗方案。

  全栈执行例子（对比 baseline）：
  - 模型推理算法层：
    Baseline → LongVA/LLaVA-Video 采样 64/96/128 帧拼接送入 LLM，逐 token 生成回答，仅依赖 SFT 阶段的弱时序对应。
    TPO → 第一阶段（数据生成）：对每个视频 V，CogVLM2 首先生成逐帧 caption → GPT-4o-mini 根据 caption 生成问题 Q → video-LMM 用 Q + 完整帧 F 生成 preferred r⁺ → video-LMM 用 Q + 不完整帧或不相关帧生成 dis-preferred r⁻ → GPT-4o-mini 评估 r⁺ vs r⁻ 的质量并过滤。第二阶段（训练）：用 (V, Q, r⁺, r⁻) 四元组进行 DPO + SFT 联合优化，β 控制 KL 散度约束，α 控制 SFT 权重。
    关键差别：TPO 不改变推理时的模型架构，仅通过 post-training 阶段的偏好信号让模型学会在推理时更好地利用时序信息。消融实验显示 TPO 在 128 帧输入下优于 baseline 在 64 帧输入下的表现，且 TPO 随帧数增长性能持续提升而 baseline 在 >64 帧后退化。
  - 系统框架层：论文未明确说明。标准 PyTorch full fine-tuning，8 × A100 80GB，batch size 64，约 4 小时训练。
  - 编译框架层：论文未明确说明。
  - kernel 调度层：论文未明确说明。
  - 硬件架构层：论文未明确说明。训练在 8 × NVIDIA A100 80GB 上。

## SPIKE-RL__Video-LLMs_meet_Bayesian_Surprise

- baseline方法是什么？
  Baseline 是 Video-LLM 的 **uniform frame sampling**（均匀帧采样）：将视频帧按固定间隔均匀采样 F 帧输入模型，所有帧的权重相同，无论帧内容如何。全栈执行例子：
  - 模型推理算法层：Video-LLM（如 Qwen2.5-VL）将视频视为 "bag of frames"，均匀采样后一次性自回归生成结果（caption / QA 答案）。没有信念演化过程，没有帧选择策略，对 routine 和 surprising 帧一视同仁。
  - 系统框架层：论文未明确说明。标准 Video-LLM 推理 pipeline，直接调用模型 API 或本地推理。
  - 编译框架层：论文未明确说明。
  - kernel 调度层：论文未明确说明。GPU 上标准 FlashAttention-2 + bfloat16 推理。
  - 硬件架构层：论文未明确说明。在 H100 GPU 上运行。

  核心缺陷：(1) **缺乏信念追踪**：模型不维护对视频故事的演化理解，无法区分 routine 和 surprising 帧；(2) **信息冗余**：均匀采样倾向于采样高频 mundane 帧，错过关键的 surprising 时刻；(3) **query-agnostic 但无原则**：虽不依赖查询，但采样策略毫无信息优先级的引导。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **SPIKE/SPIKE-RL** 将 Bayesian Surprise 引入 Video-LLM，通过显式信念追踪和 surprise 引导的帧采样解决 uniform sampling 的三个缺陷：
  
  **(1) 信念追踪替代 bag-of-frames**：SPIKE 在每个时间步维护显式概率分布 P(belief | context)，生成文字化信念假设 B_t = {b_{t,1}, ..., b_{t,N}}（如 "the man will continue walking"），然后计算 P_prior（仅用历史上下文 H_t + 前序帧 W_t）和 P_post（加入当前观察帧 O_t），通过 KL散度 D_KL(P_post || P_prior) 量化 surprise。这给 Video-LLM 注入了人类式的"预期-现实"对比机制。
  
  **(2) surprise-weighted 采样替代 uniform 采样**：将 F 帧预算按 surprise 得分比例分配：p_i = softmax(S_i/τ_s)，高 surprise 段可被多次采样，确保关键事件帧不被遗漏。τ_s 控制采样集中度（τ_s=0.7 实验设置）。
  
  **(3) GRPO 优化信念质量**：SPIKE-RL 用 RL 训练假设生成器，reward 来自最终 caption 与 ground truth 的 LLM-Match 相似度，通过策略梯度反向优化中间信念假设质量。

  全栈执行例子（对比 baseline）：
  - 模型推理算法层：Baseline → Qwen2.5-VL 均匀采样 F 帧一次性推理。SPIKE → 先对视频做 W 帧滑动窗口，每一步生成 N=3 个文字假设 + 计算 Bayesian Surprise（KL散度），再按 surprise 概率采样 F 帧送入 Qwen2.5-VL 推理。SPIKE-RL → 额外使用 GRPO 在 2000 视频集上训练假设生成器，3 条 rollout 轨迹、LLM-Match reward、Z-score 归一化 advantage。
  - 系统框架层：论文未明确说明。SPIKE 是即插即用的推理时模块，替换 Video-LLM 的 uniform sampling layer。
  - 编译框架层：论文未明确说明。
  - kernel 调度层：论文未明确说明。训练使用 DeepSpeed ZeRO-3 offload，推理使用 FlashAttention-2。
  - 硬件架构层：论文未明确说明。训练在 4×H100 单节点上。

## StreamingVLM__Real-Time_Understanding_for_Infinite_Video_Streams

- baseline方法是什么？
  Baseline 是现有 VLM 处理长视频的三种方式：
  
  **(a) Full Attention**：对视频全部帧计算 full causal attention。O(T²) 计算复杂度，内存无界，视频超出训练长度后性能退化。执行例子：Qwen2.5-VL-7B-Instruct 对完整视频做 full attention → 一次自回归生成 caption，所有 vision tokens 在所有层 attend 到所有历史 tokens。2-5 分钟后超出 training context length，latency 急剧上升直至 OOM。
  
  **(b) Sliding Window Attention (w/o Overlap)**：将长视频切分为固定长度 chunk，每个 chunk 内独立处理，上下文在 chunk 边界重置。执行例子：LiveCC-7B-Instruct 在 chunk 模式下，每 100s 视频重置 KV cache，只读入当前 chunk 的 frames + 前序 text 作为 prompt 生成解说。短 chunk 破坏跨块连贯性（coherence），长 chunk 则延迟高且仍会超出 training length（Figure 6）。
  
  **(c) Sliding Window Attention (w/ Overlap)**：相邻 chunk 有重叠，window 滑动时 recompute 重叠部分的 attention。执行例子：维护固定长度窗口（如 100s），每次新帧到来重新计算窗口内全部 tokens 的 attention，KV cache 不跨窗口复用。维持了 coherence 但 computation redundancy 严重，latency 高且不稳定（Figure 7 中 SI. w/ Overlap 曲线）。
  
  训练无关的 KV cache 驱逐方法（如 ReKV）：训练时使用 full attention，推理时强行驱逐 tokens 会破坏模型期望的 attention pattern，对 fine-tuned 模型常常导致无输出（Table 2）。
  
  全栈执行例子（以 Sliding Window w/o Overlap 为例）：
  - 模型推理算法层：VLM（Qwen2.5-VL / LiveCC）将视频按 100s chunk 切分，每个 chunk 独立做 full attention，chunk 间通过 previous text 传递上下文 → 自回归生成解说。没有 KV cache 跨 chunk 复用，跨块长程依赖丢失。
  - 系统框架层：论文未明确说明。标准 VLM 推理 pipeline，逐 chunk 调用模型。
  - 编译框架层：论文未明确说明。
  - kernel 调度层：论文未明确说明。GPU 上标准 FlashAttention，无自定义 kernel。
  - 硬件架构层：论文未明确说明。在 H100 GPU 上运行。
  
  核心缺陷：(1) **缺乏训练-推理一致性**：训练时用 full attention，推理时用 sliding window，模型遭遇 distribution shift；(2) **计算冗余**：overlap 模式反复 recompute attention，w/o overlap 模式丢失上下文连贯性；(3) **位置编码漂移**：native RoPE 索引随视频增长超出训练范围，导致 long-horizon 性能退化；(4) **不支持真正的无限流式输入**：现有方法处理有限长度 video clip，无法应对 continuous/infinite 视觉流。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **StreamingVLM** 通过统一训练-推理框架 + KV cache 复用 + contiguous RoPE 解决上述四个缺陷：

  **(1) 训练-推理一致性：Overlapped-Chunk Full-Attention Training**
  训练时将长视频切为 W=24s 的 overlapped chunk（O=12s），每个 chunk 内做 full attention，vision/text tokens 以 1s 间隔交错排列（而非传统 VLM 的 vision-then-text 布局）。这种 overlapped full-attention 的 effective attention pattern 与推理时的 "sink tokens + 近期 text 长窗口 + 近期 vision 短窗口" 高度近似（Figure 4 右侧），使模型学到 recency bias 而非跨 chunk 的突然重置。只在 text position 计算 loss，并在无解说词的秒插入占位符 "..."，训练模型同时学会"何时说话、何时沉默"的流式行为。

  **(2) 计算效率：Streaming-Aware KV Cache + Asymmetric Retention**
  推理时维护紧凑 KV cache，复用历史 KV 而非 recompute。非对称保留策略：attention sink tokens（Tsink=512，system prompt + 早期 text）保证 attention 稳定性；近期 text 长窗口（Twindow=512）保留长期语言记忆；近期 vision 短窗口（Vwindow=16s）跟踪连续动作。旧 vision tokens 优先驱逐（视觉冗余高），旧 text 仅在超 budget 时驱逐（语义信息密度高）。此设计消除了 Sliding Window w/ Overlap 的重计算，使 latency 保持低且稳定（Figure 7），单 H100 达到 8 FPS 实时性能。

  **(3) 位置编码稳定性：Contiguous RoPE**
  当旧 tokens 被驱逐后，后续及新 tokens 的 RoPE 位置索引左移，保持与最后保留 token 的数值连续性。当视频长度超出总窗口尺寸后，effective RoPE 索引停止增长，保持有界。这使位置编码始终在训练分布内，防止 long-horizon 性能退化。Ablation（Table 4）显示 native RoPE 在 infinite stream 上急速退化（win rate 25.09 vs. GPT-4o），而 contiguous RoPE 维持 66.18%。对 Qwen-VL 的 3D RoPE（time, height, width），同样应用 contiguous 左移规则。

  **(4) 无限流式输入：数据 + 推理协同设计**
  构建 Inf-Streams 数据集（>4000 小时体育解说 SFT + 14K 高质量 annealing 样本）和 Inf-Streams-Eval benchmark（20 场完整比赛，平均 2.12h，per-second 帧-文本对齐），验证真正无限流式理解能力。推理时按 1s 步进接收新视觉帧，KV cache 增量更新，自回归生成解说或等待下一个信息 token，实现 closed-loop streaming behavior。

  全栈执行例子（对比 baseline）：
  - 模型推理算法层：Baseline → VLM full attention 或 sliding window 逐 chunk 处理。StreamingVLM → SFT 后的 Qwen2.5-VL 接收每 1s 新帧 → vision encoder 编码为 V_new tokens → interleave text tokens → contiguous RoPE 计算 bounded 位置索引 → attention 计算时 Q 仅 attend 到 KV cache 中保留的 sink+text_window+vision_window tokens（复用历史，不 recompute）→ KV cache 按 asymmetric policy 驱逐 → 自回归生成解说（无解说时输出 silence placeholder）→ 循环。训练时 overlapped chunk（W=24s, O=12s）内 full attention 的 attention pattern 与推理时 sink+sliding window 的 effective 注意力模式一致。
  - 系统框架层：论文未明确说明。基于 Qwen2.5-VL 的推理管线，修改 token interleaving layout + KV cache management。
  - 编译框架层：论文未明确说明。
  - kernel 调度层：论文未明确说明。使用标准 FlashAttention，GPU 上 bfloat16 推理。
  - 硬件架构层：论文未明确说明。训练 128 H100-days，推理单卡 H100。

- baseline方法是什么？
  Baseline包括两类方法：
  (a) **DIRECT范式**：给定采样的视频帧，通过单轮推理（single sequence prediction）直接输出最终答案。代表：Qwen3-VL系列（直接将128帧concat后一次性输入MLLM自回归生成答案）、Video-R1（GRPO训练但仅用option-matching和ROUGE作为reward）、VideoRFT（semantic-consistency reward）、LongVILA-R1（用序列并行支持数千帧的RL训练，但同样用string-matching reward）。这些方法的核心局限：对所有video duration一视同仁——短视频和2小时长视频都用相同计算量处理，且依赖string-matching reward导致在open-ended问题上无效。
  (b) **AGENT范式（Over-reliance on Temporal Grounding）**：多轮agent系统（VideoAgent, VideoMind, VideoExplorer, LVAgent等），核心依赖temporal grounder在整个视频范围内迭代定位事件。局限：(i) 缺乏鲁棒的长视频时序定位模型，(ii) 系统设计中缺乏知识驱动推理能力，(iii) 过度工程化于MCQ问题导致open-ended性能差。

  Baseline全栈执行例子（以Qwen3-VL-8B-Instruct DIRECT模式，128 frames输入为例）：
  - 算法层：视频→均匀采样128帧(2 FPS)→每帧ViT编码→temporal pooling(factor=2)→约128×min_tokens=16384 visual tokens→LLM单轮自回归生成答案。不区分短视频和长视频的处理方式，对2小时视频只采样128帧导致无法回答需要时间定位的问题。
  - 系统框架层：PyTorch模型 + vLLM serving。无专用调度。
  - 编译框架层：论文未明确说明。
  - kernel调度层：FlashAttention标准kernel。无自定义kernel。
  - 硬件架构层：NVIDIA H100 GPU集群。DIRECT RL baselines（Video-R1等）也需要多GPU训练。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  SAGE通过三方面设计解决Baseline的缺陷：

  **(a) 缺陷1：DIRECT方法对所有时长一视同仁，缺乏自适应推理能力** → Any-Horizon Agent System
  SAGE将统一单轮推理改为自适应Agent系统。SAGE-MM（orchestrator VLM）根据任务难度和视频时长自主决定：简单问题单轮直接回答，复杂/长视频问题通过多轮tool calling逐步推理。Nmax=11限制步数。关键设计：
  - Stage-1（Context VLM）：一次性产出video_context + query_intent + 首步action
  - Stage-2（Iterative Reasoner）：迭代判断answerable，每步产出tool call或final answer
  - 5 tools + analyze tool提供多种信息获取渠道，不依赖单一temporal grounding

  **(b) 缺陷2：AGENT系统过度依赖temporal grounding，缺乏知识驱动推理** → Knowledge-Driven Multi-Tool System
  现有AGENT系统几乎完全依赖temporal grounder在整个视频中定位事件。SAGE引入web-search和speech transcription工具，使系统能利用外部知识和语音信息智能缩小搜索空间。案例：知道F1 2024赛季排名后，看2025 livery reveal视频时可以推理出目标片段的大致时间区间。实验验证（Table 10）：去除web-search/parse-website导致overall降2.5%，去除transcribe-speech降5.5%（verbal问题降36.5%）。

  **(c) 缺陷3：现有RL recipe依赖string-matching reward，对open-ended问题无效** → Multi-Reward GRPO + LLM-as-Judge
  Video-R1/VideoRFT/LongVILA-R1等用option-matching和ROUGE作为reward，只能处理MCQ问题。SAGE使用：
  - GPT-4o作为LLM-Judge判断答案语义正确性（binary correctness verdict）
  - 多层reward设计：format(+0.05/-0.10) + reasonable-tool(+0.10/-0.10) + args-repeat(-0.05×√rep) + args-valid(-0.1/0) + accuracy(-2.0~+1.25)
  - 正确+使用visual tools额外+0.25奖励（鼓励视觉信息利用）
  - 前100步Nmax=6稳定训练，防止RL初期因长trajectory方差过大导致不收敛
  结果：RL后SFT模型改善4.1%（Qwen3-VL-4B，Table 7），open-ended从51.1%→57.4%。

  对比Baseline的全栈执行例子（SAGE, Qwen3-VL-8B-Instruct SFT+RL SAGE-MM）：
  - 算法层：视频→128帧采样→SAGE-MM接收T|F|Q|M→Stage-1输出video_context + tool_call → 执行tool（如transcribe 2分钟segment, web search）→ Stage-2迭代（平均1.74-3.54 turns根据视频时长）→ final answer。对于短视频（<60s），单轮直接回答率更高（Tab 9）。对于长视频（>600s），平均2.49 turns的multi-turn推理。
  - 系统框架层：vLLM serving所有模型。Tool执行链：Serper Google Search API（web-search）+ Whisper-large-v3（transcribe-speech）+ Qwen3-VL-30B-A3B-Instruct（ground-event & analyze）。推理耗时8.6s/sample，远快于VideoMind 24.7s和VideoAgent 1445s。
  - 编译框架层：论文未明确说明。
  - kernel调度层：标准FlashAttention，无自定义kernel。直接复用vLLM PagedAttention。
  - 硬件架构层：NVIDIA H100 GPU集群（训练16×H100）。冻结visual encoder和projector，仅训练LLM部分。

- baseline方法是什么？
  （第二组baseline视角——合成数据生成方法）现有的长视频QnA数据生成方法（如LongVILA、Eagle 2.5）采用bottom-up pipeline：将视频切割为10-30秒的subclip，分别用模型处理生成caption或QnA pairs，再聚合。对1小时视频需处理120个subclip，单个subclip 10秒即需20分钟。

  Paper方法：一次性利用Gemini-2.5-Flash的长上下文能力（支持2小时视频），通过carefully designed prompt直接生成10-20个覆盖全视频时间跨度的QnA pairs。通过percent_video_parsed字段强制模型按时间顺序生成并覆盖至少90%视频。成本约为人工标注的1/100，时间约为subclip pipeline的1/10。人工验证1700+样本仅5%错误率。

- baseline方法是什么？
  Baseline 是传统的**固定分辨率全量编码**（Vanilla）以及两类主流后编码压缩方法：
  (a) **Model-side compression**：在视觉编码后对 token 进行剪枝或合并（ToMe token merging, VisionZip attention-guided pruning, FlashVid spatiotemporal tree-based merging）。这些方法"接受编码器的全分辨率输入作为固定成本"，先付出全部计算再试图压缩，一旦细粒度证据被丢弃就无法恢复。不规则 token 布局还会破坏 FlashAttention 和 vLLM 等优化内核的兼容性。
  (b) **Output-side agentic reasoning**：通过迭代检索或缩放步骤多次调用 backbone（VideoAuto-R1 等）。虽可恢复覆盖范围，但每步检索需要独立 backbone 调用，首轮粗视图常欠采样目标证据。

  Baseline（Vanilla Qwen2.5-VL-7B，32 frames 全分辨率）全栈执行例子：
  - 算法层：视频 T=32 frames → 每帧固定 res 448×448 → ViT pathify (P=14) → 32×32×32=32768 visual tokens → LLM backbone 自回归生成答案。计算代价与像素量二次方关系，但回答复杂查询所需的证据在时间上高度稀疏。
  - 系统框架层：PyTorch + vLLM/SGLang for serving。无专用 Serving 框架修改。
  - 编译框架层：论文未明确说明。
  - kernel调度层：FlashAttention 标准 kernel，无自定义 kernel。但 model-side 压缩方法产生的不规则 token 布局会破坏这些优化内核。
  - 硬件架构层：NVIDIA H100 GPU 集群（训练 32×H100），推理在 4-GPU vLLM engine。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  ResAdapt 通过**输入侧自适应（Input-side Adaptation）**原则将干预点从"编码后"移到"编码前"，用 RL 训练的 Allocator 预测 query-aware 的每帧分辨率预算，解决了 Baseline 的三大缺陷：

  **(a) 缺陷1：编码后的压缩无法恢复已丢失的细粒度证据** → 输入侧自适应分配
  Model-side 方法（ToMe/VisionZip）在 32 帧全分辨率编码后才剪枝 token，此时高分辨率编码的计算成本已全部付出，且被剪枝的细粒度证据无法恢复。ResAdapt 的 Allocator 在编码前接收粗粒度特征 + query，预测每帧缩放因子 st ∈ [0.2, 1.8]，只让 backbone 处理 resize 后的像素。在 ~10% retention 时，ResAdapt 在 VideoMMMU 上达 45.7，显著优于 ToMe 的 39.2 和 VisionZip 的 39.1。关键设计：
  - Beta 分布参数化保证连续动作空间：s_t 从 Beta(α_t, β_t) 采样后线性映射到 [s_min, s_max]
  - 分配策略完全兼容 FlashAttention、vLLM、SGLang，无需定制 kernel

  **(b) 缺陷2：Naive accuracy-cost penalty 导致策略崩溃** → CAPO 非对称 reward shaping
  若直接使用 Lagrangian R = Q(x,y) − λC(s)，策略会无条件向最小预算崩溃——任何成本降低都获得等量奖励，无论答案质量。CAPO 通过三项机制克服：
  - Dynamic cost pivot：τ_dyn = κ_mix·c̄_group + (1−κ_mix)·τ_fix，同时提供局部比较基线和全局压缩锚点
  - Asymmetric shaping：正确+低成本 → 中等奖励 λ_+；错误+高成本 → 强惩罚 λ_−（λ_− > λ_+ > 0）。这种不对称性是防止崩溃的核心——降低成本的激励只在保持正确性时才有效
  - 对正确 rollout 施加正下限 ε_+ > 0，确保正确低成本的 rollout 始终获得正向学习信号

  **(c) 缺陷3：相邻冗余帧上的均匀分配浪费预算** → Temporal Similarity Regularizer
  没有 L_sim 时，Allocator 对视觉相似的相邻帧分配近乎相同的 scale。L_sim 通过余弦相似度门控权重 w_t = σ((cos(f_t, f_{t+1}) − τ_sim)/γ_sim) 激活，惩罚冗余联合高预算分配，迫使策略区分视觉相似的相邻帧。消融实验（Figure 7）显示：去除 L_sim 后 scale trace 坍缩为接近 FixedScale 的常数分布；恢复 L_sim 后恢复锐利的帧级分化。

  对比 baseline 的全栈执行例子（ResAdapt, Qwen2.5-VL-7B, T=32 frames, ρ≈11%）：
  - 算法层：视频 T=32 frames → SmolVLM 轻量编码器提取粗粒度特征 [T, 1024] → Transformer decoder（时序self-attn + gated cross-attn to query）→ Beta 头预测 (α_t, β_t) → 采样 s_t ∈ [0.2, 1.8] → bilinear resize 每帧 → backbone 接收约 3604 visual tokens（vs 32768）→ 单次自回归生成。CAPO 训练中，GRPO 循环：M=16 allocations × N=1 rollout → CAPO advantages → 交替更新 Allocator（PPO clip + L_sim + L_con）和 Backbone（token-level PPO）。
  - 系统框架层：VeRL + DeepSpeed ZeRO + vLLM 分布式训练（32×H100）。推理时单 GPU Allocator + 4-GPU vLLM engine。在 128 frames, R≈28% 时，E2E 延迟从 4877ms 降至 1977ms（−59.5%）。
  - 编译框架层：论文未明确说明。ResAdapt 保持 backbone 的 native token 接口，无需编译框架修改。
  - kernel调度层：完全兼容 FlashAttention 和 vLLM PagedAttention，无自定义 kernel。与需要处理不规则 token 布局的 model-side 方法形成对比。
  - 硬件架构层：NVIDIA H100 GPU（训练 32×H100）。推理延迟节省在长序列下最显著——128 frames + ~28% retention 时 attention FLOPs 降低为 ρ² ≈ 0.012（~83×），叠加 Allocator 固定开销后仍有净加速。空间预算节省还可 reinvest 为时间覆盖：在同计算量下可处理 16× 更多帧，长视频推理相对增益 >15%。

- baseline方法是什么？
  Baseline是传统frame-centric密集视觉编码器（以SigLIP2为代表），其核心假设是：视频中的所有空间patch和时间帧同等重要，因此需要均匀密集处理。

  Baseline（以SigLIP2 ViT-L/16 dense帧采样为例）全栈执行例子：
  - 算法层：视频→均匀8帧采样（约sparse temporal sampling）→每帧dense patchify（16×16 patches, 256 patches/frame）→ViT编码全部256×8=2048个patches→attentive pooling聚合→class embeddings。所有空间区域（前景/背景、运动/静止）以相同计算量处理。问题：大量计算（>75%）浪费在静态背景和不变区域，且8帧稀疏采样可能完全错过关键瞬时动作（如短暂倾倒、击球瞬间）。
  - 系统框架层：PyTorch + HuggingFace Transformers + Flash Attention。无专用Serving框架修改。
  - 编译框架层：论文未明确说明。
  - kernel调度层：Flash Attention 2标准实现。
  - 硬件架构层：NVIDIA A800/H100 GPU集群。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  OV-Encoder通过**Codec Patchification**将编解码器的信息论分解原理引入ViT设计，根本性地将"密集均匀计算"改为"只编码信息熵高的patch"，解决了baseline的三大缺陷：

  **(a) 缺陷1：密集计算浪费** → Codec Patchification
  传统方法对所有空间patch平均分配计算，背景静态区域和运动前景区域同等处理。OV-Encoder利用HEVC的运动矢量（motion vectors）和预测残差（residuals）作为patch级信息熵的代理信号，仅在64帧密集输入中选择3.1%-25%的显著patch（motion rich + high residual），剩余87.5%-96.9%的patch完全跳过。关键设计：
  - I/P-frame分解：每GOP（32帧）保留1个I-frame全量编码（建立完整空间上下文）+ 31个P-frame稀疏编码（仅运动显著区域）
  - Clip-level全局Top-K选择：不按帧独立选patch，而是在整个64帧clip中全局排序显著性，确保token budget最优分配到真正需要的时间位置
  - 可视patch indices机制：被跳过的patches通过visible_indices标记其时空位置用于3D-RoPE，保留temporal coverage

  **(b) 缺陷2：均匀采样导致关键帧丢失** → 密集时间覆盖
  传统方法均匀采样8帧（限制于token budget）意味着采样间隔内发生的快速/短暂动作可能完全丢失。OV-Encoder保留全部64帧，但P-frames仅取显著区域，实现"时间密集、空间稀疏"——例如Diving场景中，64帧覆盖连续的pose transitions，而8帧均匀采样可能跳过来回翻滚的过渡姿态。Case study 1和2论证了连续运动场景和离散关键帧场景下codec采样的优势。

  **(c) 缺陷3：缺乏语义结构化** → 百万级聚类判别
  传统contrastive learning（CLIP/SigLIP）用instance-level discrimination + text supervision，无法建模intra-class consistency和fine-grained inter-class relationship。OV-Encoder用frozen metaCLIP提取嵌入，k-means聚类为2M图像类中心和400K视频类中心，用multi-label sigmoid BCE监督，同时建模物体级（object-level）和动作级（motion-level）语义，无需外部语言监督。

  对比baseline的全栈执行例子（OV-Encoder Codec, 64 frame input, budget=2048）：
  - 算法层：原始视频→HEVC解码提取motion vectors + residuals→按patch聚合为saliency score→全局Top-K选择2048个显著patches（512来自2个I-frame所有patches + 1536来自62个P-frame最显著patches）→3D-RoPE编码时空位置→24层ViT编码（Flash Attention 2）→attentive pooling聚合→image branch对比2M object centroids / video branch对比400K motion centroids→sigmoid BCE loss。保留密集64帧时间覆盖的同时token减少87.5%。
  - 系统框架层：PyTorch + Flash Attention 2。128×A800 GPUs（16 nodes × 8）分布式训练。无Serving框架修改。
  - 编译框架层：论文未明确说明。
  - kernel调度层：Flash Attention 2标准实现，无自定义kernel。Codec处理（motion vector extract + residual decode）在CPU上进行。
  - 硬件架构层：NVIDIA A800 128 GPU预训练（Stage 1: 13B samples; Stage 2: 4B samples）。Attention probing on 8×A800。

## ReVisionLLM__Recursive_Vision-Language_Model_for_Temporal_Grounding_in_Hour-Long_Videos

- baseline方法是什么？
  Baseline是**VTimeLLM**（一个标准的非递归VLM用于时序定位），结合**CONE的CLIP相似度排序方法**。VTimeLLM使用均匀帧采样（如从小时级视频中均匀采样100帧）提取CLIP特征后送入LLM预测事件边界，然后使用CLIP相似度（平均池化帧特征与文本特征的dot product）对所有候选段进行排序，选取Top-K预测。

  Baseline（VTimeLLM + CONE, MAD dataset, 约110分钟视频）全栈执行例子：
  - 算法层：输入小时级视频（约110分钟）→ 均匀采样100帧（丢失大量时序细节，尤其在moment-to-video比极低如4.1s/110min的场景下）→ Frozen CLIP ViT-L/14提取每帧CLS token (100×768) → 线性投影到LLM嵌入空间 (100×4096) → Vicuna-7B LLM预测事件边界 "From s to e" → 将视频分割为段，对每段重复预测 → CLIP相似度排序（每段mean pooled frame CLS dot product with text CLS）→ 选Top-K。问题：(1) 均匀100帧采样导致严重时序信息丢失（Table 2中R1@.1=0.0, 所有Recall=0）；(2) CLIP相似度排序置信度校准极差——ECA@IoU=0.1高达0.6231，大量高置信度假阳性；(3) 缺少对比训练，模型只能见过正样本段，从未被训练判断"事件不存在"，在长视频中产生大量误检。
  - 系统框架层：PyTorch + HuggingFace Transformers，标准VLM推理pipeline。无专用Serving框架修改。
  - 编译框架层：论文未明确说明。
  - kernel调度层：标准PyTorch操作，无自定义kernel。
  - 硬件架构层：8×NVIDIA A100 GPUs用于训练。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  ReVisionLLM通过**递归层次化视觉感知**将时序定位从"一次性全局预测"改为"由粗到细的递归聚焦"，解决了Baseline的三大缺陷：

  **(a) 缺陷1：均匀帧采样导致时序信息丢失（R1@.1=0.0）** → 递归层次化处理
  Baseline从110分钟视频均匀采样100帧，等价于每次采样间隔约66秒，4.1秒的事件可能被完全跳过。ReVisionLLM使用**三层递归**：
  - Top层（hierarchy=3）：用稀疏特征（每段压缩为1个768维token）扫描全视频（~150分钟→100个段，段长125秒，步长25秒），粗定位5分钟级的感兴趣区域。输入token数从100降至100个sparse tokens（每段1个）。
  - 中层（hierarchy=2）：在上层预测区域附近聚焦（约50分钟→33个段），进一步缩小到分钟级。
  - Bottom层（hierarchy=1）：在最终选定的少数段内，使用250帧密集特征（每帧都保留），精确定位起止时间（秒级精度）。
  通过sparse-to-dense的递归搜索策略，ReVisionLLM默认仅处理57%的视频帧（vs baseline 100%），却将R1@.1从0.0提升至15.0%。

  **(b) 缺陷2：置信度校准极差（ECE=0.62）导致大量高置信度假阳性** → Contrastive Segments + LLM内部置信度
  Baseline仅用正样本训练（只见过包含目标事件的段），且使用CLIP相似度排序，导致无法有效区分真假阳性。ReVisionLLM引入：
  - **Contrastive Segments训练**：Stage 1从小时级视频中随机采样不含目标事件的高迷惑性段（同视频内不重叠于ground truth的段），训练模型输出"Not Present."或对存在性判断"Does <event> happen? Answer yes or no." → 负样本回答"No"，直接训练模型辨识视觉输入的信心。
  - **LLM熵基置信度**：推理时计算LLM生成每个词的概率分布熵，取平均熵倒数作为置信度 $R^i = 1 / \text{mean}(H_k^i)$，而非依赖CLIP的跨模态相似度。ECE从0.6231降至0.4614（Table S1）。
  累积消融（Table 2）：+Contrastive Segments: R1@.1 1.4%→4.8%, +Calibration (-CONE): R1@.1 4.8%→8.4%。

  **(c) 缺陷3：单层处理无法应对极低moment-to-video比和长视频扩展性** → 渐进式训练 + 层次化适配器
  Baseline尝试在训练时直接处理完整长视频会因显存和计算资源爆炸而失败。ReVisionLLM的渐进式训练：
  - Stage 1：仅用短片段（~125秒段）训练模型识别事件存在性和精确边界，计算开销小。
  - Stage 2：冻结Hierarchical Adapter，引入稀疏特征压缩（段级压缩比高达250:1），仅微调新LoRA模块处理长视频。
  同时，Hierarchical Adapter设计为轻量级（2层Cross-Attn + 2层Self-Attn vs CLIP 24层），几乎不增加额外计算开销。Ablation on Video Length（Figure 5）证明递归方法可将性能从2h稳定扩展到10h，非递归方法在10h完全失败。

  对比baseline的全栈执行例子（ReVisionLLM, 默认Top-to-Bottom, MAD dataset）：
  - 算法层：输入110分钟视频→ Frozen CLIP ViT-L/14提取每帧CLS token (T×768) → 滑动窗口分段（125s段, 25s步长, 每段uniform采样250帧）→ Hierarchical Adapter生成稀疏特征（Cross-Attention对齐文本 + Self-Attention压缩为1×768 per segment）和密集特征（Linear Projection 768→4096）→ **Hierarchy 3**: LLM接收100个稀疏token + "when can we see <event> happening?" → 粗粒度预测 → **Hierarchy 2**: LLM接收33个稀疏token（聚焦区域）→ 中等粒度预测 → **Hierarchy 1**: LLM接收250个密集token（选定段）→ 精确边界秒级输出 "From 4562 to 4577" → 计算LLM输出熵的置信度排序 → Top-K最终预测。Epochs: 5(MAD)/1(VidChapters-7M) for Stage 1, 2 for Stage 2. LoRA r=64, α=128.
  - 系统框架层：PyTorch + HuggingFace Transformers, 8×A100 GPUs, AdamW optimizer, cosine LR decay。LoRA高效微调，无需全参数更新。
  - 编译框架层：论文未明确说明。
  - kernel调度层：标准PyTorch操作（nn.Linear, nn.MultiheadAttention），无自定义CUDA kernel。
  - 硬件架构层：8×NVIDIA A100 GPUs集群。每GPU batch size=16（total 128）for Stage 1短片段训练, batch size=1（total 8）for Stage 2长视频训练。推理仅需单GPU。

## Owl-1__Omni_World_Model_for_Consistent_Long_Video_Generation

- baseline方法是什么？
  Baseline是传统的**迭代式长视频生成方法**（iterative temporal autoregressive paradigm），典型代表包括 StreamingT2V, SEINE, DynamiCrafter 等。这些方法将长视频生成分解为逐段生成短clips，每轮使用上一clips的**last frame** 作为下一轮生成的条件。

  Baseline（以DynamiCrafter迭代生成7s视频为例）全栈执行例子：
  - 算法层：输入首帧I和文本描述 → Video Diffusion Model 生成clip_0 (2s) → 取clip_0最后一帧作为image condition → 输入Video Diffusion Model生成clip_1 (2s) → 取clip_1最后一帧 → 生成clip_2 (2s) → ... → 拼接为长视频。条件仅包含最近一帧的像素信息（short-term fine-grained visual clues），缺少对整体场景风格、角色身份、背景等长期信息的记忆。问题：(1) long-term inconsistency —— 远距离clips之间风格/角色/背景漂移；(2) 内容同质化 —— 缺乏对未来动态的预测，反复生成相似内容；(3) 时序感受野受限 —— 仅依赖相邻帧的short-term信息。
  - 系统框架层：PyTorch + Video Diffusion Model inference pipeline。无专用Serving框架修改。
  - 编译框架层：论文未明确说明。
  - kernel调度层：标准扩散模型推理，无自定义kernel。
  - 硬件架构层：NVIDIA A800 GPU集群。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  Owl-1 通过构建**Omni World Model**（全向世界模型），从根本上将长视频生成从"像素级前后帧拼接"改为"隐式世界建模+显式视频拍摄"，解决了baseline的三大缺陷：

  **(a) 缺陷1：long-term inconsistency（仅用last frame导致远距离风格/角色/背景漂移）** → Comprehensive Condition from Latent State
  Baseline仅用last frame（像素级short-term信息）作为条件。Owl-1设计 latent state variable s_t，它是所有历史观测的聚合：s_{t+1} = h(s_0, o_0, ..., o_t)（Eq. 4）。state通过LMM的大感受野（causal self-attention over entire history sequence）编码完整的历史演进信息，作为下一轮生成的综合条件（Eq. 1: o_t = D(s_t, o_{t-1})），其中s_t负责长期一致性，o_{t-1} 负责短期平滑。在 VBench-Long 上，Owl-1 的 Subject Consistency (98.29) 和 Background Consistency (98.61) 在开源方法中均为最佳，验证了latent state对一致性的提升。

  **(b) 缺陷2：homogeneous content（反复生成同质内容）** → Anticipation of Future Dynamics
  Baseline忽略视频内容在长时序上的变化，导致反复生成相似内容。Owl-1显式建模世界动态 d_t（Eq. 2: d_t = f(s_t, o_t)），从当前观测和状态预测未来事件的文本描述，并将预测的动态融入状态演化 d_t → s_{t+1}（Eq. 3），驱动世界向前推进。在定性可视化（Figure 5）中，Owl-1可生成从"手部特写"到"整体修剪效果"的逻辑演进，体现了动态预测能力。

  **(c) 缺陷3：缺乏world-level的理解（仅做pixel-level condition传递）** → Closed-loop State-Observation-Dynamics Triplet
  Baseline在像素空间做条件传递，缺乏对世界的抽象理解。Owl-1构建闭环三元组（state → observation → dynamics → state），用LMM（Chameleon）的通用推理能力建模三者的关系。LMM以自回归方式处理序列 [..., s_t, o_t, d_t, ...]（Eq. 5），利用大规模预训练的常识知识理解世界演化规律。这种从"像素条件"到"世界状态条件"的范式转换，是论文的核心设计理念。

  对比baseline的全栈执行例子（Owl-1, 24s长视频, 3 scenes × 8s）：
  - 算法层：首帧I + text d_0 → SD2.1生成首帧 → LMM编码初始化 s_0 (128 learnable queries) → Video Diffusion Model (DynamiCrafter) 以s_0替代text condition生成 o_0 (8s) → LMM从前序序列 [..., s_0, o_0, d_0] 预测 d_1 并更新 s_1 → 跨场景切换时丢弃image_cond仅用s_1生成 o_1 → LMM预测 d_2 并更新 s_2 → 生成 o_2 → 拼接为24s长视频。State变量在全程保持一致的风格/角色/背景，Dynamics驱动不同scene之间的内容演进。
  - 系统框架层：PyTorch + Chameleon LMM + DynamiCrafter Video Diffusion Model。8×A800训练。
  - 编译框架层：论文未明确说明。
  - kernel调度层：标准LMM自回归推理 + 扩散模型denoising推理。无自定义kernel。
  - 硬件架构层：NVIDIA A800 GPU集群（80G）。训练：Stage 1 (1天) + Stage 2 (5天) + Stage 3 (1天)。

## Molmo2__Open_Weights_and_Data_for_Vision-Language_Models_with_Video_Understanding_and_Grounding

- baseline方法是什么？
  Baseline分为三类：(1) **闭源API模型**（GPT-5, Gemini 2.5/3 Pro, Claude Sonnet 4.5）—— 强大的视频理解能力但不公开训练数据/权重/recipe，训练中可能使用自产VLM互蒸馏。(2) **开源权重模型**（Qwen3-VL, InternVL3.5, Eagle2.5, GLM-4.1V）—— 公开权重但不公开训练数据和recipe，数据集可能由GPT-4等闭源VLM生成，存在循环依赖。(3) **全开源模型**（PLM, LLaVA-Video, VideoChat-Flash）—— 公开权重+数据，但数据严重依赖闭源VLM蒸馏（GPT-4V生成captions/QA pairs），受闭源bias污染。
  这些baseline的共性问题：(a) **视频grounding能力严重缺失**——即使是闭源API模型，视频pointing F1仅2.2-20.0，视频tracking HOTA仅~30，无法进行pixel级时空定位；(b) **训练数据依赖闭源VLM蒸馏**——形成封闭循环，全开源社区无法自主迭代优化；(c) **视频caption dataset偏短/偏粗**——现有开源数据描述长度~75-547 words/video，细节不足，无法支撑细粒度视频理解；(d) **长视频理解弱**——开源模型在10min+视频上性能急剧下降（open models long QA avg仅56.2-60.4 vs closed 66.4-80.4）；(e) **缺少多图像grounding**——图像pointing仅限单图，不支持跨多图的pointing QA。

  Baseline（以Qwen3-VL-8B为例）全栈执行例子：
  - 算法层：视频→帧均匀采样→SigLIP/ViT逐帧编码→每帧生成vision tokens→LLM处理→生成文本答案。视频pointing几乎完全不可用（Molmo2-VP F1仅1.5）。视频tracking HOTA平均~16.5（Molmo2-Track across all categories）。video counting在高count区间(25-60)准确率0.0%。
  - 系统框架层：PyTorch + HuggingFace。无专用Serving框架修改。
  - 编译框架层：论文未明确说明。
  - kernel调度层：FlashAttention标准实现。
  - 硬件架构层：Nvidia H100/A100 GPU集群。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  Molmo2通过以下设计系统性解决baseline缺陷：

  (1) **9个全新自建数据集（无需闭源VLM）** → 解决fully-open数据生态缺失。Molmo2-Cap用人工语音描述+转录+LLM润色+Molmo frame-level caption补充的pipeline，产出avg 924 words/video的dense captions（vs 基线75-547 words）。Molmo2-VideoPoint/Molmo2-VideoTrack用人工点标注+时戳记录，产出650k pointing queries和15k tracking queries。Molmo2-CapQA用自训练(基于Molmo2-Cap)的video captioner + LLM合成QA，避免闭源VLM依赖。Molmo2-AskModelAnything用人工提问+LLM辅助answer+人工refine。全流程仅使用text-only LLM辅助（非VLM），打破闭源VLM蒸馏循环。

  (2) **视频Grounding：spatio-temporal pointing + tracking + counting** → 解决视频grounding缺失。将2D image pointing (PixMo的<x,y>点格式)扩展到3D spatio-temporal域：引入timestamp + obj_id + (x,y)的压缩HTML-like格式。Pointing→counting：先point objects再count IDs（"point then count"策略，vs 直接预测count更优 Table 9a）。Tracking→point tracks with IDs：每帧标注object点+关联ID，支持HOTA等tracking指标（MOTA测量association accuracy）。相比API模型的bbox centerpoint策略（论文实验表明API模型无法生成准确point tracks），Molmo2的native point generation在video pointing F1达38.4（vs Gemini 3 Pro 20.0），video counting close accuracy 35.5（vs GPT-5 35.8），video tracking HOTA 57.5 across all categories（vs Gemini 3 Pro 29.1）。

  (3) **训练技术创新** → 解决训练效率和数据不平衡。(a) Packing + Message Trees：动态规划solver pool=48，平均3.8 examples/packed sequence，~15x训练效率；message tree允许同一video/image有多个annotations并存，用custom attention mask防止跨分支attention。(b) Token weighting：video caption weight=0.1, pointing weight=0.2, 其他 √(4/n)策略，防止少数长输出样本主导loss。(c) Bidirectional attention on vision tokens：vision tokens可互相attend（cross-frame/image），提升性能（Table 8b: 无bidir导致QA avg -0.4, Cap F1 -1.0）。(d) Pointing预训练：在pre-training阶段引入pointing数据，使SFT阶段不再需要学习basic pointing format，整体pointing更稳定。(e) SlowFast encoding：推理时用query-based frame selection + 3×3 slow/9×9 fast pooling，在~43% fewer tokens下匹配224 frame性能（Table 20）。

  对比baseline的全栈执行例子（Molmo2-8B, 128 frame video + tracking query）：
  - 算法层：视频→torchcodec 2fps抽128帧→SigLIP 2 ViT逐帧编码（384px, 27 layers）→取layer 3和layer 9 hidden states concat→Connector MH pooling 3×3 window→每帧81 visual tokens→128×81≈10,368 vision tokens。Vision tokens双向attention（可跨帧attend）→LLM 36层处理。Point output format：
    `<tracks coords="0.0 1 635 522;0.5 1 606 490;1.0 1 515 164">person in red</tracks>`
    `timestamp obj_id x y;...` 多个frames按timestamp排序，相同obj_id关联为track。64s超长视频→SFT 128帧处理；若384帧则需long-context SFT + context parallelism（8 GPUs Ulysses attention）。
  - 系统框架层：PyTorch + FSDP2 + SDPA（非FlashAttention, 因custom attention mask不兼容）+ torch.compile（静态shape）+ AMP bfloat16。HuggingFace model + vLLM serving。Multi-node 128 H100 SFT training。
  - 编译框架层：torch.compile用于LLM和ViT的静态编译优化吞吐量。论文未额外修改编译框架。
  - kernel调度层：PyTorch SDPA（Scaled Dot Product Attention），因需要custom attention mask（packing + message trees + bidir vision）无法使用FlashAttention。Context parallelism用Ulysses attention all-gather实现。
  - 硬件架构层：Nvidia H100 128节点SFT训练（8.1k GPU hours for 8B）。推理384 frames + greedy decoding on single H100。

  核心差异映射：
  | Baseline缺陷 | Molmo2解决方案 |
  |---|---|
  | 无视频grounding能力 | Spatio-temporal pointing format + native point generation（pointing F1 38.4 vs API 20.0） |
  | 训练数据依赖闭源VLM蒸馏 | 9个自建数据集，仅用text-only LLM辅助（+Molmo自用frame-captioner） |
  | 视频caption粗短 | 人工语音+transcribe+润色+frame visual detail补充，924 words/video |
  | 长视频理解弱 | Long-context SFT (384 frames, 36864 tokens) + SlowFast query-based inference |
  | 缺少多图grounding | Molmo2-MultiImagePoint (470k examples) + canonical label cross-image一致性算法 |
  | 训练效率低 | Packing (15x) + message trees + token weighting + bidir vision attention |
  | 开源模型使用闭源数据 | 全栈fully-open (权重+数据+代码)，可被community完全reproduce和extend |

## LongLive__Real-time_Interactive_Long_Video_Generation

- baseline方法是什么？
  Baseline 分为两类：(1) **扩散模型**（Wan2.1, SkyReels-V2, LTX-Video）—— 基于 DiT 架构的双向注意力（bidirectional attention）视频生成模型。双向注意力使所有帧之间的注意力关系为非因果（non-causal），导致 KV cache 机制无法使用，每生成一个视频片段必须重新计算全部帧的注意力。例如 SkyReels-V2 生成 60s 视频在单 H100 上需要约 50 分钟。(2) **自回归（AR）模型**（Self-Forcing, CausVid）—— 基于因果注意力（causal attention）的 frame-level 或 chunk-wise AR 视频生成模型。因果注意力天然支持 KV cache 加速推理，但由于训练长视频成本高，普遍采用 train-short-test-long 策略（仅在短视频上训练，在 rollout 长视频时用模型自己的输出做上下文），导致误差累积、内容漂移和一致性下降。

  Baseline（以 Self-Forcing 为例）全栈执行例子：
  - **算法层**：Wan2.1-T2V-1.3B (DiT) → 适配为 chunk-wise causal AR 模型（ODE initialization + DMD distillation）→ 训练仅 5s clips（train-short）→ 推理时通过滚动式 KV cache rollout 长视频（test-long）。Cross-attention: visual tokens Q attend to text prompt K/V。Self-attention: causal mask，生成帧的 KV 缓存。（缺陷：train-short-test-long 导致随视频变长质量下降；prompt 切换时若丢弃 KV cache 导致视觉断裂，保留 KV cache 导致 prompt 不跟随）。
  - **系统框架层**：PyTorch + HuggingFace Diffusers。无多请求 Serving 框架修改。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：标准 FlashAttention for causal self-attention。
  - **硬件架构层**：NVIDIA H100 GPU。推理时 dense causal attention 复杂度 O(L²)（L=总帧数），生成 180s 视频需处理超百万 token（以 Wan2.1 参考）。

  Baseline 的核心缺陷：
  1. **扩散模型中双向注意力禁止 KV cache**：每步推理需重算全部帧注意力，延迟随视频长度平方增长（O(L²)），导致长视频生成极慢（SkyReels-V2 ~50min/60s）。
  2. **AR 模型 train-short-test-long 不匹配**：训练仅在短视频上进行，推理时长 rollout 中模型输出不断作为自身输入，误差累积使上下文逐渐劣化，内容一致性随时间下降。
  3. **Prompt 切换时 KV cache 困境**：丢弃 KV cache → 视觉断裂、时间不连续；保留 KV cache → 旧 prompt 语义残留在 cache 中导致新 prompt 延迟响应或不跟随。
  4. **无有效的长视频高效推理策略**：dense causal attention 的 O(L²) 计算不可持续。此前 attention sink 在视频模型中被报告为无效（Self-Forcing 验证），因长期 rollout collapse 使得 sink 失去锚定作用。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  LongLive 通过三个核心设计系统性地解决上述缺陷：

  (1) **KV-recache** → 解决 Prompt 切换困境。在 prompt 切换边界，重新用已生成视频前缀当视觉上下文配对新 prompt 计算 KV cache（单次 forward pass 过交叉注意力层），清除旧 prompt 残留语义（cross-attention 中旧 prompt embedding 被替换）但保留自注意力中的运动与视觉连续性信号。训练时同步集成 recache（teacher 也接收新 prompt 做 DMD 监督），消除 train-inference mismatch。多 switch 泛化：推理时 n+1 个 prompt 有 n 个 switch 边界，每个边界执行一次 recache 即可。

  (2) **Streaming Long Tuning** → 解决 train-short-test-long 不匹配。每次 iteration 基于上一 iteration 的 KV cache 滚动生成下一个 5s clip（而非重新采样），仅对当前 clip 计算 DMD loss（teacher=Wan2.1-T2V-14B 对每个 5s clip 独立监督，确保 teacher 在自身能力范围内），gradient 只流经当前 clip（detach 历史帧梯度）。这直接将模型暴露于长 rollout 中自己生成的退化帧，训练即推理条件，使模型学会在长序列中自我纠错、抑制误差累积。同时显存仅按 clip 时长控制（O(clip) 而非 O(full_video)），避免 naive long tuning 的 OOM。

  (3) **Short Window Attention + Frame Sink** → 解决长视频推理效率。推理时注意力仅作用于最近 W 帧（如 W=9 latent frames）+ 永久保留的首帧 chunk（S=3 sink tokens），注意力复杂度从 O(L²) 降至 O(W+S+T)。Frame sink 仅在 streaming long tuning 解决了长期 rollout collapse 后才生效——作为全局语义锚点，将场景身份、色调、风格等持久信息缓存于 sink token 中。Train-test 对齐：在 streaming long tuning 中同样使用 short window + frame sink，resident KV size = O(W+T+S) 不随视频长度增长，避免 OOM。

  对比 baseline 的全栈执行例子（LongLive, 832×480, 1.3B, single H100）：

  - **算法层**：Wan2.1-T2V-1.3B (DiT) → chunk-wise causal AR 适配（ODE init + DMD, short window W=9, frame sink S=3）→ Streaming Long Tuning (60s, LoRA rank=256, 350M trainable)：每次 rollout 5s clip → [KV cache] extend next 5s → DMD loss only on current clip → repeat 至 60s → prompt switch: KV-recache (对已生成前缀用新 prompt 重算 KV) → 继续 rollout。（训练 3000 iters, 64 GPU × 12h = 32 GPU-days）
  - **系统框架层**：PyTorch + DMD pipeline。无多请求 Serving 框架修改。LoRA 微调 27% 参数。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：标准 FlashAttention with causal mask。Short window attention 通过 attention mask 实现（仅 mask 掉 window 外的 KV 位置），与 FlashAttention 兼容。Sink tokens 实现为 KV 序列的前缀拼接。
  - **硬件架构层**：NVIDIA H100 GPU。推理吞吐 20.7 FPS（vs Self-Forcing 17.0 / chunk-wise，8.9 / frame-wise）。Short window + sink 降低端到端计算时间 28%，峰值显存 17%（on H100）。支持 240s 视频生成 on single H100。INT8 PTQ: 2.7GB → 1.4GB (1.9×), 5090 GPU 上 16.4 FPS。

  LongLive 与 baseline 的核心差异：
  | 维度 | Baseline (Self-Forcing et al.) | LongLive |
  |------|-------------------------------|----------|
  | 训练策略 | train-short-test-long（5s） | train-long-test-long（streaming 60s/240s） |
  | Prompt 切换 | KV cache 全弃或全留 | KV-recache（单次重算，清除旧语义保留视觉） |
  | 长视频推理 | dense causal attn O(L²) | short window + frame sink O(W+T+S) |
  | Frame sink | 无效（因长期 collapse） | 有效（streaming tuning 解决 collapse） |
  | Train-test 对齐 | 不匹配（仅短期监督） | 完全对齐（training 模拟 rollout + same window） |

## LLaVA-Mini__Efficient_Image_and_Video_Large_Multimodal_Models_with_One_Vision_Token

- baseline方法是什么？
  Baseline 方法是 LLaVA-v1.5 (Liu et al., 2023b)，标准 LMM 架构：CLIP ViT-L/336px vision encoder → Projection → Vicuna-7B LLM backbone。单张图像被编码为 576 个 vision token（24×24 patches），和 text token 一起输入 LLM 的 32 层 Transformer 做逐层自注意力，最终自回归生成回复。

  Baseline（LLaVA-v1.5, 336px）全栈执行例子：
  - 算法层：图像 → CLIP ViT-L/14 (patch size 14) → 24×24=576 vision tokens → Linear Projection → [576, 4096]；文本 → Vicuna-7B embedding → [l_q, 4096] → Concat → [576+l_q, 4096] → Vicuna-7B 32-layer causal self-attention → next-token generation。每张图像 576 个 vision token 全部参与 LLM 逐层计算，FLOPs 8.55T，延迟 A100 约 113ms。
  - 系统框架层：PyTorch + HuggingFace Transformers。无 Serving 框架修改。
  - 编译框架层：论文未明确说明。
  - kernel调度层：标准 FlashAttention。
  - 硬件架构层：NVIDIA A100/A800 GPU。
  - 视频场景：8 秒 1fps → 4608 vision tokens (576×8)，VRAM 随帧数线性增长，24GB GPU 处理上限约 100 帧。

  Baseline 的缺陷：
  1. **Vision token 数量过多**：每张图 576 个 vision token 全部输入 LLM，导致 FLOPs 巨大（8.55T），延迟高（>100ms），难以实现实时交互。
  2. **高分辨率扩展困难**：高分辨率需要更多 token（如 LLaVA-v1.5-672px 需要 4 倍 token = 2304 个），FLOPs 急剧增加到 40.49T。
  3. **长视频不可行**：1fps 抽取下每张图 576 token，8 秒视频需 4608 token，VRAM 消耗大，无法处理超长视频。
  4. **Vision token 在深层冗余**：论文分析发现 vision token 主要在 LLM 前几层被 text token 用来"融合"视觉信息，深层中 vision token 被关注的注意力急剧下降（80%+ 注意力转向 instruction token），后层移除 vision token 对性能影响很小。因此深层中大量 vision token 是浪费的。
  5. **直接 token 合并损害性能**：先前方法（PruMerge, MQT-LLaVA, VoCo-LLaMA 等）在 vision encoder 输出后直接合并 token，因视觉信息未预先融入 text token 而导致 5% 平均性能下降。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法：LLaVA-Mini 通过 insight-driven 的设计将 vision token 压缩与视觉-文本融合解耦到 LLM 之前执行：

  (1) **Query-based Compression** → 解决 vision token 数量过多。引入可学习压缩 query Q^v 通过 cross-attention 与全部 vision token 交互，使用 2D sinusoidal PE 保留空间信息，输出 C^2 个压缩 vision token（C=1 即 1 token）。相比 average pooling，query-based compression 可自适应关注关键视觉区域（如 OCR 中的文字、价格等），额外仅增加 2.42G FLOPs。

  (2) **Modality Pre-fusion** → 解决 visual information loss during compression。在 LLM 之前放置 N_fusion=4 个与 LLM 同构的 Transformer decoder 块，将全部 vision token 和 text token 拼接后通过 pre-fusion 模块，使 text token 提前吸收融合视觉信息。这模拟了 LLM 早期层中 text token attend vision token 的过程，但将其移到了 LLM 之外。即使之后 vision token 被极端压缩（甚至到 1 个），融合后的 text token 已携带所需视觉信息。

  (3) **模块放置在 LLM 外部** → 解决兼容性与压缩质量。a) 压缩放在 LLM 外部可避免 LLM 内部层赋予 vision token 上下文信息导致压缩模块难以区分 token；b) 保持 LLM backbone 不变，兼容几乎所有 LLM 加速框架。

  对比 baseline 的全栈执行例子（LLaVA-Mini, 336px, C=1）：
  - 算法层：图像 → CLIP ViT-L/14 → 576 vision tokens [576, 4096] → 同时走两条路径 — (a) Compression: learnable queries [1,4096] cross-attend 576 vision tokens → Ĥ^v [1,4096]；(b) Pre-fusion: 576 vision tokens + l_q text tokens 拼接 → 4-layer Transformer decoder → Ĥ^q [l_q, 4096] → Concat([1,4096], [l_q,4096]) → Vicuna-7B 32-layer → response。LLM 仅需对 1+l_q 个 token 做 self-attention（而非 576+l_q）。FLOPs 1.96T (下降 77%)，延迟 A100 38.64ms (加速 2.9×)，VRAM per image 从 360MB 降至 0.6MB。
  - 系统框架层：PyTorch + HuggingFace Transformers。8×A800 训练，A100/RTX 3090/A800 推理。
  - 编译框架层：论文未明确说明。
  - kernel调度层：标准 FlashAttention，pre-fusion 模块也用标准 Transformer — 与 LLM backbone 同构。
  - 硬件架构层：NVIDIA A800 (训练), A100/RTX 3090/A800 (推理)。RTX 3090 24GB 可处理 >10000 帧视频（~3 小时），而 LLaVA-v1.5 在同一硬件上仅能处理 ~100 帧。
  - 视频场景：1fps 抽取 M 帧 → M×1 vision token + l_q 融合 text token（M 帧 fusion token 经 pooling 聚合），远超 LLaVA-v1.5 的 M×576 token 规模。训练时只用 <60 帧视频，推理时可外推至 7200+ 帧（2 小时）且性能良好（MLVU 42.8, Video-LLaVA 36.4）。

- baseline方法是什么？
  Baseline 方法是标准 CLIP 预训练 + 微调范式：Vision Encoder (ViT) + Text Encoder（轻量自回归模型，约 1/3 ViT 参数量），通过对比损失在数亿到数百亿 image-text pairs 上训练，将图像和文本嵌入共享表示空间。Text encoder 上下文窗口限制为 77 tokens，对长/复杂 caption 理解能力不足。

  Baseline（以 SigLIP2-SO/14, 224px, 原始 CLIP text encoder 为例）全栈执行例子：
  - 算法层：图像 → ViT (SO/14, 428M) → visual embedding [d=1152]；文本 → CLIP text encoder (autoregressive, ~1/3 ViT params, 77-token limit) → text embedding [d=1152] → L2 normalize → cosine similarity。预训练于 ~40B image-text pairs。在 Flickr30K 短文本 I2T 93.9/T2I 82.9，ShareGPT4V 长文本 I2T 90.2/T2I 87.2。
  - 系统框架层：PyTorch 分布式训练，大规模 batch size 训练（如 SigLIP 使用 sigmoid loss 替代 softmax 以支持更大 batch）。
  - 编译框架层：论文未明确说明。
  - kernel调度层：标准 FlashAttention-2。
  - 硬件架构层：NVIDIA A100 GPU 集群。

  Baseline 的缺陷：
  1. **文本编码器能力弱**：CLIP text encoder 是轻量自回归模型（~100M 参数级），其语言理解和世界知识远不如现代 LLM（8B 参数级）。对长/复杂 captions、多语言文本、细粒度空间关系/对象描述的语义抽取能力严重不足。
  2. **上下文窗口限制**：原始 CLIP text encoder 仅支持 77 tokens 输入，对 dense captions 必须截断或使用变通方法（summarization/segmentation/positional encoding fine-tuning），信息丢失严重。
  3. **LLM 嵌入不可直接使用**：直接将 LLM 嵌入注入 CLIP 训练会导致性能退化——原始 LLM 嵌入对 image captions 的可分离性极差（Llama3-8B 在 COCO caption-to-caption retrieval Top-1 仅 5.2%，而 CLIP text encoder 为 25.2%），无法为对比学习提供有效监督。
  4. **训练成本高**：CLIP 预训练本就昂贵，naively 联合微调 LLM 会进一步推高成本。直接 fine-tune CLIP 对短文本提升微弱（Directly Finetune 仅从 74.4/72.0 提升到 74.5/72.3），说明单纯增加训练数据无法有效注入 LLM 能力。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法：LLM2CLIP 通过两阶段高效微调框架将 LLM 能力注入预训练 CLIP：

  (1) **Stage 1: Caption Contrastive (CC) Fine-tuning** → 解决 LLM 嵌入可分离性差。对 LLM 进行"embedding化"改造使其适合 CLIP 场景：(a) 移除 causal mask → 启用双向注意力；(b) 使用 average pooling 代替 [EOS] token 获得句子嵌入；(c) LoRA 参数高效微调激活文本理解能力；(d) 监督 SimCSE 对比损失 —— 同一图像的两个 caption 作为正样本对，大幅提升嵌入对 caption 语义的区分能力。LLM 从不可用（Top-1 5.2%）提升到超越 CLIP text encoder（Top-1 29.5%）。

  (2) **Stage 2: Adapter + Vision Encoder Fine-tuning** → 解决训练成本与架构融合。将 CC fine-tuned LLM 作为文本编码器完全替换原始 CLIP text encoder，冻结 LLM 梯度，在其输出后附加 4 层 Linear Adaptor（inverted bottleneck MLP, 67.1M params）作为可学习桥梁，与 CLIP Vision Encoder 进行跨模态对比学习。关键优势：(a) LLM 梯度冻结 —— 完全不更新 LLM，GPU 显存消耗大幅降低；(b) Offline-loading —— 预计算所有 caption 的 LLM 嵌入存盘，训练时直接加载，将 LLM 推理开销从多 epoch 降至单次 pass，batch size 可从 704 增至 16384；(c) 训练时间从 17h (LLM LoRA) 降至 1.3h (Frozen + Offline-loading)，同时性能更高（83.9/82.1 → 85.9/83.3）。

  (3) **LLM 的开放世界知识注入** → 解决长文本和多语言理解不足。LLM 训练于海量文本语料，拥有开放世界知识，能理解 dense captions 中的空间关系、对象间关系、细粒度描述。即使 LLM2CLIP 仅用英语数据训练 15M samples，其在 XM3600 的 36 语言检索上仍超越用 12B alt-texts（含 109 语言）训练的 SigLIP2 text encoder。

  对比 baseline 的全栈执行例子（LLM2CLIP + SigLIP2-SO/14, 224px, 60M data）：
  - 算法层：图像 → ViT (SO/14, 428M, 梯度全开) → visual embedding [d=1280]；文本 → Llama 3.1 8B（双向注意力, avg pooling, 梯度冻结）→ sentence embedding [d=4096] → 4-layer Linear Adaptor (FuseMix MLP, 梯度全开, 67.1M params) → text embedding [d=1280] → L2 normalize → cosine similarity。训练时 LLM 不加载到 GPU（offline precomputed embeddings），batch size 4096（offline-loading 可达 16384）。对比原始 SigLIP2：短文本 +1.0/+1.9 (I2T/T2I)，长文本 +14.8/+15.8，多语言 +11.9/+15.2。推理时 LLM 需加载一次计算文本嵌入。
  - 系统框架层：Stage 1 使用 32 A100，LoRA fine-tuning LLM 1 epoch。Stage 2 使用 2×8 A100 40GB，ViT 全梯度 + Adaptor 训练 4 epochs。Offline-loading 策略将文本预计算与视觉训练解耦。数据配比：50% 真实 short caption + 50% MLLM-generated dense caption。
  - 编译框架层：论文未明确说明。
  - kernel调度层：FlashAttention-2 用于 LLM 和 ViT 的注意力计算。bfloat16 混合精度训练。
  - 硬件架构层：NVIDIA A100 40GB GPU 集群（Stage 1: 32卡，Stage 2: 16卡）。Offline-loading 后训练仅 1.3h，batch size 达 16384。

  核心洞察：只需百万级训练样本和与标准 CLIP fine-tuning 几乎相同的计算预算，即可将 LLM 的文本理解能力注入预训练 CLIP，显著提升跨模态表示质量。CC fine-tuning 是使 LLM 嵌入可用于 CLIP 的关键前提——跳过此步骤的 LLM 嵌入反而会损害原始 CLIP 性能。

## Growing_a_Twig_to_Accelerate_Large_Vision-Language_Models

- baseline方法是什么？
  Baseline方法是基于attention map的视觉token剪枝方法（如FastV、SparseVLM、VisionZip），通过在VLM早期层利用attention scores选择并保留重要visual tokens来加速prefilling阶段，但存在两个主要缺陷：(1) 早期层attention信号对任务不敏感，导致剪枝后精度大幅下降；(2) KV-cache机制和FFN block使decode阶段加速有限，长response生成时速度瓶颈严重。

  Baseline（以FastV on LLaVA-1.5-7B，K=2, R̄=64为例）全栈执行例子：
  - 算法层：Image → CLIP Vision Encoder → 576 visual tokens → 拼接text tokens → LLaVA-1.5-7B 前2层处理 → 取第2层attention map，计算text-to-visual attention scores → 选择top-R(=41)最重要的visual tokens → 丢弃其余 → 剩余30层处理pruned序列 → KV-cache缓存 → decode阶段逐token自回归生成。88.9% pruning ratio下RelAcc仅77.0%（FastV）。
  - 系统框架层：HuggingFace Transformers推理。剪枝仅在prefilling阶段生效，decode阶段FFN计算量不变，KV-cache使SA加速有限。
  - 编译框架层：论文未明确说明
  - kernel调度层：标准FlashAttention（causal mask），decode阶段每步只处理1个token，GPU利用率极低
  - 硬件架构层：8×NVIDIA A100 GPU服务器

  Baseline的缺陷：
  1. **早期attention信号质量差**：第2层attention对multimodal语义理解不充分，选出的visual tokens与prompt无关（论文Fig.2可视化：D=2选出的token在不同prompt间几乎相同），大量有用visual信息被错误丢弃。更深层attention(D=18)虽能提供更精准信号，但若在深层pruning则前面的计算冗余已产生。
  2. **Decode阶段加速有限**：KV-cache机制下SA block加速效果被削弱，FFN block完全无加速。当response length≥32 tokens时，prefilling时间可忽略，但decode时间线性增长（论文Fig.3）。FastV仅在prefilling阶段加速，长response(MM-Vet, S̄≈100)下RelSpd仅~104%。
  3. **剪枝信号未专门优化**：attention map的token选择能力仅作为next-token prediction训练的副产品出现，未针对剪枝任务直接优化。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法：TwigVLM/TwigVLM++通过三项核心设计解决上述缺陷：

  (1) **Twig-guided Token Pruning (TTP)** → 解决早期attention信号差。在base VLM第K层后附加T层twig block（初始化为VLM第K+1至K+T层权重），使用twig最后一层的attention map指导token剪枝。由于twig最后一层靠近prediction head（距离loss函数更近），其attention对multimodal关系理解更精准（Fig.2证实attention quality随depth增加而提升）。同时twig仅需轻量post-training（~10% base VLM训练时间），不修改base VLM权重。配合FinalWipe策略在Kf层后移除所有visual tokens（因深层visual tokens贡献极小），在固定R̄下允许更大的R，进一步提升精度。

  (2) **Self-Speculative Decoding (SSD)** → 解决decode阶段加速不足。利用TwigVLM天然的一体两用架构：浅层子网络Ms（前K层+twig）作draft model、深层子网络Mb（完整base VLM）作target model。Draft自回归生成5个候选tokens → target并行验证并接受匹配tokens。关键优势：(a) draft和target共享前K层KV-cache，减少冗余计算；(b) draft model极浅(K+T≪L)，生成开销低；(c) 并行验证充分利用GPU并行能力。长response场景(MM-Vet)RelSpd达154%（vs FastV 104%）。TwigVLM++的Tree-based SSD进一步通过构建token tree（E=10, K=10, D=4）增加每次验证的候选路径覆盖，RelSpd达~197%。

  (3) **Multi-head Twig + RL-based Pruning Optimization (TwigVLM++)** → 解决剪枝信号未专门优化。解耦D-Head（next-token prediction）和P-Head（专用于token重要性评分）。Stage-1通过PredKL蒸馏（teacher=base VLM, student=shallow VLM）和AttnKL蒸馏（teacher=deep layer attention, student=P-Head score）提供额外监督。Stage-2用GRPO式RL直接最大化pruned输入下的参考答案log-probability：P-Head产生token重要性分布 → 无放回采样R个位置得action → reward = pruned输入生成参考答案的mean log-prob → group-level advantage归一化 → 纯on-policy policy gradient更新。Dynamic pruning ratio schedule使单个模型支持多种pruning ratio。88.9% pruning下LLaVA-1.5-7B RelAcc从96.0%提升到97.7%。

  对比baseline的全栈执行例子（TwigVLM++ on LLaVA-1.5-7B, K=2, T=3, Kf=24, R̄=64）：
  - 算法层：Image → CLIP Vision Encoder → 576 visual tokens → base VLM前2层 → twig block 3层（D-Head预测next token + P-Head计算token importance score s via gated attention Eq.7）→ TTP按s选择top R=41 visual tokens → FinalWipe在24层移除所有visual tokens → 剩余深层的base VLM处理pruned序列 → 同时twig作为draft用tree-based方式构建token tree → base VLM用tree attention并行验证多条路径 → 从根遍历接受匹配tokens → 追加bonus token → 迭代至生成EOS。
  - 系统框架层：两阶段训练：Stage-1用LLaVA-665K (665K samples) + L_NTP + α·L_PredKL + γ·L_AttnKL，仅更新twig block (~10 GPU hours)；Stage-2用50K SFT samples + GRPO式RL，仅更新P-Head参数。推理时draft/target共享前K层KV-cache，tree attention使用topology-aware causal mask。
  - 编译框架层：论文未明确说明。
  - kernel调度层：FlashAttention v2.3.2。Tree-based SSD的tree attention用topology-aware causal mask替代标准causal mask，在单次前向中处理整个token tree（最多60个candidate nodes）。标准SSD每step验证1条sequence、接受~3 tokens；tree-based SSD每step验证K·E^D条路径、接受更多tokens。
  - 硬件架构层：8×NVIDIA A100 GPU服务器。Prefilling阶段TTP剪枝88.9% visual tokens，compute减少显著；Decoding阶段tree-based SSD每step处理batch of tree nodes（而非单token），GPU利用率大幅提升。长response (MM-Vet) RelSpd达~197%，短response (TextVQA) RelSpd达~139%。

## EasyAnimate__A_High-Performance_Long_Video_Generation_Method_based_on_Transformer_Architecture

- baseline方法是什么？
  Baseline 方法是基于 Diffusion Transformer 的 3D Full Attention 视频生成模型（如 CogVideoX、HunyuanVideo、OpenSora），使用 CLIP + T5 双文本编码器，DDPM 采样，以及按固定分辨率和帧数训练的 naive training strategy。

  Baseline（以典型 3D Full Attention DiT + CLIP/T5 为例）全栈执行例子：
  - 算法层：文本 prompt → CLIP 编码（限制 77 tokens）+ T5 编码 → 拼接视频 tokens → 48 层 DiT Attention（每层对所有 F×H×W tokens 做 3D full attention，计算复杂度 O((F·H·W)^2)）→ 多步 DDPM denoising → 3D VAE decode → 视频帧序列。生成 1024×1024×49 frames时，单卡 A100 需约 30 分钟。
  - 系统框架层：PyTorch 分布式训练（DDP/FSDP），naive training 下不同分辨率/帧数样本 token 数差异大，导致 GPU 利用率不均（部分 GPU 闲置等待）。
  - 编译框架层：论文未明确说明
  - kernel 调度层：FlashAttention 用于标准 3D attention，但无针对视频注意力的特殊优化
  - 硬件架构层：NVIDIA A100 GPU 集群

  Baseline 的缺陷：
  1. **3D Full Attention 计算复杂度随序列长度二次增长**：对于高分辨率长视频（1024×1024×49 frames），F×H×W tokens 产生的序列长度极大，full attention 的 O(N^2) 计算和 O(N^2) 显存需求使训练/推理成本极高。Spatial-temporal decoupled attention 虽降低复杂度但显著损害生成质量（受限于 3D 感受野）。
  2. **Naive training 导致 GPU 利用率不均**：不同分辨率和帧数的视频 token 数不同，在同一 batch 中导致不同 GPU 处理不同数据量，部分 GPU 提前完成后空闲等待，训练吞吐量低。
  3. **CLIP/T5 文本编码器能力有限**：CLIP 限制输入 77 tokens，T5 对复杂场景和细粒度文本理解不足，导致文本-视频语义对齐差。
  4. **生成视频与人类偏好偏差**：大规模 web 数据训练的扩散模型在美学质量、文本遵循度上不足，现有 reward 相关方法仅用于 U-Net + DDPM 架构，在 DiT + rectified flow 架构上未探索。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法：EasyAnimate 通过四项设计分别解决上述缺陷：

  (1) **Hybrid Windows Attention** → 解决 O(N^2) 复杂度问题。提出 6 方向滑动窗口注意力（fhw/fwh/hfw/hwf/wfh/whf），将注意力头分为 6 组，每组沿不同 3D 方向重排 token 序列后执行滑动窗口注意力，仅需一次 FlashAttention 调用。然后将 window attention 层与 full attention 层交替排布（window attention 放在中间层 12-36），在保留全局上下文的条件下大幅降低计算量。1024 分辨率下训练加速 22.39%，推理加速 25.53%。

  (2) **Training with Token Length** → 解决 GPU 利用率不均问题。将相似 token 数的视频分组到同一训练 step，如 512^2×49 frames 与 768^2×21 frames token 数相近，同组训练。每次迭代训练的 token 数从 6.17M 提升到 13.63M（+120.91%）。

  (3) **Qwen2-VL-7B 文本编码器** → 解决文本理解不足。使用 MLLM 替代 CLIP/T5，支持多语言和长文本输入，通过 RMSNorm + FC 层处理文本特征以对齐视频特征 L2 norm。VBench Total Score 从 80.42% 提升到 81.57%。

  (4) **Reward Backpropagation + Rectified Flow** → 解决人类偏好偏差。使用 HPSv2.1 + MPS 可微分 reward model 组合，通过 LoRA 微调 DiT。关键适配：K=10（因 rectified flow 下梯度 norm 比 DDPM 小，仅优化最后一步不稳定）、F=1（因果 VAE 的首帧解码能力足够，多帧导致 dynamics 损失和 reward hacking）。VBench Total Score 从 81.57% 提升到 83.42%。

  对比 baseline 的全栈执行例子（EasyAnimate + Qwen2-VL + HWA + Reward BP）：
  - 算法层：文本 prompt（多语言）→ Qwen2-VL-7B 提取倒数第二层 hidden features → RMSNorm → FC 线性变换对齐 → 拼接视频 noised latents → 48 层 MMDiT（层 1-12 用 3D full attention 建立全局上下文 → 层 12-36 用 6-direction sliding window attention 降低计算量 → 层 36-48 用 3D full attention 维持稳定性）→ Rectified Flow 快速采样（比 DDPM 少步数）→ 3D Causal VAE 逐帧因果解码（缓存前帧 latent）→ 高质量视频帧序列。后训练阶段通过 Reward BP + LoRA 微调，每步只优化最后 K=10 denoising 步骤、只计算第一帧 reward。
  - 系统框架层：PyTorch 分布式训练 + TTL 策略（按 token 数分组每步样本，均衡 GPU 负载）
  - 编译框架层：论文未明确说明
  - kernel 调度层：FlashAttention（window attention 仍兼容 FlashAttention sliding window 参数）
  - 硬件架构层：NVIDIA A100 GPU 集群，训练耗时 1024×1024 下 59.79s/iter（vs full attention 77.04s/iter），推理 21.32s/iter（vs 28.63s/iter）

## Free-MoRef__Instantly_Multiplexing_Context_Perception_Capabilities_of_Video-MLLMs_within_Single_Inference

- baseline方法是什么？
  Baseline 方法：现有的 Video-MLLM（LLaVA-Video-7B）在长视频处理中面临两种典型策略：(1) 直接扩展输入帧数 —— 将更多帧通过 Vision Encoder 编码为 token 序列，送入 LLM 的 full self-attention 推理。当帧数超过 64 时，vision token 数量暴增（每帧 182 tokens），序列长度超出 Qwen2-7B 的 32768 token 阈值，导致 OOM 或显著的性能退化（如 256 frames full attention FLOPs 为 64 frames 的 1600%）；(2) Token Compression（如 FastV、LLaMA-Vid）—— 在推理前压缩/剪枝 vision tokens，高压缩率导致信息损失；(3) Streaming Inference —— 多次调用 LLM，复用历史 KV Cache，延迟与上下文长度成正比。

  Baseline（LLaVA-Video-7B, 64 frames, full attention）全栈执行例子：
  - 算法层：视频 → FPS=1 采样最多 64 帧 → SigLIP Vision Encoder 逐帧编码为 182 tokens/frame → spatial pooling (2×2) → 64×182=11648 vision tokens → Projector 映射到 LLM embedding space → 拼接 system prompt + vision tokens + question text → Qwen2-7B 28 层 causal self-attention（每层对全部 11648 tokens 做 full FlashAttention）→ 自回归 decode → 答案。若扩展至 128 frames，序列长度 23296 tokens，full attention 的 FLOPs 为基准的 400%；256 frames 则达 46592 tokens（1600% FLOPs），单卡 A100 直接 OOM。
  - 系统框架层：基于 HuggingFace Transformers + lmms-eval 评估框架，使用 accelerate 工具包管理显存
  - 编译框架层：论文未明确说明
  - kernel 调度层：标准 FlashAttention（causal mask），无特殊 kernel 优化
  - 硬件架构层：单张 NVIDIA A100 GPU

  Baseline 的缺陷：
  1. **O(N²) 计算复杂度导致长上下文难以扩展**：vision token 数量随帧数线性增长，但 full attention 计算代价按 O(n²) 增长，128 frames（2×帧数）需要 4× FLOPs，256 frames（4×帧数）需要 16× FLOPs，512 frames（8×帧数）需要 64× FLOPs。单卡 A100 在 256 frames 时直接 OOM。
  2. **Token Compression 导致信息丢失**：压缩/剪枝 vision tokens 可以控制序列长度，但高压缩率意味着关键视觉信息可能被丢弃，尤其在需要细粒度推理的长视频问答中损失严重。
  3. **Streaming Inference 引入线性延迟增长**：KV Cache 复用虽支持任意长上下文，但延迟与上下文长度成正比，无法在单次推理中高效完成长视频理解。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法：Free-MoRef 是一个 training-free 方法，受 MoE（Mixture-of-Experts）范式启发，通过将长 vision token 序列划分为多个 short reference chunks（Multi-Reference Partition），在 shallow decoder layers 中用 MoRef Attention 并行 query 各 chunk 并聚合统一 activation，在 mid-layer 通过 Reference Fusion 合并 parallel chunks 为 global reference 后在 deep layers 中标准推理。

  对比 baseline 的全栈执行例子（Free-MoRef@LLaVA-Video-7B, 512 frames, 8 parallel chunks）：
  - 算法层：视频 → FPS=1 采样 512 帧 → SigLIP Vision Encoder 编码 → 512×182=93184 vision tokens → Multi-Reference Partition：M=64 temporal units, N=8 chunks → 每个 chunk 约 11648 tokens → 拼接相同 system prompt + question → 输入 LLM。Shallow layers 0-11（L=12）：各 chunk 独立执行 FlashAttention，产生 O_i = [O_i^sys, O_i^vis, O_i^ques]，计算跨模态注意力 A_i = softmax(Q^ques × (K^vis)^T) → gating weights w_i = max(A_i) / Σ max(A_j) → 聚合 O^fusion = Σ w_i · O_i^ques → 组装 MoRef 输出并替换 O^ques → 残差 + FFN。在 layer 12 执行 Reference Fusion：基于 E_i = mean(A_i, dim=ques)，每个 chunk 保留 top 1/8 vision tokens 并聚合为 global reference（约 11648 vision tokens）。Deep layers 12-27：仅用 global reference 做标准 self-attention → 自回归 decode → 答案。
  - 系统框架层：基于 HuggingFace Transformers + lmms-eval，使用 accelerate toolkit 辅助显存管理。512 frames@Free-MoRef 仅需 400% FLOPs（vs baseline 6400%），且可直接在单卡 A100 上运行（baseline 在 256 frames 已 OOM）。
  - 编译框架层：论文未明确说明
  - kernel 调度层：标准 FlashAttention（MoRef Attention 仍然兼容 FlashAttention 的 causal 接口），额外仅需一次 query-vision cross-modal attention（计算量可忽略）
  - 硬件架构层：单张 NVIDIA A100 GPU，512 frames Free-MoRef 可在不使用 accelerate 的情况下推理，而 baseline 在 256 frames 已 OOM

  解决 baseline 缺陷的对应关系：
  1. **Multi-Reference Partition + MoRef Attention → 解决 O(N²) 计算复杂度**：将长度为 N·L 的序列划分为 N 个长度为 L 的 chunk 并行处理，每层 attention 计算复杂度从 O((N·L)²) 降至 O(N·L²) ≈ O(1/N · full attention)。128 frames 时 FLOPs 仅 110.4%（vs full attention 400%），256 frames 时 163.2%（vs 1600%），512 frames 时 400%（vs 6400%），实现了随帧数线性增长的 FLOPs 而非二次增长。
  2. **MoRef Attention 的全感知机制 → 解决 Token Compression 的信息丢失**：通过 query-vision cross-modal attention 计算 gating weights 来聚合各 chunk 的 question token activation，使得所有 vision tokens 都参与到每个 decoder layer 的 question token 更新中，实现了"equivalent to full attention"的全上下文感知，无需丢弃任何 token。
  3. **单次推理并行处理 → 解决 Streaming Inference 的延迟增长**：所有 chunks 在单个 forward pass 中并行处理，first token latency 保持恒定（与 64-frame baseline 相当），不支持额外的逐 chunk 串行推理延迟，实现 "instant responses"。

## EVA__Efficient_Reinforcement_Learning_for_End-to-End_Video_Agent

- baseline方法是什么？
  Baseline 方法是 perception-first 的视频理解范式，分为两类：(1) Passive MLLM 方法（Qwen2.5-VL、LongVA、LongVila、Video-R1）—— 将整个视频或均匀采样帧作为静态 context 输入 MLLM，不进行选择性注意或自适应推理。模型被动消费所有帧后一次性生成答案。(2) Adaptive Agent 方法（FrameThinker、VideoAgent、VideoMTR）—— 引入外部帧选择工具，但仍遵循"先给均匀采样帧 + query，再 tool call"的 perception-first 流程，且工具控制维度单一（仅可调时间范围，不能调帧数和分辨率），工作流基于固定参数和刚性规则，探索能力受限。

  Baseline（Qwen2.5-VL-7B, 32 frames uniform sampling, LSDBench）全栈执行例子：
  - 算法层：输入一段长视频（>6600s）+ 问题 "figure out action sequences" → 以固定间隔均匀采样 32 帧 → Vision Encoder 逐帧编码为 650 tokens/frame → 32×650=20.8K visual tokens → Projector 映射 → 与 text prompt tokens 拼接 → LLM 28 层 prefill + decode → 生成答案。对于超长视频，32 帧均匀采样严重欠采样，关键帧被跳过，模型只能猜测。
  - 系统框架层：基于 HuggingFace Transformers 或 vLLM 推理
  - 编译框架层：论文未明确说明
  - kernel调度层：标准 FlashAttention，论文未明确说明
  - 硬件架构层：推理用 vLLM on GPU（720p 视频），论文未明确说明

  Baseline 的缺陷：
  1. **被动识别导致冗余视觉处理**：MLLM 被当作被动识别器，预处理整个视频或均匀采样帧，无法选择性关注信息量高的时刻。对于超长视频（>6600s），32 帧均匀采样导致关键帧完全丢失，模型只能猜测。即使在 agent 方法中，模型也是在看到均匀采样帧后才开始推理——先被灌入大量可能无关的视觉 token，影响后续规划。
  2. **固定刚性工具控制维度**：FrameThinker 等 agent 方法仅允许调整时间范围，无法同时控制帧数（nframes）和空间分辨率（resize）。当需要放大关键区域细节时只能增加时间精度但无法提高空间分辨率，导致信息丢失。
  3. **缺乏自主探索策略**：现有方法依赖手工设计的固定工作流（如"先均匀采样 32 帧→再 tool call 特定时间段"），无法根据 query 动态调整策略。模型不能决定"先低分辨率全局浏览，再高分辨率聚焦关键段"这样的自主策略。
  4. **训练与推理行为脱节**：Agent 方法通常在推理时使用 tool call，但在训练时缺乏探索性 tool-use 的奖励信号，导致 trained behavior（SFT 格式跟随）与 expected behavior（高效自适应探索）之间存在 gap。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法：EVA 通过以下设计实现 planning-before-perception 的自主视频 Agent：
  (1) **Planning-before-perception 范式**：初始仅提供 query，模型先基于文本推理生成 plan，再调用 tool 获取视觉信息。通过迭代 summary–plan–action–reflection 循环逐步完善感知和推理。
  (2) **灵活四参数帧选择工具**：start_time、end_time、nframes（帧数）、resize（空间下采样比），同时控制时间和空间粒度，支持"低分辨率全局浏览 → 高分辨率聚焦"等灵活策略。
  (3) **三阶段训练 pipeline**：SFT Cold-Start（学习 tool-call 格式和基本帧选择策略）→ KTO Correction（从典型失败中学习，纠正猜测、欠采样等模式）→ Data-Enhanced GRPO（在线 RL 优化，通过收集失败案例为 teacher MLLM 生成新 QA 来增强训练多样性）。

  对比 baseline 的全栈执行例子（EVA + Qwen2.5-VL-7B, 同一超长视频 + 同一问题）：
  - 算法层：
    1. Round 1（仅 query，无视觉输入）：s_0 = {q, [], []} → MLLM Planning: "需要先获取视频全貌，先低分辨率快速浏览" → Action: frame_select(start=0, end=6600, nframes=10, resize=0.1) → 提取 10 帧低分辨率全局帧 → Summary: 描述每帧内容 → Reflection: "全局帧中看到多个场景...需要聚焦 [200,250] 时间段的高分辨率信息"
    2. Round 2：MLLM Planning: "在 [200,250] 时间段观察到了目标动作的迹象，需要高分辨率确认" → Action: frame_select(start=200, end=250, nframes=100, resize=0.4) → 提取 100 帧高分辨率聚焦帧 → Summary + Reflection: "已获得足够证据" → 生成正确答案
    3. 总 visual tokens: 10×650×0.1 + 100×650×0.4 ≈ 650 + 26000 tokens（但仅展示采样帧中的部分帧给 MLLM，实际使用远少于计算值）
    ↑ LSDBench accuracy: 51.0%（vs baseline 49.2%），仅用 ~76.9 frames/10.3K tokens（vs baseline 32 frames/21K tokens）
  - 系统框架层：vLLM serving（temperature=0），720p 原始视频
  - 编译框架层：论文未明确说明
  - kernel调度层：论文未明确说明
  - 硬件架构层：32 × NVIDIA H100（训练），推理用 vLLM

  解决对应关系：
  | Baseline 缺陷 | EVA 解决方案 |
  |---|---|
  | 被动识别导致冗余视觉处理 | Planning-before-perception：先基于 query 推理规划，再有针对性地获取视觉信息。round 1 用低分辨率全局浏览，round 2+ 聚焦关键段高分辨率。LSDBench: 10.3K vs 21K tokens 且 +1.8% accuracy |
  | 固定刚性工具控制维度 | 四参数工具 (start/end/nframes/resize)：同时控制空间和时间粒度。Case study 展示从 resize=0.1 全局到 resize=0.4 聚焦的自适应切换 |
  | 缺乏自主探索策略 | 三阶段 RL 训练：SFT 学习格式 → KTO 纠正失败模式 → GRPO 在线优化 exploration-exploitation 平衡。消融实验证明 GRPO 阶段 agent 从"格式跟随者"进化为"策略探索者"（更多轮次但每轮更精准 token 分配） |
  | 训练与推理脱节 | KTO 从真实失败轨迹学习（63% correct + 37% rejected），GRPO 用 ROUGE/CSV reward 实现端到端优化，使训练目标与推理表现对齐。SFT→KTO 减少帧数和轮数但提升 3-4%，KTO→GRPO 增加轮数但最高 accuracy |

  训练阶段渐进演化（Ablation 核心发现）：
  - SFT only: 大量帧数 + 多轮交互 → 最低 accuracy（学会了 tool-call 格式但不会高效探索）
  - +KTO: 显著减少帧数和轮数 → 大幅提升 accuracy（学会了避免错误策略：猜测、欠采样、过采样）
  - +GRPO: 帧数再减少但轮数增加 → 最高 accuracy（学会了 multi-round deliberate reasoning + 精准 token 分配）
  - 这揭示了一个**策略演化路径**：格式跟随者 → 纠错后谨慎探索者 → 策略性主动探索者

- baseline方法是什么？
  Baseline 分为两个层面：(1) **无压缩 baseline** —— LLaVA-OneVision VLLM 完整处理所有输入帧的 visual tokens（32 frames × 196 tokens/frame = 6272 visual tokens for 7B），prefilling 阶段计算所有 token 的 QKV 并填充 KV cache，decoding 阶段每步对完整 KV cache 计算注意力，生成每个新 token 在 7B 模型上耗时约 42ms；(2) **One-shot token pruning 方法** —— FastV [3] 在 prefilling 阶段基于第 5 层 attention score 一次性评估 visual token 重要性，保留 top-35% tokens 到 KV cache，之后 decoding 不再改变剪枝结果；LLaVA-PruMerge [39] 基于 CLIP 视觉编码器的 attention score 一次性选择关键 visual token（保留约 55%），同样在整个 decoding 阶段固定不变。两种方法都是 "先评估，一次性剪枝，不再调整" 的单阶段静态策略。

  Baseline（Full tokens + LLaVA-OV-7B, 32 frames, MVBench）全栈执行例子：
  - 算法层：输入一段 32 帧视频 + question → CLIP vision encoder 逐帧编码为 196 tokens (N_v=196) → 视觉嵌入 Z_v ∈ R^{6272×D} → projector 映射到文本空间 H_v' → concat[H_v', H_q] → LLM (28 transformer layers, d=3584, m=18944) prefilling 计算 6272 + N_q 个 token 的 QKV → 全部写入 KV cache → decoding 每步对 6272 visual tokens 做完整 attention → 自回归生成答案。FLOPs 约 41.4T。
  - 系统框架层：LMMs-Eval 评估框架，PyTorch，LLaVA-NeXT 推理代码
  - 编译框架层：论文未明确说明
  - kernel调度层：论文未明确说明（兼容 Flash Attention，无额外 kernel 修改）
  - 硬件架构层：NVIDIA A6000 (48GB) / RTX 4090 (24GB) / A100 (80GB)

  Baseline 的缺陷（基于论文 Figure 2 的核心发现）：
  1. **时间注意力漂移（Temporal Attention Shift）**：论文通过可视化 LLaVA-OV-7B decoding 阶段各迭代步的 attention score 分布（Figure 2）发现：不同预测 token 关注不同的视觉 token，且某些视觉 token（如 frame #1）随 decoding 进行重要性下降，而另一些（如 frame #16）重要性上升。这意味着 one-shot pruning 在 prefilling/首步 evaluation 后固定剪枝集，在后续 decoding 中必然错误剪除后期变得重要的 token 或保留错误 token，导致信息丢失。
  2. **时空冗余未充分利用**：视频存在大量 temporal redundancy（相邻帧相似内容）和 spatial redundancy（单帧内冗余 token），但 one-shot 方法仅基于单次 attention score 剪枝，未利用帧间时序相关性进行合并压缩，无法在保持性能的同时最大化压缩率。
  3. **永久性信息丢失**：FastV 和 PruMerge 一旦剪枝就永久丢弃 token 的 KV cache，如果后续 decoding 需要重新关注这些 token，模型无法恢复。这是 "one-shot" 的根本局限。
  4. **Prefilling 阶段注意力不可靠**：FastV 在 prefilling 阶段基于 prompt token 对 visual token 的注意力来评估重要性，但 prefilling 阶段的注意力分布与 decoding 阶段各步的实际需求不一致，导致 "错误地剪除 LLM 在 decoding 中需要的 token"（原文：FastV incorrectly prunes important tokens overlooked by the LLM during the prefilling phase）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法：DyCoke 首次引入 **training-free 的动态 token 压缩** 策略，通过两阶段设计系统性地解决上述缺陷：(1) Token Temporal Merging (TTM) —— 利用帧间 cosine similarity 合并相似 token，解决时序冗余；(2) KV Cache Dynamic Pruning —— 在 decoding 阶段动态评估并调整 KV cache 中的 visual token 集合，配合 DP cache（Dynamic Pruning cache）实现 token 的可召回机制。

  对比 baseline 的全栈执行例子（DyCoke + LLaVA-OV-7B, K=0.7, L=3, P=0.7, 32 frames, MVBench）：
  - 算法层：
    1. 视觉编码器产出 6272 visual tokens → TTM 阶段：滑动窗口 (window=4) 采样 → O/E 组余弦相似度计算 → E 组高相似 token 剪枝 + O 组内首帧全保留/其余剪枝 → visual tokens 压缩 70%（K=0.7, 保留约 1882 tokens）→ concat 文本 token 送入 LLM prefilling → 计算 QKV 并填充 KV cache。
    2. Decoding 第 t=1 步：在第 L=3 层计算预测 token 对 visual tokens 的 cross-attention A^(3) → Softmax(QK^T/√D) → 取 top P=70% 高 attention token 保留在 KV cache → 剩余 30% 移入 DP cache（~565 visual tokens 在 KV cache, ~1317 在 DP cache）。
    3. Decoding 后续步：监控相邻迭代 attention 分布的 cosine similarity。当 similarity 下降（模型关注点变化），重新在第 3 层计算 cross-attention → 从 DP cache 将注意力回升的 token 动态加回 KV cache → 同步将 KV cache 中注意力下降的 token 移回 DP cache → KV cache 和 DP cache 动态双向流动。每帧平均约 15 tokens 参与注意力矩阵计算。
    4. FLOPs 从 41.4T 降至 17.9T（约 43%），实测 latency 1.49s/example (1.54× speedup), GPU memory 从 34GB 降至 24GB。
  - 系统框架层：LMMs-Eval 评估框架，PyTorch，LLaVA-NeXT（兼容 Flash Attention，仅在第 L 层额外计算 cross-attention）
  - 编译框架层：论文未明确说明
  - kernel调度层：论文未明确说明（兼容 Flash Attention，无自定义 kernel）
  - 硬件架构层：NVIDIA A6000 (48GB) / RTX 4090 (24GB) / A100 (80GB)

  方法 vs Baseline 缺陷的对应映射：
  1. **动态剪枝解决 "时间注意力漂移"**：通过在 decoding 每一步动态重新评估 token 重要性（而非 prefilling 一次评估），模型可以随时调整保留的 token 集合。论文消融实验证明：去除 DP（动态剪枝→one-shot 剪枝）后 VideoDC 性能显著下降（Tab. 5, green），验证了动态机制的有效性。
  2. **TTM 解决 "时空冗余未充分利用"**：TTM 专门利用视频帧间的 temporal coherence（帧间余弦相似度）在 prefilling 阶段合并冗余 token，而非仅靠 attention score 剪枝。随机替换 TTM 的相似度选择为随机选择后性能剧烈下降（Tab. 5, random pruning），验证了基于时序相关性合并的正确性。
  3. **DP Cache 解决 "永久性信息丢失"**：DP cache 存储被剪枝的 token，后续 iteration 可重新召回。双向流动机制（KV cache ↔ DP cache）保证了 token 信息不会永久丢失。这直接对比 baseline 的 "一次性丢弃" 策略。
  4. **TTM 预处理 + Dynamic 评估解决 "Prefilling attention 不可靠"**：TTM 不依赖 attention score（而是语义相似度），避免了 prefilling 阶段注意力不可靠的问题；后续 Dynamic Pruning 在 decoding 真实 attention 分布上评估，比 prefilling 更准确。
  5. **Cost-Effectiveness（同计算预算性能提升）**：压缩后相同 FLOPs 预算下可处理更多帧 → VideoMME 上 32 frames (DyCoke, 17.91T FLOPs) 性能 58.3% vs Full 16 frames (18.99T FLOPs) 56.2%（Tab. 6），验证了压缩的实际效益。

- baseline方法是什么？
  Baseline 是标准的 uniform frame sampling（UNI）策略：从长视频中以固定间隔均匀采样 N 帧，每帧用 56 visual tokens 表示，拼接后送入 LMM（Qwen2.5-VL-7B/32B 或 Qwen3-VL-8B）进行视频问答推理。这种方法是完全的 query-agnostic（查询无关）策略，对所有类型的查询一视同仁。

  Baseline（uniform sampling + Qwen2.5-VL-7B, N=32 frames, MLVU）全栈执行例子：
  - 算法层：输入一段 12 分钟的叙事视频 + 问题 "What color is the man's bike at 3:15?" → 以固定间隔 = (12×60×fps)/32 均匀采样 32 帧 → 每帧经 vision encoder 编码为 56 tokens → 32×56=1792 visual tokens + Q text tokens → LLM 自回归生成答案。过程中，与"3:15 时刻骑车人"无关的帧（如开头风景、结尾字幕）同样占据 token 配额，关键帧可能恰好落在采样间隔之间被跳过 → 准确率随帧数增加反而下降（Figure 2 和 Figure 3 证明：localized queries 在 N=32 时比 N=8 性能更低）。
  - 系统框架层：LMMs-Eval 评估框架 + vLLM 推理加速
  - 编译框架层：论文未明确说明
  - kernel调度层：论文未明确说明
  - 硬件架构层：8 × NVIDIA A100 GPU

  Baseline 的缺陷：
  1. **噪声累积（Noise Accumulation）**：均匀采样给 localized query 注入了大量无关帧作为噪声。Figure 3 和 Figure 5 证明：随着帧数 N 增大，localized query 性能显著下降（Qwen2.5-VL-7B 在 MLVU 上 N=8→32, LQ accuracy 明显下降），而 global query 性能保持稳定。根本原因是 LQ 只需少数特定时间段的帧即可回答，多余帧不仅无益反而干扰模型注意力。
  2. **信息遗漏（Information Missing）**：固定间隔采样可能恰好跳过关键事件所在的帧。长视频中信息并非均匀分布，固定采样无法保证覆盖到查询相关的短暂时段。
  3. **效率低下（Efficiency Gap）**：对所有查询（包括 global query）都使用 query-aware selection 是一种浪费。现有方法（AKS, Q-Frame, BOLT）对每个查询都执行耗时的帧搜索，但论文证明对于 global query，uniform sampling 已足够（Figure 5 右侧两图），高级选择方法在 GQ 上收益递减甚至无益。
  4. **不可扩展（Scalability Limitation）**：AKS 和 Q-Frame 在 frame 数增大（>128）时性能退化，甚至会低于 uniform sampling（Table 1 中红框标记的退化结果），因为它们的候选帧采样池（如 Q-Frame 限于 128 帧）或搜索策略在长视频中不能很好地扩展。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法：DIG（DIvide, then Ground）是一个 training-free 的帧选择框架，通过"先分类再定向"的策略解决上述缺陷：(1) Query Identification —— 用 LLM（Qwen3-Next-80B-A3B）以 CoT 方式自动将查询分为 global 或 localized；(2) 对 global query 直接用高效 uniform sampling；(3) 对 localized query 启动专门 pipeline：CAFS（基于 DINOv2 语义内容的自适应代表性帧选择）→ LMM Reward Assignment（用 LMM 自身而非 CLIPScore 评估帧与查询的相关性，含二维评分：直接有用性 + 相邻帧补充信息潜力）→ Video Refinement（迭代式 reward 阈值化 + 窗口合并，构建仅含查询相关段的 refined video）→ uniform sampling。

  对比 baseline 的全栈执行例子（DIG + Qwen2.5-VL-7B, N=32 frames, 同一视频 + 问题 "What color is the man's bike at 3:15?"）：
  - 算法层：
    1. Query Identification: Qwen3-Next-80B-A3B 分析问题 → "man's bike", "at 3:15" 是具体 referents → isGlobal=false (localized)
    2. CAFS: video 2 fps 采样 → DINOv2 逐帧提取 768-d feature → 计算余弦距离序列 → 检测 prominence>0.1 的峰值 → 选相邻峰值间中间帧为 r-frames（如 47.9 个 r-frames 代表 10 分钟视频）
    3. Reward Assignment: 对每个 r-frame, Qwen2.5-VL-32B 执行 {"description": "...", "reward": 85} — 帧中有自行车和人物（直接有用性 70）+ 暗示相邻帧有其他角度拍摄（补充信息 15）= 总分 85
    4. Video Refinement: 迭代 thresholding（R̄=45 → 更新 rewards → 下一轮 R̄=28 → 稳定 → positive r-frames = 5 个）→ wlen=2 窗口合并 [K_{j-2}, K_{j+3}] → refined video 仅含"3:15 附近骑自行车"相关段
    5. Uniform sample 32 frames from refined video → LMM 推理 → 答案 "Red"
    MLVU accuracy: 70.69%（DIG）vs 61.91%（UNI 32 frames）= +8.78%
  - 系统框架层：LMMs-Eval 评估框架 + vLLM（query identification 和 reward assignment 加速）
  - 编译框架层：论文未明确说明
  - kernel调度层：论文未明确说明
  - 硬件架构层：8 × NVIDIA A100 GPU

  方法如何对应解决 Baseline 缺陷：
  1. **噪声累积** → Query Identification 区分 GQ/LQ + CAFS 只对 LQ 启动 pipeline，且在 refined video 中排除了无关段。Figure 5 左侧三图证明：pipeline 在 LQ 上持续大幅超越 uniform sampling。
  2. **信息遗漏** → CAFS 用内容语义变化（DINOv2 距离峰值）自适应选取 r-frames 而非固定间隔，保证覆盖所有语义场景。Figure 6 证明 CAFS 在长视频（>10min）上的 GIC/LoC 均优于 UNI 和 FPS。
  3. **效率低下** → Query Identification 使全局查询走 uniform sampling 快速路径，节省约 13-20% 的总选择时间（Table 11: VideoMME 节省 19.9%，MLVU 节省 13.3%），同时保持性能（Figure 5 右侧两图：GQ 上 uniform ≈ pipeline）。
  4. **不可扩展** → CAFS 自适应选择的 r-frames 数量随视频内容密度而非长度线性增长（Figure 10 证明），且 LMM-based reward 比 CLIPScore 更可靠（Table 2 证明 LMM reward 在所有 frame 数上一致优于 CLIPScore），使 DIG 在 256 甚至 768 frames 仍保持增益（Table 5: 768 frames 时 MLVU +4.7%, LVB +3.7% vs UNI）。

- baseline方法是什么？
  Baseline 方法是 naive training-free extension of LLaVA-NeXT 到视频领域（Zhang et al., 2024），采用两种静态压缩策略：(1) Uniform Frame Sampling —— 从视频中均匀采样固定数量帧，不考虑帧内容的信息密度差异；(2) Spatial Average Pooling —— 对每帧的 visual token 做空间平均池化压缩，对所有空间位置一视同仁。这种方式将图像预训练的 VLM 直接用于视频输入，不做任何额外的训练或自适应处理。

  Baseline（LLaVA-NeXT 7B, naive training-free video extension, 10 frames, EgoSchema）全栈执行例子：
  - 算法层：输入一段长视频 + 问题 "What did the person do after entering the kitchen?" → 均匀采样 10 帧（固定间隔）→ CLIP 视觉编码器逐帧编码 → 每帧产生 ~576 个 visual tokens → Spatial Average Pooling 压缩 → 拼接为 10 × compressed_tokens 的序列 → 与问题 text tokens 拼接 → Vicuna-7B LLM 前向推理 → 生成答案。整个过程对关键动作帧（如"拿杯子"）和冗余静态帧（如"站立等待"）完全等同处理，关键视觉细节在平均池化中被模糊。
  - 系统框架层：基于 HuggingFace Transformers 的自定义推理代码，论文未明确说明 Serving 框架
  - 编译框架层：论文未明确说明
  - kernel调度层：论文未明确说明
  - 硬件架构层：单卡 NVIDIA RTX A6000 GPU

  Baseline 的缺陷：
  1. **感知瓶颈（Perception Bottleneck）**：静态压缩策略（uniform frame sampling + spatial average pooling）对所有内容等同处理，丢弃了在时间和空间维度上不均匀分布的关键信息。在 EgoSchema 5-frame 实验（Figure 2a）中，uniform sampling + spatial pooling 比无压缩 baseline 精度显著下降，证明静态压缩损失了关键视觉线索。
  2. **Token 过载（Token Overload）**：即使经过压缩，视频输入的 visual token 数量仍远超静态图像，超过了图像预训练 VLM 的处理容量。在 EgoSchema 10-frame 实验中（Figure 2b），baseline 在增加 token 数时性能先升后饱和（plateau），证明模型无法有效利用超出其容量的额外 token。
  3. **Temporal Perception Blindness**：均匀采样无法感知视频内容的时间变化密度——可能在信息密集段采样不足（丢失关键动作），在冗余段采样过多（浪费 token 配额）。
  4. **Spatial Redundancy Unaddressed**：平均池化等静态空间压缩无法区分高信息量 token（如物体边界、人脸）和低信息量 token（如纯色背景），导致空间维度信息损失。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法：D-CoDe 通过两个 training-free 组件解决上述问题：
  (1) **Dynamic Compression（内容感知动态压缩）**：时间维度——在均匀采样基础上，从剩余帧中迭代选择与已选帧语义最不相似的 supplementary frame（基于 CLIP global feature 余弦相似度），确保保留信息丰富的关键片段；空间维度——对每帧 token 按 ℓ2 norm 计算 salience，保留高激活 token（top-β），再通过余弦相似度（τ）贪婪聚类合并冗余 token（anchor + cluster 取平均），在减少冗余的同时保留语义信息。
  (2) **Question Decomposition（问题分解）**：用 GPT-3.5 将复杂问题分解为聚焦于视频不同方面的子问题（如角色位置、动作、交互、场景转换），每个子问题独立用压缩后 visual tokens 推理得到子答案，然后将子答案拼接作为辅助信息与原始问题一起送入 LLM 生成最终答案，引导模型关注视频的不同方面，实现对大量 visual tokens 的全面理解。

  对比 baseline 的全栈执行例子（D-CoDe + LLaVA-NeXT 7B, 同一视频 + 同一问题）：
  - 算法层：输入同一段长视频 + 问题 "What did the person do after entering the kitchen?" → Dynamic Temporal Selection: 均匀采样 12 帧（α=0.85, N=15）→ 从剩余帧中选 3 帧与已选帧语义最不相似的帧（基于 CLIP global feature cosine similarity）→ 共 15 帧 → 每帧: ℓ2 norm salience 计算 → 保留 top 62.5% 高激活 token → 余弦相似度 >= 0.9 的 token 合并为代表 token（mean pooling）→ 拼接压缩后 visual tokens F_final → Question Decomposition: GPT-3.5 生成子问题 ["Where is the person at the start?", "What actions does the person perform?", "What objects does the person interact with?"] → 每个子问题独立用 F_final 推理 → 子答案拼接 + 原始问题 + F_final → LLM 生成最终答案。EgoSchema accuracy 从 44.8%（baseline）提升至 58.0%（+13.2%）。
  - 系统框架层：基于 HuggingFace Transformers + OpenAI API (GPT-3.5-turbo-0125)，论文未说明 Serving 框架
  - 编译框架层：论文未明确说明
  - kernel调度层：论文未明确说明
  - 硬件架构层：单卡 NVIDIA RTX A6000 GPU

  解决对应关系：
  | Baseline 缺陷 | D-CoDe 解决方案 |
  |---|---|
  | 感知瓶颈：静态压缩丢弃关键信息 | Dynamic Compression: 时间维度 supplementary frame selection 补充语义多样帧，空间维度 salience-based pruning + similarity-based merging 保留关键视觉细节（EgoSchema +5.8% over baseline w/ uniform+pooling） |
  | Token 过载：模型无法有效利用超量 token | Question Decomposition: 将复杂问题分解为聚焦子问题，引导模型关注视频的不同方面，使模型能有效利用更多 visual token（Figure 2b: decomposition 的 accuracy 随 token 数持续增长，无 plateau） |
  | 时间感知盲区：均匀采样忽略内容密度差异 | Supplementary frame selection 基于 CLIP semantic dissimilarity（Eq.3-4）：从信息丰富的候选中补充帧，确保不遗漏关键动作。NExT-QA +3.7% (65.4→68.3) |
  | 空间冗余未处理：平均池化无法区分 token 重要性 | ℓ2-norm salience pruning (Eq.5-6) + greedy cosine-similarity merging (Eq.7-9)：低激活 token 被剪枝，相似 token 被合并，保留语义信息同时减少冗余。Ablation: +dynamic spatial compression → +5.8% on EgoSchema |

  效率分析（Table 16, EgoSchema）：
  - Baseline: 44.8% accuracy, 3.927 s/sample
  - + Dynamic Compression: 51.8% (+7.0%), 6.115 s/sample (+55.7% latency)
  - + Question Decomposition: 58.0% (+6.2%), 37.395 s/sample (+511% latency)
  - 轻量变体：替换为更小 CLIP（35% params）→ 58.2%, 35.466 s/sample；限制子问题数 = 5 → 56.0%, 26.273 s/sample；限制子问题数 = 7 → 57.8%, 33.704 s/sample

## Bridging_Episodes_and_Semantics__A_Novel_Framework_for_Long-Form_Video_Understanding

- baseline方法是什么？
  Baseline 方法是现有的长视频理解方法，主要包括两类：(1) Memory-based LLM 方法 —— 如 MA-LMM[14]（对每个 incoming frame 计算与 memory bank 中相邻帧的相似度，仅合并相邻帧）、MovieChat[32]（short-term memory 用 FIFO 机制，long-term memory 用 ToMe token merging）；(2) 传统视频分类方法 —— 如 S5[41]（selective state-space models）、VIS4mer/TranS4mer（state-space + transformer hybrid）、FACT[21]（frame-action cross-attention）。这些方法的共同局限：要么将 memory update 限制在相邻帧（MA-LMM 仅考虑 temporal adjacency，忽略非连续但语义相似的帧），要么用 heuristic 机制（FIFO/random）管理 memory，且缺乏对高层次语义信息的显式建模。

  Baseline（MA-LMM）全栈执行例子：
  - 算法层：输入长视频（14000 帧）→ 采样 2048 帧 → ViT 逐帧编码 → 对每帧：与 memory bank 中相邻帧（最近 K 帧）计算 similarity → 合并到最相似的相邻帧 → memory bank 膨胀 → LLM 生成答案。整个过程中 memory 操作是 local 的（仅看相邻帧），无法发现非相邻但内容相似的帧（如开头和结尾的同一场景），也无法提取跨整个视频的高层次语义主题。
  - 系统框架层：基于 HuggingFace Transformers 的自定义推理代码，论文未明确说明 Serving 框架
  - 编译框架层：论文未明确说明
  - kernel调度层：论文未明确说明
  - 硬件架构层：论文未明确说明

  Baseline 的缺陷：
  1. **Memory 操作仅限于局部相邻帧**：MA-LMM 对每帧仅与其在 memory bank 中 temporally adjacent 的帧进行合并——当相同场景在视频中相隔数分钟再次出现时（如 flashback 场景），无法跨时间距离聚合；而真实的 episodic memory 应将所有语义相似的帧集中到同一 episode，不论其 temporal proximity。
  2. **缺乏对高层次语义信息的显式建模**：MA-LMM 和 MovieChat 仅维护一种 memory 表示（连续帧的压缩），无法区分"特定事件"（episodic: "母亲和父亲争吵"）和"总体主题"（semantic: "家庭关系"）两个认知层次——两个层次往往需要不同时间尺度的聚合。
  3. **FIFO/Random 等 heuristic memory update 策略低效**：FIFO 最先丢弃最早到来的帧，随机策略完全不可控——两者都无法基于帧内容的信息量决定保留哪些帧（Table 6 显示 FIFO accuracy 77.1 vs ECO 78.6）。
  4. **Memory 与 Query 处理脱节**：传统方法在 visual memory 和 Q-Former queries 之间缺乏双向的 episode-level 聚合——memory 和 query 各自独立处理，query 无法感知 episode 结构。
  5. **计算效率低**：MA-LMM 需要处理 2048 帧，推理耗时 467s（Figure 3），且对每帧都要计算与 memory bank 的 pairwise similarity。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法：HERMES 受人类认知中 episodic memory + semantic memory 双系统启发，通过两个可插拔模块解决问题：
  (1) **ECO (Episodic COmpressor)**：维护一个最大容量为 E 的 episode memory buffer。接收新窗口帧特征后，若 buffer 有空间则直接追加；否则将 buffer 和新帧临时拼接，迭代找全局最相似的帧对（cosine similarity）并合并（取平均），直到大小回到 E。这种 **global similarity-based merge** 超越了 temporal adjacency——无论两帧在视频中相隔多远，只要内容相似就会被合并到同一 episode。同时 ECO 也应用于 Episodic Q-Former 内部，对跨窗口 query 进行 episode-level 聚合（ECO_q）。
  (2) **SeTR (Semantics reTRiever)**：以 stride=k 将全部帧分为保留组 K 和被压缩组 K̄，计算跨组 dot-product similarity，将每个 K̄ 帧合并到最相似的 K 帧。经过 Hierarchical Q-Former（frame-level → video-level 两级处理）后，输出高浓缩的 semantic representations。Semantic stream 与 episodic stream 互补——前者提供"整段视频在讲什么"的高层概括，后者提供"具体发生了什么事"的时序细节。
  (3) **Dual-stream 架构 + modular design**：ECO stream（episodic）+ SeTR stream（semantic）→ concat → project → LLM。两个模块可以独立插入现有 SOTA 模型（MA-LMM, LongVA, LLaVA-OneVision）而不需额外训练。

  对比 baseline 的全栈执行例子（HERMES, 同一长视频 14000 帧）：
  - 算法层：输入长视频 → 采样仅 100 帧（vs MA-LMM 2048 帧）→ window size=10 分批 ViT-G/14 编码 → **ECO stream**: 每个 window 在线处理后, 通过 global cosine-similarity merge 压缩到 E=20 个 episodes → Episodic Q-Former 产出 episode-aware queries Q → **SeTR stream**: 100 帧经 stride=5 分组（keep_ratio=0.2）→ 保留 20 帧作为语义代表 → fQFormer 独立增强 → vQFormer 全局聚合 → semantic queries Q_sem → concat[Q, Q_sem] → linear projection → Vicuna-7B 生成答案。整个推理耗时 259s（vs MA-LMM 467s, -44%），accuracy 78.6%（vs MA-LMM 73.3%, +5.3%）。对于"跨时间场景"问题（如电影中角色关系判断），ECO 的 global merge 能力将分处不同时间窗但内容相关的帧聚合到同一 episode——这在 LVU 的 Relationship 和 Writer 子任务上表现突出（+15.4%/+6.8% over S5）。
  - 系统框架层：基于 PyTorch + HuggingFace Transformers 的自定义推理代码，单 V100 推理，论文未说明 Serving 框架
  - 编译框架层：论文未明确说明
  - kernel调度层：论文未明确说明
  - 硬件架构层：论文未明确说明

  解决对应关系：
  | Baseline 缺陷 | HERMES 解决方案 |
  |---|---|
  | Memory 仅局部分配（MA-LMM 仅合并相邻帧） | ECO global cosine-similarity merge：任何帧对无论 temporal distance 均可比较和合并——相似场景（如 flashback）跨越时间被聚合到同一 episode |
  | 缺乏高层次语义信息的显式建模 | SeTR 独立的 semantic stream：stride-based 语义帧选择 + Hierarchical Q-Former → 提供与 episodic 互补的全局主题理解。Table 7 显示移除 SeTR 导致 5% accuracy 下降 |
  | FIFO/Random 等 heuristic memory 策略 | ECO 基于内容的自适应压缩：每步选全局最相似帧对合并，保留最有代表性的 episode prototypes。Table 6: w/o → 55.1%, Random → 76.9%, FIFO → 77.1%, ECO → 78.6% |
  | Memory 与 Query 处理脱节 | Episodic Q-Former (ECO_q)：将 ECO 的 episode 聚合思想扩展到 query 空间——query 也按 episode 组织，使 memory 和 query 在 episode level 对齐 |
  | 计算效率低（MA-LMM 467s） | 仅需 100 帧（vs 2048）+ ECO 在线压缩减少冗余 → 259s 推理（-44%）+ 22 FPS（接近实时）。ECO 插入 MA-LMM 后将其推理时间减少 43%（Table 5） |

  额外优势：由于 ECO 和 SeTR 是 training-free 的前处理模块，它们可以作为 plugin 插入任何现成 VLM 而不需修改模型权重或重新训练。在 LongVA 上 +ECO 将 GPU 内存从 42.5GB 降至 22.9GB（-46%），在 LLaVA-OneVision 上 +ECO 减少 35% 延迟且 +0.67% accuracy（Table 3-4）。

## Atlas__Multi-Scale_Attention_Improves_Long_Context_Image_Modeling

- baseline方法是什么？
  Baseline 方法是 Windowed Self-Attention (WA) / Window-ViT，即标准 ViT 中将 self-attention 限制在局部 k×k 窗口内的变体。WA 的 computational complexity 为 O(N·k²)（相对于全局 self-attention 的 O(N²)），但存在两个关键局限：(1) Limited Receptive Field —— 每个窗口独立处理信息，不同图像区域之间无法直接通信；(2) Boundary Effects —— 跨越多窗口的物体/特征无法在单次 attention 操作内建模。其他 baseline 包括：MambaVision（hybrid SSM+attention，线性复杂度但长上下文表现差）、FasterViT（Hierarchical Attention）、LongViT（Dilated Attention，也声称 O(log N) 通信但未利用局部性）、Swin（Shifted Window，两阶段窗口通信但每个 token 仅能看到相邻窗口）。

  Baseline（Window-ViT, Base-scale, 1024×1024 HR-IN100）全栈执行例子：
  - 算法层：输入 1024×1024×3 图像 → Conv Stem → patchify 为 64×64=4096 tokens（patch_size=16）→ 每个 MSA-like block：将 feature map 划分为 16×16=256 个 non-overlapping windows，window_size=16×16 → 每个 window 内执行标准 Multi-Head Self-Attention（QKV projection + Softmax(QK^T/√d) @ V）→ 所有 window 同时计算 → FFN → 下一个 block → 重复 N 层 → 最终对所有 token 取 mean/CLS token 做 readout → 输出 100-way classification logits。虽然 WA 每个 block 的复杂度仅为 O(4096×256)=O(1M)（远小于全局 attention 的 O(16.8M)），但不同窗口之间的 token 从未直接交互——跨窗口的信息融合仅在最终 readout 层发生。
  - 系统框架层：基于 PyTorch + timm library 训练，论文未明确说明 Serving 框架
  - 编译框架层：论文未明确说明
  - kernel调度层：标准 PyTorch attention kernel（论文未明确说明是否使用 FlashAttention）
  - 硬件架构层：8×H100 GPU 节点，论文未进一步说明硬件细节

  Baseline 的缺陷：
  1. **有限的感受野（Limited Receptive Field）**：如图 3 所示，WA 的每个 token 在 attention 中仅能看到同窗口内的 K=256 个 token（而非全部 4096 个）——特征在深层之前无法跨越窗口通信，对于高分辨率图像（如 4096×4096→65536 tokens），同一物体跨越多窗口的概率显著增大，局部注意力丢失关键全局语义。
  2. **边界效应（Boundary Effects）**：物体跨越窗口边界时被分割为独立 token 组分别处理，其关系只能在最终 readout 层间接学习，导致高分辨率下有效特征交互不足。
  3. **SSM-based（MambaVision）的序列累积限制**：状态空间模型虽为线性复杂度，但其递归/卷积性质在极长序列上信息积累不足——信息在序列中单向/双向传播需 O(N) 步，导致 4096px 分辨率仅 23.36% 精度（Table 2）——远逊于注意力机制的任意 token 对直接交互。
  4. **Dilated Attention（LongViT）未利用局部性**：LongNet 的 dilated attention 通过指数间隔采样实现 O(log N) 通信，但忽略了图像的 2D 空间局部性——视觉任务中相邻像素高度相关，远距离 dilation 跳过邻近关键 token 导致信息丢失，Table 3 显示 MSA 比 Dilated Attention 快 2.39× 且准确率高 20.9%。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法：Multi-Scale Attention (MSA) + Atlas 架构通过以下设计解决问题：
  (1) **Multi-Scale Hierarchical Representation**：使用 4×4 strided max-pool 从 finest scale 开始迭代生成 O(log_S N) 个空间尺度的粗粒度摘要表示。粗尺度 token 用极少的 token 数（如 scale-4: 仅 1 window×256 tokens = 整个 image 的全局摘要）概括大范围区域信息。
  (2) **Bi-directional Cross-Scale Communication**：Top-Down: 每个细粒度窗口内的 token 通过 cross-attention 读取所有更粗尺度的对应 child window tokens——scale-1 的每个 token 可同时 attend 到 scale-1/2/3/4 的所有 K 个 token，其中 scale-4 的 child tokens 已是经过所有中间尺度积累的全局上下文。Bottom-Up: 每个粗粒度 token 通过 cross-attention 从直接 parent window 恢复局部细节（在 strided max-pool 中丢失的 fine-grain 信息）。
  (3) **O(log N) 通信复杂度 + O(N·K·log N) Runtime**：每个 token 到任意其他 token 通过至多 O(log N) 个中间粗尺度 token 传播——通信步数远少于 WA（无法直接通信）和 SSM（O(N) 步），runtime 优于全局 attention（O(N²)）和 dilated attention（同样声称 O(log N) 但实际更慢 2.39×）。
  (4) **Progressive Scale-Dropping (Atlas Architecture)**：以 L 个 macro-stage 逐步放弃最精细尺度，聚焦计算资源于高层特征。如 D={2,2,2,6} 配置：前 2 个 block 积累所有尺度 cross-scale 信息 → 第 3-4 block 仅保留 scale-2/3/4 → 第 5-6 block 仅保留 scale-3/4 → 第 7-12 block 仅处理 scale-4（最粗全局尺度）。比传统 Conv downsampling 更快（38m vs 40m）且更准（70.09% vs 56.14%）。
  (5) **QKV Caching（Appendix C）**：缓存每个 scale 的 QKV 投影以复用跨 cross-attention 操作，避免多尺度场景下的重复计算。

  对比 baseline 的全栈执行例子（Atlas-B/16, D={d1,...,d_L}, 1024×1024 HR-IN100）：
  - 算法层：输入 1024×1024×3 图像 → Conv Stem (4×4 patches) → patchify 为 64×64=4096 tokens（X^(1)）→ 初始多尺度构建: S(stride=4,16): X^(1)→X^(2): 16×16=256 tokens, X^(2)→X^(3): 4×4=16 tokens → L=3 scales → Stage 1 (前 d1 个 MSA blocks): 每个 block —— Summarize: 更新 X^(2)+=MaxPool(X^(1)), X^(3)+=MaxPool(X^(2)) → Top-Down: X^(1) 每个 window(16×16 tokens) cross-attend to [window, X^(2) child(4×4), X^(3) child(1×1)] → X^(2) cross-attend to [window, X^(3) child] → X^(3) 做标准 window-attention (scale=L) → Bottom-Up: X^(2) cross-attend parent X^(1), X^(3) cross-attend parent X^(2) → Stage 2: scale-1 discarded, 仅处理 X^(2..3) → Stage 3: scale-2 discarded, 仅处理 X^(3) → X^(3) 经 readout 输出 100-class logits → 每个 token 的全局上下文路径: X^(1)_token_i → (Top-Down) 读取 X^(3) child token（该 token 已通过 3 层 bottom-up 聚合了 image 所有其他区域的 fine-grain 信息）→ 信息混合完成，至多 log_16(4096)≈3 步
  - 系统框架层：基于 PyTorch + timm library 训练循环，论文未明确说明 Serving 框架
  - 编译框架层：论文未明确说明
  - kernel调度层：标准 PyTorch attention kernel（cross-attention 使用标准 MHA），QKV caching 减少跨 scale 重复投影；论文未说明使用 FlashAttention 但 MSA 的 windowed/cross-attention 模式与之兼容
  - 硬件架构层：8×H100 GPU 节点

  解决对应关系：
  | Baseline 缺陷 | Atlas/MSA 解决方案 |
  |---|---|
  | WA 有限感受野：token 仅见同窗口内 256 tokens | 多尺度表示：scale-4 的 256 个 token 是整个 image 的全局摘要，scale-1 的每个 token 通过 top-down cross-attention 直接读取该全局上下文 |
  | WA 边界效应：跨窗口物体无法直接建模 | Cross-scale communication 不限窗口：每个 token attend to 同 scale 窗口内的 token + 所有 coarser scale 对应 region 的 summary tokens → 跨窗口信息通过粗尺度 summary 间接传播 |
  | SSM (MambaVision) 长序列信息衰减：4096px 仅 23.36% | MSA bi-directional: 每个 token 在至多 O(log N) 步内与任意其他 token 交互（通过粗尺度中间 token）——4096px 达 55.84%（+32.48%） |
  | Dilated Attention (LongViT) 忽略局部性 | MSA 的 windowed cross-attention 仅在粗尺度执行跨区域通信、细尺度保持局部窗口 → 同时利用局部性（同一窗口）和全局性（coarse scale summary） |

## AdaptVision__Efficient_Vision-Language_Models_via_Adaptive_Visual_Acquisition

- baseline方法是什么？
  Baseline 方法是使用固定压缩比的 passive visual token 压缩方法，主要包括：(1) FastV —— 在 layer 2 之后按 attention score 固定剪枝 50% visual tokens；(2) SparseVLM —— 基于跨模态 relevance 选择语义相关 visual tokens，固定 50%/70% retention；(3) VisionZip —— 保留语义重要的 visual tokens，固定 50%/70% retention；(4) VisionThink —— 使用 RL 在低分辨率（25% token）和高分辨率（100% token）之间二选一，但限于 coarse-grained 决策；(5) Down-Sample baseline —— 固定 1/4 分辨率 25% tokens 直接回答。所有方法都是被动、固定比例压缩，无法自适应不同任务复杂度所需的 token 数量。

  Baseline（FastV 50%, Qwen2.5-VL-7B-Instruct）全栈执行例子：
  - 算法层：用户上传一张高分辨率图表（2048×1024） + 问题 "What is the value at Q3?" → Vision Encoder (ViT) 编码为 2678 个 visual tokens → Projector 对齐 → 与 system prompt 和 question tokens 拼接 → LLM decoder layer 1-2: 全量 visual tokens prefill → layer 2 之后: 按累积 attention score 排序，固定剪枝 50%（保留 1339 tokens） → 其余 token 的 KV cache 被丢弃 → layer 3-28: 在 1339 个 visual tokens + text tokens 上继续 prefill + decoding → 生成答案
  - 系统框架层：基于 HuggingFace Transformers 推理，论文未明确说明 Serving 框架
  - 编译框架层：论文未明确说明
  - kernel调度层：使用标准 FlashAttention
  - 硬件架构层：论文未明确说明

  Baseline 的缺陷：
  1. **被动固定压缩比无法适应任务难度差异**：简单任务（如 POPE 物体存在性判断）仅需极少 visual tokens 即可正确回答，但 FastV 仍保留 50% tokens 造成浪费；复杂任务（如 MathVerse 数学推理、ChartQA 图表问答）50% 压缩损失关键细节导致精度下降。
  2. **全局固定压缩忽略图像空间局部性**：视觉问答通常仅依赖图像中少数关键区域（如表格中的特定单元格、文档中的特定段落），但 uniform pruning 在整个图像上均匀丢弃 token，无法聚焦关键区域。
  3. **VisionThink 的二选一粗粒度决策**：只能整体切换低/高分辨率，无法精细定位需要高分辨率的具体区域，导致高分辨率模式下仍消耗 100% tokens。
  4. **缺乏 human-like active vision 的 coarse-to-fine 处理**：人类视觉系统先获取场景 gist（低空间频率），再选择性关注 salient 区域（高空间频率），现有方法未建模这种自适应信息获取过程。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法：AdaptVision 通过 RL 训练 VLM 实现自适应 coarse-to-fine visual token 获取：
  (1) **Coarse-to-fine 框架**：始终以 1/4 低分辨率图像（25% tokens）起步，模型自主决策是直接回答还是调用 bounding box tool 裁剪高分辨率关键区域。最大化在简单样本上节省 token，在困难样本上精准获取关键高分辨率信息。
  (2) **Decoupled Turn Policy Optimization (DTPO)**：解耦策略损失为 Tool Token 和 Answer Token 分别归一化（解决 imbalanced optimization），并分别计算 outcome advantage 和 tool advantage（解决 ambiguous credit assignment），使 RL 训练稳定收敛到自适应 tool-use 策略。
  (3) **精细奖励设计**：Outcome Reward（准确度 + 格式 + 平衡惩罚防止 tool 过度使用或 lazy guessing）+ Tool Reward（裁剪区域正确性 - 面积惩罚鼓励最小化 crop 区域）。
  (4) **Fine-grained visual acquisition**：通过 bbox tool 精准裁剪关键区域，而非整体切换分辨率，使 token 消耗更具针对性。

  对比 baseline 的全栈执行例子（AdaptVision + Qwen2.5-VL-7B-Instruct，同一图表问答）：
  - 算法层：用户上传同一张高分辨率图表（2048×1024） + 问题 "What is the value at Q3?" → I_low = resize(2048×1024 → 512×256) → Vision Encoder 编码 ≈670 tokens → Projector 映射 → 拼接 x_sys + V_low + q → LLM 首轮自回归生成 → 模型推理：`<think> 需要在图表中定位 Q3 对应的数值，低分辨率下无法辨认细节...</think> <tool call>{"name":"request_local_region","arguments":{"bbox_2d":[420,180,680,320]}}</tool call>` → 从 I_high 裁剪 bbox 区域（260×140 pixels） → Vision Encoder + Projector 得到 ≈170 个 crop visual tokens → 拼接续推 → `<think> Q3 对应数值为 47.2</think> <answer>47.2</answer>` → 总 visual tokens ≈ 670 + 170 = 840（仅 31.4% of baseline 2678），远少于 FastV 50% 的 1339 tokens，且保留了关键高分辨率信息
  - 系统框架层：veRL 框架进行 RL 训练（4 节点 × 8 H20 GPU），vLLM 框架推理（temperature=0）
  - 编译框架层：论文未明确说明
  - kernel调度层：标准 FlashAttention
  - 硬件架构层：论文未明确说明

  对应解决：
  | Baseline 缺陷 | AdaptVision 解决方案 |
  |---|---|
  | 固定压缩比无视任务难度 | RL 学习自适应策略：POPE 直接回答（25% token），ChartQA 频繁 tool call（~33% avg token） |
  | 全局压缩忽略空间局部性 | Bbox tool 精准裁剪关键区域，面积惩罚确保 crop 最小化 |
  | VisionThink 粗粒度二选一 | Fine-grained bbox 定位：不切换全局分辨率，仅获取必要区域的高分辨率信息 |
  | 缺乏 human-like active vision | Coarse-to-fine 框架 + RL 训练：先 gist（低分辨率）→ 选择性 attention（crop 关键区域）= 模拟人类视觉 |

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

## Accelerating_Vision_Transformers_with_Adaptive_Patch_Sizes

- baseline方法是什么？
  Baseline 方法是 ViT 中使用的均匀固定 patch 划分（uniform patchification）：无论图像内容如何，每个 p×p 区域分配一个 token，产生固定数量 N = HW/p² 个 token。高分辨率图像下 token 数平方增长，而图像中大面积区域（天空、纯色背景、模糊虚化区域）信息冗余度高，与复杂区域（人脸、物体边缘、纹理丰富区域）被同等对待，造成大量注意力计算浪费。代表性 baseline 实现：(1) Vanilla ViT（timm library + FlashAttention）—— 图/像分为均匀 p×p patch，线性投影为 d_embed 维 token，全注意力计算 O(N²)；(2) 输入级 baseline（Random masking / Resizing-only）—— Random 按 APT 的压缩比随机丢弃 patch（FLIP 风格），Resizing 将大 patch resize 到 p×p 仅用单一路径编码（Quadformer 风格）；(3) 层级 token 合并 baseline（EViT, ToMe, PPT, DTEM）—— 在 ViT 各层之间合并相似 token，但固定合并比例且通常不兼容 FlashAttention。

  Baseline（Vanilla ViT，ViT-L/14@336×336）全栈执行例子：
  - 算法层：输入 336×336×3 图像 → 均匀划分为 24×24=576 个 14×14 patch → 每个 patch 经线性层 E 投影为 1024 维 token → 加上可学习位置编码 → 576+1 个 token 送入 24 层 Transformer（每层 self-attention 576² 次交互，共 ~8M FLOPs per layer）→ 取 CLS token 输出分类 logits
  - 系统框架层：论文基于 timm library 实现，使用 PyTorch 原生训练循环，未涉及 Serving 框架
  - 编译框架层：论文未明确说明
  - kernel调度层：使用 FlashAttention-2 (Dao et al., 2022; Dao, 2024) 加速标准 scaled dot-product attention，xFormers (Lefaudeux et al., 2022) 处理 sequence packing 的 block-diagonal mask
  - 硬件架构层：论文未明确说明

  Baseline 的缺陷：
  1. **固定 patch 大小不考虑内容冗余度**：均匀背景（如纯蓝天空）与复杂纹理（如鸟类羽毛）被等粒度编码，前者信息密度远低于后者但占用同等计算。高分辨率下，冗余 token 比例更高——336² 产生 576 token vs 224² 仅 196 token。
  2. **高分辨率/大模型下注意力计算瓶颈加剧**：ViT-L@448 的 GFLOPS 为 ViT-L@224 的 10.8 倍（645 vs 59.7），训练时间 31.4h vs 15.9h。
  3. **层级合并方法 token 减少有限且训练 Inference gap 大**：ToMe/EViT 在训练时可加速但在推理时由于不规则 shape 和 padding 而产生实际 wall-clock 加速远小于理论 FLOPs 减少；且多数不兼容 FlashAttention，用 weighted attention 导致更慢。
  4. **固定比例合并对图像复杂度不敏感**：纯白图像只合并 50% token 不足够，繁忙城市街景合并 50% token 可能丢弃关键信息。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法：Adaptive Patch Transformer (APT) 通过以下设计解决问题：
  (1) **多尺度熵驱动的自适应 patch 大小分配**：对图像以 quadtree 层级方式从粗到细计算像素熵 H(P) = -∑p_i log₂ p_i，低熵区域分配大 patch，高熵区域分配小 patch。低熵意味着高冗余度（如纯色背景），可用更少的 token 无损表达。
  (2) **双路 Patch Aggregation + Zero-initialized MLP**：大 patch 同时 (a) resize 到 p×p 经 E 嵌入 和 (b) 拆分为 p×p 子 patch、经 E 嵌入后 Conv2d 降采样聚合，两路通过 zero-initialized MLP 融合。ZeroMLP 初始输出为零，保证用于预训练 ViT 时初始性能不退化（无训练时等价于纯 Resizing），仅需 1 epoch 微调即可匹配原始性能。
  (3) **Sequence Packing + Block-diagonal Mask**：不同图像产生不同数量 token，拼接为单一序列 + attention mask，兼容 FlashAttention/xFormers 而不引入 padding 开销。
  (4) **位置编码插值**：大 patch 的位置编码从小 patch 网格双线性插值获得，无需学习新参数。

  对比 baseline 的全栈执行例子（APT，ViT-L/14@336×336, τ₁=5.75, τ₂=4.0）：
  - 算法层：输入 336×336×3 图像 → CPU 多核算熵：先以 56×56 patch 扫描（6×6 网格），每个 patch 计算 H(P)，H<4.0 则分配 56×56 patch；H≥4.0 则拆为 4 个 28×28 子区域，每个子区域计算 H(P')，H'<5.75 则分配 28×28 patch；否则再拆为 14×14 最终 patch → 得到变长 patch 列表（典型约 400~420 个 patch，相比 baseline 576 减少 ~28%） → 每个 patch 经双路 Aggregation（resize+子 patch Conv2d 聚合 + ZeroMLP）→ 得到 ~400+ 个 d_embed=1024 token → 位置编码从 24×24 网格插值到各 patch 对应位置 → batch 内多图 token 拼接 + block-diagonal mask → 送入 24 层 ViT（每层 attention 从 576² 降至 ~420²，GFLOPS 从 174.7 降至 76.8） → CLS token 分类 → 训练 wall-clock 从 15.9h 降至 9.9h（+61% speedup）
  - 系统框架层：论文基于 timm library 实现，使用 PyTorch 原生训练循环
  - 编译框架层：论文未明确说明
  - kernel调度层：FlashAttention-2 处理标准 self-attention，xFormers 处理 sequence packing 的 block-diagonal mask；熵计算在 CPU dataloader 上多核并行并与 GPU 前向计算重叠，无额外 GPU 开销
  - 硬件架构层：论文未明确说明

  解决对应关系：
  - Baseline 缺陷 1（固定 patch 忽略内容冗余）→ 多尺度熵自适应 patch：纯蓝天空→64×64 patch（1 token），鸟类羽毛→16×16 patch（4 token），从 ~576 token 降至 ~400 token，GFLOPS 降 56%（174.7→76.8），且精度匹配（88.1 vs 88.2）
  - Baseline 缺陷 2（高分辨率/大模型瓶颈）→ 分辨率越高压缩越大：ViT-L@448 从 645 GFLOPS → 268 GFLOPS（-58%），speedup +86%；ViT-H@336 speedup +50%
  - Baseline 缺陷 3（层级合并不兼容 FlashAttention/训练推理 gap）→ APT 在模型前向之前完成 token 压缩，使用标准 FlashAttention，无 weighted attention 或动态 token shape
  - Baseline 缺陷 4（固定比例不感知复杂度）→ 每张图像根据内容自适应 token 数量分布（图 5：从接近上限到约 30% 下限），检测/分割任务可通过降低 τ 适应对细节的更高要求

- baseline方法是什么？
  Baseline 方法是现有的 fixed compression ratio 或 hand-crafted metric 的 visual token 压缩/剪枝方法，包括：(1) PDrop[7] —— 使用 LLM 深层（layer K之后）的第一个生成 token 对 visual token 的 cross-attention 作为重要性指标进行剪枝；(2) VisionZip[4] —— 使用视觉 token 间的相似度矩阵进行合并/剪枝；(3) VScan[5] —— 基于固定 cross-attention 指标进行 token 选择；(4) DivPrune[23]、CDPruner[20] —— 基于相似度/多样性的 token 选择。这些方法共同特点：使用固定的压缩比例和手工设计的 token 重要性度量。
  
  Baseline（PDrop 为代表的手工 cross-attention 方法，Qwen2.5-VL-7B）的全栈执行例子：
  - 算法层：用户上传高分辨率图像 + "What player is number 21?" → Vision Encoder 编码为 N_v 个 visual tokens → Projector 映射到 textual embedding space → 与 text tokens 拼接送入 LLM decoder → 完成全部 L 层 prefill → 生成第一个 response token → 取第一个 token 在深层（如 layer 28）的 cross-attention 作为 visual token 的重要性 → 基于此手工度量剪枝（若采用 step-by-step pruning） → 继续 autoregressive decode
  - 系统框架层：论文未明确说明部署 Serving 框架
  - 编译框架层：论文未明确说明
  - kernel调度层：论文未明确说明，但指出 VisionZip/VScan 的 dense similarity matrix 计算与 FlashAttention2 不兼容，导致 OOM
  - 硬件架构层：论文未明确说明

  Baseline 的缺陷：
  1. **手工度量不可靠（free-form generation 场景）**：如图 2 所示，当不要求简洁回答时，cross-attention 初始阶段聚焦于无关区域（而非答案相关视觉 token），导致 imprecise pruning 和 degraded output。Table 1 定量证明：PDrop 从 w/ brief 的 0.753 降至 w/o brief 的 0.406。
  2. **固定压缩比无法适应场景复杂度**：对所有输入使用相同压缩比，对小目标场景（如 DocVQA，目标区域仅占 6.8% 面积）可能剪枝不足，对大目标场景（如 VSR，目标区域占 41.6%）可能剪枝过度。
  3. **Per-step pruning 计算低效**：关键 cross-attention 仅在 LLM 深层（K层之后）出现，意味着必须保留前 K 层的全部 visual token KV cache，节省的计算和内存有限。
  4. **Dense similarity matrix 内存爆炸**：VisionZip 和 VScan 需要计算 visual token 间的稠密相似度矩阵，与 FlashAttention 的内存高效设计冲突，高分辨率输入下直接 OOM。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法：GlimpsePrune 通过以下设计解决问题：
  (1) **Glimpse Token + VIP 学习 Data-driven 剪枝度量**：在 prefill 阶段第 K 层，提取 glimpse token 对所有 visual token 的 cross-attention，与多层 hierarchical visual features 一起输入轻量 VIP 网络，输出每个 visual token 的重要性概率。该度量通过 20K GQA 样本训练得到（语言 loss + Dice/BCE 定位 loss，LVLM 参数冻结），能泛化到各种 VQA 场景。
  (2) **动态压缩比**：VIP 输出 per-token 概率分布而非固定比例，对不同图像自适应决定保留多少 token——对小目标场景（DocVQA）自动保留极低比例（3.6%），对大目标场景（VSR）保留较高比例（39.4% unrestricted）。
  (3) **One-shot 全深度 KV cache 剪枝**：仅在第 K 层执行一次剪枝，同步移除前 K 层和本层的无关 KV cache 条目——使剩余 L-K 层 prefill 和全部 decoding 阶段均在缩减序列上运行，最大化 decoding 阶段内存和 I/O 节省。
  (4) **线性复杂度 compatible with FlashAttention**：cross-attention from single query to all visual tokens 为 O(N_v × D) 线性复杂度，与 FlashAttention 兼容，不产生 OOM。

  对比 baseline 的全栈执行例子（GlimpsePrune, Qwen2.5-VL-7B, K=19）：
  - 算法层：用户上传高分辨率图像 + 问题 → Vision Encoder 编码 → Projector → 插入 glimpse token → LLM prefill 前 K=19 层（标准 forward，glimpse token 每层加上可学习嵌入） → 第 19 层提取 glimpse token cross-attention A(N_v × 28 heads) + 4层 hierarchical visual features V → VIP(M=4 self-attention blocks + 2D RoPE conditional attention) → importance map P(N_v,) → Top-K 选择保留 N_v' 个 visual token → 一次性移除不重要 token 在第 1~19 层 KV cache 中对应条目 → 丢弃 glimpse token → 剩余 9 层 prefill 在 N_v' + N_t 序列上执行 → Decoding 阶段 autoregressive 生成，每个 step attention 开销 O((N_v' + N_t) × D) ≈ O(202 × D) vs baseline O(5074 × D)
  - 系统框架层：论文未明确说明 Serving 框架
  - 编译框架层：论文未明确说明
  - kernel调度层：论文未明确说明，但方法设计保证与 FlashAttention2 兼容
  - 硬件架构层：论文未明确说明

  解决对应关系：
  - Baseline 缺陷 1（手工度量不可靠）→ Glimpse Token + Dice/BCE 定位 loss 训练：让模型学会数据驱动的剪枝度量，在 free-form generation 中稳定工作（PDrop w/o brief 0.406 → GlimpsePrune 0.939，Table 1）
  - Baseline 缺陷 2（固定压缩比）→ VIP 输出 per-token 概率 + Top-K 自适应选择：对 DocVQA 自动保留 3.6% token（仍保持 accuracy 0.962），对 VSR 自动保留 39.4%（accuracy 0.618 vs baseline 0.620）
  - Baseline 缺陷 3（per-step pruning 低效）→ One-shot pruning at layer K：prefill FLOPs 降至 69.1%，decoding 初始 KV cache 长度从 5074 降至 202.5 tokens，峰值内存降至 72.8%
  - Baseline 缺陷 4（dense similarity OOM）→ Linear-cost cross-attention + FlashAttention 兼容：高分辨率下不会 OOM，而 VisionZip/VScan 在同条件下 OOM

- baseline方法是什么？
  Baseline方法分为两类：(1) Vanilla —— 直接以固定分辨率（如448×448）将所有图像输入预训练LVLM（InternVL2-8B或Qwen2VL-7B），不对分辨率做任何特殊处理；(2) All-XN —— 将所有图像均等倍率上采样（最高pixel数放大N倍）后输入LVLM，以提升文本识别能力。两者均不区分图像重要性。
  
  全栈执行例子（All-X4，InternVL2-8B，MP-DocVQA场景）：
  - 算法层：用户上传M张文档图像+文本问题 → 所有M张图像统一被crop为16个sub-images（448×448） → ViT编码 → 与global image拼接 → 每张图像产生约1500+个visual tokens → M张图像共MnL_v个visual tokens → 与text tokens拼接送入LLM → 生成答案
  - 系统框架层：InternVL2动态分辨率处理策略(crop+resize)，论文未说明部署Serving框架
  - 编译框架层：论文未明确说明
  - kernel调度层：论文未明确说明
  - 硬件架构层：论文未明确说明

  Baseline的缺陷：对不相关图像同样进行高分辨率编码，大量visual tokens与问题无关（论文指出仅12.5%的visual tokens与正确答案相关），导致GPU内存浪费和推理延迟增加，在token数受限场景下难以扩展。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法：ADMIRE通过三个training-free模块动态调整多图像的分辨率：(1) TIE利用LLM第一层attention map，计算text token对所有visual token的注意力分数，聚合为每张图像的重要性分数并归一化；(2) KIE仅对Top-k张very important图像上采样，而非全部；(3) DVD对less important图像按attention保留50% token、对not important图像直接丢弃。

  对比baseline的全栈执行例子（ADMIRE-Top5-X4，Qwen2VL-7B，MP-DocVQA场景）：
  - 算法层：用户上传M张文档图像+文本问题 → TIE先用第一层LLM attention计算出每张图像的text-guided重要性分数S → Top5选为very important(p_kie)、低于0.5γ的为not important(p_Idvd)、介于0.5γ到1.5γ的为less important(p_Vdvd) → KIE对5张very important图像上采样4×(max pixels)后重新ViT编码，产生额外高分辨率visual tokens并插入原tokens之前 → DVD丢弃p_Idvd中图像的visual tokens、对p_Vdvd中图像只保留attention score最高的50% visual tokens → 最终visual tokens约1766个（vs Vanilla 1448 vs All-X4 2788） → 与text tokens拼接送入LLM → 生成答案
  - 系统框架层：论文未明确说明
  - 编译框架层：论文未明确说明
  - kernel调度层：论文未明确说明
  - 硬件架构层：论文未明确说明

  解决对应关系：
  - Baseline缺陷1（token浪费）：DVD多级token压缩机制——对not important图像整张丢弃，对less important图像保留50%高注意力token，直接减少冗余visual tokens
  - Baseline缺陷2（无法聚焦关键信息）：TIE利用attention的文本引导评分，自动发现evidence candidate图像，使模型聚焦于包含答案的图像
  - Baseline缺陷3（全图增强导致token爆炸）：KIE仅对Top-k张图像增强分辨率，总计5*n*L_v + (M-5)*L_v个visual tokens，远小于All-XN的M*n*L_v，同时通过DVD压缩不相关图像以平衡总token数

## Beyond_Accuracy__Evaluating_Grounded_Visual_Evidence_in_Thinking_with_Images__ViEBench

- baseline方法是什么？
  Baseline 方法是现有的 outcome-oriented 多模态 benchmark 评估范式，代表性 benchmark 包括：(1) V* Bench —— 191 个 QA pairs，仅做 perception 评估、无 reasoning 任务、无 BBox 标注、无过程评估；(2) HRBench —— 1600 个 QA pairs，仅做 perception 评估；(3) InfoVQA —— 2801 个 QA pairs，仅做 perception 评估；(4) VisualProbe —— 515 个 QA pairs，仅做 perception 评估。所有现有 benchmark 依赖单一最终答案 accuracy 作为唯一指标，将模型视为"黑盒"，无法诊断性能退化来源于 grounding 失败还是 reasoning 不足。此外，现有 benchmark 的任务设计以 fine-grained recognition 为主，不要求多步逻辑推理。

  Baseline（现有 accuracy-only 评估范式，以 V* Bench 为代表）全栈执行例子：
  - 算法层：给定高分辨率图像 + 问题 "What brand is the coffee machine on the third shelf?" → 模型（如 Qwen3-VL-32B）处理图像 → 生成裁剪/回答 → evaluator 仅比较最终答案与标准答案 → 输出 accuracy score。在此范式下，若模型聚焦于完全无关的区域（如天花板）但凭文本先验猜对品牌，仍被判为"正确"——accuracy 无法区分 faithful reasoning 与 lucky guessing
  - 系统框架层：VLMEvalKit (Duan et al., 2024) 作为统一评估框架，论文未涉及 Serving 框架修改
  - 编译框架层：论文未明确说明
  - kernel调度层：论文未明确说明
  - 硬件架构层：论文未明确说明

  Baseline 的缺陷：
  1. **仅衡量最终答案、无过程级诊断**：模型通过 irrelevant visual region 猜对答案（Ungrounded Correct Answer）与基于正确视觉证据推理得到答案（Valid Grounded Reasoning）在 accuracy 指标下无法区分——传统 benchmark 系统性高估了模型的可靠性。
  2. **无视觉证据标注（BBox）**：现有 benchmark 不提供 expert-annotated 黄金 BBox，无法定量验证模型的视觉操作（zooming/cropping）是否聚焦于正确的图像区域。
  3. **任务类型单一（仅 perception）**：现有 benchmark 以 OCR、object counting、attribute recognition 等纯感知任务为主，不评估模型整合局部视觉线索与先验知识进行多步推理的能力——这与真实应用（如工业检测中判断设备故障、城市导航中识别违规行为）的巨大差距。
  4. **缺乏高空间稀疏性设计**：现有 benchmark 未刻意控制关键视觉证据的空间占比，许多问题可通过全局视图直接回答，无法强制模型执行 zooming/cropping 操作。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法：ViEBench 通过以下设计实现从 outcome-oriented 到 process-verifiable 的评估范式转变：
  (1) **Expert-annotated Gold BBox**：为每个样本标注"最小不可或缺证据"的精确 BBox，作为 IoA 计算的 ground truth，使视觉操作的准确性可量化。
  (2) **Dual-Axis Capability Matrix**：基于 IoA 构建 Grounding 轴（G⁺/G⁻）× Answer 轴（A⁺/A⁻）的二维诊断矩阵，将模型表现分解为四个象限：G⁺·A⁺（faithful reasoning）、G⁺·A⁻（定位成功但推理失败）、G⁻·A⁺（无根据的正确答案——hallucinatory reasoning）、G⁻·A⁻（双重失败）。
  (3) **Reasoning + Perception 双任务设计**：引入 reasoning 任务要求模型在定位视觉线索后整合先验知识进行多步逻辑推理，暴露 accuracy-only 下不可见的 capability collapse。
  (4) **Extreme Spatial Sparsity**：关键证据平均仅占 0.32%-0.63% 图像面积，在全局视图下 sub-perceptual，强制模型执行精确本地 zooming 操作。

  对比 baseline 的全栈执行例子（ViEBench 评估, Qwen3-VL-32B-Instruct 在 reasoning 任务上）：
  - 算法层：给定 2048×1536 工业场景图像 + reasoning 问题 "Is the pressure gauge reading within the safe operating range?" → 模型自主 zooming 到 pressure gauge 区域 → B_pred 由模型生成 → evaluator 计算 IoA(B_pred, B_gt)：B_gt 为专家标注的 pressure gauge 精确 BBox（仅占 ~0.6% 图像面积） → IoA = max(coverage, concentration) → G⁺ if IoA>0.5 else G⁻ → 同时判断最终答案正确性 → 分配到四象限之一 → 汇总指标：Acc=74%, GS=68%, G⁺·A⁺=56%, G⁺·A⁻=13%, G⁻·A⁺=17%, G⁻·A⁻=15% → 诊断：13% 的样本模型成功定位了 pressure gauge 但推理错误（可能误读刻度），17% 的样本模型在错误区域获得了正确答案（可能靠文本先验"/pressure gauge 一般在安全范围内"猜测）
  - 系统框架层：各 agentic model 官方仓库的评估 pipeline，End-to-end models 使用 VLMEvalKit；论文未涉及 Serving 框架修改
  - 编译框架层：论文未明确说明
  - kernel调度层：论文未明确说明
  - 硬件架构层：论文未明确说明

  解决对应关系：
  | Baseline 缺陷 | ViEBench 解决方案 |
  |---|---|
  | 仅衡量最终答案、无过程诊断 | Dual-Axis Capability Matrix：G⁺·A⁻ 直接揭示"定位成功但推理失败"的 semantic reasoning bottleneck（Mini-o3 reasoning 高达 28%）；G⁻·A⁺ 暴露"无根据正确答案"的 superficial correctness（DeepEyes reasoning 33%） |
  | 无视觉证据标注 | Expert-annotated Gold BBox + IoA：双向 IoA 容忍 expansive coverage 和 tight focus 两种策略，Fig.5 显示 Qwen3-VL 系列使用 expansive coverage（高 IoA(B_pred,B_gt)），DeepEyes 使用 tight focus（高 IoA(B_gt,B_pred)） |
  | 任务类型单一（仅 perception） | ViEBench-R (reasoning)：要求多步逻辑推理整合视觉线索与先验知识，Mini-o3 perception Acc=73% → reasoning Acc=58%（-15%），而 GS 保持一致（78%），证明瓶颈在 reasoning 而非 perception |
  | 缺乏高空间稀疏性 | Gold BBox 平均占 0.32%-0.63%：Qwen3-VL-32B 在 perception 上 TR=93%（频繁 tool call）、reasoning 上 TR=95%，验证了稀疏性设计成功迫使模型执行 zooming |

## CurveStream__Boosting_Streaming_Video_Understanding_in_MLLMs_via_Curvature-Aware_Hierarchical_Visual_Memory_Management

- baseline方法是什么？
  Baseline 方法是现有的流视频记忆管理方法（多应用于无限长 streaming video 场景，统一受限于 token economy，固定 GPU 内存预算下必须持续管理视觉记忆队列），主要包括两类：(1) 视觉信息保持策略（Visual Retention）—— Uniform Sampling（均匀采样 1fps 或 64fps 下行采样）、Cosine Similarity-based selection（帧间余弦相似度做低层次筛选）、Optical Flow / Pyramid Optical Flow（密集光流计算物理运动强度）；(2) 记忆管理机制（Memory Management）—— HERMES（规则驱动的 KV cache 逐出，使用滑窗和时空冗余度量被动丢弃旧 token）、FreshMem（frequency-space hybrid memory，频率域混合记忆）、ReKV（外部存储 + post-hoc query-driven retrieval，延迟查询驱动的特征检索）。
  
  Baseline（Qwen2.5-VL-7B + 1fps uniform sampling, StreamingBench 场景）全栈执行例子：
  - 算法层：无限长流视频持续输入 → 每 1 秒均匀采样 1 帧 → 固定窗口内保留最近 64 帧 → Vision Encoder 编码为 visual tokens → 与 text query 拼接送入 LLM decoder → 当窗口满时 FIFO 丢弃最旧帧 → LLM 生成答案。整个过程对所有帧一视同仁，不区分帧的语义信息价值。对于长时间段中稀缺但关键的事件（如一个短促的动作变化），uniform sampling 可能在大量静态背景帧（如长时间观察静止物体）上浪费 token 配额，导致关键帧在 FIFO 驱逐中被挤压出去（catastrophic forgetting）。
  - 系统框架层：论文未明确说明 Serving 框架
  - 编译框架层：论文未明确说明
  - kernel调度层：论文未明确说明
  - 硬件架构层：单张推理 GPU，论文未明确 GPU 型号和硬件细节

  Baseline 的缺陷：
  1. **语义碎片化（Semantic Fragmentation）**：Uniform sampling 或 cosine similarity 等低层次物理度量缺乏内在语义感知——均匀选择或基于像素相似度过滤的帧难以保持上下文连贯性。例如长时间固定场景中，uniform 可能在语义不变时持续积累冗余帧，而真正的语义突变帧（如新物体突然出现）可能因队列刚满被逐出。
  2. **信息模糊化（Information Blurring）**：无差别的 Feature Compression（如 HERMES 对不活跃 token 的全局压缩）不可逆地模糊了短暂但关键的语义转换点——这些转换点是因果推理链的核心锚点。例如在动作识别中，一个 0.5s 的"闪避"动作可能被压缩为模糊背景区域的一部分。
  3. **延迟感知（Delayed Perception）**：ReKV 和类似 retrieval 机制的框架依赖 post-hoc 查询在外部存储中检索相关帧——这在无限流场景中本质上是被动且滞后的，限制了对未知事件的实时主动感知能力。
  4. **对局部噪声过度敏感**：Cosine similarity 和 optical flow 等物理度量在动态场景中易受局部运动噪声影响——平滑的相机平推（背景全局大位移）和真正的语义突变（新物体进入/动作边界）在物理度量下表现相似，导致选择性模糊或过度保留无关帧。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法：CurveStream 通过特征流形几何曲率驱动的层级化视觉记忆管理解决上述问题：
  (1) **Curvature-Aware Scorer (CAS)**：使用冻结的 DINOv2-small 编码器提取每帧特征 F_t，计算一阶 Motion Variation M_t 和二阶 Geometric Curvature C_t = 1 - cos(d1, d2)（特征位移向量的角度偏差）。C_t 在微分几何视角下等价于 1/2 ||T2 - T1||²（单位切向量变化的平方），即流形曲率的离散近似——当特征演化方向突变时 C_t 急剧增大（语义转换尖峰），恒速运动/平滑相机平移时 C_t ≈ 0（几何惩罚自然抑制物理噪声）。最终 CS_t = M_t + λ·C_t。
  (2) **Hierarchical Visual Memory Management (HVMM)**：使用 EMA 在线更新曲率分数的分布参数 (μ_t, σ_t²)，构建 K-Sigma 动态双阈值 g1/g2。每帧根据 CS_t 动态路由为 Clear Memory（CS_t ≥ g2，保留原始高分辨率）、Blurred Memory（g1 ≤ CS_t < g2，降采样 224×224 作为平滑过渡态）或 Discard（CS_t < g1，丢弃冗余帧），队列以 FIFO 严格控制 |M_t| ≤ N_max = 20。
  (3) **Semantic-decoupled geometric prior**：曲率度量的理论优势在于：C_t 在恒速物理运动中近似为零（免疫平移/旋转噪声），在语义突变时产生显著尖峰（对方向导数而非模长敏感），实现了从低层次物理运动到高层次语义转换的数学解耦。

  对比 baseline 的全栈执行例子（CurveStream + Qwen2.5-VL-7B, N_max=20, StreamingBench）：
  - 算法层：无限长流视频输入 → 每帧经 DINOv2-small 提取 F_t → CAS 计算 CS_t（融合一阶运动和二阶曲率）→ HVMM 在线更新 (μ_t, σ_t²) 并生成自适应双阈值 → 动态路由：高曲率帧（如物体突现/动作翻转）→ Clear Memory（原始高分辨率），中等曲率帧（场景平移中的过渡状态）→ Blurred Memory（224×224），低曲率帧（长时间静态观察）→ Discard → 当 |M_t| > 20 时 FIFO 驱逐最旧 token → 查询时刻 t_q 的帧强制执行 Clear Memory → visual tokens + text query 送入 MLLM decoder → 生成答案。Clear Memory 占比自适应维持在 ~50%（图 3b），既保持关键语义锚点的高保真，又通过 Blurred Memory 的低分辨率过渡保持动作连贯性和因果链完整。最终 StreamingBench accuracy 84.00%（+10.69% over Qwen2.5-VL-7B uniform baseline），OVOBench 73.48%（+13.58%）。
  - 系统框架层：论文未明确说明 Serving 框架
  - 编译框架层：论文未明确说明
  - kernel调度层：论文未明确说明
  - 硬件架构层：单张推理 GPU，论文未明确型号

  解决对应关系：
  | Baseline 缺陷 | CurveStream 解决方案 |
  |---|---|
  | 语义碎片化（低层次物理度量缺乏语义感知） | CAS 二阶曲率度量：C_t 对方向导数敏感而非模长，恒速运动 ≈ 0、语义突变 → 尖峰。Table III: Uniform=69.04%, Cosine Similarity=73.28%, Optical Flow=46.54%, Curvature=77.31%（训练无关方法中最优） |
  | 信息模糊化（无差别压缩不可逆模糊关键语义转换点） | Clear/Blurred/Discard 三层级路由：高曲率帧保留原始高分辨率（Clear Memory），过渡帧低分辨率维持因果连贯（Blurred Memory），冗余帧直接丢弃。图 3b: 自适应 ~50% Clear Memory ratio 在 accuracy 和 token 成本间达到最优 |
  | 延迟感知（post-hoc retrieval 被动滞后） | CAS+HVMM 构成在线主动感知闭环：曲率分数和 K-Sigma 阈值均实时计算和更新，不依赖用户查询触发——帧路由决策在帧到达瞬间完成，实现真正的实时主动语义感知 |
  | 局部噪声敏感（Cosine similarity/optical flow 混淆物理运动和语义突变） | C_t = 1/2 \|\|T2-T1\|\|² 几何等价性（Appendix C 证明）：免疫恒速运动（T1≈T2 → C_t≈0），仅对特征演化方向突变敏感。Table III: Optical Flow 仅 46.54% accuracy（被像素噪声严重干扰），Curvature 达 77.31% |

  组件协同效应（Ablation Table IX/X）：
  - CAS only: +9.12% (StreamingBench), +8.39% (OVOBench) —— 曲率度量提供了精确的语义感知
  - HVMM only: +9.76% (StreamingBench), +4.69% (OVOBench) —— 层级记忆架构在无感知评分时退化为均匀交替分配，但二值 Clear+Blurred 结构本身即提供比 FIFO 更宽的上下文覆盖
  - CurveStream (CAS+HVMM): +12.00% (StreamingBench), +10.66% (OVOBench) —— 组合增益超过各自贡献之和（非线性协同放大），证明感知和调度之间深度互补：CAS 标记高曲率转换点，HVMM 将其锚定为 Clear Memory，同时将低曲率段平滑压缩为 Blurred Memory，共同构建紧凑的因果拓扑链

## Adaptive_Keyframe_Sampling_for_Long_Video_Understanding

- baseline方法是什么？
  Baseline 方法是 uniform keyframe sampling（UNI），即当前主流 MLLM（Qwen2VL、LLaVA-OV、LLaVA-Video）在长视频理解中的默认策略：从输入视频中以固定帧率均匀采样 M 个帧（如 32 或 64 帧），不区分每帧与 prompt 的相关性，不考虑帧在时间轴上的覆盖分布。这是一种使用 dummy VL scorer 的退化 BIN sampling 策略（$s(\mathbf{Q}, \mathbf{F}_t)$ 为常数，所有帧平等对待）。

  Baseline（LLaVA-Video-7B, 64 frames, uniform sampling）全栈执行例子：
  - 算法层：用户上传一段长视频 + 文本 prompt "What is the male protagonist's 5th outfit?" → 视频帧采样器以等间隔从视频中抽取 64 帧（如 600s 视频每 ~9.4s 抽一帧）→ SigLIP Visual Encoder 逐帧编码 → Projector 对齐到 LLM embedding space → 64 组 visual tokens 与 text question tokens 拼接 → LLM (Qwen2-7B) decoder prefill → autoregressive 生成答案 → 随机猜答错误（Figure 1 示例）
  - 系统框架层：基于 HuggingFace Transformers 推理，论文未明确说明 Serving 框架
  - 编译框架层：论文未明确说明
  - kernel调度层：标准 attention 计算，论文未明确说明具体 kernel
  - 硬件架构层：论文未明确说明

  Baseline 的缺陷：
  1. **完全忽略 prompt-frame 相关性**：uniform sampling 的 $KS_M(\cdot)$ 函数不使用 $\mathbf{Q}$ 和 $\mathbf{F}$，无法区分哪些帧包含回答问题所需的信息。如图 1 所示，uniform sampling 可能抽到完全无关的帧（如风景空镜），导致 MLLM 只能随机猜测。
  2. **忽略时间轴上的信息覆盖分布**：当关键信息集中在某个时间段时（如 LongVideoBench 中大量问题聚焦于单一时钟），uniform sampling 在非关键时段浪费宝贵的 keyframe 配额；当问题需要从多个时刻收集信息时（如 VideoMME 中统计某事件发生次数），uniform sampling 可能恰好遗漏那些关键时刻。
  3. **单一策略无法适应不同问题类型的需求**：LongVideoBench 的问题倾向于 single-moment focus（"在某时间点在做什么？"），需要集中高质量帧于关键时段；VideoMME 的问题倾向于 multi-moment comprehension（"发生了多少次？"），需要在时间轴上均匀分布但仍有选择性的帧。Uniform sampling 对所有问题一视同仁。
  4. **缺乏对相邻帧冗余的感知**：均匀采样可能选中多张内容几乎相同的相邻帧，造成视觉 token 冗余，挤占其他时段有用信息的空间。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法：Adaptive Keyframe Sampling (AKS) 将 keyframe 选择形式化为 relevance + coverage 双目标优化，并通过 ADA（Adaptive Sampling）算法自适应平衡两者：
  (1) **Relevance 计算**：使用轻量 VL 模型（默认 BLIP ITM）计算每个候选帧与 prompt 的匹配分数 $s(\mathbf{Q}, \mathbf{F}_t)$，量化每帧对回答问题有用信息的含量。
  (2) **Coverage 估计**：基于 Ripley's K-function 的递归 binning 机制——将时间轴 [0, T) 递归二分为 bin，通过统计每个 bin 内 keyframe 数量分布的不均匀程度定义 coverage penalty，防止冗余相邻帧挤占其他时间段。
  (3) **ADA 分层优化算法**：在每层递归中计算 $s_{\text{top}} - s_{\text{all}}$（Top-M 帧与全帧平均分的差距），若差距超过阈值 $s_{\text{thr}}$，认为帧间区分度足够高 → 直接选 Top-M 高分帧（TOP 模式，最大化 relevance）；否则拆分当前 bin 并均分 keyframe 配额（BIN 模式，最大化 coverage）。这种自适应折中使得 AKS 在 single-moment 问题上表现为 TOP（集中资源于高相关性时刻），在 multi-moment 问题上表现为 BIN（在时间轴上分布覆盖）。

  对比 baseline 的全栈执行例子（AKS + LLaVA-Video-7B, 64 keyframes, 同一 long video from VideoMME）：
  - 算法层：用户上传同一段长视频 + prompt → 以 1 fps 采样全部候选帧 → 逐帧经 BLIP ITM 计算 $s(\mathbf{Q}, \mathbf{F}_t)$（预计算存储）→ ADA(level=0, L=5, s_thr=0.8, M=64): Level-0: s_top-s_all < s_thr → 拆分 [0,T/2)/[T/2,T)，各 32 帧 → Level-1: 对 [T/2,T) 内 s_top-s_all > s_thr → 直接选 Top-32 高分帧（该段包含主人公换装场景）；对 [0,T/2) 继续拆分 → ... → 最终选中 64 帧覆盖视频中的 5 次换装关键时刻 → SigLIP 编码 → Projector → 64 组 visual tokens → LLM 推理 → 生成正确答案 "The male protagonist changed 5 outfits in total"
  - 系统框架层：基于 HuggingFace Transformers，BLIP ITM 预计算 relevance score 在 MLLM 推理之前完成
  - 编译框架层：论文未明确说明
  - kernel调度层：论文未明确说明
  - 硬件架构层：论文未明确说明

  解决对应关系：
  - Baseline 缺陷 1（忽略 prompt-frame 相关性）→ Relevance 计算：BLIP ITM 计算每个候选帧与 prompt 的匹配分数，确保高分帧被优先选中。消融实验：TOP sampling（仅用 relevance，λ=0）在 LongVideoBench 上从 58.9%（UNI）提升至 62.4%（+3.5%），证明 relevance 信号至关重要。
  - Baseline 缺陷 2（忽略 coverage 分布）→ Coverage 估计 + ADA 分配：递归 binning 确保 keyframes 不能全部集中在狭窄时间段。BIN sampling（仅用 coverage，λ→∞）在 VideoMME 上从 64.4%（UNI）提升至 65.2%（+0.8%），证明了 coverage 约束在多时刻问题上的价值。
  - Baseline 缺陷 3（单一策略无法适应不同问题类型）→ ADA 自适应切换：通过 $s_{\text{top}} - s_{\text{all}}$ 和阈值 $s_{\text{thr}}$ 自动在 TOP 和 BIN 模式间切换。在 LongVideoBench 上 ADA=62.7%（> TOP 62.4%，> BIN 60.2%），在 VideoMME 上 ADA=65.3%（> TOP 63.7%，> BIN 65.2%），证明自适应折中在两个互补 benchmark 上同时达到最优。
  - Baseline 缺陷 4（冗余相邻帧）→ Coverage penalty 设计：递归 binning 的 $|m_1 - m_2|$ penalty 隐含地惩罚了相邻帧的过度集中——若大量 keyframe 落入同一个 bin，penalty 增大，促使算法将配额分散到其他 bin。
  - 额外优势（prompt-adaptive）：如图 6 所示，同一段视频针对不同问题，AKS 会选出不同的 keyframe 集合——这使固定的 MLLM 能灵活适应不同场景，而 uniform sampling 对所有问题返回相同的帧集合。

## FOCUS__Efficient_Keyframe_Selection_for_Long_Video_Understanding

- baseline方法是什么？
  Baseline 方法是 (1) uniform sampling（当前 MLLM 默认策略：固定间隔采样 N 帧，完全 query-agnostic）；(2) Top-K keyframe selection——用 Vision-Language encoder（如 BLIP）计算每帧与 query 的 relevance score，选 top-K 最高分帧，但需要预过滤（pre-filtering: downsampling to 1 fps）以控制计算开销；(3) AKS (Tang et al., 2025)——adaptive keyframe sampling，通过 split-and-judge 递归策略平衡 relevance 和 coverage，同样依赖预过滤（1 fps）降低候选帧数。

  Baseline（以 Top-K / AKS 的典型流程为例）全栈执行例子：
  - 算法层：输入一小时长视频 (108K frames @ 30fps) + query → 预过滤阶段（downsampling to 1 fps → 3600 候选帧，丢弃 97% 原始帧）→ 逐候选帧经 BLIP ITM 计算 r_t = cos_sim(e_t, e_q) → Top-K 选最高分 k 帧 / AKS 递归 split-and-judge 分配 → k frames → Vision Encoder + LLM decoder → 答案。预过滤丢弃了大量未评分帧，可能恰好漏掉关键帧。
  - 系统框架层：HuggingFace Transformers，BLIP 逐帧串行或小 batch forward
  - 编译框架层：论文未明确说明
  - kernel调度层：论文未明确说明
  - 硬件架构层：单卡 H100 (80GB)

  Baseline 的缺陷：
  1. **预过滤导致信息丢失**：Top-K 和 AKS 都先 uniform downsample 到 1 fps 才评分——对于一小时视频意味着丢弃 97% 的帧不做评分。关键视觉信息可能恰好存在于被丢弃的帧中（如一个持续仅 0.5 秒的短暂事件），预过滤从根本上违背了 keyframe selection 的目标（从全部帧中选最优帧）。
  2. **全帧评分计算不可行**：若不做预过滤，108K 帧全用 BLIP 评分需约 10^11-10^12 FLOPs（论文估算），对应 AKS w/o pre-filtering 的 255 GPU hours——在实际应用中完全不可行。
  3. **均匀采样无法感知 query 相关性**：Uniform sampling 对所有 query 返回相同帧集合，在 LongVideoBench（问题聚焦于特定场景/事件）上准确率极低（如 Long 视频上仅 51.8% with LLaVA-Video-7B）。
  4. **AKS 需要调节 coverage vs relevance 的超参数（s_thr, L）**：不同 benchmark（LongVideoBench vs VideoMME）需要不同的超参数组合才能达到最优，泛化性受限。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法：FOCUS 通过将 keyframe selection 建模为 Combinatorial Pure-Exploration (CPE) multi-armed bandit 问题，消除预过滤并提供理论保证：
  (1) **Clip-level bandit 建模**：将视频按固定长度（16s）划分为 M 个 clip 作为 bandit arms，目标是选出 top-m 个预期 relevance 最高的 arm。基于视频帧间强时间相关性（Figure 1: median ACF > 0.5 for ~5 seconds）的观察，clip 内相邻帧高度相似，因此只需少量采样即可估计整个 clip 的 relevance。
  (2) **Bernstein confidence radius + 乐观探索**：每个 arm 维护经验均值 μ̂_a 和方差自适应 Bernstein confidence radius β_a(n)，用 optimistic mean μ̃_a = μ̂_a + β_a 指导 arm 选择——既利用高均值 arm 又探索高不确定性 arm。
  (3) **两阶段并行批处理**：将原始串行 bandit 算法（Algorithm 1）简化为两次并行批处理（Algorithm 2）：Stage I 所有 arm 各采样 q 帧 → Stage II 仅对 top α*m optimistic arm 再采样 z 帧 → 选 top-m arm by empirical mean。充分利用 GPU 批处理能力。
  (4) **无需预过滤**：FOCUS 直接对所有 clip 并行采样，不预先丢弃任何帧。通过只对少量帧做 BLIP forward 实现高效探索——仅处理 ~1.6% 的总帧数，5.5 GPU hours。

  对比 baseline 的全栈执行例子（FOCUS + LLaVA-Video-7B, k=64, clip=16s, α=0.25, 同一小时长视频）：
  - 算法层：
    1. 将视频划分为 M 个 16s clip → M 个 bandit arms
    2. Stage I: 每个 arm 并行采样 q 帧 → 批量 BLIP forward 计算 r_t —— 仅需 ~1.0% 总帧数的 BLIP forward
    3. 计算每个 arm 的 μ̂_a, σ̂_a², β_a(n), μ̃_a(n)
    4. A_coarse = TopM(μ̃, α*m) ← 选 optimistic mean 最高的 α*m arms
    5. Stage II: 对 A_coarse 中每个 arm 并行采样 z 帧 → 批量 BLIP forward
    6. A_fine = TopM(μ̂, m) ← 基于无偏经验均值选最终 m 个 arms
    7. 在选中 arms 内 nearest-neighbor 插值 + 概率采样 k_a 帧 → 64 keyframes
    8. Vision Encoder + LLM decoder → 答案
    → LongVideoBench Long videos (>20min) accuracy: 63.7%（vs Uniform 51.8%, +11.9%; vs Top-K 60.5%, +3.2%）
    → GPU hours: 5.5h（vs AKS w/ pre-filtering 9.3h, vs AKS w/o pre-filtering 255h）
  - 系统框架层：HuggingFace Transformers，BLIP 批量 forward 在单卡 H100 上
  - 编译框架层：论文未明确说明
  - kernel调度层：论文未明确说明
  - 硬件架构层：单卡 NVIDIA H100 (80GB)

  解决对应关系：
  | Baseline 缺陷 | FOCUS 解决方案 |
  |---|---|
  | 预过滤导致信息丢失（丢弃 97% 帧） | 无需预过滤：所有 arm（覆盖所有帧）都参与 Stage I coarse exploration，通过只对少量帧做 BLIP forward（~1.6%）实现高效探索。LPU hours 5.5h vs AKS 255h（w/o pre-filtering） |
  | 全帧评分计算不可行（10^11-10^12 FLOPs） | Bandit 采样：实际只有 ~1.6% 帧被 BLIP 评分。Table 3 对比：FOCUS 1.6% frames seen vs AKS 3.7%（w/ pre-filtering）vs 100%（w/o pre-filtering） |
  | Uniform 忽略 query 相关性 | Bandit arms 的 reward 基于 BLIP 计算的 query-frame cosine similarity，选出 query-relevant top-m arms。LongVideoBench Long: +11.9% over uniform |
  | AKS 需要 tune coverage 超参数 | Bandit 框架的探索机制（UCB/confidence radius）自动平衡 exploration 和 exploitation，无需 coverage 约束。α 虽可调但对 accuracy 影响小（0.1-0.5 范围内 accuracy 62.9-63.6%） |

  核心技术贡献：
  - **Bernstein confidence radius 的方差自适应探索**：比标准 UCB 更鲁棒。Table 8 消融：FOCUS-M（仅用经验均值）62.3/58.1/63.0 vs FOCUS（加 Bernstein）63.5/60.7/63.5（LLaVA-Video/Qwen2-VL/LLaVA-OV）
  - **理论保证**：Bernstein confidence bound 保证 |μ̂_a - μ_a| ≤ β_a 以 ≥ 1-6/n 概率成立（Theorem B.1）；Algorithm 2 以 ≥ 1-6(M-m)/n 概率返回 oracle top-m set（Theorem C.1）
  - **Two-stage 批处理设计**：FOCUS-C（仅 coarse）61.7/58.4/62.3, FOCUS-F（仅 fine）61.5/57.7/62.5, FOCUS（两阶段）62.3/60.7/63.5 — 两阶段互补，coarse localization + fine exploitation

## GroundVTS__Visual_Token_Sampling_in_Multimodal_Large_Language_Models_for_Video_Temporal_Grounding

- baseline方法是什么？
  Baseline 方法是现有的 Vid-LLM 使用 **uniform frame sampling** 的视频处理方法：给定视频后，以固定帧率（如 2 FPS）均匀采样帧，每帧经 Vision Encoder 编码为 visual tokens，经 Projector 映射后全部送入 LLM 进行时序推理和定位。这种方法将所有 visual token 平等对待，不区分其与查询的相关性。部分改进方法（如 KeyVideoLLM、VideoTree）在视频输入端引入 query-guided frame selection（基于 CLIP 等外部编码器计算跨模态相似度选帧），但仅是粗粒度帧级过滤且依赖外部编码器。

  Baseline（Qwen2.5VL-7B, uniform frame sampling @ 2 FPS, Charades-STA）全栈执行例子：
  - 算法层：输入一段视频 + 查询 "a person takes a book off a shelf" → 以固定 2 FPS 均匀采样所有帧 → Vision Encoder 逐帧编码为 visual tokens → Multimodal Projector 映射到 LLM embedding space → **所有 visual tokens 不加区分地** 拼接 text query embeddings → LLM 28 层 causal self-attention（每层对所有 tokens 做 full attention）→ 自回归生成时间边界预测 → "from 4.5s to 10.3s"（与 ground truth 6.2-12.0s 偏移）。关键帧（0-13s 书架上取书的动作）与非关键帧（13s 后的无关内容）获得相同的 token 配额和注意力计算。
  - 系统框架层：基于 HuggingFace Transformers 推理，论文未明确说明 Serving 框架
  - 编译框架层：论文未明确说明
  - kernel调度层：标准 FlashAttention，论文未明确说明
  - 硬件架构层：GPU 推理（具体型号论文未明确说明）

  Baseline 的缺陷：
  1. **均匀采样稀释关键时刻信息**：Figure 1(a) 和 Figure 2 证明，uniform frame sampling 对每个时间段分配相同的 visual token 配额，当查询相关事件稀疏时，关键帧可能被大量无关帧稀释甚至漏掉。在 Charades-STA 上，frame rate 从 0.2 FPS 增至 2.4 FPS 时 mIoU 先升后降（峰值 47.8%），继续增至 3.0 FPS 时 mIoU 急剧下降——说明冗余 visual token 稀释了关键时序信号。
  2. **粗粒度帧级选择精度不足**：Figure 1(b) 中的方法（KeyVideoLLM、VideoTree）在视频输入端基于外部编码器做帧级筛选，但仅能粗粒度选帧而无法细粒度区分帧内哪些空间位置（token）与查询相关。此外依赖外部编码器增加了计算开销和解耦误差。
  3. **Visual token 冗余导致注意力分散**：LLM 的 self-attention 对所有 visual token 做全局计算，大量无关 token 不仅浪费计算，还可能在 attention 中引入噪声，干扰模型对关键时序边界的判断。
  4. **非均匀 token 分布下 LLM 无法有效适应**：即使强行用 query-guided 方式产生非均匀 visual token 分布，预训练 LLM 在均匀分布上训练的注意力机制难以直接适应这种分布偏移，导致训练不稳定和性能退化。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法：GroundVTS 通过以下设计解决：

  (1) **Visual Token Sampling (VTS) 模块** —— 在 visual encoder + projector **之后**、LLM **之前**插入 query-guided token 采样，实现 **token 级别**的细粒度选择。Token Scoring: w = softmax(W_v V · W_q Pool(Q)^T / τ)，计算每个 visual token 与 query 的语义相关性。Differentiable Top-K Selection: 通过 Gumbel-Softmax + STE 实现可微分的 top-K 选择（forward 用 hard mask, backward 通过 soft relaxation 传播梯度），使 VTS 可端到端训练。

  (2) **非均匀 token 分布 + 位置编码保留** —— VTS 输出的 \tilde{V} 在时间维度上非均匀分布：高 query 相关性区域 token 密度高，低相关性区域 token 稀疏或为零，但保留 dense sampling 时的原始位置编码（mask 未选中 token 的位置），确保 selected tokens 的时间位置信息不变。

  (3) **三阶段渐进式优化** —— Stage 1 (VTS Warm-up): 冻结 LLM 训练 VTS，使其学习稳定的 token-query 相关性估计；Stage 2 (Joint LoRA Adaptation): LoRA 微调 LLM 使模型适应非均匀 token 分布（关键设计——直接用非均匀 token 微调 LLM 会因分布偏移导致训练不稳定）；Stage 3 (Grounding Fine-tuning): 在 Grounding-FT 数据集上精细微调 VTG 能力。

  对比 baseline 的全栈执行例子（GroundVTS-Q, ρ=0.5, 2 FPS, 同一视频 + 同一查询）：
  - 算法层：输入同一段视频 + 查询 "a person takes a book off a shelf" → Vision Encoder 编码全部帧 → Projector 映射 → **VTS 模块**: W_v 投影 visual tokens → W_q 投影 query → softmax 计算 每个 token 的 query-relevance w → Gumbel-Softmax + STE 选择 top 50% (ρ=0.5) 的 visual tokens → 重要区域（0-13s, 书架取书动作）token 密度高形成峰值，无关区域（13s 后）token 几乎全被抑制 → MLP 重编码 + 重归一化权重 → 保留原始位置编码 → LLM 在稀疏但高相关性的 token 序列上推理 → "from 6.0s to 12.0s"（与 ground truth 6.2-12.0s 高度吻合, mIoU 50.1 vs baseline QwenVL-G 31.7 = +18.4）
  - 系统框架层：基于 HuggingFace Transformers 推理，论文未明确说明 Serving 框架
  - 编译框架层：论文未明确说明
  - kernel调度层：标准 FlashAttention，VTS 的 Gumbel-Softmax + STE 为纯 PyTorch 操作，无自定义 kernel
  - 硬件架构层：GPU 训练/推理（具体型号论文未明确说明）

  解决对应关系：
  | Baseline 缺陷 | GroundVTS 解决方案 |
  |---|---|
  | 均匀采样稀释关键信息 | VTS token-level query-guided 采样：仅在 query 相关的 spatio-temporal 区域保留高密度 token (ρ=0.5)，无关区域 token 被抑制。Figure 4: GroundVTS 在极低 token density (FPS×ρ=0.4) 时仍达 R1@0.7=29.2，远超 baseline QwenVL-G (10.2) |
  | 粗粒度帧级选择精度不足 | Token-level 细粒度选择：不是选帧而是选 token，可区分同一帧内不同空间位置的 query 相关性。Table 5: Token-Level VTS mIoU 50.1 vs Frame-Level 41.6 (+8.5 on Charades-STA) |
  | Visual token 冗余分散注意力 | ρ=0.5 即保留 50% 视觉 token，以一半的 token 预算超越全量 baseline (R1@0.7: 34.2 vs 30.5)。Figure 4(b) 展示 token efficiency 大幅领先 |
  | LLM 无法适应非均匀分布 | Stage 2 Joint LoRA Adaptation 在 LLaVA-Video-178K 上让 LLM 学习解释 query-guided 非均匀 token 序列。Table 4 消融：去掉 Stage 2 (仅 1+3) 后 R1@0.7 从 34.2 降至 15.2 |
  | 训练不稳定 | Stage 1 VTS Warm-up 先单独训练 VTS 学习稳定采样行为，Table 4: 跳过 Stage 1 (仅 2+3) 仍有 30.5 R1@0.7，但加 Stage 1 升至 34.2 |

  跨架构泛化验证（GroundVTS-I on InternVL3.5-8B）：
  - InternVL3.5 使用 fixed-number frame sampling（非 fixed-rate），VTS 仍持续提升：Charades-STA R1@0.7 +3.5, QVHighlights mAP +20.6, Hit@1 +48.6
  - 证明 VTS 的 token-level sampling 与底层 frame sampling 策略解耦，对不同 Vid-LLM 架构通用

  关键创新点总结：
  - **首次在 Vid-LLM pipeline 内部做 token-level query-guided sampling**（而非输入端 frame selection），实现细粒度 spatio-temporal 注意力分配
  - **Gumbel-Softmax + STE 使离散 token 选择可端到端训练**，无需外部 reward model 或 RL
  - **三阶段渐进式优化解决分布偏移问题**：warm-up → joint adaptation → task fine-tuning，消融实验证明每个阶段不可或缺
  - **Token-level > Frame-level**: 同一帧内可区分空间位置与 query 的相关性，比帧级选择精度高 8.5 mIoU

## HORNet__Task-Guided_Frame_Selection_for_Video_Question_Answering_with_Vision-Language_Models

- baseline方法是什么？
  Baseline 方法是现有的 frame-sampling 策略用于 VideoQA with VLMs，主要包括两类：(1) **Uniform Sampling** —— 以固定间隔从视频中均匀采样 T 帧（如 T=32），每帧经 Vision Encoder 编码为 visual tokens 后送入 VLM（Qwen3-VL）进行 QA 推理。这是最广泛使用的 baseline，完全不考虑帧内容与问题的相关性；(2) **Learned Selection 方法** —— SeViLA（chain Localizer+Answerer fine-tuned from BLIP-2, pseudo-label self-refinement）、Frame-Voyager（enumerate frame combinations + supervised selector via prediction loss ranking）、ReFoCUS（autoregressive frame selector via VLM confidence margin rewards）、ViaRL（co-evolve selector+answerer via iterated amplification RL）。这些方法或需微调 VLM（SeViLA, Frame-Voyager, ViaRL），或需修改 VLM 架构（ReFoCUS 部分微调），参数效率低且不通用。

  Baseline（Uniform Sampling + Qwen3-VL-2B, MSVD-QA）全栈执行例子：
  - 算法层：输入视频（~10s, ~300 frames @ 30fps） + 问题 → 均匀采样 T=32 帧（fps=2, 约覆盖 1/12 原始帧） → Qwen3-VL Vision Encoder 逐帧编码为 visual tokens → Projector 映射 → 拼接 text tokens → LLM 28 层 prefill + decode → 生成答案。32 帧中大量冗余帧（静态背景、重复动作）与关键帧（动作瞬间、物体交互）被等同处理，关键帧可能落在采样间隔之间被跳过 → F1-Lev = 0.3483, Qwen Proc. 0.28s, Avg. Frames=11.65（实际均匀采样后送入 Qwen 的帧数可能因内部处理而异）。
  - 系统框架层：基于 HuggingFace Transformers 推理，论文未明确说明 Serving 框架
  - 编译框架层：论文未明确说明
  - kernel调度层：标准 FlashAttention，论文未明确说明
  - 硬件架构层：单张 NVIDIA A100 40GB GPU

  Baseline 的缺陷：
  1. **Uniform sampling 无法优化下游回答质量**：固定间隔采样完全忽略帧内容与问题的相关性——对于"the boy on the green disc goes down"这样的时序问题，关键交互瞬间可能仅占 32 帧中的 2-3 帧，且不一定落在均匀采样点。论文 Table 5 显示 NExT-QA 上 uniform 仅 64.24% vs HORNet 71.50%（+7.3 points），证明对于需要时序/因果推理的长视频，盲采样严重不足。
  2. **冗余帧注入噪声、增加计算**：MSVD（~10s 视频）中均匀采样的 11.65 帧大多冗余——HORNet 仅用 4 帧就达到 +1.7% F1 提升，证明大多数帧是噪声而非信号。MSRVTT（~15.5s）47.52 帧 → 4 帧，NExT-QA（~43.7s）1157.88 帧 → 8 帧，冗余比例极高。
  3. **现有 learned selection 方法参数效率低**：SeViLA fine-tunes BLIP-2（~1B+ params），Frame-Voyager 需 combinatorial enumeration，ReFoCUS 需修改 VLM——这些方法在 small-data 场景（数据稀缺、标注昂贵）下不可行，也无法在不同 VLM 间 transfer。
  4. **GRPO 此前仅用于优化 VLM 输出，未探索优化 VLM 输入**：Video-R1、R1-VL、DeepVideo-R1 等均用 GRPO 优化 VLM 生成分布（outputs），但没有人用 GRPO 优化"VLM 看到什么"（inputs）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法：HORNet 首次将 GRPO 从优化 VLM 输出重定向到优化 VLM 输入，通过三项核心设计解决 baseline 缺陷：

  (1) **Select Any Frames (SAF) 问题形式化** → 将帧选择形式化为 RL 问题：学习参数化策略 π_θ(V,q) → 二进制选择 mask b ∈ {0,1}^T，最大化 E[R(M(b⊙V, q), a)]。SAF 无时序顺序或连续性约束——策略可自由选择时间上稀疏的关键事件、短关键片段或密集运动段，完全取决于什么能最大化任务驱动的 reward。

  (2) **GRPO-trained lightweight policy (<1M params)** → 解决参数效率。TimeSFormer-Tiny encoder 提取 per-frame spatiotemporal features → MLP (768→512→256→1, GELU, sigmoid) 输出 per-frame keep probability → 每步生成 K=8 个候选 mask（top-k sweep + stochastic Bernoulli） → frozen Qwen3-VL 回答 → 计算 reward (0.1·F1 + 0.9·EditSim) → GRPO 更新（group-normalized advantage, critic-free）。仅 MLP + encoder 可训练，VLM 始终冻结，参数 <1M。

  (3) **GRPO redirected from outputs to inputs** → 概念创新。传统 GRPO（DeepSeek-R1, Video-R1 等）优化 π_θ(a|V,q) 即生成分布；HORNet 优化 π_θ(b|V,q) 即选择分布。这一概念转换使方法更参数高效（不需修改 VLM）、更通用（policy 可 transfer 到不同 VLM answerer）。

  对比 baseline 的全栈执行例子（HORNet + Qwen3-VL-2B, MSVD-QA, 同一视频 + 问题）：
  - 算法层：输入同一视频 + 问题 → 均匀采样 T=32 帧 → TimeSFormer-Tiny 提取每帧 spatiotemporal 特征 F ∈ R^{32×768}（spatial self-attention per frame → temporal self-attention across frames → spatial avg pool） → MLP Policy 输出 p = [p_1,...,p_32] ∈ (0,1)^32 → 排序选 top-4 概率帧 → 仅 4 帧送入 frozen Qwen3-VL-2B → 生成答案。32→4 帧压缩 87.5%，且 F1-Lev 从 0.3483 提升到 0.3543 (+1.7%)。
  - 系统框架层：基于 HuggingFace Transformers，Qwen3-VL-2B 全程冻结，仅 MLP + TimeSFormer-Tiny 训练
  - 编译框架层：论文未明确说明
  - kernel调度层：标准 FlashAttention（TimeSFormer 的 spatial-temporal factorized attention 仍与 FlashAttention 兼容）
  - 硬件架构层：单张 NVIDIA A100 40GB GPU，训练两阶段：Stage 1 (MSVD+MSRVTT, F1-Lev reward) → Stage 2 (NExT-QA, MCQ accuracy reward)

  解决对应关系：
  | Baseline 缺陷 | HORNet 解决方案 |
  |---|---|
  | Uniform sampling 无法优化回答质量 | SAF + GRPO: task-grounded reward signal（VLM QA accuracy）直接驱动帧选择策略优化。NExT-QA 上 HORNet 71.50% vs Uniform 64.24% (+7.3), 证明 GRPO 学到了时序关键帧的选择 |
  | 冗余帧注入噪声 | MLP policy 学会丢弃噪声帧：MSVD 11.65→4 帧 +1.7% F1, MSRVTT 47.52→4 帧 -5.6% F1（可控 trade-off）。短视频上甚至提升质量（丢弃噪声聚焦关键内容） |
  | 现有 learned selection 参数效率低 | <1M trainable params, VLM frozen: 比 SeViLA（fine-tune BLIP-2）、Frame-Voyager（combinatorial）、ViaRL（co-evolve）参数效率高 3 个数量级。适合 small-data/资源受限场景 |
  | GRPO 仅用于优化 VLM 输出 | 首次将 GRPO 从 output optimization 重定向到 input optimization。这一"概念转换"被论文 Table 4 验证：GRPO OOD generalization (MSRVTT 0.3029) 优于 PPO (0.2948) 和 SFT (0.2882)，证明 group-relative advantage 学到的选择策略更可迁移 |
  | 策略难以跨 VLM transfer | 同一 HORNet policy 换 Qwen2.5-VL-3B 后 F1-Lev 从 0.3543 提升到 0.3846 (+8.5% relative)，无需任何 retraining。Policy 与 answerer 解耦 |

  训练策略的关键设计：
  - **两阶段训练**：Stage 1 用短视频+one-word answer 学习"有用帧识别"（F1-Lev reward），Stage 2 用长视频+MCQ 学习"因果/时序推理所需帧选择"（accuracy reward）
  - **Candidate generation 策略**：K=8 = 7 top-k sweep (deterministic exploitation) + 1 Bernoulli (stochastic exploration)，平衡探索与利用
  - **Reward 设计**：0.1·F1_token + 0.9·EditSim (lemmatized)，比 exact match 对 minor lexical variations 更鲁棒
  - **GRPO 优势**：Table 4 消融证明 GRPO 的 OOD generalization 优于 PPO 和 SFT——group-relative advantage estimation 学到更可迁移的选择策略，而 PPO/SFT 更容易 overfit 到训练分布

## HiPrune__Training-Free_Visual_Token_Pruning_via_Hierarchical_Attention_in_Vision-Language_Models

- baseline方法是什么？
  现有 VLM visual token 剪枝方法主要基于两种思路：(1) LLM 内部剪枝——FastV 在 LLM decoder 前面几层丢弃低注意力 token，PyramidDrop 逐层金字塔式减少 token；(2) 静态度量选择——VisionZip 基于 token 相似度/diversity 选择保留 token，DivPrune 基于 diversity 选择。这些方法的共同缺陷是：未充分利用视觉编码器本身的内在注意力结构，部分方法依赖 CLS token（SigLIP 等无 CLS token 的编码器无法使用），且大多需要针对不同模型精心调参。

  Baseline 全栈执行例子：
  - 算法层：FastV → ViT 编码 576 tokens → projector → LLM 第 2 层后丢弃 attention score 低的 visual token → 后续解码层仅处理保留 token
  - 系统框架层：HuggingFace Transformers + LMMs-Eval 评估
  - 编译框架层：论文未明确说明
  - kernel调度层：标准 PyTorch attention（或 FlashAttention），token 丢弃发生在 LLM decoder 内部
  - 硬件架构层：NVIDIA A100 40GB，单卡推理

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 HiPrune，基于视觉编码器 ViT 的**分层注意力模式（Hierarchical Attention Pattern）**进行 training-free token 剪枝。核心发现：(1) 中间层注意力集中在图像 main object（IoU 验证：中间层 top-10% token 与 COCO segmentation mask 重叠度最高）；(2) 深层注意力均匀分布，编码全局信息。基于此设计三种 token 类型：
  - **Anchor Tokens**：从中间 object layer l 选 attention score 最高的 token，保留 object-centric 细节
  - **Buffer Tokens**：Anchor 的空间邻居（上下左右），抗注意力噪声
  - **Register Tokens**：从 ViT 输出层选高注意力 token，补充全局上下文信息
  - **HiPrune++ 可选**：额外保留与 text embedding 余弦相似度高的 token，增强指令跟随

  论文方法全栈执行例子：
  - 算法层：Image → ViT CLIP-L/14 (24 layers) → 在第 9 层提取 `mid_attn` (object layer attention) → topk 选 Anchor + spatial neighbor Buffer → 在第 24 层（输出层）提取 `deep_attn` → 补充 Register → 可选 text cosine similarity 补充 (HiPrune++) → 576 tokens → N' (如 192) tokens → projector → LLM → 生成
  - 系统框架层：HuggingFace Transformers + LMMs-Eval + calflops
  - 编译框架层：论文未明确说明
  - kernel调度层：HiPrune 在 ViT 输出后、projector 前执行，与 FlashAttention 完全兼容（不修改 attention 内部计算）
  - 硬件架构层：NVIDIA A100-PCIE 40GB，部分实验 RTX 5090

  解决对应关系：
  | Baseline 缺陷 | HiPrune 解决方案 |
  |---|---|
  | 未利用 ViT 内部注意力结构（仅用 LLM 内部 attention 或静态相似度指标） | 从 ViT 中间层（object-centric）和输出层（global）分层提取注意力信号，无需 training 或 external guidance。Table 1：中间层 top-10% token IoU 0.80×~1×（CLIP-L/SigLIP/DeiT 等均适用），证明模式跨编码器通用 |
  | 依赖 CLS token 的方法（SparseVLM 等）无法用于 SigLIP | 使用 global mean attention（Eq.3: a_i = mean_head sum_n A[h,n,i]），不依赖 CLS token，Qwen2.5-VL (SigLIP) 上 SOTA (Table 4: 11.1% tokens 保持 93.0%) |
  | 需要额外 training/merging（ToMe、PuMer） | 完全 training-free、plug-and-play：仅需设置 object layer l 和 α=0.1。LLaVA-1.5 上 l=9/LLaVA-NeXT 上 l=9/Qwen 上 l=16，通过 dispersion-based searching 自动确定 |
  | Token merging 与 FlashAttention 不兼容 | HiPrune 是纯 pruning（select tokens by index），不修改 attention 计算，与 FlashAttention 完全兼容 |
  | 低 token budget 下指令跟随能力严重退化 | HiPrune++ 通过 text cosine similarity 补充 β=0.1 token 缓解：LLaVA-1.5 64 tokens (11.1%), HiPrune 92.7% vs HiPrune++ 96.1%, POPE 73.0% vs 84.3%，证明 text guidance 在低 budget 下关键 |
  | 高分辨率场景（LLaVA-NeXT 2880 tokens）压缩效率差 | 保留 2/9 tokens (640) 保持 99.7% (HiPrune++), 甚至 5.6% tokens (160) 仍保持 94.4% (HiPrune++) |

  消融实验关键发现：
  - Token types: 去掉 Register → 性能下降最显著 (Table 6b: w/o Register Avg 97.9%)，证明全局信息最关键

## Investigating_Video_Reasoning_Capability_of_Large_Language_Models_with_Tropes_in_Movies

- baseline方法是什么？
  Baseline 方法是现有的 LLM-based Video Reasoning 三类范式：(1) **Captioner-Reasoner (LLoVi)** —— VLM (BLIP-2) 将视频帧 tokenize 为文本 caption，LLM 做多轮摘要压缩后做二分类判断；(2) **LMM Instruction Fine-tuning (SeViLA, LLaMA-VID)** —— 通过 projection layer 将视觉特征对齐到 LLM token space，SeViLA 用 localizer 选 16 帧（从 120 帧中），LLaMA-VID 将每帧压缩为 2 tokens 处理长视频；(3) **Visual Programming (ViperGPT)** —— LLM (GPT-4) 生成 Python 代码调用 VLM API 做逐步推理，但原始设计中缺乏角色识别工具（仅有通用 "person" 检测），且将 NExT-QA 式的简单 temporal localization 策略直接用于电影叙事，无法处理复杂 trope 定义。

  Baseline (ViperGPT + BLIP-2 VLM + GPT-4 code generator, 16 frames, TiM Mainset) 全栈执行例子：
  - 算法层：输入电影片段 (16 帧 via SeViLA keyframe selector) + trope query "Is the trope Big Bad present?" + trope definition → GPT-4 生成 Python 程序：for frame in frames: person = frame.find("person") → action = frame.simple_query("What is this person doing?") → 逐帧收集 actions → video_segment.select_answer(info, query) → 输出 {True/False}。整个过程仅对通用 "person" 对象做查询，无法将不同帧中的同一人物关联（无 face_identify），也无法将 trope 的抽象定义（如 "direct cause of all bad happenings"）分解为具体可检验的子问题。TiM Mainset(V+D) F1=20.98 (ViperGPT 16 frames)。
  - 系统框架层：ViperGPT Python 执行引擎 + BLIP-2/GPT-4 API 调用
  - 编译框架层：论文未明确说明
  - kernel调度层：论文未明确说明
  - 硬件架构层：论文未明确说明

  Baseline 的缺陷（在 TiM 数据集暴露的 Abstract Perception 和 Long-range Compositional Reasoning 挑战）：
  1. **Abstract Perception 缺失**：现有方法擅长感知 concrete 元素（动作、物体、属性），但在抽象概念（情感、动机、幽默、判断）上严重不足。ViperGPT 原始设计仅有通用 "person" 检测，无法识别电影中数十个不同角色的身份和交互——而这些角色交互是理解 "Big Bad" 等 trope 的核心。ViperGPT F1=24.39 (120 frames, V+D)，显著低于 FEVoRI 的 32.79。
  2. **Long-range Compositional Reasoning 不足**：电影可长达数小时、数千帧，trope 查询需分解为多个相互依赖的嵌套子查询（如判断 "Big Bad" → 需先识别负面事件 → 再归因到具体角色 → 再验证一致性）。ViperGPT 的简单 NExT-QA 式 prompt（直接 temporal localization + VLM query）无法处理这种多层嵌套推理。在 TiM 上，LLoVi 甚至低于 random baseline (F1=18.97 vs 19.54)，SeViLA/LMM-IF 倾向于盲目猜 "yes"（高 recall 但极低 precision），ViperGPT 虽有较好 precision 但 recall 严重不足。
  3. **Context 与 Query 未解耦**：ViperGPT 将电影上下文和 trope 查询混在一起推理，导致 LLM 在 program generation 时难以同时处理冗长的叙事细节和复杂的 trope 定义。GPT-4 与 GPT-3.5 在 program generation 上仅有 0.17 F1 差异，说明瓶颈不在 code generation 能力，而在 task decomposition 策略。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法：(1) **FEVoRI** → 解决 Abstract Perception。在 ViperGPT API 中集成 DeepFace 人脸识别工具 (`face_identify`)，通过 ICL example 引导 LLM 生成包含角色跟踪的 Python 程序——逐帧将检测到的 "person" 映射到具体角色 ID，累积同一角色的行为和其因果效应，将抽象的 trope 概念（如 "Big Bad"）映射为可检验的角色行为链。(2) **ConQueR** → 解决 Long-range Compositional Reasoning。系统性地将 context 和 query 解耦并渐进分解：先将 trope 定义拆分为多个可检验维度 → 逐帧提取人物/动作/事件 → 将事件与 trope 各维度逐一匹配 → 最终汇聚判断。(3) **ABCD** → 量化验证。通过 VP 生成代码的 AST 节点数/边数和 VLM call/Token 数量化数据集的 Abstract Perception 和 Long-range Compositional Reasoning 水平。

  对比 baseline 的全栈执行例子（FEVoRI+ConQueR, ViperGPT + DeepFace + GPT-4 + BLIP-2, 120 frames, TiM Mainset V+D）：
  - 算法层：输入电影 120 帧 + trope query "Is the trope Big Bad present?" + trope definition → GPT-4 生成 Python 程序（含 FEVoRI+ConQueR ICL example 引导的推理模板）：
    1. **Character Identification**（FEVoRI）：逐帧 `frame.find("person")` 检测人物 → `video_segment.face_identify(character)` (DeepFace) 分配唯一角色 ID → 查询角色外观描述 `person.simple_query("Describe appearance in 10 words")` → 记录到 `character_infos[person_id]`
    2. **Action Tracking**（FEVoRI + ConQueR）：对每个角色查询 `person.simple_query("Describe action in the scene")` → 记录到 `character_infos[pid]["actions"]`
    3. **Negative Event Detection**（ConQueR解耦）：`frame.simple_query("Is there any negative event in the scene?", to_yesno=True)` → 如果是，`frame.simple_query("What's happening in the scene")` 提取 event 描述
    4. **Progressive Matching**（ConQueR核心）：对每个负面事件，遍历所有角色信息，逐一匹配：
       - `person_query = f"Is person '{character_description}' a potential cause of '{event}'?"` → 匹配人物
       - `action_query = f"Is action '{prev_action}' a potential cause of '{event}'?"` → 匹配动作
       若任一匹配，将该角色标记为 potential_cause
    5. **Global Aggregation**：`video_segment.select_answer(info, query, possible_answers)` 汇聚所有帧的角色-事件因果链，判断 trope 是否存在 → F1=39.64 (FEVoRI+ConQueR) vs 20.98 (ViperGPT baseline) = +18.66 F1
  - 系统框架层：ViperGPT Python 执行引擎 + BLIP-2/Gemini VLM API + DeepFace 人脸识别 + GPT-4 code generation
  - 编译框架层：论文未明确说明
  - kernel调度层：论文未明确说明
  - 硬件架构层：论文未明确说明

  解决对应关系：
  | Baseline 缺陷 | 论文方法 | 效果 |
  |---|---|---|
  | Abstract Perception 缺失：仅通用"person"检测，无法识别角色身份和交互 | FEVoRI: 集成 DeepFace face_identify + ICL example 引导角色跟踪，将抽象 trope 概念映射为可检验的角色行为链 | FEVoRI +8.5 F1 over ViperGPT (24.39→32.79)，CT 类别 +7.61 F1, RI 类别 +9.93 F1 |
  | Long-range Compositional Reasoning 不足：简单 temporal localization 无法处理多层嵌套推理 | ConQueR: context/query 解耦 + 渐进维度分解 + 逐帧角色-事件匹配，将复杂 trope 定义拆解为可独立验证的子问题 | ConQueR +6.9 F1 over FEVoRI (32.79→39.64)，recall +11.48 |
  | Context 与 Query 未解耦：LLM 难以同时处理冗长叙事和复杂 trope 定义 | ConQueR 渐进推理流水线：先提取角色信息 → 再检测事件 → 再逐维度匹配 → 最后汇聚，将复杂推理拆分为管道式小步推理 | ConQueR AST Nodes +18.6, AST Edges +27.1 vs baseline TiM (Table 4)，ABCD 定量证实推理复杂度提升 |
  | 全局性能仍远低于人类 (65 F1) | 即使 FEVoRI+ConQueR 仅达 40 F1 vs Human 65 F1，但揭示了未来方向：更强的 VLM (Gemini 4.5 F1 gain)、更高帧率 (everyshot +1.5 F1)、更复杂 program generation
  - Buffer: 不用 Buffer → 99.7% (Table 6b: w/o Buffer)，去掉 Buffer+Anchor → 99.7%，证明 Register 可单独支撑，但 Anchor+Buffer 精确保留细节
  - Buffer scheme: Cross(4)/Square(8)/Row(2) 差异微小 (Table 6c: all ~100%)，只要 buffer 覆盖足够即可
  - Attention pattern: Global mean attention 略优于 CLS token (Table 6a)，且 universal（不依赖 CLS token 存在与否）

## LiveStar__Live_Streaming_Assistant_for_Real-World_Online_Video_Understanding

- baseline方法是什么？
  Baseline 方法是基于 EOS（End-Of-Sequence）prediction 的在线视频理解方法（VideoLLM-online, VideoLLM-MoD, LION-FS）。核心思路：在流式视频推理中，模型对每个 incoming frame 进行前向传播，若当前帧不需要输出则生成 EOS token 表示沉默，若需要输出则生成正常字幕。训练目标为 `max P(EOS | [Ctx^{<t_i}], [Frm^{t_i}])`（非响应帧）或 `max P([Txt] | [Ctx], [Frm])`（响应帧）。

  Baseline（VideoLLM-online, Ego4D Narration Stream, 3 fps）全栈执行例子：
  - 算法层：视频帧流 → Vision Encoder 逐帧编码 → 拼接文本 prompt → LLM 前向传播 → 每帧输出 token（EOS=沉默 或 字幕token=响应）→ streaming EOS prediction 决定何时输出。关键特征：每个帧都需要完整 decoding step 生成至少 1 个 token（EOS 或 response token），EOS token 作为普通 vocabulary token 与正常文本 token 竞争。
  - 系统框架层：PyTorch + HuggingFace Transformers，continuous KV cache 保持历史上下文
  - 编译框架层：论文未明确说明
  - kernel调度层：标准 FlashAttention，论文未明确说明
  - 硬件架构层：NVIDIA A800 GPU 训练和推理

  Baseline 的缺陷：
  1. **Response-Silence Imbalance（响应-沉默失衡）**：EOS 输出帧远超正常响应帧。例如 1 分钟视频 @3fps 含 5 个响应区间，响应:沉默 = 1:35，EOS 成为最主要的预测目标，模型过度偏向沉默输出。
  2. **Consecutive Frame Inconsistency（连续帧不一致）**：相邻视觉相似帧产生冲突输出——一帧输出完整叙述而相邻帧仅输出 EOS，这种不一致在微调时破坏模型收敛。
  3. **Pre-training Misalignment（预训练失配）**：预训练阶段对齐 image-text pairs（视觉→有意义的语言），而 EOS-based 训练要求部分帧映射到 EOS token，这与预训练目标（始终产生有意义的视觉-语言对应）直接矛盾。
  4. **Vocabulary Confusion（词表混淆）**：EOS token 作为普通 vocabulary token 频繁出现在响应中污染语义连贯性，引入歧义并与正常输出冲突。
  5. **全部帧都需要完整解码**：即使 silent frame 也需要至少生成 1 个 token（EOS），推理效率低。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法：LiveStar 通过 streaming response-silence 范式解决 EOS-based 的所有缺陷：

  (1) **SCAM (Streaming Causal Attention Masks) 训练 → 解决 Pre-training Misalignment + Consecutive Frame Inconsistency**。不再依赖 EOS token 标记沉默，而是使用交错帧-字幕序列格式。对每个语义片段 C_k = {t_m, ..., t_n}，所有帧共享相同语义字幕 [Cap^k]，训练目标变为 `max P([Cap_i^k] | [Ctx {Mask}], [Frm^{t_i}])`。SCAM Mask 阻止对同一语义片段中已生成字幕的注意（防止 trivial copying），但保留前一语义片段终端字幕的可见性（传递场景边界信息）。这保持了标准 multimodal pre-training 的一致性（每个视频帧始终对齐有意义的语言内容）。

  (2) **SVeD (Streaming Verification Decoding) → 解决 Response-Silence Imbalance + Vocabulary Confusion + 全部帧解码开销**。彻底移除 EOS token 依赖！SVeD 不通过 vocabulary token 决定沉默/响应，而是通过 perplexity 变化检测语义边界：每帧仅需单次 forward pass 计算 PPL([Dec])，若 PPL 变化超过 α·threshold 则激活解码 gate 生成新字幕，否则保持沉默。关键优势：(a) 沉默帧无需 decoding —— 仅需计算 PPL（比 decode 快得多）；(b) 无 EOS token 污染 —— Dec 是真实语义内容，不存在 EOS 导致的词表混淆；(c) 响应-沉默决策基于语义变化（PPL 增加）而非 supervised EOS 分类。

  (3) **Peak-End Memory Compression → 解决长视频 OOM**。受认知科学中人类记忆优先保留"峰值"（关键帧）和"终点"（最近事件）的规律启发，利用 SVeD 预计算的 PPL 作为帧重要性评分，以概率方式剪枝低重要性旧帧，配合终端字幕摘要，将长视频上下文压缩到可控范围内。

  (4) **Streaming KV Cache → 解决推理效率**。双级缓存（intra-dialogue + inter-dialogue）消除历史帧的重计算，在 SVeD swap 操作后保持 cache 序列完整性，实现 1.53× 推理加速。

  对比 baseline 的全栈执行例子（LiveStar + SCAM + SVeD, 同一 1 分钟视频 @3fps 含 5 个响应区间）：
  - 算法层：视频帧流（180 帧）→ InternViT 逐帧编码为 16 visual tokens → MLP Projector 映射 → 逐帧输入 LLM。SCAM 训练后的模型对每帧产生有意义的字幕（而非 EOS）。SVeD 推理：每 frame 通过 1 次 forward pass 计算当前 Dec 的 PPL（约 1ms），仅在 PPL 变化 > α·PPL_ref 时触发 decoding gate（约 5 次完整解码对应 5 个语义变化 → 共 5 次 decoding + 180 次 verification passes）。对比 baseline：180 次完整 decoding（每帧至少生成 EOS）= 36× 更多 decoding 开销。
  - 系统框架层：自建 streaming inference pipeline，双级 KV Cache 管理 + SVeD swap 兼容 + Peak-End 压缩
  - 编译框架层：论文未明确说明
  - kernel调度层：标准 PyTorch attention，论文未明确说明
  - 硬件架构层：NVIDIA A800 GPU，5 分钟视频 FPS 从 2.50（无 KV cache）提升至 3.82（双级 KV cache + Peak-End），1.53× 加速

  解决对应关系：
  | Baseline 缺陷 | LiveStar 解决方案 | 效果 |
  |---|---|---|
  | Pre-training Misalignment: EOS token 映射与 vision-language pretraining 矛盾 | SCAM: 所有帧始终对齐有意义的字幕内容（无 EOS 依赖），保持与 pretraining 范式一致 | TokAcc 0.62 vs 0.49/0.48 (VideoLLM-online/MoD) |
  | Response-Silence Imbalance: 沉默帧数远超响应帧 | SVeD: 沉默帧仅需 light verification pass（无 decoding），无需 supervised EOS 分类 | 19.5% SemCor 提升 + 18.1% TimDiff 降低 |
  | Consecutive Frame Inconsistency: 相邻帧输出矛盾 | SCAM: 同一语义片段所有帧共享统一字幕 + causal mask 确保时序一致性 | SemCor 4.62 vs 3.01/2.89 (VideoLLM-online/MoD, offline) |
  | Vocabulary Confusion: EOS 污染语义连贯性 | SVeD: 无 EOS token，Dec 始终为真实语义内容 | TimRedun 0.95 vs 2.15/2.49 (VideoLLM-online/MoD) |
  | 全部帧 decoding 开销 | SVeD verification 仅 forward pass 计算 PPL（无 token generation）；90%+ 帧为 silent pass | FPS 3.82 vs 3.37/3.41 (VideoLLM-online/MoD) |
  | 长视频 OOM | Peak-End Memory Compression：概率剪枝低重要性旧帧 | 支持 10+ min videos @3fps，SemCor 3.19 vs 3.04 (Uniform)/3.07 (FIFO) |
  | 历史上下文重复计算 | Streaming KV Cache 双级缓存 | 1.53× FPS 提升 |

## LongLLaVA__Scaling_Multi-modal_LLMs_to_1000_Images_Efficiently_via_Hybrid_Architecture

- baseline方法是什么？
  Baseline 方法是标准的 Transformer-only 多模态大语言模型架构（如 LLaVA-1.5/LLaVA-1.6），使用纯 Transformer decoder 作为 LLM backbone，CLIP 视觉编码器输出 576 tokens/image，无 token 压缩，所有 vision tokens 与 text tokens 拼接后参与 LLM 每一层的 full self-attention 计算。

  Baseline（LLaVA-1.6, 13B Vicuna Transformer, 128 images @ FP16）全栈执行例子：
  - 算法层：128 张图像 → CLIP ViT 逐张编码为 576 tokens/image → Projector 映射到 LLM embedding space → 128×576 = 73,728 vision tokens → 与 text tokens 拼接 → Vicuna-13B (40 Transformer layers) 逐层 full causal self-attention (O((N_vision + N_text)²) per layer) → KV cache 存储全部 token → 自回归 decode。100K tokens 输入时 Prefill 34.0s, Throughput 14.7 tokens/s, Memory 79.4 GB, Max throughput 14.7 tokens/s（单卡 A100 80GB）。
  - 系统框架层：vLLM Serving 框架 + Int8 Quantization (GPTQ)，提供批处理推理加速。论文用于效率对比，未修改。
  - 编译框架层：论文未明确说明。
  - kernel 调度层：标准 FlashAttention，利用 GPU 并行处理 causal self-attention。
  - 硬件架构层：NVIDIA A100 80GB / A800 GPU。Transformer 的 O(N²) 复杂度使 compute 随 token 数二次增长，100K tokens 时 already 79.4 GB memory，不支持 176K+ tokens。

  Baseline 的缺陷：
  1. **O(N²) 计算复杂度导致多图/长视频不可扩展**：Transformer self-attention 的计算复杂度和 KV cache 内存消耗都随序列长度 N 二次增长。当处理 100K tokens 或近千张图像时，单卡 A100 80GB 内存 (79.4 GB) 几乎耗尽，Prefill 时间达 34s，Throughput 仅 14.7 tokens/s。扩展到 176K tokens 训练序列直接 OOM。
  2. **纯 Mamba 架构虽然线性复杂度但 ICL 能力弱**：Falcon-mamba-7B（最大开源纯 Mamba LLM）虽有 O(N) 复杂度（100K tokens: Prefill 14.3s, Throughput 72.6, Memory 32.1 GB），但在 VL-ICL (Visual In-Context Learning) 任务上多 shot 性能远不如 Transformer（如 5-shot: 53.2 vs 58.9 的 Transformer），因其缺乏显式 attention 机制导致上下文检索/推理能力不足。
  3. **每张图 576 个 vision tokens 造成冗余**：CLIP 编码每张图像为 576 个 patch tokens，多图场景下视觉序列冗长。直接 1D pooling 虽压缩但丢失空间信息（2D pooling 保留 12×12 layout 维持空间关系更好）。
  4. **混合训练 (mixed training) 对多图任务效果差**：单阶段混合训练所有类型的 data 导致多图长上下文能力训练不充分，模型无法有效区分 temporal vs spatial 依赖。
  5. **一次性压缩牺牲细粒度信息**：现有 token 压缩方法（如 MiniGPT-v2）在 encoder 输出后 hard 压缩信息，导致高分辨率/小物体识别场景下性能大幅下降。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法：LongLLaVA 通过 Hybrid Mamba-Transformer 架构 + 2D Token Compression + 专用数据协议 + Progressive Training 四项设计系统解决：

  (1) **Hybrid Mamba-Transformer (Attention:Mamba = 1:7)** → 解决 O(N²) vs ICL 的矛盾。LLM backbone 使用 4 组 hybrid stack，每组 8 层中 1 层 Transformer Attention (全注意力用于保留 ICL/检索能力) + 7 层 Mamba SSM (线性复杂度用于降低计算开销)。另加 MoE 每隔一层集成 (16 experts, top-2)。结果：100K tokens: Prefill 25.5s (vs Transformer 34.0s), Throughput 37.6 tokens/s (vs 14.7), Memory 79.1 GB (接近，因 attention layer 仍存 KV cache), Max Throughput 37.6 (vs 14.7)。同时 VL-ICL 5-shot: 61.3 (vs Transformer 58.9, vs pure Mamba 53.2)，证明了 hybrid 在 ICL 和效率间的平衡。

  (2) **2D Bilinear Token Compression (576→144, 12×12 layout)** → 解决 vision token 冗余。Vision encoder CLIP ViT 输出 24×24=576 tokens → 2D bilinear pooling (2×2) → 12×12=144 tokens → MLP projector → LLM。相比 1D pooling，2D 保留 patch 间 2D 空间位置关系使模型能更好理解图像结构 (GQA: 61.3 vs 60.4; SEED: 67.4 vs 66.3; Mile: 37.7 vs 36.2)。每图 token 从 576 → 144 = 75% 减少，支持处理更多图像/帧。

  (3) **Data Processing Protocol (特殊 token 区分 temporal/spatial)** → 解决 mixed training 缺陷。设计 `<img>`/`</img>` 包围图像 token、`<vid>`/`</vid>` 包围视频帧、`<t>` 表示帧间时间依赖、`\n` 分隔高分辨率图像的子图行。使模型训练时通过特定 token 明确区分 temporal dependency (video frames) 和 spatial layout (patched high-res image)。

  (4) **Three-Stage Progressive Training** → 解决多图长上下文训练。Stage I (Single-image Alignment): 仅训练 projector，对齐 visual-text modality (600K captions)。Stage II (Single-image Instruction Tuning): 训练 projector + LLM (932K QA pairs)。Stage III (Multi-image Instruction Tuning): 全面多图训练 (700K+ instances) + Replay 机制保留单图/文本能力。Progressive Training 在 Mile 多图指标上 46.5 vs Mixed Training 42.2 (+4.3)。

  (5) **Image Partitioning 缓解 Token Compression 信息丢失** → 对细粒度任务补强。将高分辨率图分区为 168×168 子块，独立编码后按 spatial layout (\n 分隔行) 输入，使模型在不增加 total token 的情况下聚焦关键区域。V* Bench (小物体定位) accuracy 从 49.6% (direct) 提升到 68.5% (partitioning)，随子图数量增加持续改善。

  对比 baseline 的全栈执行例子（LongLLaVA-A13B, 128 images @ FP16）：
  - 算法层：128 张图像 → CLIP ViT 逐张编码 576 raw tokens → 2D bilinear pooling 压缩为 144 tokens/image → 128×144 = 18,432 vision tokens → 数据协议包装 (\<img\>...\</img\>) → 与 text tokens 拼接 → Hybrid LLM (4 stacks of Attention:Mamba=1:7, MoE 16x top-2): Transformer attention layers 做 full causal self-attention (保留 ICL 能力) → Mamba SSM layers 做 selective scan (线性复杂度, 无 KV cache 增长) → MoE layers top-2 gating FFN → 自回归 decode。100K tokens: Prefill 25.5s, Throughput 37.6 tokens/s, Memory 79.1 GB, nearly 1000 images 单卡 A100 80GB 可处理。
  - 系统框架层：vLLM + Int8 Quantization (GPTQ)。训练: 3×8 A800 GPU, sequence packing to 176K tokens, cosine schedule, peak lr=1e-5, AdamW。
  - 编译框架层：论文未明确说明。
  - kernel 调度层：FlashAttention (Transformer attention layers) + Mamba selective scan kernel (SSM layers)。Mamba 层无 KV cache，仅 attention 层保留 KV cache，大幅降低 memory footprint。
  - 硬件架构层：NVIDIA A100 80GB / A800 GPU。单卡 A100 80GB 可处理 ~1000 张图像（Needle-In-A-Haystack 评估），Video-NIAH 1200 帧评估 accuracy near 100%。训练 3×8 A800，176K token sequence length。

  解决对应关系：
  | Baseline 缺陷 | LongLLaVA 解决方案 |
  |---|---|
  | O(N²) 计算不可扩展 | Hybrid Architecture: 7 Mamba layers O(N) + 1 Attention layer O(N²) per stack → quasi-linear 复杂度。100K tokens TP 37.6 vs 14.7 (2.6× speedup) |
  | Pure Mamba ICL 弱 | Hybrid retains Attention layers for full ICL: VL-ICL 5-shot 61.3 vs Mamba 53.2 |
  | 每图 576 tokens 冗余 | 2D bilinear pooling: 576→144 (75%↓), 保留 12×12 spatial layout。Mile: 37.7 (2D) vs 36.2 (1D) |
  | Mixed training 对多图差 | Progressive 3-stage: Mile 46.5 (progressive) vs 42.2 (mixed) |
  | Token compression 丢失细粒度信息 | Image Partitioning: V* 49.6%→68.5% accuracy；整体性能可保持 competitive (mitigation strategy in Sec 5.2) |

## LongVILA__Scaling_Long-Context_Visual_Language_Models_for_Long_Videos

- baseline方法是什么？
  Baseline方法可以从两个维度分析：(a) 长视频VLM训练：现有方法如LongVA采用"长上下文LLM+短上下文数据训练"的策略，LongVLM使用token压缩规避上下文扩展；均缺乏完整的训练pipeline和系统协同设计；(b) 分布式训练系统：ZigZag-RingAttn使用Ring风格的P2P通信做序列并行，所有GPU间均用P2P传输KV blocks，忽视intra-node NVLink (900 GB/s)和inter-node InfiniBand (50 GB/s)的18×带宽差异；DeepSpeed-Ulysses使用All-to-All按head维度并行但扩展性受限于attention head数量（8B模型32 Q heads/8 KV heads）；HuggingFace Pipeline Parallelism推理逐层串行，仅1 GPU同时活跃且首卡内存瓶颈。

  Baseline全栈执行例子（8帧短视频推理，ZigZag-RingAttn+HF Pipeline推理）：
  - 算法层：VILA标准3阶段训练（对齐→预训练→短SFT），8帧视频，32K context，未做context extension
  - 系统框架层：FSDP数据并行，无法处理超长单序列
  - 编译框架层：论文未明确说明
  - kernel调度层：Ring-Attention P2P传输KV，所有GPU间统一P2P，intra-node也用P2P浪费NVLink带宽；通信-计算overlap占用SM资源导致attention kernel forward慢18.6%
  - 硬件架构层：H100 8卡，NVLink 900 GB/s + IB 50 GB/s，但Ring P2P未区分快慢通道

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法：(1) 五阶段训练pipeline：在VILA的3阶段基础上增加Stage4(文本context extension 8K→262K, RoPE基频增大+LoRA, 渐进式训练)和Stage5(长视频SFT, LongVILA_SFT 15K视频带caption+QA)，使模型从8帧扩展到2048帧；(2) MM-SP系统：两阶段sharding（视觉编码按帧、LLM解码按token）+ 2D-Attention（intra-node A2A + inter-node P2P分离快慢通道）+ 推理模式SP（全GPU并发）。

  解决Baseline缺陷的对应关系：
  - 对抗token压缩/LongVA的间接方案：LongVILA通过完整的训练pipeline（特别是Stage4 context extension + Stage5长视频SFT）直接扩展VLMs的有效帧数，而非规避问题
  - 对抗Ring P2P忽视网络异构：2D-Attention将通信分层——intra-node高频A2A走NVLink(900GB/s)，inter-node低频P2P走InfiniBand(50GB/s)，避免18×带宽差异导致的低效
  - 对抗Ring P2P通信开销不可隐藏：2D-Attention的A2A通信量更小且与计算更好重叠，不存在Ring-style中通信占用SM资源的问题（Table 2证明Ring overlap使kernel forward慢18.6%）
  - 对抗Ulysses head数限制：2D-Attention将SP维度分解为head dim × ring dim，ring dim不受head数限制，在256 GPU上可支持2M+ tokens（8×于Ulysses）
  - 对抗HF Pipeline推理低效：MM-SP推理所有GPU并发计算（8.2×加速），内存均匀分布（2.9×更长序列）

  论文方法全栈执行例子（256帧长视频推理，MM-SP 8 GPU 4×2 mesh）：
  - 算法层：五阶段pipeline训练模型，2048 frame上下文能力，RoPE扩展
  - 系统框架层：MM-SP monkey-patch HuggingFace Transformers，FSDP+SP混合并行
  - 编译框架层：论文未明确说明（Triton实现kernel，可port到C++）
  - kernel调度层：2D-Attention (4×2 mesh) —— Stage1按帧均分8帧→每GPU32帧视觉编码；Stage2全局tokens按seq dim均分→每GPU持有1/8 tokens；逐层A2A(4卡intra-node)重分布QKV按head dim→P2P(2组inter-node)传输KV→FlashAttention2本地注意力→Reverse A2A恢复分布；A2A利用NVLink高带宽，P2P仅跨节点传输
  - 硬件架构层：H100 8卡，NVLink 900 GB/s用于A2A(约27μs)，IB 50 GB/s用于P2P(约40μs)，Tensor Cores执行FlashAttention2

## LongVT__Incentivizing__Thinking_with_Long_Videos__via_Native_Tool_Calling

- baseline方法是什么？
  Baseline是Qwen2.5-VL-7B-Instruct，采用标准text-only Chain-of-Thought (CoT)推理配合uniform frame sampling的前向推理流程。模型对输入视频均匀采样64或512帧，一次性编码所有vision tokens后送入LLM进行单轮文本推理生成答案。这种passive frame consumption方式存在三个核心缺陷：(1) 均匀采样无法自适应捕获关键视觉证据——长视频中证据稀疏且时间上分散，uniform sampling容易错过fine-grained决定性时刻；(2) 纯文本CoT推理缺乏视觉grounding——模型在不确定时倾向"blindly rephrasing"而非回到视频中核实，导致幻觉；(3) SFT仅为imitation-driven，存在exposure bias，无法泛化到分布外query和未见视频模板。
  
  Baseline全栈执行例子（Qwen2.5-VL-7B，64-frame uniform sampling，单轮长视频QA推理）：
  - 算法层：Qwen2.5-VL-7B-Instruct，uniform采样64帧，visual encoder编码→projector→LLM decoder自回归生成文本，无tool calling能力，无temporal grounding supervision
  - 系统框架层：vLLM inference engine，continuous batching serving，无工具调用协议
  - 编译框架层：论文未明确说明
  - kernel调度层：标准FlashAttention/VLLM PagedAttention，无crop-resample pipeline
  - 硬件架构层：NVIDIA A800-SXM4-80GB 8卡推理

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法：(1) iMCoTT (interleaved Multimodal Chain-of-Tool-Thought)：将LMMs的潜在temporal grounding能力激活为native video cropping tool，实现global-to-local推理——先全局skim形成假设时间窗，再调用crop_video(start_time, end_time)重采样细粒度帧进行验证，支持最多5轮self-reflection；(2) 三阶段训练pipeline：SFT cold-start教会模型tool调用范式→Agentic RL (GRPO)用joint answer-temporal grounding reward (R = R_acc + R_format + R_time)优化探索→Agentic RFT蒸馏高质量rollouts稳定行为；(3) VideoSIAH数据套件：半自动pipeline生成247.9K SFT样本 + 1.6K RL样本 + 15.4K RFT样本 + 652条VideoSIAH-Eval基准。

  解决Baseline缺陷的对应关系：
  - 对抗uniform sampling错过关键证据：iMCoTT通过crop_video工具实现on-demand temporal retrieval，模型可根据全局预览自主选择感兴趣时间段进行细粒度重采样，而非被动消费均匀帧。RL阶段的时间grounding reward (IoU)进一步优化窗口提案精度，使模型学会定位稀疏证据。
  - 对抗纯文本CoT幻觉：iMCoTT使推理过程grounded在实际视觉证据上——模型先think形成假设，再调用crop_video获取验证证据，基于新证据重新think，可自我纠正初始错误（如Figure 8案例中模型通过re-check将pink纠正为blue）。避免了"blindly rephrasing"导致的虚假回答。
  - 对抗SFT imitation-driven限制：GRPO-based RL的exploratory rollouts + joint奖励函数使模型超越SFT distribution——IoU reward抑制span inflation（对比Recall reward的reward hacking），RFT阶段用高质量rollouts (answer正确 AND IoU≥0.3) 提供in-distribution supervision稳定优化。Table 3证实SFT+RL+RFT全pipeline显著优于单一阶段。
  - 对抗inference latency：尽管有multi-turn tool interactions，LongVT-RFT inference速度反而快于单轮baselines（Table 4），因为证据grounded的回答更简洁，避免了hallucination-driven verbose generation。

  论文方法全栈执行例子（LongVT-7B-RFT，512 frames，多轮tool calling，长视频推理）：
  - 算法层：(1) 全局skim 64 frames → visual encoder → projector → LLM生成初始假设窗口 <think> [t_s, t_e] </think>；(2) 调用 <tool_call>{"name":"crop_video","arguments":{"start_time":t_s,"end_time":t_e}}</tool_call> → 外部executor从原始视频[t_s, t_e]段重采样64帧 → 再次visual encoder → projector → vision tokens返回；(3) LLM基于新vision tokens重新think验证证据 → <answer>最终答案</answer>；最多5轮。LongVT-7B-RFT模型经过SFT→RL→RFT三个阶段训练以优化此过程。
  - 系统框架层：LMMs-Engine (SFT训练，stream packing buffer 51200 tokens)，verl + SGLang (RL训练，multi-turn multimodal tool-augmented rollouts，16 rollouts/prompt)，vLLM (推理serving，MCP server + continuous batching)
  - 编译框架层：论文未明确说明（使用AdamW optimizer, Liger Kernel for SFT/RFT）
  - kernel调度层：crop_video执行在外部executor上，非kernel级别优化；visual encoder的FlashAttention处理vision tokens；SGLang prefix caching复用多次tool calling间的共享prefix tokens
  - 硬件架构层：NVIDIA A800-SXM4-80GB，SFT用32卡，RL用64卡，RFT用64卡，推理评估用8卡

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

## Machine_Mental_Imagery__Empower_Multimodal_Reasoning_with_Latent_Visual_Tokens

- baseline方法是什么？
  Baseline 分为两类：
  (1) **文本-only 推理 VLM**（Zero-Shot / Direct SFT / CoT SFT / GRPO / CoT SFT+GRPO）：VLM 仅输出文本 token，通过 chain-of-thought prompting 或 RL fine-tuning 延长文本推理轨迹。VLM 的 vision encoder 编码输入图像后，通过 MLP projection 映射到 LLM 的 text embedding space，随后全部 token 走 LLM 自回归解码。缺陷：视觉推理需要将视觉信息"翻译"为自然语言描述（verbalize），在 jigsaw puzzle、spatial navigation 等需要视觉想象的任务上，语言中介成为瓶颈，丢失了隐空间的视觉结构信息。
  (2) **Unified multimodal 模型**（Anole, MVoT）：在 Chameleon 类统一 token-based 框架下训练模型同时输出 text tokens 和 image tokens（pixel patches），再通过 external image decoder 渲染成显式图像。缺陷：(a) 大规模像素级图像生成预训练的计算开销极大，且常损害推理质量（Wang et al., 2025 指出同时优化逻辑推理和像素合成会导致推理退化）；(b) 生成的显式图像难以与输入图像交互形成 interleaved trajectory；(c) Anole 论文复现在 spatial planning 任务上甚至无法产生有效答案。

  Baseline（以 CoT SFT 为例）全栈执行例子：
  - **算法层**：Qwen2.5-VL-7B-Instruct → 输入 (image, text_query) → vision encoder (ViT) 输出 patch features → MLP projection → 与 text token embeddings 拼接 → LLM 自回归生成 text-only CoT → LM head 输出每个 text token 的概率分布 → 最终答案。SFT loss = CE(text_tokens)。无 latent visual 通道。
  - **系统框架层**：PyTorch + HuggingFace Transformers。标准 VLM 推理流程。GRPO 使用 VERL 框架。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：标准 Transformer 推理 kernel（FlashAttention for self-attention）。无自定义 kernel。
  - **硬件架构层**：单 NVIDIA H100 GPU 训练。Qwen2.5-VL 7B 标准 batch_size=8。

  Baseline 的核心缺陷：
  1. **视觉推理被迫语言化（Verbalization Bottleneck）**：VLM 必须将视觉空间推理转化为自然语言描述，对 jigsaw 的边缘连续性、spatial navigation 的空间关系等需要"视觉想象"的任务，语言是次级表示，丢失精确的几何与空间结构。
  2. **Unified model 的推理-生成冲突**：同时学习逻辑推理和图像合成导致两个目标冲突，推理能力退化。
  3. **隐空间视觉信道未被利用**：VLM 的 hidden states 本身富含视觉信息（vision encoder 输出经过 LLM 各层后的中间表示），但 text-only decoding 将这些信息全部丢弃（仅 LM head projection 到 vocab space），浪费了 LLM 内部已编码的视觉知识。
  4. **合成推理链的质量依赖**：CoT SFT 的性能受限于合成 reasoning chain 的质量（由 Qwen2.5-VL-32B 生成），且长 CoT 文本会导致推理效率下降。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  Mirage 通过三阶段训练范式实现 latent visual token 的 interleaved reasoning：

  (1) **Latent Visual Tokens → 解决 Verbalization Bottleneck**。不生成显式图像或文本描述视觉信息，而是直接复用 VLM 当前 hidden state 作为 compact visual embedding（k=4 个连续向量），插入文本 token 流中。这些 latent tokens 承载 task-relevant 视觉线索（如 jigsaw 的边缘匹配信息、navigation 的路径空间关系），供后续 token 的 attention 机制直接访问。类比人类 "mental imagery"——不生成照片级画面，而是构建简化的心智草图。

  (2) **Stage 1 Joint Supervision → 解决隐空间视觉信道未利用**。先用 helper image I 生成 compressed visual embeddings {ê_j} 作为 ground-truth 信号，通过 cosine similarity loss 将模型在 <latent> slot 的 hidden states 锚定到 visual subspace。同时训练左右文本段的 CE loss，让模型学会在文本推理中自然编织 visual cues。L_1 = Σ cos_sim(ê_j, h_j) + γ·Σ CE(text)。γ=0.1 控制视觉约束强度。

  (3) **Stage 2 Text-Only Relaxation → 解决 Unified model 推理-生成冲突**。移除 visual alignment loss，仅保留文本 CE loss。模型自回归生成自己的 latent tokens {e_i}，梯度通过 o_post 的 CE loss 反向传播到这些连续变量（fully differentiable）。这使 latent tokens 在 visual subspace 内自适应偏移，不再强制匹配固定 ground-truth embedding，从而更灵活地服务于最终答案生成。即 stage 1 提供 grounding（锚定语义），stage 2 提供 flexibility（任务自适应）。

  (4) **Stage 3 GRPO RL → 进一步优化 interleaved trajectory**。使用 VERL + GRPO，基于 accuracy + format rewards 优化整个 interleaved 序列。latent tokens 同样接收梯度（但排除 KL penalty），使模型可探索更优的 latent-text 交织模式。

  对比 baseline 的全栈执行例子（Mirage, Qwen2.5-VL 7B, VSP Spatial Reasoning, k=4）：

  - **算法层**：Qwen2.5-VL-7B → 输入 (map_image, text_query) → vision encoder 编码图像 → MLP projection → LLM 开始自回归生成。生成触发机制：当模型需要 "think visually" 时，输出特殊 token `<latent>` → 此时不通过 LM head 映射到 vocab，而是直接取当前最后一层的 hidden state h ∈ R^d 作为 latent visual token e_1 → 将 e_1 作为连续向量拼入 key-value context（类比 KV cache 中的一条）→ 继续生成 e_2, e_3, e_4（共 k=4 个 latent tokens）→ 这 4 个连续向量携带压缩后的视觉空间信息 → 后续 o_post text tokens 的 self-attention 可以 attend 到这些 visual embeddings → LM head 输出最终答案。对比 baseline text-only：{image → text_thoughts → answer}；Mirage：{image → text_thoughts_pre → [e_1, e_2, e_3, e_4] → text_thoughts_post → answer}。
  - **系统框架层**：PyTorch + HuggingFace Transformers for SFT；VERL framework for GRPO RL。修改点：在 LLM decoding loop 中插入 latent token generation path（bypass LM head, 直接取 hidden state）；训练时需要 helper image 预处理 pipeline（vision encoder forward + average pooling compression）。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：标准 Transformer kernel。latent tokens 的生成和梯度反向传播均遵循标准 PyTorch 计算图。latent token attention 与 text token attention 共享同一 self-attention kernel（隐向量作为连续的 key-value 条目参与 scaled dot-product attention）。
  - **硬件架构层**：单 NVIDIA H100 GPU。Stage 1 ~3.5h, Stage 2 ~7.2h (VSP task)。与 text-only CoT SFT (~5.5h) 相比，Mirage 总训练时间 10.7h ≈ 2× CoT SFT，但性能提升显著（VSP Spatial Planning 从 47% → 58%, Spatial Reasoning 从 84% → 87%）。推理时 latent tokens 增加的计算开销极小（k=4 个向量参与 attention，相比 576+ vision tokens 可忽略）。

## Multimodal_Long_Video_Modeling_Based_on_Temporal_Dynamic_Context

- baseline方法是什么？
  - **Baseline 代表**：VideoLLaMA2、Video-LLaVA、LLaMA-VID 等现有视频 MLLM。典型方法：从视频中固定采样少量帧（如 8 或 16 帧），每帧独立通过 CLIP/SigLIP 编码为视觉 token，直接拼接到文本 token 送入 LLM。视觉和音频模态分开编码后简单拼接，缺乏模态间交互。
  - **全栈执行例子（VideoLLaMA2 baseline）**：
    - **算法层**：输入 120s 视频 → 采样 16 帧 → SigLIP encoder 每帧得到 72 visual tokens → BEATs encoder 每帧约 50 audio tokens → 拼接 visual + audio tokens (16 × (72+50) = 1952 tokens) → 与 question text tokens 拼接送入 LLM (Qwen2-7B) → 自回归生成答案。长视频场景下，16 帧均匀采样丢失大量时序细节；简单拼接各模态 token 导致 LLM 难以区分融合多模态信息。
    - **系统框架层**：PyTorch + HuggingFace Transformers。LLM 通过 causal self-attention 处理所有输入 token。无 token 压缩机制，token 数量随帧数线性增长。
    - **编译框架层**：论文未明确说明。使用标准 PyTorch eager mode 推理。
    - **kernel调度层**：标准 Transformer attention kernel（Flash Attention 或 PyTorch native SDPA）。无自定义 kernel。
    - **硬件架构层**：NVIDIA GPU（论文未明确型号）。Baseline 中每帧独立编码、简单拼接，无跨帧/跨模态融合计算。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **缺陷 1**：固定少量帧采样导致信息丢失 → TDC 以 1fps 密集采样全部帧，并通过语义场景分割保证时序一致性。
  - **缺陷 2**：长视频 token 数量随帧数线性增长 → TDC 将首帧作为静态参考（完整保留），后续帧通过 Q-Former 压缩为 K=16 个 context tokens，使每帧平均 token 数从 194 降至 16。
  - **缺陷 3**：视觉和音频模态独立编码、简单拼接 → TDC 将 visual + audio tokens 一起送入 Q-Former 做 cross-attention，在统一视频上下文中融合多模态信息；同时注入 instruction text (F_s) 使压缩过程自适应于用户问题。
  - **缺陷 4**：超长视频无法整体处理 → LVCoT 将视频等分为 M 段，逐段推理生成中间答案，再汇总为最终答案，利用分段推理链提升长视频理解深度。
  - **全栈执行例子（TDC 7B）**：
    - **算法层**：输入 T 秒视频 (T 可能 > 1000) → 1fps 密集采样 T 帧 → DINOv2 提取每帧 embedding，cosine similarity 找 S-1 个低相似度分割点 → 视频分割为 S≤24 个场景 → 每场景 sliding window 内首帧完整保留 (144 visual + 50 audio tokens) → AvgPool 从首帧 visual tokens 得 K=16 个 query tokens → 后续每帧 visual+audio tokens 送入 Q-Former (BERT initialized) 与 query tokens 做 cross-attention，同时注入 question text (F_s) → 输出压缩后的 context tokens(16/frame) → 场景表示 F_TDC = [static_tokens · <Sep> · context_tokens] → LLM (Qwen2-7B/LLaMA3.2-3B) 自回归生成。超长视频额外触发 LVCoT: 均分 3 段 → 每段独立 TDC 编码+推理 → 汇总段级答案作为 chain-of-thought → 最终全局推理。
    - **系统框架层**：PyTorch + HuggingFace Transformers。修改点：在 video encoding pipeline 中插入 scene segmentation module (DINOv2 similarity)、TDC Q-Former compressor（可训练）、LVCoT multi-pass 推理循环。三阶段训练：Stage 1 vision-language alignment (3.2M), Stage 2 video instruction tuning (2M/540K), Stage 3 audio-video instruction tuning (300K/120K + LoRA)。
    - **编译框架层**：论文未明确说明。使用标准 PyTorch eager 推理。
    - **kernel调度层**：Q-Former cross-attention 使用标准 Transformer attention kernel。无特殊 kernel 优化。场景分割的 cosine similarity 矩阵计算为标准矩阵乘法。
    - **硬件架构层**：论文未明确说明 GPU 型号。推理计算量主要来自 (1) per-frame encoder forward (SigLIP + BEATs), (2) Q-Former cross-attention (每帧 16 queries × ~194 key-value tokens), (3) LLM decoder self-attention。TDC 将每帧平均 token 数从 ~194 压缩至 16，LLM attention 成本降为原来的 ~(16/194)² ≈ 0.7%。LVCoT 每段独立推理引入额外 forward pass 但提升长视频准确性。

## PEARL__Personalized_Streaming_Video_Understanding_Model

- baseline方法是什么？
  Baseline 是现有的**个性化图像/视频理解方法**，可分为两类：
  1. **离线模型**（LLaVA-OV-7B, Qwen2-VL-7B, InternVL3.5-8B, Qwen3-VL-8B）：均匀采样 64 帧（frame-level）或 64 秒窗口 1fps（video-level）处理视频，无显式记忆机制。每次查询独立处理当前窗口，无法跨时间步维护概念和视觉证据。
  2. **在线模型**（ReKV, StreamForest-7B, TimeChat-Online-7B）：支持流式视频输入和多轮对话，但将历史信息压缩为固定大小的状态表示，缺乏概念级（concept-grounded）检索能力，不能精确检索与用户定义概念相关的历史视觉证据。

  Baseline 全栈执行例子（以离线 LLaVA-OV-7B 64帧均匀采样为例）：
  - **算法层**：长视频 → 均匀采样 64 帧（丢失关键时刻） → ViT 编码所有帧 → LLM 自回归生成答案。所有查询独立处理，无概念记忆，无历史检索。概念定义信息在下一个查询中消失，必须重复提供。问题：(a) 64 帧限制无法保留长范围历史证据用于 Past-Time QA；(b) 无概念存储机制使 Real-Time QA 无法可靠地将个性化名字链接到视觉实体；(c) 无检索机制意味着历史视觉证据完全不可访问。
  - **系统框架层**：PyTorch + HuggingFace Transformers。视频帧通过 OpenCV/decord 解码，按 instruction template 与文本 token 拼接送入 VLM。无流式视频管理、无外部记忆模块。
  - **编译框架层**：论文未明确说明。使用标准 PyTorch eager 推理。
  - **kernel调度层**：标准 Transformer attention kernel（Flash Attention 或 PyTorch native SDPA）。无自定义 kernel。
  - **硬件架构层**：NVIDIA H200 GPU。

  在线模型（如 ReKV）虽然支持流式输入，但其 KV-cache 压缩将历史信息丢失到固定大小的压缩状态中，无法为 Past-Time QA 提供精确的检索式历史证据访问。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  PEARL 通过 **Dual-grained Memory System + Concept-aware Retrieval Algorithm** 解决 baseline 的三大核心缺陷：

  **(a) 缺陷1：无概念记忆 → Concept Memory（概念记忆）**
  Baseline 在每次查询中都不知道用户定义了哪些个性化概念。PEARL 的 Concept Memory 在 Concept-Definition QA 触发时，从当前 clip 提取视觉证据（frame-level 取最后一帧，video-level 取 clip），用 VLM 生成聚焦永久/稳定特征的紧凑文本描述（frame-level：性别/面部/发型/体型；video-level：核心运动学/动作序列），将概念名、视觉证据和文本描述结构化存储。后续查询通过概念名匹配快速检索 Csub，使模型在任何时间点都知道"Adaliz 是一个年轻女性，长黑发"。消融实验：加 Concept Memory 使 Real-Time 准确率从 15.84% 飙升至 51.41%（+35.57%）。

  **(b) 缺陷2：无法访问历史视觉证据 → Streaming Memory + Concept-aware Retrieval**
  Baseline 的 64 帧窗口无法覆盖 Past-Time QA 所需的历史 clip（例如查询"Adaliz 做饭时穿什么颜色"需要检索 30 分钟前的 cooking 场景）。PEARL 的 Streaming Memory 将视频流增量归档为 (clip, embedding) 对，Concept-aware Retrieval 通过 Query Rewriting 将概念名替换为视觉描述后编码为嵌入，与 Streaming Memory 中所有历史 clip 嵌入做余弦相似度匹配，精确检索 Top-K 最相关历史 clip。消融实验：加 Streaming Memory 使 Past-Time 准确率从 25.43% 提升至 45.69%（+20.26%）。

  **(c) 缺陷3：个性化名称无法被通用嵌入模型理解 → Query Rewriting（查询重写）**
  通用多模态嵌入模型未见过用户定义的个性化名称（如"Adaliz"）。PEARL 在检索前将查询中出现的概念名替换为对应的视觉描述文本（如"a young female with long black hair"），使重写后的查询能被嵌入模型有效编码，从而与 clip 嵌入进行语义匹配。消融实验：加 Query Rewriting 进一步提升 Avg 准确率 4.28%。

  对比 baseline 的全栈执行例子（PEARL + Qwen3-VL-8B, 1fps）：
  - **算法层**：流式视频 → PySceneDetect 检测场景边界（HSV delta threshold=27.0, min 1s/max 8s clip） → 每个新 clip 经 Qwen3-VL-Embedding-2B 编码为 embedding → 存入 Streaming Memory。用户定义概念时：从当前 clip 提取视觉证据 → VLM 生成概念描述 → 存入 ConceptMemory{(name, evidence, desc)}。用户查询时：(1) 从 Q 中提取概念名 → 检索 ConceptMemory 获取 Csub；(2) VLM 重写 Q → Q̃（替换概念名为描述）；(3) Qwen3-VL-Embedding-2B 编码 Q̃ → e^Q → 与 StreamingMemory 中所有 ei 计算 cosine similarity → Top-K=4 clips + N=1 邻接扩展 → Vcontext；(4) Csub + Vcontext + X^tq + Q → VLM decoder → 生成答案 A。全程 training-free，不更新任何模型参数。
  - **系统框架层**：PyTorch + Qwen3-VL-Embedding-2B + PySceneDetect。修改点：(a) 在 VLM 推理 pipeline 外挂 Dual-grained Memory System（StreamingMemory 增量归档线程 + ConceptMemory 注册/检索接口）；(b) 在 VLM 推理前插入 Concept-aware Retrieval 预处理步骤（Query Rewriting + embedding-based clip retrieval）；(c) 多 GPU 评估 pipeline（server/ 启动 VLM 和 embedding server，scripts/ 协调多 GPU 并行推理 → eval.py 聚合评估指标）。
  - **编译框架层**：论文未明确说明。使用标准 PyTorch eager 推理。
  - **kernel调度层**：标准 Transformer attention kernel。PEARL 的额外计算来自：(a) embedding 编码（Qwen3-VL-Embedding-2B 前向，~constant cost）；(b) cosine similarity 矩阵计算（clip 数 × d_embed 维向量，线性于 StreamingMemory 规模）；(c) VLM Query Rewriting（一次额外 VLM 推理，仅处理文本长度级别的 token）。延迟分解（Fig.5）显示 PEARL 核心模块（Concept Retrieval + Query Rewriting + Streaming Memory Retrieval）的延迟极低且跨模型恒定，主要瓶颈仍是 LLM 推理。
  - **硬件架构层**：NVIDIA H200 GPU。PEARL 与 baseline 共享相同硬件，额外的检索和重写模块仅引入可忽略的计算开销。LLaVA-OV-7B+PEARL 端到端延迟 775ms（vs 670ms baseline），以 105ms 额外延迟换取 8.55% 平均准确率提升。

## Representation_Shift__Unifying_Token_Compression_with_FlashAttention

- baseline方法是什么？
  Baseline 是基于 attention map 的 token 剪枝方法（如 EViT, BAT, Zero-TPrune, vid-TLDR, DynamicViT, AdaViT），在 Vision/Video Transformer 推理时通过 attention scores 评估 token 重要性并剪除低重要性 token。核心依赖 self-attention 计算过程中产生的 attention map 作为 token 重要性的代理信号，例如 EViT/BAT 使用 class token 对 key tokens 的 attention scores（s = Softmax(q_cls K^T/√C)），vid-TLDR 使用 averaged attention across all query vectors（s = (1/N) Σ A_i）。

  Baseline（以 EViT with DeiT-S 为例）全栈执行例子：
  - 算法层：图像 224×224 → 14×14=196 patches + 1 class token → DeiT-S 12层。第3层：每 token 对其余 tokens 的全连接 self-attention → Attention Map A ∈ R^(197×197) → 取 class token row A_cls = A[0, 1:] ∈ R^196 → Softmax → top-K 选保留 tokens → 丢弃其余 → 剩余层处理缩减后的 tokens。问题：(1) attention map 在早期层不可靠，class token 注意力分布尚未收敛（Figure 3/6 显示 early layer attention 近乎随机）；(2) FlashAttention 不暴露 intermediate attention maps（为减少 HBM I/O），使 attention-based 方法完全不兼容。
  - 系统框架层：PyTorch + HuggingFace Transformers。标准 self-attention（非 FlashAttention），因需要访问 attention map。
  - 编译框架层：论文未明确说明。
  - kernel调度层：标准 PyTorch attention（nn.MultiheadAttention）或手动实现，不使用 FlashAttention fused kernel。attention map 必须显式存储在 HBM 中供剪枝使用。
  - 硬件架构层：单 GPU（训练和推理相同），未使用 FlashAttention 加速。

  Baseline 的核心缺陷：
  1. **Attention map 依赖与 FlashAttention 不兼容**：FlashAttention 将 attention 计算融合为单 kernel，避免构建完整 attention map 及写入 HBM，从而大幅加速（DeiT-S 1.5×, UMT-B 2.7× speedup）。但 attention-based token pruning 需要 attention map 来确定 token 重要性，两者不可兼得。这意味着在享受 FlashAttention 加速的同时无法进一步通过 token pruning 降低计算量。
  2. **Early layer attention 信号质量差**：Transformer 前几层 attention map 不可靠（Figure 3 显示 early attention 近乎随机，Figure 6 定性对比 attention vs rep shift 在 L=1 层的差异）。早期剪枝基于 unreliable 信号会错误丢弃重要 token。
  3. **需要额外训练或参数**：DynamicViT、AdaViT、A-ViT 等方法引入额外可学习网络预测 token 重要性，需要 re-training/fine-tuning，不适用于 training-free 场景。
  4. **架构局限**：attention-based scoring 仅适用于 Transformer（需要 self-attention 机制），无法扩展到 CNN 和 SSM 等其他架构。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法：Representation Shift——一种训练无关（training-free）、模型无关（model-agnostic）的 token 重要性度量。核心公式：s = Δx = ||F(x) - x||₂，其中 F(·) 为层的变换函数（选定 MLP 层），Δx 量化每个 token 经过该层后的表示变化量。直观理解：对任务关键的 token 会被网络强调（大 representation shift），冗余 token 几乎不变（小 shift）。无需 attention map，与 FlashAttention 完全兼容。

  **(a) 缺陷1：Attention map 与 FlashAttention 不兼容 → 不需要 attention map 的重要性度量**
  Representation shift 计算 token 在 MLP 前后的 L2 距离，完全独立于 attention 机制。由于不需要 attention map，可以在所有层使用 FlashAttention 的同时在特定层（早期层）应用 token pruning。UMT-B 从 32 vid/s（Base, standard attention）加速到 175 vid/s（+FlashAttention + rep shift pruning），相比之下 attention-based pruning（standard attention）仅 57 vid/s。FlashAttention 本身提供约 2.7× speedup，token pruning 在 FlashAttention 基础上再加速约 2×。

  **(b) 缺陷2：Early layer attention 信号质量差 → MLP-based representation shift 在 early layer 更可靠**
  Figure 3/5a 消融实验显示：(i) 基于 Attention 的 representation shift 不如 MLP-based——因 attention 层进行跨 token 信息交换，transformation 更扩散（diffuse），而 MLP 逐 token 独立操作，产生更具判别性的 representation shift；(ii) Figure 6 定性对比中，rep shift 在第1层即成功检测前景物体（"handles foreground object well"），而 attention map 在早期层近乎随机。L2 距离在深度上一致优于 L1 和 cosine（Figure 5b）。

  **(c) 缺陷3：需要额外训练 → 完全 training-free**
  Representation shift 仅需一次前向传播计算 token 的 L2 差，无需任何额外参数或训练。直接应用于预训练模型。Table 5, 6, 7 的实验均无额外训练（CNN 的 fine-tuning 是为了适应 resolution change，非学习 importance scoring）。

  **(d) 缺陷4：仅适用 Transformer → 扩展到 CNN 和 SSM**
  Representation shift 的 "模型无关" 特性使其可计算任何层的输入输出差。CNN（ResNet）：在各 stage 后计算 feature map 变化，通过行/列级剪枝减少分辨率；SSM（Vision Mamba）：替换 ToP-ViM 的激活值基分数为 rep shift。Table 6/7 展示了 CNN/SSM 上 real throughput gain。

  对比 baseline 的全栈执行例子（Representation Shift + UMT-L + FlashAttention, video-text retrieval）：
  - **算法层**：视频 12 frames × 224² → 2352 tokens → Layer 0：FlashAttention（fused SRAM kernel，不暴露 A）→ MLP(LN(x')) → Δ = ||MLP(LN(x')) - x'||₂ → Top-80% → prune 20% → ×3 layers progressive prune → 1204 tokens remaining → 后续 9 层正常 Transformer（FlashAttention）→ text and video embeddings similarity → R@K retrieval。全程无 attention map 依赖。吞吐量 66 vid/s（vs Base 12 vid/s = 5.5×），FLOPs 从 984.6G 降至 478.5G。
  - **系统框架层**：PyTorch + FlashAttention fused kernel（通过 `scaled_dot_product_attention` 或 flash-attn 库）。修改：在指定层（drop_layers）的 MLP 后插入 rep shift 计算 + token pruning 模块；其余层不变。无 Serving 框架修改。
  - **编译框架层**：论文未明确说明。使用标准 PyTorch eager 模式。
  - **kernel调度层**：FlashAttention fused kernel 用作标准 self-attention 后端。Representation shift 仅增加 L2 norm 计算（O(N × C)），开销可忽略（< 1% of total FLOPs）。无自定义 kernel 修改。关键的 kernel 兼容性：FlashAttention 的 SRAM-resident 计算不产生 attention map，而 rep shift 不需要 attention map，两者正交兼容。
  - **硬件架构层**：单 NVIDIA RTX A6000 GPU。FlashAttention 减少 HBM I/O 实现 2.7× speedup，rep shift-based pruning 减少 token 数实现额外 2× speedup，两者叠加总 speedup 5.5×（UMT-L）。

  核心洞察：representation shift 的成功源于一个经验观察——"网络中信息被放大的 token 对任务更重要"。MLP 的逐 token 独立变换使得这一信号的 distinguishability 最优。L2 距离简单但比 cosine（angular）和 L1（robust but less discriminative in deeper layers）更一致。这一发现使 token pruning 首次实现了与 FlashAttention 的 superposition——两种正交加速技术的组合产生乘法级 speedup（1.5× FlashAttention × 2× pruning ≈ 3-5.5×）。

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

## Scaling the Long Video Understanding of Multimodal Large Language Models via Visual Memory Mechanism (FlexMem)

- baseline方法是什么？
  Baseline 包含两类方法：
  **(a) MLLM 原生 uniform sparse sampling**：LLaVA-Video 7B 均匀采样 64 帧（13k tokens）或 LLaVA-OneVision 7B 均匀采样 32 帧（7k tokens），将所有帧一次性 concat 送入 LLM 做 full self-attention 并自回归生成答案。核心缺陷：(1) 输入帧数受限于 LLM 的 sequence length 上限（超过 200k tokens 即无法处理），均匀采样 64 帧相当于丢弃了 88%+ 的视频信息；(2) 对所有帧等权处理，缺乏对关键片段的聚焦能力；(3) 缺乏跨 clip 的信息传递和历史记忆，无法理解跨越多个采样的长期依赖。
  **(b) 现有高效长视频理解方法**：VideoRAG（如 AKS）通过相似度检索关键帧再输入 MLLM，但缺乏时序连续性理解，对需要全局/整体理解的任务表现差；视觉压缩方法（如 AdaRETAKE）逐 clip 压缩 KV cache 但最终仍需输入所有 compressed features，上下文长度随视频时长线性增长，存在计算瓶颈。

  Baseline（LLaVA-Video 7B, 64 frames uniform sampling）全栈执行例子：
  - 模型推理算法层：长视频 → 均匀采样 64 帧 → SigLIP Vision Encoder 逐帧编码为 ~182 tokens/frame → spatial pooling → 64×182≈11.6k visual tokens → Projector → 与 text prompt tokens 拼接 → Qwen2-7B 28 层 causal self-attention（对所有 11.6k tokens 做 full attention）→ 自回归解码生成答案。对所有视频等权均匀采样，无法区分关键帧和冗余帧。2h 视频仅用 64 帧相当于仅看 0.5fps，严重欠采样。
  - 系统框架层：PyTorch + HuggingFace Transformers，标准 Video-MLLM 推理 pipeline。无专用 Serving 框架修改。
  - 编译框架层：论文未明确说明。
  - kernel 调度层：标准 FlashAttention（causal mask），无自定义 kernel。
  - 硬件架构层：单张 NVIDIA RTX 3090 GPU (24GB)。64 帧 uniform sampling 下可正常运行，但帧数扩展至 128+ 帧时视觉 token 数超 23k，24GB 显存紧张。

  核心缺陷总结：(1) **输入上限**：受 sequence length 和显存限制，无法处理 100+ 帧，大量视频信息被丢弃；(2) **缺乏记忆机制**：无跨 clip 的信息传递，前 30 分钟视频的信息在后 30 分钟完全丢失；(3) **缺乏聚焦能力**：对所有帧等权处理，无法针对问题聚焦关键片段；(4) **RAG 和压缩方法各有局限**：RAG 丢失时序连续性，压缩方法仍受限于线性增长的上下文。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  FlexMem 通过**视觉记忆机制（Visual Memory Mechanism）**将长视频理解从"一次性全部输入"改为"迭代观看-形成记忆-召回相关片段"，解决了 baseline 的四大缺陷：

  **(a) 缺陷1：输入上限（sequence length + 显存限制）** → 迭代式记忆编码 + 双路径 KV Cache 压缩
  Baseline 必须一次性输入所有帧，64 帧即接近上限。FlexMem 将视频分片为 N 个 clips（每 clip 8 帧），每次仅处理当前 clip 的 KV cache + 前序 context memory，通过 Dual-Pathway Compression（DPC）将每 clip 压缩为极少的 context 和 local memory tokens。最终解码时仅召回最相关的 na 个 clip 的 memory，而非全量 compressed tokens。LLaVA-Video 解码时仅用 13k tokens（vs AdaRETAKE 的 40k）。理论上可处理无限长视频，实验已验证 1024 帧（16× baseline）。

  **(b) 缺陷2：缺乏跨 clip 信息传递和长期记忆** → Context Memory 链式传递 + Memory Bank
  Baseline 前 30 分钟和后 30 分钟视频信息完全独立。FlexMem 通过两种设计实现跨 clip 信息流：(i) Context Memory C 的链式传递——每步编码时 MLLM 接收前 ns 个 context memory {C_{i-ns}, ..., C_{i-1}}，将历史视频信息持续传递；(ii) Visual Memory Bank M_bank——所有 local memory Mi 被持久存储，并在需要时通过 memory recall 召回长期记忆 `<Ml>`。消融实验（Table 5 Block 2）证实 context + local 组合显著优于单独使用任一种。

  **(c) 缺陷3：缺乏对问题的聚焦能力** → Memory Recall（记忆召回）
  Baseline 对所有帧等权处理。FlexMem 在观看完全部视频后，通过 memory recall 从 M_bank 中召回最相关的记忆片段：(i) Encoding-based Reading 利用 MLLM 在 encoding 时的 cross-modal attention 计算各 clip 与问题 Tq 的 relevance score g_i，Top-K 选择最相关片段；(ii) MemIndex 通过线性回归学习 encoding-based reading 的 relevance 分布，用选定的 cache 层（K=3）和压缩视觉索引（k=5 tokens）做快速点积匹配，完全独立于 memory encoding，适合多问题和 streaming 场景。消融（Table 5 Block 3）验证 memory reading 远优于 indiscriminate loading of all memory。

  **(d) 缺陷4：RAG 和压缩方法各有局限** → 结合两者优势
  FlexMem 同时具备：(i) 压缩方法的全面理解——通过 context memory 链式传递保持时序连续性；(ii) RAG 方法的精确定位——通过 memory recall 从完整 M_bank 中精确召回相关片段。在单 3090 上，FlexMem 在五个 benchmark 上全面超越 AKS（RAG 代表）和 AdaRETAKE（压缩代表），且在 24GB 受限下仅损失 0.5% 性能（vs AKS 和 AdaRETAKE 的显著退化）。

  对比 baseline 的全栈执行例子（FlexMem + LLaVA-Video 7B, 512 frames, 单 RTX 3090）：
  - 模型推理算法层：
    (1) 视频 V → 均匀分 N=64 clips（每 clip 8 帧）→ 共 512 帧
    (2) First encoding: MLLM(V1, Tq) → MLLM 逐层计算 attention → DPC:
        Context path: s_j^l = Σ_{k∈C} a_{jk}^l + Σ_{h∈Vi} a_{hj}^l → top α_c 选 c_1^l
        Local path: ŝ_j^l = Σ_{k∈Vi} a_{kj}^l → top α_s 选 m_1^l
        → C1 = {c_1^1..c_1^L}(用于下一轮), M1 = {m_1^1..m_1^L}(→ M_bank)
    (3) Iterative (i=2..64): MLLM(C_{i-ns},...,C_{i-1}, Vi) → DPC → Ci, Mi → M_bank
    (4) Recall: 从 M_bank 计算 g_i → top na 连续 memory → MLLM(M_i..M_{i+na-1}, Tq) → Y
    关键差异：baseline 一次性输入 64 帧 full tokens 解码；FlexMem 迭代压缩 512 帧后仅用 13k tokens 解码，帧覆盖量 8× baseline。
  - 系统框架层：PyTorch + HuggingFace Transformers。Training-free 即插即用，无 fine-tuning，无框架修改。
  - 编译框架层：论文未明确说明。
  - kernel 调度层：标准 FlashAttention。DPC 的 attention score 计算完全在 MLLM 已有的 forward pass 中完成，无额外 kernel。
  - 硬件架构层：单张 NVIDIA RTX 3090 GPU (24GB)。FlexMem 在 24GB 下可处理 1024 帧（baseline 仅 64 帧），且性能仅损失 0.5%。

## See It, Say It, Sorted: An Iterative Training-Free Framework for Visually-Grounded Multimodal Reasoning in LVLMs

- baseline方法是什么？
  Baseline 是标准的 LVLM greedy decoding（自回归逐 token 生成）。在长链 CoT 推理中，base LVLM 在每一步根据 p_VLM(x_i | x_{<i}) 选择 top-1 token。由于随着上下文增长，语言先验逐渐压倒视觉线索，单步幻觉 token 会级联传播至后续步骤，最终导致错误答案（Fig. 1(a)）。
  
  Baseline 全栈执行例子（以 Qwen2.5-VL-7B 回答 TreeBench 视觉推理问题为例）：
  - 算法层：自回归 greedy decoding。LVLM 编码图像和文本指令后，逐 token 解码 CoT 推理链，每步选择 p_i 中概率最高的 token。无外部监督，无证据注入。若某步模型将颜色"blue"错选为"red"，后续的定位、描述和最终判断全部基于错误前提。
  - 系统框架层：PyTorch + HuggingFace Transformers。标准 generate() 调用，无额外 decoding wrapper。单次 forward pass 输出 logits → argmax → 追加到 prefix。
  - 编译框架层：论文未明确说明。
  - kernel 调度层：标准 FlashAttention 实现 attention 计算。Decoding 阶段每步 attention 为增量计算（仅计算新增 token 的 Q 与历史 KV）。
  - 硬件架构层：单张 H20-NVLink GPU。Greedy decoding 下 latency 由 t_0 描述（V*Bench 8.98s、MathVista 12.92s 等，per question），visual decider 调用次数 r=0（δ=0）。

  问题：greedy decoding 在 hallucination-prone 步无防御机制。一旦某个中间 token 偏离视觉事实，后续逻辑推理——即使形式正确——也全部基于错误前提，导致 cascading failure。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法 ECRD（Evidence-Constrained Reweighting Decoding）通过在 test time 注入视觉证据来监督每一步的 token 选择，解决 greedy decoding 中"单步幻觉→全链崩溃"的问题。不像 RL-based 方法（DeepEyes、Pixel-Reasoner）需要在训练时学习"何时看图"，ECRD 在推理时按需获取视觉证据。

  核心设计映射到 baseline 缺陷：
  (1) **分布监督器替代 greedy argmax**：不再直接取 top-1，而是先 knee truncation 选出候选集 C_i，再用证据池中的文本证据计算证据诱导分布 r_i(w)，最终与 base 分布 p_i(w) 通过自适应权重 α_i = p_{(1)} 混合。当 base 分布尖锐时（α→1），保持 base 主导；当 base 分布平坦时（α 小），证据获得更多权重。—— 解决了"自信但有偏差的 token 选择"问题。
  (2) **不确定性触发的 visual decider**：当 k*>1 且混合分布 margin Δ_i ≤ δ 时，调用 GRIT-3B 读图并输出微观察证据句。证据句强制提交正确 token + 追加到证据池供后续步骤复用。—— 解决了"关键歧义步无外部仲裁"的问题。
  (3) **文本证据池累积与复用**：证据以文本形式存储（非像素），可在后续步骤中被 supervisor 的 scoring 函数直接参考（式 5-7），无需反复编码图像。—— 解决了 RL-based 方法中"每次看图需重新编码裁剪区"的效率问题。

  ECRD 全栈执行例子（以 TreeBench 问题"直接位于 favorita 品牌香蕉纸箱后面的物体是什么？"为例）：
  - 算法层：
    ```
    Step i（关键歧义步）：
    C_i = {"5", "3"}  # knee truncation 选出候选
    base: p("5")=0.498, p("3")=0.483
    evidence pool 评分 → evidence-induced: r("5")=0.503, r("3")=0.478
    alpha = p_{(1)} ≈ 0.498（base 不自信，alpha 小）
    p_mix: ("5", 0.501), ("3", 0.480)
    margin = 0.021 ≤ δ=0.08 → 触发 decider
    GRIT 读图 + 当前 prefix → w*="3", 
      E_i="The number behind the cardboard box with the 'favorita' brand and banana illustration is '300'."
    强制选 "3"，证据句追加到池
    Step i+1: evidence pool 含上述证据句
      supervisor 评分 → "0" 获得证据支持 → 选 "0"
    Step i+2: 同上 → 选 "0"
    最终答案: "300"（正确）vs greedy 选 "5" 导致 "5XX"（错误）
    ```
  - 系统框架层：PyTorch + HuggingFace Transformers。ECRD 作为 decoding wrapper 包裹 frozen LVLM，不修改模型权重。Visual decider（GRIT-3B）独立部署在另一 backend（FP16 on CPU），仅在触发时调用。证据 scoring 的计算复杂度 O(k*|E_i|)，k* 为个位数，|E_i| 增长缓慢，GPU 压力可忽略。
  - 编译框架层：论文未明确说明。
  - kernel 调度层：标准 FlashAttention + 推理框架自带的 kernel。Evidence scoring 在 CPU 上完成（precomputed log-likelihoods），不占用 GPU compute。
  - 硬件架构层：单张 H20-NVLink GPU。ECRD 的 overhead 来自两部分：(i) 证据评分 —— O(k*|E_i|)，<0.1s per step；(ii) visual decider 调用 —— l_0 ≈ 1.12-1.46s/call，δ=0.08 时每问题平均调用 r 在低个位数。总 latency T(0.08) ≈ t_0 + l_0·r ≈ 10-15s，相比 t_0(≈9-13s) overhead 控制在 20-30%，而 accuracy 提升 4.5-10.9 个点。

  关键差异对比：
  | 维度 | Baseline (Greedy) | ECRD (Ours) |
  |------|------------------|-------------|
  | Token 选择 | argmax(p_i) | 协商混合 p_i + r_i，自适应 α |
  | 视觉监督 | 无（仅初始编码一次图像） | 证据池持续评分 + 按需 decider 注入微观察 |
  | 幻觉处理 | 无防御，单步错→全链错 | margin 检测歧义步，decider 仲裁 |
  | 训练需求 | 无（但性能差） | 无（training-free，frozen models） |
  | 证据形式 | 无 | 文本（可复用，无需重编码图像） |
  | Cost | 最低（单次 forward/step） | 少量 overhead（证据评分 + 按需 decider） |

## SlowFast-VGen: Slow-Fast Learning for Action-Driven Long Video Generation

- baseline方法是什么？
  Baseline 方法分为两类：
  
  **(A) 动作条件视频生成模型**：AVDC（image-space diffusion，仅在 image space 做 video policy）、AnimateDiff（个性化 T2I 动画扩展）、SEINE（short-to-long 生成过渡帧）、iVideoGPT（交互式自回归 transformer）。这些模型通过在大规模数据上预训练（类比 slow learning）来构建世界模型，但受限于 context window（如仅 16 帧/4秒），无法记忆超出当前窗口的轨迹，导致长视频中 temporally distant frames 不一致。
  
  **(B) 长视频生成模型**：Streaming-T2V（conditional attention + appearance preservation module + video enhancer）是最先进的 long video generation 方法。它通过 anchoring 一个 anchor frame 来保持全局 context，配合 conditional attention 逐 chunk 生成。然而其 appearance preservation module 仅使用单一 anchor frame，无法存储完整的情节记忆（episodic memory），导致回访先前场景时一致性差。
  
  全栈执行例子（以 Streaming-T2V 为例）：
  - 算法pipeline层：给定 text prompt → 生成首帧 → 用 conditional attention module 以 anchor frame + 前 chunk 条件生成后续 chunk → 逐 chunk 串联成 long video。仅使用最后一个 chunk 和 anchor frame，超出窗口的内容被遗忘。
  - 系统框架层：PyTorch + Diffusers 生态。模型为 latent video diffusion model（基于预训练 text-to-video model），使用 3D UNet denoiser。推理时在 latent space 操作，不经过 pixel 编解码循环。
  - 编译框架层：论文未明确说明。
  - kernel 调度层：标准 VAE encoder/decoder + UNet 推理 kernel，FP16 推理。
  - 硬件架构层：论文中 baseline 实验均在 V100 GPU 上运行（统一比较），AVDC 需 image-space diffusion 故显存和推理时延更高。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 SLOWFAST-VGEN，通过三层设计解决上述缺陷：
  
  **(1) Slow Learning — Masked Conditional Video Diffusion（解决 baseline A 的 slow learning 质量问题）**：基于 ModelScopeT2V 修改，对前序 chunk 做 masked conditioning（past frames 保持 clean 不做 denoising loss），生成后续 chunk。配合自采集的 200k 多场景 action-video 数据集，大幅提升动作条件生成的 FVD（514 vs baseline 最高 782）。
  
  **(2) Fast Learning — TEMP-LORA（解决 baseline B 的记忆存储问题）**：核心创新。在推理过程中，每当生成新 chunk（输入 X_i + action C_i → 输出 Y_i），将 X_i 和 Y_i 的 latent 拼接后添加噪声，通过去噪训练更新 TEMP-LORA 参数 Θ_i。与 Streaming-T2V 的 anchor frame 不同，TEMP-LORA 参数中存储了整个生成轨迹的情节记忆。关键设计选择：
  - 对拼接序列**全加噪**（不含干净条件帧），使模型学习整个 trajectory 而非局部 transition
  - 训练时**不含文本条件**，专注于轨迹记忆
  - 遵循 local learning rule：ΔW 仅依赖当前迭代的局部 input-output 对
  
  **(3) Slow-Fast Learning Loop（解决长时规划任务中需跨 episode 泛化的问题）**：内层 fast learning 循环在每个 episode 上快速适配并积累 TEMP-LORA 参数；外层 slow learning 循环固定 TEMP-LORA，利用多 episode 的 (input, output, Θ) 数据更新核心权重 Φ，实现从单 episode 记忆到跨 episode 技能泛化。
  
  全栈执行例子（SLOWFAST-VGEN）：
  - 算法pipeline层：输入初始帧 X_0 + action 序列 → 逐 chunk 生成：Y_i = (Φ + Θ_i)(X_i, C_i) → 拼接 X_i' = X_i ⊕ Y_i → 加噪去噪 train TEMP-LORA 更新 Θ_{i+1} → Θ 参数累积整个 trajectory 的情节记忆 → 后续 chunk 生成时 Θ 保留了之前场景信息（如回访 Loc1 时场景一致）。可生成长达 1000 帧无明显退化。
  - 系统框架层：基于 ModelScopeT2V（latent video diffusion，CLIP text encoder + VAE + 3D UNet with spatial-temporal blocks）。Slow learning 阶段冻结 VAE/CLIP，仅训练 UNet (Φ)。Fast learning 阶段在 LoRA 低秩矩阵 (Θ, rank=32) 上做推理时训练，不修改 Φ。Video Planning 采用 UPDP：ChatGPT→子目标分解→逐 chunk 生成 video→逆动力学模型→action 执行。
  - 编译框架层：论文未明确说明。
  - kernel 调度层：标准 diffusion model UNet 推理 + LoRA 低秩矩阵乘加。TEMP-LORA 训练时额外一次前向+反向传播（单 V100，仅更新 LoRA 参数）。推理 overhead 仅 +6.8% 时延（12.93s→13.81s），显存增加 +3.7%（9579MB→9931MB）。
  - 硬件架构层：64×V100 训练，1×V100 推理。SCuts 从 0.89（Streaming-T2V）降至 0.37，FVD 从 782 降至 514，SRC 从 91.02 提升至 93.71。

  关键差异对比：
  | 维度 | Baseline (Streaming-T2V) | SLOWFAST-VGEN (Ours) |
  |------|--------------------------|----------------------|
  | 慢学习 | Text-to-video 预训练 | Masked conditional video diffusion + 200k 多场景 action-video 数据 |
  | 快学习 | 无（仅 anchor frame + conditional attention） | TEMP-LORA 参数存储全轨迹情节记忆 |
  | 记忆范围 | 单一 anchor frame（丢失中间轨迹） | 完整 trajectory（逐 chunk 累积 Θ） |
  | 记忆形式 | 图像像素（anchor frame 始终可见） | 低秩参数（ΔW = AB^T, r=32） |
  | 训练/推理 | 纯推理（无推理时训练） | 推理时 fast learning + 可选 slow-fast loop fine-tuning |
  | 学习回环 | 无 | Inner fast learning + Outer slow learning |
  | SCuts | 0.89 | 0.37 |
  | 最长生成 | 未明确（chunk-by-chunk 但内容漂移） | 1000 帧无明显退化 |

## T2Vid__Translating_Long_Text_into_Multi-Image_is_the_Catalyst_for_Video-LLMs

- baseline方法是什么？
  Baseline 是标准视频数据 fine-tuning 方案：使用纯视频 instruction 数据（ShareGemini 视频-描述对 + Video-ChatGPT 视频-指令对）对预训练 image-LLM 进行端到端 fine-tuning。全栈执行例子：
  - **算法层**：加载预训练 image-LLM（InternVL-4B 或 MiniCPM-8B），输入视频按 FPS=1 提取帧（InternVL max 64 帧，MiniCPM max 24 帧，超出则均匀降采样），每帧经 ViT 提取 visual features → MLP Projector 投影到 LLM embedding 空间 → 与 text token 拼接 → LLM 自回归生成答案。训练目标：最小化 $-\log p_\theta(\mathbf{A} \mid \mathbf{V}, \mathbf{Q})$。
  - **数据层**：ShareGemini（100K 视频-描述对，9 种模板变体的 "Describe this video in detail"）和 Video-ChatGPT（100K 视频-问答对，半自动标注）。两类数据可 1:1 混合采样。
  - **系统/训练层**：全量端到端训练（InternVL 冻结 vision encoder），lr=5e-6，关闭动态分辨率 patchifying。200K 全量数据需 276.8 GPU hours。
  - **kernel/硬件层**：论文未明确说明 GPU 型号，标准 PyTorch + Flash-Attention 2 训练环境。

  Baseline 的缺陷：
  1. **Instruction 多样性不足**：ShareGemini 仅用 9 种模板变体生成 instruction（t-SNE 可视化呈 9 个清晰聚类），Video-ChatGPT 因 self-instruction 和固定 prompting 模板也缺乏多样性。结果：即使数据量从 30K 扩到 100K（3.3×），Video-MME 整体准确率仅从 55.8 提升到 56.3（+0.5 points），呈对数增长趋势——数据效率极低。
  2. **数据冗余高**：短视频数据（多数 <30s）中视觉信息逐帧高度冗余，增加采样帧数（24→48）不能提升长视频理解性能（甚至略降），因为只引入更多冗余信息而非新信息。
  3. **标注成本高**：高质量视频 instruction 数据需要 Gemini-1.5-Pro API 调用或人工标注，规模化成本高。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **Sparrow** 数据增强方法，将纯文本 instruction 数据转化为类视频格式（text-to-image synthesis），混合真实视频数据训练，在不增加视频标注成本的前提下丰富 instruction 多样性。全栈执行例子（以 InternVL-4B + Sparrow 30K 混合数据为例）：

  - **数据合成层（核心创新）**：
    1. 从 LongAlpaca/LongQLora 取 (long_context, instruction, answer) 三元组
    2. NLTK 按 ~115 词分割 long_context 为多段 chunk
    3. 每段 chunk 用 Pillow ImageFont 渲染为 448×448 白底黑字图像（20pt Arial Regular, 黑色, 左右 20px margin）
    4. 输出 (synthetic_images[], instruction, answer) → 格式与真实视频完全一致
    5. 合成数据与真实视频以 1:2 比例混合（10K syn + 20K video = 30K hybrid）

  - **算法/训练层**：合成图像与真实视频帧以相同 pipeline 处理——ViT 编码 → Projector 投影 → LLM。训练协议与 baseline 完全相同（lr=5e-6, InternVL 冻 vision encoder），因此是完全 plug-and-play 的数据增强方案，不修改模型架构和训练代码。

  - **系统层**：30K hybrid 数据仅需 33.6 GPU hours（vs 276.8 GPU hours for 200K），效率 8.2× 提升。因为只需处理 15% 的数据量，存储和 I/O 开销也等比降低。

  - **kernel/硬件层**：论文未明确说明，与 baseline 相同环境（PyTorch + Flash-Attention 2）。

  解决 Baseline 缺陷的对应关系：
  1. **丰富 instruction 多样性** → 文本数据注入：文本 instruction 数据天然具有远高于视频 instruction 的多样性（LongAlpaca/LongQLora 覆盖书籍章节、学术论文、长文档等领域的问答），t-SNE 可视化显示 Sparrow 混合后 instruction 分布显著扩展。效果：30K hybrid 达到与 200K 纯视频数据相当的 Video-MME 性能（56.7 vs 56.3），且随数据缩放保持线性增益（100K hybrid vs 100K video: MVBench +4.3 points），消除 baseline 的对数增长瓶颈。
  2. **降低数据冗余** → 合成图像序列提供紧凑信息密度：一段 500 词的文本渲染为 4-5 张图像，每张图像内含密集文字信息（而非冗余视频帧），迫使模型学习从高信息密度视觉输入中提取语义，可能间接提升对关键帧的敏感度。
  3. **零额外标注成本** → 复用现有文本数据：无需调用任何 vision API 或人工标注，仅使用 PIL 渲染文本为图像。合成 10K 样本的成本几乎为零（纯 CPU 计算）。合成数据集已开源：https://huggingface.co/datasets/xjtupanda/Sparrow-Synthetic
  4. **意外收益：提升长视频理解** → 长文本上下文的时序推理迁移：即使训练数据不含长视频，hybrid 训练的模型在 LongVideoBench 上比 pure video baseline 高 6.6 points（100K 规模）。因为长文本中包含因果、时序、情节推理模式，这些推理能力可通过统一的 LLM backbone 迁移到长视频理解的跨帧时序建模中。
  5. **纯文本不够，必须转图像** → modality gap 桥接：如果直接用原始文本（不转为图像）混入训练，Video-MME Overall 仅 55.8（vs Sparrow 56.7），因为文本-视觉模态差异导致 training-inference mismatch。将文本转为图像使合成样本在视觉编码路径上与真实视频一致，消除了这个 gap。

  关键设计取舍验证：
  - 纯文本 vs 文本转图像：纯文本混合训练反而降低性能（Video-MME Long 从 48.1 降至 47.7），验证了 text-to-image 转换的必要性
  - 只用合成数据训练不可行：TOPA/T3 的纯文本合成方案极易饱和甚至降级（缺少真实视觉模式），合成数据只能作为正则化补充而非替代
  - 稠密采样帧无效：48 帧 vs 24 帧训练对长视频无增益（短视频冗余），更长上下文需从 LLM backbone 层面解决（continue pretraining）

## Speak_While_Watching__Unleashing_TRUE_Real-Time_Video_Understanding_Capability_of_Multimodal_Large_Language_Models

- baseline方法是什么？
  Baseline 是 Interleave Streaming（交替流式推理），基于 Qwen2.5-VL 的原生全局连续位置编码。全栈执行例子：
  - **算法层**：Qwen2.5-VL 使用 3D RoPE 位置编码（x, y, t），所有 token（视觉 + 文本）共享全局连续的位置索引空间。视觉 token 从 ViT 编码后经 MLP projector 映射到 LLM 嵌入空间，文本 token 由 LLM embed_tokens 产生，两者混合在一个连续位置空间中。
  - **推理流程**：在流式场景（wait-K 策略）下，第 i 步接收帧 → vision encoder 输出 m_i 个视觉 token → LLM 做 prefill（计算 KV cache）→ LLM 自回归解码生成 k_i 个文本 token → 第 i+1 步接收下一帧。每一步的视觉 token 位置紧跟在上一步文本 token 之后，形成严格的全局连续索引链 $0,1,...,E_{i-1},E_i,...$。
  - **系统层**：prefill 和 decode 必须严格串行——因为文本生成长度 k_i 不可预知，下一帧视觉 token 的起始位置 $E_i + k_i + 1$ 无法提前确定，导致 prefill 和 decode 无法并行。
  - **kernel/硬件层**：论文未明确说明。使用 PyTorch 标准推理，可选 Flash-Attention 加速。

  Baseline 的缺陷：
  1. 全局位置连续性约束 → prefill 和 decode 强制串行，延迟累加（$T_{\text{total}} = \sum_i (m_i/R_v + k_i/R_t)$）。
  2. 流式输出中视觉 token 插入打断文本序列（文本 token → 视觉 token → 文本 token 的交错 attention 路径），导致生成不连贯、重复、碎片化，BLEURT 从 Offline 的 53.21 骤降至 44.11，流利度从 4.84 降至 2.84。
  3. 对调度扰动（如不规则的帧到达率或生成速率变化）极度敏感——Random schedule 下 BLEURT 进一步降至 40.56，流利度大幅下降。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出三种打破全局位置连续性的位置编码策略（OSPE, GDPE, GIPE），实现视觉感知和文本生成的并行流式推理。以最优的 GDPE 为例，全栈执行例子：
  - **算法层**：GDPE 将视觉 token 和文本 token 分配为两个独立的位置组，每组内部连续但组间解耦。视觉组从 pos_v=0 开始递增，文本组从 pos_a=0 开始递增，互不依赖。训练时通过自定义 causal mask 确保 $V_{i+1}$ 只 attend 到 $V_{1..i}$，$A_i$ 只 attend 到 $V_{1..i}$ 和 $A_{1..i}$。仅需在 Qwen2.5-VL 上做少量 SFT（20K 样本），无需修改模型架构。
  - **推理流程**：第 i 步：vision encoder 处理第 i+1 帧（prefill）与 LLM 自回归生成第 i 步的 k_i 个文本 token 并行执行。视觉 token 位置基于 pos_v 独立递增，文本 token 基于 pos_a 独立递增，不再互相阻塞。
  - **系统层**：理论上可用双 GPU 或双计算流并行执行 prefill 和 decode，将每步延迟从 $T = m/R_v + k/R_t$ 降低到 $T = \max(m/R_v, k/R_t)$。代码仓库目前实现的是位置编码层面的并行设计（单 GPU 逻辑并行），真实多 GPU 并行未实现。
  - **kernel/硬件层**：论文未明确说明。

  解决 Baseline 缺陷的对应关系：
  1. **打破串行依赖** → 位置空间解耦：GDPE 通过独立位置计数器消除了 "必须等文本生成完才知道下一视觉 token 起始位置" 的依赖，使 prefill 和 decode 可重叠执行，理论加速比最高 2×（$r \approx 1$ 时）。
  2. **修复文本连贯性** → causal mask 重排：GDPE/GIPE 的 causal mask 确保文本 token 不再被后续视觉 token 打断 attention 路径，保持文本序列的连续注意力。结果：Streaming 下 GDPE 流利度 4.56（vs Interleave 2.84），GIPE 流利度 4.85（接近 Offline 的 4.84）。
  3. **提升鲁棒性** → 独立索引空间：视觉和文本的索引空间独立后，即使帧到达率/生成速率波动，两者的位置分配互不干扰。Random schedule 下 GDPE BLEURT 51.76（甚至略优于 fixed 的 51.53），而 Interleave 从 44.11 降至 40.56。

  三种策略的 trade-off：
  - OSPE：视觉和文本从同一 max 位置起共享索引，文本段内连续但跨段非连续 → 流利度 (4.48) 低于 GDPE/GIPE，BLEURT (50.62) 居中。
  - GDPE：视觉和文本完全独立组，组内连续 → 综合最优平衡，流利度 4.56，BLEURT 51.53，语义捕捉能力最好。
  - GIPE：GDPE 基础上在两组间加入大数值 gap → 流利度最高 (4.85)，但语义捕捉 (CIDEr/BLEU) 略低于 GDPE。

## StreamingEval__A_Unified_Evaluation_Framework_for_Streaming_Video_Understanding

- baseline方法是什么？
  现有流式视频理解的评估方法存在三个核心缺陷：(1) **评估设置不统一** —— 部分基准（如 VStream-QA, StreamingBench, OVO-Bench）使用 pseudo-streaming 设置：视频在 query 时间戳处截断但仍以离线方式处理，未模拟真实的增量帧到达和因果约束；(2) **指标单一** —— 仅关注 answer accuracy，完全忽略延迟（TTFT）、吞吐（MaxFPS）、资源消耗（memory usage）等部署关键指标；(3) **比较不公平** —— 在线模型和离线模型使用不同的评估协议（离线模型可访问全视频，在线模型受因果约束），且不同模型的 visual token 维度不同导致相同 token 数的实际内存占用不一致。

  Baseline 全栈执行例子（以典型 "pseudo-streaming" 评估 + offline VideoLLM 为例）：
  - **算法层**：加载预训练 VideoLLM（如 Qwen3-VL-8B），在评估时加载完整视频的所有帧到 GPU → Vision Encoder 一次性编码所有帧 → visual tokens + text tokens 拼接 → LLM 自回归解码。对于时间戳 query，仅使用 query timestamp 之前的帧子集，但仍以批量 offline 方式处理（可同时访问 future frames 做 context）。
  - **Serving/系统层**：标准 HuggingFace Transformers 推理 pipeline（model.generate），无帧级增量处理，无 memory budget 约束，KV cache 可无限增长直到 OOM。
  - **评估层**：仅计算 accuracy（如 OVO-Bench 的 Real-Time/Backward/Forward 三类任务平均分），不测量编码延迟、解码延迟、内存占用。
  - **kernel/硬件层**：FlashAttention-2 + Accelerate 加速，单卡 RTX 4090 BF16 推理。论文未明确说明 baseline 评估框架的细节。

  Baseline 的缺陷：
  1. Pseudo-streaming 无法反映真实部署条件：模型实际可一次性加载全视频做 context，与真实在线场景下帧逐帧到达、仅能访问过去的约束不一致。
  2. 仅用 accuracy 评估误导：离线模型 accuracy 高但实际部署时可能因 encoding 太慢（<1fps）无法跟上视频流，或 decoding 延迟过大破坏交互体验。例如 VideoChatOnline-4B 在 OVO-Bench accuracy 仅为 40.40 但 MaxFPS 仅 0.14（远低于 1fps 要求），声称 "在线" 但无法实际部署。
  3. Token 数预算不公平：不同模型 visual token embedding 维度不同，相同 "256 visual tokens" 内存占用差异可达 2-3×。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 StreamingEval，一套统一的流式视频理解评估框架。以评估 Qwen3-VL-8B 在 OVO-Bench 上为例，全栈执行例子：

  - **算法层**：加载预训练 VideoLLM（Qwen3-VL-8B），不做任何模型修改。离线模型通过 bounded-memory adapter 接入：视觉 encoder 逐帧编码 → MLP projector 映射到 LLM embedding 空间 → 写入固定容量 memory bank（FIFO 淘汰）。在线模型（如 Flash-VStream）保留原生 streaming mechanism（增量编码 + 记忆/状态更新 + retrieval policy）。

  - **Serving/系统层（核心创新）**：StreamingEval 实现三进程异步 pipeline：
    1. **Frame Player**：以 1fps 固定帧率提取和发送视频帧到下个进程。
    2. **Encoder & Memory Updater**：接收帧 → vision encoder 编码（$z_i = g_\theta(v_i)$）→ 按照模型特定更新规则更新 memory state $M_{\tau_i^+} = \mathcal{U}(M_{\tau_i^-}, z_i; B, \pi)$，其中 B 是字节级 memory budget，π 是淘汰策略（离线模型用 FIFO，在线模型用原生策略）。
    3. **Responder**：用户 query 到达时，编码 query，读取当前 memory snapshot $M_{t1}$，条件于对话历史 $C_{t1}$ 和 query $q_{t0}$ 做自回归生成 $R_{t1} \sim p_\phi(\cdot | q_{t0}, C_{t1}, M_{t1})$。
    三个进程通过 inter-process queues 通信，无额外同步阻塞。

  - **评估指标层**：四维指标 + 综合 StreamingScore：
    - MaxFPS：模型可维持的最大输入帧率（编码吞吐上限）
    - TTFT (Time-to-First-Token)：从 query 到达到首个 token 生成的 wall-clock 时间
    - Memory_bank：在线可用历史视觉缓存的字节预算
    - Accuracy：流式在线 QA 的正确率
    - StreamingScore = (MaxFPS^{w_f} × Acc^{w_a}) / (TTFT^{w_t} × M^{w_r})，其中 M = Mem × ln(Params)，权重可按部署偏好调节（Best Answer/Interaction First/Resource-Saving/Throughput First）

  - **字节级统一资源预算**：不按 visual token 数量约束，而是：
    $\text{Mem}_i(B) = B \cdot d_i \cdot s_{\text{emb}} + B \cdot 2L_i \cdot h_i^{\text{kv}} \cdot s_{\text{kv}}$
    计算 visual token embedding + 关联 KV cache 的总字节数，反推出模型特定的 visual token 上限 $B_i = \lfloor M_{\text{bytes}} / (d_i s_{\text{emb}} + 2L_i h_i^{\text{kv}} s_{\text{kv}}) \rfloor$。

  - **kernel/硬件层**：单卡 RTX 4090 48GB (BF16)，FlashAttention-2 + Accelerate 加速。三进程 pipeline 的 inter-process 通信开销可忽略。

  解决 Baseline 缺陷的对应关系：
  1. **真实因果约束** → 三进程异步 pipeline：Frame Player 按固定帧率发送 → Encoder 逐帧增量编码 → Responder 仅能访问 t1 时刻的 memory snapshot（不含未来帧）。严格保证了 streaming 评估的真实性，消除了 pseudo-streaming 设置的失真。结果：VideoChatOnline 的 MaxFPS 仅 0.14（远低于 1fps），揭示了声称 "在线" 的模型实际无法部署。
  2. **多维部署导向评估** → 四维指标 + StreamingScore：超越 accuracy 单一指标，同时量化延迟、吞吐、资源消耗。例如 Qwen3-VL 在 OVO-Bench accuracy 最高（58.00 vs StreamForest 55.57），但 StreamForest 的 StreamingScore 在特定权重下超过 Qwen3-VL（因更低延迟和内存），揭示了 accuracy 和 deployability 之间的系统性 trade-off。
  3. **统一资源预算** → 字节级 memory budget：将不同 embedding 维度的模型归一化到相同字节预算下比较，消除了 token 数预算的不公平。Memory_bank 从 0.1G→1.5G 的 sensitivity 实验表明 accuracy 在 1.0G 以上近乎饱和，为实际部署的资源分配提供了量化指导。
  4. **场景感知评估** → 可调权重的 StreamingScore：支持 Best Answer (w_a=0.4)、Interaction First (w_t=0.4)、Edge Resource-Saving (w_r=0.4)、Throughput First (w_f=0.4) 四种部署偏好，不同场景下模型排名可互换（如 Qwen3-VL 在 Best Answer 排第 1 但 Flash-VStream 在 Interaction First/Resource-Saving/Throughput First 均排第 1），但整体趋势统计稳健（Spearman ρ ∈ [0.972, 0.993]）。

> 注意：本论文属于评估框架/基准测试论文，非新的模型算法或系统实现。核心贡献是标准化评估协议，使不同模型在统一流式约束下可公平比较，并量化 accuracy-latency-throughput-resource 的多维 trade-off。

## TSPO__Temporal_Sampling_Policy_Optimization_for_Long-form_Video_Language_Understanding

- baseline方法是什么？
  Baseline 为 Video-MLLM 的 **uniform frame sampling**（均匀帧采样）和 **training-free keyframe search**（无训练关键帧搜索）。全栈执行例子：
  - 模型推理算法层：Video-MLLM（如 LLaVA-Video-7B）对长视频以固定 FPS（如 1 FPS）均匀采样 64 帧，所有帧权重相同，不考虑查询内容。Training-free 方法如 LongVU 使用 DINOv2-1B 提取帧间差异选择关键帧，或 CoS 使用 LLaVA-1.5-13B 进行查询相关帧过滤——但这些 selector 是预训练模型，无法针对 Video-MLLM 的最终任务进行优化。所有方法中帧采样和语言生成是两个独立阶段。
  - 系统框架层：论文未明确说明。使用标准 Video-MLLM 推理流程：视频解码→帧采样→视觉编码→token 拼接→LLM 自回归生成，无专门的调度或编译框架。
  - 编译框架层：论文未明确说明。
  - kernel调度层：论文未明确说明。使用 PyTorch + DeepSpeed 标准训练栈，Flash-Attention 加速注意力计算。
  - 硬件架构层：论文未明确说明。运行在 8×NVIDIA A800 80GB GPU 上。

  Baseline 的核心缺陷：
  1. **无监督性 (Unsupervised)**：通用视频理解训练中缺乏帧级标注，uniform sampling 无法知道哪些帧对回答关键，training-free 方法依赖预训练 selector 的跨模态理解能力，无优化空间。
  2. **不可微性 (Non-differentiable)**：帧采样是离散子集选择问题，输出为帧索引而非连续变量，无法通过 SFT 反向传播直接优化采样策略。
  3. **计算冗余**：Training-free 方法如 CoS 额外调用 MLLM-13B 做帧选择，推理开销大（28.4s frame time vs TSPO 1.2s）。
  4. **查询无关**：Uniform sampling 对所有查询采样相同帧，忽略查询-事件关联。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  TSPO 将关键帧选择和语言生成建模为联合决策过程，通过强化学习（GRPO）端到端优化时序采样策略。全栈执行例子：
  - 模型推理算法层：
    (1) **Event-aware Temporal Agent**：基于 CLIP-Large（400M 冻结）+ 3.5M 可学习参数。输入候选帧（1FPS 均匀采样）和查询文本的 CLIP 特征 → local window attention 注入事件感知和时序位置编码 → 融合 event-level 和 frame-level 的 cross-modal similarity → Gumbel-Softmax 概率化 TopK 采样 → 输出关键帧索引和概率。
    (2) **TSPO RL 优化**：将采样策略 π_ts(V_s|q,V_c) 和语言生成 π_l(o|q,V_s,V_c) 联合建模为 π(o,V_s|q,V_c) = π_l · π_ts。Video-MLLM (π_l) 保持冻结，仅通过 GRPO 优化 Temporal Agent (π_ts)。GRPO 对每组 query 采样 G 个关键帧组合，以 rule-based reward（答案准确性 R_A + 时序定位 R_T）计算组内相对优势 A_i，最大化期望奖励。无需帧级标注——语言级答案正确性（多选题选项匹配）直接监督帧选择策略。
    (3) **双风格训练数据**：Comprehensive Temporal Data（过滤太易/太难的多选题，保留需多关键帧的题目）+ Video Needle-in-a-Haystack Data（合成超长视频 10∼60min，训练长程时序定位能力）。
    推理时去除 Gumbel 噪声，直接确定性采样 64 帧（可降至 32 帧仍超 baseline），比 CoS 节省 90% 帧提取时间。
  - 系统框架层：论文未明确说明。基于 DeepSpeed 分布式训练，Video-MLLM 骨干可替换（迁移实验验证了 LLaVA-Video→Qwen2VL/Qwen2.5VL 的 zero-shot 迁移能力）。
  - 编译框架层：论文未明确说明。
  - kernel调度层：论文未明确说明。
  - 硬件架构层：论文未明确说明。

  **解决 Baseline 缺陷的映射**：
  1. 无监督性 → **语言级奖励替代帧级标注**：TSPO 利用最终回答的准确性（R_A）和粗粒度定位准确率（R_T）作为奖励信号，language supervision 通过 GRPO 的期望最大化间接指导帧选择，无需任何帧级 ground-truth 标注。
  2. 不可微性 → **RL 替代 SFT 反向传播**：Gumbel-Softmax 提供可微的离散采样近似，GRPO 的 policy gradient 方法天然处理离散动作空间（帧索引选择），避免了对不可微采样的直接梯度需求。
  3. 计算冗余 → **轻量级 Temporal Agent (3.5M)**：相比 CoS 使用 MLLM-13B 做帧选择（28.4s），TSPO 的 CLIP-based agent 仅需 1.2s 帧提取时间，且推理时可降低采样帧数（32 帧实现 token 减半、LLM 时间减半）。
  4. 查询无关 → **查询驱动的自适应采样**：Event-aware agent 计算帧-查询 cross-modal similarity，对每个查询动态选择不同的关键帧组合，而非对所有查询使用相同帧。

## Test-Time Temporal Sampling for Efficient MLLM Video Understanding

- baseline方法是什么？
  Baseline 是标准 MLLM 视频推理 pipeline：从视频 F 帧中均匀或规则采样子集 N 帧，每帧经视觉编码器 E_v 编码为 M 个 patch token，共 L = N×M 个视觉 token。可选地经压缩器 C 缩短后，与文本 token 拼接送入 MLLM 做自回归解码。全栈执行例子：
  - 算法层：规则帧采样（如 Qwen2.5-VL 的均匀采样），全量 token 送入 self-attention，每个 token 与所有 L 个 token 计算 attention（O(L²)）。关键帧可能因采样稀疏被遗漏，且相邻帧的冗余 patch 浪费 attention 计算。
  - 系统框架层：VLMEvalKit 评估工具包，单次前向传播处理完整长序列。
  - 编译框架层：论文未明确说明。
  - kernel调度层：论文未明确说明。
  - 硬件层：单 GPU 推理，长序列 self-attention 导致 GPU 显存和延迟瓶颈。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 T3S，通过随机多试次采样和聚合，将单长序列替换为多个短且多样化的子序列，利用视频时空冗余在降低 attention 成本的同时保持或提升覆盖率。全栈执行例子：
  - 算法层：对视频随机采样 m 个独立的 N 帧子集并分别做 token 子采样（保留率 αᵢ），m 个短子序列打包在一个前向传播中处理（块对角线 attention mask），最后对各试次 logit 进行聚合（均值/置信度加权/双试次交叉验证）。Attention 复杂度从 O(L²) 降为 O(∑αᵢ²L²)，m=2、α₁=0.5、α₂=0.3 时理论降为 0.34L²。随机采样的无偏性保证多试次统计上覆盖关键时间片段，弥补单试次可能的遗漏。
  - 系统框架层：VLMEvalKit + 自定义 inference wrapper，序列打包（packing）实现多子序列单次前向传播。对 Qwen2.5-VL-7B 在 LongVideoBench 上准确率提升 3.1%，首次 token 延迟降低 2.04×。
  - 编译框架层：论文未明确说明。
  - kernel调度层：论文未明确说明。
  - 硬件层：单 GPU 推理，由于每个 attention 块更小且打包后总序列长度更短（0.8L vs L），显著减少 GPU 计算和显存占用。论文 4.5 节指出单 GPU 上各 chunk 并行计算已使硬件饱和，多 GPU 可将各试次分配到独立设备进一步加速。

  **Baseline 缺陷 → 论文设计对策**：
  1. 缺陷：规则采样无语义感知 → 对策：随机多试次采样，统计上无偏覆盖时间轴，避免遗漏关键帧。
  2. 缺陷：全量 token 的 O(L²) attention 开销 → 对策：token 子采样 + 多短序列打包，每个 attention 块更小，总复杂度降低。
  3. 缺陷：需要额外训练或模型修改（如 learned selector、memory summarization）→ 对策：完全训练无关，即插即用于任何预训练 MLLM。
  4. 缺陷：学习型选择器在推理时仍需先处理所有帧再选择 → 对策：随机采样在前，无需预处理全量帧。

## TimeLens: Rethinking Video Temporal Grounding with Multimodal LLMs

- baseline方法是什么？
  Baseline 是使用 Qwen2.5-VL-7B（或 Qwen3-VL-8B）基础 MLLM 直接做 Video Temporal Grounding（VTG），未经专门的 VTG 后训练。模型使用 MRoPE（Multimodal Rotary Position Embedding）将视频帧的空间和时间维度编码到 position embedding 中，通过 SFT 在通用多模态数据上训练，未针对时间定位任务进行专门优化。

  Baseline 在模型推理算法-系统框架全栈的执行例子：

  - **算法/模型推理层**：给定视频 v 和文本查询 q（如 "When does the person turn off the light?"），Qwen2.5-VL-7B 按 2 FPS 采样视频帧，vision encoder 将相邻两帧合并为一个 patch embedding（每两个连续帧 merge），通过 MRoPE 注入 frame 的时间位置信息。LLM 接收 interleaved visual-text 序列，autoregressive 生成时间片段 `(t_start, t_end)`。由于 MRoPE 需要对 LLM 的 RoPE 机制进行底层修改，且未在大规模 VTG 数据上专门训练，模型缺乏精确的时间感知能力。
  - **系统框架层**：论文未明确说明。推断使用标准的 HuggingFace Transformers / vLLM 推理框架，无专门的 VTG 优化。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：论文未明确说明。
  - **硬件架构层**：论文未明确说明。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **TimeLens**，通过 **Data Curation + Algorithmic Design** 两大维度解决 Baseline 的三个核心缺陷：

  **缺陷 1**：Baseline 缺乏高质量 VTG 训练/评估数据 → 时间定位不准。
  **解决**：手动审查并重标注三大 benchmark（Charades-STA、ActivityNet Captions、QVHighlights）产出 TimeLens-Bench，自动化重标注训练数据产出 TimeLens-100K。原始 benchmark 中 Charades-STA 有 20.6% 样本违反 query 唯一性、34.9% 存在标注精度问题；TimeLens-100K 替代 noisy 训练数据后，mIoU 从 35.6 提升至 48.3（Charades-TimeLens）。

  **缺陷 2**：MRoPE position embedding 需要底层修改 LLM 的 RoPE 机制，难以在大规模重训中实用化，且时间感知精度不足。
  **解决**：采用 **Interleaved Textual Timestamp Encoding** — 将每帧的原始时间戳（如 "10.2s"）通过 LLM text tokenizer 转为文本 token，交错插入到对应帧的 visual tokens 之前。这无需修改 LLM 底层结构，利用 MLLM 现有的文本理解能力直接感知时间。实验中 Interleaved Textual + raw timestamp 在所有 encoding 方案中效果最优（mIoU: 48.3 vs MRoPE 36.6 on Charades-TimeLens）。

  **缺陷 3**：Baseline 的 SFT 训练范式在 VTG 任务上效率低，且 thinking-based RLVR 的显式推理对感知主导型任务无益。
  **解决**：采用 **Thinking-free RLVR (GRPO)** — 模型直接输出 `(t_start, t_end)` 而非 "think-then-answer" 格式。奖励函数简化为单一的 `r(y) = IoU(Ŝ, S*)`，无 format reward。训练效率 1.0×（约 4h10m on 8×H20），而 thinking-based RLVR 需要 1.9× 训练时间且性能更差。原因是 VTG 本质上是感知任务（perception-driven），显式推理过程被模型学成 bypass 的空操作（论文观察到 thinking 长度随训练收敛至简单内容）。

  TimeLens 方法在模型推理算法-系统框架全栈的执行例子：

  - **算法/模型推理层**：给定视频 v 和 query q，(1) 视频按 1 FPS 采样帧，每帧复制为两份以绕过 Qwen2.5-VL 的 frame merge 机制（同时让计算量等同 2 FPS）；(2) 每帧前插入文本时间戳 token（如 "10.2s"），使用 LLM text tokenizer 编码；(3) vision encoder 对每帧独立提取 visual tokens（frozen）；(4) 形成 interleaved 序列：`[prompt_tokens, timestamp_0, visual_0, timestamp_1, visual_1, ..., timestamp_T, visual_T]`；(5) LLM 在 GRPO 训练后直接 autoregressive 输出 `"The event happens in 5.2 - 12.7 seconds"`。推理时无需 thinking 过程，latency 低于 thinking-based 方法。
  
  - **训练/RLVR 层（系统框架层）**：(1) 离线阶段：用待训练模型对 TimeLens-100K 做 offline inference，计算每个样本的 difficulty `d_i = 1 - IoU(Ŝ_i, S*_i)`；(2) 按高斯分布 g(d; μ=0.05, σ=0.2) 进行 density-corrected 采样，获得约 12K 困难样本；(3) GRPO 训练：per prompt 采样 G=8 个 responses，对每个 response 计算 IoU reward，以 group 内 relative advantage `A^(g) = r^(g) - mean(r)` 更新策略；(4) 追踪 temporal IoU reward 和 group reward std，当两者 plateau 时 early stop（约 310 steps）。

  - **编译框架层**：论文未明确说明。使用 HuggingFace Transformers 标准训练流程（DeepSpeed ZeRO 或 FSDP 推断）。
  
  - **kernel调度层**：论文未明确说明。GRPO 的 8× roll-out 采样可能受益于 batch 推理加速，但论文未深入讨论。
  
  - **硬件架构层**：8 × NVIDIA H20 GPU 训练，vision encoder frozen 降低显存需求。推理时视频帧采样和 tokenization 在 CPU 预处理。

  对比总结（Baseline vs TimeLens on Qwen2.5-VL-7B, Charades-TimeLens）：
  | 维度 | Baseline (Qwen2.5-VL-7B) | TimeLens-7B |
  |---|---|---|
  | Timestamp Encoding | MRoPE (position embedding) | Interleaved Textual Prefix + Raw Timestamps |
  | 训练范式 | SFT (多任务通用) | Thinking-free RLVR (GRPO) |
  | 训练数据 | 通用多模态 SFT 数据 | TimeLens-100K (高质量 VTG 专用) |
  | 数据采样 | 随机 | Difficulty-based Gaussian Sampling |
  | 停止策略 | 固定 epoch | Early Stopping (reward plateau) |
  | mIoU | 39.3 | **48.8** (+9.5) |
  | R1@0.5 | 37.8 | **55.6** (+17.8) |

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

## VideoAuto-R1: Video Auto Reasoning via Thinking Once, Answering Twice

- baseline方法是什么？
  Baseline 是现有的 **"always-thinking" 视频推理模型**（Video-R1, Time-R1, VideoChat-R1），以及标准的 GRPO CoT 训练策略（RL with Thinking）。

  全栈执行例子（以 Video-R1 + Qwen2.5-VL-7B 为例）：
  - 模型推理算法层：Base model Qwen2.5-VL-7B-Instruct → GRPO RL 训练，对每个 prompt 采样 G=16 个候选输出 → CoT 格式为 `<think> reasoning trace </think> \boxed{final_answer}` → 使用 rule-based verifiable rewards（QA: exact match 0/1, Temporal Grounding: tIoU ∈ [0,1]）进行 group-normalized advantage 计算 → 单次 reward 仅监督最终答案 → 模型学习对所有输入强制生成长推理链。推理时对所有 query 均执行完整的 CoT 生成（greedy decoding, temperature=0），始终输出所有 reasoning tokens 再给出最终答案。Video-R1 在 VideoMME 上 CoT 推理生成平均 386 tokens 但仅 64.3% accuracy（比 direct 的 64.6% 下降 0.3%）。
  - 系统框架层：DeepSpeed + vLLM 加速 GRPO 训练（32 H100 GPU, ~35h）。推理使用 lmms-eval 框架 + Qwen2.5-VL/Qwen3-VL 标准推理 pipeline。无 Serving 框架修改。
  - 编译框架层：论文未明确说明。
  - kernel 调度层：标准 PyTorch + HuggingFace Transformers，无自定义 kernel。
  - 硬件架构层：32 NVIDIA H100 GPU（训练），无硬件定制。

  核心缺陷：
  (1) **视频 CoT 并非普遍有效**：表 1 表明在 VideoMME、LongVideoBench、MMVU 等感知导向 benchmark 上，Direct 推理与 CoT 推理准确率相当甚至更好（Video-R1: Direct 64.6 vs CoT 64.3；Time-R1: Direct 65.9 vs CoT 63.8）。CoT 仅在推理密集型 benchmark（VideoMMMU）上有 +1~3.4% 的有限增益。
  (2) **推理效率极低**：always-thinking 模型生成大量冗余 tokens（Video-R1 平均 386 tokens，RL with Thinking 标准 CoT 平均 149 tokens），而 direct answering 仅需 2-17 tokens。自回归 LLM 的解码延迟与 token 数线性相关，长推理链显著增加端到端延迟和推理成本。
  (3) **可能的过思考（Overthinking）效应**：感知导向任务（如物体识别、动作识别）中 CoT 冗余描述视频内容或逐步对比选项，但最终结论与 direct answer 相同。更严重的是，推理链中的单步幻觉可能将正确初步判断覆写为错误答案（图 7 示例：CoT 将 D 错误推理为 E）。
  (4) **视频领域"必须思考"样本稀缺**：与数学/编程等符号推理任务不同，视频中真正需要多步推理的样本极少（VideoMMMU 上 CoT-Direct gap 仅 +1~3.4%），这使得训练 mode-switching policy（think/no-think 标签）极不稳定。训练中若强制二分类决策，易导致 mode collapse（始终 think 或始终 no-think）。
  (5) **SFT Cold-Start 反而有害**：表 17 显示使用 Video-R1 CoT 数据进行 SFT 后 Qwen2.5-VL 性能从 66.0 退化至 60.1（VideoMME）。低质量 CoT 监督会扭曲强基模型的已有能力，而直接 RL 避免这种伤害。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **VideoAuto-R1** 通过三个核心设计解决上述缺陷：

  **(1) "Thinking Once, Answering Twice" 训练范式 → 解决缺陷 (1)(3)(4)(5)**

  Baseline → 每个样本被分流为 think 或 no-think 模式，需要 per-sample 标签，SFT 初始化有害。
  VideoAuto-R1 → 采用统一的 $\boxed{a_1} \rightarrow$ `<think>` $r$ `</think>` $\rightarrow \boxed{a_2}$ 模板。模型**始终学习生成初始答案+推理+审查答案**，而非训练二分决策。dual-answer reward $R = w_1 R_{task}^{(1)} + w_2 R_{task}^{(2)} + \lambda R_{fmt} + \alpha R_{fallback}$（$w_2 > w_1$）同时监督两个答案，鼓励最终答案优于初始答案。这消除了对 think/no-think 标签的需求，避免了 mode-switching 稳定性问题。直接 RL 训练（无 SFT cold-start）通过精心设计的 system prompt（Table 2）工程实现格式约束，无需预收集 CoT 数据。

  训练曲线（Figure 6）显示 $R_{task}^{(2)}$ 始终高于 $R_{task}^{(1)}$，验证推理阶段确实改善了答案。Fallback reward $\alpha = 0.3$ 在 $a_1$ 为 "Let's analyze..." 且 $a_2$ 正确时提供额外奖励（1.1+0.3=1.4 vs 错误猜测的 0），鼓励模型在无法立刻回答时诚实延迟而非低置信度猜测。

  **(2) Confidence-Based Early-Exit 推理策略 → 解决缺陷 (2)**

  Baseline → 推理时始终生成完整 CoT + 答案，冗余 tokens 大量增加延迟（149-386 tokens avg）。
  VideoAuto-R1 → 推理时解码至第一个 `<think>` tag 即暂停，提取 $\boxed{a_1}$ 的 tokens，计算 length-normalized mean log probability $s(a_1) = \frac{1}{L} \sum_{\ell=1}^{L} \log p_{\theta}(t_{\ell} \mid t_{<\ell}, q)$。若 $s(a_1) \geq \log \tau$（默认 $\tau = 0.97$），早停返回 $a_1$，避免生成后续推理链；否则继续生成 $r$ 和 $a_2$。

  注意这是 test-time only 决定，非训练时学习的切换策略。表 8 验证置信度与任务难度的相关性：感知 benchmark 上平均置信度 >0.93（think ratio ~25-28%，gain 仅 +0.1~0.4），推理 benchmark 上平均置信度降至 0.874（think ratio 51%，gain +4.0）。Recall of think-needed samples（$a_1$ 错误但 $a_2$ 正确的样本被路由到 reasoning mode）在 MVBench/MMVU/VideoMMMU 上分别为 100%/100%/94%，说明 confidence signal 有效捕获真正需要推理的样本。

  效果：平均响应长度从 149 tokens（RL with Thinking）降至 44 tokens（~3.3× 减少），Think ratio 自适应：MVBench 25% → VideoMMMU 51%。

  **(3) Multi-modal Training Data + Difficulty Filtering → 解决缺陷 (4)**

  Baseline → 仅用视频数据训练的 reasoning 模型偏向感知而非推理。
  VideoAuto-R1 → 混合 Text（6.4K DAPO-Math）+ Image（27.5K ViRL/ThinkLite-Hard）+ Video（49.4K）数据，增强符号推理能力。Difficulty filtering：每个样本采样 8 个 responses，全对或全错者丢弃（对 QA tasks），仅保留可学习的样本。表 11 显示 Text+Image+Video filtered 配置在 VideoMMMU 上从纯 Video 的 55.1 提升至 56.4，同时数据集减小 40%（138K→83K）。

  全栈执行例子（VideoAuto-R1 + Qwen2.5-VL-7B）：
  - 模型推理算法层：
    训练阶段 → 视频 v + 问题 q + system prompt（Table 2）输入 → base model Qwen2.5-VL-7B-Instruct（visual encoder 冻结，仅微调 projector+LLM）→ 使用 vLLM 加速 G=16 rollouts（temperature=1.0）→ 解析每 output 的 $\boxed{a_1}$、`<think> r </think>`、$\boxed{a_2}$ → 计算 dual-answer reward → GRPO group-normalized advantage → AdamW 更新。推理阶段 → greedy decoding 至第一个 `<think>` tag → 提取 $\boxed{a_1}$ → 计算 confidence $s(a_1)$ → 若 $s(a_1) \geq \log(0.97)$ 则早停返回 $a_1$（平均 ~10 tokens），否则继续生成 $r$ 和 $a_2$（平均 ~91 tokens）。按样本综合平均 44 tokens。
  - 系统框架层：DeepSpeed ZeRO + vLLM for rollout generation（训练），lmms-eval for evaluation（推理）。无 Serving 框架修改。
  - 编译框架层：论文未明确说明。
  - kernel 调度层：标准 PyTorch + HuggingFace Transformers，无自定义 kernel。
  - 硬件架构层：32 NVIDIA H100 GPU（训练），无硬件定制。

## VideoLLaMB: Long-context Video Understanding with Recurrent Memory Bridges

- baseline方法是什么？
  Baseline 是 **PLLaVA**（基于 Adaptive Pooling 的视频压缩方法）以及现有视频压缩策略（sampling、aggregation、semantic consolidation、resampling、video segmentation）。

  全栈执行例子（以 PLLaVA + Vicuna-7B 为例）：
  - 模型推理算法层：视频 V={v_1,...,v_n} → ViT-L/14 逐帧提取特征 → Adaptive Pooling 将 n 帧特征池化到固定 M 帧（如 16 帧）→ Linear Projector 投影 → Vicuna-7B 生成答案。核心机制是通过 pooling 丢弃部分帧信息来压缩视频长度。训练数据为 Video-LLaVA 视频数据 + LLaVA-1.5 图像数据，16 帧训练，推理可使用 32 帧（但需复用 16 帧训练权重）。EgoSchema 上 PLLaVA 16→16 帧仅 45.6%，32→16 帧降至 43.8%。
  - 系统框架层：基于 LLaVA-1.5 初始化，Video-LLaVA 的 image/video encoder，无 Serving 框架修改。
  - 编译框架层：论文未明确说明。
  - kernel 调度层：标准 PyTorch + HuggingFace Transformers，无自定义 kernel。
  - 硬件架构层：4× NVIDIA A800 GPU（训练），单 A100/A800（推理），无硬件定制。

  核心缺陷：
  (1) **Pooling 导致关键视觉信息丢失**：Adaptive pooling 将 n 帧强制压缩到固定数量，无论原始视频长度多少，必然丢弃大量视觉细节。在 NIAVH benchmark 中，pooling 策略在 haystack:needle 比例较高时几乎无法定位目标帧（Figure 3a），因为 needle 帧的视觉信息被 pooling 平均化淹没了。
  (2) **位置外推能力差**：基于 position extrapolation + sampling 的方法（如 LLaVA-NeXT-Video-DPO）在训练时仅见 32 帧，推理时对长于训练的序列预测能力急剧下降（Figure 3b），无法有效利用更长视频提供的信息。
  (3) **视频分割破坏语义流**：将视频均匀或按时间切分为短片段的方法（如 VideoStreaming、Video-XL）切断了跨片段的语义连贯性，使模型难以理解跨越场景边界的因果和时序关系。
  (4) **Resampler 压缩容量有限**：MA-LMM 等方法使用 resampler + memory consolidation，但 resampler 的压缩比固定，超长视频的信息编码最终受限于 resampler 的容量瓶颈（Figure 3c）。
  (5) **GPU 显存随视频长度线性增长**：无压缩方法（如 LongVA）直接将所有帧的视觉 token 送入 LLM，显存开销与帧数成正比，限制了可处理的视频长度。PLLaVA 通过 pooling 缓解但信息损失严重。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **VideoLLaMB** 通过三个协同设计的模块解决上述缺陷：

  **(1) SceneTiling 语义分割 → 解决缺陷 (3)**

  Baseline → 均匀切分破坏跨场景语义连贯性。
  VideoLLaMB → 基于 ViT [CLS] token 帧间余弦相似度计算 depth score $d_i = (cl_i + cr_i - 2c_i)/2$，按 μ+α·σ 阈值自适应检测语义边界，将视频划分为 K 个语义独立的 segment。每个 segment 内部帧高度相关，跨 segment 边界处为语义转换点，此处的信息损失不破坏语义完整性。消融实验（Table 8）显示替换为 uniform segment 后性能下降 1.8 点（EgoSchema: 53.8→52.0）。

  **(2) Recurrent Memory Bridge Layers → 解决缺陷 (1)(4)**

  Baseline → Adaptive pooling 丢弃信息；Resampler 有压缩容量上限。
  VideoLLaMB → 单层 Transformer Bridge Layer，每个语义段前 prepend 32 个 memory tokens，通过 self-attention 将当前段信息压缩入 memory tokens：$[m_{i+1}; o_i] = \text{BridgeLayer}([m_i; s_i])$。memory tokens 递归遍历所有语义段，逐步累积全视频信息，**不主动丢弃任何帧**。最终 LLM 仅接收 memory-token-augmented 的当前段视觉表示，输入 token 数固定（约 32+M 个 token per step），GPU 显存线性缩放（Figure 4），支持 320 帧处理（仅训练于 16 帧）。消融实验（Table 8）：移除 recurrent 机制（mean pooling）→ 51.61%（-2.19），移除（adaptive pooling）→ 49.4%（-4.4）。

  **(3) Memory Cache with Retrieval → 解决缺陷 (2)(4)**

  Baseline → 位置外推能力差；长视频中早期信息被遗忘。
  VideoLLaMB → 维护 MemoryCache = [m_1, ..., m_i]，以当前 m_i 为 query、历史 memory cache 为 key/value 进行 cross-attention 检索：$m_{i+1} = \text{Softmax}(W_i^Q m_i (W_i^K M_i)^\top / \sqrt{d_k}) W_i^V M_i$。实现 BPTT 绕过：梯度通过检索路径传播（仅到当前几步），避免 RNN 式的长时间反向传播梯度消失。在 NIAVH 320s 测试中（Figure 3d），VideoLLaMB 在 depth=12 处（needle 在视频 30% 位置）仍然保持最高分数 5.73（vs LLaVA-NeXT-Video-DPO 1.72, PLLaVA 1.82）。消融实验（Table 8）：移除 retrieval → 52.2%（-1.6）。

  全栈执行例子（VideoLLaMB + Vicuna-7B）：
  - 模型推理算法层：
    训练阶段 → 16 帧视频 → ViT-L/14 编码 → SceneTiling 分为 4 段 → 初始化 32 个 memory tokens → Bridge Layer（单层 Transformer, 8 heads, hidden=1024）逐段递归处理 → Memory Retrieval（单层, 8 heads, hidden=1024）更新 memory → Projector → Vicuna-7B 生成。仅训练 Bridge Layer + LLM（LoRA），冻结 ViT。Learning rate=2e-4, batch=8, epoch=1, warmup=0.03，4×A800 GPU 训练。推理阶段 → 支持动态段数（SceneTiling 自适应）或静态 4 段 → 处理 320 帧时 GPU 显存线性增长，4.21s 推理 300s 视频（Table 6），2.3s 用于特征处理 + 1.91s 用于生成。3000s 视频需 31.5s（23.4s 特征 + 8.1s 生成，Table 7）。
  - 系统框架层：基于 LLaVA-1.5 初始化，Video-LLaVA image/video encoder。开源代码包含 CLI、streaming、Gradio demo。streaming 模式下 SceneTiling 仅用左侧相似度 $d_i = (cl_i - c_i)/2$ 实时检测边界，无需预知全视频。
  - 编译框架层：论文未明确说明。
  - kernel 调度层：标准 PyTorch + HuggingFace Transformers，无自定义 kernel。Memory Cache 随视频长度线性增长但 token 数极少（每段仅 32 个 memory token），额外显存可忽略。
  - 硬件架构层：训练 4× NVIDIA A800 GPU，推理单 A100 (80GB)/A800 GPU。处理 320 帧仅需单卡 A100，显存线性缩放。

  对比 baseline 的解决效果：
  | 缺陷 | Baseline 表现 | VideoLLaMB 解决方式 |
  |------|-------------|-------------------|
  | Pooling 信息丢失 | PLLaVA EgoSchema 45.6%，NIAVH 1.82 | VideoLLaMB EgoSchema 53.8%（+8.2），NIAVH 5.73（+3.91） |
  | 位置外推差 | LLaVA-NeXT 32→长视频准确率骤降 | 16帧训练→320帧推理性能保持（Figure 2 dynamic segments） |
  | 语义流破坏 | uniform segmentation | SceneTiling 语义分割，+1.8 点 vs uniform |
  | 压缩容量瓶颈 | Resampler (MA-LMM) NIAVH 3.39 | Recurrent memory 无损累积，NIAVH 5.73 |
  | 显存线性增长 | LongVA 全 token 输入显存随帧数增长 | Memory Bridge 固定输入长度，显存线性缩放但仍可控 |
  | 推理时间 | MA-LMM 14.5s, MovieChat 143.7s (300s video) | VideoLLaMB 4.21s（压缩视觉输入使 LLM 处理时间最短） |

## VideoRoPE: What Makes for Good Video Rotary Position Embedding

- baseline方法是什么？
  **M-RoPE** (Wang et al., 2024a, Qwen2-VL) 是当前 Video LLM 最广泛采用的 3D position embedding 方案。它将 d=128 维 head 分为三组：前 32 维（高频率，θ_n = β^{-2n/d}）用于 temporal t，中间 48 维（中频）用于 horizontal x，后 48 维（低频）用于 vertical y。这种设计的核心缺陷通过 V-NIAH-D 任务暴露：

  **缺陷 1 — 时间维度高频分配导致周期性振荡**：低维对应的高频 θ_n 产生短单调区间，cos(θ_n·t) 在远距离上周期性重复。M-RoPE 中 t 使用前 16 个旋转角（如 θ_13, θ_14, θ_15），当帧号从 0 到 3000 时，cos(θ_n·t) 多次经过 0 产生"hash collision"——距离很远的两个位置有几乎相同的 temporal embedding。这使 distractor 帧（距 needle 200 帧插入）在 temporal 维度与 needle 不可区分，模型被误导。注意力可视化显示 M-RoPE 的 temporal 注意力集中在对角线附近（仅关注局部帧），实际定位 needle 依赖的是 vertical 维度而非 temporal 维度。

  **缺陷 2 — 无空间对称性（Spatial Asymmetry）**：M-RoPE 的 visual token 位置索引在每帧内从 (0,0) 到 (W-1,H-1)，导致每帧最后一个 visual token 总在 (W-1,H-1) 处形成"corner stack"，preceding text end 到 visual start 的距离 ≠ visual end 到 subsequent text start 的距离。

  **缺陷 3 — 无时间索引缩放**：M-RoPE 中所有维度使用相同的 index increment=1，不区分 temporal frame spacing 和 spatial pixel spacing 的差异。

  全栈执行例子（M-RoPE + Qwen2-VL-7B）：
  - 模型推理算法层：视频 → ViT 编码 → 每帧 W×H patches → 拼合文本→visual→文本序列 → M-RoPE 分配 position IDs：t 维度前 32 维（高频），x 维度中间 48 维，y 维度后 48 维 → 计算 RoPE rotation → Qwen2-7B LLM 前向 → 生成答案。
  - 系统框架层：vLLM 推理（>32K token 序列），无 Serving 框架修改。
  - 编译框架层：论文未明确说明。
  - kernel 调度层：标准 PyTorch + HuggingFace Transformers，FlashAttention 加速，无自定义 kernel。
  - 硬件架构层：NVIDIA A100 GPU 推理。704 GPU hours fine-tuning。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **VideoRoPE** 通过三个协同设计模块解决 M-RoPE 的缺陷：

  **(1) Low-frequency Temporal Allocation (LTA) → 解决缺陷 1**

  Baseline (M-RoPE) → t 使用低维（高频 θ_n），cos(θ_n·t) 在远距离上周期性重复。
  VideoRoPE → t 使用高维（低频 θ_n，dims 48-63 即 θ_48 到 θ_63），θ 值极小（如 θ_63 ≈ 10000^{-126/128} ≈ 0.00011），cos(θ_n·t) 在 3000 帧范围内几乎单调不减，不会产生 hash collision。x 和 y 维度交叉排列在低维（dim 0-47），因为它们处理的是分辨率受限的空间信息，高频足以覆盖所有空间位置。效果：V-NIAH-D 87.11%（M-RoPE 74.67%），temporal 维度在注意力中成功捕获长程 needle。

  **(2) Diagonal Layout (DL) → 解决缺陷 2**

  Baseline (M-RoPE) → 每帧 visual token 从 (0,0) 到 (W-1,H-1) 排列，corner stack。
  VideoRoPE → 整个输入沿对角线排列。第 0 帧中心 patch 坐标为 (Ts, Ts, Ts)，第 τ 帧中心 = (Ts+δτ, Ts+δτ, Ts+δτ)，其他 patch 偏移 ±(w-W/2, h-H/2)。preceding text (0..Ts-1 → 0..Ts-1) 到 visual start (Ts) 的距离 = visual end 到 subsequent text start 的距离，满足 Eq.5 对称性。

  **(3) Adjustable Temporal Spacing (ATS) → 解决缺陷 3**

  Baseline (M-RoPE) → 所有维度 index increment=1。
  VideoRoPE → 引入 δ=2，帧间 temporal index 增量为 2，而 spatial 和 text index 增量为 1，解耦时间与空间尺度差异。t = T_s + δ(τ-T_s)，x = t + w - W/2，y = t + h - H/2。

  全栈执行例子（VideoRoPE + Qwen2-VL-7B）：
  - 模型推理算法层：视频(128帧, 2FPS) → ViT 编码 → 构建 3D position IDs(t=T_s+2·f_idx, x=t+w-W/2, y=t+h-H/2) → d=128: dims[0:48] 交叉 x/y, dims[48:64] 为 t（低频 LTA）→ 计算 RoPE 旋转 → Qwen2-7B LLM → 生成。
  - 系统框架层：vLLM Serve-API 推理，Qwen2-VL fine-tuning pipeline（LR=1e-5, cosine scheduler, warmup=0.01）。训练 8K context，推理支持 128K。
  - 编译框架层：论文未明确说明。
  - kernel 调度层：标准 PyTorch + HuggingFace Transformers + FlashAttention。VideoRoPE 仅修改 position ID 计算和 RoPE dimension allocation，不涉及 kernel 修改。
  - 硬件架构层：训练 704 NVIDIA A100 GPU hours，推理单 A100。128K 推理 vLLM 支持。

  对比 baseline 的解决效果：
  | 缺陷 | M-RoPE 表现 | VideoRoPE 解决方式 | 效果 |
  |------|-------------|-------------------|------|
  | 时间维度高频振荡 | V-NIAH-D 74.67% (-4.0 vs V-NIAH) | LTA: 低频分配, 避免 hash collision | V-NIAH-D 87.11% (+12.44) |
  | 无空间对称性 | Corner stack, text-visual 非对称 | DL: 3D 对角线排列 | 施加式对称性 |
  | 时间/空间尺度统一 | index incr=1 所有维度 | ATS: δ=2 temporal scaling | Avg 60.92 (δ=2) |
  | LongVideoBench 64K | 54.35 | LTA+DL+ATS | 57.26 (+2.91) |
  | MLVU 64K | 61.10 | LTA+DL+ATS | 65.56 (+4.46) |
  | Temporal Hallucination | 29.0 | VideoRoPE | 58.5 (+29.5) |
  | 128K extrapolation | 51.45 (LVB) | VideoRoPE | 55.64 (+4.19) |

## VideoSeek: Long-Horizon Video Agent with Tool-Guided Seeking

- baseline方法是什么？
  - 现有 video agents（如 VideoAgent [ECCV 2024], VideoTree [CVPR 2025], DrVideo [CVPR 2025], MR. Video, DVD [NeurIPS 2025]）主要依赖两种范式：(a) 密集预处理：以 0.2–2 FPS 将全视频转为详细文本描述或结构化记忆数据库，然后基于数据库检索或推理回答；(b) 贪婪搜索：逐段扫描视频并调用 CLIP 等检索相关帧。这些方法在长视频上计算开销随视频长度线性增长，且论文发现 LVBench 中 >80% 的问题仅需 <5% 的帧即可回答——证明 exhaustive parsing 极低效。
  - 全栈执行例子：
    - 算法pipeline：DVD agent 在推理前以 2 FPS 构建 multi-granular 视频数据库（8,074 帧 for LVBench），再在数据库上执行 tool-augmented search。p(τ | X, Q) 被替换为预先构建的静态数据库查询，trajectory 不随推理动态演化。
    - 系统框架：DrVideo 先用 0.2 FPS（493 帧 for Video-MME）将长视频转为长文档，再用 LLM 索引和检索。推理和感知分离：感知固定在前处理阶段，推理阶段只能从预建文档中检索。
    - 编译框架/kernel调度/硬件架构：论文未明确说明（这些层次与 video agent 的算法设计无关）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - VideoSeek 不依赖预构建视频数据库，而是通过 think-act-observe 循环利用视频逻辑流（temporal order, causality）主动 seek 答案关键帧。三个多粒度工具（overview→skim→focus）按需调用，trajectory 随时间动态累积观测，使得每步的工具选择基于"前序观测揭示了什么 + 还需要什么"。
  - 全栈执行例子（沿一个 query 的数据流穿过整个 agent pipeline）：
    - 算法pipeline：给定 query Q 和视频 X，初始化 τ=[I, Q]。Turn 1：θ_think(GPT-5) 读 τ，判断零先验，调用 overview 获得 16α 帧的全局 storyline（如 "1480s 两人在 mall 入口交谈，旁有 B1 标志"）。Turn 2：基于 overview 的推测，θ_think 规划调用 skim(1465-1510s)，获得粗粒度扫描（4α 帧）确认 "1465-1497s: 两人在 B1 标识旁说话；1503s: 镜头转向红色高楼"。Turn 3：θ_think 决定调用 focus(1499-1507s, 1 FPS) 精确读取高楼上的中文文字，获得 "祝全市人民新春快乐"。Turn 4：θ_think 判断证据充分，调用 answer 工具输出最终答案 D。全过程仅观看约 93 帧（vs DVD 的 8,074 帧）。
    - 系统框架：VideoSeek 是一个 model-agnostic 框架，thinking model θ_think 可替换（GPT-5/o4-mini/GPT-4.1），视觉解释也由 θ_think 完成。系统 prompt（Figure 7-8）定义了 Role/Environment/State/Workflow/Toolkit/Operational Rules 六部分，规定了工具调用的约束（每轮仅一个工具，overview 仅用于冷启动，skim→focus 渐进细化）。
    - 编译框架/kernel调度/硬件架构：论文未明确说明（VideoSeek 是纯 prompt-based agent，不涉及编译/算子/硬件层次）。
  - Baseline 缺陷 → 方法设计选择的直接映射：
    - 缺陷 1：baseline 的密集预处理在长视频上计算开销不可接受。→ 方法：用 logic flow 引导的主动 seeking，仅在需要时采样帧，LVBench 仅用 ~92 帧（vs DVD 8,074 帧）。
    - 缺陷 2：baseline 的预建数据库是静态的，无法根据推理中间结果动态调整证据收集策略。→ 方法：全 trajectory τ 作为上下文，每步基于完整历史决定下一步工具和行为。
    - 缺陷 3：baseline 缺乏 coarse-to-fine 的分层探索能力，要么全局过粗要么局部过细。→ 方法：三工具互补设计——overview 提供全局 map，skim 快速缩小搜索空间，focus 精确验证细节。

## VisiPruner: Decoding Discontinuous Cross-Modal Dynamics for Efficient Multimodal LLMs

- baseline方法是什么？
  Baseline 是标准 MLLM（以 LLaVA-v1.5 7B 为典型代表）的 **dense 全 token 推理**：ViT 编码图像为 576 个视觉 token → MLP projector 映射到 LLM 维度 → 与 text token 拼接 → 32 层 LLaMA 2 transformer 逐层处理，每层对所有 visual tokens 执行 full cross-attention（Q_text 与 K_visual、V_visual 计算 softmax attention）+ visual self-attention + text self-attention + FFN，然后 autoregressive 解码。这种 dense 模式也被现有 training-free token pruning 方法（FastV、SparseVLM、PyramidDrop、FitPrune）继承，它们主要依赖 cross-attention weights 来选择保留哪些 token。

  全栈执行例子（以 LLaVA-v1.5 7B + LLaMA 2 7B 推理为例）：
  - 模型推理算法层：输入 336×336 图像 → CLIP-ViT-L/14（24层，patch size=14）编码为 576 tokens × 1024 dims → MLP Projector 映射至 4096 dims → 与 text instruction tokens（~74 tokens）拼接为 650 tokens 序列 → 送入 LLaMA 2 7B（32 layers, d=4096, 32 heads, GQA with 32 kv_heads, FFN intermediate=11008）。每层：① text tokens 对 full visual tokens 做 cross-attention（QKV 投影后 softmax 得到 N_text × 576 attention matrix），② visual tokens 之间做 self-attention，③ text tokens 之间做 causal self-attention，④ 各自过 FFN。32 层后 autoregressive 生成。首 token 延迟严重受 N_v=576 的 attention O(N_text × N_v) 影响；生成长回答时 KV cache 需存储全部 576 个 visual tokens × 32 层 = 18,432 个 visual KV entries，每个 entry 为 4096 × 2(K+V) × 2 bytes(FP16) = 16KB，总计 ~295 MB 仅用于 visual KV。
  - 系统框架层：LLaVA 标准推理 pipeline（HuggingFace Transformers 加载 LLaMA 2 + CLIP ViT），无特殊 serving 修改。推理时 visual token 的 attention 计算随 token 数平方增长。
  - 编译框架层：论文未明确说明。
  - kernel 调度层：标准 PyTorch nn.MultiheadAttention / scaled_dot_product_attention，无自定义 kernel。
  - 硬件架构层：论文未明确说明具体 GPU 型号（仅提及 models up to 13B）。

  Baseline 的核心缺陷：
  (1) **浅层视觉计算完全冗余**：现有方法假设浅层 cross-attention 负责跨模态融合（基于高 attention score 的观察），但实验表明浅层 cross-attention scores 与输入指令无关（静态 attention pattern），mask 高 attention token 对性能无影响，视觉 token 的作用仅为 attention sink 而非信息传递——计算完全浪费。
  (2) **attention-based token 选择不可靠**：FastV/SparseVLM 等基于 attention weights 选择关键 token，但由于 visual attention sink 现象、attention 分布的分散性（难以隔离单 token 影响）、以及固定阈值缺乏任务适应性，导致选中的 token 并非真正关键。
  (3) **深层仍保留视觉 token 产生噪声**：深层模型行为已切换为纯语言 refinement，继续处理视觉 token 不仅无益，反而引入噪声（论文实验：skip layer 26 的视觉处理反而比继续处理性能更好）。
  (4) **缺乏对跨模态交互阶段性的理解**：现有方法未区分浅/中/深层的不同角色，一刀切地减少 token，导致在高压缩率（-97.6% attention）下性能崩溃（如 PDrop retained=64 时 MMB 降至 33.3）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **VisiPruner** 基于对 MLLM 跨模态交互三阶段规律的实验发现，提出分层剪枝策略：
  (1) **浅层 Attention Merging**（解决缺陷 1）：在 layer 1 将所有视觉 cross-attention 合并到单个随机 token 作为 attention sink（公式 A^{(1)}_{i,j} = Σ_{v∈V} A^{(1)}_{i,v} if j=k else 0）；layer 2+ 完全跳过 cross-attention 和 visual self-attention，仅保留 FFN 对视觉 token 的处理。
  (2) **中层 Influence-based Token Selection**（解决缺陷 2）：不依赖 attention scores，而是直接评估每个视觉 token mask 后对最后输入 token 的 attention output 的影响，使用 cosine similarity（方向变化）和 L2 distance（幅度变化）双指标——这捕获了 token 的实际信息贡献而非 attention 中的权重分布。以 cosine < 0.995 定义过滤层，L2 < 0.2 丢弃，平均 576 → 10.3 tokens。
  (3) **深层 Vision Exit**（解决缺陷 3）：持续追踪保留 token 的 influence，一旦连续两层无影响则在此层移除所有视觉 token，后续层纯文本处理。
  (4) **三阶段统一框架**（解决缺陷 4）：训练无关、即插即用，不修改模型权重，在 -98.3% visual attention 压缩下 MMB 仅从 64.3→62.0（相比之下 PDrop retained=64 的 -97.6% 压缩下 MMB 降至 33.3）。

  全栈执行例子（VisiPruner + LLaVA-v1.5 7B 推理）：
  - 模型推理算法层：LLaVA-v1.5 结构不变，在前向过程中插入三个剪枝阶段。**浅层（layer 1-8）**：layer 1 cross-attention 合并到随机令牌（N_text×576 → N_text×1 attention），layer 2+ cross-attention 矩阵全零（非可视），visual self-attention mask 全零，仅 FFN 处理 H_v。此时每个 visual token 仅经历 W_up → SiLU → W_down 的 FFN 变换，无 attention 交互。**中层（layer 9+）**：在每个 layer 末尾计算最后一个 text token 对每个 visual token 的 cross-attention output 影响（逐个 mask → 重算 O'_masked → cosine+L2），第一次 cosine<0.995 时确定为过滤层。过滤层保留 L2≥0.2 的 visual tokens（平均 10.3 个），其余丢弃。后续层仅对这 10.3 个关键 tokens 做 attention。**深层（平均 layer 23.9 起）**：检测到连续两层无 influence → ℓ_exit，此后 H_v 为空，仅剩 H_t 做纯语言 causal self-attention + FFN 直到生成完成。
  - 系统框架层：在 LLaVA 推理代码中插入 `cli_pruning.py`（位于 GitHub repo 的 `llava/` 目录），对 HuggingFace LLaMA 2 模型的每层 forward 进行 hook 修改：① 修改 attention mask（cross-attention 合并/mask），② 注入 token influence 计算逻辑，③ 动态丢弃 token 并调整 KV cache。配置参数：`shallow_mid_layer`（浅层/中层分界）、`layer_threshold`（cosine threshold）、`tokens_threshold`（L2 threshold）。
  - 编译框架层：论文未明确说明。
  - kernel 调度层：使用标准 PyTorch attention 算子，通过减少参与计算的 token 数量实现加速（减少 softmax 计算和 matmul 中的 N_v 维度），无自定义 kernel。
  - 硬件架构层：论文未明确说明。

  设计思路核心：**用 influence（对 attention output 的实际改变）替代 attention weights（softmax 概率分布）作为 token 重要性度量**。这是因为 attention weights 受 attention sink 效应污染（高 attention 不代表高信息贡献），且 softmax 归一化使得权重分布分散，难以精准定位关键 token。Influence-based 方法直接在值空间（V × attention output）操作，捕获了 token 对残差流状态的实际扰动，从而实现了比 attention-based 方法（FastV/SparseVLM/PyramidDrop）高得多的压缩精度。

## WorldMM: Dynamic Multimodal Memory Agent for Long Video Reasoning

- baseline方法是什么？
  Baseline是**基于文本摘要的固定尺度记忆检索方法**，代表工作包括EgoRAG、HippoMM、M3-Agent等。这些方法的通用设计是：
  (1) 将长视频按固定时间粒度（如30s）分段，每段用Video LLM生成caption/text摘要；
  (2) 构建单一文本形式的外部记忆（层级摘要、知识图谱或实体关系图）；
  (3) 检索时按固定策略返回预定数量的文本片段（如3个30s片段），用这些文本片段作为LLM推理上下文。
  
  **全栈执行例子（以M3-Agent/EgoRAG为代表性baseline）**：
  - **算法pipeline层**: 视频V（周级→44.3h）按固定30s粒度划分为~5,316段，每段采样0.5fps→768帧上限，VideoLLM输出caption文本。对caption按层级聚合生成event摘要（EgoRAG三层层级：moment→event→activity），构建纯文本知识图谱（如HippoRAG PPR图）。用户query q到来时，用固定k=3检索文本片段→LLM基于3个30s片段文本生成答案。缺点：(a) **纯文本丢失视觉细节**——同一段caption"[I stand and walk to dining table, discuss AC temperature with Shure]"无法传达场景中正在"吃火锅(hot pot)"的视觉证据；(b) **固定时间粒度无法适配不同问题**——"Where did I leave my glasses?"只需几秒，"What happened in the soccer match second half?"需要数十分钟——固定30s粒度要么信息不足要么冗余噪声；(c) **固定检索策略**对所有问题统一返回固定数量片段，无法按需动态扩展。
  - **系统框架层**: 论文未明确说明具体Serving框架。baseline方法通常将记忆构建作为离线预处理（caption生成→图构建），在线推理时用LLM API（如GPT-5/Gemini）进行检索+RAG生成。预处理开销大（如M3-Agent需要实体识别），但检索时延较低。
  - **编译框架层**: 论文未明确说明。
  - **kernel调度层**: 论文未明确说明。检索基于图算法（PPR等）和向量相似度计算，使用标准CPU/GPU运算。
  - **硬件架构层**: 论文未明确说明。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  WorldMM通过三大设计解决上述三个缺陷：
  
  **(A) 多模态记忆替代纯文本记忆** — 解决缺陷(a):
  WorldMM显式构建三类互补记忆：(i) Episodic Memory（文本KG，事实事件），(ii) Semantic Memory（文本KG，长期关系/习惯），(iii) Visual Memory（特征向量 + 原始帧索引）。与baseline纯文本相比，Visual Memory保留了baseline caption丢失的视觉细节（如场景外观、物体状态、空间关系）。当问题需要视觉推理（如"eating hot pot"需要识别桌上火锅）时，Retrieval Agent能自主从Visual Memory获取帧级视觉证据，而非仅依赖可能缺失视觉信息的文本caption。
  
  **(B) 多时间尺度记忆替代固定粒度** — 解决缺陷(b):
  Episodic Memory构建多尺度知识图谱 M_e = {G_{30s}, G_{3min}, G_{10min}, G_{1h}}（以EgoLifeQA为例）。检索时采用coarse-to-fine策略：先从各尺度用PPR分别检索top-k候选，再由LLM cross-scale reranker联合评估所有尺度的候选，选择最相关的时间范围并输出top-m结果。相比baseline只能返回30s固定片段，WorldMM能动态组合小时级摘要（如"下午的会议讨论了什么"）和秒级细节（如"19:30我把眼镜放在哪儿"）。
  
  **(C) 自适应多轮检索替代固定策略** — 解决缺陷(c):
  Retrieval Agent以LLM驱动多轮迭代检索。每轮Agent决定：(i) 用哪个记忆类型（episodic/semantic/visual），(ii) 用什么搜索关键词，(iii) 是否已收集足够信息（输出STOP）。通过迭代式策略，WorldMM能根据query复杂度自适应扩展检索范围：简单问题1-2轮即够（STOP早），复杂问题多轮深化（最多5轮）。EgoLifeQA上5轮vs1轮提升9.3%。特别地，Retrieval Agent可以跨记忆类型混合检索——先用episodic memory定位时间戳("DAY2 18:34:01")，再用visual memory按时间戳获取对应帧——实现文本+视觉的跨模态推理。

  **全栈执行例子（WorldMM）**：
  - **算法pipeline层**: 视频V（44.3h，EgoLifeQA）→ 离线构建：(i) M_e = {G_{30s}, G_{3min}, G_{10min}, G_{1h}} [四尺度的(entity, action, entity)三元组KG]，(ii) M_s [Consolidation增量更新的语义关系KG]，(iii) M_v = M_v^f ∪ M_v^I [VLM2Vec-V2编码特征 + 时间戳帧索引]。在线推理：query q="What were we doing last time we discussed the air conditioning temperature?" → Round1: Search/Memory:Episodic/Query:"discussing AC temperature"，检索到[DAY2 13:36-13:39]文本证明讨论了AC但未说明活动 → Round2: Search/Memory:Episodic/Query:"air conditioning"（更泛化），检索到[DAY2 18:34:01-18:34:29]描述"Shure set AC to 26°, eating..."，文本指向食物但未明确具体种类 → Round3: Search/Memory:Visual/Query:"DAY2 18:34:01-18:34:29"，获取对应帧图像（画面中桌上火锅+投影屏幕）→ Round4: Decision:Answer，基于文本+视觉综合证据选(A) Eating hot pot。整个过程是文本→文本+视觉的跨模态证据链构建。
  - **系统框架层**: 论文未明确说明具体Serving框架。WorldMM的预处理与baseline类似（caption→triplet extraction→graph construction→semantic consolidation），但论文指出支持在线操作——记忆每10s固定间隔更新，每段预处理可在间隔窗口内完成，consolidation增量合并无需重建。LLM推理使用GPT-5 API（闭源）或Qwen3-VL-8B本地部署。
  - **编译框架层**: 论文未明确说明。
  - **kernel调度层**: 论文未明确说明。PPR图检索、cosine向量检索等使用标准库实现，无自定义kernel。
  - **硬件架构层**: 论文未明确说明。

  设计思路核心：**用多模态+多尺度+自适应迭代三元设计替代单一的文本固定检索范式**。关键在于将"记忆构建(what to remember)"和"记忆检索(how to retrieve)"解耦并对每个维度进行专门化设计——记忆维度上分离episodic/semantic/visual三类互补表示，时间维度上构建多粒度KG层级，检索维度上用LLM agent实现跨类型/跨尺度的自适应多轮调度。这使得模型能根据query特性按需组合不同记忆类型和时间粒度，避免了baseline"一刀切"策略带来的信息缺失或噪声干扰。

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

## mPLUG-Owl3__Towards_Long_Image-Sequence_Understanding_in_Multi-Modal_Large_Language_Models

- baseline方法是什么？
  Baseline 多样化，核心问题在两类：
  **(1) Concatenate-based MLLMs（如 LLaVA-1.5/LLaVA-Next/LLaVA-Interleave/InternVL/Mantis）**：视觉编码器提取特征后，直接拼接到文本 token 序列中送入 LLM。输入序列长度 = 文本 tokens + 图像 patches × 图像数。对于多图/长视频场景，视觉 token 数量迅速膨胀，超出 LLM context window，导致 O(N²) 的 self-attention 开销剧增，推理延迟和显存消耗随图像数线性甚至平方增长。例如 LLaVA-Interleave 在 80GB VRAM 下仅能处理 ~20 张图。

  **(2) Flamingo-style Cross-Attention MLLMs（如 Flamingo/IDEFICS/EVLM）**：在每个 transformer block 中额外插入 cross-attention 层。虽不占用 LLM context window，但三个缺陷：(a) 引入大量新参数（每层增加完整 cross-attention 模块），训练和推理开销大；(b) LLM 预训练知识无法直接惠及跨模态融合，因为 cross-attention KV 完全独立；(c) cross-attention 不考虑图像在交织序列中的原始位置，导致多图场景性能差。

  **全栈执行例子（Concatenate-based Baseline，以 LLaVA-Interleave 为例）**：
  - **算法 Pipeline**：输入 `S = [T1, <image>, T2, <image>, T3]`，对每张图由 ViT 编码为 576 个 patch tokens，经 MLP 投影后插入 `<image>` 占位符位置，全序列送入 Qwen2，标准 causal self-attention 在所有 token 对上计算。
  - **系统框架层**：基于 transformers 库推理，使用 HuggingFace 原生 generate()。图像 tokens 占据大量 KV cache，每张 384×384 分辨率图 ≈ 576 tokens。100 张图 ≈ 57.6K visual tokens + text tokens，self-attention O(L²) 使 80GB GPU 在 ~20 张图时 OOM。
  - **编译框架层**：论文未明确说明。
  - **kernel 调度层**：使用标准 FlashAttention-2 kernel 计算 causal self-attention。视觉 tokens 与文本 tokens 在 attention 计算中无区分。
  - **硬件架构层**：训练 TP=4×V100-32G，推理单张 V100-32G。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **Hyper Attention Transformer Block (HATB)**，核心思想是将 cross-attention 与 self-attention **并行执行**，并用四个关键设计解决 Baseline 缺陷：

  **(a) 并行而非串行**：HATB 在同一个 transformer block 内，self-attention 处理文本内部关系，cross-attention 从视觉特征中提取文本当前语义所需要的视觉信息。两者共享 Query，因此 cross-attention 无需单独计算 Query，大幅减少参数。

  **(b) 稀疏集成**：仅替换 LLM 中少量层（Qwen2 的 28 层中选 4 层 [0, 9, 17, 25]），其余层保持纯文本 self-attention。实验证明 4 层 > 8 层（稠密反而导致 zero-shot 退化），2 层也足够但略弱。这直接解决 Flamingo 参数量过大的问题。

  **(c) Modality-Specific KV Projection + Shared LayerNorm**：视觉的 K/V 投影权重用 LLM 预训练 KV 权重初始化（W_img_KV ∈ R^{2D×D}），使跨模态融合受益于语言模型的预训练知识。LayerNorm 也复用了 LLM 原生的 LN，保证视觉输入分布的兼容性。对比 Flamingo 的独立参数方案，参数量从 O(N_layers × D²) 降至 O(N_HATB × 2D²)。

  **(d) MI-Rope (Multimodal-Interleaved Rotary Position Embedding)**：为每张图的所有 patch 赋予其文本占位符 T_img 的 RoPE 位置编码。跨图共享位置索引确保图像间的顺序信息被保留，因果 cross-attention mask 确保自回归特性。这直接解决了 Flamingo 无位置编码导致多图场景差的缺陷。

  **(e) Adaptive Gating**：基于文本语义的门控 `g = Sigmoid(W_gate^T · H_text)`，动态决定每 token 从 self-attention（文本内在）和 cross-attention（视觉补充）各取多少信息。相比 Flamingo 的固定 learnable scale，自适应门控在单图和多图场景均有提升。

  **全栈执行例子（mPLUG-Owl3 Hyper Attention）**：
  - **算法 Pipeline**：输入 `S = [T1, T_img, T2, T_img, T3]` → WordEmbedding → H_text。视觉编码器提取特征 → Linear Projection → H_img。在 4 个 HATB 层中，H_text 同时做 self-attention 和 cross-attention(H_img)，通过 Adaptive Gate 融合 → FFN → 下一层。其余 24 层纯文本 self-attention。H_img 不进入 LLM context window，视觉 token 数量增长不会增加 self-attention 序列长度。
  - **系统框架层**：基于开源训练框架，Stage 1 仅训练 Linear Projection + Visual KV Projection + Adaptive Gate；Stage 2/3 全模型训练。TP=4 (tensor parallelism) 降低单 GPU 显存至 32-40GB。推理时视觉特征仅在 4 个 HATB 层参与 cross-attention，LLM self-attention 序列长度恒定（仅文本 tokens）。
  - **编译框架层**：论文未明确说明。
  - **kernel 调度层**：Hyper Attention 的 cross-attention 使用标准 attention 操作（matmul Q·K^T → softmax → ×V），可与 FlashAttention 兼容。视觉 feature 序列长度远小于 context window（每图 576 tokens 已由 ViT 编码完成，H_img 维度固定），cross-attention 复杂度 O(L_text × L_img) 远小于 O((L_text + L_img large)²)。
  - **硬件架构层**：训练和推理均在 V100-32G GPU 上进行。mPLUG-Owl3 在 V100-32G 上可输入 128 frames 视频（LLaVA-Interleave 仅 ~8 frames），展现了显著的显存效率优势。Distractor Resistance 测试中可处理 400 张图，而 LLaVA-Interleave 最多 50 张即 OOM。

## VTPerception-R1: Enhancing Multimodal Reasoning via Explicit Visual and Textual Perceptual Grounding

- baseline方法是什么？
  Baseline 方法为 **RLVR-only 的多模态推理方法**（如 GRPO/DAPO 目标 + 格式奖励 + 答案正确性奖励），代表方法包括 MM-Eureka、Vision-R1、R1-VL、Visionary-R1 等。这类方法的核心思路是将文本大语言模型的 RLVR 直接迁移到多模态场景：通过强化学习优化答案正确性和结构化推理格式，让模型在 GRPO/DAPO 目标下自动探索更好的推理路径。
  
  全栈执行例子：
  - **算法 Pipeline**：输入（图像 x_img + 问题 q）→ MLLM 编码 → 自回归生成推理链（可能包含隐式视觉参考）→ 最终答案 a。RL 只奖励答案匹配度 R_acc 和格式合规 R_fmt，感知过程完全隐式（模型自行决定关注图像的哪些部分）。
  - **系统框架层**：基于 EasyR1 或类似 DAPO/GRPO 实现框架，Ray 分布式 RL 训练，DeepSpeed ZeRO-3 部署。训练流程：前向生成 → 规则奖励计算 → 策略梯度更新。
  - **编译框架层**：论文未明确说明。
  - **kernel 调度层**：标准 Transformer attention（Causal Self-Attention + Cross-Attention for visual tokens），Qwen2.5-VL 原生实现，无特殊 kernel 优化。
  - **硬件架构层**：论文未明确说明 GPU 型号，使用 bf16 精度训练。

  Baseline 的核心缺陷（论文系统研究发现）：
  (1) **感知错误是 RLVR 失败的主因**：PAPO 人工审计发现 67% 的 GRPO 错误源于感知问题。
  (2) **正确性奖励不足以改善感知**：纯粹的 answer-correctness RLVR 无法有效提升模型的视觉/文本感知能力。
  (3) **小模型感知能力更弱，结构化 prompting 反而有害**：实验发现 7B 模型在 structured visual grounding prompting 下性能下降，因为其自身感知能力不足以支撑结构化描述，产生幻觉性观察。
  (4) **缺乏文本感知**：现有方法几乎仅关注视觉感知，忽略了文本线索（OCR、数值、约束条件）对推理的关键影响。
  (5) **推理与感知耦合导致不可审计**：隐式感知使得无法检查模型是否"看到了正确的证据"。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **VTPerception-R1**，一个显式解耦感知与推理的两阶段训练框架，系统性地解决上述缺陷。

  **对应解决关系**：
  
  (a) **针对"感知错误是主因"→ 引入感知增强 SFT（Stage I）**：通过 `<description>` 字段显式训练模型提取视觉/文本证据，将感知从隐式过程变为显式、可检查的输出。SFT 数据经过自动化清洗（VLM dense caption + Grounding DINO + EasyOCR → 重建 CoT → 质量评分），确保感知训练数据的质量。

  (b) **针对"正确性奖励不足以改善感知"→ 引入感知感知 RL（Stage II）**：在 DAPO 目标上增加三个感知专用奖励——视觉关键信息 R_vkey（衡量 description 覆盖标注关键视觉元素的比例）、文本关键信息 R_tkey（衡量 think 覆盖关键文本线索的比例）、一致性 R_cons（确保推理引用的实体/属性/数值被感知证据支持）。这些奖励直接提供感知级别的学习信号，而非仅依赖下游答案正确性的间接信号。

  (c) **针对"小模型感知弱"→ Perception-First 加权调度**：训练早期增大 R_vkey 和 R_tkey 权重，先建立稳健的感知基础，后期才切换到以 R_acc 为主。这种渐进式策略尤其适合 7B 等感知能力较弱的模型。

  (d) **针对"缺乏文本感知"→ R_tkey 奖励**：专门衡量模型是否在推理中使用了问题中的关键文本信息（OCR 文本、数值、单位、约束、常识），确保推理不是纯视觉驱动的。

  (e) **针对"推理与感知耦合不可审计"→ R_cons 奖励**：检查 `<think> + <answer>` 中引用的实体/属性/数值是否在 `<description> + question` 中有据可查，存在冲突时直接给 0 分奖励。这使得模型的推理链可审计——任何人可以检查 reasoning 是否 grounded in perception。

  **全栈执行例子（VTPerception-R1）**：
  - **算法 Pipeline**：
    1. 输入（图像 x_img + 问题 q）
    2. MLLM 编码（Qwen2.5-VL-7B-Instruct 全参数）
    3. Stage I SFT 训练：模型学习生成 `<description>`（提取视觉/文本证据）→ `<think>`（基于证据推理）→ `<answer>`（输出答案），损失 L_SFT = -Σ log π_θ(y_t|x, y_<t)
    4. Stage II RL 训练：对 prompt x 采样 G 个 response {o_i}，计算 R = R_acc + R_fmt + R_vkey + R_tkey + R_rep + R_cons，用 DAPO token-level clipped policy gradient 更新
    5. 推理时：模型生成 `<description> d <think> t <answer> a`，d 可被外部检查验证感知是否正确
  - **系统框架层**：基于 EasyR1-perc（DAPO 实现），Ray 分布式（1 主节点 + 1 ORM 节点），DeepSpeed ZeRO-3 + bf16，TP=4。RL 数据通过教师模型集成（72B 级模型）→ 预算验证（top-B by log-probability → correctness + coherence scoring）→ 关键信息提取流水线构建。
  - **编译框架层**：论文未明确说明。
  - **kernel 调度层**：标准 Qwen2.5-VL attention 实现，无特殊 kernel 优化。SFT 阶段有梯度检查点（gradient checkpointing）优化显存。
  - **硬件架构层**：论文未明确说明 GPU 型号，仅说明使用 DeepSpeed ZeRO-3 分布式训练。

  关键对比：Baseline 方法（如 Vision-R1）仅奖励 "最终答案对不对"（R_acc + R_fmt），VTPerception-R1 还同时奖励 "感知到了什么"（R_vkey）"有没有用文本线索"（R_tkey）和 "推理是否忠于感知"（R_cons），从而让模型在强化学习中主动优化感知质量。
