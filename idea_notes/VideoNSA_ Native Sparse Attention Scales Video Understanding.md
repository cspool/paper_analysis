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
