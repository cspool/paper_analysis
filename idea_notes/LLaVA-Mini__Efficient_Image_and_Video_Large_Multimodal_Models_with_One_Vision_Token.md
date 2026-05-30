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
