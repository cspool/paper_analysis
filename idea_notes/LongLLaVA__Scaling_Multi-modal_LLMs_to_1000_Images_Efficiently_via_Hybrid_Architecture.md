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
