## DeepSeek-VL2: Mixture-of-Experts Vision-Language Models for Advanced Multimodal Understanding

- baseline方法是什么？
  Baseline 为 DeepSeek-VL（LLaVA-style 架构，hybrid vision encoder: SigLIP-384 + SAM-B-1024, dense LLM 7B, 固定双分辨率 384×384 和 1024×1024）。具体痛点：(1) **固定分辨率限制**：DeepSeek-VL 的 hybrid vision encoder 仅支持固定 1024×1024 和 384×384 两种分辨率，无法高效处理极端宽高比的高分辨率图像（如 InfographicVQA 中的超长图），导致细节信息丢失。(2) **Dense LLM 参数效率低**：DeepSeek-VL 使用 7B dense LLM，所有参数在每次推理时均激活，计算和显存效率低，模型规模扩展困难。(3) **视觉定位能力缺失**：DeepSeek-VL 不支持视觉定位（visual grounding），无法输出目标物体的 bounding box，限制了在 embodied AI 和 agent 场景中的应用。(4) **训练数据质量不足**：开源图像描述数据集质量参差不齐（短描述、文本不匹配、幻觉），影响模型的多模态理解能力。

  **Baseline 全栈执行例子（以 DeepSeek-VL 7B, 单张 1024×1024 图像 + 文本 query decode 为例）**：
  - **算法层**: Hybrid Encoder: SigLIP-384 (coarse) + SAM-B-1024 (fine) -> Concat -> MLP Projector -> Dense 7B LLM Decoder。固定双分辨率策略，不支持动态 tile。无 visual grounding 能力。Next-token prediction only (visual+text tokens)。
  - **系统框架层**: HAI-LLM 框架，标准数据并行+流水线并行训练。dense LLM 全参数激活，每 token 计算量固定。
  - **编译框架层**: 论文未明确说明。
  - **Kernel调度层**: 标准 Transformer attention kernel, FFN kernel on A100 Tensor Cores。
  - **硬件架构层**: NVIDIA A100 GPU 集群训练，7B dense 模型可部署在单 GPU。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **(1) Dynamic Tiling Strategy**：替代 DeepSeek-VL 的 hybrid fixed-resolution encoder。通过候选分辨率集 C={(m·384, n·384)|1≤m,n≤9} 最小化 padding 面积选择最佳分辨率，将高分辨率图像动态切分为 m×n 个 384×384 local tiles + 1 global thumbnail tile。所有 tile 通过单个 SigLIP-SO400M-384 共享编码，再经 2×2 pixel shuffle 压缩（27×27→14×14, 196 tokens/tile），通过 <tile_newline> 和 <view_separator> special tokens 组织 visual sequence。优势：(a) 支持任意宽高比高分辨率图像，不限固定 1024×1024；(b) tile 数可控（≤81+1），视觉 token 数随分辨率线性增长而非平方增长（local attention 特性）；(c) 统一使用单一 SigLIP 编码器，简化架构。

  **(2) DeepSeekMoE LLM with MLA**：将 dense LLM 替换为 MoE + MLA 架构。MLA 通过低秩压缩 K_t^{C}=W^{UK}·(W^{DKV}·h_t)，V_t^{C}=W^{UV}·(W^{DKV}·h_t) 大幅减少 KV cache（rank=512 vs embedding dim=2048/2560），提高推理吞吐。MoE 使用 2 shared + 64~72 routed experts，每 token 仅激活 Top-6 routed+2 shared=8 experts，3B→0.57B/16B→2.4B/27B→4.1B activated params。优势：(a) 稀疏激活大幅降低每 token 计算量；(b) MLA 压缩 KV cache 使长序列推理更高效；(c) 总参数大但激活参数少，训练推理效率高。

  **(3) 精细化三阶段训练 + 数据质量管控**：Stage 1 VL Alignment: 冻结 LLM 训练 vision encoder+MLP 实现 visual-textual 对齐。Stage 2 VL Pretraining: 全参数训练 ~800B tokens (70% VL + 30% text-only)，通过内部 captioner + DeepSeek Chat 质量评分过滤低质量描述。Stage 3 SFT: ~20B tokens 涵盖 VQA/OCR/文档/图表/数学/定位/grounded conversation 等多任务。数据增强包括：negative samples 防止幻觉定位、中英双语多轮对话消除语言混用、Visual Prompt QA 支持箭头/框/圈/涂鸦理解。

  **(4) Visual Grounding**：引入 <|ref|>, <|/ref|>, <|det|>, <|/det|>, <|grounding|> special tokens 实现：(a) 视觉定位——给定文字描述输出 bounding box；(b) Grounded conversation——在对话回复中引用具体目标位置；(c) In-context visual grounding——跨图像目标的参照理解。

  **论文方法全栈执行例子（以 DeepSeek-VL2, 4.5B activated, 单张高分辨率 3000×1000 图像 + text query decode 为例）**：
  - **算法层**: Dynamic Tiling 选择 m=8,n=2 分辨率 (3072×768, padding=174,336)，切 16 local tiles + 1 global thumbnail。SigLIP-SO400M-384 编码 17 tiles × 196 tokens = 3,332 visual tokens + special tokens → ~3,400 total visual tokens。MLA (rank=512) 压缩 KV: c_KV->k_C,v_C (latent dim 512, up-projected per head), decoupled RoPE (d_h^R=64)。DeepSeekMoE: 2 shared + 72 routed, Sigmoid gating+expert bias correction, Top-6 activation -> 8 experts per token。MLA+MoE: 每 token 仅 4.1B activated / 8 experts active，KV cache 大小为同等 dense MHA 的 512/2560≈20%。
  - **系统框架层**: HAI-LLM 框架，pipeline parallelism（fine-grained vision encoder layer division 防止 pipeline bubble）+ tensor parallelism + expert parallelism。Image tile load balancing across data parallel ranks。双 pipeline strategy 按数据是否为纯文本切换。
  - **编译框架层**: 论文未明确说明。
  - **Kernel调度层**: 标准 FlashAttention kernel on A100。MLA attention 需自定义 fused kernel（低秩压缩+上投影+RoPE 融合）。MoE all-to-all dispatch/combine 通信。
  - **硬件架构层**: NVIDIA A100 GPU 集群（336 GPUs for DeepSeek-VL2, 7×8 GPU nodes），节点内 NVLink，节点间 InfiniBand。FP32 optimizer states，无 BF16 optimizer（Tiny/Small）或 BF16 optimizer（DeepSeek-VL2）。推理部署在单 GPU (80GB)。
