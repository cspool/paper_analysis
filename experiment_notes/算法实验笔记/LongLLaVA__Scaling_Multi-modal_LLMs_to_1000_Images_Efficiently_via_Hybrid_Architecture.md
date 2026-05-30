## LongLLaVA__Scaling_Multi-modal_LLMs_to_1000_Images_Efficiently_via_Hybrid_Architecture

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：LongLLaVA —— 提出一种 Hybrid Mamba-Transformer 多模态大语言模型架构，通过三个层面扩展 MLLM 的长上下文多图理解能力：
  (1) **Hybrid LLM Architecture**：在 LLM backbone 中使用 4 组 hybrid layer stack，每组以 Attention:Mamba = 1:7 的比例交替集成 Transformer 层和 Mamba 层。同时采用 Mixture of Experts (MoE) 每隔一层集成，共 16 个 experts，每 token 激活 top-2。使用 RMSNorm 层归一化、Grouped Query Attention (GQA)、SwiGLU 激活函数。总参数 53B，推理时激活参数 13B (LongLLaVA-A13B)。LongLLaVA-9B 变体通过保留仅 Expert-0 构建，激活参数 9B。
  (2) **2D Bilinear Token Compression**：使用 CLIP (openai/clip-vit-base-patch32) 作为视觉编码器，两层 MLP 作为 projector。在 projector 前应用 2×2 bilinear pooling，将每张图像 token 数从 576 压缩到 144，同时保持 patch 间空间关系。
  (3) **Data Processing Protocol**：设计特殊 token 区分不同模态信息 —— `<img>` / `</img>` 包围图像 token 序列；`<vid>` / `</vid>` 包围视频帧序列；`<t>` 插入连续帧之间表示时间依赖；`\n` 用于高分辨率子图空间布局（分隔全局图与子图块、子图行）。
  (4) **Three-Stage Progressive Training**：Stage I (Single-image Alignment) —— 使用 600K image-caption pairs (ALLaVA-Caption + ShareGPT4V)，仅训练 projector，冻结 vision encoder 和 LLM；Stage II (Single-image Instruction Tuning) —— 使用 932K QA pairs (LLaVA-1.5 + Mantis-Single)，训练 projector + LLM，冻结 vision encoder；Stage III (Multi-image Instruction Tuning) —— 使用 200K Mantis + 200K VideoChat2 + 50K ShareGPT4Video + 200K Replay (single-image) + 50K Replay (pure-text) + 50K Sub-Image，全面多图能力训练。

  实验比较：
  (a) **Multi-image Evaluation** —— MileBench (Temporal/Semantic/IR)、Video-MME (128 frames w/o subs, Short/Medium/Long)、MVBench (20 video tasks)、LongVideoBench，对比 GPT-4V/GPT-4o/Gemini-1.5-Pro/Claude3-Opus (商业) 和 LongVA/InternVL2/InternVL2.5/OmChat/LongVILA/Qwen2-VL/Qwen2.5-VL/VideoLLaMA2/mPLUG-Owl3/Phi-3-Vision/Cobra/VideoChat2 (开源)。
  (b) **Diagnostic Long-Context Evaluation** —— VNBench (retrieval/ordering/counting atomic capabilities) 和 Video-NIAH (1200 images, needle-in-a-haystack)，对比 GPT-4o/GPT-4V/Qwen2-VL/VideoLLaMA2。
  (c) **Ablation Studies** —— MLLM Backbone 对比 (Vicuna-13B vs Jamba-9B)、Token Compression (no pooling / 1D pooling / 2D pooling)、Dataset Construction (single-image only / multi-image addition)、Training Strategy (mixed vs progressive)、Replay Data ablation。
  (d) **Scaling Analysis** —— Frames scaling (Video-MME 1→256 frames)、Shots scaling (VL-ICL few-shot vs fine-tuning)。
  (e) **Token Compression Impact** —— 5 general VL benchmarks + V* Bench (small object localization) 不同 token count 下的性能与推理开销，以及 Sub-Image Partitioning 缓解策略。
  (f) **Efficiency Analysis** —— 处理 100K tokens 的 Prefill time、Throughput、Memory usage、Max Throughput，对比 Falcon-mamba (Mamba only) 和 LLaVA-1.6 (Transformer only)。
  (g) **Applications** —— Healthcare (Pathology: VQA-RAD + PathVQA; 3D CT: CT-RATE) + Remote Sensing (FIT-RSFG-VQA + STAR dataset)。
  (h) **Single-image Evaluation** —— GQA/MME/MM-Vet/ScienceQA/SEED-Bench-v1/MMBench/MMMU/BLINK/ChartQA/DocVQA，对比商业和开源模型。

- 硬件平台是什么，配置是什么。
  训练：3 × 8 NVIDIA A800 GPU（共 24 卡）。数据序列随机采样拼接至 176K tokens，`<eos>` 分隔。单 epoch 训练，cosine learning rate schedule，warmup ratio 0.03，peak learning rate 1e-5。
  推理效率测试：单张 A100 80GB GPU (或 A800 80GB)。Efficiency metrics 使用 vLLM 框架 + Int8 quantization 评估 100K tokens 输入的 Prefill/Throughput/Memory。所有评估默认使用 Int8 quantization 降低计算开销，FP16 precision。

- 模型是什么。数据集和bench分别是什么。
  模型：Vision Encoder — CLIP ViT (openai/clip-vit-base-patch32)。Projector — 2-layer MLP。LLM Backbone — Hybrid Mamba-Transformer (Attention:Mamba=1:7, MoE 16 experts top-2)。LongLLaVA-9B (9B active params, Expert-0 only) 和 LongLLaVA-A13B (53B total, 13B active)。每张图像 token 数 144 (经 2×2 bilinear pooling)。
  训练数据 — Stage I: ALLaVA-Caption + ShareGPT4V (~600K captions)；Stage II: LLaVA-1.5 + Mantis-Single (~932K QA pairs)；Stage III: Mantis 200K + VideoChat2 200K + ShareGPT4Video 50K + Replay (single-image 200K + pure-text 50K) + Sub-Image 50K。Pure-text Instruction Tuning 使用 Evol-instruct-GPT4 + WildChat + SmolTalk + Tulu3 (DEITA) + LongAlign (~813K entries)。
  Benchmarks Multi-image: MileBench, Video-MME (30 sub-fields, 128 frames), MVBench (20 tasks), LongVideoBench (QA up to 1h), VNBench (synthetic video, retrieval/ordering/counting), VL-ICL (Matching Image task), Video-NIAH (1200 images)。
  Benchmarks Single-image: GQA, MME (perception), MM-Vet (6 VL capabilities), ScienceQA, SEED-Bench-v1 (image), MMBench (20 dimensions), MMMU (183 subfields, 30 image types), BLINK, ChartQA, DocVQA。
  Applications: Healthcare — VQA-RAD, PathVQA, CT-RATE (1304 samples, 512-1024 px, 100-984 slices)。Remote Sensing — FIT-RSFG-VQA, STAR dataset (1024×768 和 3327×4083 px)。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址：https://github.com/FreedomIntelligence/LongLLaVA

  算法 pipeline 伪代码：

  ```
  # ===== LongLLaVA 完整推理流程 =====

  # -- 视觉编码与压缩 --
  # 输入: N 张图像 (336×336) 或视频帧
  vision_encoder = CLIP_ViT()  # openai/clip-vit-base-patch32
  projector = 2-layer_MLP()

  for image_i in images:
      # CLIP 编码: 336×336 → 576 tokens (24×24 patch grid)
      H_v_raw = vision_encoder(image_i)  # [576, d_clip]
      # 2D 双线性池化: 2×2 pooling → 144 tokens (12×12 layout)
      # Reshape: [576, d] → [24, 24, d] → pool → [12, 12, d] → [144, d]
      H_v = bilinear_pool_2d(H_v_raw)      # [144, d_llm]
      H_v = projector(H_v)                 # [144, d_llm]

      # -- 数据格式包装 --
      if is_video:
          # 视频: <vid>\n<image><t><image>...\n</vid>
          token_seq += [<vid>, <image>, H_v_i, <t>, ..., </vid>]
      elif is_multi_image:
          token_seq += [<img>, H_v_i, </img>]
      else:  # single image
          token_seq += [<img>, H_v_i, </img>]

      if is_high_res:
          # 高分辨率: 全局图 + 子图分块 (pad to multiple of 168, 168×168 blocks)
          # 子图按 raster scan 排列, \n 分行
          sub_images = partition_and_pad(image_i, block_size=168)
          for row_blocks in sub_images:
              for block in row_blocks:
                  token_seq += [<img>, encode_and_pool(block), </img>]
              token_seq += [\n]

  # -- LLM Backbone (Hybrid Mamba-Transformer) --
  # 4 个 stack，每个 stack 内 Attention:Mamba = 1:7
  # 每隔一层有 MoE (16 experts, top-2 gating per token)
  for each stack in range(4):
      for layer in range(stack_size):
          if layer % 8 == 0:  # Attention layer
              # GQA: Q [1, d_head×n_query], K,V [seq, d_head×n_kv]
              H = RMSNorm(H)
              Q, K, V = W_Q(H), W_K(H), W_V(H)
              attn_out = FlashAttention(Q, K, V, causal=True, num_kv_heads=n_kv)
              H = H + attn_out
              if is_moe_layer:
                  # MoE FFN: 16 experts, top-2 routing
                  gate_logits = router(H)
                  top2_experts, top2_weights = topk(softmax(gate_logits), k=2)
                  H_moe = sum(w * expert_i(H) for i, w in zip(top2_experts, top2_weights))
                  H = H + H_moe
              else:
                  H = H + SwiGLU_FFN(H)
          else:  # Mamba layer
              H = RMSNorm(H)
              H = H + MambaBlock(H)  # SSM: Δ, A, B, C 参数扫描
              if is_moe_layer:
                  H = H + MoE_FFN(H)
              else:
                  H = H + SwiGLU_FFN(H)

  # 自回归生成
  response = autoregressive_decode(H, max_new_tokens)
  ```

  关键张量计算流程（LongLLaVA-A13B, single image 336×336）：
  - Vision Encoder: 336×336×3 → CLIP ViT-B/32 → [576, d_clip] → bilinear 2×2 pool: reshape to [24,24,d] → avg pool 2×2 → [12,12,d] → flatten → [144, d_clip] → 2-layer MLP projector → [144, d_llm]
  - Hybrid LLM: [144 + l_text, d_llm] 输入。每 8 层为 1 组 hybrid stack (1 Attention + 7 Mamba)。Attention layers 中 QKV 计算使用 GQA (n_kv < n_heads)。Mamba layers 中 SSM 输入 → 1D Conv + SiLU → Δ/B/C/A projection → selective scan → output gate。MoE layers 中 router 输出 16-dim logits → top-2 gating → expert FFNs 加权求和。
  - FlashAttention 兼容：Attention layers 支持标准 FlashAttention (causal mask)。Mamba layers 使用 Mamba SSM kernel (selective scan)。
  - Int8 Quantization: 评估时使用 Int8 量化降低计算开销，LLM backbone 权重 + 激活量化。

  关键设计：
  - 1:7 Attention:Mamba 比例：在 1.3B 模型上训练实验验证 1:3 与 1:7 性能差距极小但 1:7 计算效率显著更高
  - Expert 仅保留 Expert-0 (LongLLaVA-9B)：MMLU 和 BBH 上不同专家选择方法差异极小
  - 2D Pooling 优于 1D Pooling：12×12 layout 保持空间关系，GQA/Mile 优于 1D
  - Progressive Training 优于 Mixed Training：Multi-image 任务上有明显提升，Single-image 持平
  - Replay Data 关键：防止 single-image 和 text 能力退化。Text replay 50K 已饱和，Single-image replay 随数据量持续改善
