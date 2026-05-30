## DeepSeek-VL2: Mixture-of-Experts Vision-Language Models for Advanced Multimodal Understanding

- 属于算法pipeline的实现是什么？实验比较什么？
  实现包括三部分核心创新：**(1) Dynamic Tiling Vision Encoding**：将高分辨率图像按候选分辨率集 C={(m·384, n·384) | 1≤m,n≤9} 动态切分为 m×n 个 384×384 local tiles + 1 个 global thumbnail tile，通过 SigLIP-SO400M-384 共享视觉编码器处理所有 tile，经 2×2 pixel shuffle 压缩（27×27→14×14=196 tokens/tile），再通过 special tokens (<tile_newline>, <view_separator>) 组织 visual sequence 送入 LLM。**(2) DeepSeekMoE LLM with Multi-head Latent Attention (MLA)**：MLA 将 KV cache 压缩为低秩 latent vector，大幅减少推理时 KV cache 内存占用，提升吞吐；MoE 使用 shared experts + routed experts 架构，Tiny/Small 使用 Softmax routing (64 experts, Top-6)，DeepSeek-VL2 使用 Sigmoid routing + expert correction bias (72 experts, Top-6)，实现稀疏激活的高效推理。**(3) 三阶段训练 + 精细化数据管线**：Stage 1 VL Alignment（冻结 LLM，训练 vision encoder + MLP，ShareGPT4V 1.2M），Stage 2 VL Pretraining（全参数训练，~800B tokens，70% VL + 30% text-only），Stage 3 SFT（全参数 fine-tuning，~20B tokens）。实验比较：(a) DeepSeek-VL2-Tiny/Small/DeepSeek-VL2 vs 同参数/同激活参数量的开源密集和 MoE 模型（LLaVA-OV, InternVL2, Qwen2-VL, Molmo, MM1.5, Aria-MoE, Phi-3.5, Pixtral）；(b) vs 闭源模型 GPT-4V/GPT-4o/Claude 3.5 Sonnet/Gemini-1.5-Pro。

- 硬件平台是什么，配置是什么。
  训练：NVIDIA A100 GPU 集群。DeepSeek-VL2-Tiny: 16 节点 × 8 A100（128 GPUs），训练 7 天。DeepSeek-VL2-Small: 33 节点 × 8 A100（264 GPUs），训练 10 天。DeepSeek-VL2: 42 节点 × 8 A100（336 GPUs），训练 14 天。训练使用 HAI-LLM 框架，pipeline parallelism + tensor parallelism + expert parallelism。推理部署：Tiny 模型可部署在 10GB 单 GPU，Small 模型 40GB 单 GPU，DeepSeek-VL2 模型 80GB 单 GPU。

- 模型是什么。数据集和bench分别是什么。
  模型：LLaVA-style 架构，Vision Encoder (SigLIP-SO400M-384, ~0.4B) + VL Adaptor (2×2 pixel shuffle + 2-layer MLP) + DeepSeekMoE LLM。三个变体：**(1) Tiny**：LLM 3B total/0.57B activated, d=1280, 10 heads, 12 layers, MHA, 64 routed+2 shared experts, Top-6 Softmax。**(2) Small**：LLM 16B total/2.4B activated, d=2048, 16 heads, 27 layers, MLA(rank=512), 64 routed+2 shared, Top-6 Softmax。**(3) DeepSeek-VL2**：LLM 27B total/4.1B activated, d=2560, 32 heads, 30 layers, MLA(rank=512), 72 routed+2 shared, Top-6 Sigmoid + expert bias correction。数据集：**Alignment**: ShareGPT4V 1.2M；**Pretraining**: 交错图文(WIT, WikiHow, OBELICS, Wanjuan)、重新标注图像描述(内部 captioner+DeepSeek Chat 质量评分)、OCR(LaTeX OCR, RenderedText)、VQA、视觉定位(Objects365, KOSMOS-2)、Grounded conversation 等，~800B tokens；**SFT**: 通用 VQA, OCR/文档, 表格/图表, 推理/数学, 视觉定位, Grounded conversation, 纯文本, ~20B tokens。Benchmarks: DocVQA, ChartQA, InfoVQA, TextVQA, OCRBench, AI2D, MMMU(Val), MMStar, MathVista(TestMini), MME, MMBench, MMBench-V1.1, MMT-Bench, RealWorldQA, RefCOCO/RefCOCO+/RefCOCOg。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  MIT License 开源，代码和预训练模型发布于 https://github.com/deepseek-ai/DeepSeek-VL2。

  **Dynamic Tiling 算法流程（单张高分辨率图像推理）**：
  ```
  Input: image I of size (H, W)
  Params: base_res=384, max_grid=9

  // Step 1: Select best resolution minimizing padding
  C = {(m*384, n*384) | 1<=m,n<=9}
  For (h_c,w_c) in C:
      scale = min(h_c/H, w_c/W)
      pad = h_c*w_c - (H*scale)*(W*scale)
  Select (m*,n*) = argmin pad

  // Step 2: Dynamic Tiling
  Resize I to (m*·384, n*·384), pad to maintain aspect ratio
  Split -> m*×n* local tiles (384×384) + 1 global thumbnail (384×384)

  // Step 3: Vision Encoding
  For each tile:
      v = SigLIP-SO400M-384(tile)  // 27×27×1152
      v = PixelShuffle(v)          // 2×2 -> 14×14=196 tokens, dim=4608

  // Step 4: Visual sequence construction
  Global: 14×(14+<tile_newline>) = 210 tokens
  Local grid: (m*·14)×(n*·14) + n*·14 <tile_newline>
  Full: [210 global] + <view_separator> + [local grid]
  Total visual tokens: 210+1+n*·14·(m*·14+1)

  // Step 5: VL Adaptor projection
  v_proj = 2-layer MLP(v_token)  // 4608->d_LLM

  // Step 6: DeepSeekMoE LLM with MLA (for Small/VL2)
  For each layer l:
      // MLA: KV compression into latent
      c_KV = W_DKV · h_t              // -> rank 512
      k_C = W_UK · c_KV               // compressed key
      v_C = W_UV · c_KV               // compressed value
      k_R = RoPE(W_KR · h_t)          // decoupled RoPE
      q = W_Q · h_t,  q_R = RoPE(q)
      o = MHA([q;q_R], [k_C;k_R], v_C)

      // MoE FFN: sparse activation
      s = Sigmoid(W_gate·h_t)         // or Softmax for Tiny/Small
      s += bias (DeepSeek-VL2 only)
      TopK = TopK(s, K=6)
      FFN_out = Σ g_i · FFN_i(h_t)    // 2 shared + 6 routed experts
  ```

  **三阶段训练**：
  - Stage 1 (VL Alignment): frozen LLM, train vision encoder+MLP, ~2B tokens, batch=256, seq=4096, Cosine LR=~4.5e-4
  - Stage 2 (VL Pretraining): unfreeze all, ~800B tokens, batch=2304~3360, seq=4096, Step LR (÷√10 at 50%/75%), pipeline+ tensor+ expert parallelism
  - Stage 3 (SFT): unfreeze all, ~20B tokens, batch=64, seq=4096, Constant LR=1.4e-5~3e-5
  - 所有阶段：AdamW (β1=0.9, β2=0.95), weight_decay=0.1, grad_clip=1.0, aux_loss_weight=0.001~0.0001
