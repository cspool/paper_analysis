## Video Large Language Model / VLLM (视频大语言模型)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Video Large Language Model (VLLM) 是一类能够理解和推理视频内容的多模态大语言模型。其推理 pipeline 通常包含：(1) 视频输入 → 帧采样（如 32 frames @ uniform），(2) Vision Encoder（如 CLIP ViT-L）逐帧编码为 visual tokens（每帧 ~196 tokens），(3) Projector（如 MLP）将 visual tokens 映射到 LLM 的 token embedding 空间，(4) 将 visual tokens 与 text prompt tokens 拼接送入 LLM（如 LLaMA/Qwen）进行 prefilling + decoding 生成答案。代表性 VLLM 包括：LLaVA-OneVision（统一图像/多图/视频）、VideoLLaMA 2（时空建模 + 音频）、VideoChat（chat-centric video understanding）、LLaVA-NeXT-Interleave（多图/视频/3D 统一）、VILA（预训练 for visual language）、Tarsier（视频描述）。主要挑战：数十帧 × 196 tokens/frame = 6272+ visual tokens，导致 attention 计算复杂度 O(n²) 爆炸，prefilling 和 decoding 延迟高，GPU 内存占用大。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
VLLM 推理 pipeline（以 LLaVA-OV-7B, 32 frames 为例）：
```
# === VLLM 推理 Pipeline ===
# 模型: LLaVA-OneVision-7B (d=3584, m=18944, T=28)
# 视频输入: 32 frames

# Step 1: 帧采样与视觉编码
frames = uniform_sample(video, 32)          # 32 frames
for frame in frames:
    z_i = VisionEnc(frame)                  # CLIP ViT → 196 tokens/frame
# Z_v shape: (32, 196, d_v) → (6272, d_v)

# Step 2: Projector 映射
H_v' = Projector(Z_v)                       # MLP → token embedding space
# H_v' shape: (6272, D=3584)

# Step 3: concat 文本 tokens
H_q = TextTokenizer(prompt)                 # shape: (N_q, 3584)
H = concat[H_v', H_q]                       # shape: (6272+N_q, 3584)

# Step 4: Prefilling (所有 token 并行计算)
for l in 1..T (28 layers):
    Q = H W_Q^l, K = H W_K^l, V = H W_V^l  # Eq.1
    out = MHA(Q, K, V) + FFN(out)
    KV_cache[l] = (K, V)

# Step 5: Decoding (逐 token 自回归)
for t in 1..max_new_tokens:
    h_t = LM_Head(LLM(KV_cache))            # 仅计算当前 token 的 K/V
    KV_cache = concat[KV_cache, (h_t W_K, h_t W_V)]  # Eq.2

# 计算负载: prefilling FLOPs ≈ T(4nd² + 2n²d + 2ndm)
# n = 6272+N_q tokens → 约 41.4T FLOPs (LLaVA-OV-7B)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
VLLM 通常基于 image LLM 扩展：在预训练的 image MLLM 基础上添加视频数据集（如 VideoChatGPT、ActivityNet）进行 instruction tuning。主流框架：LLaVA-NeXT/OneVision（PyTorch）、VideoLLaMA（时空 Q-Former）、VILA（大规模预训练）。评估：LMMs-Eval 框架支持 VideoMME、MVBench、ActivityNet-QA、PerceptionTest、VideoDetailCaption 等 benchmark。部署：支持 Flash Attention 加速，支持 4-bit/8-bit 量化（LLaVA-1.5），训练需要 A100/H100 多卡，0.5B~72B 参数规模。token 压缩是当前 VLLM 推理加速的核心方向，包括 token merging（TTM, HoliTom, TTF）、token pruning（FastV, PruMerge）、dynamic pruning（DyCoke）等 training-free 方法。

涉及论文标题：
- DyCoke__Dynamic_Compression_of_Tokens_for_Fast_Video_Large_Language_Models
- T2Vid__Translating_Long_Text_into_Multi-Image_is_the_Catalyst_for_Video-LLMs

补充（来自 Sparrow 论文）：Sparrow 揭示了一个重要的 VLLM 训练数据效率问题——当基于 Image-LLM 通过 fine-tuning 开发 VLLM 时，简单地扩大视频数据量（如从 30K 到 200K 样本）带来的性能增益呈对数增长趋势（Video-MME 仅从 55.8 → 56.3），原因在于视频 instruction 数据的多样性不足。Sparrow 发现 ShareGemini 数据集仅使用 9 种固定模板变体（"Describe this video in detail"），而 Video-ChatGPT 的自 instruction 方式也缺乏真正的多样性。解决方案是引入文本域的长上下文数据，通过 text-to-image 合成（PIL/Pillow 将 text segments 渲染为 448×448 图像）模拟视频帧序列结构，以 2:1 比例混合真实视频数据和合成数据训练。Sparrow 用 30K 混合数据（15% 样本量）达到了与 200K 全量视频数据相当的 Video-MME 性能，GPU hours 从 276.8 降至 33.6（8.2× 效率提升），并意外地在长视频理解上获得了 +6.6 points 的提升（100K 规模，LongVideoBench/MLVU）。
