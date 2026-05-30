## Training-Free Token Compression (免训练Token压缩)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Training-Free Token Compression 指不需要对模型进行额外训练或微调、直接降低视觉 token 数量的推理加速技术。与需要训练的方法（如 xGen-MM-Vid 需要微调将多帧 token 映射到 compact 集合、VILA 需要大规模视频数据训练）不同，training-free 方法作为 plug-and-play 模块嵌入现有 VLLM 推理流程，保持模型参数完全冻结。核心优势：(1) 零训练成本——无需 GPU 小时训练；(2) 即插即用——可直接应用于不同 VLLM 架构和多规模模型；(3) 保持原始推理能力——不改变模型权重，理论上可无损恢复到 full-token 模式。主要方法包括：(a) Token Merging——基于相似度合并冗余 token（ToMe, TTM, HoliTom, TTF, TempMe）；(b) Token Pruning——基于 attention score 或 salience 剪枝不重要 token（FastV, PruMerge, PyramidDrop, VisionZip, SparseVLM）；(c) Dynamic Pruning——decoding 阶段动态调整剪枝集（DyCoke）；(d) Hierarchical Attention Pruning——基于 ViT 内部不同层 attention 语义差异的剪枝（HiPrune/HiPrune++）；(e) Test-Time Temporal Sampling (T3S)——在推理时生成 m 个短且多样化的子序列，打包到单次前向传播中处理，通过 logit 聚合输出最终预测，利用视频时间冗余同时降低 attention 复杂度（O(L²)→O(∑αᵢ²L²)）并扩展有效时间覆盖。DyCoke 在 training-free 方向上达到 SOTA：1.5× speedup + 1.4× memory reduction，性能不降反升。T3S 在 Qwen2.5-VL-7B 上实现 2.04× speedup 且准确率提升 3.1%。HiPrune 在 LLaVA-1.5 上以 1/3 token 保持 99.3% 准确率，FLOPs 减少 58.7%，并证明 training-free 方法可跨 VLM 架构（LLaVA、Qwen、Video-LLaVA）通用。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Training-free 方法的共同模式——在 VLLM 推理流程中插入压缩操作而不改变模型权重：
```
# === Training-Free Token Compression 通用模式 ===
# 前提: VLLM 模型权重完全冻结 (no gradient, no fine-tuning)

# 方式 A: Pre-LLM 压缩 (在视觉 token 进入 LLM 之前)
visual_tokens = VisionEncoder(video_frames)        # (M*N_v, d)
compressed_tokens = compress(visual_tokens)          # 训练无关压缩函数
H = concat[Projector(compressed_tokens), text_tokens] # 送入 LLM

# 方式 B: In-LLM 剪枝 (在 LLM 推理过程中剪枝)
H = concat[Projector(visual_tokens), text_tokens]
KV_cache = LLM_prefill(H)
for t in decoding:
    KV_cache = prune_KV_cache(KV_cache, attention_scores)  # 动态调整
    output_t = LLM_decode(KV_cache)

# 方式 C: 混合 (DyCoke = Pre-LLM TTM + In-LLM Dynamic Pruning)
visual_tokens = VisionEncoder(video_frames)
compressed = TTM(visual_tokens, K=0.7)              # Pre-LLM merging
H = concat[Projector(compressed), text_tokens]
KV_cache = LLM_prefill(H)
for t in decoding:
    KV_cache, DP_cache = dynamic_prune(KV_cache)    # In-LLM dynamic
    output_t = LLM_decode(KV_cache)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
DyCoke 通过 PyTorch 实现，在 lmms-eval 中通过模型参数传入 dycoke=True 启用。关键超参数仅 3 个：K（TTM 保留比）、L（评估层）、P（动态剪枝保留比）。评估使用统一 FLOPs 指标确保与其他 training-free 方法（FastV, PruMerge）公平对比。类似工具：FastV (github.com/pkunlp-icler/FastV)、PruMerge (github.com/42Shawn/LLaVA-PruMerge)、HoliTom (github.com/cokeshao/HoliTom)。所有 training-free 方法的核心 trade-off：token 保留率 vs 性能保持，通常保留 15-35% tokens 即可保持 >99% 性能。

涉及论文标题：
- DyCoke__Dynamic_Compression_of_Tokens_for_Fast_Video_Large_Language_Models
- HiPrune__Training-Free_Visual_Token_Pruning_via_Hierarchical_Attention_in_Vision-Language_Models
- SCOPE: Saliency-Coverage Oriented Token Pruning for Efficient Multimodal LLMs
- Test-Time_Temporal_Sampling_for_Efficient_MLLM_Video_Understanding
