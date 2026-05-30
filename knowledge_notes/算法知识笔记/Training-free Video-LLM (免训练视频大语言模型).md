## Training-free Video-LLM (免训练视频大语言模型)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Training-free Video-LLM 是一类将图像预训练的 VLM 直接扩展到视频理解的方法，其核心特点是**不做任何额外训练或微调**（与 Training-required Video-LLM 形成对比）。Training-required 方法通常需要在大规模视频数据集上微调视觉编码器、跨模态连接器或 LLM（如 Video-ChatGPT, Video-LLaVA, LLaVA-NeXT-Video 等），计算成本高。Training-free 方法利用图像和视频之间的结构相似性，通过设计推理时的压缩、采样、聚合策略来适配视频输入，同时保持预训练 VLM 的所有参数冻结。代表性方法包括：IG-VLM（构建 grid-view 图像）、FreeVA（帧级时间聚合）、SF-LLaVA（slow-fast 架构）、TS-LLaVA（thumbnail-and-sampling 策略）、D-CoDe（dynamic compression + question decomposition）。Training-free 方法的优势在于零额外训练成本、可插拔性（直接应用于不同的预训练 VLM），但通常需要仔细设计的压缩策略来平衡信息保留和 token 预算。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Training-free Video-LLM 的通用 pipeline：
```
# === Training-free Video-LLM 推理流程 ===
# 输入: 视频 V = {I_1, ..., I_T}, 问题 Q
# 所有模型参数冻结（不更新）

# Step 1: 帧选择/压缩（training-free 的关键设计空间）
frames = frame_selection_strategy(V, max_frames=N)
# 策略变体:
#   - Uniform: 均匀间隔采样
#   - IG-VLM: 构造 grid-view 图像
#   - SF-LLaVA: slow (密集采样) + fast (稀疏采样) 双路径
#   - TS-LLaVA: thumbnail (全局缩略图) + sampling (均匀采样)
#   - D-CoDe: uniform + supplementary (基于语义多样性)

# Step 2: 帧编码（冻结的视觉编码器）
for frame in frames:
    tokens_frame = Frozen_ViT(frame)    # CLIP/SigLIP 等

# Step 3: Token 压缩/聚合（可选, training-free）
tokens_compressed = token_compression_strategy(all_tokens)
# 策略变体:
#   - Average Pooling: 空间平均
#   - D-CoDe: salience pruning + similarity merging

# Step 4: LLM 推理（冻结）
answer = Frozen_LLM(concat([tokens_compressed, text_emb(Q)]))
```

D-CoDe 在 training-free 方法中的定位（Table 2, EgoSchema）：
- IG-VLM: 35.8%
- SF-LLaVA: 47.2%
- TS-LLaVA: 50.2%
- D-CoDe: 58.0%（第一个超越所有 training-required 方法的 training-free 方法）

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Training-free Video-LLM 的实现通常基于 HuggingFace Transformers，核心代码是帧采样器 + token 处理模块 + 冻结模型推理的 Python 脚本。D-CoDe 的实现代码在 `run_inference_multiple_choice_qa.py` 和 `run_inference_video_qa.py` 中，不涉及任何模型训练/微调代码。优势：(1) 可插拔——可直接应用于不同预训练 VLM；(2) 低成本——单卡 RTX A6000 即可运行；(3) 零数据需求——不需要视频训练数据。劣势：(1) 推理延迟通常高于 training-required 方法（Question Decomposition 引入 511% 额外延迟）；(2) 性能上限受限于基础 VLM 的能力；(3) 对频繁场景切换的视频适应性较差（D-CoDe 在 MSRVTT-QA 的频繁切换子集上从 64.2 降至 56.0%）。

涉及论文标题：
- D-CoDe__Scaling_Image-Pretrained_VLMs_to_Video_via_Dynamic_Compression_and_Question_Decomposition
