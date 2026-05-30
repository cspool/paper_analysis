## BEATs (Bidirectional Encoder representation from Audio Transformers)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
BEATs 是 Microsoft 提出的自监督音频预训练框架（Chen et al., ICML 2023 Oral），核心理念：通过迭代方式联合优化 acoustic tokenizer 和 audio SSL model。先用随机投影 tokenizer 生成离散标签（cold start），再通过知识蒸馏训练 self-distilled tokenizer，tokenizer 生成的离散标签用于下一轮 SSL 预训练，逐步抽象高层语义。架构：ViT-like 12层 Transformer encoder，convolutional relative position embedding，gated relative position bias，DeepNorm。预训练任务：Masked Audio Modeling (MAM)——随机 mask 75% 输入 patches，预测 masked positions 的离散标签。输入预处理：16kHz 重采样 → 128维 Mel filterbank (25ms window, 10ms hop) → 16×16 patches。~90M 参数，AudioSet-2M mAP 48.6%（单模型 SOTA）。TDC 论文中用作冻结的音频编码器，每 1 秒音频输出约 50 tokens。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# BEATs 在 TDC 中的使用 (frozen feature extractor)

# Preprocessing
audio_16k = resample(raw_audio, target_sr=16000)
mel_spec = mel_filterbank(audio_16k, n_mels=128, win=25ms, hop=10ms)
mel_norm = normalize(mel_spec, mean=0, std=0.5)
patches = split_into_patches(mel_norm, patch_size=16)  # 16x16 patches

# BEATs forward (frozen)
audio_tokens = BEATs_encoder(patches)          # (≈50, 768) per second

# In TDC pipeline: concat with visual tokens → Q-Former
F_ai = audio_tokens
F_xi = SigLIP(video_frame)                     # (144, D)
F_Q_i = QFormer(Q, [F_xi · F_ai], F_s)         # cross-modal compression
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
BEATs 开源: github.com/microsoft/unilm/tree/master/beats (MIT License)。加载：通过 HuggingFace/fairseq 加载预训练权重。下游任务达到 SOTA: ESC-50 98.1%, AudioSet-2M 48.6%, KS 98.1%。TDC 中作为冻结特征提取器（不训练），输出约 50 Hz temporal resolution 的 patch-level audio features。

涉及论文标题：
- Multimodal_Long_Video_Modeling_Based_on_Temporal_Dynamic_Context
