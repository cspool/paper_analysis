## Frame-Query Relevance Scoring via BLIP ITM for Keyframe Selection

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Frame-Query Relevance Scoring 是 FOCUS 及多数 training-free keyframe selection 方法（AKS, Q-Frame, Top-K）使用的核心信号——用预训练 vision-language encoder（默认 BLIP ITM）计算每帧与文本查询的语义相关性分数，作为该帧对回答查询的信息贡献的代理（proxy）。在 FOCUS 中，relevance r_t = cosine_similarity(BLIP.encode_image(x_t), BLIP.encode_text(q))，作为 bandit 的 reward 信号。FOCUS 理论框架建模为 r_t = y_t + ε_ψ，其中 y_t 是真值 frame-level utility（不可直接观测），ε_ψ 是 encoder 噪声（零均值，方差 σ_ψ²）——即 r_t 是 y_t 的无偏估计。BLIP (Li et al., ICML 2022) 是 Salesforce 提出的统一视觉-语言理解与生成框架，其 ITM (Image-Text Matching) 头通过 cross-attention 融合图文特征输出匹配概率，捕获细粒度对齐。BLIP-2 (Li et al., ICML 2023) 引入 Q-Former 连接冻结 ViT 和 LLM。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# BLIP ITM 用于 frame-query relevance scoring
# 输入: 帧 x_t (3×H×W), 查询文本 q

# Vision Encoder: ViT → image feature
e_img = BLIP.visual_encoder(x_t)    # shape: (d,), d≈768/1024

# Text Encoder: BERT → text feature
e_txt = BLIP.text_encoder(q)        # shape: (d,)

# ITM: cross-attention + binary classifier
# BLIP 内部: image features as K,V, text features as Q → [CLS] token → sigmoid
# 或简化版 (FOCUS 实际可能使用):
r_t = (e_img · e_txt) / (||e_img|| * ||e_txt||)  # cosine similarity ∈ [0,1]
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在 FOCUS 中，BLIP 参数完全冻结不做微调。FOCUS Table 10 encoder 消融：BLIP (63.5%) > SigLIP (60.9%) > CLIP (60.2%) > Uniform (58.9%)，所有 encoder 均优于 uniform baseline，框架对 encoder 选择鲁棒。每个 BLIP forward 约 10^8-10^9 FLOPs，全量评分 1h 视频需 10^11-10^12 FLOPs——即 255 GPU hours（Table 3: AKS w/o pre-filtering），所以需要 bandit 采样。BLIP ITM 的局限：(1) 对需要世界知识的复杂推理查询评分不准；(2) 无法区分同 object 不同 context 的情况。FOCUS 通过 bandit 采样仅评分 1.6% 帧，将开销降至 5.5 GPU hours。

涉及论文标题：
- FOCUS__Efficient_Keyframe_Selection_for_Long_Video_Understanding
