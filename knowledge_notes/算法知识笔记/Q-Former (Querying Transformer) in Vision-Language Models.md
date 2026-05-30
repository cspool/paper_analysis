## Q-Former (Querying Transformer) in Vision-Language Models

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Q-Former (Querying Transformer) 是 BLIP-2 (Li et al., ICML 2023) 的核心创新组件，作为连接冻结的视觉编码器与冻结的 LLM 之间的轻量级信息瓶颈（information bottleneck）。架构由两个共享 self-attention 层的 Transformer 子模块组成：(1) Image Transformer —— 通过 cross-attention 与冻结的图像编码器输出交互，从视觉特征中提取与文本语义最相关的信息；(2) Text Transformer —— 同时作为文本编码器和文本解码器使用。核心机制：32 个可学习的 query 向量（每向量 768 维）通过 self-attention 相互联系，通过 cross-attention 与冻结图像特征交互，将全图的大量 visual tokens（如 ViT 的 257 tokens）压缩为仅 32 个最具信息量的 query 输出。两阶段预训练：(1) 视觉-语言表示学习（ITC+ITM+ITG 三个联合损失）；(2) 视觉到语言的生成学习（Q-Former 输出通过线性投影馈入冻结 LLM）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
标准 Q-Former 前向过程：
```
Q = learnable_query_vectors                   # (32, 768)
I_emb = frozen_ViT(image)                     # (257, 1024) ViT output
# Self-Attention on queries
Q = SelfAttention(Q, causal_mask=False)
# Cross-Attention: queries attend to image features
Q = CrossAttention(Q=Q, KV=I_emb)             # (32, 768)
# 输出: 32 个信息密集的视觉 features
output = Linear(Q)                            # 投影到 LLM embedding 维度
```

HERMES 对 Q-Former 的两个扩展：
(1) **Episodic Q-Former**：在标准 Q-Former 中插入 ECO 模块——self-attention on queries → cross-attention to visual episodes M → ECO_q（在 query 空间应用与 ECO 相同的 cosine-similarity iterative merging，将跨 window 的 queries 也聚合为 query episodes）。公式：$Q = ECO_q(CA(SA(Q_0), M))$。
(2) **Hierarchical Q-Former**：两级设计——Frame Q-Former (fQFormer) 独立增强每帧语义 → Frame-to-Sequence Adapter (Linear) → Video Q-Former (vQFormer) 全局聚合所有帧信息。公式：$Q_{sem} = vQFormer(Linear(fQFormer(F')))$。消融实验（Table 8）：HQFormer=95.2% > vQFormer=94.1% > fQFormer=93.2% on Breakfast。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Q-Former 在 HERMES 中的实现：(1) 权重初始化自 InstructBLIP（已包含视觉-语言对齐）；(2) 参数在 fully-supervised 设置下可微调；(3) 作为 visual features → LLM embedding 的桥梁，输出与 SeTR semantic features concat 后馈入 Vicuna-7B；(4) Q-Former 的 32 query 设计在视频场景下通过 Episodic Q-Former 扩展为 episode-level query 组织。Q-Former 的替代方案：LLaVA 使用简单线性层（而非 Q-Former）连接 ViT 和 LLM——更简单但缺乏 Q-Former 的信息瓶颈和跨模态对齐能力。开源：BLIP-2 (https://github.com/salesforce/LAVIS/tree/main/lavis/models/blip2_models)。

TDC 论文对 Q-Former 的扩展使用：用于视频帧的 temporal dynamic context 压缩。具体做法：(1) 对每个视频场景 segment，首帧完整保留作为 static frame；(2) 对首帧 visual tokens 做 AvgPool 得到 K=16 个 query tokens（而非 learnable queries）；(3) 后续每帧的 visual+audio tokens 拼接后与 query tokens 做 cross-attention，同时注入 instruction text F_s 使压缩自适应于用户问题；(4) Q-Former 的 query output 作为该帧的压缩表示，形成 temporal dynamic context F_TDC。消融实验表明 AvgPool queries 优于 learned queries，且 text instruction 可提升各 benchmark 性能（MVBench +0.4, MLVU +0.2/+1.6, VideoMME +1.2）。Q-Former 由预训练 BERT 初始化。

涉及论文标题：
- Bridging_Episodes_and_Semantics__A_Novel_Framework_for_Long-Form_Video_Understanding
- LLaVA-Mini__Efficient_Image_and_Video_Large_Multimodal_Models_with_One_Vision_Token
- Multimodal_Long_Video_Modeling_Based_on_Temporal_Dynamic_Context
