## Mental Imagery for VLM Reasoning（VLM 推理中的心智图像）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Mental Imagery 源于认知心理学（Shepard & Metzler 1971 的 mental rotation 实验，Kosslyn 1996 的 imagery debate），指人类在推理时不生成照片级精确画面，而是构建和操作简化的内部心智表征（mental sketches），仅捕获任务关键信息。Mirage 将这一认知理论引入 VLM 多模态推理：通过 latent visual tokens 在隐空间内构建类似 mental images 的紧凑视觉线索，替代显式图像生成。核心类比：(1) "压缩": 人类只记住碎片轮廓而非整个房间 → Mirage 用 k=4 个 average-pooled vectors 替代全部 n 个 patch features；(2) "灵活表征": 人类的心智草图是抽象而非照片级还原 → Stage 2 允许 latent tokens 从精确 visual match 中偏离，自适应任务需求。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Mental Imagery 的三个认知阶段在 Mirage pipeline 中的对等实现：
```
# 1. 编码 (Encoding): 感知输入 → 内部表征
#    人类: 看拼图碎片 → 提取边缘轮廓的简化特征
#    VLM: vision_encoder(image) → patch features → projection
#         → LLM hidden representations (multi-layer)

# 2. 操作 (Manipulation): 在心智中操纵表征
#    人类: 脑中旋转拼接碎片
#    VLM: self-attention 在 latent visual tokens e_{1:k} 与 text tokens
#         之间进行信息融合，latent tokens 作为 key-value 供后续查询

# 3. 提取 (Extraction): 从心智表征得出结论
#    人类: 判断匹配/不匹配
#    VLM: text_post tokens attend to e_{1:k} → LM head → answer
```

t-SNE 可视化 (Fig. 7) 验证了 mental imagery 的设计：Stage 2 后 latent tokens (red dots) 聚集在 visual cluster (yellow dots) 外侧而非内部——保持了 Stage 1 的 visual subspace 亲和性，同时体现了 Stage 2 的任务导向偏移。"simplified sketch, not photorealism."

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现通过两阶段训练：(1) Stage 1: helper image → compressed visual embeddings → cosine similarity loss 锚定 latent tokens 到 visual subspace；(2) Stage 2: 移除 cosine loss，仅文本 CE 监督，梯度反传使 latent tokens 在 visual subspace 附近自适应优化。使用场景：需要 "视觉想象" 而非 "视觉识别" 的多模态推理任务。对比纯文本 CoT：textualization 是对视觉信息的二次编码损失，mental imagery 保留了 first-order 视觉结构。

涉及论文标题：
- Machine_Mental_Imagery__Empower_Multimodal_Reasoning_with_Latent_Visual_Tokens
