## Instruction Diversity in Video-LLM Training（视频-LLM训练中的指令多样性）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Instruction Diversity（指令多样性）指 MLLM 训练数据中 instruction 文本的语义丰富程度，包括句式、任务类型、推理深度、领域覆盖等维度的变化。Sparrow 论文通过 t-SNE 可视化首次系统性地揭示了视频 instruction 数据的多样性不足问题：ShareGemini 数据集的 instruction 仅来自 9 种固定模板变体（如 "Describe this video in detail"），t-SNE 图中呈现 9 个清晰聚类；Video-ChatGPT 数据集的 instruction 虽然包含具体视频内容相关问题（视频摘要、内容问答、创造性任务），但由于 self-instruction 的本质——基于固定 prompting 模板由 LLM（GPT-3.5）生成——其多样性同样有限。这种不足导致的直接后果是数据效率低下：当视频样本量从 30K 扩大到 100K（3.3×），模型性能仅从 55.8 提升到 56.3（+0.5 points）呈对数增长。这一定量缺陷此前在视频-LLM 领域未被系统量化和解决。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Sparrow 论文中的 instruction 多样性分析方法：
```
# Instruction 多样性评估流程
def analyze_instruction_diversity(dataset, n_samples=5000):
    # Step 1: 采样 instruction 文本
    instructions = random_sample(dataset, n_samples)

    # Step 2: 将 instruction 编码为 embedding
    # 使用 sentence transformer（如 all-MiniLM-L6-v2, d=384）
    embeddings = SentenceTransformer.encode(instructions)
    # embeddings shape: [5000, d=384]

    # Step 3: t-SNE 降维可视化
    tsne = TSNE(n_components=2, perplexity=30)
    reduced = tsne.fit_transform(embeddings)

    # Step 4: 分析聚类特征
    # ShareGemini → 9 个清晰的聚类簇（对应 9 种模板变体）
    # Video-ChatGPT → 相对分散但覆盖范围仍有限
    # Sparrow hybrid 混合后 → 分布范围显著扩展
    return reduced, cluster_labels

# Sparrow 增强 instruction 多样性的方法
def sparrow_augment_diversity(video_data, text_data, mix_ratio=2):
    """
    video_data: (ShareGemini + Video-ChatGPT, 1:1 采样)
    text_data:  (LongAlpaca + LongQLora, 1:1 采样)
    每个 text sample: (long_context, instruction, answer)
      - long_context → split by ~115 words → render as images
      - instruction: 书籍摘要、论文问答、文档理解等多样化任务
    """
    syn_samples = [text_to_images(s) for s in text_data]
    # 混合采样: video:synthetic = 2:1
    mixed_data = SampleConcat(video_data, syn_samples, ratio=mix_ratio)
    return mixed_data
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
提高 instruction 多样性有几种策略：
1. **数据源多样化**（Sparrow 方法）：引入文本域的 instruction 数据（LongAlpaca 覆盖书籍/论文的长上下文问答，LongQLora 覆盖长文档对话）。文本域 instruction 天然具有更高的提问多样性，无需额外标注，通过 text-to-image 合成转化为视觉格式混合训练。
2. **Human-in-the-loop 精炼**：人工标注者审查和丰富 instruction（如 Video-ChatGPT 的小部分数据），成本高但质量好。
3. **模板扩展**：在已有数据集上通过改写/重述扩展 instruction 模板，但覆盖范围受限于数据集固有内容。
4. **Multi-source 混合**：混合多个不同来源的数据集，利用不同数据集的自然分布差异增加多样性。

评估指标：当前主要通过 (a) t-SNE 可视化定性评估 instruction embedding 分布的覆盖范围；(b) 数据缩放实验的 learning curve 定量评估多样性改善效果——更陡峭/更持久的 scaling curve 说明多样性更足，更平的对数曲线说明多样性不足。

涉及论文标题：
- T2Vid__Translating_Long_Text_into_Multi-Image_is_the_Catalyst_for_Video-LLMs
