## TextTiling

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
TextTiling 是 Marti Hearst 在 1993 年 ACL 会议提出、1997 年在 Computational Linguistics 期刊上正式发表的文本分割算法。它是一种无监督、领域无关的算法，用于将说明性文本自动分割为连贯的多段子主题段落。核心思想基于词汇衔接理论（Halliday & Hasan, 1976）：当文本子主题变化时，显著比例的词汇也会随之变化。算法步骤：(1) Tokenization：将文本分为固定大小的"伪句"（pseudo-sentences，通常每 20 个 token），去除功能词；(2) 相似度计算：以滑动窗口（block size k，通常 k=6）计算相邻块间的余弦相似度，基于词频向量；(3) 边界识别：平滑相似度曲线，计算每个 gap 的 depth score（衡量相似度在 gap 两侧下降的幅度），将 depth score 最大的 gap 作为子主题边界。TextTiling 被集成在 NLTK 中，广泛应用于信息检索（段落检索）、文本摘要（选取每个子主题的代表句）和下游 NLP 任务（如词义消歧的上下文窗口选择）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
TextTiling 算法伪代码（与 SceneTiling 对比）：

```
# === TextTiling (NLP) ===
# 输入: 文档 D = {w_1, ..., w_N} (词序列)
# 参数: w=20 (伪句大小), k=6 (block size)

# Step 1: Tokenization → 伪句
pseudo_sentences = group_by_size(remove_stopwords(D), w)  # 每组20个实词
# 输出: PS = {ps_1, ps_2, ..., ps_T}

# Step 2: Block 间余弦相似度
for i = 1 to T-1:
    block_left = TF_vector(ps_{i-k+1}, ..., ps_i)   # k个伪句的词频向量
    block_right = TF_vector(ps_{i+1}, ..., ps_{i+k})
    sim_i = CosineSimilarity(block_left, block_right)

# Step 3: Depth Score 计算
for i = 1 to T-1:
    d_i = (max_left_sim(i) + max_right_sim(i) - 2*sim_i) / 2
    # 识别相似度的"低谷"位置

# Step 4: 边界选取
boundaries = top_m_depth_scores(d)  # 选取 m 个最大 depth score 位置

# === SceneTiling (Video) ===  [本文，与 TextTiling 对比]
# 差异1: 用 ViT [CLS] token 替代 TF 词频向量
# 差异2: 逐帧计算而非伪句分组
# 差异3: Block size k 在视频域不适用，直接比较相邻帧
# 差异4: 阈值选取用 μ+ασ 而非固定 top-m 选取
```

TextTiling 的核心假设——"子主题变化 ≈ 词汇变化"——在视频域中被 SceneTiling 映射为"场景变化 ≈ 视觉特征 [CLS] token 变化"。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
TextTiling 在 Python 中可通过 NLTK 直接使用：`from nltk.tokenize import TextTilingTokenizer`。参数可配置：(1) w (pseudosentence size, default=20)；(2) k (block size, default=10)；(3) smoothing_width (default=2)；(4) cutoff_policy (hc/top/diff, 控制边界数量)。在 VideoLLaMB 中，SceneTiling 将 TextTiling 从文本域迁移到视频域，核心修改：(1) 将词频向量替换为 ViT [CLS] token 表示；(2) 将伪句概念替换为单帧；(3) 将 block-based 相似度替换为直接相邻帧相似度（因为视频的时序性远强于文本）；(4) 使用阈值 μ+ασ 替代固定 top-m 选取，使分割数自适应于视频内容。局限性：(1) TextTiling 假设线性子主题结构，不支持层级式主题组织（Hierarchical TextTiling 已提出但未用于 SceneTiling）；(2) 仅依赖词汇分布（或视觉特征），不利用语义理解或领域知识。

涉及论文标题：
- VideoLLaMB__Long-context_Video_Understanding_with_Recurrent_Memory_Bridges
