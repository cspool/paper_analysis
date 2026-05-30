## SimCSE (Simple Contrastive Learning of Sentence Embeddings)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SimCSE (Simple Contrastive Learning of Sentence Embeddings) 是一种通过对比学习训练句子嵌入的方法，由 Gao et al. (EMNLP 2021) 提出。核心思想：将同一个输入句子通过不同的 dropout mask 传递两次（unsupervised 版本），或使用标注的正样本对（supervised 版本），将这些变体作为正样本对，batch 内其他句子作为负样本，通过对比损失（NT-Xent loss）训练编码器使正样本对的嵌入彼此接近、与负样本嵌入远离。训练目标等价于最大化正样本对 cosine similarity 同时最小化与负样本的 similarity。LLM2CLIP 使用 supervised SimCSE 变体：正样本对为同一图像的两个不同 caption（由系统 prompt "Given a caption, retrieve a similar relevant caption" 构建），负样本为 batch 内其他 caption。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Supervised SimCSE for CC Fine-tuning
# 输入: batch of paired captions (c_i, c_j) from same image

def simcse_loss(encoder, batch_pairs, τ=0.05):
    # batch_pairs: [(c_1_i, c_1_j), ..., (c_B_i, c_B_j)]
    c_i_batch = [pair[0] for pair in batch_pairs]  # B captions
    c_j_batch = [pair[1] for pair in batch_pairs]  # B captions

    # 编码两个视图
    h_i = encoder(c_i_batch)  # [B, d]
    h_j = encoder(c_j_batch)  # [B, d]

    # L2 normalize
    z_i = h_i / ||h_i||_2   # [B, d]
    z_j = h_j / ||h_j||_2   # [B, d]

    # NT-Xent loss (symmetric)
    sim = z_i @ z_j.T / τ   # [B, B]
    labels = arange(B)       # (0,1,...,B-1)

    # 两个方向: i→j 和 j→i
    loss_i2j = CrossEntropy(sim, labels)
    loss_j2i = CrossEntropy(sim.T, labels)
    loss = (loss_i2j + loss_j2i) / 2

    return loss

# Unsupervised SimCSE (for comparison):
# def unsupervised_simcse(encoder, sentences):
#     # 同一句子经过两次带不同 dropout 的前向
#     z1 = encoder(sentences, dropout=True)  # 第1次 dropout
#     z2 = encoder(sentences, dropout=True)  # 第2次 dropout
#     # 其余计算相同
```

Annotations: `τ` 为 temperature (通常 0.05)；`d` 为 embedding 维度；supervised SimCSE 使用标注的正样本对（LLM2CLIP 中为同一图像的不同 caption），unsupervised 依赖 dropout 噪声产生正样本变体。LLM2CLIP Table A5 显示 supervised SimCSE (Avg I2T 80.4) 显著优于 unsupervised (59.2)。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
SimCSE 的使用方式：(1) Unsupervised: 仅需未标注文本，通过 dropout 增强产生正样本，适合大规模无标注数据场景。(2) Supervised: 需要标注的正样本对（如 NLI 数据、paraphrase 数据、或同一图像的不同 caption），性能优于 unsupervised。(3) 在 LLM2CLIP 中，supervised SimCSE 是 CC Fine-tuning 阶段的核心损失函数，其有效性源于：DreamLIP 为每张图像提供多个 dense captions → 天然的正样本对来源 → 训练 LLM 学习"两个相似语义的 caption 应具有相似的嵌入表示"这一能力 → LLM 特征空间获得 caption 语义可分离性。LLM2CLIP 的消融显示 SimCSE 是最关键的损失组分——仅用 MNTP 无 SimCSE 时 Avg I2T 从 80.4 降至 70.1。

涉及论文标题：
- LLM2CLIP__Powerful_Language_Model_Unlock_Richer_Visual_Representation
