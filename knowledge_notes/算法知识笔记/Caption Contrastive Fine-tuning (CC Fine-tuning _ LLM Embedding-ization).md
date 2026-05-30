## Caption Contrastive Fine-tuning (CC Fine-tuning / LLM Embedding-ization)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Caption Contrastive Fine-tuning (CC Fine-tuning) 是 LLM2CLIP 提出的 Stage 1 训练方法，目的是将 LLM 改造为适合 CLIP 跨模态对比训练的文本嵌入模型（即 "embedding-ization"）。核心问题：原始 LLM 的 token 输出层是 classification head（预测离散文本 token），其最后的 hidden state 对不同 caption 的语义可分离性极差——LLM2CLIP Table A1 显示 Llama3-8B 在 COCO caption-to-caption retrieval 上 Top-1 仅 5.2%，远低于 CLIP text encoder 的 25.2%。CC Fine-tuning 通过三项设计解决此问题：(1) **模型架构改造**：移除 causal attention mask → 启用 bidirectional attention；使用 average pooling 聚合所有 output tokens 获得句子嵌入；通过 LoRA (r=16, α=32) 参数高效微调。(2) **监督 SimCSE 对比损失**：使用同一图像的两个不同 caption 作为正样本对，以 in-batch 其他 caption 为负样本，最大化正样本对 cosine similarity。(3) **混合训练数据**：DreamLIP 30M captions + Echo Embeddings 1.5M 纯文本对，保证文本区分能力的泛化性。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Stage 1: CC Fine-tuning
# LLM: Llama 3.1 8B, LoRA (r=16, α=32)
# 移除 causal attention mask, 使用 full bidirectional attention
# 从 DreamLIP 数据中采样: (c_i, c_j) 为同一图像的两个 caption

def cc_finetune_step(llm, captions_pairs, optimizer):
    # c_i, c_j: tokenized captions, shape [B, L]
    # 构建系统 prompt: "Given a caption, retrieve a similar relevant caption."
    c_i = [system_prompt + cap for cap in c_i]
    c_j = [system_prompt + cap for cap in c_j]

    # 前向: bidirectional attention (无 causal mask)
    h_i = llm(c_i, causal_mask=False)  # [B, L, d_llm]
    h_j = llm(c_j, causal_mask=False)  # [B, L, d_llm]

    # Average pooling (而非 [EOS] token)
    e_i = h_i.mean(dim=1)  # [B, d_llm]
    e_j = h_j.mean(dim=1)  # [B, d_llm]

    # 监督 SimCSE 对比损失
    sim = cosine_similarity(e_i, e_j) / τ  # [B, B]
    labels = arange(B)  # 对角线表示 (e_i[k], e_j[k]) 为正样本对
    loss = CrossEntropy(sim, labels)

    loss.backward()
    optimizer.step()  # 仅更新 LoRA 参数
    return loss

# 训练配置: AdamW lr=2e-4, 300-step warmup, seq_len=512
#           有效 batch_size=2048, 1 epoch over 30M samples
#           32 A100 GPUs, bfloat16 + FlashAttention-2
```

Annotations: `d_llm` = 4096 (Llama 3.1 8B hidden dim)；`τ` 为 temperature 参数；in-batch negatives 利用 batch 内其他样本的 caption 作为负样本；DreamLIP 为每个图像提供多个 dense captions，因此可以采样两个不同 caption 作为正样本对。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
CC Fine-tuning 的关键消融发现（LLM2CLIP Table 6/Table A5）：(1) Supervised SimCSE >> Unsupervised SimCSE >> MNTP alone；(2) Bidirectional attention 与 causal attention 性能相近（80.4 vs 80.0 Avg I2T），但 bidirectional 能更好建模文本双向关系；(3) Average pooling 优于 [EOS] token (80.4 vs 80.0)；(4) LoRA 是必需的——冻结 LLM + 仅训练 Adaptor 性能显著下降 (74.1 vs 80.4)；(5) MNTP + SimCSE 组合不优于 SimCSE alone。CC fine-tuned LLM 的分离能力超越原始 CLIP text encoder（Top-1 29.5% vs 25.2%）。CC Fine-tuning 后 LLM 的特征空间已具备充分的 caption 区分能力，可作为 Stage 2 中 CLIP 视觉编码器训练的有效文本监督信号。

涉及论文标题：
- LLM2CLIP__Powerful_Language_Model_Unlock_Richer_Visual_Representation
