## Offline Embedding Precomputation for LLM-based Training (Offline-loading / 离线嵌入预计算)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Offline Embedding Precomputation (Offline-loading) 是 LLM2CLIP Stage 2 提出的训练效率优化策略：在 CLIP 跨模态对比训练开始之前，用冻结的 CC fine-tuned LLM 对所有训练数据的文本 caption 预先计算文本嵌入并存入磁盘；训练时直接从磁盘加载预计算嵌入，通过 Adaptor 参与对比损失计算，完全避免将 LLM 加载到 GPU 显存中。核心优势：(1) LLM 推理从每个训练 epoch 执行一次 → 整个训练过程仅执行一次；(2) 训练时 GPU 显存无需容纳 LLM（8B 参数模型 bf16 约 16GB + optimizer states），可将节省的显存用于增大 batch size；(3) batch size 从 LLM LoRA 时 ~704 → offline-loading 时 16384（提升 23×）；(4) 训练时间从 17h (LLM LoRA) → 1.3h (offline-loading)。该策略可行性的前提是 LLM 参数在 Stage 2 完全冻结，嵌入为固定值不随训练变化。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Phase 1: Offline Precomputation (一次性)
def precompute_text_embeddings(llm, all_captions):
    llm.eval()
    embeddings = {}
    with torch.no_grad():
        for idx, caption in enumerate(all_captions):
            tokens = tokenize(caption)  # [1, L], seq_len ≤ 512
            h = llm(tokens, bidirectional=True)  # [1, L, 4096]
            emb = h.mean(dim=1)  # avg pooling → [1, 4096]
            embeddings[idx] = emb.cpu()  # 存入 CPU/磁盘
    return embeddings  # 持久化到磁盘

# Phase 2: Stage 2 Training with Offline-loading
def stage2_training_with_offline(caption_emb_path, images):
    precomputed = load(caption_emb_path)  # 加载预计算嵌入

    for batch_images, batch_ids in dataloader:
        # 视觉编码: ViT 在 GPU 上运行
        v_feat = ViT(batch_images)                      # [B, 1280]

        # 文本编码: 从磁盘加载预计算 LLM 嵌入 → Adaptor
        pre_emb = precomputed[batch_ids].to(device)     # [B, 4096]
        t_feat = Adaptor(pre_emb)                        # [B, 1280]

        # LLM 完全不加载到 GPU显存
        # CLIP contrastive loss
        loss = clip_loss(v_feat, t_feat)
        loss.backward()
        optimizer.step()  # 仅更新 ViT + Adaptor
```

Annotations: 预计算使用 32 A100 GPU 对 LLM 做单次前向推理；训练时 batch size 可达 16384 (2 nodes × 8 A100 40GB)；仅 ViT (307M~428M) + Adaptor (67.1M) 在 GPU 上训练，显存占用远低于同时加载 LLM (8B)；LLM 参数、optimizer states 完全不出现在训练显存中。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
LLM2CLIP Table A4 的 efficiency analysis 对比了三种策略的 trade-off：(1) LLM LoRA 在线训练: batch_size=704, 训练时间 17h, Avg I2T/T2I 85.4/82.5；(2) LLM Frozen + Linear Adaptor 在线: batch_size=4096, 5.5h, 83.9/82.1；(3) LLM Frozen + Adaptor + Offline-loading: batch_size=16384, 1.3h, 85.9/83.3。Offline-loading 不仅训练最快，性能也最高——更大的 batch size 对对比学习有益（更多 in-batch negatives）。适用条件：(a) 文本编码器参数冻结；(b) 训练数据在训练前完全已知（不需要在线生成文本）；(c) 存储空间足够容纳所有预计算嵌入（15M captions × 4096 × 2 bytes (bf16) ≈ 115GB，可控）。不适用场景：训练中需要实时生成或动态变化文本数据的情况。

涉及论文标题：
- LLM2CLIP__Powerful_Language_Model_Unlock_Richer_Visual_Representation
