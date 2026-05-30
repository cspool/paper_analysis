## Cosine Similarity Loss for Hidden State Alignment（隐状态对齐的余弦相似度损失）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
这是 Mirage Stage 1 中用于将模型 latent slot 位置的 hidden states 锚定到 visual embedding subspace 的损失函数。损失形式：L_visual = (1/k) Σ_{j=1..k} [1 - cos_sim(ê_j, h_j)]，其中 ê_j 是从 helper image I 压缩得到的 target visual embedding（通过 average pooling），h_j 是模型在对应 latent slot 位置的 hidden state（prediction）。余弦相似度 cos_sim(a, b) = a·b / (||a||·||b||) 度量两个向量的方向对齐程度。选择 cosine similarity 而非 MSE 的理由：(1) hidden states 和 visual embeddings 处于不同 latent subspace，但方向 encode 关键语义；(2) 对向量模长不敏感，防止模型通过放大隐藏状态模长 "cheat"；(3) 在高维空间 (d=4096) 中方向比尺度更重要。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Stage 1: Cosine Similarity Loss for Latent Grounding

# Input: helper image I, 模型 VLM f_θ, k=4

# Step 1: 提取压缩后的目标 visual embeddings
patch_feats  = f_θ.vision_encoder(I)        # [n_patches, d_model]
target_embeds = avg_pool(patch_feats, k=4)   # [4, d_model] = {ê_1,...,ê_4}

# Step 2: Forward pass 获取 hidden states at latent slots
h_1 = f_θ.hidden_state(x, o_pre)                    # 第一个 latent slot
h_2 = f_θ.hidden_state(x, o_pre, ê_1)               # 第二个, conditioned on ê_1 (teacher forcing)
h_3 = f_θ.hidden_state(x, o_pre, ê_1, ê_2)          # 第三个
h_4 = f_θ.hidden_state(x, o_pre, ê_1, ê_2, ê_3)    # 第四个

# Step 3: 计算 Cosine Similarity Loss
L_visual = 0
for j in range(1, 5):
    cos_sim_j = (ê_j · h_j) / (||ê_j|| * ||h_j||)
    L_visual += (1 - cos_sim_j)
L_visual /= 4

# Step 4: 联合文本 CE Loss
L_text = CE(o_pre) + CE(o_post | o_pre, ê_{1:k})
L_total = L_visual + 0.1 * L_text  # γ=0.1
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
PyTorch: `F.cosine_similarity(h_j, ê_j, dim=-1)` 返回 (cos_sim+1)/2 带范围 [-1,1]，或直接 `1 - F.cosine_similarity(...)`。γ=0.1 意味着 visual alignment loss 权重约为 text CE 的 10 倍，确保 Stage 1 latent grounding 有效性。消融 (Tab 5): γ=0.1 → 87% avg, γ=0.5 → 84%, γ=1 → 83%；γ→∞ (跳过 Stage 1) → 21% (w/o Stage 2 setting)，证明 cosine loss 提供的 visual grounding 是关键初始化。使用场景：任何需要将模型内部表征对齐到特定 target embedding 的跨模态场景（multimodal alignment, knowledge distillation, latent reasoning）。

涉及论文标题：
- Machine_Mental_Imagery__Empower_Multimodal_Reasoning_with_Latent_Visual_Tokens
