## Latent Visual Tokens（隐空间视觉 Token）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Latent Visual Tokens 是 Mirage 框架的核心创新：在 VLM 自回归解码过程中，将模型最后一层 hidden state 直接作为紧凑的连续视觉 embedding（而非通过 LM head 映射到离散 vocabulary），插入文本 token 序列中供后续 token 的 self-attention 访问。与显式图像生成 (Anole, MVoT, Chameleon) 不同，latent visual tokens 不需要 external image decoder，也不产生 pixel-level output。它们本质上是对 VLM 内部已编码的视觉信息的高效"回放"——通过 bypass LM head、reuse hidden state，将视觉推理信息以连续向量的形式保留在 multi-modal embedding space 中，供后续 reasoning step 直接 attend。k 个 latent tokens 通过 average pooling 压缩自输入 helper image I 的 patch-level features（{e_1,...,e_n} → {ê_1,...,ê_k}），k 默认=4。这种机制受到人类 mental imagery 的启发：人类在推理时不生成照片级画面，而是构建简化的心智草图。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# === Latent Visual Token 生成 (Mirage Inference) ===
# 模型: Qwen2.5-VL-7B, k=4

# Stage 1: 生成文本前半部分 o_pre
o_pre = VLM.generate(x, stop_at="<image_placeholder>")

# Stage 2: 生成 k 个 latent visual tokens (bypass LM head)
e = []  # list of continuous embeddings
for j in range(k):
    hidden = VLM.last_hidden_state(x, o_pre, e)  # forward through LLM
    e_j = hidden[-1]  # shape: (d_model,) = (4096,) for 7B
    e.append(e_j)
    # e_j directly used as embedding for next token position
    # NOT mapped through LM head → NOT discrete token

# Stage 3: 基于 latent tokens 生成后续文本
o_post = VLM.generate(x, o_pre, e)  # o_post attends to e_{1:k}
# self-attention: Q_text @ [K_text_pre, K_e1..K_ek, K_post_<t]^T
answer = extract_answer(o_post | e_{1:k})

# === Stage 1 训练: Joint Supervision ===
patch_feats = VLM.vision_encoder(helper_image_I)  # {e_1,...,e_n}
target_embeds = avg_pool(patch_feats, k=4)         # {ê_1,...,ê_4}
# L_visual = Σ_j cos_sim(ê_j, h_j), h_j 为模型在 latent slot 的 hidden state
# L_1 = L_visual + 0.1 * L_text

# === Stage 2 训练: Latent Relaxation ===
e_j = VLM.hidden_state(x, o_pre, e_{<j})  # 模型自回归生成
# L_2 = CE(o_pre) + CE(o_post | e_{1:k})  # 仅文本 CE loss
# 梯度通过 o_post 的 CE loss 反向传播到 e_j
```

对比 Coconut (LLM continuous thought): Coconut 在纯文本 LLM latent space 中操作，无需视觉 grounding；Mirage 在 VLM 多模态空间中操作，Stage 1 提供 visual embedding distillation 锚定机制——消融显示 w/o Stage 1 会使性能从 58% 降至 21% (VSP Spatial Planning)。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现点：(1) LLM decoder loop 中检测 `<image_placeholder>` token 或通过 gating mechanism 切换 latent generation path；(2) bypass LM head: forward pass 产生 hidden state h ∈ R^d 后，h 直接作为嵌入向量缓存到 KV cache，而非映射为离散 token 的概率分布；(3) 训练时 k 个 latent slots 的 hidden states 在 Stage 1 通过 cosine similarity 对齐 target embeddings，Stage 2 自回归生成并通过 downstream CE loss 接收梯度。开源：https://github.com/UMass-Embodied-AGI/Mirage。适用场景：需要视觉想象的多模态推理任务（jigsaw, spatial planning, navigation），特别适合 bypass 图像生成、避免 unified model 推理-生成冲突的场景。局限性：k>6 时性能下降（latent sequence 误差累积），目前限于 spatial reasoning benchmarks。

涉及论文标题：
- Machine_Mental_Imagery__Empower_Multimodal_Reasoning_with_Latent_Visual_Tokens
