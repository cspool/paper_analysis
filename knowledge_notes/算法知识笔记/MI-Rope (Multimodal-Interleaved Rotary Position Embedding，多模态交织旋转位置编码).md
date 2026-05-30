## MI-Rope (Multimodal-Interleaved Rotary Position Embedding，多模态交织旋转位置编码)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
MI-Rope 是 mPLUG-Owl3 提出的用于多图交织场景的位置编码方法。核心思想：在图像-文本交织输入中，每张图的所有 patch tokens 共享该图在文本序列中占位符 T_img 的 RoPE 位置编码。与 MRoPE（Qwen2.5-VL，使用 3D (T,H,W) 位置 ID）不同，MI-Rope 仅使用 1D 位置 ID——即文本序列中的 token 索引位置。这确保了：(1) 图像间的相对顺序由占位符的文本位置自然编码；(2) 图像与前后文本的上下文关系被保留；(3) 配合 causal attention mask，每个文本 token 仅能 cross-attend 到已出现的视觉特征，保持自回归生成的一致性。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
MI-Rope 的位置分配逻辑：
```
# 输入: S_text = [T1, T_img, T2, T_img, T3] (文本+占位符序列)
# 图像特征: I1 patches ∈ R^{P1×D}, I2 patches ∈ R^{P2×D}

# Step 1: 记录每张图的占位符位置
pos_images = []
for token_idx, token in enumerate(S_text):
    if token == "<|image|>":
        pos_images.append(token_idx)

# Step 2: 为所有 visual patches 分配位置编码
# I1 的所有 P1 个 patches 共享 S_text 中第一个 T_img 的位置
# I2 的所有 P2 个 patches 共享 S_text 中第二个 T_img 的位置
# 例: S_text = [0:T1, 1:T_img, 2:T2, 3:T_img, 4:T3]
#     I1 patches → pos=1, I2 patches → pos=3

# Step 3: 在 cross-attention 中应用 RoPE
Q_rope = rotary_embed(Q_text, pos_text)        # 文本 Q: 自身序列位置
K_img_rope = rotary_embed(K_img, pos_images)   # 视觉 K: 占位符位置
```

与 MRoPE 的关键区别：
- MRoPE (Qwen2.5-VL): pos = (temporal_id, height_id, width_id)，三维位置，三组频率分别旋转不同维度段
- MI-Rope (mPLUG-Owl3): pos = 占位符在文本序列中的 1D 索引，所有 patches 共享同一位置

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
MI-Rope 在 PyTorch 中实现为：预计算每个 batch 的 pos_images 张量（shape: [total_img_patches]），在 HATB 的 cross-attention 中调用标准 RoPE 实现（如 transformers 库的 LlamaRotaryEmbedding），但传入的位置索引为占位符位置而非原始图像网格位置。消融实验验证：去掉 MI-Rope 后多图 benchmark（NLVR2, Mantis-Eval）性能显著下降，视频 benchmark 影响较小（视频帧有时间顺序可被隐式建模）。MI-Rope 与 Shared LayerNorm 和 Adaptive Gating 协同工作，共同构成 HATB 的完整设计。

涉及论文标题：
- mPLUG-Owl3__Towards_Long_Image-Sequence_Understanding_in_Multi-Modal_Large_Language_Models
