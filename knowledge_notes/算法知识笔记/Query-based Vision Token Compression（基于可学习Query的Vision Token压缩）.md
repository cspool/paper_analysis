## Query-based Vision Token Compression（基于可学习Query的Vision Token压缩）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Query-based Vision Token Compression 是 LMM 中一种通过可学习 query 向量与 vision token 进行 cross-attention 来压缩视觉信息的技术。LLaVA-Mini (Zhang et al., 2025) 将其推向极致——将 CLIP ViT-L 输出的 576 个 vision token 压缩到仅 1 个 token（压缩率 0.17%），同时通过 modality pre-fusion 弥补压缩带来的信息损失，性能与 LLaVA-v1.5 可比。核心机制：(1) 引入 C×C 个可学习压缩 query Q^v（默认 C=1，即 1 个 query）；(2) 对 query 和原始 vision token 施加 2D sinusoidal positional encoding 保留空间位置信息；(3) Q^v 通过 cross-attention 与全部 vision token 交互，产生注意力矩阵 A [C^2, N^2]；(4) 压缩输出 Ĥ^v = A · H^v [C^2, d_h]。相比 average pooling，query-based 压缩可自适应关注关键区域（如 OCR 文字、价格标签），仅增加 2.42G FLOPs 但带来显著的精度提升。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Query-based Compression 的计算过程：
```
# 输入: H_v = [576, 4096] (576 vision tokens, d_h=4096 Vicuna-7B)
Q_v = learnable_compression_queries    # [C^2, 4096], C=1 → [1, 4096]
pos = 2D_Sinusoidal_PE()               # 2D 正弦位置编码

# Cross-attention: queries attend vision tokens
Q_with_pos = Q_v + pos(Q_v)            # [C^2, 4096]
K_with_pos = H_v + pos(H_v)            # [576, 4096]
A = Softmax(Q_with_pos @ K_with_pos.T) # [C^2, 576] 注意力矩阵

# 加权聚合压缩
H_v_compressed = A @ H_v               # [C^2, 4096]
```
关键设计：(1) 2D sinusoidal PE 保留 patch 的 2D 空间位置信息，这对图像理解至关重要——2D PE 比 1D PE 能更好地保留相邻 patch 的空间关系。(2) Cross-attention 无 causal mask——所有 query 平等 attend 所有 vision token。消融实验（Table 8）：1 token 时 query-based 77.6/60.9/65.6 (VQA-v2/GQA/MMB) vs average pooling 76.1/59.8/64.0，FLOPs 增加仅 2.42G（1.96T 总 FLOPs 的 0.12%）。可视化（Figure 12）：压缩 attention 在关键信息集中区域（文字、产品标签）聚焦明显，在主体不明确时分布更分散。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现上，compression queries 为 nn.Parameter，cross-attention 复用标准 MultiHeadAttention（Q=queries, K=V=vision_tokens）。查询数 C^2 为超参数：标准分辨率 C=1 → 1 token；HD 高分辨率 C=8 → 64 tokens。在 LLaVA-Mini 的两阶段训练中，compression module 在 Stage 2 引入并端到端训练。C 值可配置以在效率-精度间 trade-off（Table 7：1 token VQA-v2 77.6, 64 tokens 78.5, 576 tokens 80.0）。相关方法对比：与 BLIP-2 Q-Former 的 32 个固定 query 不同，LLaVA-Mini 将 query 数压到 1 并配合 pre-fusion 补偿；与 PruMerge（基于 token 相似度合并）和 VoCo-LLaMA（用 LLM 压缩）也不同。开源：https://github.com/ictnlp/LLaVA-Mini。

涉及论文标题：
- LLaVA-Mini__Efficient_Image_and_Video_Large_Multimodal_Models_with_One_Vision_Token
