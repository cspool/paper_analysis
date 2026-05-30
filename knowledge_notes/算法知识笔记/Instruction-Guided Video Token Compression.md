## Instruction-Guided Video Token Compression

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Instruction-Guided Video Token Compression 是 TDC 论文提出的将用户指令注入视频 token 压缩过程的技术。在 Q-Former cross-attention 阶段，将 instruction text tokens F_s 作为额外 key-value 输入（公式: F_Q^i = QFormer(Q, [F_xi · F_ai], F_s)），使 Q-Former 根据问题语义自适应决定每个 query token 从视觉/音频 tokens 中提取什么信息。相比无 instruction，加入 F_s 后压缩不仅捕捉帧间时序变化，还能聚焦与问题相关的细节。消融（Table 4d）：with text vs without text: MVBench 62.7 vs 62.3, MLVU Long 59.6 vs 58.0 (-1.6), VideoMME 52.7 vs 51.5 (-1.2)。instruction 对长视频帮助更大。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Q-Former with Instruction-Guided Compression
# Input: Q(16 query tokens), F_xi(144 visual), F_ai(50 audio), F_s (text)

T_s = tokenize(F_s); E_s = embed(T_s)              # text embeddings
K = linear_k(concat(F_xi, F_ai, E_s))              # (144+50+L_s, d)
V = linear_v(concat(F_xi, F_ai, E_s))
Q_proj = linear_q(Q)                                # (16, d)

attn = softmax(Q_proj @ K.T / sqrt(d))              # (16, 144+50+L_s)
F_Q_i = attn @ V                                    # (16, d) compressed
```
对比 prior work: LongVU 基于 visual similarity + query relevance 压缩后再筛选，TDC 在压缩过程中（Q-Former 内部）注入 instruction。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
TDC 在 Q-Former forward 中增加 text_cross_attention 参数接收 instruction embeddings。instruction text 仅参与 key-value 计算，不改变输出维度（仍 K tokens）。计算开销：key-value 长度增加 L_s (通常 10-50 tokens)，可忽略。适用场景：任何 cross-attention compressor 均可加入 instruction guidance 提升压缩质量。

涉及论文标题：
- Multimodal_Long_Video_Modeling_Based_on_Temporal_Dynamic_Context
