## Cross-modal Attention (Text-to-Image Attention for VLM Pruning)

术语是什么？通过联网搜索让回答具体和精准。
Cross-modal Attention在VLM中指text tokens与image tokens之间的attention交互。在Vision-Language Model的Transformer attention层中，QK^T SoftMax矩阵包含四个block：image-to-image (M×M)、image-to-text、text-to-image (T×M)、text-to-text。Text-to-Image block的每个元素I[i,j]表示第i个text token对第j个image token的attention score，反映了"语言查询"对"视觉内容"的关注程度。Focus利用这个cross-modal attention做prompt-aware token pruning：对每个image token j计算其从所有text tokens和attention heads中接收到的最大attention score作为importance指标，从而根据实际文本prompt动态选择相关视觉区域。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Cross-modal attention在VLM attention层的计算流程：
```
# Attention computation in VLM Transformer layer
Q = Linear_q(concat(image_tokens, text_tokens))  # (M+T) × d
K = Linear_k(concat(image_tokens, text_tokens))
V = Linear_v(concat(image_tokens, text_tokens))
S = Q @ K^T  # (M+T) × (M+T) attention scores
# S is partitioned into 4 blocks:
# S[0:M, 0:M]     → image-to-image
# S[0:M, M:M+T]   → image-to-text
# S[M:M+T, 0:M]   → text-to-image (cross-modal)
# S[M:M+T, M:M+T] → text-to-text

# SEC extracts text-to-image block for importance estimation
I = S[M:M+T, 0:M]  # T×M matrix
for each image token j:
    importance[j] = max_{i=1..T, k=1..n_heads} I^{(k)}[i,j]
# This captures how much ANY text token attends to image token j
# across ALL attention heads
```
当prompt问"What is the type of the dog?"时，attention集中在狗的位置；当问"What is the color of the flower?"时，attention转向花的位置。SEC利用这个prompt-dependent attention模式做动态token pruning。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Cross-modal attention是standard VLM Transformer attention的天然产物，无需额外计算。Focus的SEC通过从已计算的SoftMax(QK^T)中提取text-to-image block来获取cross-modal attention scores，不引入额外attention层或模块。在硬件实现中，SEC的importance analyzer从systolic array输出的attention SoftMax结果中直接stream text-to-image attention columns到parallel max units进行计算，完全on-chip、streaming。此方法可推广至任何使用cross-modal attention的VLM架构。

涉及论文标题：
- Focus: A Streaming Concentration Architecture for Efficient Vision-Language Models

