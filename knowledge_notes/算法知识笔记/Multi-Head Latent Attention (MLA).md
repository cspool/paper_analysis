## Multi-Head Latent Attention (MLA)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Multi-Head Latent Attention (MLA) 是DeepSeek-V2/V3/R1系列模型提出的注意力机制变体，通过低秩联合压缩Key和Value进入共享latent空间来大幅减少KV cache内存占用。与MHA（Multi-Head Attention）和GQA（Grouped Query Attention）不同，MLA不直接存储每个head的K和V，而是将K/V投影到一个低秩latent表示 c_KV ∈ R^{d_latent}（d_latent << num_heads × head_dim），推理时仅缓存latent vector而非完整的KV cache。decoding时从latent vector上投影回各head的K和V。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

MLA的算法pipeline（简化）：

```
# 压缩阶段（prefill，每个token）:
c_KV = x @ W_down     # x∈R^d, c_KV∈R^{d_latent}，latent压缩
# 缓存 c_KV 而非完整 K/V

# Decompression（decode，每个query token）:
K_i = c_KV @ W_UK_i + x @ W_KR_i   # 第i个head的K，from latent + rope部分
V_i = c_KV @ W_UV_i                 # 第i个head的V，from latent

# Attention计算（仍使用online softmax）:
S_i = Q_i × K_i^T / sqrt(d_k)
O_i = softmax(S_i) × V_i
```

MLA的核心trade-off：decode从memory-bound变为compute-bound（因latent decompression引入额外计算，但减少了HBM KV cache读取量）。BLASST paper验证了其在MLA上的兼容性：DeepSeek-R1使用MLA + BLASST，在60% sparsity下GPQA Diamond/Mmlu Pro/LiveCodeBench准确率几乎无退化（Table 11）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

MLA已在DeepSeek-V2、DeepSeek-V3、DeepSeek-R1等模型中实现，开源代码在DeepSeek的HuggingFace模型仓库中。MetaAttention框架将MLA作为一种RowNorm-based attention variant支持。从实现角度看，MLA的关键是latent space的维度选择（d_latent通常为512或576），以及rope positional encoding与latent-compressed部分和non-compressed部分的组合方式。

涉及论文标题：
- BLASST: Dynamic BLocked Attention Sparsity via Softmax Thresholding
