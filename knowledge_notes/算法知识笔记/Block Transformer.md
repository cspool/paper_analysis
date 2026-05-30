## Block Transformer

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Block Transformer 是 Ho et al. (KAIST/LG AI/Google DeepMind, NeurIPS 2024) 提出的分层全局到局部（hierarchical global-to-local）Transformer 架构，将标准自注意力分解为 block 级全局注意力和 token 级局部注意力以系统性地降低自回归推理开销。架构包含：(1) **Embedder**：lookup table 将每 $L_B$ 个 subword token embedding 拼接为 block embedding；(2) **Block Decoder**：在 block 序列上运行的标准自回归 Transformer（序列长度 $L/L_B$），全局 causal self-attention，输出 context embedding；(3) **Token Decoder**：以 context embedding 投影出的 prefix tokens 为全局上下文来源，仅对当前 block 内 $L_B$ token 做局部 self-attention，解码 individual tokens。主配置 $L_B=4$, prefix=2, block:token 参数比 1:1。核心收益：Block Decoder 将 KV cache 大小降 $L_B$ 倍、KV cache IO 降 $L_B^2$ 倍；Token Decoder 将 KV cache IO 从 $O(L^2)$ 降至 $O(L \cdot L_B)$（线性复杂度），prefill 可完全跳过。开源：https://github.com/itsnamgyu/block-transformer。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以 Block Transformer (LB=4, prefix=2, L=2048) 推理 pipeline：
```
# Embedder: lookup table 拼接
E_emb = Embedding(V, D/LB)
block_emb[i] = Concat([E_emb[tok_{i*LB+j}] for j in range(LB)])  # [B, 512, D]

# Block Decoder: block-level global SA
h = BlockDecoder(block_emb)   # M_b layers causal SA, 512 blocks
context_emb = h[:, -1, :]     # [B, D] — 最后一个block位置的输出

# Token Decoder: token-level local SA
prefix = Linear(context_emb).view(B, P, D)       # [B, 2, D]
tok_embs = E_tok(curr_block_tokens)               # [B, 4, D]
h_tok = TokenDecoder(Concat([prefix, tok_embs]))  # [B, 6, D], 仅6token间SA
logits = Classifier(h_tok[:, -LB:, :])            # [B, 4, V]
```
KV cache 对比：Block Decoder 512 blocks (↓4× vs 2048), Token Decoder 6 tokens (↓341×)。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
基于 GPT-NeoX + HuggingFace Transformers + DeepSpeed ZeRO 实现。8×A100 40GB 训练 300B tokens (Pile)，H100 推理。吞吐量达 vanilla 的 10-25× (prefill-heavy/decode-heavy)。支持从预训练 vanilla transformer uptraining，仅需 10% 训练数据，为从现有大模型迁移到 Block Transformer 架构提供低成本的训练路径。

涉及论文标题：
- Block Transformer Global-to-Local Language Modeling for Fast Inference (NeurIPS 2024)

---
