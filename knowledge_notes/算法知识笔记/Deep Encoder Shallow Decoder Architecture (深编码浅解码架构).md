## Deep Encoder Shallow Decoder Architecture (深编码浅解码架构)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Deep Encoder Shallow Decoder 是一种为机器翻译任务设计的 encoder-decoder Transformer 架构变体，将 encoder 层数设为 decoder 层数的约 2 倍。传统 Transformer 的 encoder 和 decoder 通常层数对称，但在自回归推理中 decoder 是性能瓶颈——encoder 只执行一次而 decoder 每生成一个 token 都要执行一次 beam search。通过减少 decoder 层数来降低自回归解码计算开销，同时保持深度 encoder 的编码能力以保证翻译质量。

论文 "Who Says Elephants Can't Run" 使用 24 encoder layers + 12 decoder layers（2:1），embedding dim 1024, FFN hidden dim 4096，每 2 层使用 MoE FFN 层代替 dense FFN。该配置来自 Kim et al. (2021) 和 Kasai et al. (2020)，实验验证为 quality-performance 的 trade-off 最优点。

从算法pipeline角度拆解术语：

Auto-regressive 推理的层执行模式：
```
# === Encoder: 执行 1 次 ===
h_enc = embed(input_tokens)           # B × S_in tokens
for l in 1..24:                       # 24 layers
    h_enc = TUPE_self_attn(h_enc) + h_enc
    if l % 2 == 0:                    # 12 MoE layers in encoder
        h_enc = MoE_FFN(h_enc) + h_enc

# === Decoder: 每 token 执行 1 次（自回归 bottleneck） ===
for t in 1..T_out:
    for l in 1..12:                   # 12 layers (half!)
        h_dec = TUPE_self_attn(h_dec) + h_dec
        h_dec = cross_attn(h_dec, h_enc) + h_dec
        if l % 2 == 0:                # 6 MoE layers in decoder
            h_dec = MoE_FFN(h_dec) + h_dec
    next_token = argmax(lm_head(h_dec[:,-1,:]))
```

为什么有效：在 beam search 中，decoder 执行成本 = B × K × T_out × L_dec × cost_per_layer，encoder 执行成本 = B × S_in × L_enc × cost_per_layer。由于 T_out × K 通常远大于 S_in，decoder 深度影响巨大。L_dec 减半 ≈ decoder 计算减半 ≈ 总延迟约减半。

术语一般如何实现？如何使用？

在 PyTorch/HuggingFace 中通过 `EncoderDecoderModel` 或自定义 `nn.Module` 配置不同的 encoder/decoder 层数参数。Kim et al. (2019) 最早在 CPU 部署中使用此架构实现极快机器翻译。Kasai et al. (2020) 发现 encoder 至少 2× decoder 深度以保证非自回归蒸馏训练质量。

涉及论文标题：
- Who Says Elephants Can't Run: Bringing Large Scale MoE Models into Cloud Scale Production
