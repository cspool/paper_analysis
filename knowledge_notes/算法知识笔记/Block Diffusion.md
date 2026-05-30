## Block Diffusion

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Block Diffusion是一种介于全扩散（bidirectional, 全并行）和全自回归（causal, 全顺序）之间的生成范式。核心思想：将序列划分为固定大小的块（blocks），块内使用bidirectional attention + masked token prediction（扩散范式），块间使用causal conditioning（自回归范式）。Fast-dLLM v2首次将block diffusion扩展到现代LLM规模（7B），提出完整的训练+推理recipe：训练时使用block-wise attention mask（M_BD+M_OBC+M_BC）+ complementary masking + token shift；推理时使用block级KV cache（跨block复用已解码上下文）+ sub-block DualCache（块内高效并行refinement）+ confidence-aware parallel decoding。仅需~1B tokens微调即可将预训练AR模型（Qwen2.5-Instruct）转化为block diffusion模型，相比Dream的580B tokens减少500×。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Fast-dLLM v2的完整Block Diffusion pipeline：

```
# === 训练阶段 ===
Input: 预训练AR模型 θ_AR, 训练数据D, block size D=32
1: 将序列padding到D的整数倍（padding token不参与loss）
2: Packing: 多个样本拼接至context length L
3: 对每个block b采样random mask m_b和互补mask m̄_b
4: 两个view放入同一batch
5: Attention mask M_full = [[M_BD, M_OBC], [0, M_BC]]
6: Token shift: 位置i-1的hidden state预测位置i的token
7: Loss: masked-token-only cross-entropy

# === 推理阶段 ===
Input: prompt p, target_len L, block_size B=32, sub_block_size S=8
1: x ← [p; [MASK]×L]
2: for k in 1..⌈L/B⌉:                       # 逐block
3:     复用block级KV cache（已解码block的K/V）
4:     for each sub-block in current block:
5:         bidirectional attention within sub-block
6:         confidence > τ token并行解码（threshold=0.9）
7:         DualCache复用sub-block prefix/suffix K/V
8:     end
9:     刷新block级KV cache
10: end
```

Block Diffusion的关键优势：(1) 块内bidirectional attention提供更丰富的context modeling；(2) 块间causal conditioning保证全局语义连贯性；(3) block级KV cache实现与AR模型类似的cache复用；(4) 训练与AR模型高度兼容（仅需1B tokens微调）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Fast-dLLM v1（训练无关的block diffusion加速）已开源：https://github.com/NVlabs/Fast-dLLM。Fast-dLLM v2代码和模型待发布。训练使用64×A100 GPU + DeepSpeed Zero-3，1.5B模型约8小时，7B模型约12小时。推理时block size=32, sub-block size=8, threshold=0.9实现2.5×加速（vs AR baseline）。Block diffusion的block size是关键的accuracy-efficiency trade-off参数：训练和推理block size应保持一致（mismatch导致显著性能退化，Table 4）；sub-block size提供推理粒度的灵活调节（Table 3，size=8最优）。

涉及论文标题：
- Fast-dLLM Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding
- Fast-dLLM v2: Efficient Block-Diffusion LLM
