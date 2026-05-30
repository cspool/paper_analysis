## Cross-Attention based MLLM Architecture（基于交叉注意力的多模态大语言模型架构）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
基于交叉注意力的 MLLM 架构是多模态大语言模型的三大架构范式之一（另两种为 Concatenation-based 和 Q-Former/Token Compression-based）。核心特征是视觉特征不直接进入 LLM 的文本序列，而是通过 cross-attention 层以"外部记忆"形式注入到 LLM 的中间表示中。代表工作：Flamingo（每层插入 cross-attention）、IDEFICS、EVLM 和 mPLUG-Owl3（稀疏 HATB）。天然优势：视觉 tokens 不占用 LLM context window，序列长度不随图像数量增长，多图/长视频场景下显存和计算效率远优于 Concatenation-based 方法。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
三种 MLLM 架构范式的对比：
```
=== Concatenation-based (LLaVA, InternVL, Mantis) ===
输入: [img_tokens_1, ..., img_tokens_N, text_tokens]
处理: 全序列进入 LLM，标准 causal self-attention
问题: 序列长度 = N×P + T, O(L²) attention
      N=100 张 384² 图 → ~57.6K visual tokens → 80GB GPU OOM

=== Token Compression-based (BLIP-2, Idefics2, MiniCPM) ===
输入: img_tokens → compressor (Q-Former/perceiver/pooling) → fixed-size tokens
处理: 压缩后 tokens 拼接进文本序列
问题: 信息压缩损失，固定 token 数不够灵活

=== Cross-Attention-based (Flamingo, mPLUG-Owl3) ===
输入: H_img ∈ R^{V×D} 作为外部 K/V 对
处理: LLM sparse 层 cross-attend 到视觉信息
H_img 不在 context window 中
优势: self-attention 复杂度 O(T²) 独立于 V
      mPLUG-Owl3 比 LLaVA 处理 ~6× 更多图像
```

mPLUG-Owl3 相对 Flamingo 的关键改进：
- 并行而非串行：cross-attention 与 self-attention 在同一 block 并行执行，共享 Q
- 稀疏而非稠密：仅 4/28 层含 cross-attention（Flamingo 每层）
- 复用 LLM 权重：W_img_KV 初始化自 LLM KV 权重，共享 LayerNorm
- 位置感知：MI-Rope 为图像赋予交织序列的位置信息

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
PyTorch 中通过修改 Attention 模块实现。每个 HATB 层额外维护 W_img_KV 和 W_gate 参数。视觉特征经 Linear Projection 对齐到 LLM 隐空间维度后，在 HATB 层作为 cross-attention 的 K/V 输入。训练需 staged training（先对齐再微调），视觉编码器和 LLM 主体通常冻结。mPLUG-Owl3 开源：https://github.com/X-PLUG/mPLUG-Owl。

涉及论文标题：
- mPLUG-Owl3__Towards_Long_Image-Sequence_Understanding_in_Multi-Modal_Large_Language_Models
- Flamingo: a Visual Language Model for Few-Shot Learning
