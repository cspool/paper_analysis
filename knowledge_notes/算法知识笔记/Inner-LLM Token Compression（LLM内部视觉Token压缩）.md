## Inner-LLM Token Compression（LLM内部视觉Token压缩）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Inner-LLM Token Compression 是指在 LVLM 的 LLM decoder 内部（forward propagation 过程中）对视觉 token 进行压缩的方法类别，与 Pre-LLM Compression（在进入 LLM 之前压缩）相对。操作时机：visual token 通过 ViT 编码和 Projector 映射后进入 LLM，在 LLM 的若干选定的 transformer layer 之间进行 token 丢弃或合并。代表性方法包括：FastV（early layer 基于 cross-attention 剪枝，ECCV 2024）、SparseVLM（cross-modal attention ranking + 自适应稀疏比 + token recycling，ICML 2025）、PDrop/PyramidDrop（渐进式金字塔型剪枝，CVPR 2025）、V2Drop（variation-aware 剪枝，CVPR 2026）。

Inner-LLM 方法的优势：与模型架构无关（architecture-agnostic），plug-and-play 可插拔，无需修改 ViT encoder 或 Projector，训练无关（training-free）。主要挑战：(1) 依赖 attention weights 的评分方法不兼容 FlashAttention；(2) attention 的位置偏见（positional bias）导致丢弃语义重要 token。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Inner-LLM token compression 的通用 pipeline：

```
Input: Vision tokens F_v ∈ R^{M×D}, Text tokens F_t, LLM layers L
Output: Generated response Y

1. F_combined = concat([F_v, F_t])  # interleaved format
2. for l = 1 to L:
3.     h = TransformerLayer_l(h)
4.     if l in pruning_layers:
5.         # Pruning strategy varies by method:
6.         # - FastV: attn_score = mean(Attn_weights[text_tokens → vision_tokens])
7.         # - SparseVLM: attn_score = cross_modal_attention_ranking()
8.         # - V2Drop: var_score = ||f_i^(l) - f_i^(l-1)||_2  # attention-free
9.         vision_tokens = select_top_k(vision_tokens, score, K_l)
10.        h = concat([vision_tokens, text_tokens])
11. return auto_regressive_decode(h)
```

典型压缩率：Inner-LLM 方法可在 LLM 浅层剪枝 50-77.8% visual token。V2Drop 在 LLaVA-1.5 上：577→288 (layer 3)→173 (layer 17)→128 (layer 22)，即 77.8% total reduction。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现方式：(a) 通过 PyTorch forward hook 在指定 LLM layer 的 attention 或 MLP 后拦截 hidden states；(b) 计算每个 visual token 的重要性分数（attention score / variation score / duplication score）；(c) 根据分数排序选择 top-K 保留；(d) 重组 sequence（丢弃的 visual token 不传入后续 layer）。代码通常集成在 HuggingFace Transformers 的 model forward 中，如 V2Drop 核心实现在 `llava/model/language_model/V2Drop.py`。使用：加载 LVLM checkpoint → 注册 pruning hooks → 正常推理。各方法开源情况：FastV (https://github.com/pkunlp-icler/FastV)，SparseVLM（未完全开源），PDrop (https://github.com/XingLuan/PyramidDrop)，V2Drop (https://github.com/xuyang-liu16/V2Drop)。

涉及论文标题：
- V2Drop__Variation-aware_Vision_Token_Dropping_for_Faster_Large_Vision-Language_Models
