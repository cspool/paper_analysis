## Prompt Compression（提示压缩）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Prompt Compression 是在 LLM 推理前减少输入 prompt 长度以降低计算量的技术。由于 attention 的计算复杂度为 $O(T^2 d)$，prompt 长度 $T$ 减少 $k$ 倍可带来约 $k^2$ 倍的 attention 计算节省。核心方法包括：(1) Token Pruning——在推理过程中逐步移除不重要的 token（如 PoWER-BERT、DynamicViT）；(2) Prompt Summarization——使用小模型或基于熵的方法压缩 prompt 文本（如 LLMLingua）；(3) Soft Prompt Compression——训练 autoencoder 将长 prompt 压缩为少量 summary vectors（如 AutoCompressors、ICAE）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

LLMLingua 的 token 级迭代压缩：

```
# LLMLingua: Coarse-to-Fine Prompt Compression
def compress_prompt(prompt, target_ratio, budget_controller):
    segments = split_into_segments(prompt)
    
    # Stage 1: Coarse-grained (segment level)
    segment_importance = []
    for seg in segments:
        entropy = compute_entropy(LLM(seg))  # per-token entropy from LLM
        segment_importance.append(mean(entropy))
    
    # Budget controller allocates compression budget per segment
    budgets = budget_controller(segment_importance, target_ratio)
    
    # Stage 2: Fine-grained (token level, iterative)
    for seg, budget in zip(segments, budgets):
        tokens = tokenize(seg)
        while len(tokens) / len(original_tokens) > budget:
            # Remove token with lowest conditional perplexity increase
            scores = [perplexity_increase(seg, i) for i in range(len(tokens))]
            tokens.pop(argmin(scores))
    
    # Stage 3: Distribution alignment (instruction tuning)
    compressed = detokenize(concatenate(all_tokens))
    return compressed
```

LLMLingua 可实现 20× 压缩比。LLMLingua-2（ACL 2024）将压缩重新定义为 token 分类问题，使用双向 encoder 替代单向 LLM，3-6× 更快且 1.6-2.9× 端到端延迟改善。LongLLMLingua 针对长上下文场景增加了位置偏差处理。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

压缩可发生在预处理或运行时：LLMLingua 系列在推理前压缩 prompt；Token pruning（如 Deja Vu、PuMer）在推理过程中动态移除 token。AutoCompressors 和 ICAE 训练专门的压缩模型将长上下文映射为少量 soft prompt token。对于 ViT，DynamicViT 和 A-ViT 根据输入复杂度自适应选择保留的 patch token 数量。使用场景：RAG 检索长文档、多轮对话压缩历史、代码补全长上下文。

涉及论文标题：
- A Survey of Resource-efficient LLM and Multimodal Foundation Models
