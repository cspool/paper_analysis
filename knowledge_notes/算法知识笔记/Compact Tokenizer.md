## Compact Tokenizer

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Compact Tokenizer（紧凑分词器）是一种针对小语言模型（Tiny Language Model）优化的词表压缩技术。其核心思想是：大语言模型（如7B+参数）使用的分词器通常包含100k+词汇量（vocabulary size），以保证对多种语言和领域的高覆盖率。但对小模型（≤1.5B参数），embedding层和输出head层的参数量在总参数量中占比极大——例如对于12层/2048宽的1B模型，100k词表的embedding+head层占比高达38.19%，远超大模型中约10%的比例。Compact Tokenizer通过统计分析发现训练语料中词频呈现长尾分布（top-48k词汇覆盖97.86%的1.6T tokens语料），识别并移除低频冗余词汇（bottom 52k+词汇仅覆盖不到3%的语料），将词表从100k压缩至48k，使embedding+head参数占比降至18.07%，释放约20%参数给Transformer本体。

从算法pipeline角度拆解术语，给出术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

在算法Pipeline中，Compact Tokenizer的训练和使用流程如下：

```
输入: 原始大词表 V_large (|V|=100883), 大语料 D (1.6T tokens)

# Phase 1: 词频分析与词表压缩
freq = Counter()  # 统计每个vocab token的出现频率
for sample in D:
    for token_id in tokenize(sample, V_large):
        freq[token_id] += 1

# 按频率降序排序，计算累积覆盖率
sorted_vocab = sort_by_freq_desc(V_large, freq)
cum_freq = 0; total = sum(freq.values())
for k, v in enumerate(sorted_vocab):
    cum_freq += freq[v]
    coverage = cum_freq / total
    if coverage >= 0.9786:  # 目标覆盖率97.86%
        V_compact_size = k + 1  # 得到 k=48k
        break

# Phase 2: 用SentencePiece BPE在新词表大小下重新训练tokenizer
V_compact = train_sentencepiece_bpe(
    corpus=D,
    vocab_size=48000,
    character_coverage=0.9995
)
# 输出: 新的BPE编码规则 → token_id映射

# Phase 3: 小模型使用Compact Tokenizer
# Embedding层: W_emb ∈ R^{48000 × d_model}  (原来: R^{100883 × d_model})
# LM Head层:  W_head ∈ R^{d_model × 48000}  (原来: R^{d_model × 100883})
# 参数占比从 38.19% 降至 18.07%
```

关键技术公式：
- PEHL (Proportion of Embedding and Head Layers) = `(2 × V × d_model) / total_params`
- 推荐 PEHL < 20%

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现方面：
1. **初始词表来源**：从已有大模型（如PanGu-π-7B）继承tokenizer作为初始词表。
2. **语料覆盖分析**：在训练语料（1.6T tokens）上进行频率统计，计算top-k词汇的累积覆盖率，确定最优k值（本论文k=48k, 覆盖率97.86%）。
3. **Tokenizer实现**：使用SentencePiece库的BPE算法，设置vocab_size=48000重新训练。
4. **超参数推荐**：词表大小应保证累积覆盖率>90%，同时PEHL<20%。过小词表（如8k，覆盖率<70%）导致性能下降。
5. **相关方法对比**：MiniMind使用6400词表的Custom BPE；AG-BPE通过注意力引导打分机制实现16k词表、3.77×压缩比；Compact框架通过后剪枝（post-hoc pruning）同时剪除低频词汇和FFN中间通道。

使用场景：
- 构建1B及以下参数量级的SLM/TLM模型
- 边缘设备部署场景，对参数量和显存严格受限
- 双语/多语言小模型的tokenizer设计
- 从大模型蒸馏到小模型时作为首步优化

涉及论文标题：
- PanGu-π Pro: Rethinking Optimization and Architecture for Tiny Language Models
