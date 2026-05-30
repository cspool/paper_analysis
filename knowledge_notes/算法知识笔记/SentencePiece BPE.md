## SentencePiece BPE

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

SentencePiece BPE（Byte-Pair Encoding with SentencePiece）是一种子词级别的文本分词方法。SentencePiece是Google开发的**纯数据驱动**、**语言无关**的分词器框架，将输入文本直接视为Unicode字符序列（无需预分词），通过BPE算法从语料中学习最优的子词合并规则。BPE（Byte-Pair Encoding）的核心思想是：从字符级开始，反复统计相邻token pair的共现频率，贪心地合并最高频pair为新的token，直到词表达到预设大小。这使模型能够处理任意未见过的词（通过子词组合），平衡语义粒度和词表大小。

从算法pipeline角度拆解术语，给出术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# BPE学习过程 (SentencePiece内部)
输入: 训练语料 D (raw text), 目标词表大小 vocab_size
输出: 子词合并规则 + token→id映射

# 1. 初始化: 每个Unicode字符 + 空格为基本unit
vocab = {char: count for char in all_chars_in_D}  # 字符级统计
# SentencePiece特性: 空格用▁(U+2581)表示，保持可逆

# 2. BPE迭代合并
while len(vocab) < vocab_size:
    # 统计所有相邻token pair的频率
    pair_freq = defaultdict(int)
    for word_freq, word_tokens in D_tokenized:
        for i in range(len(word_tokens) - 1):
            pair_freq[(word_tokens[i], word_tokens[i+1])] += word_freq
    
    if not pair_freq:
        break  # 无可合并pair
    
    # 贪心选择最高频pair
    best_pair = max(pair_freq, key=pair_freq.get)
    new_token = best_pair[0] + best_pair[1]  # 合并为新的subword
    
    # 更新: 添加新token，在原序列中替换出现位置
    vocab[new_token] = pair_freq[best_pair]
    D_tokenized = merge_pair(D_tokenized, best_pair, new_token)

# 输出: (token, token_id)的映射表
# 例: "自然语言处理" → ["▁自然", "语言", "处理"] → [1234, 567, 890]

# 分词推理 (inference)
def encode(text):
    # 按最长匹配原则应用BPE合并规则
    chars = list(text)
    tokens = apply_bpe_rules(chars, merge_rules)  # 按学习顺序应用规则
    return [vocab_to_id[t] for t in tokens]
```

关键公式：
- BPE合并准则: max_{pair} freq(pair) = freq(token_a, token_b) over all token sequences in corpus
- SentencePiece Loss: 基于unigram language model，优化p(x) = Π_i p(x_i) under subword segmentation

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现和使用：
1. **SentencePiece安装与基本用法**：
```python
import sentencepiece as spm
# 训练
spm.SentencePieceTrainer.train(
    input='corpus.txt', model_prefix='m',
    vocab_size=48000, character_coverage=0.9995,
    model_type='bpe', num_threads=16
)
# 使用
sp = spm.SentencePieceProcessor(model_file='m.model')
tokens = sp.encode("Hello world!", out_type=str)  # ['▁Hello', '▁world', '!']
ids = sp.encode("Hello world!")  # [123, 456, 78]
```

2. **SentencePiece vs 传统BPE**：SentencePiece将输入视为原始字符序列（无预分词步骤），消除语言特定的tokenization假设（如英语依赖空格分词），真正语言无关；使用▁元字符标记词边界，保证tokenization可逆（detokenization无歧义）。

3. **Typical训练参数**：
   - `vocab_size`：词表大小，大模型通常100k+，小模型推荐32k-48k
   - `character_coverage`：字符覆盖率，推荐0.9995（覆盖99.95%的Unicode字符）
   - `model_type`：bpe（最常用）或unigram
   - `num_threads`：并行训练线程数

4. **本论文中的使用**：基于PanGu-π-7B的100k BPE tokenizer，在1.6T tokens语料上进行频率分析，然后使用SentencePiece BPE算法训练48k紧凑tokenizer。

涉及论文标题：
- PanGu-π Pro: Rethinking Optimization and Architecture for Tiny Language Models
