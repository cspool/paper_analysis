## Associative Recall in Language Models

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Associative Recall (AR)是语言模型在上下文中识别和使用之前出现过的key-value对的能力。Arora et al.(2023, Zoology)将AR定义为与ICL质量高度相关的核心技能：若某token完成的bigram在上下文中之前出现过，且该bigram在训练中罕见(未被memorize)，则模型须依靠上下文而非参数知识预测。AR形式化为MQAR任务——上下文中多个(key,value)对，给定key预测value。Transformer通过attention的O(N²)全局匹配天然擅长AR；固定memory循环模型需在有限state中选择性存储，成为recall-intensive ICL的主要瓶颈。JRT论文通过Pile perplexity slicing (AR slice vs Other slice)量化各架构的AR能力。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
// AR基本形式:
Context: "In 1957, Dr. Seuss wrote ... In 1982, Dr. ___"
// "Dr. Seuss" 是AR bigram——预测"Seuss"需回忆前面出现的"Dr."

// Pile AR slice定义(JRT Section 5.2):
// "AR hit": token完成一个训练中低频(<1000次)的re-occurring bigram
// 按bigram frequency和distance分组分析perplexity
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
JRT论文使用10M Pile训练文档计数bigram频率，在3200 seq/2048 token的Pile test上计算最后1024 token/seq的AR/Other slice perplexity。JRT-RNN在AR slice显著优于causal decoder-only baseline，在Other(non-recall) slice略差(因MLM训练仅见65% NTP tokens)。MQAR评测代码集成在LM-Eval Harness和RULER benchmark中。

涉及论文标题：
- Just_read_twice__closing_the_recall_gap_for_recurrent_language_models

---
