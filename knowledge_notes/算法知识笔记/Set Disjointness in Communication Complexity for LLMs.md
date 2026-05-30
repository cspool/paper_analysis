## Set Disjointness in Communication Complexity for LLMs

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Set Disjointness (SD)是通信复杂度理论的经典问题：两方各持集合A和B，需通过最少通信判断A∩B是否为空，Ω(n)通信下界为领域基础结果(Chattopadhyay & Pitassi, 2010)。JRT论文将SD与语言模型的associative recall建立等价：循环模型可视为处理A和B集合的streaming algorithm，需在固定内存中存储足够信息以判断交集。Causal模型存储需求为Ω(min(|A|,|B|))——小集合在前内存需求最小，即"正确数据顺序"的理论依据。JRT-Prompt将下界降至Ω(min(|A|,|B|)/p)。纯卷积架构(BaseConv)无法从JRT获益(Theorem G.6/G.7/G.11)。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
// SD合成任务(Algorithm 1):
Input: [prefix_token], A, [sep_token], B, [answer_token], [t]
  A/B: random tokens from disjoint halves of vocab |V|=2048
  t: intersecting token from A also inserted into B
Output: predict t

// 理论(Theorem 3.2):
// Based(BaseConv+MLP+LA+MLP) in JRT-prompt以O(min{|A|,|B|}·n)空间解SD
// n=token bit width, |A|,|B|∈{1..1024}
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
JRT论文构建合成SD任务训练4层Based模型，sweep model dim∈{36..128}和feature dim∈{4..24}控制state size，评估不同数据顺序下准确率。代码基于Zoology合成仓库(https://github.com/HazyResearch/zoology)。训练数据：20000×12种(|A|,|B|)组合(mixture)。评估时需长度外推至训练未见的序列长度组合。该框架适用于分析新循环架构的memory-recall tradeoff和设计prompting策略。

涉及论文标题：
- Just_read_twice__closing_the_recall_gap_for_recurrent_language_models

---
