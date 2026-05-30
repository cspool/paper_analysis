## Token Shift / Shifted-Label Strategy

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Token Shift（或称Shifted-Label Strategy）是Fast-dLLM v2用于在block diffusion训练中保留预训练AR模型representation quality的技术。在标准masked diffusion中，每个masked位置i使用自身的hidden state h_i来预测token x^i。而Token Shift改用前一个位置i-1的hidden state h_{i-1}来预测token x^i：logit用于预测x^i的位置是i-1而非i。这使得预测的计算路径与AR模型的next-token prediction保持一致（AR模型中position i-1预测position i），让dLLM在支持block内bidirectional attention的同时维持AR-like的temporal representation。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# 标准masked diffusion预测（无token shift）:
for each masked position i:
    h_i = transformer_output[i]          # 使用位置i的hidden state
    logit_i = lm_head(h_i)               # 预测位置i的token
    
# Token Shift预测（Fast-dLLM v2）:
for each masked position i:
    h_{i-1} = transformer_output[i-1]    # 使用位置i-1的hidden state
    logit_i = lm_head(h_{i-1})           # 预测位置i的token（shifted）
    
# 效果：position i-1的hidden state负责预测position i
# 与AR next-token prediction的形式一致: p(x_i | h_{i-1})
```

Token Shift与complementary masking协同工作：masked位置i使用i-1的hidden state（i-1在complementary view中可能是可见的），使得模型能利用完整的prefix context进行预测。消融实验（Table 2）中"naive token shift"即为仅使用token shift但无complementary masking和padding的baseline（avg=41.3）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现方式：在计算loss时对logits做offset索引——对每个masked位置i，取logits[i-1]而非logits[i]来计算cross-entropy。这与Dream的预训练方法（Ye et al., 2025b）一致。适用范围：从AR模型（使用next-token prediction训练）微调为diffusion模型时，token shift是保持AR预训练质量的关键技术。论文未明确说明此技术的原创来源，Dream论文中已有类似设计。

涉及论文标题：
- Fast-dLLM v2: Efficient Block-Diffusion LLM
