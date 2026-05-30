## Complementary Masking (互补掩码)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Complementary Masking（互补掩码）是Fast-dLLM v2提出的block diffusion训练策略。对于每个训练样本，采样一个随机binary mask m ∈ {0,1}^D（D为block size），其中m_i=1表示位置i被替换为[MASK] token。同时生成互补mask m̄ = 1 - m，两个view（m和m̄）放入同一个batch中训练。这确保每个token既在masked上下文（被mask时从其他可见token预测自己）又在unmasked上下文（可见时帮助预测其他masked token）中被训练。由于互补性，m和m̄的masked token集合完全不重叠，两个view的loss覆盖了序列中所有L个位置——使得无需在loss中除以mask比例（无需归一化系数1/t），总监督信号量恒定为L。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Block size D=32, training sample x_0 of length L
# 每个block独立采样mask

m = random_binary_mask(shape=[D])      # 每位置以p=0.5概率为1（mask）
m_complement = 1 - m                   # 互补mask

# View 1: 对x_0应用m → masked位置替换为[MASK]
x_t^1 = x_0.copy()
x_t^1[m == 1] = [MASK]

# View 2: 对x_0应用m_complement → 互补的masked位置替换为[MASK]
x_t^2 = x_0.copy()
x_t^2[m_complement == 1] = [MASK]

# 两个view放入同一batch，model同时处理
# Noised x_t和clean x_0沿sequence维度拼接（总长2L）
# 使用block-wise attention mask [[M_BD, M_OBC], [0, M_BC]]

# Loss（无需1/t归一化，因两个view覆盖所有L个位置）:
L = -Σ 1[x_t^1_i=[MASK]]·log p(x_0^1_i|x_t^1)  # view 1的masked位置
  + -Σ 1[x_t^2_i=[MASK]]·log p(x_0^2_i|x_t^2)  # view 2的masked位置
# 每个样本总贡献L个token的loss（完整序列监督）
```

消融实验（Table 2）证明：+pad+CM（complementary masking）比naive token shift提升+3.7 avg accuracy，是训练配方中最关键的组件。互补掩码还与token shift协同：masked位置使用i-1的hidden state预测token i，保留AR模型的representation quality。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现要点：(1) 训练时将两个view构造为同一batch中的两个sample；(2) 使用flex-attention实现自定义block-wise attention mask同时处理noised和clean序列；(3) 适用于从预训练AR模型微调为block diffusion模型的场景。与标准masked language modeling（MLM）的区别：MLM通常mask 15% token，每个样本仅mask一次；CM mask ~50% token两次（互补），监督信号更密集。

涉及论文标题：
- Fast-dLLM v2: Efficient Block-Diffusion LLM
