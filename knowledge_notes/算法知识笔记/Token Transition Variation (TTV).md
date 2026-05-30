## Token Transition Variation (TTV)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

TTV（Token Transition Variation）是 TransPrune 提出的用于评估 LVLM 中视觉 token 重要性的训练无关准则。TTV 的核心思想是：token 在 Transformer 模块中传播时的表征变化（transition）幅度和方向能够反映该 token 的语义重要性，而无需依赖 token 间的 attention 计算。具体而言，对每个 Transformer 子模块（self-attention 或 FFN），TTV 同时测量两个维度：(1) **幅度变化** m = ||T_out||₂ / ||T_in||₂，衡量 token 表征在模块传递后的 L2 norm 变化率；(2) **方向变化** d = cos(T_out, T_in) = (T_out · T_in) / (||T_out||₂ · ||T_in||₂)，衡量表征向量方向的旋转程度。最终的 TTV 计算为：TTV = Softmax(1 - |d|) · m，其中 Softmax 在所有 token 上归一化方向变化值，乘以幅度变化作为权重。每层 l 的 TTV = TTV(Attention) + TTV(FFN)。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**TTV 的核心计算流程**（TransPrune 在每层 Transformer 中）：

```
# 输入: visual tokens T_I [N, d], 当前层索引 l
# 当前层 self-attention 模块
T_attn_out = SelfAttention(T_I)  # 标准 QKV attention + projection
# 计算 self-attention 的 TTV
m_attn = ||T_attn_out||_2 / ||T_I||_2        # [N], 幅度变化率
d_attn = (T_attn_out · T_I) / (||T_attn_out||_2 · ||T_I||_2)  # [N], 余弦相似度
TTV_attn = Softmax(1 - |d_attn|) * m_attn    # [N], Equation (2)

# 当前层 FFN 模块
T_ffn_out = FFN(T_attn_out)  # SwiGLU 或其他 FFN
m_ffn = ||T_ffn_out||_2 / ||T_attn_out||_2
d_ffn = cos(T_ffn_out, T_attn_out)
TTV_ffn = Softmax(1 - |d_ffn|) * m_ffn

# 当前层 TTV = attention + FFN 贡献之和
TTV[l] = TTV_attn + TTV_ffn                     # Equation (3)

# Accumulation: 从 accumulation start 到当前层累积 TTV
if l in pruning_layers:
    TTV_acc = sum(TTV[j] for j in range(acc_start, l+1))  # Equation (4)
```

关键设计动机：(1) 使用 1-|d| 而非 d —— 实验中 1-|d| 效果更好（见论文 supplementary）；(2) Softmax 归一化使方向变化在不同 token 间可比；(3) 乘以 m 赋予幅度变化更大的 token 更高权重（幅度变化大的 token 通常语义更丰富）；(4) TTV 仅依赖 token 自身的输入→输出变化，不计算 inter-token 依赖，天然避免 attention 三角 mask 的 positional bias。

术语一般如何实现？如何使用？

TTV 在 TransPrune 中通过 hook Transformer 子模块的输入/输出 tensor 实现，无需修改模型结构或进行训练。具体实现要点：(1) TTV 在 accumulation layers（TransPrune 默认 layers 7-12）计算——论文实验表明中间层（而非浅层 1-6 或深层 13+）的 token transition 最能反映语义重要性（Table 10，中间层 MME^P=1540 vs 浅层 MME^P=1515）；(2) TTV accumulation 跨层累积避免单层噪声——消融实验（Table 11）显示引入 accumulation 后 MME^P 从 1530 提升到 1540；(3) TTV 额外计算开销 O(sd) 与 stage 数 s 和维度 d 线性相关，在总计算中占比可忽略；(4) 与 FlashAttention 完全兼容——TTV 仅需模块输入/输出 tensor，不访问内部 attention matrix；(5) TTV 的幅度和方向组件均带来增益，magnitude 贡献更大（Table 12：IGA+Magnitude MME^P=1532 vs IGA+Direction MME^P=1521 vs IGA+TTV MME^P=1540）。代码将开源于 https://github.com/liaolea/TransPrune。

涉及论文标题：
- TransPrune: Token Transition Pruning for Efficient Large Vision-Language Model

---
