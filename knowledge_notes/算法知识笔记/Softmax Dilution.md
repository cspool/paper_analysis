## Softmax Dilution

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Softmax Dilution 是 Focus 论文附录 A 中解释"为何稀疏注意力能超越全注意力"的三个机制之一：在全注意力中，softmax 将概率质量分布在全部 n 个 token 上。当一个 token（如位置 800 的代词）需要关注其语义相关 token（如位置 200 的先行词）时，它必须与其余 n-2 个无关 token 竞争 softmax 概率质量。结果是对相关 token 的注意力权重被稀释，每个无关 token 都从真正重要的 pair 那里夺取一小部分注意力权重。Focus 通过限制 softmax 到同组 token + 局部窗口，将竞争集合从 n 缩小到约 n/K + w，概率质量集中在更小但更相关的候选集上，产生更锐利、更有信息量的注意力分布。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Softmax dilution 的数学本质：
```
全注意力:
  scores = [q_i·k_1, q_i·k_2, ..., q_i·k_n]    # n 个 score
  weights = softmax(scores)                       # n 个权重, sum=1
  # 当 n 很大时, 即使最强的 score 也只得到 1/n ~ 少量概率质量
  # 每个无关 token 贡献: weight_j * v_j (虽小但 n 个累积)

Focus:
  scores = [q_i·k_j for j where g(i)=g(j) or |i-j|≤w]  # ~n/K + w 个
  weights = softmax(scores)                              # ~n/K + w 个权重
  # 竞争集缩小 ~K 倍, 关键 token 获得更多概率质量
```

论文 Appendix A 给出的三个机制：
1. **Softmax dilution**（上述）：减少竞争 token 数量，集中概率
2. **Noise removal**：无关 KV pair 不仅浪费计算，还主动降质——每个无关 token 向输出添加微量噪声，12 层 × 12 heads 累积后显著。Focus 完全消除这些 pair
3. **Implicit structural constraint**：全注意力在小模型（124M）上会记忆训练数据中的虚假长程相关性。限制注意力到语义相关 group 充当结构性先验（类似 L1 正则化），防止过拟合噪声注意力 pattern

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Softmax dilution 的实践意义：
- 解释了为什么 top-k=2（约 50% 远距离 pair）产生比 top-k=3/4（更多 pair）更好的 PPL——更多 pair 重新引入 dilution
- 在小模型（124M）上效果最显著：Focus FT 30.3 vs full attention FT 31.4 (+1.1 PPL)
- 在大模型（7B+）上效果递减：更大模型天然有更强能力区分相关/无关 token，dilution 影响较小
- 设计启示：稀疏注意力的目标不是近似全注意力，而是通过消除稀释和噪声来超越它

涉及论文标题：
- Why Attend to Everything? Focus is the Key (Composing Sparse Attention via Learned Grouping)

涉及论文标题：
- TransPrune: Token Transition Pruning for Efficient Large Vision-Language Model
