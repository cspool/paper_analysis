## Compensated Attention Scoring (补偿注意力评分)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Compensated Attention Scoring 是 ZSMerge 中用于修正 token 合并后 KV Cache 表示偏差的注意力评分机制。核心公式（Eq. 8）：

$$\hat{a}_t^{(T)} = \frac{\exp\left(\mathbf{q}_T^\top \mathbf{k}_t / \sqrt{d} + \alpha \log w_t\right)}{\sum_{i=1}^T \exp\left(\mathbf{q}_T^\top \mathbf{k}_i / \sqrt{d} + \alpha \log w_i\right)}$$

其中 $w_t$ 为 token t 的融合计数（未压缩 token w_t=1，残差 slot w_t = 该 slot 合并的 token 数），$\alpha \in [0,1]$ 为 scale factor。

补偿机制解决两个关键问题：
1. **表示偏差修正**：合并 token 的 key 为多个原始 token 的均值，与原 value 分布不匹配——log w_t 偏置项修正此偏差。
2. **attention mass 守恒**：保证压缩 token 的 attention 占比不"过度膨胀"——Theorem 1 证明：$\forall$ 未压缩 token i，$\hat{a}_i^{(T)} \geq a_i^{(T)}$（原 attention 分数），即未压缩 token 在压缩后仍保持相对优势。

定理 1 的证明依赖于 Jensen 不等式（指数函数的凸性）：残差 slot r 的 attention numerator 上界为 sum of individual token attention numerators，因此分母受压缩影响有限，未压缩 token 的相对分数在压缩后不降低。

$\alpha=0$ 退化为纯驱逐式注意力（无补偿），$\alpha=1$ 为完全补偿。ZSMerge 实验固定 $\alpha=1$，消融实验显示 $\alpha$ 从 0→1 的 ROUGE 提升 1-5%。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
# 标准 attention（无补偿）
scores = Q @ K_B.T / sqrt(d)       # [1, B]
attn = softmax(scores)
output = attn @ V_B

# 补偿 attention（ZSMerge Eq. 8）
w_all = [1, 1, ..., w[0], w[1], ...]  # 未压缩=1, 残差 slot = 合并数
log_bias = alpha * log(w_all)          # 对数偏置项
scores_compensated = Q @ K_B.T / sqrt(d) + log_bias  # [1, B]
attn_compensated = softmax(scores_compensated)
output = attn_compensated @ V_B
```

关键性质：log w 偏置使得合并 slot 的 attention 分数增加（因为 log w > 0），但 Theorem 1 保证此增加不会压倒未压缩 token——Jensen 不等式约束 compressed token 的 attention numerator ≤ sum of individual numerators。

术语一般如何实现？如何使用？

实现为对标准 softmax 的一行修改——在 logit 中加 α·log w_all 后调 softmax。O(B) 额外开销。无需训练或校准数据，α=1.0 为推荐默认值。与任何 attention kernel（SDPA、FlashAttention）兼容——只需改变 softmax 输入。

涉及论文标题：
- ZSMerge: Zero-Shot KV Cache Compression for Memory-Efficient Long-Context LLMs
