## Key Centering (for LSH-based Attention)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Key Centering是MagicPIG中在LSH哈希表构建前对key向量进行的预处理操作：对每个attention head的K cache进行中心化，即k̄_i = k_i - (1/n)Σ_{j=1}^n k_j。由于Softmax对输入同时加常数具有平移不变性（Softmax(q·(K + c)/√d) = Softmax(qK^T/√d + constant) = Softmax(qK^T/√d)），centering不改变attention计算的数学结果。该操作的必要性源于LLM中key向量的几何特性——key平均方向k_avg与attention sink的key方向k_sink几乎相反（余弦相似度-0.9~-0.8），且k_sink的朝向在不同输入下几乎不变（相似度>0.99）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
// Key Centering预处理
// K = [k_1, k_2, ..., k_n] ∈ R^{n×d}
k_mean = (1/n) * Σ_{i=1}^n k_i
K_centered = K - k_mean  // 每行减去均值

// 不centering的后果（Figure 9a）：
//   - q和k方向几乎相反 → 随机投影无法区分key
//   - <0.1%的key能被query采样
//   - 检索任务准确率降至接近0
//   - 聚合任务准确率降至65%

// centering后的效果：
//   - key分布在query周围，随机投影能有效区分
//   - 准确率恢复到接近全注意力水平
```

术语一般如何实现？如何使用？

Centering是MagicPIG成功的关键ablated组件。论文Ablation（Section 5.3, Figure 9a）验证：不centering时准确率在检索(NIAH)中降至接近0，FWE降至65%。Centering后key向量不再集中在query的相反方向，SimHash投影能有效区分不同key的相似度。该操作在所有attention head上独立执行，计算量为O(nd)，在KV cache构建时一次性完成，不影响解码效率。

涉及论文标题：
- MagicPIG: LSH Sampling for Efficient LLM Generation
