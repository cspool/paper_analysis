## TopK Attention

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

TopK attention是一种稀疏注意力近似方法：仅选择attention scores最高的K个key-value对参与加权平均计算，丢弃其余token的贡献。数学上，设w_{r_1} > ... > w_{r_K} > ... > w_{r_n}为排序后的attention scores，则TopK attention的计算为o^{TopK} = Σ_{i=1}^K w_{r_i} v_{r_i} / Σ_{i=1}^K w_{r_i}。Quest、Loki等方法是TopK attention的搜索近似（用近似搜索替代精确TopK排序以降低检索开销）。TopK attention是有偏估计——丢弃低score tokens的系统性偏差无法通过增加K来消除（除非K=n）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**TopK attention的缺陷分析**（MagicPIG Section 3）：

1. **长尾分布问题**：在许多层，Top20% tokens仅覆盖70~80% attention scores（Figure 2a），丢弃的30~20% scores导致不可忽略的估计误差（15-20%，Figure 4）。
2. **Attention Sink误导稀疏性**：首token（attention sink）吸收了大部分attention mass，使分布看起来稀疏，但剩余token间分布更均匀（Figure 2b）。
3. **搜索开销大**：IVF等搜索方法需要访问>30%的key states才能获得精确TopK（Liu et al., 2024a）。

**下游任务退化**：在聚合任务（Common/ Frequent Word Extraction）中，即使exact TopK也严重退化（Figure 1, Figure 9b-c）。检索任务（Needle-in-a-Haystack）中TopK表现可接受，因为所需信息集中在少数token上。

术语一般如何实现？如何使用？

Quest (Tang et al., 2024) 是TopK搜索的代表实现：将KV cache按page_size分页，计算q与每页summary的内积近似估计该页的重要性，TopK页被选中参与attention。page_size=16时检索开销Cost_1=1/16=6.25%。但Quest在lm-eval-harness中期上下文任务（GSM8K, COQA, MMLU）上准确率显著低于MagicPIG（Table 1）。

涉及论文标题：
- MagicPIG: LSH Sampling for Efficient LLM Generation

---
