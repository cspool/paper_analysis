## Reconstruction Correlation (r-bar / Attention Reconstruction Correlation)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Reconstruction Correlation r̄ 是 TriAttention 提出的评估指标，用于量化三角函数级数从 Q/K 中心预测实际 attention pattern 的准确度（Eq 5）：$\bar{r} = \frac{1}{N} \sum_{i=1}^N \rho(\mathbf{a}_i, \hat{\mathbf{s}})$，其中 a_i 是 query i 的实际 attention logits（在对数间隔的距离上采样），ŝ 是三角函数级数从 Q/K 中心预测的 attention logits，ρ 是 Pearson 相关系数。

物理意义：r̄ 量化"Q/K Concentration → 可预测距离偏好"这条因果链的强度。r̄ 高表示 Q/K 聚集确实导致了可被三角函数级数捕获的 attention 模式。跨模型分布：Qwen3-8B、Qwen2.5-7B、Llama-3-8B 的 r̄ 均右偏，均值 > 0.5，峰值在 0.6-0.9。

术语一般如何实现？如何使用？

实现：纯 Python + numpy，在大约 10K token 序列上计算一次。对数间隔采样 {1,2,4,8,...} 确保跨距离尺度平衡覆盖（避免相邻距离的样本非独立而高估相关性）。

使用场景：(1) 验证 Q/K Concentration 的因果效应；(2) 诊断哪些 head 适合用 S_trig——r̄ > 0.5 的 head（约 53.5% in Qwen3）三角函数级数预测有效；(3) 跨模型/跨架构比较——在 GQA 和 MLA 上 r̄ 分布相似，证明聚集现象是架构无关的通用规律。

涉及论文标题：
- TriAttention: Efficient Long Reasoning with Trigonometric KV Compression

---
