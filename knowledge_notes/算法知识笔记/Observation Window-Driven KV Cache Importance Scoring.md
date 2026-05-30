## Observation Window-Driven KV Cache Importance Scoring

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Observation Window-Driven KV Cache Importance Scoring 是 WindowKV 提出的 token 重要性评估机制：以输入末尾 α 个 token（紧邻生成位置）作为 observation window，计算其对 review context 中各 token 的累积注意力 $t_j = \sum_{i \in [n-\alpha, n]} \mathbf{A}_{ij}$，用于后续窗口级重要性评分。

优势对比：(1) vs H2O 的全 query 平均注意力——不易被 attention outliers 主导；(2) vs PyramidKV 仅用 instruction tokens——observation window 紧邻生成位置，天然携带当前生成阶段最相关的上下文需求。

模型相关配置：Qwen2.5-1.5B α=4(loc)/16(agg)；LLaMA3-8B α=16(loc)/32(agg)。选择逻辑：aggregation 任务需要更大的 observation window 来识别各窗口中的关键 token，localization 任务只需足够定位相关窗口即可。

术语一般如何实现？如何使用？

取 attention 矩阵 A[n-α:n, :n-α] 子矩阵，沿 query dim sum 得 score vector t ∈ R^{n-α}。仅 group-first layer 执行，其余层共享 indices。与 FlashAttention 兼容（full attention 仅首层）。

涉及论文标题：
- WindowKV: Task-Adaptive Group-Wise KV Cache Window Selection for Efficient LLM Inference

---
