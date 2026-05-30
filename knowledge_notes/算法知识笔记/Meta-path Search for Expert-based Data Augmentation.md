## Meta-path Search for Expert-based Data Augmentation

术语解释
Meta-path over Experts 是 MELD 提出的基于 expert 序列的数据增强策略。给定 fixed experts E={e_1,...,e_n}，meta-path E_i={e_{j1},...,e_{jm}} 是一个有序 expert 序列，沿序列依次查询 experts 对训练数据进行增强。

术语是什么？
- Meta-path 将多个 DP task 的 experts 串联成流水线，利用前序 expert 输出为后续 expert 提供额外特征
- 搜索算法：贪心搜索，目标 argmax_{E_i} Eval(e_i, X_i^{E_i})，用户定义 sub-optimal paths 缩减搜索空间

从算法pipeline角度拆解术语。
以 EM task meta-path E_EM = {e_Blocking, e_DI, e_AVE, e_EM} 为例：
```
q = (t1="Apple iPhone 13", t2="iPhone 13 by Apple")
→ e_Blocking(q): 候选对筛选 → 过滤噪声
→ e_DI(t1): 填补缺失属性 → t1' (enriched)
→ e_AVE(t1'): 提取关键属性值 → 附加特征
→ e_EM(t1', t2): 最终match/mismatch
```
半结构化数据提升显著（Semi-Text-Watch F1: 55.07→70.78）。

术语一般如何实现？如何使用？
- 贪心搜索避免 expert 组合穷举
- 用户定义 sub-optimal paths 基于领域知识（如 EM 常用 {e_Blocking, e_EM}）
- 搜索完成后 meta-path 固定，训练/推理时直接使用

涉及论文标题：
- Efficient Mixture of Experts based on Large Language Models for Low-Resource Data Preprocessing
