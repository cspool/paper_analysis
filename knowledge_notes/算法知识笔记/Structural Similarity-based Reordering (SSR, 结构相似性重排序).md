## Structural Similarity-based Reordering (SSR, 结构相似性重排序)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Structural Similarity-based Reordering（SSR，结构相似性重排序）是 PT²-LLM 提出的列重排序策略，用于替换 GPTQ 固定顺序或 Hessian 重要性重排序。SSR 的动机：三值化（仅有 3 个量化级别）对块内权重分布极为敏感——离群列和散乱的列间分布会严重扭曲三值网格，使大量权重被错误映射。SSR 利用列间余弦相似度衡量结构相关性，每次选块时从残差矩阵中选取与均值参考向量最相似的 top-k 列，使块内列结构对齐、数值接近，形成更紧凑的分布。排列通过置换矩阵 P 实现（W'=WP, X'=XP），保证输出不变，推理时零额外开销。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# SSR 伪代码 (集成在 GPTQ 逐块量化框架中)
# 输入: W ∈ R^{n×m}, block_size k=128
col_remaining = list(range(m))
while len(col_remaining) >= k:
    W_rem = W[:, col_remaining]             # 残差子矩阵
    w_bar = mean(W_rem, axis=1)             # 列均值参考向量 (n,)
    # 计算剩余每列与 w_bar 的余弦相似度
    sim = [(j, dot(W[:,j], w_bar)/(||W[:,j]||*||w_bar||)) for j in col_remaining]
    block_cols = [j for j,_ in sorted(sim, key=lambda x:-x[1])[:k]]
    ATQ_quantize(W[:, block_cols])           # ITF + AGA 三值化
    error_compensate(W[:, col_remaining])    # GPTQ Hessian 补偿
    col_remaining -= block_cols
```
效果：LLaMA-2-7B 上 SSR vs 无重排 PPL 从 13.06→11.56，优于 Hessian 重排（12.35）和随机重排（12.84）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
SSR 实现要点：(1) 列置换是纯索引重排，推理零开销；(2) 余弦相似度计算 O(n·m_remaining) 每步，总开销远低于 ATQ 量化本身；(3) w_bar 作为"代表性列"代理——选与 w_bar 最相似的列等价于选与当前残差方向最一致的列群；(4) 块内方差可视化证实 SSR 使权重分布更紧凑（论文图 3 右侧）。

涉及论文标题：
- PT²-LLM Post-Training Ternarization for Large Language Models

---
