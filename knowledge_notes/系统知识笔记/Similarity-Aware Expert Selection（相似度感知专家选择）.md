## Similarity-Aware Expert Selection（相似度感知专家选择）

术语是什么？
Similarity-Aware Expert Selection是FineMoE的核心决策机制：根据检索到的历史expert map与当前context的similarity score动态决定预取多少expert。公式为delta=clip(1-score, 0, 1)，从高概率到低概率选择expert直到累积概率超过delta，且至少选择top-K所需数量。低相似度时delta高→多预取降低miss risk；高相似度时delta低→少预取节省GPU cache。

从系统架构角度拆解术语：
```
# 给定检索到的历史expert map P[l][e]（第l层专家e的概率分布），similarity score s：
delta = clip(1 - s, 0, 1)
sorted_experts = sort_by_probability_desc(P[l])
selected = []
cum_prob = 0
for expert in sorted_experts:
    selected.append(expert)
    cum_prob += P[l][expert]
    if cum_prob >= delta and len(selected) >= K:
        break
# selected为需要预取的expert集合
```
动态阈值机制在hit rate和GPU memory之间实现fine-tune：s低（不相似context）→delta接近1→几乎预取所有expert→确保low miss但占用更多显存；s高（高度相似context）→delta接近0→只预取top-K所需→节省显存但依赖高命中率。

术语一般如何实现？
在FineMoE中以PyTorch ops实现，与Expert Map Searcher集成。每次inference iteration的目标layer都执行动态选择，确保预取策略随context变化自适应调整。对比baseline的固定stride或LRU策略，similarity-aware方式使cache容量利用更高效。

涉及论文标题：
- Taming Latency-Memory Trade-Off in MoE-Based LLM Serving via Fine-Grained Expert Offloading
