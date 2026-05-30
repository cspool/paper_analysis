## Structured Query Refinement（结构化查询精炼 / Structured Reasoning）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Structured Query Refinement（结构化查询精炼），在 Vgent 中也称为 Structured Reasoning，是一种后检索（post-retrieval）验证机制。其核心思想是：检索到的 Top-N clips 中存在 hard negatives（与 query 词义相似但实际不包含关键信息的 clips），直接将这些 clips 输入 LVLM 会稀释关键信息导致推理失败。Structured Reasoning 引入一个中间推理步骤——将原始 query 分解为一组结构化 subqueries，每个 subquery 的预期答案为二元 (yes/no) 或数值 (count)，用这些结构化的、容易验证的问题对每个检索 clip 逐一筛选，仅保留至少有一个 subquery 正向匹配的 clip，消除 hard negatives。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Vgent 的 Structured Reasoning pipeline：
```
# === Step 1: Subquery Generation ===
Q_struct = LVLM.generate_subqueries(query, keywords)
# 例: query="Did I open the laptop?"
# Q_struct = [
#     {"type": "binary", "text": "Is there a laptop in the video?"},
#     {"type": "binary", "text": "Is the laptop open?"},
#     {"type": "binary", "text": "Is someone interacting with the laptop?"}
# ]

# === Step 2: Per-Clip Structured Verification ===
R_prime = []
for v_i in R_topK:  # Top-20 retrieved clips
    responses = [LVLM.answer(q.text, v_i) for q in Q_struct]
    # binary: 1(yes) or 0(no); numeric: count value
    if any(r > 0 for r in responses):
        R_prime.append(v_i)

R_prime = R_prime[:5]  # max r=5 clips retained

# === Step 3: Information Aggregation ===
summary = LVLM.aggregate(
    subquery_results, refined_clips
)

# === Step 4: Final Generation ===
answer = LVLM.generate(query, context={
    "video_clips": R_prime,
    "reasoning_summary": summary
})
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Vgent 实验：(1) Structured Reasoning 在 GraphRAG 基础上额外提升 MLVU 2.6%、VideoMME 1.6%，总体比 base model 平均提升 3.4%。(2) Structured Reasoning 的效果依赖底层检索质量——应用于 NaiveRAG 时仅带来 65.4→68.6 (+3.2)，仍低于 base model 的 68.8。(3) confidence-based refinement（让模型自我反思 clip 相关性）仅带来 0.2% 提升——验证了结构化验证优于模型自反思路径。(4) r=5（最多保留 5 个 clips）在实验中取得最佳性能。(5) Count 和 Order 任务上提升最为显著——Count: 41.7→58.7 (+17.0), Order: 61.0→67.1 (+6.1)——因为这些任务最需要多 clip 信息聚合。

涉及论文标题：
- Vgent__Graph-based_Retrieval-Reasoning-Augmented_Generation_For_Long_Video_Understanding
