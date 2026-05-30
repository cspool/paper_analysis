## Semantic-based Expert Map Search（基于语义的专家图搜索）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Semantic-based Expert Map Search 是 FineMoE 的两种 expert map 检索方式之一，利用 MoE 模型中 embedding layer 输出的 semantic embedding 与 Expert Map Store 中历史 semantic embeddings 的 cosine similarity，检索最相似的 historical expert map。核心假设：语义相似的 prompts 具有相似的 expert 选择模式（此假设经 Pearson correlation 验证，semantic similarity 与 expert hit rate 正相关）。主要用于前 d 层（prefetch distance 内，尚无足够 trajectory history 可用时）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
Semantic-based Expert Map Search 流程：

Input: new_prompt_tokens ∈ R^{B×seq_len}
        Expert Map Store: {sem_old ∈ R^{C×h}, map_old ∈ R^{C×L×J}}
        prefetch_distance d

# Step 1: 提取 semantic embedding
sem_new = embedding_layer(new_prompt_tokens)  # R^{B×h}, h=4096 for Mixtral

# Step 2: pairwise cosine similarity
score_sem ∈ R^{B×C} = (sem_new · sem_old^T) / (||sem_new|| · ||sem_old||)
# 每个 batch 元素与 C 个历史 prompts 的 pairwise similarity

# Step 3: 选择最相似 historical iteration
for b in range(B):
    best_iter[b] = argmax(score_sem[b, :])  # 第 y 个历史 iteration

# Step 4: 使用 best_iter 的 expert map 指导前 d 层 prefetch
for l in range(1, d+1):
    P_l = map_old[best_iter, l, :]  # 第 y 个历史 iteration 第 l 层的概率分布
    prefetch_experts_with_similarity_aware_selection(P_l, score_sem)

# 仅用于 l ∈ [1, d] 的层 → 之后切换为 trajectory-based search
```

关键设计决策：使用模型的原始 embedding layer 而非额外训练的 encoder，因为 "words that appear in similar contexts will have similar embeddings" (Mikolov et al., 2013)。Embedding layer 的输出天然捕获了 prompt 的语义特征，且无需额外计算开销。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
FineMoE 中语义搜索用 PyTorch native cosine_similarity 实现。sem_old 和 map_old 均为预先填充的历史数据（70% prompts 用于 Expert Map Store）。语义搜索的有效性由 Pearson correlation analysis 验证（图 9）：所有 6 个 model-dataset 组合中，semantic similarity 与 expert hit rate 的 Pearson coefficient > 0，表明正相关。搜索开销极小（<50ms），与 expert prefetching 均为异步执行，不进入 critical path。

涉及论文标题：
- Taming Latency-Memory Trade-Off in MoE-Based LLM Serving via Fine-Grained Expert Offloading
