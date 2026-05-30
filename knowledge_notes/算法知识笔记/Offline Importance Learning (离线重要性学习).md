## Offline Importance Learning (离线重要性学习)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Offline Importance Learning 是 AdaSkip 在 Prefilling 阶段使用的 sublayer 重要性学习方法。由于 Prefilling 阶段前无任何可用信息指导 sublayer skipping 决策，AdaSkip 利用历史推理任务中累积的 IO Similarity 统计量为新任务的 skipping 提供依据。

核心 Insight：历史 Prefilling 特征与新的 Prefilling 特征之间具有高度相关性——跨数据集 top-K hit rate 实验（Table 1）验证：如用 TriviaQA 学习的 ATTN similarity 在 MultiFieldQA 上 top-10 hit rate 达 9.31/10，FFN 跨数据集 hit rate 也在 9.38-9.56/10。说明 IO Similarity 分布是模型内在特性，可跨任务共享。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
// Input: N 个历史推理样本，M 个 Transformer Layer，共 2M 个 sublayer

// Step 1: 累积各 sublayer 的 IO Similarity 和 Scale Factor
for sublayer j in 1..2M:
    Simi_j = 0.0; Scale_j = 0.0; total_tokens = 0
    for sample i in 1..N:
        for token t in 1..|T_i|:
            a = get_input_hidden(sublayer=j, token=t)
            b = get_output_hidden(sublayer=j, token=t)
            Simi_j += (a·b)/(|a||b|)        // 公式(2)
            Scale_j += |b|/|a|               // 公式(3)
            total_tokens += 1

// Step 2: 归一化
Simi_j /= total_tokens; Scale_j /= total_tokens

// Step 3: 排序——Similarity 越高越应跳过
sorted_all = sort(zip(Simi, range(2M)), by=Simi, descending=True)

// Step 4: 根据加速比 α 确定跳过数量
m_skip = M - M/α; num_skip = 2 * m_skip
skipped_set = sorted_all[0 : num_skip]
// 在 Prefilling 中：j ∈ skipped_set → output = Scale_j * input
```

术语一般如何实现？如何使用？

一次性 profiling 过程：选定代表性历史数据集（如 AdaSkip 使用的 TriviaQA、MultiFieldQA-en、2WikiMQA），运行一次完整 Prefilling 推理并 hook 各 sublayer 输入/输出 hidden states，计算 average IO Similarity 和 Scale Factor 存储为 per-model metadata。后续对该模型的所有推理任务复用此 metadata。

涉及论文标题：
- AdaSkip: Adaptive Sublayer Skipping for Accelerating Long-Context LLM Inference
