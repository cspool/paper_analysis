## Negative Sample in KV Cache Compression (KV Cache 压缩的负样本分析)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Negative sample（负样本）是在 KV cache 压缩上下文中的一个评估概念：指那些在原始未压缩 LLM 下表现正常（benign），但在应用 KV cache 压缩后精度显著退化的样本。论文 "Rethinking KV Cache Compression" 首次系统性地引入此概念，并使用 Algorithm 1 定义收集流程：给定数据集 $\mathcal{D}$、阈值 $\theta$、LLM $\mathcal{M}$、baseline 算法 $\mathcal{A}_b$ 和压缩算法集合 $\mathcal{A}$，若某样本 $d_i$ 在所有压缩算法下的 accuracy 均低于 $(1-\theta) \times p_{\text{base}}$（其中 $p_{\text{base}}$ 为 baseline 下的 accuracy），则该样本被标记为 negative sample，加入 $\mathcal{D}_{neg}$。关键发现：即使压缩算法整体平均 accuracy 损失很小（<1%），仍存在大量 negative samples（例如 threshold=10% 时数百个），揭示压缩算法在不同样本和任务类型上的脆弱性不均衡。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**Negative Sample 收集流程（论文 Algorithm 1 复述）**：
```
Input: Dataset D, Threshold theta (e.g., 0.10),
       LLM M, Baseline Algorithm A_base, Compression Algorithm Set A = {A1, A2, ...}
Output: Negative Dataset D_neg

D_neg = empty_set()

for each sample d_i in D:
    # Step 1: 获取 baseline 精度
    p_base = Accuracy(A_base, M, d_i)

    # Step 2: 只评估 benign samples（baseline 精度 >= 平均值）
    if p_base < average(D, A_base):
        continue

    # Step 3: 检查所有压缩算法是否都 fail
    negative = true
    for each A_j in A:
        p_comp = Accuracy(A_j, M, d_i)
        if p_comp >= (1 - theta) * p_base:
            negative = false    # 至少一个算法通过
            break

    # Step 4: 所有算法都 fail → negative sample
    if negative:
        D_neg.insert(d_i)

return D_neg
```

**Annotations**: 阈值 $\theta$ 控制严格度——$\theta=0.10$ 意味着压缩后 accuracy 不得低于 baseline 的 90%。论文使用 LongBench 数据集和 LLaMA-3.1-8B-instruct 评估，发现 negative samples 集中在 summarization 和 QA 任务类型（这些任务严重依赖长上下文信息，KV cache 压缩导致的关键信息丢失对它们影响最大）。多个压缩算法（KIVI+GEAR 或 H2O+StreamingLLM）联合时 negative samples 减少但不能完全消除。

术语一般如何实现？如何使用？

论文提供的工具链（https://github.com/LLMkvsys/rethink-kv-compression）包含 negative sample evaluator：从 LongBench 收集的、经 10% threshold 筛选的 negative samples 构成 benchmark 数据集，用于评估新的 KV cache 压缩方法在困难样本上的表现。论文 Table 7 显示：LLaMA-3.1-8B-instruct 上 baseline FP16 在 summarization/QA/code 上得分为 31.6/52.0/97.0，而 KIVI 降至 24.8/28.8/30.0——code 任务退化最严重（97→30）。论文推荐的方向：(1) 用 lightweight model 预测请求的 task type，(2) 开发 task-specific 压缩策略，(3) 对不同 task 使用不同压缩级别。

涉及论文标题：
- Rethinking Key-Value Cache Compression Techniques for Large Language Model Serving
