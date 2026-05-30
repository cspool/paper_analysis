## LLM Entropy-Based Confidence Calibration / LLM熵基置信度校准

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
ReVisionLLM 提出的替代 CLIP similarity ranking 的预测排序方法。对底层 hierarchy 的每个候选预测，计算 LLM 自回归生成每个词时输出概率分布的熵，取均值再取倒数作为置信度：R^i = 1 / mean_k(H_k^i)，其中 H_k^(i) = -Σ_w p(w|T_<k, D^(i)) log p(w|T_<k, D^(i))。直觉：LLM 对确信的视觉输入输出集中低熵分布→高置信度；不确时输出分散高熵分布→低置信度。按 R^i 降序排列选 Top-K。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
def confidence_score(prediction):
    total_H = 0
    for k, logits_k in enumerate(prediction.logits):  # (vocab_size,) per step
        probs = softmax(logits_k)
        H_k = -sum(probs * log(probs + ε))
        total_H += H_k
    return 1.0 / (total_H / K)  # K=生成词数

ranked = argsort([confidence(p) for p in predictions], descending=True)
top_k = [predictions[i] for i in ranked[:K]]
```
Annotaions: ε 防 log(0)。仅用于底层 hierarchy (dense features)。Table 2: +Calibration (-CONE) 将 R1@.1 从 4.8% → 8.4%，ECE 从 0.6231 → 0.4614。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
有效性源于 Stage 1 contrastive training → LLM 学会对不确定输入输出高熵。与 TimeJudge (logit-space calibration offset) 不同，ReVisionLLM 依赖训练阶段的隐式校准。要求推理框架保留 LLM 原始 logits。代码: https://github.com/Tanveer81/ReVisionLLM。

涉及论文标题：
- ReVisionLLM__Recursive_Vision-Language_Model_for_Temporal_Grounding_in_Hour-Long_Videos
