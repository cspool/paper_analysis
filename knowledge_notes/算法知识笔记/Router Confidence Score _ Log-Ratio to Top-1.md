## Router Confidence Score / Log-Ratio to Top-1

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Router Confidence Score 是 LYNX 提出的衡量 MoE router 对每个 token-expert assignment "置信度"的度量。定义为 token 对某个 expert 的 router logit 相对于 top-1 expert logit 的差值（log-ratio）：confidence(t, e) = logit[t][e] - logit[t][top1]。由于 softmax 概率比的对数等价于 logit 差（log(P(e)/P(top1)) = logit[e] - logit[top1]），log-ratio 直接反映了 router 在各 expert 间的区分度。

LYNX 通过实验验证了该度量的有效性：high-confidence tokens（router 强烈偏好某个 expert，各 expert 分数差异大）的 expert assignment 必须保留；low-confidence tokens（各 expert 分数接近）可以安全地重映射到其他 expert 而不影响输出质量。该区分能力源于 MoE 训练中 load-balancing loss 的副作用——训练时强制均匀利用 expert，导致 router 对许多 token 产生弱偏好（各 expert 分数接近），这些弱偏好在 inference 时是冗余的。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Router Confidence 计算（单 token，MoE layer l）
输入: router_logits z ∈ R^N  (N experts)

# 标准 softmax
p_i = exp(z_i) / Σ_j exp(z_j)    # i ∈ [1, N]

# Top-k selection
topk_idx = argsort(p, descending=True)[:k]

# Confidence: log-ratio to top-1
top1_logit = z[topk_idx[0]]
for e in topk_idx[1:]:
    conf[e] = z[e] - top1_logit   # ≤0, 越接近 0 表示越 confident

# 与概率比的关系
# conf[e] = log(p_e / p_top1)  因为 log(p_e/p_top1) = log(e^{z_e}/e^{z_top1}) = z_e - z_top1
```

LYNX 实验发现 (Figure 6)：随着 confidence threshold 提高，high-confidence 和 low-confidence token 在 remapping 后的 accuracy impact 差异显著扩大。这表明 log-ratio 确实是区分 critical vs redundant token-expert mapping 的可靠信号。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

在 LYNX 实现中，Confidence Analyzer (Kernel 1) 直接拦截 router 输出的未归一化 logits（而非 softmax 后概率），计算 log-ratio 并做 AffinityBinning 离散化。使用 logits 而非概率避免了 softmax 的数值稳定性问题，且在 GPU 上 logit 差值比概率除法更高效。对于使用 sigmoid-based routing（如 DeepSeek-V2/V3）的模型，使用 pre-sigmoid scores 的差值。

涉及论文标题：
- LYNX: Enabling Efficient MoE Inference Through Dynamic Batch-Aware Expert Selection
