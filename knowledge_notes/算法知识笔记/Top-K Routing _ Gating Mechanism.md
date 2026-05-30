## Top-K Routing / Gating Mechanism

术语解释
Top-K路由是MoE模型中最常用的专家选择策略，对每个token选择门控概率最高的K个专家进行处理，其余专家不参与计算。

术语是什么？
路由（Gating/Routing）机制决定每个输入token应该激活哪些专家。Top-K路由的具体计算过程：
1. 门控网络：θ = Softmax(R(x))，R通常是线性层W_r·x
2. Top-K选择：E_selected = TopK(θ, K)，保留概率最高的K个
3. 稀疏化：将未选中专家的权重置零，仅K/N的专家参与计算

K的典型取值为1~8。K=1（top-1 routing）最大稀疏性但可能性能不足；K=2（top-2 routing）是常用折中（如Mixtral-8x7B）。更细粒度的MoE（如DeepSeek-V2）使用更小的专家和更大的K和N。

从算法pipeline角度拆解术语。
```
def topk_gating(x, router_weight, K, N):
    # x: [batch, seq_len, d_model]
    # router_weight: [d_model, N]
    logits = x @ router_weight           # [batch, seq_len, N]
    probs = softmax(logits, dim=-1)      # [batch, seq_len, N]
    topk_vals, topk_idx = topk(probs, K, dim=-1)
    # 归一化选中的权重
    topk_vals = topk_vals / topk_vals.sum(dim=-1, keepdim=True)
    # 未选中的专家权重置零
    mask = zeros_like(probs)
    mask.scatter_(-1, topk_idx, topk_vals)
    return mask
```
固定Top-K的问题：所有token使用相同数量的专家，无法根据token复杂度自适应分配——简单token浪费计算，困难token可能计算不足。

术语一般如何实现？如何使用？
- 实现：通常为nn.Linear(d_model, N)，后接Softmax+TopK
- 辅助负载均衡损失：防止某些expert被过度使用或完全不被使用
- 容量因子：限制每个expert处理的token数上限，防止热点

Adaptive Gating (Li et al., EMNLP 2023) 从实证角度揭示了固定 top-2 的浪费：≥55% 的 token 其 top-1 与 top-2 概率差异显著，这些 token 仅需单 expert。固定 top-2 为所有这些 token 浪费了 1 个 expert 的 FLOPs，且训练时 all-to-all 通信量也翻倍。

Ada-K 论文定量分析了固定 Top-K 的具体缺陷：Mixtral-8x22B 降低 k=2→1 导致平均准确率下降 15.80 点，Mixtral-8x7B 降低 7.68 点。Ada-K 通过动态路由在减少 34.4% 专家激活的同时提升性能 +0.77。

涉及论文标题：
- A Survey on Inference Optimization Techniques for Mixture of Experts Models
- A Survey on Mixture of Experts in Large Language Models
- Ada-K Routing Boosting the Efficiency of MoE-based LLMs
- Adaptive Gating in Mixture-of-Experts based Language Models
- Beyond Distillation Task-level Mixture-of-Experts for Efficient Inference
- Dense Backpropagation Improves Training for Sparse Mixture-of-Experts

**Task-MoE 的路由粒度分类（Kudugunta et al., EMNLP 2021）**：
路由决策可在三个粒度级别：
1. **Token-level**（标准）：GATE(x_s)，每 token 独立选择 experts → 推理时需全部 experts
2. **Sentence-level**：GATE(mean(x_{1:S}))，整句共享相同 experts → 效果较差
3. **Task-level**（提出）：GATE(task_id)，同 task 所有 token 共享 experts → 允许 sub-network extraction

Hybrid 策略（Token encoder + Task decoder）在 WMT 上 BLEU 最高（23.6 vs Token/Token 22.6），decoder 推理成本占 200x encoder step time。

---
