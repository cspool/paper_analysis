## Mixture of LoRA Experts (MoLE)

术语解释
Mixture of LoRA Experts (MoLE) 是将 Mixture of Experts 架构与 LoRA 结合的参数高效微调范式，在每层用多个 LoRA expert 替代单个 LoRA adapter，通过路由机制为不同输入动态选择和组合 LoRA 专家。

术语是什么？
MoLE 的核心思想：在 Transformer 层的目标权重矩阵（如 attention Q/K/V/O）上，创建 N 个独立的 LoRA adapter（各 rank = r_total / N），一个可训练的路由器计算 Softmax 选择概率，Top-K 选出最相关的 LoRA 专家：
$$h = W_0 x + \sum_{i \in \text{TopK}(p, K)} \frac{p_i}{\sum_{j \in \text{TopK}(p, K)} p_j} \cdot B_i A_i x$$

其中 p = Softmax(W_g x) 为路由器输出。

**与标准 MoE 的差异**：MoLE 的 expert 是 LoRA adapter（仅修改权重更新 ΔW 而非完整 FFN），路由作用于 fine-tuning 时的参数更新而非模型前向计算路径。

**关键设计维度**：
1. **专家粒度**：每 expert 作用于单个权重矩阵，N 专家共享总秩（参数等价于单 LoRA）
2. **门控策略**：top-k（k=1,2,3）、固定阈值 τ=1/N、动态阈值 τ(x)（AdaMoLE）
3. **层级位置**：通常应用于 self-attention 四矩阵（Wq, Wk, Wv, Wo）

从算法pipeline角度拆解术语。
```
# MoLE Forward (top-2 gating on self-attention Q projection)
def mole_forward(x, W_q, lora_experts, router):
    base = x @ W_q.T                       # pretrained output
    logits = router(x)                     # [batch, seq, N]
    probs = F.softmax(logits, dim=-1)
    topk_probs, topk_indices = torch.topk(probs, k=2, dim=-1)
    topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
    
    delta = torch.zeros_like(base)
    for k in range(2):
        for expert_idx in topk_indices[:,:,k].unique():
            mask = (topk_indices[:,:,k] == expert_idx)
            delta[mask] += topk_probs[:,:,k][mask].unsqueeze(-1) * lora_experts[expert_idx](x[mask])
    return base + delta
```

AdaMoLE 将 top-k 替换为动态阈值：激活所有 p_i ≥ τ(x) 的 expert，τ(x) 由阈值网络生成。

术语一般如何实现？如何使用？
- 配置：N × r = total_rank（如 N=8, r=4 → total=32）
- 门控变体：top-k（MoLE/MoLA）、固定阈值 τ=1/N、动态阈值 τ(x)（AdaMoLE）
- 训练：frozen W_0 + load balancing loss（λ=1e-3），防止 expert 坍塌
- 推理：所有 expert 矩阵常驻 GPU 显存，router 每次 forward 动态门控

涉及论文标题：
- AdaMoLE Fine-Tuning Large Language Models with Adaptive Mixture of Low-Rank Adaptation Experts

---
