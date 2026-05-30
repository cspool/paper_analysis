## MoE-based LoRA (LoRA-MoE / Mixture of LoRA Experts / MoLE)

术语解释
MoE-based LoRA（又称 LoRA-MoE、Mixture of LoRA Experts / MoLE）是将 Mixture-of-Experts 架构与 Low-Rank Adaptation (LoRA) 结合的参数高效微调方法：将单一 LoRA 模块替换为 n 个并行的 LoRA expert，通过 Router 稀疏激活部分 expert 来处理异构数据。

术语是什么？
标准 LoRA 对预训练权重 W₀ 引入低秩矩阵 A, B：
$$y' = W_0 x + BA x$$

LoRA-MoE 将 BA 模块替换为 n 个 expert {E_i = B_i A_i}，通过 Router g(x; G) 控制激活：
$$y' = W_0 x + \sum_{i=1}^{n} g_i(x; G) \cdot E_i(x)$$

其中 g_i(x; G) 经 top-k routing 稀疏化后，仅 top-k 个 expert 的权重非零。

LoRA-MoE 的核心优势：
1. **容量-计算权衡**：n 个 expert 提供 n× 参数容量，但每次仅激活 k 个（通常 k=2），计算量接近 k/n 倍
2. **异构数据处理**：不同 expert 可隐式专门化于不同任务/数据分布，适合 multi-task fine-tuning
3. **与全量 MoE 的区别**：LoRA-MoE 的 expert 是轻量低秩矩阵（r × (d_in + d_out) 参数），而非完整 FFN 层

从算法pipeline角度拆解术语：
```
# LoRA-MoE Layer Forward (以 CoMoE 配置: n=4, k=2, r=16)
class LoRAMoELayer:
    def __init__(self, W_0, n_experts=4, rank=16):
        self.W_0 = W_0                    # frozen pretrained weights
        self.experts = []                 # n LoRA experts
        for i in range(n_experts):
            A_i = nn.Linear(d_in, rank, bias=False)
            B_i = nn.Linear(rank, d_out, bias=False)
            self.experts.append((A_i, B_i))
        self.router = nn.Linear(d_in, n_experts)  # gating network
    
    def forward(self, x):
        # 1. Router: 每个 token 对所有 expert 的 affinity score
        gate_logits = self.router(x)      # [batch, seq, n]
        
        # 2. Top-k selection
        topk_weights, topk_idx = torch.topk(
            torch.softmax(gate_logits, dim=-1), k=2, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        
        # 3. Expert computation (sparse)
        output = self.W_0 @ x             # frozen path
        for i in range(self.n_experts):
            mask = (topk_idx == i).any(dim=-1)  # tokens routed to expert i
            if mask.any():
                A_i, B_i = self.experts[i]
                expert_out = B_i(A_i(x[mask]))  # LoRA: B·A·x
                weight = topk_weights[mask][topk_idx[mask] == i]
                output[mask] += weight.unsqueeze(-1) * expert_out
        
        # 4. (CoMoE 额外) 收集所有 expert 表示用于对比损失
        expert_reprs = [B_i(A_i(x)) for i in range(n_experts)]
        return output, expert_reprs, topk_idx
```

术语一般如何实现？如何使用？
- **实现框架**：HuggingFace PEFT（peft library）提供 MoE-LoRA 支持；MixLoRA (https://github.com/TUDB-LAB/MixLoRA) 是最常用的开源实现之一
- **关键变体**：
  - **MixLoRA** (Li et al., 2024)：resource-efficient sparse MoE，top-k routing + load balance loss
  - **MOELoRA** (Liu et al., 2023)：面向 multi-task medical applications
  - **MiLoRA** (Zhang et al., 2024)：prompt-aware routing 降低延迟
  - **OMoE** (Feng et al., 2025)：orthogonal fine-tuning 强制 expert 多样性
  - **LoRAMoE** (Dou et al., 2024)：token-based routing 缓解知识遗忘
  - **HydraLoRA** (Tian et al., 2024)：非对称 LoRA 结构
- **典型配置**：LLaMA-2 7B 上 r=16, n=4~8, k=2，应用在 Q/K/V/O/Up/Down/Gate 投影层
- **可训练参数占比**：约 0.7%~3%（取决于 n, r, 应用层数）
- **局限**：expert 功能冗余（缺乏专业化约束导致 expert 学到相似功能）、负载不均（部分 expert 过度使用/闲置）

涉及论文标题：
- CoMoE: Contrastive Representation for Mixture-of-Experts in Parameter-Efficient Fine-tuning

---
