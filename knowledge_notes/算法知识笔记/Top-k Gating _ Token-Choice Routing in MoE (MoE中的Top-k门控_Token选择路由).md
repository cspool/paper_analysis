## Top-k Gating / Token-Choice Routing in MoE (MoE中的Top-k门控/Token选择路由)

术语解释
Top-k Gating（Token-Choice Routing）是 MoE 模型中最主流的路由策略：每个 token 独立选择 top-k 个 experts，仅被选中的 experts 参与该 token 的前向计算。这是稀疏 MoE 实现"条件计算"的核心——每个 token 只激活总 expert 数的一小部分（8/64 = 12.5%）。

术语是什么：
$$s_{t,e} = x_t^\top \cdot W_{\mathrm{gate}}[e], \quad \mathcal{K}_t = \mathrm{TopK}(\{s_{t,e}\}, k)$$
$$w_{t,e} = \frac{\exp(s_{t,e})}{\sum_{j \in \mathcal{K}_t} \exp(s_{t,j})} \text{ if } e \in \mathcal{K}_t \text{, else } 0$$
$$h_t = x_t + \sum_{e \in \mathcal{K}_t} w_{t,e} \cdot \mathrm{FFN}_e(x_t)$$

C3PO 不改变 gating 机制本身，而是通过修改 routing weights w_{t,e}（或 gate logits s_{t,e}）来 re-mix expert 贡献。

从算法pipeline角度拆解术语：
```
class SparseMoELayer:
    def forward(self, x, routing_override=None):
        gate_logits = self.gate(x)
        if routing_override is not None:
            gate_logits = gate_logits + routing_override  # C3PO 注入点
        
        topk_weights, topk_indices = torch.topk(
            torch.softmax(gate_logits, dim=-1), k=self.top_k, dim=-1)
        
        output = torch.zeros_like(x)
        for e in range(self.num_experts):
            mask = (topk_indices == e).any(dim=-1)
            if mask.any():
                output[mask] += topk_weights[mask][...].unsqueeze(-1) * self.experts[e](x[mask])
        return output
```

术语一般如何实现？如何使用？
- k 是关键超参数：OLMoE k=8/E=64, Mixtral k=2/E=8, DeepSeekMoE k=6/E=64+2 shared
- C3PO 利用 last token 的 routing weights 做优化——last token 在自回归生成中承载最多决策信号
- CoMoE 在 LoRA-based MoE PEFT 中基于 top-k routing 构建对比学习：激活的 top-k expert 作为正样本，其余 n-k 个非激活 expert 作为负样本，通过 InfoNCE loss 促进 expert 专业化

涉及论文标题：
- C3PO Critical-Layer, Core-Expert, Collaborative Pathway Optimization for Test-Time Expert Re-Mixing
- CoMoE: Contrastive Representation for Mixture-of-Experts in Parameter-Efficient Fine-tuning
