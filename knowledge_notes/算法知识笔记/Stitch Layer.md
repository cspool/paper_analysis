## Stitch Layer

术语解释
Stitch Layer 是 BTS 算法核心的轻量级可学习模块，插入在 Seed（Hub）模型和 Expert（Spoke）模型之间，负责加权合并多个模型的隐藏表示（hidden states）。每个 Stitch Layer 包含两种可学习参数：线性投影 $\{w_{\text{proj}_i} \in \mathbb{R}^{\dim \times \dim}\}_{i=1}^n$ 用于跨模型空间映射，以及线性门控 $w_{\text{gate}} \in \mathbb{R}^{\dim \times \dim \times (n+1)}$ 用于计算各模型表示的贡献权重。BTS 中交替使用两种 Stitch Layer 类型。

术语是什么？
两种 Stitch Layer：

1. **Experts-into-Hub Stitch Layer**（将 Expert 信息合并到 Hub）：
   - Gate 采用 softmax（dropout 后）归一化到 [0,1] 且 sum=1
   - Expert hidden states 投影到 Hub 空间：$\tilde{h}_i = w_{\text{proj}_i}(h_i)$
   - Hub 更新为各表示的加权和：$\tilde{h}_0 = h_0 * g_0 + \sum_{i=1}^n g_i * \tilde{h}_i$
   - Expert hidden states 不变

2. **Hub-into-Experts Stitch Layer**（将 Hub 信息注入 Expert）：
   - Gate 采用 sigmoid（dropout 后）输出 [0,1] 独立权重
   - Hub 保持不变：$\tilde{h}_0 = h_0$
   - 每个 Expert 混入 gated Hub 信息：$\tilde{h}_i = (1 - g_i) * h_i + g_i * w_{\text{proj}_i}(h_0)$

从算法pipeline角度拆解术语。
```
def StitchLayer(xs, merge_into_hub=True):
    x_hub = xs[0]
    x_experts = xs[1:]
    g = dropout(w_gate(x_hub))  # [bs, seq_len, dim, n+1]
    
    if merge_into_hub:  # Experts-into-Hub
        g = g.softmax(dim=-1)
        h_experts = [w_proj[i](x_experts[i]) for i in range(n)]
        h_hub = (g * stack([x_hub] + h_experts, dim=-1)).sum(-1)
    else:  # Hub-into-Experts
        g = g.sigmoid()
        h_experts = [(1-g[...,i+1])*x_experts[i] + g[...,i+1]*w_proj[i](x_hub) 
                     for i in range(n)]
        h_hub = x_hub
    
    return stack([h_hub] + h_experts, dim=-1)
```

Softmax vs Sigmoid gate 选择的设计理由：Experts-into-Hub 使用 softmax 使所有 Expert 的贡献归一化后竞争性地加权（总和为 1），适合 Hub 侧的信息聚合；Hub-into-Experts 使用 sigmoid 使每个 Expert 独立决定融入 Hub 信息的比例（各 Expert 门控值互不影响），适合 Expert 侧的信息吸收。

术语一般如何实现？如何使用？
- 实现：PyTorch nn.Module，w_gate 为 nn.Linear，w_proj 为 nn.ModuleList of nn.Linear
- 参数总量：$K \times (n \times \dim^2 + \dim \times (n+1))$。BTS 中 K=4, n=3, dim=3072 → 约 264M 可训练参数
- 训练仅针对 Stitch Layer，使用 next-token prediction loss from Hub output
- 消融结果：(1) 4 层 vs 10 层性能相近，1 层明显不足；(2) 交替架构对 cross-capability 至关重要；(3) Seed 作 Hub 明显优于 Expert 作 Hub

涉及论文标题：
- BTS Harmonizing Specialized Experts into a Generalist LLM

---
