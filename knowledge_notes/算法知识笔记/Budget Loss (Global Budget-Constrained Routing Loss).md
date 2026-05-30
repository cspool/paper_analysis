## Budget Loss (Global Budget-Constrained Routing Loss)

术语解释
Budget Loss 是 Duo-LLM 提出的替代传统 per-layer Load Balancing Loss 的路由训练损失。它对所有层的 big 模块使用比例施加全局约束，允许 router 跨层灵活分配计算，而非强制每层内均匀使用。

术语是什么？
标准 MoE load balancing loss 在每层内独立计算：L_aux = α·N·Σ f_i·P_i（per-layer）。这强制每层内各 expert 使用率相似，但限制了跨层的灵活计算分配。Budget Loss 改为跨所有层的全局约束：

$$L_{budget} = \left(\frac{\sum_{i=1}^{L} P_{i,\text{big}}}{L} - \text{budget}\right)^2$$

其中 P_{i,big} 是第 i 层 router 分配给 big 模块的 softmax 概率，budget 是目标 big 模块使用比例（如 0.33 表示 33% 的层用 big）。总训练损失 L_total = L_CE + α·L_budget。

从算法pipeline角度拆解术语：
```
# Budget Loss Training (Duo-LLM Stage 3)
# L layers, each layer has W_{r,l} router weight

def budget_loss_training_step(x, labels, budget=0.33, alpha=0.01):
    total_ce_loss = 0
    P_big_all = []
    
    for l in range(L):
        # Router: learnable linear layer per layer
        logits_l = x @ W_r[l]                    # [batch, 2]: big/small
        P_l = softmax(logits_l / tau)            # [batch, 2]
        P_big = P_l[:, 0].mean()                  # scalar
        P_big_all.append(P_big)
        
        # Soft combination of big and small FFN outputs
        H_big = BigFFN_l(x)
        H_small = SmallFFN_l(x)
        # x_{out} = P_big * H_big + P_small * H_small  (per-token)
        # NOTE: 论文使用 soft combination 或 hard routing
        x = layer_forward(x, P_l, H_big, H_small)
    
    # Cross-entropy loss
    ce_loss = CrossEntropy(x @ W_vocab, labels)
    
    # Budget loss: global constraint across all layers
    avg_p_big = mean(P_big_all)
    budget_loss = (avg_p_big - budget) ** 2
    
    total_loss = ce_loss + alpha * budget_loss
    return total_loss
```

术语一般如何实现？如何使用？
- Budget 参数控制全局计算量：如 0.33 表示约 4/12 层使用 big 模块
- α 控制 budget 约束的强度，论文未给出具体值
- Soft routing 使用温度参数 τ：训练初期 τ 较小（soft routing），逐渐增大使 P 趋近 one-hot（hard routing）
- 对比 per-layer load balancing：Budget Loss 允许某些层 80% 用 big、另一些层 10% 用 big，只要全局均值满足 budget
- 优势：释放 layer-level routing flexibility，理论上可接近 oracle 的跨层不均匀分配模式

涉及论文标题：
- Duo-LLM: A Framework for Studying Adaptive Computation in Large Language Models

---
