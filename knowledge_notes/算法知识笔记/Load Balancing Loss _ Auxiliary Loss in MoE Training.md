## Load Balancing Loss / Auxiliary Loss in MoE Training

术语解释
Load Balancing Loss（又称 Auxiliary Loss / Load Balance Loss）是 MoE 训练中附加在主要语言建模损失上的辅助损失项，鼓励 Router 将 token 均匀分配到各 expert，防止某些 expert 被过度使用而其他 expert 完全不参与训练（dead experts）。

术语是什么？
标准形式（Switch Transformer, Fedus et al. 2022）：
$$L_{aux} = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i \cdot P_i$$

其中 f_i = 路由到 expert i 的 token 比例，P_i = expert i 的平均 gating probability，N = expert 总数，α = 辅助损失系数。当 f_i 和 P_i 均为 1/N（完全均匀）时 L_aux 取得最小值 α。

从算法pipeline角度拆解术语：
```
# Standard Auxiliary Loss (per micro-batch)
def load_balancing_loss(gate_probs, topk_indices, N_experts, alpha=0.01):
    # gate_probs: [B, N] softmax router outputs
    # topk_indices: [B, K] selected expert indices
    
    # Expert selection frequency f_i
    mask = one_hot(topk_indices, N_experts).sum(dim=-2)  # [B, N]
    f_i = mask.sum(dim=0) / mask.sum()                    # [N]
    
    # Average gating probability P_i
    P_i = gate_probs.mean(dim=0)                          # [N]
    
    # Load balancing loss
    L_aux = alpha * N_experts * sum(f_i * P_i)
    return L_aux

# Total training loss
L_total = L_lm + L_aux
```

**Globally Reduced Auxiliary Loss (Qiu et al., 2025)**: 将 f_i 的计算从 micro-batch 级别改为 global-batch 级别。跨所有 Data Parallel ranks 同步 f_i（仅 N_E 维向量，通信量极小），用全局 f̄_i 替换本地 f_i。这放松了均衡约束——允许每个 micro-batch 内 expert 使用不均，但整个 global batch 均衡——从而促进 expert domain specialization。

术语一般如何实现？如何使用？
- **系数 α**: 典型值 0.01（Switch Transformer 风格）。α 过小 → expert 负载不均或 dead expert；α 过大 → router 过度均匀分配，抑制 expert specialization
- **变体**: 
  - z-loss: 额外的门控 logit 正则项，稳定训练（DeepSeek-V2/V3）
  - Auxiliary-Loss-Free 策略 (Wang et al., 2024): 通过可学习 expert bias b_i 动态调整路由，完全替代 auxiliary loss
  - Group-Level Load Balancing (ARIA): 对 expert 组施加约束（详见对应条目）
- **DefaultMoE 的使用**: α=0.01，使用 globally reduced auxiliary loss（跨节点计算），不使用 z-loss 和 jitter（该规模下无益）。global-batch LBL 使 baseline 性能显著提升，因此 prior routing method 声称的"free lunch"改进被削弱
- **与 Dense Backpropagation 的关系**: DefaultMoE 的 dense gradient 通过减少 Router 梯度误差进一步提升训练稳定性，允许使用更大 learning rate（9×10⁻⁴ vs baseline 的 7×10⁻⁴）而不出现 loss spike
- **Duo-LLM 的 Budget Loss 变体**: 不同于 per-layer 负载均衡，Duo-LLM 使用全局 Budget Loss 约束所有层的 big 模块总使用比例：L_budget = (mean(P_big across all layers) - target_budget)²。这允许 router 跨层灵活分配计算——某些层可以更多使用 big 模块，另一些层更多使用 small 模块——只要全局满足预算。配合 soft routing（温度 τ 逐渐增大实现硬分配），router 被鼓励发现跨层的复杂路由模式而非 per-layer 均匀分配。

涉及论文标题：
- CuMo: Scaling Multimodal LLM with Co-Upcycled Mixture-of-Experts
- Dense Backpropagation Improves Training for Sparse Mixture-of-Experts
- Demons in the Detail: On Implementing Load Balancing Loss for Training Specialized Mixture-of-Expert Models
- Dense Training, Sparse Inference Rethinking Training of Mixture-of-Experts Language Models
- Duo-LLM: A Framework for Studying Adaptive Computation in Large Language Models
- Every Expert Matters: Towards Effective Knowledge Distillation for Mixture-of-Experts Language Models

**CuMo 中的 bzloss 使用**：CuMo 使用标准 load balancing loss (α_b=0.1) + router z-loss (α_z=0.01)，合称 "bzloss"，分别独立应用于 MLP connector、CLIP vision encoder 和 LLM 的每个 MoE 块。总损失 L = L_ce + 0.1·L_b + 0.01·L_z。CuMo 的消融实验（Table 3）表明加入 bzloss 后在 MMVet 上取得明显提升（32.3 → 33.1），验证了负载均衡对 MoE 多模态训练的正面影响。α_b=0.1 相比标准 α=0.01 更高，论文未解释原因。

**DS-MoE 中的 MI Loss**：DS-MoE (Pan et al., 2024) 引入基于信息论的 Mutual Information (MI) Loss 作为负载均衡替代方案。与 switch loss 的双线性形式不同，MI Loss 分为两项：
1. **最大化 expert 分布熵** H(e) = -Σ p(e) log p(e)：促进全局负载均衡（所有 expert 被均匀使用）
2. **最小化条件熵** H(e|X) = -Σ p(e|x) log p(e|x)：鼓励 Router 对每个 token 产生集中的、确定性的 expert 分配（sparse concentration）
总损失：L_MI = -H(e) + (1/|X|) Σ H(e|X)，总训练 loss：L = L_LM + α·L_MI，其中 α 控制 sparsity 程度。

MI Loss 的特殊优势：(a) 不需要 fixed K——支持训练后灵活选择 inference sparsity 级别；(b) 自我平衡——H(e) 推动均衡、H(e|X) 推动集中，两者形成对抗平衡；(c) 训练后 Router 自然产生 sparsity，可仅凭阈值 ε 或 TopK 选择激活 expert。DS-MoE 的 α 参数：MoA 层 3.5e-4 (1B) / 2e-4 (3B/6B)，MLP 层 6.3e-4 (1B) / 4e-4 (3B) / 2e-4 (6B)。α 越大 → sparsity 越高（模型在高 sparsity 下性能保持更好），但可能在低 sparsity 下性能略差。

---
