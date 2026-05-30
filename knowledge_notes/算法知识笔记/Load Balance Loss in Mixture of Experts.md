## Load Balance Loss in Mixture of Experts

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Load Balance Loss 是 MoE 训练中的辅助损失函数，防止 gate 将所有 token 路由到少数 expert 导致利用率极度不均衡。Switch Transformers 中定义为：

$$L_{LB} = N \cdot \sum_{i=1}^{N} f_i \cdot P_i$$

其中 N 为 expert 总数，$f_i$ 为路由到 expert i 的 token 比例，$P_i$ 为 gate 分配给 expert i 的平均路由概率。该 loss 最小化 $f_i$ 和 $P_i$ 之间的差异。

FedMoE 客户端本地训练中：$\mathcal{L}_k = \mathcal{L}_{CE} + \alpha \mathcal{L}_{LB}$，$\alpha$ 通常取 0.01。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Load Balance Loss (Switch Transformers)
def load_balance_loss(gate_logits, top1_indices):
    E = gate_logits.shape[1]  # expert 数
    P = softmax(gate_logits, dim=-1).mean(dim=0)   # (E,) 平均路由概率
    f = one_hot(top1_indices, E).float().mean(dim=0)  # (E,) token 分配比例
    return E * (f * P).sum()  # N·Σ f_i·P_i
```

FedMoE 的子模型仅包含部分 expert，load balance loss 仅在子模型包含的 expert 之间作用。

Flex-MoE 中，Load Balance Loss 有独特变体：在 Expert Specialization 阶段，top-1 gate 已被 S-Router 通过 cross-entropy loss 强制绑定到目标 modality combination expert，因此 load/importance balancing loss 仅对**剩余 top-(k-1) 个 expert** 计算（$E \setminus e_{\text{top-1}}$），以避免对已锁定 expert 重复施加平衡约束。公式为：

$$\mathcal{L}_{\text{balance}} = \text{CV}^2\left(\sum_{j}^{N} \text{importance}_j\right) + \text{CV}^2\left(\sum_{j}^{N} \text{load}_j\right)$$

$$\text{importance}_e = \sum_{i}^{N} g_{ie}, \quad \text{load}_e = \sum_{i}^{N} \delta(g_{ie} > 0), \quad \forall e \in E \setminus e_{\text{top-1}}$$

其中 $\text{CV}^2(x) = (\sigma(x) / \mu(x))^2$ 为变异系数平方。loss coefficient 设为 0.01，与 task classification loss 和 cross-entropy loss 联合优化。

GatePro 揭示了 Load Balance Loss 的局限性——它仅解决 token 分配的统计均衡，但不解决 expert 选择的功能多样性（diversity）问题。功能相似的 expert 仍可被同时激活（只要 token 分配均衡），产生冗余计算。GatePro 通过局部竞争机制补充了 diversity 维度，与 LBL 形成互补：LBL 保证"资源利用效率"，GatePro 保证"资源利用质量"。实验证明 GatePro + LBL 的组合收敛最快。

涉及论文标题：
- FedMoE Personalized Federated Learning via Heterogeneous Mixture of Experts
- Flex-MoE: Modeling Arbitrary Modality Combination via the Flexible Mixture-of-Experts
- GLaM: Efficient Scaling of Language Models with Mixture-of-Experts
- GatePro Parameter-Free Expert Selection Optimization for Mixture-of-Experts Models
- HMoE: Heterogeneous Mixture of Experts for Language Modeling
- Layerwise Recurrent Router for Mixture-of-Experts
- LocMoE: A Low-overhead MoE for Large Language Model Training

**RMoE 对 Load Balance Loss 梯度的分析**：RMoE 论文通过分析训练过程中 router 梯度的两个来源（LM loss 和 LB loss），揭示了 linear router 与 GRU router 的行为差异：对于 linear router，LB loss 在训练早期主导梯度（LB grad: 0.433 vs LM grad: 0.625 at step 0.1k），随后 LB grad 迅速衰减（10k 步后 LB grad 仅 0.001-0.011），表明 linear router 过早收敛于 LB loss 的次优解。对于 GRU router，LB loss 梯度在训练早期较稳定（0.337→0.014→0.003），且 LM loss 梯度持续下降，表明 GRU router 更优地优化了 LM loss 与 LB loss 的权衡。结论：跨层 recurrent router 能有效控制 LB loss 的影响，避免其过早主导训练。

**LocMoE 的 Load Balance Loss 用法**：LocMoE 使用与 Switch Transformer 相同的 aux loss 公式 ($L_{aux} = \alpha \cdot n \cdot \sum f_i \cdot P_i$)，α=0.01。区别在于 LocMoE 将其与 locality loss 联合使用：$L_{task} = L_{aux} + L_{loc} + L_{cross}$，同时约束负载均衡和局部性。

**LLEP 对 Load Balance Loss 的替代视角**：

LLEP 指出了 Load Balance Loss（及 moving-average routing bias）的局限性：这些方法在训练过程中强制统计均衡，但在 post-training 或推理阶段不可行（会改变预训练的 routing behavior，破坏模型完整性）。LLEP 提出了系统级的替代方案——在 dispatch-combine 通信阶段动态重新分配 token 到 GPU，而不修改 gate network 的输出或 expert FFN 计算。这使得 LLEP 适用于 post-training（SFT、RLHF）、推理、甚至训练（支持 backward pass），而 Load Balance Loss 仅适用于预训练。LLEP 的 exact computation 属性（保证数学输出与传统 EP 完全一致）是其区别于所有修改模型行为方案的核心优势。
