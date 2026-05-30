## Mixture of Parameter-Efficient Experts (MoPE)

术语解释
MoPE 将 MoE 的门控机制与参数高效微调（PEFT，如 LoRA、Adapter、Prefix-tuning）结合，每个专家本身是一个 PEFT 模块，实现多任务微调下的参数效率和任务隔离。

术语是什么？
将 MoE 的多专家架构与 PEFT 的低参数量结合：专家由 LoRA 矩阵/Adapter 块/(IA)^3 向量构成，仅更新专家和门控，base model 冻结。按放置位置分为：FFN-level, Attention-level, Transformer Block-level, Every Layer-level。

从算法pipeline角度拆解术语。
```
# LoRAMoE (FFN-level MoPE)
def loramoe_forward(x, base_ffn, lora_experts, router):
    y_base = base_ffn(x)  # frozen FFN
    weights = softmax(router(x))
    y_lora = sum(weights[i] * (x @ A_i @ B_i) for i in range(N))
    return y_base + y_lora
```

术语一般如何实现？如何使用？
- LoRAMoE: 专家分为两组（任务学习 + 知识保持），局部平衡约束
- MoV: (IA)^3 向量专家，MoV-10 仅用 ~40 个向量
- MoLA: 不同层使用不同数量的专家
- 适用场景：多任务微调、领域适配、指令调优

涉及论文标题：
- A Survey on Mixture of Experts in Large Language Models
- AT-MoE: Adaptive Task-planning Mixture of Experts via LoRA Approach

### AT-MoE 的任务特定 LoRA 专家训练方法

AT-MoE 提出了一种与标准 MoPE 不同的训练策略：先在各任务数据上**分别独立训练**任务特定 LoRA 专家（冻结 LLM 权重 W_0），确保每个专家有明确的任务领域属性，再冻结所有 LoRA 专家，训练路由模块。这与标准 MoPE（专家和路由器在混合数据上联合训练）的根本区别在于：
- **任务级专业化**：每个 LoRA 专家 ΔW_j = B_j A_j（B_j ∈ R^{d×r}, A_j ∈ R^{r×k}）在特定任务领域上训练，而非混合数据
- **专家可解释性**：专家有明确的任务标签（如医学中：诊断专家、处方专家、分诊专家、外科专家、放射科专家）
- **可控融合**：路由器可根据复合指令的多个意图，按层次（群组→组内）分配专家权重

此外，AT-MoE 还训练了一个在所有任务混合数据上训练的"预合并 LoRA 专家"W_p 作为通用后备专家，通过平衡参数 λ 控制任务特定专家与通用专家的融合比例：y_i = (λ·F_G(W̄_e) + (1-λ)·W_p)x_i + W_0·x_i。

---
