## Co-Upcycling

术语解释
Co-Upcycling 是 CuMo 论文提出的稀疏 MoE 初始化策略：在多模态 LLM 的视觉编码器（CLIP ViT）和 MLP 连接器中，将每个 dense MLP 块的权重同时（co-）复制为对应 MoE 专家的初始权重，而非随机初始化专家。这是对 Sparse Upcycling 的扩展——不仅 upcycle 单个模块，而是跨多个模块（视觉编码器 + MLP 连接器）协同 upcycle。

术语是什么？
CuMo 在视觉指令微调阶段引入 Co-Upcycling：(1) 预训练 MLP 连接器和预微调全模型（无 MoE 块）；(2) 将每个 dense MLP 块替换为 Top-2-in-4 稀疏 MoE 块；(3) 将预训练/预微调阶段的同位置 MLP 权重复制到对应 MoE 块中的每个 expert；(4) Router 网络（Top-K gating）随机初始化，从头训练。

```
# Co-Upcycling: 同时 upcycle 多个模块的 MLP → MoE
def co_upcycle(pretrained_model):
    # Step 1: Upcycle MLP connector (两层 MLP)
    for mlp_block in pretrained_model.mlp_connector:
        moe_block = TopK_MoE(num_experts=4, top_k=2)
        for expert in moe_block.experts:
            expert.weight = mlp_block.weight.clone()  # 从预训练 MLP 复制
        moe_block.router = Router(num_experts=4)       # 随机初始化
        replace(mlp_block, moe_block)

    # Step 2: Co-Upcycle CLIP vision encoder 的每个 transformer 层
    for layer in pretrained_model.clip_vit.layers:
        moe_block = TopK_MoE(num_experts=4, top_k=2)
        for expert in moe_block.experts:
            expert.weight = layer.mlp.weight.clone()   # 从同层 MLP 复制
        moe_block.router = Router(num_experts=4)
        layer.mlp = moe_block

    return pretrained_model  # 继续 visual instruction tuning
```

从算法pipeline角度拆解术语：
Co-Upcycling 位于三阶段训练的第三阶段（视觉指令微调）开始时。前两阶段（MLP connector 预训练 + 全参数预微调）产生 warm-up 后的模型参数。第三阶段开始时，dense MLP → MoE 替换发生，expert 初始化来自 warm-up MLP 权重。这避免了从头训练 MoE 的不稳定性——论文报告若从随机初始化训练 MoE blocks，模型无法收敛，即使降低学习率也无法达到 baseline 性能。

关键对比：
- Sparse Upcycling：仅 upcycle LLM 的 MLP → MoE（如 Mistral-7B → Mistral-7B-MoE）
- Co-Upcycling：同时 upcycle CLIP ViT + MLP connector 的 MLP → MoE，而 LLM 使用 pre-trained Mixtral-8×7B（因为论文实验表明 upcycled LLM-MoE 效果不如 pre-trained LLM-MoE）

术语一般如何实现？如何使用？
- 前提：已有完成 MLP connector 预训练和全参数预微调的 dense checkpoint
- CuMo 实验表明 Co-Upcycling 比随机初始化 MoE 专家显著更好（Table 3-4），甚至随机初始化导致模型不收敛
- 学习率：Co-Upcycling 后的视觉指令微调使用 2e-6 ~ 4e-6，比常规微调更低
- 配合辅助损失 bzloss（L_balance α=0.1 + L_z α=0.01）维持 expert 负载均衡

涉及论文标题：
- CuMo: Scaling Multimodal LLM with Co-Upcycled Mixture-of-Experts

---
