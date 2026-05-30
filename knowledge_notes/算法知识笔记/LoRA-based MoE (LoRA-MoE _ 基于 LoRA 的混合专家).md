## LoRA-based MoE (LoRA-MoE / 基于 LoRA 的混合专家)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
LoRA-based MoE 是一类将 LoRA（低秩适配）与 MoE（混合专家）架构结合的 PEFT 方法：在预训练 dense 模型的 transformer 子层中插入多个 LoRA adapter 作为 expert，用可训练 router 为每 token 选择激活的 expert 子集，以低成本扩展容量并获得 MoE 的多任务泛化优势。MixLoRA 的关键设计区别于其他 LoRA-MoE 方法：expert = 共享冻结 FFN 权重 + 独立 LoRA adapter（而非LoRA 本身作为 expert），更贴近 Mixtral 等预训练 MoE 架构；attention 层使用独立 LoRA 适配器（非 MoE）。解决标准 LoRA 的多任务 catastrophic forgetting 和容量受限问题。

从算法pipeline角度拆解术语（MixLoRA 与其他 LoRA-MoE 流派对比）：

| 方法 | Expert构造 | 位置 | Router | LB | 特点 |
|------|----------|------|--------|----|------|
| MOELoRA | sub-rank per module | Attn+FFN | 有 | 有 | 对比学习路由 |
| LoRAMoE | 多LoRA+FFN | FFN only | 有 | 有 | 防知识遗忘 |
| MOLA | 层级expert数 | Attn+FFN | 有 | 有 | 高层多expert |
| PESC | dense→sparse | FFN only | 有 | 有 | dense2sparse |
| **MixLoRA** | **共享FFN+LoRA expert** | **FFN(MoE)+Attn(独立LoRA)** | **Top-2** | **有** | **对齐预训练MoE** |

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 开源：https://github.com/TUDB-Labs/MixLoRA
- 关键参数：N=8 experts, K=2, r=16（远小于标准 LoRA r=80）。共享 FFN 计算减少 30% token 延迟；多模型 batching 减少 40% GPU memory。
- 适用：consumer GPU（24GB）多任务微调、multi-tenant LoRA serving。

涉及论文标题：
- MixLoRA: Enhancing Large Language Models Fine-Tuning with LoRA based Mixture of Experts
- MoDE: Effective Multi-task Parameter Efficient Fine-Tuning with a Mixture of Dyadic Experts
- MoLA: MoE LoRA with Layer-wise Expert Allocation (applies LoRA-MoE to ALL dense weight matrices in Transformer, including attention Wq/Wk/Wv/Wo and MLP Wgate/Wdown/Wup; introduces layer-wise expert allocation — different layers have different numbers of LoRA experts; lower layers have more expert redundancy, middle/upper layers benefit from more experts)

MoDE 论文对 LoRA-MoE 的贡献：(1) 通过 PCA 分析发现 down-projection 向量跨任务聚类，提出共享 down-projection 矩阵 A（LoRA-MoE-SD），节省 ~64% 参数同时提升性能；(2) 将 LoRA 分解为 dyadic sum Σ(a_j ⊗ b_j)，对每个 rank j 独立配置 m 个 rank-one adapter 并 per-rank routing，实现 m^r 种组合空间（vs 传统 m 种）；(3) 泛化为 rank-p adapter（MoDE m×r×p）。

---
