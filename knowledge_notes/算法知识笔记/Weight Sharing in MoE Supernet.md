## Weight Sharing in MoE Supernet

术语解释
Weight Sharing in MoE Supernet 是 AutoMoE 提出的专用权重共享技术，使 Supernet（最大 MoE 配置）和其子架构之间可以通过"提取前 rows/columns"的方式共享权重。这是将 Supernet training 从 dense Transformer 扩展到稀疏 MoE 架构的关键技术。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。
核心操作：给定 Supernet 中某层的 router 权重 W_super ∈ R^{M×d_max}（M 个 expert，d_max 最大 embedding 维度），子架构需要 W_sub ∈ R^{n×d}（n < M 个 expert，d < d_max embedding 维度）。提取方式为：
- W_sub = W_super[:n, :d]（取前 n 行、前 d 列）

对 Expert FFN 权重也是类似操作。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
# Supernet Router: W_super ∈ R^{M × d_max}
# Subnet Router (n experts, d embed): W_sub ∈ R^{n × d}
W_sub = W_super[:n, :d]  # front n rows, front d columns

# Supernet Expert FFN Layer 1: W_ffn1_super ∈ R^{d_ff_max × d_max}
# Subnet Expert FFN Layer 1 (d_ff=2048, d=512):
W_ffn1_sub = W_ffn1_super[:2048, :512]  # front 2048 rows, front 512 columns

# Supernet Expert FFN Layer 2: W_ffn2_super ∈ R^{d_max × d_ff_max}
# Subnet Expert FFN Layer 2 (d=512, d_ff=2048):
W_ffn2_sub = W_ffn2_super[:512, :2048]  # front 512 rows, front 2048 columns

# 异构 expert sizes 示例：
# Layer 有 4 个 expert，FFN sizes 分别为 [3072, 2048, 2048, 1024]
expert_0_W1 = W_ffn1_super[:3072, :512]  # 最大 expert
expert_1_W1 = W_ffn1_super[:2048, :512]  # 中等 expert
expert_2_W1 = W_ffn1_super[:2048, :512]  # 中等 expert（可共享相同位置）
expert_3_W1 = W_ffn1_super[:1024, :512]  # 最小 expert

# 关键约束：子架构中不存在的 expert 不提取权重
# 若子架构只有 2 个 expert，第 3、4 个 expert 的 Supernet 权重不参与该步训练
```

这种设计的核心假设：Supernet 的前几行/列权重经过了最充分的训练（因为几乎所有子架构都包含它们），因此提取的前行/前列权重质量最高。

术语一般如何实现？如何使用？
- 适用于任何可通过"维度截断"表达的搜索维度（expert count, FFN size, embedding size, hidden size）
- 在 training loop 中，每次采样后动态提取对应权重，训练后权重更新回 Supernet
- 这种机制自然地支持异构 expert 尺寸：每个 expert 提取不同数量的 rows/columns
- 无法处理非单调的搜索维度（如"是否使用某类 attention head"），需其他机制
- AutoMoE 代码：https://aka.ms/AutoMoE（基于 fairseq）

涉及论文标题：
- AutoMoE: Heterogeneous Mixture-of-Experts with Adaptive Computation for Efficient Neural Machine Translation

---
