## Fairness-Oriented Loss (FOL)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

FOL（Fairness-Oriented Loss）是 Fair-MoE 论文提出的公平性损失函数，创新地将 MoE load balance 中使用的方差（variance）度量同时用于公平性优化。FOL 由五个组件组成：

$$FOL = F_{EI} + F_{ET} + F_{FI} + F_{FT} + L_{distance}$$

其中：
- **F_EI**：图像 embedding-based MoE 的方差损失
- **F_ET**：文本 embedding-based MoE 的方差损失
- **F_FI**：图像 feature-based MoE 的方差损失
- **F_FT**：文本 feature-based MoE 的方差损失
- **L_distance**：Sinkhorn distance loss（继承自 FairCLIP）

以 F_EI 为例，核心公式：

$$F_{EI} = \sum_{p \in P} \sum_{j=0}^{M^1-1} (Var(O_{N_j}) - Var(O_{N|p_j}))^2$$

其中 O_N 是从整个数据集采样的 N 个样本的 gate weight 矩阵（所有 expert 的权重），O_{N|p} 是从特定受保护属性组 p 采样的 gate weight 矩阵，Var(·) 计算每列（每个 expert）的方差，P 是某属性的所有组集合（如 race 的 {White, Black, Asian}）。

**核心设计思想**：FOL 同时优化两个维度的公平性：(1) L_distance 最小化不同属性组分布之间的**距离**（位置对齐）；(2) 四个方差项最小化不同属性组分布的**离散度差异**（形状对齐，即方差对齐）。方差优化同时服务于 load balancing（防止 MoE 训练中 expert 使用不均衡导致的训练不稳定），从而让 Fair-MoE 能更好地利用 MoE 的学习能力提取公平特征。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

FOL 的计算流程：

```
# 输入: 训练一个 batch 的数据 (image, text, protected_attr_labels)
# 对于四个 MoE 模块分别计算方差差异

# 以图像 embedding-based MoE 为例:
# 从全数据集和每个属性组分别累积 gate weights
O_N = []      # 全数据集的 weights
O_N_race_0 = []  # race=White 的 weights
O_N_race_1 = []  # race=Black 的 weights
# ... 类似地累积其他属性组

for batch in dataloader:
    W^1 = FO_MoE_image_emb.gate(batch.images)  # gate weights
    O_N.append(W^1)
    for p in protected_groups:
        mask = (batch.attr == p)
        O_N_p.append(W^1[mask])

# 计算 F_EI
F_EI = 0
for p in protected_groups:        # 遍历每个属性组
    for j in range(M^1):          # 遍历每个 expert
        var_all = Var(O_N[:, j])
        var_group = Var(O_N_p[:, j])
        F_EI += (var_all - var_group)^2

# 类似地计算 F_ET, F_FI, F_FT
# L_distance 使用 Sinkhorn distance (最优传输距离)
# 最终 FOL = F_EI + F_ET + F_FI + F_FT + L_distance
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

FOL 在训练过程中作为辅助 loss 与 CLIP 的对比学习 loss 联合优化：L_total = L_CLIP + λ · FOL。方差通过 PyTorch 的 `torch.var()` 在累积的 gate weight 矩阵上计算（需要采样足够多数据以获得稳定的方差估计）。FOL 适用于任何使用 MoE 架构且需要考虑公平性的场景，特别是医疗影像分析中多个受保护属性（race, gender, ethnicity, language）共存的情况。消融实验证明：移除 FOL 导致 Race AUC 下降 2.56%，Gender ES-AUC 下降 2.34%，验证了方差优化对公平性和有效性的双重贡献。

涉及论文标题：
- Fair-MoE: Fairness-Oriented Mixture of Experts in Vision-Language Models
