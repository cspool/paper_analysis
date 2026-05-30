## Dynamic Rank Selection (Energy-based)（基于能量的动态秩选择）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Dynamic Rank Selection 是 X-EcoMLA 中提出的一种基于奇异值累积能量的自适应 rank 选择策略。与 Fixed Rank Selection（所有 transformer 层使用统一的 r_q 和 r_kv）不同，Dynamic Rank Selection 根据每层权重矩阵的奇异值分布，自动确定该层所需的 rank 值——信息丰富的层（奇异值衰减慢）自动分配更高 rank，冗余的层（奇异值衰减快）分配更低 rank。

逻辑链：对 W^Q（或 [W^K, W^V]）做 SVD 得到奇异值序列 σ_1 ≥ σ_2 ≥ ... ≥ σ_min(d, n_h·d_h) → 计算总能量 E = Σ σ_j² → 设定能量阈值 δ（如 0.90 或 0.95）→ 选择最小的 rank R 使得 Σ_{j=1}^R σ_j² ≥ δ·E → R 是该层在保留 δ 比例信息的条件下的最优低秩维度。

从算法pipeline角度拆解术语，给出具体例子。

**动态 Rank 选择算法：**

```
def dynamic_rank_selection(W, delta=0.95):
    """
    W: weight matrix [d_out, d_in]
    delta: energy preservation threshold (0 < delta <= 1)
    """
    _, Σ, _ = torch.linalg.svd(W, full_matrices=False)  # 经济型 SVD
    energies = Σ ** 2                                     # 奇异值平方
    total_energy = energies.sum()                         # 总能量 E
    cumulative_energy = torch.cumsum(energies, dim=0)     # 累积能量
    # 找到最小 rank 使得累积能量 >= delta * total_energy
    mask = cumulative_energy >= delta * total_energy
    rank = mask.nonzero(as_tuple=True)[0][0].item() + 1  # 1-indexed
    return rank

# 使用
r_q_i = dynamic_rank_selection(W_Q_layer_i, delta_q)      # 第 i 层 Q 的 rank
r_kv_i = dynamic_rank_selection(
    torch.cat([W_K_layer_i, W_V_layer_i], dim=-1), 
    delta_kv
)                                                          # 第 i 层 KV 的 rank
```

**X-EcoMLA 实验中的动态 rank 选择效果（论文 Table 1、Table 6-7）**：
- δ=0.95 时 KV size 约 54.7%（Llama3.2-1B），Avg score 53.12（vs baseline 52.85）——动态 rank 自动平衡了压缩与精度的 tradeoff
- 动态 rank 效果与固定 rank 相当或略优，但无需手动调参，对超参数不敏感

术语一般如何实现？如何使用？

Dynamic Rank Selection 可与 Fixed Rank Selection 互换使用（论文 Table 10 显示二者性能接近）。实际使用中：(1) 先用动态 rank 确定每层的 r_q、r_kv 值（一次性计算，耗时极短）；(2) 然后按确定的 rank 值进行 SVD 初始化；(3) 后续训练与固定 rank 相同。能量阈值 δ 通常设为 0.85-0.95（论文中的典型配置）。

涉及论文标题：
- X-EcoMLA__Upcycling_Pre-Trained_Attention_into_MLA_for_Efficient_and_Extreme_KV_Compression

---
