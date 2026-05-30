## IsoFLOP Analysis（等计算量分析）

术语是什么？
IsoFLOP Analysis 是由 Borgeaud et al. (2022b) 引入的实验方法论：固定总计算预算（FLOPs），联合变化模型大小和训练 token 数，在验证集上绘制"模型大小 vs perplexity"曲线（isoFLOP 曲线）。曲线上的每个点具有相同的总计算成本，曲线最低点对应的模型大小即为该 FLOP 预算下的 compute-optimal 模型。IsoFLOP 分析避免了训练多个 model size 到收敛的高昂成本，用相对较小的 FLOP 预算来表征模型大小的缩放行为。PEER 使用 isoFLOP 分析（6e18 和 2e19 FLOPs）比较 PEER vs Dense FFW vs Coarse-grained MoE vs PKM 的性能-计算 trade-off。

从算法pipeline角度拆解术语：
IsoFLOP 分析流程：
```
固定 FLOP 预算 F:
for model_size in [M_1, M_2, ..., M_n]:
    # 计算 FLOPs per training step: flops_per_step(model_size, batch, seq_len)
    num_steps = F / flops_per_step
    train model with (model_size, num_steps) on C4
    record validation perplexity
plot: x = model_size, y = validation perplexity (isoFLOP curve)
compute_optimal = argmin(perplexity)
```
关键假设：同一 FLOP 预算下训练不同大小的模型，验证 perplexity 是模型大小的 U 形函数（过小欠拟合，过大过拟合/欠训练）。PEER 中 isoFLOP 曲线显示：稀疏替代方案（MoE/PKM/PEER）将曲线向下和向右移动——引入更大 P 但使用更少或相等的 P_active。

术语一般如何实现？
PEER 的具体参数：FLOP 预算 = 6e18 和 2e19，batch size = 128，sequence length = 2048，训练步数 = FLOP 预算 / 每步 FLOPs。对于每个方法（Dense/MoE/PKM/PEER），从同一 dense backbone 开始，取不同 model size（通过变化层数、attention heads、d_model），中间一层替换为对应方法，训练至相同 FLOP 预算后在 C4 验证集评估 perplexity。IsoFLOP 曲线在双对数坐标中呈现 U 形。

涉及论文标题：
- Mixture of A Million Experts
