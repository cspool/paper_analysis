## AKI-Pro

术语解释
AKI-Pro 是 AquilaMoE 对 bert2BERT AKI 方法的改进版初始化策略，包含两个关键改进：(1) 深度扩展使用 Interpolation 替代 Stacking；(2) 适配 Group Query Attention (GQA) 架构。

术语是什么？
AKI-Pro 的两个改进：
1. **Depth Growing via Interpolation**: 原 bert2BERT 使用 StackBERT 的 stacking（层堆叠）扩展深度，即 W'_l = W_{l mod L_1}。但 stacking 导致 L_1-1 层的输出空间与第 0 层输入空间不匹配，训练初期不稳定。AKI-Pro 改为 interpolation：W'_l = floor(l × L_2 / L_1)，相邻层在输出/输入空间上平滑过渡，连续训练更稳定。
2. **GQA Compatibility**: 原 AKI 仅支持 MHA。AKI-Pro 在源和目标模型 GQA group 数一致的约束下，将每个 group 视为独立 MHA block 进行 AKI 扩展，QKV projection 的扩展操作与 MHA 完全一致。

从算法pipeline角度拆解术语：
```
# 1. Depth Interpolation (L1=3, L2=6):
# 源层: W[0], W[1], W[2]
# Stacking: W'[0]=W[0], W'[1]=W[1], W'[2]=W[2], W'[3]=W[0], W'[4]=W[1], W'[5]=W[2]
#   → W'[2].out != W'[3].in: 输出空间不匹配
# Interpolation: 
#   W'[0]=W[0], W'[1]=W[1], W'[2]=W[1] (插值), W'[3]=W[2], W'[4]=W[2] (插值), W'[5]=W[2]
#   → 平滑过渡，训练更稳定

# 2. GQA 扩展（8 groups, heads_per_group=5）:
for group in range(8):  # 每组独立
    for head in range(5):  # 组内 head 扩展同 MHA
        W_qkv_large[group, head] = AKI_expand(W_qkv_small[group, head])
# 约束: num_groups_small == num_groups_large
```

术语一般如何实现？如何使用？
- 验证结果：AKI-Pro-Interpolation loss 7.81 vs AKI-Stacking loss 9.56 at M(32,4096)——interpolation 显著降低初始 loss
- 用于 AquilaMoE 的 Scale-Up 阶段：1.3B→7B (24→32 layers, 2048→4096 hidden), 7B→16B (32→40 layers, 4096→5120 hidden)
- GQA 适配假设源和目标模型 group 数相同；无法处理 group 数不同的情况
- Depth Interpolation 参考文献：Pan et al. (2024) "Preparing Lessons for Progressive Training on Language Models"

涉及论文标题：
- AquilaMoE Efficient Training for MoE Models with Scale-Up and Scale-Out Strategies

---
